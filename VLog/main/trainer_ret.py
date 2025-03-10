import pdb
import time
import wandb
import torch
import torch.nn as nn
import pynvml
from model import utils
from model.utils import pad_sequences_1d
import torch.distributed as dist

def train_epoch(train_loader, model, vocab_model, tokenizer, criterion, optimizer, epoch, scheduler, writer, args, logger=None):
    """Main training loop."""
    pynvml.nvmlInit()
    device_id = args.local_rank if args.local_rank > 0 else 0
    handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)

    ngpus_per_node = torch.cuda.device_count()
    batch_time = utils.AverageMeter('Iter(s):', ':4.2f')
    epoch_time = utils.AverageMeter('Epoch(h):', ':4.2f', utils.Summary.NONE)
    cap_time = utils.AverageMeter('CaptioningTime:', ':4.2f')
    data_time = utils.AverageMeter('Data:', ':4.2f')
    losses = utils.AverageMeter('Loss:', ':.2e')
    ret_losses = utils.AverageMeter('RetLoss:', ':.2e')
    lm_losses = utils.AverageMeter('LmLoss:', ':.2e')
    top1 = utils.AverageMeter('Acc@1:', ':4.2f')
    top5 = utils.AverageMeter('Acc@5:', ':4.2f')
    memory = utils.AverageMeter('Mem(Gb):', ':4.2f')

    progress = utils.ProgressMeter(
        args.steps_per_epoch,
        [batch_time, epoch_time, losses, lm_losses, ret_losses, memory],
        prefix="Epoch: [{}]".format(epoch),
        logger=logger)

    model.train()
    end = time.time()

    if args.main_node:
        if not args.debug:
            wandb.watch(model, log="all", log_freq=10)

    if args.train_class and args.class_list is not None:
        all_cls_embeds = vocab_model.get_txt_embeds(args.class_list)
        all_cls_embeds = all_cls_embeds.cuda(device_id, non_blocking=True)
        if args.precision == 'fp16':
            all_cls_embeds = all_cls_embeds.float()
        elif args.precision == 'bf16':
            all_cls_embeds = all_cls_embeds.bfloat16()

    for i, batch in enumerate(train_loader):
        input_embeds = batch["input_embeds"]
        input_masks = batch["input_masks"]
        metadata = batch["metadata"]

        if batch["output_embeds"] is not None and isinstance(batch["output_embeds"][0], str):
            output_embeds = vocab_model.get_txt_embeds(batch["output_embeds"])
            output_embeds, output_masks = pad_sequences_1d([e for e in output_embeds], dtype=torch.float32, fixed_length=None)
        else:
            output_embeds = batch["output_embeds"]
            output_masks = batch["output_masks"]

        query_ids = batch["query_ids"]
        query_masks = batch["query_masks"]
        response_ids = batch.get("response_ids", None)
        response_masks = batch.get("response_masks", None)

        actual_step = epoch * args.steps_per_epoch + i + 1
        # measure data loading time
        data_time.update(time.time() - end)

        if torch.cuda.is_available():
            input_embeds = input_embeds.cuda(device_id, non_blocking=True)
            if output_embeds is not None:
                output_embeds = output_embeds.cuda(device_id, non_blocking=True)

        if args.precision == 'fp16':
            input_embeds = input_embeds.float()
            if output_embeds is not None:
                output_embeds = output_embeds.float()
        elif args.precision == 'bf16':
            input_embeds = input_embeds.bfloat16()
            if output_embeds is not None:
               output_embeds = output_embeds.bfloat16()

        mode_start = time.time()

        ret_embeds = model(
            input_embeds=input_embeds, input_masks=input_masks, 
            output_embeds=output_embeds, output_masks=output_masks, 
            query_ids=query_ids, query_masks=query_masks,
            response_ids=response_ids, response_masks=response_masks)

        if args.distributed and ngpus_per_node > 1:
            all_ret_embeds = [torch.zeros_like(ret_embeds) for _ in range(dist.get_world_size())]
            dist.all_gather(all_ret_embeds, ret_embeds)
            all_ret_embeds[dist.get_rank()] = ret_embeds
            all_ret_embeds = torch.cat(all_ret_embeds, axis=0)

            all_gts_embeds = [torch.zeros_like(output_embeds) for _ in range(dist.get_world_size())]
            dist.all_gather(all_gts_embeds, output_embeds)
            all_gts_embeds[dist.get_rank()] = output_embeds
            all_gts_embeds = torch.cat(all_gts_embeds, axis=0)

            batch_metadata = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(batch_metadata, metadata)
            batch_metadata[dist.get_rank()] = metadata
            batch_metadata = [item for sublist in batch_metadata for item in sublist]
        else:
            all_ret_embeds = ret_embeds
            all_gts_embeds = output_embeds
            batch_metadata = metadata

        memory.update(meminfo.used / 1024 / 1024 / 1024)
        cap_time.update(time.time() - mode_start)

        if args.train_class and args.class_list is not None:
            # use closed-set list for supervision
            cls_idx = [x['index'] for x in batch_metadata]
            ret_loss = criterion(all_ret_embeds, all_cls_embeds, cls_idx=cls_idx)
        else:
            ret_loss = criterion(all_ret_embeds, all_gts_embeds)

        output_size = all_ret_embeds.size(0)
        # ret_loss = criterion(all_ret_embeds, all_gts_embeds)
        ret_losses.update(ret_loss.item(), output_size)

        loss = args.scale_ret_loss * ret_loss
        loss = loss / args.grad_accumulation_steps
        losses.update(loss.item(), output_size)
        loss.backward()

        # Update weights
        if ((i + 1) % args.grad_accumulation_steps == 0):
            # compute gradient and do SGD step
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            optimizer.zero_grad()

        # measure elapsed time
        batch_time.update(time.time() - end)
        epoch_time.update(batch_time.avg * args.steps_per_epoch / 3600)
        end = time.time()

        if (i + 1) % args.print_freq == 0:
            ex_per_sec = args.batch_size / batch_time.avg
            # if args.distributed:
            #     batch_time.all_reduce()
            #     epoch_time.all_reduce()
            #     data_time.all_reduce()
            #     ex_per_sec = (args.batch_size / batch_time.avg) * ngpus_per_node

            #     losses.all_reduce()
            #     lm_losses.all_reduce()
            #     top1.all_reduce()
            #     top5.all_reduce()
            #     cap_time.all_reduce()
            #     memory.all_reduce()

            if args.main_node:
                progress.display(i + 1)
                writer.add_scalar('train/loss', losses.avg, actual_step)
                writer.add_scalar('train/lm_loss', lm_losses.avg, actual_step)
                writer.add_scalar('train/ret_loss', ret_losses.avg, actual_step)
                writer.add_scalar('metrics/total_secs_per_batch', batch_time.avg, actual_step)
                writer.add_scalar('metrics/total_secs_captioning', cap_time.avg, actual_step)
                writer.add_scalar('metrics/data_secs_per_batch', data_time.avg, actual_step)
                writer.add_scalar('metrics/examples_per_sec', ex_per_sec, actual_step)
                if not args.debug:
                    wandb.log({"epoch": epoch, 
                        "loss": losses.avg,
                        "batch_time": batch_time.avg,
                        "epoch_time": epoch_time.avg},
                        step=actual_step)

        if i == args.steps_per_epoch - 1:
            break

        scheduler.step()
        curr_lr = scheduler.get_last_lr()
        if args.main_node:
            if (i + 1) % args.print_freq == 0 and not args.debug:
                # Write current learning rate to Tensorboard.
                writer.add_scalar('train/lr', curr_lr[0], actual_step)
                
    return actual_step