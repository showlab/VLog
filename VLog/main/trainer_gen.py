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

    for i, batch in enumerate(train_loader):
        input_embeds = batch["input_embeds"]
        input_masks = batch["input_masks"]
        
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

        model_outputs = model(
            input_embeds=input_embeds, input_masks=input_masks, 
            output_embeds=output_embeds, output_masks=output_masks, 
            query_ids=query_ids, query_masks=query_masks,
            response_ids=response_ids, response_masks=response_masks)

        lm_loss = model_outputs.loss
        lm_losses.update(lm_loss.item(), input_embeds.size(0))

        loss = args.scale_lm_loss * lm_loss
        loss = loss / args.grad_accumulation_steps
        losses.update(loss.item(), input_embeds.size(0))
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