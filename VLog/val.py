
import os
import pdb
import time
import tqdm
import json
import pickle
import pynvml
import numpy as np
import collections
from PIL import Image
import language_evaluation
import evaluate

import torch
import torch.distributed as dist
from torch.utils.data import Subset
from torch.utils.tensorboard import SummaryWriter
import torchvision
from bert_score import score
torch.set_printoptions(threshold=torch.inf)

from main import losses as losses_utils
from main import utils
from main.utils import format_time, sort_list_by_time
from dataset import data

import sys

def compute_metrics(predictions, references):
    bleu = evaluate.load('bleu')
    rouge = evaluate.load('rouge')
    meteor = evaluate.load('meteor')
    # cider = evaluate.load('cider')

    bleu_score = bleu.compute(predictions=predictions, references=references)
    rouge_score = rouge.compute(predictions=predictions, references=references)
    meteor_score = meteor.compute(predictions=predictions, references=references)
    # cider_score = cider.compute(predictions=predictions, references=references)

    return dict(bleu=bleu_score['bleu'], rougel=rouge_score['rougeL'], meteor=meteor_score['meteor']) # cider=cider_score)

def validate(val_loader, model, tokenizer, criterion, epoch, writer, args, logger):
    evaluator = language_evaluation.CocoEvaluator()
    num_samples = len(val_loader.dataset)
    
    # steps-per-epoch
    if args.val_steps_per_epoch == -1:
        args.val_steps_per_epoch = len(val_loader)
    
    pynvml.nvmlInit()
    device_id = args.local_rank if args.local_rank > 0 else 0
    handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)

    ngpus_per_node = torch.cuda.device_count()
    batch_time = utils.AverageMeter('IterTime(s):', ':5.2f')
    epoch_time = utils.AverageMeter('EpochTime(h):', ':5.2f', utils.Summary.NONE)
    losses = utils.AverageMeter('Loss:', ':.4e', utils.Summary.AVERAGE)
    lm_losses = utils.AverageMeter('LmLoss:', ':.4e', utils.Summary.AVERAGE)
    top1 = utils.AverageMeter('Acc@1:', ':5.2f', utils.Summary.AVERAGE)
    top5 = utils.AverageMeter('Acc@5:', ':5.2f', utils.Summary.AVERAGE)
    bleu1 = utils.AverageMeter('BLEU@1:', ':5.2f', utils.Summary.AVERAGE)
    bleu2 = utils.AverageMeter('BLEU@2:', ':5.2f', utils.Summary.AVERAGE)
    bleu3 = utils.AverageMeter('BLEU@3:', ':5.2f', utils.Summary.AVERAGE)
    bleu4 = utils.AverageMeter('BLEU@4:', ':5.2f', utils.Summary.AVERAGE)
    meteor = utils.AverageMeter('METEOR:', ':5.2f', utils.Summary.AVERAGE)
    rouge_l = utils.AverageMeter('ROUGE_L:', ':5.2f', utils.Summary.AVERAGE)
    cider = utils.AverageMeter('CIDEr:', ':5.2f', utils.Summary.AVERAGE)
    spice = utils.AverageMeter('SPICE:', ':5.2f', utils.Summary.AVERAGE)
    bertscore = utils.AverageMeter('BERTScore:', ':5.2f', utils.Summary.AVERAGE)
    memory = utils.AverageMeter('GpuMemory(gb):', ':5.2f')
    accuracy = utils.AverageMeter('Gen Acc:', ':5.2f')
    
    progress = utils.ProgressMeter(
        len(val_loader) + (args.distributed and (len(val_loader.sampler) * args.world_size < len(val_loader.dataset))),
        [batch_time, epoch_time, losses, lm_losses, top1, top5, memory],
        prefix="Eval: [{}]".format(epoch),
        logger=logger)
    
    # switch to eval mode
    model.eval()
    end = time.time()
    all_generated_caps = []
    all_groundtruth_caps = []
    all_metadata = []
    
    for i, batch in enumerate(val_loader):
        metadata = batch["vid_meta"]
        vid_inputs = batch["vid_inputs"]
        vid_masks = batch["vid_masks"]
        pre_inputs = batch["pre_inputs"]
        pre_masks = batch["pre_masks"]
        post_inputs = batch["post_inputs"]
        post_masks = batch["post_masks"]

        if torch.cuda.is_available():
            vid_inputs = vid_inputs.cuda(args.gpu, non_blocking=True)
            vid_masks = vid_masks.cuda(args.gpu, non_blocking=True)
            pre_inputs = pre_inputs.cuda(args.gpu, non_blocking=True)
            pre_masks = pre_masks.cuda(args.gpu, non_blocking=True)
            post_inputs = post_inputs.cuda(args.gpu, non_blocking=True)
            post_masks = post_masks.cuda(args.gpu, non_blocking=True)

        if args.distributed:
            device = torch.device("cuda", args.local_rank)
            vid_inputs = vid_inputs.to(device)
            vid_masks = vid_masks.to(device)
            pre_inputs = pre_inputs.to(device)
            pre_masks = pre_masks.to(device)
            post_inputs = post_inputs.to(device)
            post_masks = post_masks.to(device)

        if args.precision == 'fp16':
            vid_inputs = vid_inputs.half()
        elif args.precision == 'bf16':
            vid_inputs = vid_inputs.bfloat16()

        if 'match_narr' in batch:
            match_narr = batch["match_narr"]
            narr_inputs = get_txt_embeds(match_narr).cuda(args.gpu, non_blocking=True)
            if args.precision == 'fp16':
                narr_inputs = narr_inputs.float()
            elif args.precision == 'bf16':
                narr_inputs = narr_inputs.bfloat16()
            if args.distributed:
                narr_inputs = narr_inputs.to(device)
        else:
            narr_inputs = None

        model_outputs, full_labels, generated_ids, groundtruth_ids = model(
            vid_inputs=vid_inputs, vid_masks=vid_masks, 
            pre_inputs=pre_inputs, pre_masks=pre_masks, 
            post_inputs=post_inputs, post_masks=post_masks,
            narr_inputs=narr_inputs,
            generate=True, temperature=args.temperature, do_sample=args.do_sample, max_len=args.max_len_eval)
        
        loss = 0
        output = model_outputs.logits

        acc1, acc5 = utils.accuracy(output[:, :-1, :], full_labels[:, 1:], -100, topk=(1, 5))
        top1.update(acc1[0], output.size(0))
        top5.update(acc5[0], output.size(0))

        memory.update(meminfo.used / 1024 / 1024 / 1024)

        lm_loss = args.scale_lm_loss * model_outputs.loss
        lm_losses.update(lm_loss.item(), output.size(0))
        
        loss += lm_loss
        losses.update(loss.item(), output.size(0))
        
        if args.distributed and ngpus_per_node > 1:
            batch_metadata = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(batch_metadata, metadata)
            batch_metadata[dist.get_rank()] = metadata
            batch_metadata = [item for sublist in batch_metadata for item in sublist]
            
            all_generated_ids = [torch.zeros_like(generated_ids) for _ in range(dist.get_world_size())]
            dist.all_gather(all_generated_ids, generated_ids)
            all_generated_ids[dist.get_rank()] = generated_ids
            all_generated_ids = torch.cat(all_generated_ids)

            all_groundtruth_ids = [torch.zeros_like(groundtruth_ids) for _ in range(dist.get_world_size())]
            dist.all_gather(all_groundtruth_ids, groundtruth_ids)
            all_groundtruth_ids[dist.get_rank()] = groundtruth_ids
            all_groundtruth_ids = torch.cat(all_groundtruth_ids)
        else:
            batch_metadata = metadata
            all_generated_ids = generated_ids
            all_groundtruth_ids = groundtruth_ids

        if args.main_node:
            generated_caps = tokenizer.batch_decode(all_generated_ids, skip_special_tokens=True)
            groundtruth_caps = tokenizer.batch_decode(all_groundtruth_ids, skip_special_tokens=True)

            all_metadata += batch_metadata
            all_generated_caps += generated_caps
            all_groundtruth_caps += groundtruth_caps

        # measure elapsed time
        batch_time.update(time.time() - end)
        epoch_time.update(batch_time.avg * args.val_steps_per_epoch / 3600)
        end = time.time()

        if args.main_node and i % args.print_freq == 0:
            progress.display(i + 1)

        if i == args.val_steps_per_epoch - 1:
            break

    if args.main_node:
        # Due to the DDP, the generated cadptions maybe duplicated, so we need to remove the duplicates
        logger.info("================================================================================")
        logger.info(f'Computing Epoch {epoch} scores on {num_samples} samples:')
        all_metadata = all_metadata[:num_samples]
        valid_generated_caps = all_generated_caps[:num_samples]
        # valid_generated_caps = [x.strip() for x in valid_generated_caps]
        valid_groundtruth_caps = all_groundtruth_caps[:num_samples]

        # save the predictions
        write_outputs = []
        for i, (meta_i, pred_i, gt_i) in enumerate(zip(all_metadata, valid_generated_caps, valid_groundtruth_caps)):
            write_outputs.append({
                'uid': meta_i['uid'],
                'cid': str(meta_i['chunk_idx']) if 'chunk_idx' in meta_i else None,
                'pred': pred_i,
                'gt': gt_i
            })
        val_meta_file = args.val_metadata.split('.')[0]
        save_json_url = os.path.join(args.log_base_dir, args.exp_name, f'{val_meta_file}_epoch{epoch}.json')
        with open(save_json_url, 'w') as f:
            json.dump(write_outputs, f, indent=4)

        # remove noisy characters (unsure)
        # valid_generated_caps = [x.strip() for x in valid_generated_caps]
        # valid_groundtruth_caps = [x.strip() for x in valid_groundtruth_caps]

        valid_acc = [1 if pred.lower() == gt.lower() else 0 for pred, gt in zip(valid_generated_caps, valid_groundtruth_caps)]
        scores = evaluator.run_evaluation(valid_generated_caps, valid_groundtruth_caps)
        bool_acc = sum(valid_acc) / len(valid_acc)
        P, R, F1 = score(valid_generated_caps, valid_groundtruth_caps, lang='en', verbose=True, rescale_with_baseline=True)
        # scores = compute_metrics(valid_generated_caps, valid_groundtruth_caps)

        for term, value in scores.items():
            logger.info(f"{term}: {value:.2%}")
        logger.info(f"bert score: P={P.mean():.2%}, R={R.mean():.2%}, F1={F1.mean():.2%}")
        logger.info(f"boolean acc: {bool_acc:.2%}")
        logger.info("================================================================================")

        target_metadata = []
        for meta_i, pred_i, gt_i in zip(all_metadata, valid_generated_caps, valid_groundtruth_caps):
            target_metadata.append(dict(pred=pred_i, gt=gt_i))

        for i, meta_i in enumerate(target_metadata[:1000]):
            logger.info("--------------------------------------------------------------------------------")
            logger.info(f"{i} th GroundTruth:  {meta_i['gt']}")
            logger.info(f"{i} th Generated:    {meta_i['pred']}")
        logger.info("================================================================================")

        # bleu1.update(scores['bleu'])
        # meteor.update(scores['meteor'])
        # rouge_l.update(scores['rougel'])
        bleu1.update(scores['Bleu_1'])
        bleu2.update(scores['Bleu_2'])
        bleu3.update(scores['Bleu_3'])
        bleu4.update(scores['Bleu_4'])
        meteor.update(scores['METEOR'])
        rouge_l.update(scores['ROUGE_L'])
        cider.update(scores['CIDEr'])
        spice.update(scores['SPICE'])
        accuracy.update(bool_acc)
        bertscore.update(F1.mean())

    if args.main_node:
        writer.add_scalar('val/lm_loss', losses.avg, epoch)
        writer.add_scalar('val/seq_top1_acc', top1.avg, epoch)
        writer.add_scalar('val/seq_top5_acc', top5.avg, epoch)
        writer.add_scalar('metrics/bleu1', bleu1.avg, epoch)
        writer.add_scalar('metrics/bleu2', bleu2.avg, epoch)
        writer.add_scalar('metrics/bleu3', bleu3.avg, epoch)
        writer.add_scalar('metrics/bleu4', bleu4.avg, epoch)
        writer.add_scalar('metrics/meteor', meteor.avg, epoch)
        writer.add_scalar('metrics/rouge_l', rouge_l.avg, epoch)
        writer.add_scalar('metrics/cider', cider.avg, epoch)
        writer.add_scalar('metrics/spice', spice.avg, epoch)
        writer.add_scalar('metrics/accuracy', accuracy.avg, epoch)
        writer.add_scalar('metrics/bertscore', bertscore.avg, epoch)
    # return rouge_l.avg
    return cider.avg