import os
import pdb
import time
import json
import wandb
import torch
import torch.nn as nn
import numpy as np
import pynvml
from tqdm import tqdm
from model import utils
from model.utils import pad_sequences_1d
from main.eval_utils import retrieval_score
import torch.distributed as dist

def save_json(save_dict, save_path):
    save_dir = os.path.dirname(save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(save_path, 'w') as f:
        json.dump(save_dict, f, indent=4)

def val_epoch(val_loader, model, vocab_model, tokenizer, criterion, epoch, global_step, writer, args, logger):
    num_samples = len(val_loader.dataset)
    model.eval()
    ngpus_per_node = torch.cuda.device_count()
    device_id = args.local_rank if args.local_rank > 0 else 0

    all_preds = []
    all_gts = []
    all_metadata = []

    total_forward_time = []
    for i, batch in enumerate(tqdm(val_loader)):
        input_embeds = batch["input_embeds"]
        input_masks = batch["input_masks"]
        
        # if batch["output_embeds"] is not None and isinstance(batch["output_embeds"][0], str):
        #     output_embeds = vocab_model.get_txt_embeds(batch["output_embeds"])
        #     output_embeds, output_masks = pad_sequences_1d([e for e in output_embeds], dtype=torch.float32, fixed_length=None)
        # else:
        output_embeds = batch["output_embeds"]
        output_masks = batch["output_masks"]

        metadata = batch["metadata"]
        query_ids = batch["query_ids"]
        query_masks = batch["query_masks"]
        response_ids = batch.get("response_ids", None)
        response_masks = batch.get("response_masks", None)

        # measure data loading time

        if torch.cuda.is_available():
            input_embeds = input_embeds.cuda(device_id, non_blocking=True)
            input_masks = input_masks.cuda(device_id, non_blocking=True)
            query_ids = query_ids.cuda(device_id, non_blocking=True)
            query_masks = query_masks.cuda(device_id, non_blocking=True)
            # if output_embeds is not None:
            #     output_embeds = output_embeds.cuda(device_id, non_blocking=True)

        if args.precision == 'fp16':
            input_embeds = input_embeds.float()
            # if output_embeds is not None:
            #     output_embeds = output_embeds.float()
        elif args.precision == 'bf16':
            input_embeds = input_embeds.bfloat16()
            # if output_embeds is not None:
            #    output_embeds = output_embeds.bfloat16()

        mode_start = time.time()
        t1 = time.time()
        with torch.no_grad():
            model_outputs, generated_ids, groundtruth_ids = model(
                input_embeds=input_embeds, input_masks=input_masks, 
                output_embeds=output_embeds, output_masks=output_masks, 
                query_ids=query_ids, query_masks=query_masks,
                response_ids=response_ids, response_masks=response_masks,
                generate=True)
        t2 = time.time()
        total_forward_time.append(t2 - t1)
        print(f"Generative Time taken for model forward pass: {t2-t1}")

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
            all_generated_ids = generated_ids
            all_groundtruth_ids = groundtruth_ids
            batch_metadata = metadata

        if args.main_node:
            generated_caps = tokenizer.batch_decode(all_generated_ids, skip_special_tokens=True)
            groundtruth_caps = tokenizer.batch_decode(all_groundtruth_ids, skip_special_tokens=True)
            all_preds += generated_caps
            all_gts += groundtruth_caps
            all_metadata += batch_metadata

    if args.main_node:
        # all_preds = torch.cat(all_preds, axis=0)
        # all_gts = torch.cat(all_gts, axis=0)
        all_preds = all_preds[:num_samples]
        all_gts = all_gts[:num_samples]
        all_metadata = all_metadata[:num_samples]

        num_correct = 0
        for pred, gt in zip(all_preds, all_gts):
            # print(pred, gt)
            if pred.lower() == gt.lower():
                num_correct += 1
        accuracy = num_correct / num_samples
        logger.info(f"Accuracy: {accuracy}")

        answer_list = [x['answer']['clip_text'] for x in all_metadata]
        for i, x in enumerate(all_metadata):
            x['gt_answer'] = x['answer']['clip_text']
            x['pred_answer'] = all_preds[i]

        eval_score=dict(accuracy=accuracy, pred=all_preds, gt=all_gts)

        avg_forward_time = sum(total_forward_time) / len(total_forward_time)
        logger.info(f"Total Forward Time: {sum(total_forward_time)}")
        logger.info(f"Average Forward Time per Iter: {avg_forward_time}")
        logger.info(f"Average Forward Time per Sample: {avg_forward_time / args.batch_size}")

        if not args.debug:
            for k, v in eval_score.items():
                wandb.log({f"{args.val_dataset}/{args.val_metadata}/{k}": v}, step=global_step)

        save_json(all_metadata, os.path.join(args.log_dir, 'tmp', f'{args.val_dataset}_{args.val_metadata}_{epoch}_tmp_dict.json'))
        save_json(eval_score, os.path.join(args.log_dir, 'tmp', f'{args.val_dataset}_{args.val_metadata}_{epoch}_res_dict.json'))
        logger.info(f"Saved tmp dict and res dict to {args.log_dir}/tmp")
        return eval_score['accuracy']