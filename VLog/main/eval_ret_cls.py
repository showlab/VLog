import pdb
import time
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


def val_epoch(val_loader, model, vocab_model, tokenizer, criterion, epoch, global_step, writer, args, logger):
    num_samples = len(val_loader.dataset)
    model.eval()
    ngpus_per_node = torch.cuda.device_count()
    device_id = args.local_rank if args.local_rank > 0 else 0

    all_preds = []
    all_gts = []
    all_metadata = []
    for i, batch in enumerate(tqdm(val_loader)):
        input_embeds = batch["input_embeds"]
        input_masks = batch["input_masks"]
        if batch["output_embeds"] is not None and isinstance(batch["output_embeds"][0], str):
            output_embeds = vocab_model.get_txt_embeds(batch["output_embeds"])
            if len(output_embeds.shape) == 1:
                output_embeds = output_embeds.unsqueeze(0)
            output_embeds, output_masks = pad_sequences_1d([e for e in output_embeds], dtype=torch.float32, fixed_length=None)

        metadata = batch["metadata"]
        query_ids = batch["query_ids"]
        query_masks = batch["query_masks"]
        response_ids = batch.get("response_ids", None)
        response_masks = batch.get("response_masks", None)

        # measure data loading time

        if torch.cuda.is_available():
            input_embeds = input_embeds.cuda(device_id, non_blocking=True)
            input_masks = input_masks.cuda(device_id, non_blocking=True)
            output_embeds = output_embeds.cuda(device_id, non_blocking=True)
            query_ids = query_ids.cuda(device_id, non_blocking=True)
            query_masks = query_masks.cuda(device_id, non_blocking=True)

        if args.precision == 'fp16':
            input_embeds = input_embeds.float()
            output_embeds = output_embeds.float()
        elif args.precision == 'bf16':
            input_embeds = input_embeds.bfloat16()
            output_embeds = output_embeds.bfloat16()

        mode_start = time.time()

        with torch.no_grad():
            ret_embeds = model(
                input_embeds=input_embeds, input_masks=input_masks, 
                output_embeds=output_embeds, output_masks=output_masks, 
                query_ids=query_ids, query_masks=query_masks,
                response_ids=response_ids, response_masks=response_masks)

        # ret_embeds = ret_embeds.cpu().detach()
        # output_embeds = output_embeds.cpu().detach()

        # all_preds.append(ret_embeds.cpu().detach())
        # all_gts.append(output_embeds.cpu().detach())

        if args.distributed and ngpus_per_node > 1:
            all_ret_embeds = [torch.zeros_like(ret_embeds) for _ in range(dist.get_world_size())]
            dist.all_gather(all_ret_embeds, ret_embeds)
            all_ret_embeds[dist.get_rank()] = ret_embeds
            all_ret_embeds = torch.cat(all_ret_embeds, axis=0)

            batch_metadata = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(batch_metadata, metadata)
            batch_metadata[dist.get_rank()] = metadata
            batch_metadata = [item for sublist in batch_metadata for item in sublist]
        else:
            all_ret_embeds = ret_embeds
            batch_metadata = metadata

        if args.main_node:
            all_metadata += batch_metadata
            all_preds.append(all_ret_embeds)

    if args.main_node:
        all_preds = torch.cat(all_preds, axis=0)
        all_preds = all_preds[:num_samples].float()
        all_metadata = all_metadata[:num_samples]

        all_gts = vocab_model.get_txt_embeds(args.candidate)
        all_idxs = [x['index'] for x in all_metadata]

        mapped_gts = all_gts[all_idxs]
        sim_score = criterion.get_sim_matrix(all_preds, all_gts)
        sim_array = sim_score.to(torch.float32).cpu().detach().numpy()
        pred_idxs = np.argmax(sim_array, axis=1).tolist()
        accuracy = [1 if pred == gt else 0 for pred, gt in zip(pred_idxs, all_idxs)]
        accuracy = np.mean(accuracy)

        eval_score=dict(accuracy = accuracy)

        logger.info("Score on {} samples".format(len(all_preds)))
        logger.info(f"{args.val_metadata} Acc: {eval_score}")

        if not args.debug:
            for k, v in eval_score.items():
                wandb.log({f"{args.val_dataset}/{args.val_metadata}/{k}": v}, step=global_step)

        return eval_score['accuracy']