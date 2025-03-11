import os
import cv2
import sys
import json
import torch
import decord
import random
import pandas as pd
import numpy as np
from model import models
from tqdm import tqdm
from model.losses import nce
from model.siglip import Siglip
from model.utils import pad_sequences_1d
from model.siglip import Siglip
from transformers import AutoProcessor, AutoModel, AutoTokenizer

from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt

from IPython.display import Video

os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)

class VlogVocab:
    # def __init__(self, vlog_model, vocab_url, args):
    #     # VLog model
    #     self.args = vars(args)
    #     self.vlog = vlog_model
    #     self.tokenizer =  vlog_model.tokenizer
    #     # Vidcab
    #     self.vocab = torch.load(vocab_url)
    #     self.scene_embeds = self.vocab['scene_embed']
    #     self.scene2vid = self.vocab['scene2vid']
    #     # VidExtractor
    #     self.siglip = AutoModel.from_pretrained("google/siglip-so400m-patch14-384").cuda()
    #     self.processor = AutoProcessor.from_pretrained("google/siglip-so400m-patch14-384")
    
    def __init__(self, args_url, ckpt_url, vocab_url):
        with open(args_url, 'r', encoding='utf-8') as file:
            args = json.load(file)

        model_args = models.ModelArgs(
            llm_model = args['llm_model'],
            lora = args['lora'],
            vis_model = args['vis_model'],
            llm_8bit = args['llm_8bit'],
            local_rank = args['local_rank'],
            freeze_lm = args['freeze_lm'],
            freeze_vm = False,
            add_special_tokens = args['add_special_tokens'],
            num_layers = args['num_layers'],
            hidden_dim = args['hidden_dim'],
            nheads = args['nheads'],
            dim_feedforward = args['dim_feedforward'],
            dropout = args['dropout'],
            droppath = args['droppath'],
            visual_input = args['visual_input'],
            n_visual_tokens = args['n_visual_tokens'],
        )
        self.args = args

        # init. tokenizer
        tokenizer = AutoTokenizer.from_pretrained(args['llm_model'], use_fast=False)
        if tokenizer.pad_token is None:
            tokenizer.pad_token_id = (0)       
        self.tokenizer = tokenizer
        
        # init. VLog
        vlog_model = models.VLog(tokenizer, model_args).cuda()
        if args['precision'] == 'fp16':
            vlog_model = vlog_model.float()
        elif args['precision'] == 'bf16':
            vlog_model = vlog_model.bfloat16()
        self.criterion = nce.NormSoftmaxLoss().cuda(args['nce_temperature'])
                    
        checkpoint = torch.load(ckpt_url)
        checkpoint['state_dict'] = {k.replace('module.',''):v for k,v in checkpoint['state_dict'].items()}
        vlog_model.load_state_dict(checkpoint['state_dict'], strict=False)
        self.vlog = vlog_model

        # init. Vidcab
        self.vocab = torch.load(vocab_url)
        self.scene_embeds = self.vocab['scene_embed']
        self.scene2vid = self.vocab['scene2vid']
        
        # VidExtractor
        self.siglip = AutoModel.from_pretrained("google/siglip-so400m-patch14-384").cuda()
        self.processor = AutoProcessor.from_pretrained("google/siglip-so400m-patch14-384")
        
    def collate_fn(self, batch):
        batched_data = dict()
        inputs_keys = batch[0].keys()
        batched_data['input_embeds'], batched_data['input_masks'] = pad_sequences_1d([e['input_embeds'] for e in batch], dtype=torch.float32, padded_sides="left")
        batched_data['query_ids'], batched_data['query_masks'] = pad_sequences_1d([e['query_ids'] for e in batch], dtype=torch.long, padded_values=self.tokenizer.pad_token_id, padded_sides="right")
        return batched_data
    
    def get_topk_scene(self, vid_embeds, topk=None, query="What is the overall activity in the video?"):
        self.scene_embeds = self.vocab['scene_embed']
        query_ids = self.tokenizer(query).input_ids
        eos_id = self.tokenizer.eos_token_id
        batch = [dict(input_embeds=vid_embeds, query_ids=query_ids+[eos_id])]
        batch = self.collate_fn(batch)
        
        input_embeds = batch["input_embeds"].cuda()
        input_masks = batch["input_masks"].cuda()
        query_ids = batch["query_ids"].cuda()
        query_masks = batch["query_masks"].cuda()
        
        if self.args['precision'] == 'fp16':
            input_embeds = input_embeds.float()
        elif self.args['precision'] == 'bf16':
            input_embeds = input_embeds.bfloat16()
        
        ret_embeds = self.vlog(input_embeds=input_embeds, input_masks=input_masks, query_ids=query_ids, query_masks=query_masks,)
        self.scene_embeds = self.scene_embeds.to(device=ret_embeds.device, dtype=ret_embeds.dtype)
        
        sim_score = self.criterion.get_sim_matrix(ret_embeds, self.scene_embeds)
        if topk is None:
            topk = len(self.scene_embeds)
        topk_values, topk_indices = torch.topk(sim_score, k=topk, dim=1)
        
        topk_list = topk_indices[0].tolist()
        topk_uid = [self.scene2vid[x]['vid_uid'] for x in topk_list]
        topk_scene = [self.scene2vid[x]['scene'] for x in topk_list]
        topk_conf = topk_values[0].tolist()
        return topk_uid, topk_conf, topk_scene

    def encode_feature(self, video_path, fps=2, bsz=64):
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        interval = int(video_fps / fps)
    
        frames = []
        frame_count = 0
        success, frame = cap.read()
    
        while success:
            if frame_count % interval == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                frames.append(pil_image)
            
            frame_count += 1
            success, frame = cap.read()
        cap.release()
    
        features = []
        for i in tqdm(range(0, len(frames), bsz), desc=f"Extracting features"):
            batch_frames = frames[i:i+bsz]
            inputs = self.processor(images=batch_frames, return_tensors="pt", padding=True)
            inputs = inputs.to('cuda')
            with torch.no_grad():
                image_features = self.siglip.get_image_features(**inputs)
            features.append(image_features.cpu())
    
        features_tensor = torch.cat(features, dim=0)
        return features_tensor
        
    def get_topn_vocab(self, vid_embeds, topk_uid=None, topn=1, topk=1, query="What is the action in the video?"):
        if topk_uid is None:
            topk_uid, topk_conf, topk_scene = self.get_topk_scene(vid_embeds, topk)

        cand_text = [self.vocab['vid2narr'][x]['text'] for x in topk_uid]
        cand_embed =  [self.vocab['vid2narr'][x]['embed'] for x in topk_uid]
        cand_text = [item for sublist in cand_text for item in sublist]
        cand_embed = torch.cat(cand_embed)
        
        query_ids = self.tokenizer(query).input_ids
        batch = [dict(input_embeds=vid_embeds, query_ids=query_ids+[self.tokenizer.eos_token_id])]
        batch = self.collate_fn(batch)

        input_embeds = batch["input_embeds"].cuda()
        input_masks = batch["input_masks"].cuda()
        query_ids = batch["query_ids"].cuda()
        query_masks = batch["query_masks"].cuda()
        
        if self.args['precision'] == 'fp16':
            input_embeds = input_embeds.float()
        elif self.args['precision'] == 'bf16':
            input_embeds = input_embeds.bfloat16()
    
        with torch.no_grad():
            ret_embeds = self.vlog(input_embeds=input_embeds, input_masks=input_masks, query_ids=query_ids, query_masks=query_masks,)
        cand_embed = cand_embed.to(device=ret_embeds.device, dtype=ret_embeds.dtype)
            
        sim_score = self.criterion.get_sim_matrix(ret_embeds, cand_embed)
        max_value, max_index = torch.max(sim_score, dim=1)
        topn_values, topn_indices = torch.topk(sim_score, k=topn, dim=1)
        
        topn_list = topn_indices[0].tolist()
        topn_narr = [cand_text[x] for x in topn_list]
        topn_conf = topn_values[0].tolist()
        # return topn_narr, topn_conf, cand_text
        return dict(topn_narr=topn_narr, topn_conf=topn_conf, cand_text=cand_text)