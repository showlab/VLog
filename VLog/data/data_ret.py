import os
import re
import pdb
import math
import json
import random
import logging
import numpy as np
import pandas as pd
from PIL import Image, ImageFont

import torch
from torch.utils.data import Dataset
import torchvision.datasets as datasets

# from dataloader import collate_fn

def read_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

class RetDataset(Dataset):
    def __init__(self, 
                dataset_dir,
                metadata,
                tokenizer,
                split: str = 'test',
                num_frame: int = 4,
                max_len: int = 128,
                max_clip_len: int = 480,
                precision: str = 'fp32',
                sep="\t",
                add_eos=True,
                num_history=0,
                past_len=0,
                fullset=True,
                hidden_dim=1152,
                ):

        metadata = os.path.join(dataset_dir, 'ego4d/metadata', f'{metadata}.json')
        metadata = read_json(metadata)[split]

        # TDL: need to be updated
        # determined by vlog data format
        self.df = []
        for sample in metadata:
            self.df.append(sample)
        self.train = split == 'train'
        self.num_frame = num_frame

        self.sample = 'rand' if self.train else 'uniform'
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.precision = precision
        self.add_eos = add_eos

        self.max_clip_len = max_clip_len
        self.num_history = num_history
        self.past_len = past_len

        self.embed_dir = f'{dataset_dir}/ego4d/siglip-so400m-patch14-384-fps2'
        
        match = re.search(r'fps(\d+)', self.embed_dir)
        if match:
            self.fps = int(match.group(1))
            print(f"Extracted fps: {self.fps}")
        else:
            raise ValueError("fps not found in embed_dir")

        self.template_input = '{query_txt}'
        self.template_output = '{response_txt}'
        self.hidden_dim = hidden_dim

    def __len__(self):
        return len(self.df)

    def _get_vid_embed(self, sample, vid):
        # vid = sample['vid_uid']
        start = sample['clip_start'] * self.fps
        end = sample['clip_end'] * self.fps
        start = math.floor(start)
        end = math.ceil(end)
        end = max(start + 1,  end)

        embed_url = os.path.join(self.embed_dir, f'{vid}.pt')
        embed_tensor = torch.load(embed_url)
        vid_embeds = embed_tensor[start:end]
        return vid_embeds

    def _get_vid_embed_w_hist(self, sample):
        history = len(sample['context'])
        vid_embeds = []
        for ctx in sample['context']:
            vid_embeds.append(self._get_vid_embed(ctx, sample['vid_uid']))
        return torch.cat(vid_embeds, dim=0)

    def _get_txt(self, sample):
        text = sample['clip_text']
        if isinstance(text, str):
            return text
        elif isinstance(text, list):
            return random.choice(text)

    def __getitem__(self, idx):
        sample = self.df[idx]
        
        query_txt = sample['question']
        response_txt = None

        try:
            input_embeds = self._get_vid_embed_w_hist(sample)
            if input_embeds.shape[0] > self.max_clip_len:
                L = input_embeds.size(0)
                indices = torch.linspace(0, L - 1, steps=self.max_clip_len).long()
                input_embeds = input_embeds[indices]
            assert len(input_embeds.shape) == 2
            assert 0 < input_embeds.shape[0] <= self.max_clip_len
        except:
            print(f"invalid input_embeds by {sample['vid_uid']}")
            input_embeds = torch.zeros(1, self.hidden_dim)
        

        output_embeds = self._get_txt(sample['answer'])

        # input should end with eos for retrieval
        if query_txt is not None:
            input_data = self.tokenizer(
                query_txt,
                truncation=True,
                max_length=self.max_len,
                padding=False,
                return_tensors=None)
            query_ids = input_data.input_ids

            if query_ids[-1] != self.tokenizer.eos_token_id:
                query_ids.append(self.tokenizer.eos_token_id)
        else:
            query_ids = [self.tokenizer.eos_token_id]

        if response_txt is not None:
            output_data = self.tokenizer(
                response_txt,
                truncation=True,
                max_length=self.max_len,
                padding=False,
                return_tensors=None)
            response_ids = output_data.input_ids

            if response_ids[-1] != self.tokenizer.eos_token_id:
                response_ids.append(self.tokenizer.eos_token_id)
            if response_ids[0] == self.tokenizer.bos_token_id:
                response_ids.pop(0)
        else:
            response_ids = None

        return dict(
            input_embeds=input_embeds, 
            output_embeds=output_embeds,
            query_txt=query_txt,
            query_ids=query_ids,
            response_txt=response_txt,
            response_ids=response_ids,
            pad_token_id=self.tokenizer.pad_token_id,
            metadata=dict(self.df[idx])
            )
        
if __name__ == "__main__":
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from tqdm import tqdm

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    metadata = "egoclip_vidcab"
    train = True

    num_frame = 1
    dataset = RetDataset(
                    # dataset_dir='/blob/v-lqinghong/data/world',
                    dataset_dir=,
                    metadata=metadata, 
                    tokenizer=tokenizer, 
                    num_frame=num_frame,
                    split='train',
                    num_history=0,
                    fullset=True,
                    max_clip_len=128,
                    )
    
    max_len = 0
    # print(len(dataset))
    for i in tqdm(range(len(dataset))):
        tmp =  dataset[i]
        max_len = max(max_len, len(tmp['input_embeds']))
        print(tmp['input_embeds'].shape)
        continue
    
    # train_loader = torch.utils.data.DataLoader(
    #         dataset, batch_size=16, shuffle=True,
    #         num_workers=4, pin_memory=True, sampler=None, collate_fn=collate_fn)