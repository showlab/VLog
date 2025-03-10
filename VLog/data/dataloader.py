from typing import Optional, Tuple
import collections
import os
import pdb
import random
import numpy as np
import pandas as pd
import torch
from torchvision import transforms as T
from PIL import Image, ImageFont
from torch.utils.data import Dataset

from data import *
from model import utils
from model.utils import pad_sequences_1d

def collate_fn(batch):
    batched_data = dict()

    inputs_keys = batch[0].keys()
    pad_token_id = batch[0]['pad_token_id']
    
    if 'metadata' in inputs_keys:
        batched_data['metadata'] = [e['metadata'] for e in batch]
    
    if isinstance(batch[0]['output_embeds'], str):
        output_embeds = [e['output_embeds'] for e in batch]
        batched_data['output_embeds'] = output_embeds
        # output_embeds = get_txt_embeds([e['output_embeds'] for e in batch])
    elif batch[0]['output_embeds'] is None:
        batched_data['output_embeds'], batched_data['output_masks'] = None, None
    else:
        batched_data['output_embeds'], batched_data['output_masks'] = pad_sequences_1d([e for e in output_embeds], dtype=torch.float32, fixed_length=None)

    batched_data['input_embeds'], batched_data['input_masks'] = pad_sequences_1d([e['input_embeds'] for e in batch], dtype=torch.float32, fixed_length=None, padded_sides="left")
    # left to ensure the last token is EOS;
    batched_data['query_ids'], batched_data['query_masks'] = pad_sequences_1d([e['query_ids'] for e in batch], dtype=torch.long, fixed_length=None, padded_values=pad_token_id, padded_sides="left")

    if batch[0]['response_ids'] is not None:
        batched_data['response_ids'], batched_data['response_masks'] = pad_sequences_1d([e['response_ids'] for e in batch], dtype=torch.long, fixed_length=None, padded_values=pad_token_id, padded_sides="right")
    
    return batched_data

class HybridDataset(torch.utils.data.Dataset):
    def __init__(self, datasets):
        self.datasets = datasets
        self.length = sum(len(ds) for ds in datasets)
        print(f"HybridDataset: {self.length} samples in total")
        for ds in datasets:
            print(f"Dataset {ds}: {len(ds)} samples")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        current_idx = idx
        for ds in self.datasets:
            if current_idx < len(ds):
                return ds[current_idx]
            current_idx -= len(ds)
        raise IndexError("Index out of range")
        
def get_dataset(args, split: str, tokenizer, precision: str = 'fp32', vocab_model=None) -> Dataset:
    if split == 'train': 
        dataset_list = args.dataset.split(',')
        metadata_list = args.metadata.split(',')
    else:
        dataset_list = args.val_dataset.split(',')
        metadata_list = args.val_metadata.split(',')

    datasets = []
    for metadata, dataset_type in zip(metadata_list, dataset_list):
        dataset_type = dataset_type.strip()
        if dataset_type == 'ret':
            datasets.append(RetDataset(
                dataset_dir=args.dataset_dir,
                metadata=metadata, 
                tokenizer=tokenizer,
                split=split, 
                num_frame=args.num_frame,
                max_len=args.max_len, 
                max_clip_len=args.max_clip_len,
                precision=args.precision, 
                add_eos=args.add_eos,
                hidden_dim=args.hidden_dim,
            ))
        elif dataset_type == 'gen':
            datasets.append(GenDataset(
                dataset_dir=args.dataset_dir,
                metadata=metadata, 
                tokenizer=tokenizer,
                split=split, 
                num_frame=args.num_frame,
                max_len=args.max_len, 
                max_clip_len=args.max_clip_len,
                precision=args.precision, 
                add_eos=args.add_eos,
                # vocab_model=vocab_model,
                hidden_dim=args.hidden_dim,
            ))
        elif dataset_type == 'coin':
            datasets.append(COIN(
                dataset_dir=args.dataset_dir,
                metadata=metadata, 
                tokenizer=tokenizer,
                split=split, 
                num_frame=args.num_frame,
                max_len=args.max_len, 
                max_clip_len=args.max_clip_len,
                precision=args.precision, 
                add_eos=args.add_eos,
                hidden_dim=args.hidden_dim,
            ))
        else:
            raise ValueError(f"Invalid dataset type: {dataset_type}")

    if len(datasets) == 1:
        return datasets[0]
    else:
        return HybridDataset(datasets)