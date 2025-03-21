import os
import cv2
import sys
import json
import torch
import decord
import random
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse
from model import models
from tqdm import tqdm
from model.losses import nce
from model.siglip import Siglip
from model.models_vocab import VlogVocab
from model.utils import get_latest_ckpt_id, read_json, save_json, display_video_frames
import matplotlib.pyplot as plt

#################################################
# Set up the model, with checkpoint, with vidcab
#################################################
save_dir = 'pretrained'
args_url = os.path.join(save_dir, 'args.json')
ckpt_url = os.path.join(save_dir, 'vlog.pth.tar')
vocab_url = os.path.join(save_dir, 'vocab_egoclip.pt')

vlog_vocab = VlogVocab(args_url, ckpt_url, vocab_url)

if __name__ == "__main__":
    # Short video clip case:
    video_path = 'assets/3c0dffd0-e38e-4643-bc48-d513943dc20b_012_014.mp4'
    display_video_frames(video_path)
    clip_embed = vlog_vocab.encode_feature(video_path, fps=2)
    print(clip_embed.shape)

    # Stay tune for Long video case.

    query="What is the action in the video?"
    # query="What is the overall activity in the video?"
    # query="What is the next action in the video?"
    # query="What is the previous action in the video?"
    
    uid_list, topk_conf, topk_scene = vlog_vocab.get_topk_scene(clip_embed, topk=10)
    print("Scene: ", topk_scene[0])
    pred_dict = vlog_vocab.get_topn_vocab(clip_embed, uid_list, query=query)
    print("Top-Narr: ", pred_dict['topn_narr'])