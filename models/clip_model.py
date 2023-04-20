import os
import cv2
import pdb
import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPVisionModelWithProjection
from transformers import logging
logging.set_verbosity_error()

class FeatureExtractor():
    def __init__(self, args):
        self.device = args.feature_extractor_device
        self.beta = args.beta
        self.processor = CLIPProcessor.from_pretrained(args.feature_extractor)
        self.model = CLIPVisionModelWithProjection.from_pretrained(args.feature_extractor).to(self.device)
        self.data_dir = args.data_dir
        self.tmp_dir = args.tmp_dir


    def __call__(self, video_path, video_id):        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_length = frame_count / fps
        sample_rate = int(fps) * self.beta

        save_path = os.path.join(self.tmp_dir, video_id + '.npz')
        if os.path.exists(save_path):
            data = np.load(save_path)
            clip_features = data['features']    
            return clip_features, video_length

        clip_features = []
        print("Extract the clip feature.")
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if cap.get(cv2.CAP_PROP_POS_FRAMES) % sample_rate == 0:
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                inputs = self.processor(images=image, return_tensors="pt").pixel_values
                inputs = inputs.to(self.device)
                
                with torch.no_grad():
                    feat = self.model(inputs)['image_embeds']
                    clip_features.append(feat.cpu().numpy())
        print("Finished.")

        clip_features = np.concatenate(clip_features, axis=0)
        np.savez_compressed(save_path, features=clip_features)
        
        return clip_features, video_length
