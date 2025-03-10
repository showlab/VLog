import os
import cv2
import pdb
import sys
import time
import numpy as np
from transformers import logging
logging.set_verbosity_error()

from models.kts_model import VideoSegmentor
from models.clip_model import FeatureExtractor
from models.blip2_model import ImageCaptioner
from models.grit_model import DenseCaptioner
from models.whisper_model import AudioTranslator
from models.gpt_model import LlmReasoner
from utils.utils import logger_creator, format_time


class Vlogger :
    def __init__(self, args):
        self.args = args
        self.alpha = args.alpha
        self.beta = args.beta
        self.data_dir = args.data_dir
        self.tmp_dir = args.tmp_dir
        self.models_flag = False
        self.init_llm()
        if not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)
        
    def init_models(self): 
        print('\033[1;34m' + "Welcome to the our Vlog toolbox...".center(50, '-') + '\033[0m')
        print('\033[1;33m' + "Initializing models...".center(50, '-') + '\033[0m')
        print('\033[1;31m' + "This may time-consuming, please wait...".center(50, '-') + '\033[0m')
        self.feature_extractor = FeatureExtractor(self.args)
        self.video_segmenter = VideoSegmentor(alpha=self.alpha, beta=self.beta)
        self.image_captioner = ImageCaptioner(model_name=self.args.captioner_base_model, device=self.args.image_captioner_device)
        self.dense_captioner = DenseCaptioner(device=self.args.dense_captioner_device)
        self.audio_translator = AudioTranslator(model=self.args.audio_translator, device=self.args.audio_translator_device)
        print('\033[1;32m' + "Model initialization finished!".center(50, '-') + '\033[0m')
    
    def init_llm(self):
        print('\033[1;33m' + "Initializing LLM Reasoner...".center(50, '-') + '\033[0m')
        os.environ["OPENAI_API_KEY"] = self.args.openai_api_key
        self.llm_reasoner = LlmReasoner(self.args)
        print('\033[1;32m' + "LLM initialization finished!".center(50, '-') + '\033[0m')

    def video2log(self, video_path): 
        video_path = video_path
        video_id = os.path.basename(video_path).split('.')[0]
        if self.llm_reasoner.exist_vectorstore(video_id):
            return self.printlog(video_id)
        try:
            self.llm_reasoner.create_vectorstore(video_id)
            return self.printlog(video_id)
        except:
            pass        

        if not self.models_flag:
            self.init_models()
            self.models_flag = True
            
        logger = logger_creator(video_id)
        clip_features, video_length = self.feature_extractor(video_path, video_id)
        seg_windows = self.video_segmenter(clip_features, video_length)
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        audio_results = self.audio_translator(video_path)

        for start_sec, end_sec in seg_windows:
            middle_sec = (start_sec + end_sec) // 2
            middle_frame_idx = int(middle_sec * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_idx)
            ret, frame = cap.read()
            
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image_caption = self.image_captioner.image_caption(frame)
                dense_caption = self.dense_captioner.image_dense_caption(frame)
                audio_transcript = self.audio_translator.match(audio_results, start_sec, end_sec)
                    
                logger.info(f"When {format_time(start_sec)} - {format_time(end_sec)}")
                logger.info(f"I saw {image_caption}.")
                logger.info(f"I found {dense_caption}.")
                    
                if len(audio_transcript) > 0:
                    logger.info(f"I heard someone say \"{audio_transcript}\"")
                logger.info("\n")
        
        cap.release()
        self.llm_reasoner.create_vectorstore(video_id)
        return self.printlog(video_id)
        
    def printlog(self, video_id):
        log_list = []
        log_path = os.path.join(self.data_dir, video_id + '.log')
        with open(log_path, 'r') as f:
            for line in f:
                log_list.append(line.strip())
        return log_list
    
    def chat2video(self, user_input):
        response = self.llm_reasoner(user_input)
        return response

    def clean_history(self):
        self.llm_reasoner.clean_history()
        return
