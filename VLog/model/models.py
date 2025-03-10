import os
import pdb
import json
import glob
import math
import numpy as np
import pickle as pkl
from peft import PeftModel, get_peft_model, LoraConfig
from einops import rearrange
from functools import partial
from collections import namedtuple
from PIL import Image, UnidentifiedImageError
from typing import Callable, List, Optional, Tuple, Union

# import clip
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from model import utils
from model.transformer import build_transformer
from model.position_encoding import build_position_encoding
from model.utils import pad_sequences_1d

# Refer to the usage of LLaMa
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
# https://huggingface.co/docs/transformers/main_classes/output#transformers.modeling_outputs.CausalLMOutput

class ModelArgs:
    def __init__(self, 
                 llm_model: str = 'gpt2',
                 rand_init: bool = False,
                 lora: bool = False,
                 vis_model: str = 'openai/clip-vit-base-patch32', 
                 llm_8bit: bool = False,
                 local_rank: int = -1,
                 freeze_lm: bool = True, 
                 freeze_vm: bool = True,
                 add_special_tokens: bool = False,
                 num_layers: int = -1,
                 hidden_dim: int = 512,
                 nheads: int = 8,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1,
                 droppath: float = 0.1,
                 visual_input: str = 'pixel', 
                 n_visual_tokens: int = 1, 
                 vis_pooling: bool = False,
                 vis_query_pooling: bool = False,
                 last_vis_mean: bool = False
                 ):
        self.llm_model = llm_model
        self.rand_init = rand_init
        self.lora = lora
        self.vis_model = vis_model
        self.llm_8bit = llm_8bit
        self.device_map = {"": local_rank if local_rank != -1 else 0}
        self.freeze_lm = freeze_lm
        self.freeze_vm = freeze_vm
        self.add_special_tokens = add_special_tokens
        
        # adapter for video tokens temporal fusion
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.nheads = nheads
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.droppath = droppath
        
        self.visual_input = visual_input
        self.n_visual_tokens = n_visual_tokens

        self.vis_pooling = vis_pooling
        self.vis_query_pooling = vis_query_pooling
        self.last_vis_mean = last_vis_mean
        
    def __str__(self):
        return str(self.__dict__)
    
class VLog(nn.Module):
    def __init__(self, tokenizer, args: ModelArgs = ModelArgs(),
                 pooling: str = 'last',
                 normalize: bool = False):
        super().__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.visual_input = args.visual_input
        self.lora_config = LoraConfig(
            task_type="CAUSAL_LM",
        )
        self.pooling = pooling
        self.normalize = normalize

        self.use_adapter = args.num_layers > 0
        ##########################
        # * LLM brancha
        ##########################
        # dtype = utils.convert_precision_dtype(self.args.precision)
        print(f"Using {self.args.llm_model} for the language model.")

        self.lm = AutoModelForCausalLM.from_pretrained(self.args.llm_model)
        if self.args.rand_init:
            print("Randomly initializing the LM.")
            self.lm.init_weights()

        if self.args.freeze_lm:
            self.lm.eval()  # all fix
            print("Freezing the LM.")
            for param in self.lm.parameters():
                param.requires_grad = False
        else:
            self.lm.train() # all train

        if self.args.lora:
            print("Using LoRA for the LLM.")
            self.lm = get_peft_model(self.lm, self.lora_config)

        # If insert new token, we need to resize the LM token embeddings
        # inspired by  https://github.com/haotian-liu/LLaVA/blob/main/llava/model/llava.py
        if self.args.add_special_tokens:
            self.num_new_tokens = tokenizer.add_tokens(["<vis>", "</vis>"], special_tokens=True)
            self.lm.resize_token_embeddings(len(tokenizer))
            
            input_embeddings = self.lm.get_input_embeddings().weight.data
            output_embeddings = self.lm.get_output_embeddings().weight.data
            
            input_embeddings_avg = input_embeddings[:-self.num_new_tokens].mean(dim=0, keepdim=True)
            output_embeddings_avg = output_embeddings[:-self.num_new_tokens].mean(dim=0, keepdim=True)

            input_embeddings[-self.num_new_tokens:] = input_embeddings_avg
            output_embeddings[-self.num_new_tokens:] = output_embeddings_avg
            
            self.orig_embeds_params = self.lm.get_input_embeddings().weight.data.clone()
            for p in self.lm.get_output_embeddings().parameters():
                p.requires_grad = False
            for p in self.lm.get_input_embeddings().parameters(): # print(p.requires_grad)
                p.requires_grad = True

        self.lm_embeddings = self.lm.get_input_embeddings()		# (#vocab, 4096)

        ##########################
        # * Vision branch
        ##########################
        # print(f"Using {self.args.vis_model} for the visual model with {self.args.n_visual_tokens} visual tokens.")
        # if 'clip' in self.args.vis_model:
        #     # self.vm = CLIPVisionModel.from_pretrained(self.args.vis_model)
        #     self.vm = CLIPVisionModelWithProjection.from_pretrained(self.args.vis_model)
        # else:
        #     raise NotImplementedError
        
        # if self.args.freeze_vm:
        #     print("Freezing the VM.")
        #     self.vm.eval()
        #     for param in self.vm.parameters():
        #         param.requires_grad = False
        # else:
        #     self.vm.train()

        if self.use_adapter:
            self.vm_adapter = build_transformer(self.args)

        # vm_projection_dim = self.vm.config.projection_dim	# 512 / 768
        vm_projection_dim = self.args.hidden_dim
        vm_embedding_dim = self.lm_embeddings.embedding_dim * self.args.n_visual_tokens		# 4096 * n
        
        self.pos_embed = build_position_encoding(vm_projection_dim)         # positional encoding dimension
        self.in_projector = nn.Linear(vm_projection_dim, vm_embedding_dim)	# (256, 768)
        self.out_projector = nn.Linear(vm_embedding_dim, vm_projection_dim)	# (768, 256)
            
    # Extract visual embeddings from the vision encoder.
    def get_vid_embeds(self, vid_inputs: torch.FloatTensor, vid_masks=None):
        if self.visual_input == 'pixel':
            # if len(vid_inputs.shape) == 5:  # videos of multiple frames (bs, t, c, h, w)
            #     bs, t, c, h, w = vid_inputs.shape
            #     vid_inputs_flatten = vid_inputs.reshape(-1, c, h, w)
            #     outputs = self.vm(vid_inputs_flatten)
            #     # encoder_outputs_flatten = outputs.pooler_output
            #     encoder_outputs_flatten = outputs.image_embeds
            #     encoder_outputs = encoder_outputs_flatten.reshape(bs, t, -1)
            # elif len(vid_inputs.shape) == 4:  # images (bs, c, h, w)
            #     outputs = self.vm(vid_inputs)
            #     # encoder_outputs = outputs.pooler_output
            #     encoder_outputs = outputs.image_embeds
            # else:
            #     raise NotImplementedError
            raise NotImplementedError

        elif self.visual_input == 'feature':
            if len(vid_inputs.shape) == 3:  # features (bs, t, c)
                encoder_outputs = vid_inputs
            elif len(vid_inputs.shape) == 2:    # features (t, c)
                encoder_outputs = vid_inputs[None,...]
            else:
                raise NotImplementedError

        # video temporal adapter
        if self.use_adapter:
            vid_pos = self.pos_embed(encoder_outputs, vid_masks)
            vid_pos = vid_pos.to(encoder_outputs.dtype)
            encoder_outputs = self.vm_adapter(encoder_outputs, ~vid_masks.bool(), vid_pos)

        # (bs, t, 512) -> (bs, t, 4096 * n)        
        vid_embeds = self.in_projector(encoder_outputs)
        if self.args.n_visual_tokens > 1:
            vid_masks = vid_masks.repeat_interleave(self.args.n_visual_tokens, dim=1)
        # (bs, t, 4096 * n) -> (bs, t * n, 4096)
        vid_embeds = torch.reshape(vid_embeds, (vid_embeds.shape[0], -1, self.lm_embeddings.embedding_dim ))
        return vid_embeds, vid_masks

    # Extract textual embeddings from the llm embedding layer.
    def get_txt_embeds(self, token_values: torch.LongTensor):
        return self.lm_embeddings(token_values)

    # add start and end tokens to the visual tokens
    def add_special_tokens(self, vid_embeds, vid_masks):
        bs = vid_embeds.shape[0]
        vis_start_token, vis_end_token = self.tokenizer.convert_tokens_to_ids(["<vis>", "</vis>"])
        tmp = self.lm_embeddings(torch.LongTensor([vis_start_token, vis_end_token]).to(vid_embeds.device))
        vis_start_embeds, vis_end_embeds = tmp[0].repeat(bs,1,1), tmp[1].repeat(bs,1,1)
        vis_start_masks, vis_end_masks = torch.ones_like(vis_start_embeds[:,:,0]), torch.ones_like(vis_end_embeds[:,:,0])
        vid_embeds_aug = torch.cat([vis_start_embeds, vid_embeds, vis_end_embeds], dim=1)
        vid_masks_aug = torch.cat([vis_start_masks, vid_masks, vis_end_masks], dim=1)
        return vid_embeds_aug, vid_masks_aug

    def _pooling(self, last_hidden_state, attention_mask):
        if self.pooling == 'last':
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_state.shape[0]
            reps = last_hidden_state[
                    torch.arange(batch_size, device=last_hidden_state.device), sequence_lengths]
        else:
            raise NotImplementedError
        if self.normalize:
            reps = torch.nn.functional.normalize(reps, p=2, dim=-1)
        return reps

    def encode_input(self, input):
        hidden_states = self.lm(**input, return_dict=True, output_hidden_states=True)
        hidden_states = hidden_states.hidden_states[-1]
        pooled_output = self._pooling(hidden_states, input['attention_mask'].bool())
        return pooled_output

    def train(self, mode=True):
        super(VLog, self).train(mode=mode)
        # first try to freeze all parameters then we activate part of them.
        if self.args.freeze_lm:
            self.lm.eval()

        # activate lora layers if lora is enabled
        if self.args.lora:
            for name, param in self.lm.named_parameters():
                if 'lora' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        # activate embedding layers
        # for p in self.lm.get_output_embeddings().parameters():
        #     p.requires_grad = False
        # for p in self.lm.get_input_embeddings().parameters():
        #     p.requires_grad = True

        if self.args.freeze_vm:
            self.vm.eval()

    def lock_lm(self):
        for param in self.lm.parameters():
            param.requires_grad = False

    def save_pretrained(self, save_directory):
        os.makedirs(save_directory, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(save_directory, "pytorch_model.bin"))

    @classmethod
    def from_pretrained(cls, load_directory):
        model = cls()
        model.load_state_dict(torch.load(os.path.join(load_directory, "pytorch_model.bin")))
        return model

    # template: [input_embeds] [input_ids] <-> [output_embeds] ([output_ids])
    def forward(self,
        input_embeds: torch.FloatTensor,
        input_masks: torch.LongTensor,
        query_ids: torch.LongTensor,
        query_masks: torch.LongTensor,
        output_embeds: str = None,
        output_masks: torch.LongTensor = None,
        response_ids: torch.LongTensor = None,
        response_masks: torch.LongTensor = None,
        generate: bool = False,
        temperature: float = 1.0,
        do_sample: bool = False,
        max_len: int = 128,
        narr_inputs: bool = None
        ):
        
        input_embeds, input_masks = self.get_vid_embeds(input_embeds, input_masks)

        if self.args.add_special_tokens:
            with torch.no_grad():
                self.lm.get_input_embeddings().weight.data[:-2] = self.orig_embeds_params[:-2].data
            input_embeds, input_masks = self.add_special_tokens(input_embeds, input_masks)
        input_labels = torch.zeros_like(input_masks, dtype=torch.int64).to(input_embeds.device) - 100

        # if query_ids is not None:
        #     query_labels = torch.zeros_like(query_ids, dtype=torch.int64).to(input_embeds.device) - 100
        #     query_embeds = self.get_txt_embeds(query_ids)
            
        #     cond_embeds = torch.cat([input_embeds, query_embeds], axis=1)
        #     cond_masks = torch.cat([input_masks, query_masks], axis=1)
        #     cond_labels = torch.cat([input_labels, query_labels], axis=1)
        # else:
        #     cond_embeds = input_embeds
        #     cond_masks = input_masks
        #     cond_labels = input_labels

        # must (at least) have query_ids as [EOS] token for retrieval learning;
        query_labels = torch.zeros_like(query_ids, dtype=torch.int64).to(input_embeds.device) - 100
        query_embeds = self.get_txt_embeds(query_ids)

        cond_embeds = torch.cat([input_embeds, query_embeds], axis=1)
        cond_masks = torch.cat([input_masks, query_masks], axis=1)
        cond_labels = torch.cat([input_labels, query_labels], axis=1)

        if self.args.last_vis_mean:
            mean_embeds = torch.mean(input_embeds, 1).unsqueeze(1)
            mean_masks = torch.ones(mean_embeds.shape[0], 1).to(input_embeds.device)
            mean_labels = torch.zeros_like(mean_masks, dtype=torch.int64).to(input_embeds.device) - 100
            cond_embeds = torch.cat([cond_embeds, mean_embeds], axis=1)
            cond_masks = torch.cat([cond_masks, mean_masks], axis=1)
            cond_labels = torch.cat([cond_labels, mean_labels], axis=1)

        # generative
        if response_ids is not None:
            res_embeds = self.get_txt_embeds(response_ids)
            res_masks = response_masks
            res_labels = response_ids

            full_embeds = torch.cat([cond_embeds, res_embeds], axis=1)
            full_masks = torch.cat([cond_masks, res_masks], dim=1)
            full_labels = torch.cat([cond_labels, res_labels], axis=1)

            if generate:
                bs = cond_embeds.shape[0]
                device = cond_embeds.device
                generated_ids_pad = torch.full((bs, max_len), self.tokenizer.pad_token_id).to(device)
                groundtruth_ids_pad = torch.full((bs, max_len), self.tokenizer.pad_token_id).to(device)

                with torch.no_grad():
                    model_outputs = self.lm(inputs_embeds=full_embeds, attention_mask=full_masks, labels=full_labels, output_hidden_states=True)
                    generated_ids = self.lm.generate(inputs_embeds=cond_embeds, attention_mask=cond_masks, 
                                                    pad_token_id=self.tokenizer.pad_token_id,
                                                    temperature=temperature, max_new_tokens=max_len)

                seq_len_gen = min(max_len, generated_ids.shape[1])
                seq_len_gt = min(max_len, response_ids.shape[1])
                
                generated_ids_pad[:, :seq_len_gen] = generated_ids[:, :seq_len_gen]
                groundtruth_ids_pad[:, :seq_len_gt] = response_ids[:, :seq_len_gt]
                return  model_outputs, generated_ids_pad, groundtruth_ids_pad
        
            model_outputs = self.lm(inputs_embeds=full_embeds, 
                                    attention_mask=full_masks, 
                                    labels=full_labels)
            return model_outputs

        # contrastive
        else:
            if self.args.vis_pooling:
                masked_embeds = input_embeds * input_masks.unsqueeze(-1)
                sum_mask = input_masks.sum(dim=1, keepdim=True)
                sum_mask = torch.clamp(sum_mask, min=1e-6)
                pooled_output = masked_embeds.sum(dim=1) / sum_mask
                pooled_output = pooled_output.to(input_embeds.dtype)
                ret_embeds = self.out_projector(pooled_output)
                return ret_embeds

            if self.args.vis_query_pooling:
                masked_embeds = cond_embeds * cond_masks.unsqueeze(-1)
                sum_mask = cond_masks.sum(dim=1, keepdim=True)
                sum_mask = torch.clamp(sum_mask, min=1e-6)
                pooled_output = masked_embeds.sum(dim=1) / sum_mask
                pooled_output = pooled_output.to(cond_embeds.dtype)
                ret_embeds = self.out_projector(pooled_output)
                return ret_embeds

            input_dict = dict(inputs_embeds=cond_embeds, attention_mask=cond_masks)
            last_embeds = self.encode_input(input_dict)
            ret_embeds = self.out_projector(last_embeds)
            return ret_embeds

        # # output the generated ids for decoding
        # if generate:
        #     raise NotImplementedError

        # res_embeds = self.get_txt_embeds(response_ids)
        # res_masks = response_masks
        # res_labels = response_ids

        # full_embeds = torch.cat([cond_embeds, res_embeds], axis=1)
        # full_masks = torch.cat([cond_masks, res_masks], dim=1)
        # full_labels = torch.cat([cond_labels, res_labels], axis=1)

        # model_outputs = self.lm(inputs_embeds=full_embeds,
        #                     attention_mask=full_masks,
        #                     labels=full_labels,
        #                     output_hidden_states=True)
        # # last_embeddings = model_outputs['hidden_states'][-1]
        # return model_outputs, full_labels

def load_model(model_dir, pth='ckpt_best.pth.tar', visual_input='feature'):
    model_args_path = os.path.join(model_dir, 'model_args.json')
    model_ckpt_path = os.path.join(model_dir, pth)
    if not os.path.exists(model_args_path):
        raise ValueError(f'model_args.json does not exist in {model_dir}.')
    if not os.path.exists(model_ckpt_path):
        raise ValueError(f'pretrained_ckpt.pth.tar does not exist in {model_dir}.')

    with open(model_args_path, 'r') as f:
        model_kwargs = json.load(f)
    model_kwargs['visual_input'] = visual_input
    
    # Initialize tokenizer.
    if model_kwargs['llm_model'] in ['gpt2']:
        model = AutoModelForCausalLM.from_pretrained(model_kwargs['llm_model']) # torch_dtype=torch.bfloat16)
        tokenizer = AutoTokenizer.from_pretrained(model_kwargs['llm_model']) 
    else:
        raise ValueError(f"Model {model_kwargs['llm_model']} not supported.")

    if tokenizer.pad_token is None:
        tokenizer.pad_token_id = (0)
    # if model_kwargs['add_special_tokens']:
    #     num_new_tokens = tokenizer.add_tokens(["<vis>", "</vis>"], special_tokens=True)

    # Initialize model for inference.
    args = namedtuple('args', model_kwargs)(**model_kwargs)
    my_model = VLog(tokenizer, args)
    my_model.eval()
    
    my_model = my_model.bfloat16()
    my_model = my_model.cuda()

    checkpoint = torch.load(model_ckpt_path)
    checkpoint['state_dict'] = {k.replace('module.',''):v for k,v in checkpoint['state_dict'].items()}
    my_model.load_state_dict(checkpoint['state_dict'], strict=False)
    return my_model, tokenizer