import os
import sys
import pdb
import time
import json
from datetime import datetime
import warnings
import numpy as np
from PIL import Image
from tqdm import tqdm

import wandb
import torch
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.utils.data
import torch.utils.data.distributed
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.optim.lr_scheduler import StepLR
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from transformers import AutoTokenizer
from warmup_scheduler import GradualWarmupScheduler

from config import parse_args
from data import dataloader
from main.trainer_ret import train_epoch as train_epoch_ret
from main.trainer_gen import train_epoch as train_epoch_gen
from main.eval_ret import val_epoch as val_epoch_ret
from main.eval_ret_cls import val_epoch as val_epoch_cls
from main.eval_gen import val_epoch as val_epoch_gen
from model import models
from model import utils

os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)

wandb.login(key=os.environ["WANDB_KEY"])

def main(args):
    args = parse_args(args)

    if 'LOCAL_RANK' in os.environ:
        if int(os.environ["LOCAL_RANK"]) in [-1, 0]:
            args.main_node = True
    elif args.local_rank in [-1, 0]:
        args.main_node = True
    else:
        args.main_node = False

    logger = None
    if args.main_node:
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        args.log_dir = os.path.join(args.log_base_dir, args.exp_id, timestamp)
        os.makedirs(args.log_dir, exist_ok=True)
        if not args.debug:
            wandb.init(
                project="VLog",
                group=args.exp_id,
                name=f'{args.exp_id}_{timestamp}',
                dir=args.log_dir,
                config=args
            )

        with open(os.path.join(args.log_dir, f'args.json'), 'w') as wf:
            json.dump(vars(args), wf, indent=4)

        with open(os.path.join(args.log_dir, f'git_info.txt'), 'w') as wf:
            utils.dump_git_status(out_file=wf)

        print(f'Logging to {args.log_dir}.')
        logger = utils.create_logger(args.log_dir + '/train.log')

        print(f"Start job {args.exp_id}")

    if args.seed is not None:
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                        'This will turn on the CUDNN deterministic setting, '
                        'which can slow down your training considerably! '
                        'You may see unexpected behavior when restarting '
                        'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                        'disable data parallelism.')

    if "WORLD_SIZE" in os.environ:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size >= 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args, logger)

    if args.main_node == 0:
        if not args.debug:
            wandb.finish()
            # writer.close()

def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Conv1d)):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)

def main_worker(gpu, ngpus_per_node, args, logger=None):
    """Setup code."""
    global best_score
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        # if args.dist_url == "env://" and args.local_rank == -1:
        if args.local_rank == -1:
            args.local_rank = int(os.environ["LOCAL_RANK"])
            print("Local Rank: ", args.local_rank)
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, local_rank needs to be the
            # global local_rank among all the processes
            args.local_rank = args.local_rank * ngpus_per_node + gpu
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend=args.dist_backend, 
                                # init_method=args.dist_url,
                                world_size=args.world_size, rank=args.local_rank) #, timeout=datetime.timedelta(seconds=100))

    # Create model
    model_args = models.ModelArgs(
        llm_model = args.llm_model,
        rand_init = args.rand_init,
        lora = args.lora,
        vis_model = args.vis_model,
        llm_8bit = args.llm_8bit,
        local_rank = args.local_rank,
        freeze_lm = args.freeze_lm,
        freeze_vm = False,
        add_special_tokens = args.add_special_tokens,
        num_layers = args.num_layers,
        hidden_dim = args.hidden_dim,
        nheads = args.nheads,
        dim_feedforward = args.dim_feedforward,
        dropout = args.dropout,
        droppath = args.droppath,
        visual_input = args.visual_input,
        n_visual_tokens = args.n_visual_tokens,
        vis_pooling = args.vis_pooling,
        last_vis_mean = args.last_vis_mean,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.llm_model, use_fast=False)

    if args.add_eos:
        tokenizer.add_eos_token = True
    if tokenizer.pad_token is None:
        tokenizer.pad_token_id = (0)
    
    # Save model args to disk.
    if args.main_node:
        with open(os.path.join(args.log_dir, 'model_args.json'), 'w') as f:
            json.dump(vars(model_args), f, indent=4)

    model = models.VLog(tokenizer, model_args)
    if args.precision == 'fp16':
        model = model.float()
    elif args.precision == 'bf16':
        model = model.bfloat16()
    
    # to freeze the get_input / output embeddings parameters after pretraining.
    if args.lock_lm:
        model.lock_lm()
    
    # Print parameters and count of model.
    if args.main_node:
        param_counts_text = utils.get_params_count_str(model)
        with open(os.path.join(args.log_dir, 'param_count.txt'), 'w') as f:
            f.write(param_counts_text)

    # Log trainable parameters to Tensorboard.
    if args.main_node:
        _, total_trainable_params, total_nontrainable_params = utils.get_params_count(model)
        writer = SummaryWriter(os.path.join(args.log_dir, 'tensorboard'))
        writer.add_scalar('params/total', total_trainable_params + total_nontrainable_params, 0)
        writer.add_scalar('params/total_trainable', total_trainable_params, 0)
        writer.add_scalar('params/total_non_trainable', total_nontrainable_params, 0)
    else:
        writer = None
    
    # load vocabulary dual encoder
    if args.vocab_model == "lavila":
        from model.lavila import Lavila
        vocab_model = Lavila(args.local_rank)
    elif args.vocab_model == "google/siglip-so400m-patch14-384":
        from model.siglip import Siglip
        vocab_model = Siglip(args.vocab_model, args.local_rank)

    # Data loading code
    train_dataset = dataloader.get_dataset(args, 'train', tokenizer, vocab_model=vocab_model)
    val_dataset = dataloader.get_dataset(args, 'test', tokenizer, vocab_model=vocab_model)

    if args.train_class and hasattr(train_dataset, 'candidate'):
        class_list = train_dataset.candidate
        args.class_list = class_list
    else:
        args.class_list = None

    if hasattr(val_dataset, 'candidate'):
        candidate = val_dataset.candidate
        args.candidate = candidate
    else:
        args.candidate = None

    if args.main_node:
        logger.info(f'Training with {len(train_dataset)} examples and validating with {len(val_dataset)} examples.')

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, drop_last=False)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False, drop_last=False)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, collate_fn=dataloader.collate_fn,
        # multiprocessing_context=mp_ctx
        )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=(args.val_batch_size or args.batch_size), shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=val_sampler, collate_fn=dataloader.collate_fn,
        # multiprocessing_context=mp_ctx
        )

    # steps-per-epoch
    if args.steps_per_epoch == -1:
        args.steps_per_epoch = len(train_loader)

    if not torch.cuda.is_available():
        print('WARNING: using CPU, this will be slow!')
        model = torch.nn.DataParallel(model)
    elif args.distributed:
        device = torch.device("cuda", args.local_rank)
        model.to(device)
        # DistributedDataParallel will divide and allocate batch_size to all
        # available GPUs if device_ids are not set
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        model = torch.nn.DataParallel(model).cuda()    

    # define loss function (criterion), optimizer, and learning rate scheduler
    if args.loss == "nce":
        from model.losses import nce
        criterion = nce.NormSoftmaxLoss().cuda(args.nce_temperature)
    else:
        raise NotImplementedError

    optimizer_cls = torch.optim.AdamW
    if args.main_node:
        print('Using torch.optim.AdamW as the optimizer.')
    optimizer = optimizer_cls(model.parameters(), args.lr,
                                betas=(args.beta1, args.beta2),
                                weight_decay=args.weight_decay,
                                eps=1e-8)

    scheduler_steplr = StepLR(optimizer, step_size=args.lr_schedule_step_size * args.steps_per_epoch, gamma=args.lr_schedule_gamma)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1.0, total_epoch=args.lr_warmup_steps, after_scheduler=scheduler_steplr)

    # optionally resume from a checkpoint
    if args.resume:
        # only load model weights from pretraining phase
        if os.path.isfile(args.resume):
            # if args.local_rank > 0:
            if args.distributed:
                device_id = 'cuda:{}'.format(args.local_rank)
                checkpoint = torch.load(args.resume, map_location=device_id)
            else:
                checkpoint = torch.load(args.resume)
                checkpoint['state_dict'] = {k.replace('module.',''):v for k,v in checkpoint['state_dict'].items()}
            model.load_state_dict(checkpoint['state_dict'], strict=False)
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
        # model, _ = models.load_model(args.resume, 'ckpt_12_best.pth.tar')
        # model.to(device)

    cudnn.benchmark = True
    # filter trainable model parameters
    args.trainable_key = utils.params_to_save(model)
    trainable_state_dict = {k: v for k, v in model.state_dict().items() if k in args.trainable_key}
    
    dataset = set(args.dataset.split(','))
    # print(dataset)
    train_epoch = train_epoch_ret
    for x in dataset:
        if x in ['gen']:
            train_epoch = train_epoch_gen
            break

    val_dataset = args.val_dataset.split(',')
    assert len(val_dataset) == 1
    val_dataset = val_dataset[0]
    if val_dataset in ['gen']:
        val_epoch = val_epoch_gen
    elif val_dataset in ['ret']:
        val_epoch = val_epoch_ret
    elif val_dataset in ['coin']:
        val_epoch = val_epoch_cls
    else:
        raise NotImplementedError

    best_score = 0
    for epoch in range(args.start_epoch, args.epochs):
        if args.evaluate:
            eval_score = val_epoch(val_loader, model, vocab_model, tokenizer, criterion, epoch, 0, writer, args, logger)
            return eval_score
        
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # pdb.set_trace()
        # train for one epoch
        global_step = train_epoch(train_loader, model, vocab_model, tokenizer, criterion, optimizer, epoch, scheduler, writer, args, logger)

        # evaluate on validation set
        eval_score = val_epoch(val_loader, model, vocab_model, tokenizer, criterion, epoch, global_step, writer, args, logger)

        if args.main_node:
            # remember best score and save checkpoint
            is_best = eval_score > best_score
            best_score = max(eval_score, best_score)

            save_or_not = is_best if args.only_best else True
            if save_or_not:
                # if args.distributed:
                #     model.module.save_pretrained(os.path.join(args.log_dir, f'ckpt_{epoch+1}'))
                # else:
                #     model.save_pretrained(os.path.join(args.log_dir, f'ckpt_{epoch+1}'))
                utils.save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': trainable_state_dict, # only save trainable parameters for efficient
                    'best_score': best_score,
                    'optimizer' : optimizer.state_dict(),
                    'scheduler' : scheduler.state_dict()
                }, is_best, os.path.join(args.log_dir,  f'ckpt_{epoch+1}'))

    if args.main_node:
        writer.close()

if __name__ == '__main__':
    main(sys.argv[1:])