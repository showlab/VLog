import argparse

llm_models = ['gpt2', 'gpt2-xl']

def parse_args(args):
    parser = argparse.ArgumentParser(description='VLog training')
    parser.add_argument('--exp_id', default='debug', type=str, help="Name of experiment, used for saving checkpoints.")
    parser.add_argument('--debug', action='store_true', default=False, help="debug mode for fast d evelopment.")

    # * Main model settings
    parser.add_argument('--llm_model', default='gpt2',	help='LLM model to use.')
    parser.add_argument('--llm_8bit', action='store_true', help="use load_in_8bit for LLM model.")
    parser.add_argument('--freeze_lm', action='store_true', help='whether to freeze the LLM during training.')
    parser.add_argument('--lora', action='store_true', help="use LoRA for LLM model.")
    parser.add_argument('--vis_model', default='openai/clip-vit-base-patch32', type=str, help="Visual encoder to use.")
    parser.add_argument('--precision', default='fp32', type=str, choices=['fp32', 'fp16', 'bf16'], help="Precision to visual inputs.")
    parser.add_argument('--n_visual_tokens', default=1, type=int, metavar='N', help='Number of visual tokens to use for each video frame / image.')
    parser.add_argument('--rand_init',  default=False, action='store_true')

    # * Video Temporal Adapter
    parser.add_argument('--num_layers', type=int, default=-1, help="Number of transformer layers of temporal adapter")
    parser.add_argument('--hidden_dim', type=int, default=1152, help="Hidden dim of temporal adapter, should align with the CLIP projector dim i.e. 512")
    parser.add_argument('--nheads', type=int, default=8, help="Number of heads of temporal adapter.")
    parser.add_argument('--dim_feedforward', type=int, default=2048, help="Dim of feedforward of temporal adapter")
    parser.add_argument('--dropout', type=float, default=0.1, help="Dropout probability of temporal adapter")
    parser.add_argument('--droppath', type=float, default=0.1, help="Droppath probability of temporal adapter")
    parser.add_argument('--vis_pooling', default=False, action='store_true')
    parser.add_argument('--vis_query_pooling', default=False, action='store_true')
    parser.add_argument('--last_vis_mean', default=False, action='store_true')

    # * Vocabulary settings
    parser.add_argument('--vocab_model', default='google/siglip-so400m-patch14-384',	choices=['lavila', 'google/siglip-so400m-patch14-384'])

    # * Dataset settings
    parser.add_argument('--dataset_dir', default='/blob/v-lqinghong/data/Ego_database', type=str, help='Dataset directory containing .tsv files.')
    parser.add_argument('--log_base_dir', default='/blob/v-lqinghong/experiments/VLog', type=str, help='Base directory to write logs and ckpts to.')

    parser.add_argument('--dataset', default='ret',  type=str)
    parser.add_argument('--metadata', default='egoclip_100k', type=str, help='specific metadata of training corpus')
    parser.add_argument('--fullset', default=False, action='store_true')

    parser.add_argument('--val_dataset', default='ret',  type=str)
    parser.add_argument('--val_metadata', default=None, type=str, help='specific metadata of evaluation')

    # * Image preprocessing
    parser.add_argument('--workers', default=4, type=int, metavar='N',	help='number of data loading workers (default: 4)')
    parser.add_argument('--visual_input', default='feature', choices=['pixel', 'feature'], help='video input may be RGB / CLIP features')
    parser.add_argument('--image_size', default=224, type=int, metavar='N', help='Size of images.')
    parser.add_argument('--num_frame', default=1, type=int, help='The number of video sample frames.')
    parser.add_argument('--add_special_tokens', action='store_true', help='whether to use <vid> and </vid> tokens.')
    parser.add_argument('--num_history', default=0, type=int, help='The number of video history.')
    parser.add_argument('--past_len', default=0, type=int, help='The number of video history.')
    parser.add_argument('--train_narrator', default="narration_pass_2", type=str)

    # * Text preprocessing
    parser.add_argument('--add_eos', default=True, action='store_false', help="add an eos token at the end of each text")
    parser.add_argument('--max_len', default=32, type=int, metavar='N', help='Maximum length to truncate captions / generations to.')
    parser.add_argument('--max_len_eval', default=32, type=int, metavar='N', help='Maximum length to truncate captions / generations to.')
    parser.add_argument('--max_clip_len', default=512, type=int, metavar='N', help='Maximum length to truncate captions / generations to.')
    parser.add_argument('--temperature', default=0.7, type=float, help='temperature for llm generation')

    # * Training settings
    parser.add_argument('--epochs', default=10, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
    parser.add_argument('--steps_per_epoch', default=-1, type=int, metavar='N', help='number of training steps per epoch')
    parser.add_argument('--val_steps_per_epoch', default=-1, type=int, metavar='N', help='number of validation steps per epoch.')
    parser.add_argument('--batch_size', default=180, type=int, metavar='N')
    parser.add_argument('--val_batch_size', default=180, type=int)
    parser.add_argument('--print_freq', default=10, type=int, metavar='N', help='print frequency (default: 10)')

    parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
    parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
    parser.add_argument('--only_best', action='store_false', help='Only save the model checkpoint with best score')

    parser.add_argument('--do_sample', action='store_true', help='evaluate model with do sample option')
    parser.add_argument('--lock_lm', action='store_true', help='whether to freeze the get_input / output embeddings parameters after pretraining.')

    # * Learning rate settings
    parser.add_argument('--lr', default=0.0003, type=float, metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--beta1', default=0.9, type=float, metavar='M', help='beta1 for Adam')
    parser.add_argument('--beta2', default=0.95, type=float, metavar='M', help='beta2 for Adam')
    parser.add_argument('--weight_decay', default=0.0, type=float, metavar='W', help='weight decay (default: 0.0)', dest='weight_decay')
    parser.add_argument('--lr_warmup_steps', default=100, type=int, metavar='N', help='Number of steps to warm up lr.')
    parser.add_argument('--lr_schedule_step_size', default=10, type=int, metavar='N', help='Number of steps before decaying lr.')
    parser.add_argument('--lr_schedule_gamma', default=0.1, type=float, metavar='N', help='Decay parameter for learning rate scheduler.')
    parser.add_argument('--grad_accumulation_steps', default=1, type=int, metavar='N', help='number of gradient accumulation steps')
    parser.add_argument('--grad_clip', default=1.0, type=float, help='gradient clipping amount')

    parser.add_argument('--loss', type=str, default='nce')
    parser.add_argument('--nce_temperature', type=float, default=0.05)
    parser.add_argument('--scale_lm_loss', type=float, default=1.0, help="Scale on language modeling loss.")
    parser.add_argument('--scale_ret_loss', type=float, default=1.0, help="Scale on retrieval loss.")
    parser.add_argument('--train_class', action='store_true', default=False)

    # * Distributed Data Parallel settings
    parser.add_argument('--main_node', default=False, type=bool, help='denote which node is used for writing.')
    parser.add_argument('--world_size', default=-1, type=int, help='number of nodes for distributed training')
    parser.add_argument('--local-rank', default=-1, type=int, help='node rank for distributed training')
    parser.add_argument('--dist_url', default='tcp://localhost:44122', type=str, help='url used to set up distributed training')
    parser.add_argument('--dist_backend', default='nccl', type=str, help='distributed backend')
    parser.add_argument('--seed', default=None, type=int, help='seed for initializing training. ')
    parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
    parser.add_argument('--multiprocessing_distributed', action='store_true',
        help='Use multi-processing distributed training to launch '
            'N processes per node, which has N GPUs. This is the '
            'fastest way to use PyTorch for either single node or '
            'multi node data parallel training')
    return parser.parse_args(args)
