from enum import Enum
import subprocess
import sys
import pdb
import shutil
import logging
import torch
import torch.distributed as dist
from torchvision.transforms import functional as F
from torchvision import transforms as T
from transformers import AutoFeatureExtractor
from PIL import Image, ImageDraw, ImageFont, ImageOps
import requests
from io import BytesIO
import numpy as np
import json
from tqdm import tqdm
import math
import Levenshtein as lev

import random
# import decord
# from pycocoevalcap.eval import COCOEvalCap
# from pycocotools.coco import COCO

def sort_list_by_time(lst, key=0):
    return sorted(lst, key=lambda x: x[key])

def fuzzy_match(text, choices):
    scores = [-lev.distance(text, choice) for choice in choices]
    return scores.index(max(scores))

def format_time(seconds):
    seconds = float(seconds)
    hours = math.floor(seconds / 3600)
    seconds %= 3600
    minutes = math.floor(seconds / 60)
    seconds %= 60
    milliseconds = math.floor((seconds % 1) * 1000)
    seconds = math.floor(seconds)
    return '{:02d}:{:02d}:{:02d}'.format(hours, minutes, seconds)

def create_logger(log_file):
    # Create a logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # Create a file handler
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.DEBUG)

    # Create a console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Create a formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

def coco_caption_eval(coco_gt_root, results_file):
    annotation_file = coco_gt_root

    coco = COCO(annotation_file)
    coco_result = coco.loadRes(results_file)

    coco_eval = COCOEvalCap(coco, coco_result)

    coco_eval.evaluate()

    score_dict = {}
    for metric, score in coco_eval.eval.items():
        print(f"{metric}: {score*100:.3f} (%)")
        score_dict[metric] = score*100

    return score_dict

def caption_eval(pred: list,  pred_path: str, gt_path: str):
  total = 0
  cap_pred = []
  for i, cap in enumerate(tqdm(pred)):
    tmp = {"image_id": i, "caption": cap}
    cap_pred.append(tmp)
  with open(pred_path, "w") as f:
    json.dump(cap_pred, f)
  return coco_caption_eval(gt_path, pred_path)

def convert_precision_dtype(precision):
    if precision == 'fp32':
        return torch.float
    elif precision == 'fp16':
        return torch.float16
    elif precision == 'bf16':
        return torch.bfloat16
    else:
        raise ValueError(f"Unsupported target data type '{precision}'.")

def dump_git_status(out_file=sys.stdout, exclude_file_patterns=['*.ipynb', '*.th', '*.sh', '*.txt', '*.json']):
  """Logs git status to stdout."""
  subprocess.call('git rev-parse HEAD', shell=True, stdout=out_file)
  subprocess.call('echo', shell=True, stdout=out_file)
  exclude_string = ''
  subprocess.call('git --no-pager diff -- . {}'.format(exclude_string), shell=True, stdout=out_file)


def get_image_from_url(url: str):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    img = img.resize((224, 224))
    img = img.convert('RGB')
    return img

def truncate_caption(caption: str) -> str:
  """Truncate captions at periods and newlines."""
  caption = caption.strip('\n')
  trunc_index = caption.find('\n') + 1
  if trunc_index <= 0:
      trunc_index = caption.find('.') + 1
  if trunc_index > 0:
    caption = caption[:trunc_index]
  return caption


def pad_to_size(x, size=256):
  delta_w = size - x.size[0]
  delta_h = size - x.size[1]
  padding = (
    delta_w // 2,
    delta_h // 2,
    delta_w - (delta_w // 2),
    delta_h - (delta_h // 2),
  )
  new_im = ImageOps.expand(x, padding)
  return new_im


class RandCropResize(object):

  """
  Randomly crops, then randomly resizes, then randomly crops again, an image. Mirroring the augmentations from https://arxiv.org/abs/2102.12092
  """

  def __init__(self, target_size):
    self.target_size = target_size

  def __call__(self, img):
    img = pad_to_size(img, self.target_size)
    d_min = min(img.size)
    img = T.RandomCrop(size=d_min)(img)
    t_min = min(d_min, round(9 / 8 * self.target_size))
    t_max = min(d_min, round(12 / 8 * self.target_size))
    t = random.randint(t_min, t_max + 1)
    img = T.Resize(t)(img)
    if min(img.size) < 256:
      img = T.Resize(256)(img)
    return T.RandomCrop(size=self.target_size)(img)


class SquarePad(object):
  """Pads image to square.
  From https://discuss.pytorch.org/t/how-to-resize-and-pad-in-a-torchvision-transforms-compose/71850/9
  """
  def __call__(self, image):
    max_wh = max(image.size)
    p_left, p_top = [(max_wh - s) // 2 for s in image.size]
    p_right, p_bottom = [max_wh - (s+pad) for s, pad in zip(image.size, [p_left, p_top])]
    padding = (p_left, p_top, p_right, p_bottom)
    return F.pad(image, padding, 0, 'constant')


def create_image_of_text(text: str, width: int = 224, nrows: int = 2, color=(255, 255, 255), font=None) -> torch.Tensor:
  """Creates a (3, nrows * 14, width) image of text.

  Returns:
    cap_img: (3, 14 * nrows, width) image of wrapped text.
  """
  height = 12
  padding = 5
  effective_width = width - 2 * padding
  # Create a black image to draw text on.
  cap_img = Image.new('RGB', (effective_width * nrows, height), color = (0, 0, 0))
  draw = ImageDraw.Draw(cap_img)
  draw.text((0, 0), text, color, font=font or ImageFont.load_default())
  cap_img = F.convert_image_dtype(F.pil_to_tensor(cap_img), torch.float32)  # (3, height, W * nrows)
  cap_img = torch.split(cap_img, effective_width, dim=-1)  # List of nrow elements of shape (3, height, W)
  cap_img = torch.cat(cap_img, dim=1)  # (3, height * nrows, W)
  # Add zero padding.
  cap_img = torch.nn.functional.pad(cap_img, [padding, padding, 0, padding])
  return cap_img


def get_feature_extractor_for_model(model_name: str, image_size: int = 224, train: bool = True):
  print(f'Using HuggingFace AutoFeatureExtractor for {model_name}.')
  feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
  return feature_extractor


def get_pixel_values_for_model(feature_extractor, img):
  pixel_values = feature_extractor(
    img.convert('RGB'),
    return_tensors="pt").pixel_values[0, ...]  # (3, H, W)
  return pixel_values

def get_pixel_values_for_video(feature_extractor, img):
  pixel_values = feature_extractor(
    img,
    return_tensors="pt").pixel_values  # (3, H, W)
  return pixel_values

def save_checkpoint(state, is_best, filename='checkpoint'):
  if is_best:
    # shutil.copyfile(filename + '.pth.tar', filename + '_best.pth.tar')
    torch.save(state, filename + '_best.pth.tar')
  else:
    torch.save(state, filename + '.pth.tar')

    


def accuracy(output, target, padding, topk=(1,)):
  """Computes the accuracy over the k top predictions for the specified values of k"""
  with torch.no_grad():
    maxk = max(topk)
    if output.shape[-1] < maxk:
      print(f"[WARNING] Less than {maxk} predictions available. Using {output.shape[-1]} for topk.")

    maxk = min(maxk, output.shape[-1])
    batch_size = target.size(0)

    # Take topk along the last dimension.
    _, pred = output.topk(maxk, -1, True, True)  # (N, T, topk)

    mask = (target != padding).type(target.dtype)
    target_expand = target[..., None].expand_as(pred)
    correct = pred.eq(target_expand)
    correct = correct * mask[..., None].expand_as(correct)

    res = []
    for k in topk:
      correct_k = correct[..., :k].reshape(-1).float().sum(0, keepdim=True)
      res.append(correct_k.mul_(100.0 / mask.sum()))
    return res


def get_params_count(model, max_name_len: int = 60):
  params = [(name[:max_name_len], p.numel(), str(tuple(p.shape)), p.requires_grad) for name, p in model.named_parameters()]
  total_trainable_params = sum([x[1] for x in params if x[-1]])
  total_nontrainable_params = sum([x[1] for x in params if not x[-1]])
  return params, total_trainable_params, total_nontrainable_params


def get_params_count_str(model, max_name_len: int = 60):
  padding = 70  # Hardcoded depending on desired amount of padding and separators.
  params, total_trainable_params, total_nontrainable_params = get_params_count(model, max_name_len)
  param_counts_text = ''
  param_counts_text += '=' * (max_name_len + padding) + '\n'
  param_counts_text += f'| {"Module":<{max_name_len}} | {"Trainable":<10} | {"Shape":>15} | {"Param Count":>12} |\n'
  param_counts_text += '-' * (max_name_len + padding) + '\n'
  for name, param_count, shape, trainable in params:
    param_counts_text += f'| {name:<{max_name_len}} | {"True" if trainable else "False":<10} | {shape:>15} | {param_count:>12,} |\n'
  param_counts_text += '-' * (max_name_len + padding) + '\n'
  param_counts_text += f'| {"Total trainable params":<{max_name_len}} | {"":<10} | {"":<15} | {total_trainable_params:>12,} |\n'
  param_counts_text += f'| {"Total non-trainable params":<{max_name_len}} | {"":<10} | {"":<15} | {total_nontrainable_params:>12,} |\n'
  param_counts_text += '=' * (max_name_len + padding) + '\n'
  return param_counts_text


class Summary(Enum):
  NONE = 0
  AVERAGE = 1
  SUM = 2
  COUNT = 3


class ProgressMeter(object):
  def __init__(self, num_batches, meters, prefix="", logger=None):
    self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
    self.meters = meters
    self.prefix = prefix
    self.logger = logger

  def display(self, batch):
    entries = [self.prefix + self.batch_fmtstr.format(batch)]
    entries += [str(meter) for meter in self.meters]
    self.logger.info('\t'.join(entries)) if self.logger else print('\t'.join(entries))
    
  def display_summary(self):
    entries = [" *"]
    entries += [meter.summary() for meter in self.meters]
    # print(' '.join(entries))
    self.logger.info(' '.join(entries)) if self.logger else print(' '.join(entries))

  def _get_batch_fmtstr(self, num_batches):
    num_digits = len(str(num_batches // 1))
    fmt = '{:' + str(num_digits) + 'd}'
    return '[' + fmt + '/' + fmt.format(num_batches) + ']'


class AverageMeter(object):
  """Computes and stores the average and current value"""
  def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
    self.name = name
    self.fmt = fmt
    self.summary_type = summary_type
    self.reset()

  def reset(self):
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0

  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / max(self.count, 1)

  def all_reduce(self):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    total = torch.tensor([self.sum, self.count], dtype=torch.float32, device=device)
    dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
    self.sum, self.count = total.tolist()
    self.avg = self.sum / max(self.count, 1)

  def __str__(self):
    if self.summary_type is not Summary.NONE:
      fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
    else:
      fmtstr = '{name} {val' + self.fmt + '}'
    return fmtstr.format(**self.__dict__)
  
  def summary(self):
    fmtstr = ''
    if self.summary_type is Summary.NONE:
      fmtstr = ''
    elif self.summary_type is Summary.AVERAGE:
      fmtstr = '{name} {avg:.3f}'
    elif self.summary_type is Summary.SUM:
      fmtstr = '{name} {sum:.3f}'
    elif self.summary_type is Summary.COUNT:
      fmtstr = '{name} {count:.3f}'
    else:
      raise ValueError('invalid summary type %r' % self.summary_type)
    
    return fmtstr.format(**self.__dict__)


def sample_frames(num_frames, vlen, sample='rand'):
    acc_samples = min(num_frames, vlen)
    intervals = np.linspace(start=0, stop=vlen, num=acc_samples + 1).astype(int)
    ranges = []
    for idx, interv in enumerate(intervals[:-1]):
        ranges.append((interv, intervals[idx + 1] - 1))
    if sample == 'rand':
        frame_idxs = []
        for x in ranges:
          if x[0] != x[1]:
            frame_idxs.append(random.choice(range(x[0], x[1])))
          else:
            frame_idxs.append((x[0] + x[1]) // 2)
    elif sample == 'uniform':
        frame_idxs = [(x[0] + x[1]) // 2 for x in ranges]
    else:
        raise NotImplementedError

    return frame_idxs

def sample_frames_start_end(num_frames, start, end, sample='rand'):
    acc_samples = min(num_frames, end-start)
    intervals = np.linspace(start=start, stop=end, num=acc_samples + 1).astype(int)
    ranges = []
    for idx, interv in enumerate(intervals[:-1]):
        ranges.append((interv, intervals[idx + 1] - 1))
    if sample == 'rand':
      frame_idxs = []
      for x in ranges:
        if x[0] != x[1]:
          frame_idxs.append(random.choice(range(x[0], x[1])))
        else:
          frame_idxs.append((x[0] + x[1]) // 2)
    elif sample == 'uniform':
        frame_idxs = [(x[0] + x[1]) // 2 for x in ranges]
    else:
        raise NotImplementedError

    return frame_idxs

def read_frames_decord(video_path, num_frames, sample='rand'):
    # video_reader = decord.VideoReader(video_path, num_threads=1)    
    video_reader = decord.VideoReader(video_path)
    vlen = len(video_reader)
    frame_idxs = sample_frames(num_frames, vlen, sample=sample)
    video_reader.skip_frames(1)
    frames = video_reader.get_batch(frame_idxs).asnumpy()
    frames = torch.from_numpy(frames)
    frames = [x.permute(2, 0, 1) for x in frames]
    return frames, frame_idxs

def read_frames_features(video_feat, num_frames, t1, t2, fps=5, sample='rand'):
    start, end = int(t1 * fps), int(t2 * fps)
    frame_idxs = sample_frames_start_end(num_frames, start, end, sample=sample)

    frames = video_feat[frame_idxs]
    frames = torch.from_numpy(frames)
    return frames, frame_idxs

def params_to_save(model):
    params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            params.append(name)
    return params

def pad_sequences_1d(sequences, dtype=torch.long, device=torch.device("cpu"), fixed_length=None, padded_values=0, padded_sides='right'):
    """ Pad a single-nested list or a sequence of n-d array (torch.tensor or np.ndarray)
    into a (n+1)-d array, only allow the first dim has variable lengths.
    Args:
        sequences: list(n-d tensor or list)
        dtype: np.dtype or torch.dtype
        device:
        fixed_length: pad all seq in sequences to fixed length. All seq should have a length <= fixed_length.
            return will be of shape [len(sequences), fixed_length, ...]
    Returns:
        padded_seqs: ((n+1)-d tensor) padded with zeros
        mask: (2d tensor) of the same shape as the first two dims of padded_seqs,
              1 indicate valid, 0 otherwise
    Examples:
        >>> test_data_list = [[1,2,3], [1,2], [3,4,7,9]]
        >>> pad_sequences_1d(test_data_list, dtype=torch.long)
        >>> test_data_3d = [torch.randn(2,3,4), torch.randn(4,3,4), torch.randn(1,3,4)]
        >>> pad_sequences_1d(test_data_3d, dtype=torch.float)
        >>> test_data_list = [[1,2,3], [1,2], [3,4,7,9]]
        >>> pad_sequences_1d(test_data_list, dtype=np.float32)
        >>> test_data_3d = [np.random.randn(2,3,4), np.random.randn(4,3,4), np.random.randn(1,3,4)]
        >>> pad_sequences_1d(test_data_3d, dtype=np.float32)
    """
    
    if isinstance(sequences[0], list):
        if "torch" in str(dtype):
            sequences = [torch.tensor(s, dtype=dtype, device=device) for s in sequences]
        else:
            sequences = [np.asarray(s, dtype=dtype) for s in sequences]

    extra_dims = sequences[0].shape[1:]  # the extra dims should be the same for all elements
    lengths = [len(seq) for seq in sequences]
    if fixed_length is not None:
        max_length = fixed_length
    else:
        max_length = max(lengths)
    if isinstance(sequences[0], torch.Tensor):
        assert "torch" in str(dtype), "dtype and input type does not match"
        # padded_seqs = torch.zeros((len(sequences), max_length) + extra_dims, dtype=dtype, device=device)
        padded_seqs = torch.full(((len(sequences), max_length) + extra_dims), padded_values, dtype=dtype, device=device)
        mask = torch.zeros((len(sequences), max_length), dtype=torch.float32, device=device)
    else:  # np
        assert "numpy" in str(dtype), "dtype and input type does not match"
        # padded_seqs = np.zeros((len(sequences), max_length) + extra_dims, dtype=dtype)
        padded_seqs = np.full(((len(sequences), max_length) + extra_dims), padded_values, dtype=dtype)
        mask = np.zeros((len(sequences), max_length), dtype=np.float32)

    for idx, seq in enumerate(sequences):
        end = lengths[idx]
        if padded_sides == 'right':
            padded_seqs[idx, :end] = seq
            mask[idx, :end] = 1
        elif padded_sides == 'left':
            padded_seqs[idx, -end:] = seq
            mask[idx, -end:] = 1
        else:
            raise NotImplementedError
    return padded_seqs, mask  # , lengths