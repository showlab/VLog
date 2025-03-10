from typing import Optional
import torch
from main import utils

def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
  return torch.nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))


def contrastive_acc(logits: torch.Tensor, target: Optional[torch.Tensor] = None, topk=(1,)) -> torch.Tensor:
  """
  Args:
    logits: (N, N) predictions.
    target: (N, num_correct_answers) labels.
  """
  assert len(logits.shape) == 2, logits.shape
  batch_size = logits.shape[0]

  if target is None:
    target = torch.arange(len(logits), device=logits.device)
    return utils.accuracy(logits, target, -1, topk)
  else:
    assert len(target.shape) == 2, target.shape
    with torch.no_grad():
      maxk = max(topk)
      if logits.shape[-1] < maxk:
        print(f"[WARNING] Less than {maxk} predictions available. Using {logits.shape[-1]} for topk.")
      maxk = min(maxk, logits.shape[-1])

      # Take topk along the last dimension.
      _, pred = logits.topk(maxk, -1, True, True)  # (N, topk)
      assert pred.shape == (batch_size, maxk)

      target_expand = target[:, :, None].repeat(1, 1, maxk)  # (N, num_correct_answers, topk)
      pred_expand = pred[:, None, :].repeat(1, target.shape[1], 1)  # (N, num_correct_answers, topk)
      correct = pred_expand.eq(target_expand)  # (N, num_correct_answers, topk)
      correct = torch.any(correct, dim=1)  # (N, topk)

      res = []
      for k in topk:
        any_k_correct = torch.clamp(correct[:, :k].sum(1), max=1)  # (N,)
        correct_k = any_k_correct.float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
      return res

