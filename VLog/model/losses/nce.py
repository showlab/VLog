import torch
import pdb
import torch.nn as nn
import torch.nn.functional as F

def sim_matrix(a, b, eps=1e-8):
    """
    added eps for numerical stability
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt

class NormSoftmaxLoss(nn.Module):
    def __init__(self, temperature=0.05):
        super().__init__()
        self.temperature = temperature

    def get_sim_matrix(self, a, b):
        return sim_matrix(a, b)

    def forward(self, a, b, cls_idx=None):
        x = self.get_sim_matrix(a, b)
        # print(x.shape)
        if cls_idx:
            # Create mask to select only positive pairs
            # cls is of shape (N,), values are in the range [0, L-1]
            positive_mask = torch.zeros_like(x, dtype=torch.bool)  # N x L
            positive_mask[torch.arange(a.size(0)), cls_idx] = True

            # Compute log-softmax over similarity matrix
            i_logsm = F.log_softmax(x / self.temperature, dim=1)

            # Extract the log-softmax values for positive pairs
            loss_i = i_logsm[positive_mask].mean()
            return -loss_i

        "Assumes input x is similarity matrix of N x M \in [-1, 1], computed using the cosine similarity between normalised vectors"
        i_logsm = F.log_softmax(x/self.temperature, dim=1)
        j_logsm = F.log_softmax(x.t()/self.temperature, dim=1)

        # sum over positives
        idiag = torch.diag(i_logsm)
        loss_i = idiag.sum() / len(idiag)

        jdiag = torch.diag(j_logsm)
        loss_j = jdiag.sum() / len(jdiag)
        return - loss_i - loss_j