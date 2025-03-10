import pdb
import numpy as np

def retrieval_score(sim_matrix, topk=[1, 5, 10, 25, 50, 100, 250]):
    N = sim_matrix.shape[0]  # N x N retrieval matrix
    gt_indices = np.arange(N)  # Ground truth matches (diagonal)

    sorted_indices = np.argsort(-sim_matrix, axis=1)
    results = {}
    total_rank = 0
    
    for k in topk:
        correct_at_k = 0
        for i in range(N):
            if gt_indices[i] in sorted_indices[i, :k]:
                correct_at_k += 1
        recall_at_k = correct_at_k / N
        results[f"Recall-{k}"] = recall_at_k * 100

    for i in range(N):
        rank = np.where(sorted_indices[i] == gt_indices[i])[0][0] + 1
        total_rank += rank

    avg_rank = total_rank / N
    results["Rank-Avg"] = avg_rank
    return results