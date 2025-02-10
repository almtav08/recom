import torch


def mean_reciprocal_rank(y_true, y_scores):
    """
    Compute the mean reciprocal rank of a list of ranks.

    Returns:
        float: The mean reciprocal rank.
    """
    ranks = torch.argsort(y_scores, descending=False)
    ideal_sum = sum(1.0 / (i + 1) for i in range(len(y_true)))
    reciprocal_ranks_base = torch.Tensor([1.0 / (r + 1) for r in range(len(y_scores))])
    correct_ranks_mask = torch.tensor([r in y_true for r in ranks])
    reciprocal_ranks = reciprocal_ranks_base * correct_ranks_mask

    return (reciprocal_ranks.sum() / ideal_sum).item()
