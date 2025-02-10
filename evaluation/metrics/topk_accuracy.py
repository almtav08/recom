import torch


def top_k_accuracy(y_true, y_scores, k):
    """
	Computes the top-k accuracy for a list of entities.	

	Args:
		y_true (list): The true entities.
		y_scores (list): The scores for the entities.
		k (int): The top-k value.

	Returns:
	    float: El top-k accuracy adaptado.
	"""
    top_k = torch.topk(y_scores, k, largest=False)
    top_k_indices = top_k.indices
    correct = 0
    for i in range(len(y_true)):
        if y_true[i] in top_k_indices:
            correct += 1
    return correct / len(y_true)
