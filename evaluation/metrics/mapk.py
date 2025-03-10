import torch


def average_precision_at_k(y_true, y_scores, k):
    """
    Calcula el Average Precision (AP) en el top-k.

    Args:
        y_true (list): Lista binaria de relevancia (1 si relevante, 0 si no).
        y_scores (list): Lista de scores para cada ítem.
        k (int): Número de elementos a considerar.

    Returns:
        float: AP@k.
    """

    # Obtener los índices de los k elementos con mayor puntuación
    top_k = torch.topk(torch.tensor(y_scores), k, largest=False)
    top_k_indices = top_k.indices.tolist()

    # Reordenar y_true en función de los índices de top_k
    y_true_sorted = [y_true[i] for i in top_k_indices]

    # Calcular la precisión en cada posición relevante
    precisions = [
        sum(y_true_sorted[: i + 1]) / (i + 1)  # Precisión acumulada
        for i in range(k)
        if y_true_sorted[i] == 1
    ]

    return sum(precisions) / sum(y_true) if sum(y_true) > 0 else 0.0
