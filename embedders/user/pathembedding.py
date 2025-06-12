import torch
import torch.nn as nn


class PathEmbeddingModel(nn.Module):
    def __init__(self, embedding_dim, max_path_length):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_path_length = max_path_length

        # Positional embedding trainable
        self.position_encoder = nn.Embedding(max_path_length, embedding_dim)

    def forward(self, resource_embeddings):
        """
        resource_embeddings: tensor de tamaño (L, D) o (B, L, D)
        """
        if resource_embeddings.dim() == 2:
            resource_embeddings = resource_embeddings.unsqueeze(0)  # (1, L, D)

        batch_size, seq_len, _ = resource_embeddings.size()
        assert (
            seq_len <= self.max_path_length
        ), "Path demasiado largo para el max_path_length"

        positions = torch.arange(seq_len, device=resource_embeddings.device)
        position_embeddings = self.position_encoder(positions)  # (L, D)
        position_embeddings = position_embeddings.unsqueeze(0).expand(
            batch_size, -1, -1
        )

        x = resource_embeddings + position_embeddings  # (B, L, D)

        # Pooling: media de los embeddings del path
        path_embedding = x.mean(dim=1)  # (B, D)
        return path_embedding
