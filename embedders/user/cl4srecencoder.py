import torch
import torch.nn as nn
import torch.nn.functional as F


class CL4SRecEncoder(nn.Module):
    def __init__(self, embedding_dim, max_seq_length, num_layers=2, num_heads=2, dropout=0.2, device="cpu"):
        super(CL4SRecEncoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.max_seq_length = max_seq_length
        self.device = device
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        self.positional_embedding = nn.Embedding(max_seq_length, embedding_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.to(device)

    def forward(self, input_seq, mask=None, pooling="mean"):
        """
        Forward pass through the CL4SRec encoder.
        Args:
            input_seq: Tensor of shape (batch_size, seq_length, embedding_dim)
            mask: Optional mask tensor for padding
            pooling: Pooling method to apply to the output
        Returns:
            Tensor of shape (batch_size, embedding_dim) after pooling
        """

        batch_size, seq_len, _ = input_seq.size()
        position_ids = torch.arange(seq_len, dtype=torch.long, device=self.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, seq_len)

        position_emb = self.positional_embedding(position_ids)
        seq_emb = input_seq + position_emb
        out = self.layer_norm(self.dropout_layer(seq_emb))

        attn_mask = mask == 0
        attn_mask = attn_mask.unsqueeze(1).expand(batch_size, seq_len, seq_len)

        # Apply pooling
        if pooling == "last":
            # Encuentra el índice del último token válido (no padding)
            if mask is not None:
                lengths = mask.sum(dim=1) - 1
                user_emb = out[torch.arange(batch_size), lengths]
            else:
                user_emb = out[:, -1, :]
        elif pooling == "mean":
            if mask is not None:
                user_emb = (out * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(
                    dim=1, keepdim=True
                )
            else:
                user_emb = out.mean(dim=1)
        else:
            raise ValueError("Unsupported pooling method: {}".format(pooling))

        return user_emb
