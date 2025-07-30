import torch
import torch.nn as nn
import torch.nn.functional as F


class BERT4RecEncoder(nn.Module):

    def __init__(
        self,
        embedding_dim,
        max_seq_length,
        num_items,
        num_layers=2,
        num_heads=2,
        dropout=0.2,
        mask_token_id=-1,
        device="cpu",
    ):
        super(BERT4RecEncoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.max_seq_length = max_seq_length
        self.num_items = num_items
        self.mask_token_id = mask_token_id
        self.device = device
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        self.position_embedding = nn.Embedding(max_seq_length, embedding_dim)

        self.mask_embedding = nn.Parameter(torch.randn(embedding_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_layer = nn.Linear(embedding_dim, num_items)

        self.to(device)

    def forward(self, input_embs, masked_pos, target_ids=None):
        """
        Forward pass through the BERT4Rec encoder.
        Args:
            input_embs: Tensor of shape (batch_size, seq_length, embedding_dim)
            masked_pos: Tensor of positions to mask
            target_ids: Optional tensor of target item IDs for prediction
        Returns:
            Tensor of shape (batch_size, seq_length, num_items) after encoding
        """
        batch_size, seq_length, embedding_dim = input_embs.size()

        # Add positional embeddings
        positions = torch.arange(seq_length, dtype=torch.long, device=self.device)
        positions = positions.unsqueeze(0).expand(batch_size, seq_length)
        position_embs = self.position_embedding(positions)

        x = input_embs.clone()

        # Sustituir por vector de máscara en posiciones enmascaradas
        mask_vector = self.mask_embedding.view(1, 1, embedding_dim)
        x[masked_pos.bool()] = mask_vector

        x = x + position_embs

        encoded = self.encoder(x)

        # Predicción solo en posiciones enmascaradas
        logits = self.output_layer(encoded)  # (B, T, num_items)

        if target_ids is not None:
            # Flatten
            logits = logits.view(-1, logits.size(-1))
            targets = target_ids.view(-1)
            loss = F.cross_entropy(logits, targets, ignore_index=-100)
            return loss
        else:
            return logits  # se puede usar para evaluar
        
    def get_user_embedding(self, input_embs, mask=None, pooling="mean"):
        """
        Get user embedding from the input sequence.
        Args:
            input_embs: Tensor of shape (batch_size, seq_length, embedding_dim)
            mask: Optional mask tensor for padding
            pooling: Pooling method to apply to the output
        Returns:
            Tensor of shape (batch_size, embedding_dim) after pooling
        """
        batch_size, seq_length, embedding_dim = input_embs.size()

        # Add positional embeddings
        positions = torch.arange(seq_length, dtype=torch.long, device=self.device)
        positions = positions.unsqueeze(0).expand(batch_size, seq_length)
        position_embs = self.position_embedding(positions)

        x = input_embs + position_embs

        encoded = self.encoder(x)

        if pooling == "last":
            if mask is not None:
                lengths = mask.sum(dim=1) - 1
                user_emb = encoded[torch.arange(encoded.size(0)), lengths]
            else:
                user_emb = encoded[:, -1, :]
        elif pooling == "mean":
            if mask is not None:
                masked_encoded = encoded * mask.unsqueeze(-1)
                user_emb = masked_encoded.sum(dim=1) / mask.sum(dim=1, keepdim=True)
            else:
                user_emb = encoded.mean(dim=1)
        else:
            raise ValueError("Unsupported pooling method: {}".format(pooling))

        return user_emb
