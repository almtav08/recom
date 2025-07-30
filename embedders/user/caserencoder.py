import torch
import torch.nn as nn
import torch.nn.functional as F


class CaserEncoder(nn.Module):
    def __init__(self, embedding_dim, max_seq_length, num_vertical_filters=16, num_horizontal_filters=16, horizontal_filter_sizes=(2, 3, 4), dropout=0.2, device="cpu"):
        super(CaserEncoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.max_seq_length = max_seq_length
        self.device = device
        self.num_vertical_filters = num_vertical_filters
        self.num_horizontal_filters = num_horizontal_filters
        self.horizontal_filter_sizes = horizontal_filter_sizes
        self.dropout_value = dropout

        # Vertical convolution: 1 filter per embedding dim (kernel_size = (seq_len, 1))
        self.vertical_conv = nn.Conv2d(
            in_channels=1,
            out_channels=num_vertical_filters,
            kernel_size=(max_seq_length, 1),
        )

        # Horizontal convolutions with different filter sizes
        self.horizontal_convs = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=1,
                    out_channels=num_horizontal_filters,
                    kernel_size=(h, embedding_dim),
                )
                for h in horizontal_filter_sizes
            ]
        )

        # Fully connected layer for user representation
        self.total_out = num_vertical_filters * embedding_dim + num_horizontal_filters * len(horizontal_filter_sizes)
        self.fc = nn.Linear(self.total_out, embedding_dim)

        self.dropout = nn.Dropout(self.dropout_value)

        self.to(device)

    def forward(self, input_embs):
        """
        Forward pass through the Caser encoder.
        Args:
            input_embs: Tensor of shape (batch_size, seq_length, embedding_dim)
            mask: Optional mask tensor for padding
            pooling: Pooling method to apply to the output
        Returns:
            Tensor of shape (batch_size, embedding_dim) after pooling
        """
        batch_size, seq_len, emb_dim = input_embs.size()
        x = input_embs.unsqueeze(1)  # (B, 1, L, D)

        # Vertical convolution → output shape: (B, num_vertical_filters, 1, D)
        v_out = self.vertical_conv(x)  # shape: (B, V, 1, D)
        v_out = v_out.squeeze(2)       # shape: (B, V, D)
        v_out = F.relu(v_out)
        v_out = v_out.view(batch_size, -1)  # flatten to (B, V * D)

        # Horizontal convolutions → each outputs (B, H, 1, 1)
        h_outs = []
        for conv in self.horizontal_convs:
            h = F.relu(conv(x))              # (B, H, L', 1)
            h = F.max_pool2d(h, kernel_size=(h.size(2), 1))  # (B, H, 1, 1)
            h = h.squeeze(2).squeeze(2)  # reduce L' y 1, pero conserva (B, H)                  # (B, H)
            h_outs.append(h)

        h_out = torch.cat(h_outs, dim=1) if h_outs else None

        # Combine horizontal and vertical outputs
        z = torch.cat([v_out, h_out], dim=1)
        z = self.dropout(z)

        # Project to embedding_dim
        user_emb = self.fc(z)  # shape: (B, embedding_dim)

        return user_emb

    def untrained_copy(self):
        return CaserEncoder(
            embedding_dim=self.embedding_dim,
            max_seq_length=self.max_seq_length,
            num_vertical_filters=self.num_vertical_filters,
            num_horizontal_filters=self.num_horizontal_filters,
            horizontal_filter_sizes=self.horizontal_filter_sizes,
            dropout=self.dropout_value,
            device=self.device,
        )
