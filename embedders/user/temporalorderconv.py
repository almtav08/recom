import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops


class TemporalOrderConv(MessagePassing):
    def __init__(self, in_channels, out_channels, device=None):
        super().__init__(aggr="add")
        self.in_channels = in_channels
        self.out_channels = out_channels

        # ¡IMPORTANTE! No sumamos +1 aquí todavía
        self.lin = None
        self.edge_weight = nn.Linear(1, 1)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, x, edge_index, pos):
        x = x.to(self.device)
        edge_index = edge_index.to(self.device)
        pos = pos.to(self.device)

        if pos.dim() == 1:
            pos = pos.view(-1, 1)

        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Inicializar self.lin si aún no lo hemos hecho
        if self.lin is None:
            in_features = x.size(1) + 1  # x_j + pos_j
            # print(f"[DEBUG] Initializing Linear with in_features = {in_features}")
            self.lin = nn.Linear(in_features, self.out_channels).to(x.device)

        return self.propagate(edge_index, x=x, pos=pos)

    def message(self, x_j, pos_i, pos_j):
        x_j_augmented = torch.cat([x_j, pos_j], dim=1)

        delta_pos = pos_i - pos_j
        weight = torch.sigmoid(self.edge_weight(delta_pos))

        # print(f"[DEBUG] x_j_augmented shape: {x_j_augmented.shape}")
        # print(f"[DEBUG] weight shape: {weight.shape}")

        return self.lin(x_j_augmented) * weight

    def update(self, aggr_out):
        return aggr_out
