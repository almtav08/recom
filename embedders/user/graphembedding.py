import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, GATv2Conv, SGConv
from torch_geometric.nn.aggr import LSTMAggregation


class GCNLSTM(nn.Module):
    def __init__(self, in_channels, hidden_channels, lstm_hidden_dim, output_dim, class_dim, device=None):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.lstm_hidden_dim = lstm_hidden_dim
        self.output_dim = output_dim
        self.class_dim = class_dim
        self.device = device if device is not None else torch.device("cpu")
        semisuma = (in_channels + hidden_channels) // 2
        self.conv = GATConv(in_channels, semisuma)
        self.conv1 = GATConv(semisuma, hidden_channels)
        self.aggr = LSTMAggregation(hidden_channels, lstm_hidden_dim)
        self.fc = nn.Linear(lstm_hidden_dim, output_dim)
        self.clas = nn.Linear(output_dim, class_dim)

    def forward(self, data):
        x, edge_index, batch = data.x.float(), data.edge_index, data.batch

        x = F.relu(self.conv(x, edge_index))

        x = F.relu(self.conv1(x, edge_index))

        # LSTM Aggregation: resumen por grafo (usando data.batch)
        x = self.aggr(x, batch)  # (num_graphs, lstm_hidden_dim)

        x = self.fc(x)  # (num_graphs, output_dim)

        return self.clas(x)

    def embed(self, data):
        x, edge_index, batch = data.x.float(), data.edge_index, data.batch

        x = F.relu(self.conv(x, edge_index))

        x = F.relu(self.conv1(x, edge_index))

        # LSTM Aggregation: resumen por grafo (usando data.batch)
        x = self.aggr(x, batch)  # (num_graphs, lstm_hidden_dim)

        return self.fc(x)  # (num_graphs, output_dim)

    def set_criterion(self, criterion):
        self.criterion = criterion

    def classify(self, data):
        x = self.forward(data)
        x = torch.sigmoid(x)
        return x

    def untrained_copy(self) -> "GCNLSTM":
        return GCNLSTM(
            in_channels=self.in_channels,
            hidden_channels=self.hidden_channels,
            lstm_hidden_dim=self.lstm_hidden_dim,
            output_dim=self.output_dim,
            class_dim=self.class_dim,
            device=self.device,
        ).to(self.device)
