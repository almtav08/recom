import torch
import torch.nn as nn
import torch.optim as optim
from .lightgru import LightGRUCell, MidGRUCell, SimpleGatedRNN
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class UserEmbeddingClassifier(nn.Module):

    def __init__(
        self, input_size, hidden_size, output_size, device=None, criterion=None
    ):
        super(UserEmbeddingClassifier, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.device = device if device is not None else torch.device("cpu")
        self.criterion = criterion

        # GRU layer
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True, device=device, num_layers=6)

        # Attention layer
        self.attention = nn.Linear(hidden_size, 1, device=device)

        # Output layer
        self.hidden = nn.Linear(hidden_size, output_size, device=device)

        self.output = nn.Linear(output_size, 1, device=device)

        # Initialize weights using xavier_uniform
        for name, param in self.gru.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

        nn.init.xavier_uniform_(self.attention.weight)
        nn.init.xavier_uniform_(self.output.weight)

        self.to(device)

    def _run_light_gru(self, path):
        batch_size, seq_len, _ = path.shape
        h = torch.zeros(batch_size, self.hidden_size, device=self.device)
        outputs = []

        for t in range(seq_len):
            x_t = path[:, t, :]  # (batch_size, input_size)
            h = self.gru(x_t, h)
            outputs.append(h.unsqueeze(1))  # keep time dimension

        return torch.cat(outputs, dim=1)  # (batch_size, seq_len, hidden_size)

    def forward(self, path, lengths = torch.Tensor([])):
        """
        Forward pass through the network.
        Args:
            path: Tensor of shape (batch_size, seq_length, input_size) containing embeddings
        Returns:
            Tensor of shape (batch_size, output_size) containing user embeddings
        """
        if lengths.size()[0] > 0:
            path = pack_padded_sequence(
                path, lengths, batch_first=True, enforce_sorted=False
            )
        # Pass through GRU
        gru_out, _ = self.gru(path)  # shape: (batch_size, seq_length, hidden_size)
        # gru_out = self._run_light_gru(path)

        if isinstance(gru_out, torch.nn.utils.rnn.PackedSequence):
            gru_out, _ = pad_packed_sequence(gru_out, batch_first=True)

        # Calculate attention weights
        attention_weights = torch.softmax(
            self.attention(gru_out), dim=1
        )  # shape: (batch_size, seq_length, 1)

        # Apply attention weights
        context = torch.sum(
            attention_weights * gru_out, dim=1
        )  # shape: (batch_size, hidden_size)
        # gru_out = torch.mean(gru_out, dim=1)  # Use mean of all outputs of GRU

        # Get final output
        output = self.hidden(context)  # shape: (batch_size, output_size)

        logits = self.output(output)  # shape: (batch_size, 1)

        return logits.squeeze(1)

    def _embed(self, path):
        """
        Forward pass through the network.
        Args:
            path: Tensor of shape (batch_size, seq_length, input_size) containing embeddings
        Returns:
            Tensor of shape (batch_size, output_size) containing user embeddings
        """
        # Pass through GRU
        gru_out, _ = self.gru(path)  # shape: (batch_size, seq_length, hidden_size)
        # gru_out = self._run_light_gru(path)

        # Calculate attention weights
        attention_weights = torch.softmax(
            self.attention(gru_out), dim=1
        )  # shape: (batch_size, seq_length, 1)

        # Apply attention weights
        context = torch.sum(
            attention_weights * gru_out, dim=1
        )  # shape: (batch_size, hidden_size)
        # gru_out = torch.mean(gru_out, dim=1)  # Use mean of all outputs of GRU

        # Get final output
        output = self.hidden(context)  # shape: (batch_size, output_size)

        return output

    def classify(self, path):
        """Devuelve probabilidad en [0,1] tras pasar por sigmoide"""
        logits = self.forward(path)
        probs = torch.sigmoid(logits)
        return probs

    def compute_loss(self, batch_input, batch_labels):
        """
        Computes the loss ```criterion``` for the positive and negative triples.
        """
        lengths = torch.tensor(
            [x.ne(0).any(dim=1).sum().item() for x in batch_input],
        )

        # Possitive scores
        logits = self.forward(batch_input, lengths)
        loss = self.criterion(logits, batch_labels.float())
        return loss

    def embed(self, path):
        return self._embed(path)

    def set_criterion(self, criterion):
        self.criterion = criterion

    def untrained_copy(self) -> "UserEmbeddingClassifier":
        return UserEmbeddingClassifier(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            output_size=self.output_size,
            device=self.device,
        )
