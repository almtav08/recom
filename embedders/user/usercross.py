import torch
import torch.nn as nn
import torch.optim as optim


class UserEmbeddingCross(nn.Module):

    def __init__(
        self, input_size, hidden_size, output_size, device=None, criterion=None
    ):
        super(UserEmbeddingCross, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.device = device if device is not None else torch.device("cpu")
        self.criterion = criterion

        # GRU layer
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True, device=device)

        # Attention layer
        self.attention = nn.Linear(hidden_size, 1, device=device)

        # Output layer
        self.hidden = nn.Linear(hidden_size, output_size, device=device)

        self.output = nn.Linear(output_size, 2, device=device)

        # Initialize weights using xavier_uniform
        for name, param in self.gru.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

        nn.init.xavier_uniform_(self.attention.weight)
        nn.init.xavier_uniform_(self.output.weight)

        self.to(device)

    def forward(self, path):
        """
        Forward pass through the network.
        Args:
            path: Tensor of shape (batch_size, seq_length, input_size) containing embeddings
        Returns:
            Tensor of shape (batch_size, output_size) containing user embeddings
        """
        # Pass through GRU
        gru_out, _ = self.gru(path)  # shape: (batch_size, seq_length, hidden_size)

        # Calculate attention weights
        attention_weights = torch.softmax(
            self.attention(gru_out), dim=1
        )  # shape: (batch_size, seq_length, 1)

        # Apply attention weights
        context = torch.sum(
            attention_weights * gru_out, dim=1
        )  # shape: (batch_size, hidden_size)

        # Get final output
        output = self.hidden(context)  # shape: (batch_size, output_size)

        logits = self.output(output)  # shape: (batch_size, 1)

        return logits

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

        # Calculate attention weights
        attention_weights = torch.softmax(
            self.attention(gru_out), dim=1
        )  # shape: (batch_size, seq_length, 1)

        # Apply attention weights
        context = torch.sum(
            attention_weights * gru_out, dim=1
        )  # shape: (batch_size, hidden_size)

        # Get final output
        output = self.hidden(context)  # shape: (batch_size, output_size)

        return output

    def classify(self, path):
        """Devuelve probabilidad en [0,1] tras pasar por sigmoide"""
        logits = self.forward(path)
        preds = torch.argmax(logits, dim=1)  # 0 = Fail, 1 = Pass
        return preds

    def compute_loss(self, batch_input, batch_labels):
        """
        Computes the loss ```criterion``` for the positive and negative triples.
        """
        # Possitive scores
        logits = self.forward(batch_input)
        loss = self.criterion(logits, batch_labels)
        return loss

    def embed(self, path):
        return self._embed(path)

    def set_criterion(self, criterion):
        self.criterion = criterion

    def untrained_copy(self) -> "UserEmbeddingCross":
        return UserEmbeddingCross(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            output_size=self.output_size,
            device=self.device,
        )
