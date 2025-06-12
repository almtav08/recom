import torch
import torch.nn as nn


class LightGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size, device=None, batch_first=False):
        super(LightGRUCell, self).__init__()
        self.hidden_size = hidden_size
        self.device = device if device is not None else torch.device("cpu")
        self.W_z = nn.Linear(input_size + hidden_size, hidden_size, device=self.device)
        self.W_h = nn.Linear(input_size + hidden_size, hidden_size, device=self.device)

    def forward(self, x, h_prev):
        combined = torch.cat((x, h_prev), dim=1)
        z = torch.sigmoid(self.W_z(combined))
        h_hat = torch.tanh(self.W_h(combined))
        h = (1 - z) * h_prev + z * h_hat
        return h


class MidGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size, device=None, batch_first=False):
        super(MidGRUCell, self).__init__()
        self.hidden_size = hidden_size
        self.device = device if device is not None else torch.device("cpu")

        # Compuertas: actualización y reinicio
        self.W_z = nn.Linear(input_size + hidden_size, hidden_size, device=self.device)
        self.W_r = nn.Linear(input_size + hidden_size, hidden_size, device=self.device)

        # Transformación del estado candidato
        self.W_h = nn.Linear(input_size + hidden_size, hidden_size, device=self.device)

    def forward(self, x, h_prev):
        combined = torch.cat((x, h_prev), dim=1)

        z = torch.sigmoid(self.W_z(combined))  # gate de actualización
        r = torch.sigmoid(self.W_r(combined))  # gate de reinicio

        # Aplicamos reinicio antes de calcular h̃ (más parecido a GRU que LightGRU)
        h_reset = r * h_prev
        combined_reset = torch.cat((x, h_reset), dim=1)
        h_tilde = torch.tanh(self.W_h(combined_reset))

        # Mezcla de estados como en la GRU
        h = (1 - z) * h_prev + z * h_tilde
        return h

class SimpleGatedRNN(nn.Module):

    def __init__(self, input_size, hidden_size, device=None, batch_first=False):
        super(SimpleGatedRNN, self).__init__()
        self.hidden_size = hidden_size
        self.device = device if device is not None else torch.device("cpu")

        # Parámetros para la puerta de actualización z_t
        self.W_xz = nn.Linear(input_size, hidden_size, device=self.device)
        self.W_hz = nn.Linear(hidden_size, hidden_size, device=self.device)

        # Parámetros para el estado candidato \tilde{h}_t
        self.W_xh = nn.Linear(input_size, hidden_size, device=self.device)
        self.W_hh = nn.Linear(hidden_size, hidden_size, device=self.device)

    def forward(self, x, h_0=None):
        batch_size, seq_len, _ = x.size()
        if h_0 is None:
            h_t = torch.zeros(batch_size, self.hidden_size, device=x.device)
        else:
            h_t = h_0

        outputs = []
        for t in range(seq_len):
            x_t = x[:, t, :]

            z_t = torch.sigmoid(self.W_xz(x_t) + self.W_hz(h_t))
            h_candidate = torch.tanh(self.W_xh(x_t) + self.W_hh(h_t))

            h_t = (1 - z_t) * h_t + z_t * h_candidate

            outputs.append(h_t.unsqueeze(1))

        outputs = torch.cat(outputs, dim=1)  # [batch_size, seq_len, hidden_size]
        return outputs, h_t
