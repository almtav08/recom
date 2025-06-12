# Bucar el mejor valor de un fichero de history

import torch
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.nn import LSTM


if __name__ == "__main__":
    tensor_a = torch.Tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    tensor_b = torch.Tensor([[0.7, 0.8, 0.9], [1.0, 1.1, 1.2], [1.3, 1.4, 1.5]])

    print(f"Tensor A Size: {tensor_a.size()}")
    print(f"Tensor B Size: {tensor_b.size()}")

    batch_X = pad_sequence([tensor_a, tensor_b], batch_first=True)
    print(f"Batch X Size: {batch_X.size()}")

    lengths = torch.tensor([2, 3])
    packed_X = pack_padded_sequence(batch_X, lengths, batch_first=True, enforce_sorted=False)
    print(packed_X)
    print(f"Packed X Size: {packed_X.data.size()}")

    lstm = LSTM(
        input_size=3, hidden_size=2, batch_first=True
    )

    packed_output, (ht, ct) = lstm(packed_X)
    print(f"Packed Output: {packed_output}")

    output, input_sizes = pad_packed_sequence(packed_output, batch_first=True)
    print(output)
