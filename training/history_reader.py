import sys
import torch
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.nn import LSTM

sys.path.append(".")
from embedders.user.sasrecencoder import SASRecEncoder


if __name__ == "__main__":
    batch_size = 4
    seq_len = 10
    embedding_dim = 64

    # Simula secuencias de embeddings de recursos educativos
    input_embs = torch.randn(batch_size, seq_len, embedding_dim)

    # Máscara opcional (1 = real, 0 = padding)
    mask = torch.ones(batch_size, seq_len).bool()

    # Instancia el modelo
    model = SASRecEncoder(embedding_dim=embedding_dim, max_seq_length=seq_len)

    # Obtiene el embedding del estudiante
    student_emb = model(input_embs, mask=mask, pooling="last")

    print("Student embedding shape:", student_emb.shape)
    # → (batch_size, embedding_dim)
