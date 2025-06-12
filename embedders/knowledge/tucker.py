import torch
import torch.nn as nn

class TuckER(nn.Module):
    def __init__(self, num_entities, num_relations, emb_dim=100):
        super(TuckER, self).__init__()
        self.E = nn.Embedding(num_entities, emb_dim)
        self.R = nn.Embedding(num_relations, emb_dim)
        self.W = nn.Parameter(torch.randn(emb_dim, emb_dim, emb_dim))
        self.input_dropout = nn.Dropout(0.3)

        nn.init.xavier_normal_(self.E.weight.data)
        nn.init.xavier_normal_(self.R.weight.data)
        nn.init.xavier_normal_(self.W.data)

    def forward(self, h, r, t=None):
        e_h = self.input_dropout(self.E(h))  # (batch, emb_dim)
        r_emb = self.R(r)  # (batch, emb_dim)
        W_mat = torch.einsum(
            "bd, dxy -> bxy", r_emb, self.W
        )  # (batch, emb_dim, emb_dim)
        x = torch.bmm(e_h.unsqueeze(1), W_mat).squeeze(1)  # (batch, emb_dim)
        
        if t is not None:
            # Si se proporcionan objetivos específicos, calcular scores solo para esos
            e_t = self.E(t)  # (batch, emb_dim)
            scores = torch.sum(x * e_t, dim=1)  # producto escalar
            return scores
        else:
            # Si no se proporcionan objetivos, calcular scores para todas las entidades
            return x @ self.E.weight.T

    def negative_sample_loss(self, h, r, t, neg_t):
        """
        Computes the loss ```criterion``` for the positive and negative triples.
        
        Args:
            h: Head entities indices
            r: Relation indices
            t: Tail entities indices (positive examples)
            neg_t: Negative tail entities indices
            
        Returns:
            Loss value for the batch
        """
        # Possitive scores
        pos_dist = self.forward(h, r, t)

        # Negative scores
        neg_dist = self.forward(h, r, neg_t)

        # Compute loss
        # target = torch.tensor([-1], dtype=torch.float, device=self.device)
        target = torch.full_like(pos_dist, -1, dtype=torch.float, device=self.device)
        loss = self.criterion(pos_dist, neg_dist, target)
        return loss


class TuckERT(nn.Module):
    def __init__(self, num_entities, num_relations, num_times, emb_dim=100):
        super().__init__()
        self.emb_dim = emb_dim
        self.ent_emb = nn.Embedding(num_entities, emb_dim)
        self.rel_emb = nn.Embedding(num_relations, emb_dim)
        self.time_emb = nn.Embedding(num_times, emb_dim)

        # Núcleo tensorial 4D: (rel_dim, time_dim, ent_dim, ent_dim)
        self.W = nn.Parameter(torch.randn(emb_dim, emb_dim, emb_dim, emb_dim) * 0.01)

        nn.init.xavier_uniform_(self.ent_emb.weight.data)
        nn.init.xavier_uniform_(self.rel_emb.weight.data)
        nn.init.xavier_uniform_(self.time_emb.weight.data)

    def forward(self, h, r, t, tau):
        h_e = self.ent_emb(h)  # (batch, d)
        r_e = self.rel_emb(r)  # (batch, d)
        t_e = self.ent_emb(t)  # (batch, d)
        tau_e = self.time_emb(tau)  # (batch, d)

        # Construir W_rt para cada muestra: suma ponderada del núcleo
        W_rt = torch.einsum("bi,bj,ijkl->bkl", r_e, tau_e, self.W)  # (batch, d, d)

        # Multiplicar: h^T W_rt t
        # Primero multiplicar W_rt * t (batch matmul)
        Wt = torch.bmm(W_rt, t_e.unsqueeze(2)).squeeze(2)  # (batch, d)

        score = -torch.sum(h_e * Wt, dim=1)  # Similaridad (negativa para distancia)
        return score
