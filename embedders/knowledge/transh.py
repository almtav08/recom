import torch
from torch.nn import Embedding
from torch.nn.init import xavier_uniform_
from embedders.knowledge.trans import Trans


class TransH(Trans):
    def __init__(
        self, num_entities, num_relations, embedding_dim, device=None, criterion=None
    ):
        super(TransH, self).__init__(
            num_entities, num_relations, embedding_dim, device, criterion
        )
        # Embeddings
        self.normal_vectors = Embedding(num_relations, embedding_dim).to(self.device)

        # Initialize embeddings
        xavier_uniform_(self.normal_vectors.weight.data)

    def forward(self, heads, relations, tails):
        """
        Computes the score for the triples.
        """
        h = self.ent_embeds(heads)
        r = self.rel_embeds(relations)
        t = self.ent_embeds(tails)
        n = self.normal_vectors(relations)

        # Project entities
        h_proj = h - torch.sum(h * n, dim=1, keepdim=True) * n
        t_proj = t - torch.sum(t * n, dim=1, keepdim=True) * n

        # Distance of the triples
        dist = torch.norm(h_proj + r - t_proj, p=1, dim=1)
        return dist

    def negative_sample_loss(self, heads, relations, tails, neg_tails):
        """
        Computes the loss ```criterion``` for the positive and negative triples.
        """
        # Possitive scores
        pos_dist = self.forward(heads, relations, tails)

        # Negative scores
        neg_dist = self.forward(heads, relations, neg_tails)

        # Compute loss
        # target = torch.tensor([-1], dtype=torch.float, device=self.device)
        target = torch.full_like(pos_dist, -1, dtype=torch.float, device=self.device)
        loss = self.criterion(pos_dist, neg_dist, target)
        return loss

    def embed(self, entity) -> torch.Tensor:
        return self.ent_embeds(entity)

    def embed_relation(self, relation) -> torch.Tensor:
        return self.rel_embeds(relation)

    def score(self, head, relation, tail) -> torch.Tensor:
        head = torch.tensor([head], device=self.device)
        relation = torch.tensor([relation], device=self.device)
        tail = torch.tensor([tail], device=self.device)

        h = self.ent_embeds(head)
        r = self.rel_embeds(relation)
        t = self.ent_embeds(tail)
        n = self.normal_vectors(relation)

        # Project entities
        h_proj = h - torch.sum(h * n, dim=1, keepdim=True) * n
        t_proj = t - torch.sum(t * n, dim=1, keepdim=True) * n

        return torch.norm(h_proj + r - t_proj, p=1).item()

    def untrained_copy(self) -> 'TransH':
        return TransH(
            self.num_entities,
            self.num_relations,
            self.embedding_dim,
            self.device,
            self.criterion,
        )
