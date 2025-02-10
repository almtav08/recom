import torch
from embedders.knowledge.trans import Trans


class TransE(Trans):
    def __init__(
        self, num_entities, num_relations, embedding_dim, device=None, criterion=None
    ):
        super(TransE, self).__init__(
            num_entities, num_relations, embedding_dim, device, criterion
        )

    def forward(self, heads, relations, tails):
        """
        Computes the score for the triples.
        """
        h = self.ent_embeds(heads)
        r = self.rel_embeds(relations)
        t = self.ent_embeds(tails)

        # Distancia de los tripletes verdaderos
        pos_dist = torch.norm(h + r - t, p=1, dim=1)
        return pos_dist

    def negative_sample_loss(self, heads, relations, tails, neg_tails):
        """
        Computes the loss ```criterion``` for the positive and negative triples.
        """
        # Possitive scores
        pos_dist = self.forward(heads, relations, tails)

        # Negative scores
        neg_dist = self.forward(heads, relations, neg_tails)

        # Compute loss
        target = torch.tensor([-1], dtype=torch.float, device=self.device)
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
        return torch.norm(h + r - t, p=1).item()

    def untrained_copy(self) -> 'TransE':
        return TransE(
            self.num_entities,
            self.num_relations,
            self.embedding_dim,
            self.device,
            self.criterion,
        )
