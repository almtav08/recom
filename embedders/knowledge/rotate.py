import torch
from embedders.knowledge.trans import Trans


class RotatE(Trans):
    def __init__(
        self, num_entities, num_relations, embedding_dim, device=None, criterion=None
    ):
        super(RotatE, self).__init__(
            num_entities, num_relations, embedding_dim, device, criterion
        )

    def forward(self, heads, relations, tails):
        """
        Computes the score for the triples.
        """
        h = self.ent_embeds(heads)
        r = self.rel_embeds(relations)
        t = self.ent_embeds(tails)

        reh, imh = torch.chunk(h, 2, dim=-1)
        rer, imr = torch.chunk(r, 2, dim=-1)
        ret, imt = torch.chunk(t, 2, dim=-1)

        rerc = torch.cos(rer)
        rers = torch.sin(rer)

        reh, imh = reh * rerc - imh * rers, reh * rers + imh * rerc
        ret, imt = ret * rerc - imt * rers, ret * rers + imt * rerc

        dist = torch.norm(reh - ret, p=1, dim=1) + torch.norm(imh - imt, p=1, dim=1)
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

        reh, imh = torch.chunk(h, 2, dim=-1)
        rer, imr = torch.chunk(r, 2, dim=-1)
        ret, imt = torch.chunk(t, 2, dim=-1)

        rerc = torch.cos(rer)
        rers = torch.sin(rer)

        reh, imh = reh * rerc - imh * rers, reh * rers + imh * rerc
        ret, imt = ret * rerc - imt * rers, ret * rers + imt * rerc

        score = torch.norm(reh - ret, p=1, dim=1) + torch.norm(imh - imt, p=1, dim=1)
        return score.item()

    def untrained_copy(self) -> 'RotatE':
        return RotatE(
            self.num_entities,
            self.num_relations,
            self.embedding_dim,
            self.device,
            self.criterion,
        )
