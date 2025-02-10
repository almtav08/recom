import torch
from torch.nn import Embedding
from torch.nn.init import xavier_uniform_
from embedders.knowledge.trans import Trans


class TransR(Trans):
    def __init__(
        self,
        num_entities,
        num_relations,
        embedding_dim,
        projection_dim,
        device=None,
        criterion=None
    ):
        super(TransR, self).__init__(
            num_entities, num_relations, embedding_dim, device, criterion
        )
        # Initialize the attributes
        self.projection_dim = projection_dim

        # Embeddings
        self.proj_relations = Embedding(
            num_relations, embedding_dim * projection_dim
        ).to(self.device)

        # Initialize embeddings
        xavier_uniform_(self.proj_relations.weight.data)

    def forward(self, heads, relations, tails):
        """
        Computes the score for the triples.
        """
        h = self.ent_embeds(heads)
        r = self.rel_embeds(relations)
        t = self.ent_embeds(tails)
        rp = self.proj_relations(relations)

        rp = rp.view(-1, self.embedding_dim, self.projection_dim)

        hp = torch.matmul(h.unsqueeze(1), rp).squeeze(1)
        tp = torch.matmul(t.unsqueeze(1), rp).squeeze(1)

        # Distancia de los tripletes verdaderos
        pos_dist = torch.norm(hp + r - tp, p=1, dim=1)
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
        # target = torch.tensor([-1], dtype=torch.float, device=self.device)
        target = torch.full_like(pos_dist, -1, dtype=torch.float, device=self.device)
        loss = self.criterion(pos_dist, neg_dist, target)
        return loss

    def embed(self, entity) -> torch.Tensor:
        return self.ent_embeds(entity)

    def embed_relation(self, relation) -> torch.Tensor:
        return self.rel_embeds(relation)

    def score(self, head, relation, tail) -> torch.Tensor:
        """
        Computes the score for a single triple (head, relation, tail).
        """

        head = torch.tensor([head], device=self.device)
        relation = torch.tensor([relation], device=self.device)
        tail = torch.tensor([tail], device=self.device)

        # Compute embeddings
        h = self.ent_embeds(head)
        r = self.rel_embeds(relation)
        t = self.ent_embeds(tail)
        rp = self.proj_relations(relation) 

        # Reshape projection matrix
        rp = rp.view(
            1, self.embedding_dim, self.projection_dim
        )

        # Project head and tail embeddings
        hp = torch.matmul(h.unsqueeze(1), rp).squeeze(1)
        tp = torch.matmul(t.unsqueeze(1), rp).squeeze(1)

        # Compute and return the distance
        score = torch.norm(hp + r - tp, p=1, dim=1)
        return score.item()

    def untrained_copy(self) -> 'TransR':
        return TransR(
            self.num_entities,
            self.num_relations,
            self.embedding_dim,
            self.projection_dim,
            self.device,
            self.criterion,
        )
