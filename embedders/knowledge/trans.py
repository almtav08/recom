import torch
from torch.nn import Embedding
from torch.nn.init import xavier_uniform_
from embedders.embedder import Embedder


class Trans(Embedder):
    def __init__(
        self, num_entities, num_relations, embedding_dim, device=None, criterion=None,
    ):
        super(Trans, self).__init__(
            device if device is not None else torch.device("cpu")
        )
        # Initialize the attributes
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.criterion = criterion

        # Embeddings
        self.ent_embeds = Embedding(num_entities, embedding_dim).to(self.device)
        self.rel_embeds = Embedding(num_relations, embedding_dim).to(self.device)

        # Initialize embeddings
        xavier_uniform_(self.ent_embeds.weight.data)
        xavier_uniform_(self.rel_embeds.weight.data)

    def forward(self, heads, relations, tails) -> torch.Tensor:
        raise NotImplementedError

    def negative_sample_loss(self, heads, relations, tails, neg_tails):
        raise NotImplementedError

    def embed(self, entity) -> torch.Tensor:
        return self.ent_embeds(entity)

    def embed_relation(self, relation) -> torch.Tensor:
        return self.rel_embeds(relation)

    def set_criterion(self, criterion):
        self.criterion = criterion

    def untrained_copy(self) -> 'Trans':
        raise NotImplementedError
