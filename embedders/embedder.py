from torch.nn import Module
from torch import Tensor, device


class Embedder(Module):

    def __init__(self, dev: device) -> None:
        super(Embedder, self).__init__()
        self.device: device = dev

    def forward(self, args: Tensor) -> Tensor:
        raise NotImplementedError

    def embed(self, args: Tensor) -> Tensor:
        raise NotImplementedError

    def untrained_copy(self) -> 'Embedder':
        raise NotImplementedError
