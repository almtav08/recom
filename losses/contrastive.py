import torch
from torch.nn import Module
from torch.nn.functional import cosine_similarity


class ContrastiveLoss(Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        similarity = cosine_similarity(output1, output2, dim=0)
        loss_contrastive = torch.mean(
            (label) * torch.pow(similarity, 2)
            + (1 - label) * torch.pow(torch.clamp(self.margin - similarity, min=0.0), 2)
        )
        return loss_contrastive
