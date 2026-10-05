import torch
from torch import nn


class Bigram(nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(3, 2))

    def forward(self, x):
        return self.logits[x]
