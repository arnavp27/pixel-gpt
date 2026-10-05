import torch
from torch import nn
import torch.nn.functional as F

from sprites.data import LENGTH, START


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.context = 4
        self.tokens = nn.Embedding(3, 16)
        self.positions = nn.Embedding(LENGTH, 16)
        self.net = nn.Sequential(
            nn.Linear(self.context * 16 + 16, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        batch, length = x.shape
        windows = F.pad(x, (self.context - 1, 0), value=START).unfold(1, self.context, 1)
        tokens = self.tokens(windows).flatten(2)
        positions = self.positions(torch.arange(length, device=x.device))
        positions = positions.expand(batch, -1, -1)
        return self.net(torch.cat((tokens, positions), dim=-1))
