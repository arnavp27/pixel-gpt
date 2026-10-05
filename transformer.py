import torch
from torch import nn

from data import LENGTH


class Attention(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(width, 3 * width)
        self.proj = nn.Linear(width, width)
        self.register_buffer("mask", torch.ones(LENGTH, LENGTH, dtype=torch.bool).tril())

    def forward(self, x):
        batch, length, width = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = [t.view(batch, length, self.heads, width // self.heads).transpose(1, 2)
                   for t in (q, k, v)]
        scores = q @ k.transpose(-2, -1) / (width // self.heads) ** 0.5
        scores = scores.masked_fill(~self.mask[:length, :length], float("-inf"))
        out = scores.softmax(dim=-1) @ v
        out = out.transpose(1, 2).contiguous().view(batch, length, width)
        return self.proj(out)


class Block(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.attention = Attention(width, heads)
        self.norm2 = nn.LayerNorm(width)
        self.ff = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width))

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        return x + self.ff(self.norm2(x))


class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        width = 64
        self.tokens = nn.Embedding(3, width)
        self.positions = nn.Embedding(LENGTH, width)
        self.blocks = nn.Sequential(Block(width, 4), Block(width, 4))
        self.norm = nn.LayerNorm(width)
        self.output = nn.Linear(width, 2)

    def forward(self, x):
        positions = torch.arange(x.shape[1], device=x.device)
        x = self.tokens(x) + self.positions(positions)
        return self.output(self.norm(self.blocks(x)))
