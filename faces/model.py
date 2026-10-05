from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class Config:
    size: int = 32
    levels: int = 16
    width: int = 256
    heads: int = 4
    layers: int = 6
    dropout: float = 0.1

    def __post_init__(self):
        if self.size < 4 or self.size % 2 or not 2 <= self.levels <= 256:
            raise ValueError("Invalid image size or gray levels")
        if self.heads < 1 or self.width < 1 or self.width % self.heads or self.layers < 1:
            raise ValueError("Width must be divisible by heads, with at least one layer")
        if not 0 <= self.dropout < 1:
            raise ValueError("Dropout must be between 0 and 1")


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = config.heads
        self.dropout = config.dropout
        self.qkv = nn.Linear(config.width, 3 * config.width)
        self.proj = nn.Linear(config.width, config.width)

    def forward(self, x, cache=None):
        batch, length, width = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = [t.view(batch, length, self.heads, width // self.heads).transpose(1, 2)
                   for t in (q, k, v)]
        offset = 0
        if cache is not None:
            offset = cache[0].shape[2]
            k, v = torch.cat((cache[0], k), dim=2), torch.cat((cache[1], v), dim=2)
        mask = None
        if offset and length > 1:
            rows = torch.arange(offset, offset + length, device=x.device)[:, None]
            mask = torch.arange(k.shape[2], device=x.device)[None, :] <= rows
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask,
                    dropout_p=self.dropout if self.training else 0.0, is_causal=offset == 0)
        out = out.transpose(1, 2).contiguous().view(batch, length, width)
        return self.proj(out), (k, v)


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.width)
        self.attention = Attention(config)
        self.norm2 = nn.LayerNorm(config.width)
        self.ff = nn.Sequential(nn.Linear(config.width, 4 * config.width), nn.GELU(),
                                nn.Linear(4 * config.width, config.width), nn.Dropout(config.dropout))

    def forward(self, x, cache=None):
        out, cache = self.attention(self.norm1(x), cache)
        x = x + out
        return x + self.ff(self.norm2(x)), cache


class Transformer(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or Config()
        c = self.config
        self.tokens = nn.Embedding(c.levels + 1, c.width)
        self.positions = nn.Embedding(c.size * c.size, c.width)
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.layers))
        self.norm = nn.LayerNorm(c.width)
        self.output = nn.Linear(c.width, c.levels)
        self.apply(self.initialize)

    @staticmethod
    def initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(self, tokens, cache=None, use_cache=False):
        offset = cache[0][0].shape[2] if cache is not None else 0
        length = tokens.shape[1]
        if length == 0 or offset + length > self.config.size ** 2:
            raise ValueError("Sequence exceeds the image size")
        positions = torch.arange(offset, offset + length, device=tokens.device)
        x = self.tokens(tokens) + self.positions(positions)
        updated = []
        for i, block in enumerate(self.blocks):
            x, state = block(x, cache[i] if cache is not None else None)
            if use_cache:
                updated.append(state)
        logits = self.output(self.norm(x))
        return (logits, updated) if use_cache else logits
