import argparse
import math
from pathlib import Path

from PIL import Image
import torch

from faces import ROOT
from faces.model import Config, Transformer
from faces.preprocess import decode, encode

CHECKPOINT = ROOT / "checkpoints" / "faces" / "model.pt"


def load_model(path=CHECKPOINT, device="cpu"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != 1 or checkpoint.get("order") != "left_then_right":
        raise ValueError("Unsupported face checkpoint")
    model = Transformer(Config(**checkpoint["config"])).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint


@torch.inference_mode()
def generate(model, left, count=1, temperature=0.8, seed=123):
    c = model.config
    half = c.size * c.size // 2
    left = torch.as_tensor(left)
    if left.ndim != 1 or left.numel() != half:
        raise ValueError("Pass only the left half of one image")
    if not torch.all((left >= 0) & (left < c.levels) & (left == left.long())):
        raise ValueError("Pixels must be integer gray-level tokens")
    if count < 1 or count > 16 or not math.isfinite(temperature) or temperature < 0:
        raise ValueError("Use 1 to 16 samples and a finite nonnegative temperature")
    model.eval()
    device = next(model.parameters()).device
    rng = torch.Generator(device=device).manual_seed(seed)
    left = left.to(device=device, dtype=torch.long).unsqueeze(0).expand(count, -1)
    start = torch.full((count, 1), c.levels, dtype=torch.long, device=device)
    logits, cache = model(torch.cat((start, left), dim=1), use_cache=True)
    right = []
    for i in range(half):
        scores = logits[:, -1]
        pixel = (scores.argmax(-1, keepdim=True) if temperature == 0 else
                 torch.multinomial((scores / temperature).softmax(-1), 1, generator=rng))
        right.append(pixel)
        if i + 1 < half:
            logits, cache = model(pixel, cache=cache, use_cache=True)
    return torch.cat((left, *right), dim=1).cpu()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--output", type=Path, default=ROOT / "runs" / "face.png")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()
    torch.set_num_threads(4)
    model, _ = load_model(args.checkpoint)
    c = model.config
    with Image.open(args.image) as image:
        left = encode(image, c.size, c.levels)[:c.size * c.size // 2]
    tokens = generate(model, left, temperature=args.temperature, seed=args.seed)[0]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    decode(tokens, c.size, c.levels).save(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
