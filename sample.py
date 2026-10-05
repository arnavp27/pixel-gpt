import argparse
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import torch

from data import LENGTH, ROOT, SIZE, START
from train import MODELS


def load_model(name, device="cpu"):
    path = ROOT / "checkpoints" / f"{name}.pt"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path.name}. Run: python train.py {name}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = MODELS[name]().to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint


@torch.no_grad()
def generate(model, count=32, prefix="", temperature=1.0, seed=123, fixed=None):
    if count < 1 or len(prefix) > LENGTH or any(c not in "01" for c in prefix):
        raise ValueError("count must be positive; prefix must contain at most 64 binary pixels")
    if temperature < 0 or not math.isfinite(temperature):
        raise ValueError("temperature must be finite and nonnegative")
    if fixed is None:
        fixed = [int(c) for c in prefix] + [None] * (LENGTH - len(prefix))
    elif prefix:
        raise ValueError("Use either prefix or fixed pixels, not both")
    if len(fixed) != LENGTH or any(p not in (None, 0, 1) for p in fixed):
        raise ValueError("fixed must contain 64 pixels, each 0, 1, or None")
    model.eval()
    device = next(model.parameters()).device
    rng = torch.Generator(device=device).manual_seed(seed)
    x = torch.full((count, 1), START, dtype=torch.long, device=device)
    for value in fixed:
        if value is not None:
            pixel = torch.full((count, 1), value, dtype=torch.long, device=device)
        else:
            logits = model(x)[:, -1]
            if temperature == 0:
                pixel = logits.argmax(dim=-1, keepdim=True)
            else:
                pixel = torch.multinomial((logits / temperature).softmax(-1), 1, generator=rng)
        x = torch.cat((x, pixel), dim=1)
    return x[:, 1:].cpu()


def save_grid(images, path, title):
    columns = min(8, len(images))
    rows = math.ceil(len(images) / columns)
    cell, gap = 8, 12
    tile = SIZE * cell
    width = columns * (tile + gap) + gap
    height = rows * (tile + gap) + gap + 28
    svg = ET.Element("svg", xmlns="http://www.w3.org/2000/svg", width=str(width), height=str(height),
                     viewBox=f"0 0 {width} {height}")
    ET.SubElement(svg, "title").text = title
    ET.SubElement(svg, "rect", width="100%", height="100%", fill="white")
    ET.SubElement(svg, "text", x=str(gap), y="21", fill="#222",
                  attrib={"font-family": "monospace", "font-size": "14"}).text = title
    for i, image in enumerate(images):
        x = gap + (i % columns) * (tile + gap)
        y = gap + 28 + (i // columns) * (tile + gap)
        ET.SubElement(svg, "rect", x=str(x), y=str(y), width=str(tile), height=str(tile),
                      fill="white", stroke="#bbb")
        for j, pixel in enumerate(image.tolist()):
            if pixel:
                ET.SubElement(svg, "rect", x=str(x + (j % SIZE) * cell), y=str(y + (j // SIZE) * cell),
                              width=str(cell), height=str(cell), fill="#171717")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(svg).write(path, encoding="unicode")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=MODELS, default="transformer", nargs="?")
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument("--prefix", default="")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output", default="results/samples.svg")
    args = parser.parse_args()
    torch.set_num_threads(4)
    try:
        model, _ = load_model(args.model)
        images = generate(model, args.count, args.prefix, args.temperature, args.seed)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))
    path = ROOT / args.output
    save_grid(images, path, args.model)
    for image in images[:4].reshape(-1, SIZE, SIZE):
        print("\n".join("".join("#" if p else "." for p in row) for row in image.tolist()))
        print()
    print(path)


if __name__ == "__main__":
    main()
