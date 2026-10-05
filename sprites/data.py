import hashlib
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "sprites.pt"
SIZE = 8
LENGTH = SIZE * SIZE
START = 2


def make_dataset(count=6000, seed=42):
    if not 200 <= count <= 100_000:
        raise ValueError("count must be between 200 and 100000")
    rng = torch.Generator().manual_seed(seed)
    density = torch.tensor([0.08, 0.35, 0.60, 0.70, 0.65, 0.45, 0.30, 0.08])[:, None]
    images, seen = [], set()
    while len(images) < count:
        left = (torch.rand(SIZE, SIZE // 2, generator=rng) < density).long()
        image = torch.cat((left, left.flip(1)), dim=1).flatten()
        key = tuple(image.tolist())
        if key not in seen:
            images.append(image)
            seen.add(key)

    images = torch.stack(images)
    images = images[torch.randperm(count, generator=rng)]
    n = count // 10
    return {
        "train": images[:-2 * n],
        "val": images[-2 * n:-n],
        "test": images[-n:],
        "seed": seed,
        "id": hashlib.sha256(images.numpy().tobytes()).hexdigest(),
    }


def load_data():
    if not DATA_FILE.exists():
        DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
        torch.save(make_dataset(), DATA_FILE)
    return torch.load(DATA_FILE, map_location="cpu", weights_only=True)


def inputs_for(images):
    starts = torch.full((len(images), 1), START, dtype=torch.long, device=images.device)
    return torch.cat((starts, images[:, :-1]), dim=1)


if __name__ == "__main__":
    data = load_data()
    for split in ("train", "val", "test"):
        print(split, tuple(data[split].shape))
