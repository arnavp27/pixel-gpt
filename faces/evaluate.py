import argparse
import json
import math
from pathlib import Path
import shutil
import time

from PIL import Image, ImageDraw
import torch

from faces import ROOT
from faces.data import DATA_FILE, DIRECTORY, load_data
from faces.preprocess import decode, mirror
from faces.sample import CHECKPOINT, generate, load_model
from faces.train import evaluate_loss, model_state, save_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "runs" / "faces" / "best.pt")
    parser.add_argument("--data", type=Path, default=DATA_FILE)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--export", action="store_true")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if not math.isfinite(args.temperature) or args.temperature < 0:
        parser.error("Temperature must be nonnegative and finite")
    if args.batch_size < 1:
        parser.error("Batch size must be positive")
    torch.set_num_threads(2)
    data = load_data(args.data)
    model, checkpoint = load_model(args.checkpoint, args.device)
    if checkpoint["data_id"] != data["id"] or checkpoint.get("overfit", 0):
        parser.error("Use a full training checkpoint from this dataset")
    if args.export and args.split != "test":
        parser.error("Use the held-out test split for the exported results")
    images = data[args.split]
    if not 1 <= args.samples <= len(images):
        parser.error("Invalid sample count")
    c = model.config
    if (c.size, c.levels) != (data["size"], data["levels"]):
        parser.error("Checkpoint preprocessing does not match the dataset")
    half = c.size * c.size // 2
    loss = evaluate_loss(model, images, args.device, args.batch_size)
    indices = torch.randperm(len(images), generator=torch.Generator().manual_seed(args.seed))[:args.samples]
    outputs, baselines = [], []
    elapsed = 0.0
    for j, index in enumerate(indices.tolist()):
        left = images[index, :half]
        if args.device == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        completion = generate(model, left, temperature=args.temperature, seed=args.seed + j)[0]
        if args.device == "cuda":
            torch.cuda.synchronize()
        elapsed += time.perf_counter() - started
        if not torch.equal(completion[:half], left):
            raise RuntimeError("Generation changed the supplied pixels")
        outputs.append(completion)
        baselines.append(mirror(left, c.size))
        if (j + 1) % 8 == 0:
            print(f"Completed {j + 1}/{args.samples}", flush=True)
    generated, copied = torch.stack(outputs), torch.stack(baselines)
    targets = images[indices]
    report = {
        "data_id": data["id"], "split": args.split, "test_count": len(images),
        "sample_count": args.samples, "size": c.size, "levels": c.levels,
        "parameters": sum(p.numel() for p in model.parameters()), "step": checkpoint["step"],
        "test_loss": loss, "val_loss": checkpoint["val_loss"],
        "model_mae": (generated[:, half:].float() - targets[:, half:].float()).abs().mean().item() / (c.levels - 1),
        "mirror_mae": (copied[:, half:].float() - targets[:, half:].float()).abs().mean().item() / (c.levels - 1),
        "seconds_per_image": elapsed / args.samples, "device": args.device,
        "temperature": args.temperature, "seed": args.seed,
        "image_ids": data["ids"][args.split][indices].tolist(),
    }
    output = ROOT / "results" / "faces" if args.export else args.checkpoint.parent / "evaluation"
    output.mkdir(parents=True, exist_ok=True)
    count = min(8, args.samples)
    tile, gap, header = 128, 10, 28
    sheet = Image.new("RGB", (4 * (tile + gap) + gap, count * (tile + gap) + gap + header), "#f8f8f2")
    draw = ImageDraw.Draw(sheet)
    for column, label in enumerate(["Visible half", "Mirror", "Model", "Original"]):
        draw.text((gap + column * (tile + gap), 8), label, fill="#243128")
    credits = {}
    metadata = json.loads((DIRECTORY / "ffhq-dataset-v2.json").read_text())
    for i in range(count):
        original = decode(targets[i], c.size, c.levels)
        visible = original.convert("RGB")
        visible.paste("#d8ded2", (c.size // 2, 0, c.size, c.size))
        pictures = [visible, decode(copied[i], c.size, c.levels), decode(generated[i], c.size, c.levels), original]
        for column, image in enumerate(pictures):
            sheet.paste(image.resize((tile, tile), Image.Resampling.NEAREST),
                        (gap + column * (tile + gap), header + gap + i * (tile + gap)))
        image_id = str(report["image_ids"][i])
        credits[image_id] = metadata[image_id]["metadata"]
        if args.export:
            (output / "examples").mkdir(exist_ok=True)
            original.save(output / "examples" / f"{int(image_id):05d}.png")
    sheet.save(output / "comparison.png")
    (output / "credits.json").write_text(json.dumps(credits, indent=2) + "\n")
    (output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    if args.export:
        (output / "examples" / "credits.json").write_text(json.dumps({f"{int(k):05d}": v for k, v in credits.items()}, indent=2) + "\n")
        CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
        save_checkpoint(model_state(model, checkpoint["data_id"], checkpoint["step"],
                                    checkpoint["val_loss"], checkpoint["seed"]), CHECKPOINT)
        history = args.checkpoint.parent / "loss.csv"
        if history.exists():
            shutil.copyfile(history, output / "loss.csv")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
