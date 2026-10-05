import argparse
import json

import torch

from data import ROOT, SIZE, load_data
from sample import generate, load_model, save_grid
from train import MODELS, evaluate_loss


def sample_metrics(images, training_images):
    grids = images.reshape(-1, SIZE, SIZE)
    matches = grids[:, :, :SIZE // 2] == grids[:, :, SIZE // 2:].flip(2)
    training = {tuple(image.tolist()) for image in training_images}
    samples = [tuple(image.tolist()) for image in images]
    return {
        "matching_pairs": matches.float().mean().item(),
        "symmetric_images": matches.flatten(1).all(1).float().mean().item(),
        "unique_samples": len(set(samples)) / len(samples),
        "training_matches": sum(sample in training for sample in samples) / len(samples),
        "filled_pixels": images.float().mean().item(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("samples must be positive")
    torch.set_num_threads(4)
    data = load_data()
    loaded = {}
    for name in MODELS:
        try:
            model, checkpoint = load_model(name)
        except FileNotFoundError as error:
            parser.error(str(error))
        if checkpoint["data_id"] != data["id"]:
            parser.error(f"{name} was trained on a different dataset; retrain it")
        loaded[name] = model, checkpoint

    results = ROOT / "results"
    results.mkdir(exist_ok=True)
    save_grid(data["train"][:32], results / "data.svg", "Training examples")
    report = {
        "data_id": data["id"],
        "split_sizes": {split: len(data[split]) for split in ("train", "val", "test")},
        "sample_count": args.samples,
        "sample_seed": args.seed,
        "temperature": 1.0,
        "torch_version": str(torch.__version__),
        "models": {},
    }
    print(f"{'model':<13} {'test loss':>10} {'pair match':>11} {'symmetric':>11} {'unique':>9} {'train match':>12}", flush=True)
    for name, (model, checkpoint) in loaded.items():
        images = generate(model, count=args.samples, seed=args.seed)
        metrics = sample_metrics(images, data["train"])
        metrics.update({
            "parameters": sum(p.numel() for p in model.parameters()),
            "test_loss": evaluate_loss(model, data["test"], "cpu"),
            "val_loss": checkpoint["val_loss"],
            "selected_step": checkpoint["step"],
            "training_seed": checkpoint["seed"],
        })
        report["models"][name] = metrics
        save_grid(images[:32], results / f"{name}.svg", name)
        print(f"{name:<13} {metrics['test_loss']:>10.4f} {metrics['matching_pairs']:>10.1%} "
              f"{metrics['symmetric_images']:>10.1%} {metrics['unique_samples']:>8.1%} "
              f"{metrics['training_matches']:>11.1%}", flush=True)
    (results / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
