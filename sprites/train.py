import argparse
import csv
import math
import time

import torch
import torch.nn.functional as F

from sprites import MODELS
from sprites.data import ROOT, inputs_for, load_data

STEPS = {"bigram": 400, "mlp": 1200, "transformer": 1600}
RATES = {"bigram": 0.1, "mlp": 0.003, "transformer": 0.002}


@torch.no_grad()
def evaluate_loss(model, images, device):
    model.eval()
    total = 0.0
    for targets in images.split(128):
        targets = targets.to(device)
        logits = model(inputs_for(targets))
        total += F.cross_entropy(logits.flatten(0, 1), targets.flatten(), reduction="sum").item()
    return total / images.numel()


def train(name, data, args):
    torch.manual_seed(args.seed)
    rng = torch.Generator().manual_seed(args.seed)
    device = args.device
    model = MODELS[name]().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr or RATES[name], weight_decay=0.01)
    steps = args.steps or STEPS[name]
    checkpoint = ROOT / "checkpoints" / "sprites" / f"{name}.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    (ROOT / "results" / "sprites").mkdir(parents=True, exist_ok=True)
    history, best = [], float("inf")
    started = time.perf_counter()
    val_loss = evaluate_loss(model, data["val"], device)
    initial_loss = val_loss
    print(f"{name}: {sum(p.numel() for p in model.parameters()):,} parameters, {device}", flush=True)
    print(f"step 0 | val {val_loss:.4f}", flush=True)
    history.append({"step": 0, "train_loss": "", "val_loss": val_loss})

    for step in range(1, steps + 1):
        model.train()
        indices = torch.randint(len(data["train"]), (args.batch_size,), generator=rng)
        targets = data["train"][indices].to(device)
        logits = model(inputs_for(targets))
        loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 100 == 0 or step == steps:
            val_loss = evaluate_loss(model, data["val"], device)
            history.append({"step": step, "train_loss": loss.item(), "val_loss": val_loss})
            print(f"step {step} | train {loss.item():.4f} | val {val_loss:.4f}", flush=True)
            if val_loss < best:
                best = val_loss
                torch.save({
                    "model": name,
                    "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    "step": step,
                    "val_loss": val_loss,
                    "initial_val_loss": initial_loss,
                    "data_id": data["id"],
                    "seed": args.seed,
                }, checkpoint)

    with (ROOT / "results" / "sprites" / f"{name}_loss.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["step", "train_loss", "val_loss"])
        writer.writeheader()
        writer.writerows(history)
    print(f"best val {best:.4f} | {time.perf_counter() - started:.1f}s | {checkpoint.name}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["all", *MODELS], default="all", nargs="?")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.batch_size < 1 or (args.steps is not None and args.steps < 1):
        parser.error("batch size and steps must be positive")
    if args.lr is not None and (args.lr <= 0 or not math.isfinite(args.lr)):
        parser.error("learning rate must be finite and positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is not available; use --device cpu")
    torch.set_num_threads(4)
    data = load_data()
    names = MODELS if args.model == "all" else [args.model]
    for name in names:
        train(name, data, args)


if __name__ == "__main__":
    main()
