import argparse
import csv
from dataclasses import asdict
import math
from pathlib import Path
import time

import torch
import torch.nn.functional as F

from faces import ROOT
from faces.data import DATA_FILE, load_data
from faces.model import Config, Transformer
from faces.preprocess import inputs_for


def right_loss(logits, targets):
    half = targets.shape[1] // 2
    return F.cross_entropy(logits[:, half:].reshape(-1, logits.shape[-1]), targets[:, half:].long().flatten())


@torch.inference_mode()
def evaluate_loss(model, images, device, batch_size=8):
    model.eval()
    total = 0.0
    for targets in images.split(batch_size):
        targets = targets.to(device=device, dtype=torch.long)
        logits = model(inputs_for(targets, model.config.levels))
        total += right_loss(logits, targets).item() * len(targets)
    return total / len(images)


def model_state(model, data_id, step, val_loss, seed, overfit=0):
    return {"format_version": 1, "order": "left_then_right", "config": asdict(model.config),
            "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "data_id": data_id, "step": step, "val_loss": val_loss, "seed": seed, "overfit": overfit}


def save_checkpoint(state, path):
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def train(args):
    torch.manual_seed(args.seed)
    rng = torch.Generator().manual_seed(args.seed)
    device = args.device
    config = Config(size=args.size, levels=args.levels, width=args.width,
                    heads=args.heads, layers=args.layers, dropout=args.dropout)
    if args.benchmark:
        data = {"train": torch.randint(config.levels, (64, config.size ** 2), dtype=torch.uint8),
                "id": "synthetic-benchmark", "size": config.size, "levels": config.levels}
    else:
        data = load_data(args.data)
        if (data["size"], data["levels"]) != (config.size, config.levels):
            raise ValueError("Model size and gray levels must match the prepared dataset")
    model = Transformer(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda" and dtype == torch.float16)
    start_step, best, history = 0, float("inf"), []
    if args.resume:
        state = torch.load(args.run / "last.pt", map_location="cpu", weights_only=True)
        if state["config"] != asdict(config) or state["data_id"] != data["id"] or state["overfit"] != args.overfit:
            raise ValueError("Resume settings do not match the checkpoint")
        for key in ("seed", "batch_size", "accumulate", "lr", "steps", "val_count"):
            if state["args"][key] != getattr(args, key):
                raise ValueError(f"Resume setting changed: {key}")
        model.load_state_dict(state["state_dict"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        rng.set_state(state["sampler_rng"])
        torch.set_rng_state(state["torch_rng"])
        if device == "cuda" and state["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        start_step, best, history = state["step"], state["best_val_loss"], state["history"]
    images = data["train"][:args.overfit] if args.overfit else data["train"]
    validation = images if args.overfit else data.get("val", images)[:args.val_count]
    if args.benchmark:
        validation = None
    else:
        args.run.mkdir(parents=True, exist_ok=True)
        initial = evaluate_loss(model, validation, device, args.batch_size)
        print(f"Initial validation loss: {initial:.4f}", flush=True)
    print(f"{sum(p.numel() for p in model.parameters()):,} parameters | {device} | "
          f"batch {args.batch_size} × {args.accumulate}", flush=True)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    started = time.perf_counter()
    for step in range(start_step + 1, args.steps + 1):
        model.train()
        warmup = min(200, max(1, args.steps // 10))
        progress = max(0, (step - warmup) / max(1, args.steps - warmup))
        rate = args.lr * min(1.0, step / warmup) * (0.1 + 0.9 * (1 + math.cos(math.pi * progress)) / 2)
        for group in optimizer.param_groups:
            group["lr"] = rate
        optimizer.zero_grad(set_to_none=True)
        train_loss = 0.0
        for _ in range(args.accumulate):
            indices = torch.randint(len(images), (args.batch_size,), generator=rng)
            targets = images[indices].to(device=device, dtype=torch.long)
            if not args.overfit:
                flip = torch.rand(len(targets), generator=rng).to(device) < 0.5
                mirrored = targets.view(-1, 2, config.size, config.size // 2).flip((1, 3)).flatten(1)
                targets = torch.where(flip[:, None], mirrored, targets)
            with torch.autocast(device_type=device, dtype=dtype, enabled=device == "cuda"):
                logits = model(inputs_for(targets, config.levels))
                loss = right_loss(logits, targets)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at step {step}")
            train_loss += loss.item() / args.accumulate
            scaler.scale(loss / args.accumulate).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        if step % args.eval_every == 0 or step == args.steps:
            elapsed = time.perf_counter() - started
            seconds = elapsed / (step - start_step)
            print(f"step {step} | train {train_loss:.4f} | {seconds:.3f}s/step", flush=True)
            if args.benchmark:
                continue
            val_loss = evaluate_loss(model, validation, device, args.batch_size)
            print(f"val {val_loss:.4f}", flush=True)
            history.append({"step": step, "train_loss": train_loss, "val_loss": val_loss})
            state = model_state(model, data["id"], step, val_loss, args.seed, args.overfit)
            if val_loss < best:
                best = val_loss
                save_checkpoint(state, args.run / "best.pt")
            state.update({"optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                          "best_val_loss": best, "history": history, "sampler_rng": rng.get_state(),
                          "torch_rng": torch.get_rng_state(), "args": {k: str(v) if isinstance(v, Path) else v
                                                                      for k, v in vars(args).items()},
                          "cuda_rng": torch.cuda.get_rng_state_all() if device == "cuda" else None})
            save_checkpoint(state, args.run / "last.pt")
            with (args.run / "loss.csv").open("w", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=["step", "train_loss", "val_loss"], lineterminator="\n")
                writer.writeheader()
                writer.writerows(history)
    if device == "cuda":
        torch.cuda.synchronize()
        print(f"Peak allocated GPU memory: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GiB", flush=True)
    print(f"Elapsed: {time.perf_counter() - started:.1f}s", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DATA_FILE)
    parser.add_argument("--run", type=Path, default=ROOT / "runs" / "faces")
    parser.add_argument("--steps", type=int, default=12000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--accumulate", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--size", type=int, default=32)
    parser.add_argument("--levels", type=int, default=16)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--val-count", type=int, default=512)
    parser.add_argument("--overfit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if min(args.steps, args.batch_size, args.accumulate, args.eval_every, args.val_count) < 1:
        parser.error("Steps and batch sizes must be positive")
    if args.overfit < 0 or not math.isfinite(args.lr) or args.lr <= 0:
        parser.error("Invalid training settings")
    if args.benchmark and (args.resume or args.overfit):
        parser.error("Benchmarking cannot resume a training run")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    torch.set_num_threads(4)
    train(args)


if __name__ == "__main__":
    main()
