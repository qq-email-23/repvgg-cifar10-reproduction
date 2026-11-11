"""CPU training with fixed validation split and resumable checkpoints."""
import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from common import ROOT, environment, file_hash, make_loaders, seed_everything, write_json, evaluate
from model import SmallRepVGG


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--variant", choices=["baseline", "conv1", "full"], required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--tag", default="main")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--stop-after", type=int, help="Stop at this epoch while keeping the planned LR schedule")
    args = p.parse_args()
    if args.workers != 0:
        p.error("Use workers=0 for deterministic checkpoint RNG restoration on this CPU setup")
    seed_everything(args.seed, args.threads)
    run = ROOT / "runs" / f"{args.tag}_{args.variant}_s{args.seed}"
    run.mkdir(parents=True, exist_ok=True)
    if (run / "config.json").exists() and not args.resume:
        p.error(f"Run exists: {run}; use --resume or a new --tag")
    train_loader, val_loader = make_loaders(args.data, args.batch_size, args.seed, args.workers)
    config = vars(args).copy()
    config["data"] = str(args.data.resolve())
    config["architecture"] = {"widths": [16, 32, 64, 128], "depths": [1, 2, 2, 2]}
    config["split_sha256"] = file_hash(ROOT / "configs" / "split_seed2026.json")
    config["source_sha256"] = {name: file_hash(ROOT / name)
                               for name in ("model.py", "train.py", "common.py")}
    config["environment"] = environment()
    model = SmallRepVGG(branches=args.variant)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    start, best, total_seconds = 0, -1.0, 0.0
    if args.resume:
        # Only load checkpoints created by this project and under your control.
        ckpt = torch.load(run / "last.pt", map_location="cpu", weights_only=False)
        for key in ("variant", "seed", "epochs", "batch_size", "lr", "split_sha256", "source_sha256", "threads"):
            if config[key] != ckpt["config"][key]:
                raise ValueError(f"Resume configuration differs: {key}")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start, best, total_seconds = ckpt["epoch"], ckpt["best_val"], ckpt["total_seconds"]
        torch.set_rng_state(ckpt["torch_rng"])
        random.setstate(ckpt["python_rng"])
        np.random.set_state(ckpt["numpy_rng"])
        train_loader.generator.set_state(ckpt["loader_rng"])
    else:
        write_json(run / "config.json", config)
    fields = ["epoch", "lr", "train_loss", "train_accuracy_pct", "val_loss",
              "val_accuracy_pct", "epoch_seconds", "total_seconds"]
    end_epoch = min(args.epochs, args.stop_after or args.epochs)
    for epoch in range(start, end_epoch):
        tick = time.perf_counter()
        model.train()
        loss_sum, correct, n = 0.0, 0, 0
        lr = optimizer.param_groups[0]["lr"]
        for x, y in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * y.numel()
            correct += (logits.argmax(1) == y).sum().item()
            n += y.numel()
        val = evaluate(model, val_loader)
        scheduler.step()
        seconds = time.perf_counter() - tick
        total_seconds += seconds
        row = dict(zip(fields, [epoch + 1, lr, loss_sum / n, 100 * correct / n,
                               val["loss"], val["accuracy_pct"], seconds, total_seconds]))
        with (run / "history.csv").open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fields)
            if epoch == 0:
                writer.writeheader()
            writer.writerow(row)
        improved = val["accuracy_pct"] > best
        best = max(best, val["accuracy_pct"])
        checkpoint = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                      "scheduler": scheduler.state_dict(), "epoch": epoch + 1,
                      "best_val": best, "total_seconds": total_seconds, "config": config,
                      "torch_rng": torch.get_rng_state(), "python_rng": random.getstate(),
                      "numpy_rng": np.random.get_state(), "loader_rng": train_loader.generator.get_state()}
        torch.save(checkpoint, run / "last.tmp")
        (run / "last.tmp").replace(run / "last.pt")
        if improved:
            torch.save({"model": model.state_dict(), "epoch": epoch + 1,
                        "best_val": best, "config": config}, run / "best.pt")
        write_json(run / "status.json", {"epoch": epoch + 1, "planned_epochs": args.epochs,
                                        "complete": epoch + 1 == args.epochs,
                                        "best_val_accuracy_pct": best, "total_seconds": total_seconds})
        print(json.dumps(row), flush=True)
    print(f"Saved: {run}", flush=True)


if __name__ == "__main__":
    main()
