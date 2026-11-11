"""Evaluate validation-selected checkpoints once after fixing the experiment protocol."""
import argparse
from datetime import datetime, timezone
import random
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from common import ROOT, environment, evaluate, file_hash, make_loaders, seed_everything, write_json
from model import SmallRepVGG, deploy_copy


def benchmark_pair(original, fused, batch_size, repeats, seed):
    models = {"before": original, "after": fused}
    x = torch.randn(batch_size, 3, 32, 32)
    samples = {name: [] for name in models}
    rng = random.Random(seed)
    with torch.inference_mode():
        for model in models.values():
            for _ in range(20):
                model(x)
        for _ in range(repeats):
            order = list(models)
            rng.shuffle(order)
            for name in order:
                t = time.perf_counter()
                for _ in range(10):
                    models[name](x)
                samples[name].append((time.perf_counter()-t)*1000/10)
    result = {}
    for name, values in samples.items():
        median = statistics.median(values)
        result[name] = {"median_ms_per_batch": median,
                        "q25_ms": float(np.percentile(values,25)),
                        "q75_ms": float(np.percentile(values,75)),
                        "images_per_second": batch_size*1000/median,
                        "raw_ms_per_batch": values}
    result["speedup_median_ratio"] = (result["before"]["median_ms_per_batch"] /
                                          result["after"]["median_ms_per_batch"])
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--repeats", type=int, default=30)
    args = p.parse_args()
    import json
    status = json.loads((args.checkpoint.parent / "status.json").read_text(encoding="utf-8"))
    if not status["complete"]:
        p.error("Training is incomplete. Finish the planned schedule before final test evaluation.")
    seed_everything(2026, args.threads)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    model = SmallRepVGG(branches=config["variant"]).eval()
    model.load_state_dict(ckpt["model"])
    fused = deploy_copy(model)
    loader = make_loaders(args.data, 256, 2026, test=True)
    before, after = evaluate(model, loader), evaluate(fused, loader)
    max_abs, diff_squared, ref_squared, disagreements, count = 0.0, 0.0, 0.0, 0, 0
    preds_a, preds_b, labels = [], [], []
    with torch.inference_mode():
        for x, y in loader:
            a, b = model(x), fused(x)
            max_abs = max(max_abs, (a-b).abs().max().item())
            diff_squared += (a.double()-b.double()).square().sum().item()
            ref_squared += a.double().square().sum().item()
            pa, pb = a.argmax(1), b.argmax(1)
            disagreements += (pa != pb).sum().item()
            count += len(y)
            preds_a.extend(pa.tolist()); preds_b.extend(pb.tolist()); labels.extend(y.tolist())
    output = ROOT / "results" / args.checkpoint.parent.name
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "test_predictions.npz", labels=labels,
                        prediction_before=preds_a, prediction_after=preds_b)
    torch.save({"model": fused.state_dict(), "config": config, "deploy": True,
                "source_checkpoint_sha256": file_hash(args.checkpoint)}, output / "deploy.pt")
    timing = {str(batch): benchmark_pair(model, fused, batch, args.repeats, 2026)
              for batch in (1, 128)}
    result = {"checkpoint": str(args.checkpoint.resolve()),
              "evaluation_time_utc": datetime.now(timezone.utc).isoformat(),
              "evaluation_source_sha256": file_hash(Path(__file__)),
              "checkpoint_sha256": file_hash(args.checkpoint),
              "deploy_sha256": file_hash(output / "deploy.pt"),
              "variant": config["variant"], "seed": config["seed"],
              "selected_epoch": ckpt["epoch"], "best_validation_pct": ckpt["best_val"],
              "test_before": before, "test_after": after,
              "equivalence": {"n": count, "max_abs_logit_error": max_abs,
                              "relative_l2": (diff_squared/max(ref_squared,1e-30))**0.5,
                              "argmax_disagreements": disagreements},
              "parameters_before": sum(p.numel() for p in model.parameters()),
              "parameters_after": sum(p.numel() for p in fused.parameters()),
              "timing": timing, "environment": environment(),
              "timing_scope": "CPU FP32; preallocated synthetic input; forward only; inference_mode; "
                              "20 warmup calls; randomized paired order; 10 forwards/sample; "
                              "excludes data loading and preprocessing"}
    write_json(output / "metrics.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
