"""Synthetic correctness and timing diagnostics; never classification results."""
import argparse
import copy
import statistics
import time

import torch
from common import ROOT, environment, seed_everything, write_json
from model import SmallRepVGG, deploy_copy


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--threads", type=int, nargs="+", default=[2, 4, 8])
    p.add_argument("--steps", type=int, default=12)
    args = p.parse_args()
    seed_everything(42, args.threads[0])
    checks = []
    for dtype in (torch.float32, torch.float64):
        for variant in ("baseline", "conv1", "full"):
            model = SmallRepVGG(branches=variant).to(dtype)
            # Update BN statistics and affine parameters before testing fusion.
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            for _ in range(3):
                x = torch.randn(8, 3, 32, 32, dtype=dtype)
                optimizer.zero_grad()
                model(x).square().mean().backward()
                optimizer.step()
            model.eval()
            fused = deploy_copy(model)
            max_error, relative_l2 = 0.0, 0.0
            with torch.inference_mode():
                for batch in (1, 17):
                    x = torch.randn(batch, 3, 32, 32, dtype=dtype)
                    a, b = model(x), fused(x)
                    torch.testing.assert_close(a, b, rtol=1e-4 if dtype == torch.float32 else 1e-10,
                                               atol=1e-5 if dtype == torch.float32 else 1e-12)
                    max_error = max(max_error, (a - b).abs().max().item())
                    relative_l2 = max(relative_l2, ((a-b).norm()/a.norm().clamp_min(1e-30)).item())
            checks.append({"variant": variant, "dtype": str(dtype), "max_abs_error": max_error,
                           "relative_l2": relative_l2})
    profiles = []
    for threads in args.threads:
        torch.set_num_threads(threads)
        for variant in ("baseline", "conv1", "full"):
            model = SmallRepVGG(branches=variant).train()
            optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
            x, y = torch.randn(128, 3, 32, 32), torch.randint(0, 10, (128,))
            times = []
            for step in range(args.steps + 3):
                t = time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                torch.nn.functional.cross_entropy(model(x), y).backward()
                optimizer.step()
                elapsed = time.perf_counter() - t
                if step >= 3:
                    times.append(elapsed)
            record = {"threads": threads, "variant": variant, "batch_size": 128,
                      "median_training_step_seconds": statistics.median(times),
                      "train_compute_minutes_per_45000_images": statistics.median(times)*352/60,
                      "note": "Synthetic compute only; excludes augmentation, validation, I/O and thermal drift"}
            profiles.append(record)
            print(record, flush=True)
    report = {"environment": environment(), "fusion_checks": checks, "profiles": profiles,
              "classification_results": False}
    write_json(ROOT / "results" / "preflight.json", report)
    print("Fusion assertions passed. Saved results/preflight.json", flush=True)


if __name__ == "__main__":
    main()
