"""Verify released checkpoints on real CIFAR-10 data without local run paths.

Only pass a trusted release ZIP: PyTorch checkpoints contain Python pickle data.
This script never overwrites archived experiments or their results.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import zipfile

import torch
from torchvision.datasets import CIFAR10

from common import environment, make_loaders, seed_everything, write_json
from model import SmallRepVGG, deploy_copy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', type=Path, required=True, help='Trusted published weights ZIP')
    parser.add_argument('--data', type=Path, default=Path('data'))
    parser.add_argument('--download-data', action='store_true')
    parser.add_argument('--variants', nargs='+', choices=['baseline', 'conv1', 'full'],
                        default=['baseline', 'conv1', 'full'])
    parser.add_argument('--seeds', nargs='+', type=int, choices=[42, 43, 44], default=[42, 43, 44])
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--output', type=Path, default=Path('verification/metrics.json'))
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output already exists. Choose a new --output path to preserve previous verification.')
    seed_everything(2026, args.threads)
    if args.download_data:
        CIFAR10(str(args.data), train=False, download=True)
    loader = make_loaders(args.data, 256, 2026, test=True)
    rows = []
    with zipfile.ZipFile(args.weights) as archive:
        manifest = archive.read('SHA256SUMS.txt').decode('utf-8').splitlines()
        expected = {}
        for line in manifest:
            digest, name = line.split('  ', 1)
            if name in expected:
                raise ValueError(f'Duplicate manifest entry: {name}')
            expected[name] = digest
            if hashlib.sha256(archive.read(name)).hexdigest() != digest:
                raise ValueError(f'Archive checksum mismatch: {name}')
        print(f'Archive manifest verified: {len(expected)} files.', flush=True)
        for variant in args.variants:
            for seed in args.seeds:
                run = f'main_{variant}_s{seed}'
                best_path = f'weights/{variant}_seed{seed}_best.pt'
                deploy_path = f'weights/{variant}_seed{seed}_deploy.pt'
                metric_path = f'experiments/{run}/metrics.json'
                for name in (best_path, deploy_path, metric_path):
                    if name not in expected:
                        raise ValueError(f'Required file missing from manifest: {name}')
                recorded = json.loads(archive.read(metric_path))
                if expected[best_path] != recorded['checkpoint_sha256']:
                    raise ValueError('Training checkpoint does not match recorded metrics')
                if expected[deploy_path] != recorded['deploy_sha256']:
                    raise ValueError('Deploy checkpoint does not match recorded metrics')
                # Only load trusted release checkpoints, after checking integrity.
                best = torch.load(io.BytesIO(archive.read(best_path)), map_location='cpu', weights_only=False)
                deploy = torch.load(io.BytesIO(archive.read(deploy_path)), map_location='cpu', weights_only=False)
                assert best['config']['variant'] == variant and best['config']['seed'] == seed
                assert deploy['source_checkpoint_sha256'] == expected[best_path]
                before = SmallRepVGG(branches=variant).eval()
                before.load_state_dict(best['model'], strict=True)
                after = SmallRepVGG(branches=variant, deploy=True).eval()
                after.load_state_dict(deploy['model'], strict=True)
                reconstructed = deploy_copy(before)
                for key, value in after.state_dict().items():
                    torch.testing.assert_close(value, reconstructed.state_dict()[key], rtol=0, atol=0)
                n, correct_before, correct_after, disagreements = 0, 0, 0, 0
                maximum, diff_sq, ref_sq = 0.0, 0.0, 0.0
                with torch.inference_mode():
                    for x, y in loader:
                        a, b = before(x), after(x)
                        pa, pb = a.argmax(1), b.argmax(1)
                        correct_before += int((pa == y).sum())
                        correct_after += int((pb == y).sum())
                        disagreements += int((pa != pb).sum())
                        maximum = max(maximum, float((a-b).abs().max()))
                        diff_sq += float((a.double()-b.double()).square().sum())
                        ref_sq += float(a.double().square().sum())
                        n += len(y)
                row = {'variant': variant, 'seed': seed, 'n': n,
                       'test_before_pct': 100*correct_before/n, 'test_after_pct': 100*correct_after/n,
                       'recorded_test_before_pct': recorded['test_before']['accuracy_pct'],
                       'recorded_test_after_pct': recorded['test_after']['accuracy_pct'],
                       'max_abs_logit_error': maximum, 'relative_l2': (diff_sq/max(ref_sq, 1e-30))**0.5,
                       'label_disagreements': disagreements,
                       'recorded_accuracy_matches': (abs(100*correct_before/n-recorded['test_before']['accuracy_pct']) < 1e-9
                                                     and abs(100*correct_after/n-recorded['test_after']['accuracy_pct']) < 1e-9)}
                rows.append(row)
                write_json(args.output, {'environment': environment(), 'runs': rows,
                                        'complete': False, 'archive_manifest_verified': True})
                print(json.dumps(row), flush=True)
    result = {'environment': environment(), 'runs': rows, 'complete': True,
              'archive_manifest_verified': True,
              'all_recorded_accuracies_match': all(r['recorded_accuracy_matches'] for r in rows)}
    write_json(args.output, result)
    print(f"Done: {args.output}; recorded accuracies match: {result['all_recorded_accuracies_match']}", flush=True)


if __name__ == '__main__':
    main()
