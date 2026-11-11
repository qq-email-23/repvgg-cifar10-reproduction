"""Run final evaluations sequentially and keep full per-run console logs."""
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

from common import ROOT, file_hash


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    runs = [ROOT / 'runs' / f'main_{variant}_s{seed}'
            for seed in (42, 43, 44) for variant in ('baseline', 'conv1', 'full')]
    for run in runs:
        cfg = json.loads((run / 'config.json').read_text(encoding='utf-8'))
        status = json.loads((run / 'status.json').read_text(encoding='utf-8'))
        rows = list(csv.DictReader((run / 'history.csv').open(encoding='utf-8')))
        assert status['complete'] and status['epoch'] == cfg['epochs'] == 60, run
        assert [int(r['epoch']) for r in rows] == list(range(1, 61)), run
        assert all(file_hash(ROOT / name) == digest for name, digest in cfg['source_sha256'].items()), run
        assert file_hash(ROOT / 'configs/split_seed2026.json') == cfg['split_sha256'], run
        assert (run / 'best.pt').is_file(), run
    print('All nine runs passed audit. Protocol fixed; evaluating every seed.', flush=True)
    logs = ROOT / 'logs'
    logs.mkdir(exist_ok=True)
    for i, run in enumerate(runs, 1):
        metrics = ROOT / 'results' / run.name / 'metrics.json'
        if metrics.exists():
            saved = json.loads(metrics.read_text(encoding='utf-8'))
            assert saved['checkpoint_sha256'] == file_hash(run / 'best.pt')
            assert saved['environment']['threads'] == args.threads
            assert saved['deploy_sha256'] == file_hash(metrics.parent / 'deploy.pt')
            print(f'[{i}/9] Already evaluated: {run.name}', flush=True)
            continue
        print(f'[{i}/9] Evaluating {run.name} ...', flush=True)
        with (logs / f'eval_{run.name}.log').open('w', encoding='utf-8') as log:
            subprocess.run([sys.executable, '-u', str(ROOT / 'evaluate_and_export.py'),
                            '--checkpoint', str(run / 'best.pt'), '--data', str(args.data),
                            '--threads', str(args.threads), '--repeats', '30'],
                           cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        saved = json.loads(metrics.read_text(encoding='utf-8'))
        print(f"  Test={saved['test_after']['accuracy_pct']:.2f}% "
              f"max_error={saved['equivalence']['max_abs_logit_error']:.3g} "
              f"label_disagreements={saved['equivalence']['argmax_disagreements']}", flush=True)
    print('All final evaluations completed.', flush=True)


if __name__ == '__main__':
    main()
