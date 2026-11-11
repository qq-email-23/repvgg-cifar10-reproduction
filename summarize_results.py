"""Create traceable tables from all predeclared runs, without selecting seeds."""
import csv
import json
import statistics

import numpy as np
import torch

from common import ROOT, file_hash
from model import SmallRepVGG


def save_csv(name, rows):
    with (ROOT / 'results' / name).open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    torch.set_num_threads(4)
    records, timing_rows, reports = [], [], {}
    for variant in ('baseline', 'conv1', 'full'):
        reports[variant] = []
        for seed in (42, 43, 44):
            name = f'main_{variant}_s{seed}'
            path = ROOT / 'results' / name
            r = json.loads((path / 'metrics.json').read_text(encoding='utf-8'))
            status = json.loads((ROOT / 'runs' / name / 'status.json').read_text(encoding='utf-8'))
            assert r['checkpoint_sha256'] == file_hash(ROOT / 'runs' / name / 'best.pt')
            assert r['deploy_sha256'] == file_hash(path / 'deploy.pt')
            assert r['test_before']['n'] == r['test_after']['n'] == r['equivalence']['n'] == 10000
            predictions = np.load(path / 'test_predictions.npz')
            for key, metric in [('prediction_before', 'test_before'), ('prediction_after', 'test_after')]:
                accuracy = float(np.mean(predictions[key] == predictions['labels']) * 100)
                assert abs(accuracy - r[metric]['accuracy_pct']) < 1e-9
            assert int(np.sum(predictions['prediction_before'] != predictions['prediction_after'])) == r['equivalence']['argmax_disagreements']
            # Verify that exported weights reload into the public deploy architecture.
            ckpt = torch.load(path / 'deploy.pt', map_location='cpu', weights_only=False)
            deployed = SmallRepVGG(branches=variant, deploy=True).eval()
            deployed.load_state_dict(ckpt['model'], strict=True)
            with torch.inference_mode():
                output = deployed(torch.zeros(1, 3, 32, 32))
                assert output.shape == (1, 10) and torch.isfinite(output).all()
            row = {'variant': variant, 'seed': seed, 'selected_epoch': r['selected_epoch'],
                   'validation_accuracy_pct': r['best_validation_pct'],
                   'test_before_pct': r['test_before']['accuracy_pct'],
                   'test_after_pct': r['test_after']['accuracy_pct'],
                   'train_minutes': status['total_seconds'] / 60,
                   'parameters_before': r['parameters_before'], 'parameters_after': r['parameters_after'],
                   'max_abs_logit_error': r['equivalence']['max_abs_logit_error'],
                   'relative_l2': r['equivalence']['relative_l2'],
                   'label_disagreements': r['equivalence']['argmax_disagreements']}
            records.append(row)
            reports[variant].append(row)
            for batch, t in r['timing'].items():
                assert len(t['before']['raw_ms_per_batch']) == len(t['after']['raw_ms_per_batch']) == 30
                timing_rows.append({'variant': variant, 'seed': seed, 'batch_size': int(batch),
                                    'before_median_ms': t['before']['median_ms_per_batch'],
                                    'before_q25_ms': t['before']['q25_ms'],
                                    'before_q75_ms': t['before']['q75_ms'],
                                    'after_median_ms': t['after']['median_ms_per_batch'],
                                    'after_q25_ms': t['after']['q25_ms'],
                                    'after_q75_ms': t['after']['q75_ms'],
                                    'speedup': t['speedup_median_ratio']})
    save_csv('all_runs.csv', records)
    save_csv('all_timing.csv', timing_rows)
    summary = []
    for variant, values in reports.items():
        accuracies = [v['test_after_pct'] for v in values]
        summary.append({'variant': variant, 'n_training_runs': 3,
                        'mean_test_pct': statistics.mean(accuracies),
                        'sample_std_test_pp': statistics.stdev(accuracies),
                        'mean_train_minutes': statistics.mean(v['train_minutes'] for v in values),
                        'parameters_before': values[0]['parameters_before'],
                        'parameters_after': values[0]['parameters_after']})
    save_csv('accuracy_summary.csv', summary)
    lines = ['# Final experimental results', '',
             'Title: 基于结构重参数化的VGG式卷积神经网络图像分类方法研究', '',
             'All three predeclared seeds (42, 43, 44) are included. The best checkpoint was selected by validation accuracy, not test accuracy. Test set: official CIFAR-10, 10,000 images. Accuracy statistics use sample standard deviation (ddof=1).', '',
             '| Variant | Test accuracy, mean ± SD (%) | Training parameters | Deploy parameters | Mean training minutes |',
             '|---|---:|---:|---:|---:|']
    for r in summary:
        lines.append(f"| {r['variant']} | {r['mean_test_pct']:.2f} ± {r['sample_std_test_pp']:.2f} | {r['parameters_before']} | {r['parameters_after']} | {r['mean_train_minutes']:.2f} |")
    lines.extend(['', '## Per-run classification and equivalence', '',
                  '| Variant | Seed | Epoch | Test before (%) | Test after (%) | Max absolute logit error | Relative L2 | Label disagreements |',
                  '|---|---:|---:|---:|---:|---:|---:|---:|'])
    for r in records:
        lines.append(f"| {r['variant']} | {r['seed']} | {r['selected_epoch']} | {r['test_before_pct']:.2f} | {r['test_after_pct']:.2f} | {r['max_abs_logit_error']:.6g} | {r['relative_l2']:.6g} | {r['label_disagreements']} |")
    lines.extend(['', '## CPU inference timing', '',
                  'CPU FP32, 4 threads, input 3×32×32; forward only, no preprocessing. Batch latency in milliseconds. Each run uses 20 warmup calls per structure, then 30 randomized paired groups with 10 forwards per group. Q1/Q3 and all raw measurements are in all_timing.csv and the per-run metrics.json files. Timing repetitions are not independent training repetitions.', '',
                  '| Variant | Seed | Batch | Before median ms | After median ms | Speedup |',
                  '|---|---:|---:|---:|---:|---:|'])
    for r in timing_rows:
        lines.append(f"| {r['variant']} | {r['seed']} | {r['batch_size']} | {r['before_median_ms']:.4f} | {r['after_median_ms']:.4f} | {r['speedup']:.3f}× |")
    lines.extend(['', '## Interpretation', ''])
    baseline = summary[0]['mean_test_pct']
    for r in summary[1:]:
        lines.append(f"- {r['variant']} minus baseline mean test accuracy: {r['mean_test_pct'] - baseline:+.3f} percentage points. Three seeds alone do not establish statistical significance.")
    max_error = max(r['max_abs_logit_error'] for r in records)
    total_disagreements = sum(r['label_disagreements'] for r in records)
    lines.append(f'- Maximum absolute logit error across all nine evaluations: {max_error:.6g}; total label disagreements across nine runs: {total_disagreements}. Each run uses the same 10,000 test examples, not nine independent datasets.')
    lines.extend(['- All variants have the same deployment topology and parameter count. Report before/after fusion speedup separately from accuracy comparisons between variants.',
                  '- This is a small-image adaptation of RepVGG, not an official RepVGG-A0/ImageNet reproduction. Accuracy improvement and speedup must be described according to these measured results.',
                  '- All nine deployment checkpoints passed strict state-dictionary reload and a finite-output forward check. Prediction files match recorded test accuracies.', '',
                  'Source: per-run metrics.json, runs/*/status.json, and test_predictions.npz. Source/checkpoint SHA256 hashes are retained in the experiment records.'])
    (ROOT / 'results' / 'RESULTS_SUMMARY.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print('\n'.join(lines[:12]))
    print('Saved results/RESULTS_SUMMARY.md and three CSV tables. All deployment weights verified.')


if __name__ == '__main__':
    main()
