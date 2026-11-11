# Final experimental results

Title: 基于结构重参数化的VGG式卷积神经网络图像分类方法研究

All three predeclared seeds (42, 43, 44) are included. The best checkpoint was selected by validation accuracy, not test accuracy. Test set: official CIFAR-10, 10,000 images. Accuracy statistics use sample standard deviation (ddof=1).

| Variant | Test accuracy, mean ± SD (%) | Training parameters | Deploy parameters | Mean training minutes |
|---|---:|---:|---:|---:|
| baseline | 88.16 ± 0.28 | 292954 | 292490 | 50.00 |
| conv1 | 88.21 ± 0.19 | 326186 | 292490 | 68.87 |
| full | 88.28 ± 0.49 | 326634 | 292490 | 68.36 |

## Per-run classification and equivalence

| Variant | Seed | Epoch | Test before (%) | Test after (%) | Max absolute logit error | Relative L2 | Label disagreements |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 42 | 55 | 87.87 | 87.87 | 1.14441e-05 | 3.89136e-07 | 0 |
| baseline | 43 | 60 | 88.43 | 88.43 | 1.14441e-05 | 3.7961e-07 | 0 |
| baseline | 44 | 58 | 88.18 | 88.18 | 1.14441e-05 | 3.87259e-07 | 0 |
| conv1 | 42 | 57 | 88.21 | 88.21 | 1.14441e-05 | 3.70905e-07 | 0 |
| conv1 | 43 | 58 | 88.40 | 88.40 | 1.14441e-05 | 3.69749e-07 | 0 |
| conv1 | 44 | 56 | 88.03 | 88.03 | 1.23978e-05 | 4.09901e-07 | 0 |
| full | 42 | 59 | 88.78 | 88.78 | 1.14441e-05 | 3.86827e-07 | 0 |
| full | 43 | 53 | 87.80 | 87.80 | 1.14441e-05 | 3.99668e-07 | 0 |
| full | 44 | 57 | 88.26 | 88.26 | 1.33514e-05 | 3.94909e-07 | 0 |

## CPU inference timing

CPU FP32, 4 threads, input 3×32×32; forward only, no preprocessing. Batch latency in milliseconds. Each run uses 20 warmup calls per structure, then 30 randomized paired groups with 10 forwards per group. Q1/Q3 and all raw measurements are in all_timing.csv and the per-run metrics.json files. Timing repetitions are not independent training repetitions.

| Variant | Seed | Batch | Before median ms | After median ms | Speedup |
|---|---:|---:|---:|---:|---:|
| baseline | 42 | 1 | 0.8560 | 0.6352 | 1.348× |
| baseline | 42 | 128 | 19.1873 | 16.9925 | 1.129× |
| baseline | 43 | 1 | 0.9324 | 0.7288 | 1.279× |
| baseline | 43 | 128 | 18.4183 | 16.5846 | 1.111× |
| baseline | 44 | 1 | 0.9089 | 0.6642 | 1.369× |
| baseline | 44 | 128 | 18.0937 | 16.5794 | 1.091× |
| conv1 | 42 | 1 | 1.4053 | 0.6742 | 2.084× |
| conv1 | 42 | 128 | 34.7427 | 18.1432 | 1.915× |
| conv1 | 43 | 1 | 1.3769 | 0.6391 | 2.155× |
| conv1 | 43 | 128 | 33.8516 | 17.7226 | 1.910× |
| conv1 | 44 | 1 | 1.3718 | 0.6414 | 2.139× |
| conv1 | 44 | 128 | 32.0577 | 16.8627 | 1.901× |
| full | 42 | 1 | 1.5129 | 0.6749 | 2.242× |
| full | 42 | 128 | 33.2559 | 16.8043 | 1.979× |
| full | 43 | 1 | 1.4856 | 0.6648 | 2.235× |
| full | 43 | 128 | 32.7901 | 17.1512 | 1.912× |
| full | 44 | 1 | 1.4646 | 0.6441 | 2.274× |
| full | 44 | 128 | 32.4634 | 17.4586 | 1.859× |

## Interpretation

- conv1 minus baseline mean test accuracy: +0.053 percentage points. Three seeds alone do not establish statistical significance.
- full minus baseline mean test accuracy: +0.120 percentage points. Three seeds alone do not establish statistical significance.
- Maximum absolute logit error across all nine evaluations: 1.33514e-05; total label disagreements across nine runs: 0. Each run uses the same 10,000 test examples, not nine independent datasets.
- All variants have the same deployment topology and parameter count. Report before/after fusion speedup separately from accuracy comparisons between variants.
- This is a small-image adaptation of RepVGG, not an official RepVGG-A0/ImageNet reproduction. Accuracy improvement and speedup must be described according to these measured results.
- All nine deployment checkpoints passed strict state-dictionary reload and a finite-output forward check. Prediction files match recorded test accuracies.

Source: per-run metrics.json, runs/*/status.json, and test_predictions.npz. Source/checkpoint SHA256 hashes are retained in the experiment records.
