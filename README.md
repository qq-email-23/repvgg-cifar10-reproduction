# 基于结构重参数化的VGG式卷积神经网络图像分类方法研究

本仓库提供论文实验的代码、配置、训练日志、测试结果与模型权重，用于核验实验结果和从头重新训练。研究基于 RepVGG 的结构重参数化思想，采用适配 CIFAR-10 的小型网络，不是官方 RepVGG-A0 或原论文 ImageNet 实验的完整复现。

## 复现入口

| 内容 | 链接 |
|---|---|
| 安装环境、验证权重与从头训练 | [REPRODUCE.md](REPRODUCE.md) |
| 权重与 ZIP 校验文件 | [v1.0 Release](https://github.com/qq-email-23/repvgg-cifar10-reproduction/releases/tag/v1.0) |
| 全部实验结果及讨论 | [RESULTS_SUMMARY.md](results/RESULTS_SUMMARY.md) |
| 逐次分类与转换结果 | [all_runs.csv](results/all_runs.csv) |
| 三种子准确率统计 | [accuracy_summary.csv](results/accuracy_summary.csv) |
| 推理延迟及四分位数 | [all_timing.csv](results/all_timing.csv) |
| 独立克隆目录的验证记录 | [reproduction_check.json](results/reproduction_check.json) |

## 实验设置

| 项目 | 设置 |
|---|---|
| 数据 | CIFAR-10 官方训练集 50,000 张，官方测试集 10,000 张 |
| 训练/验证 | 从官方训练集分层划分 45,000/5,000，每类 4,500/500；划分种子 2026 |
| 划分索引 | [configs/split_seed2026.json](configs/split_seed2026.json) |
| 网络 | 四个 stage，通道数 16/32/64/128，块数 1/2/2/2；后三个 stage 首块 stride=2 |
| 分类头 | 全局平均池化、128→10 全连接 |
| 数据增强 | padding=4 的随机裁剪、随机水平翻转 |
| 归一化 | mean=(0.4914, 0.4822, 0.4465)，std=(0.2470, 0.2435, 0.2616) |
| 优化器 | SGD，初始学习率 0.05，momentum=0.9，weight_decay=5e-4 |
| 训练 | 余弦学习率，60 epochs，batch=128，CPU FP32，4 线程 |
| 随机种子 | 42、43、44，各方法均包含全部三个种子 |
| 模型选择 | 仅按验证准确率选择；相同时保留较早轮次；测试集不参与选择 |

| 方法 | 训练结构 |
|---|---|
| baseline | 3×3 Conv-BN-ReLU |
| conv1 | 3×3 Conv-BN 与 1×1 Conv-BN 相加后 ReLU |
| full | 在 conv1 基础上增加合法位置的恒等 BN 分支，相加后 ReLU |

恒等分支仅在通道相等且 stride=1 时使用。转换在 eval 模式下融合 BN 运行统计，将 1×1 核补零、恒等映射表示为卷积后相加。三种方法转换后具有相同的部署拓扑和参数量。

## 真实实验结果

以下准确率来自官方 10,000 张测试图像，统计包含全部三个种子，标准差采用样本标准差（ddof=1）。

| 方法 | seed=42 | seed=43 | seed=44 | 均值±标准差 | 训练参数量 | 部署参数量 |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 87.87% | 88.43% | 88.18% | 88.16%±0.28% | 292,954 | 292,490 |
| conv1 | 88.21% | 88.40% | 88.03% | 88.21%±0.19% | 326,186 | 292,490 |
| full | 88.78% | 87.80% | 88.26% | 88.28%±0.49% | 326,634 | 292,490 |

full 的平均测试准确率高于 baseline **0.12 个百分点**，但不同种子存在波动，三个种子不足以支持显著提升的结论。

**转换一致性：**每次运行均在相同的 10,000 张测试图像上比较转换前后输出。九次运行标签不一致数均为 **0**，转换前后准确率相同；跨运行最大的 logits 绝对误差为 **1.33514×10⁻⁵**。各次相对 L2 误差见完整结果表。

**推理性能：**full 转换前后延迟中位数之比为 **2.235～2.274×（batch=1）**和 **1.859～1.979×（batch=128）**。原硬件为 AMD Ryzen 7 5800H，CPU FP32、4 线程、输入 3×32×32。每个结构预热 20 次，随后交错随机顺序计时 30 组，每组 10 次前向。只计算预分配输入的前向耗时，不包含数据读取与预处理。计时重复不是独立训练重复。

这些加速比比较同一模型转换前后，不代表 full 的部署结构比 baseline 的部署结构更快。不同设备的绝对延迟和加速比可能不同。

## 快速验证已训练权重

环境安装、数据下载及校验步骤见 [复现指南](REPRODUCE.md)。环境安装完成、权重 ZIP 下载到仓库根目录后，无需解压即可运行：

```powershell
# 验证完整方法 seed=42，首次下载官方 CIFAR-10 数据
.\.venv\Scripts\python.exe verify_release.py --weights .\repvgg_cifar10_v1.0_weights.zip --data .\data --download-data --variants full --seeds 42 --output .\verification\full42.json

# 验证全部九次实验
.\.venv\Scripts\python.exe verify_release.py --weights .\repvgg_cifar10_v1.0_weights.zip --data .\data --download-data --output .\verification\all_nine.json
```

程序核对包内哈希、严格加载训练与部署权重、重新融合并比较参数，随后真实推理测试集，报告准确率、logits 误差、标签不一致数及是否与原记录一致。新输出保存在 verification/，不覆盖原始实验结果。

## 环境与权重

原实验使用 Windows、Python 3.11.2、PyTorch 2.9.1+cpu、torchvision 0.24.1+cpu、NumPy 2.2.6、Pillow 12.1.0。依赖见 [requirements.txt](requirements.txt)，运行无需 NVIDIA GPU。

Release 的 repvgg_cifar10_v1.0_weights.zip 包含 **18 份权重**：三个方法×三个种子×训练/部署两种结构，并附配置、每轮日志、测试指标、逐样本预测、加载说明和文件校验和。SHA256SUMS.txt 用于校验整个 ZIP。数据集由 torchvision 从官方来源获取，未随仓库分发。

原始训练与评估代码提交为 `4053d74bcfdb784e902c592b69ce703df46dec04`。后续补充文档和验证入口，model.py、train.py、common.py 保持原实验版本。

## 文件说明

| 路径 | 用途 |
|---|---|
| model.py | 三种网络分支配置、BN 融合与部署转换 |
| train.py | 固定划分训练、验证选模、断点续训 |
| verify_release.py | 直接验证 Release 权重及原始测试结果 |
| evaluate_and_export.py | 完成训练后重新测试、转换、导出及测速 |
| check_and_profile.py | 合成输入融合自检和训练步测速，不是分类实验 |
| runs/main_*/ | 九次原始实验的配置、60 轮 CSV 日志、完成状态 |
| results/main_*/metrics.json | 原始测试结果、计时样本、环境及权重哈希 |
| results/ | 汇总表与独立克隆验证记录 |
| final_evaluation.py、summarize_results.py | 原始完整实验的评估与汇总工具 |
| run_all.ps1、run_evaluation.ps1 | 原实验本机启动脚本，外部复现使用 REPRODUCE.md 中的命令 |

日志中的绝对路径只记录当时的运行位置，外部验证不依赖这些路径。从头训练使用新标签 retrain，避免与已提交的 main_* 实验记录冲突，步骤见复现指南。

## 验证范围

在独立克隆目录中，使用 Release 对应 ZIP 对九个模型重新执行测试集推理，全部准确率与原记录一致，转换前后标签不一致数均为 0。该检查复用原电脑的环境和已有数据，尚未完成另一台设备的全新依赖安装测试。跨后端、版本和硬件重训不保证权重逐位一致，测速应记录实际环境。

## 方法来源

Ding X, Zhang X, Ma N, Han J, Ding G, Sun J. **RepVGG: Making VGG-Style ConvNets Great Again.** CVPR, 2021: 13733–13742.

- [原论文](https://openaccess.thecvf.com/content/CVPR2021/html/Ding_RepVGG_Making_VGG-Style_ConvNets_Great_Again_CVPR_2021_paper.html)
- [官方实现](https://github.com/DingXiaoH/RepVGG)
- [CIFAR-10 数据集](https://www.cs.toronto.edu/~kriz/cifar.html)
