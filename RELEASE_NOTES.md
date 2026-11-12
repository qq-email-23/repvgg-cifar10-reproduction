# v1.0：RepVGG CIFAR-10 课程论文复现实验

本 Release 提供论文「基于结构重参数化的 VGG 式卷积神经网络图像分类方法研究」的原始模型权重和配套记录。

## 下载附件

- **repvgg_cifar10_v1.0_weights.zip**：baseline、conv1、full × 种子 42/43/44 × 训练/部署两种结构，共 **18 份权重**；附配置、每轮日志、测试指标、逐样本预测、加载说明及包内校验文件。
- **SHA256SUMS.txt**：整个权重 ZIP 的 SHA-256 校验值。

GitHub 自动生成的 Source code 压缩包不包含模型权重。请下载上面两个附件；运行仓库验证程序时无需解压权重 ZIP。

## 复现入口

1. 查看 [仓库首页](https://github.com/qq-email-23/repvgg-cifar10-reproduction)。
2. 按 [REPRODUCE.md](https://github.com/qq-email-23/repvgg-cifar10-reproduction/blob/main/REPRODUCE.md) 安装 Python 3.11 环境、下载附件并校验。
3. 使用 `verify_release.py` 在 CIFAR-10 的 10,000 张真实测试图像上验证已有权重；也可按指南从头训练并重新测速。

数据集未打包，按指南通过 torchvision 从官方来源获取。无需 NVIDIA GPU。

## 原始实验结果

| 方法 | 测试准确率均值±样本标准差（3 个种子） |
|---|---:|
| baseline | 88.16%±0.28% |
| conv1 | 88.21%±0.19% |
| full | 88.28%±0.49% |

九次实验转换前后测试准确率相同，预测标签不一致数均为 0，跨运行最大绝对 logits 误差为 `1.33514e-5`。full 在原设备上的转换前后推理加速比为 2.235～2.274×（batch=1）、1.859～1.979×（batch=128）。计时条件为 AMD Ryzen 7 5800H、CPU FP32、4 线程，仅统计前向计算；其他设备的速度可能不同。

本文是 RepVGG 思想在 CIFAR-10 小型网络上的复现与分析。full 平均准确率高于 baseline 0.12 个百分点，三个种子不足以支持显著提升的结论。

原始训练与评估代码提交：`4053d74bcfdb784e902c592b69ce703df46dec04`。`main` 分支包含后续完善的复现指南和启动脚本；本次完善未替换原始权重或实验结果。
