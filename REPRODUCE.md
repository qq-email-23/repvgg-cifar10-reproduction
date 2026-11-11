# 从 GitHub 复现结果

本文题目：**基于结构重参数化的VGG式卷积神经网络图像分类方法研究**。

代码：https://github.com/qq-email-23/repvgg-cifar10-reproduction

权重：https://github.com/qq-email-23/repvgg-cifar10-reproduction/releases/tag/v1.0

原始训练与评估代码提交：`4053d74bcfdb784e902c592b69ce703df46dec04`。后续提交补充外部验证入口与文档，保持 `model.py`、`train.py`、`common.py` 不变。原始实验配置中的本地绝对路径仅是历史记录，验证程序不使用这些路径。

**已执行的复现检查**：在独立克隆目录中，使用待发布 ZIP 对全部九个模型重新运行真实测试集推理。九次准确率均与原记录一致，所有融合前后标签不一致数均为 0，报告见 `results/reproduction_check.json`。该检查复用了作者本机 Python 环境和已有数据；尚未验证另一台电脑的全新依赖安装、数据下载或公开 Release 下载，不把它称为跨设备复现。

## 1. 下载代码并安装环境

下面命令适用于 Windows PowerShell。建议 Python **3.11**；原实验使用 CPU PyTorch 2.9.1。首次安装依赖与下载数据需要联网。

```powershell
git clone https://github.com/qq-email-23/repvgg-cifar10-reproduction.git
Set-Location repvgg-cifar10-reproduction
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

不必激活虚拟环境，也不依赖作者电脑的 D 盘目录。若不用 Windows，使用系统 Python 3.11 创建虚拟环境，并把下文的解释器路径替换为相应路径。

## 2. 下载公开权重并校验

在上方 Release 页面下载这两个附件，放在刚克隆的仓库根目录：

- `repvgg_cifar10_v1.0_weights.zip`
- `SHA256SUMS.txt`

公开 Release 必须发布后才能下载；如果页面不存在，请联系作者完成发布。GitHub 自动提供的 Source code 压缩包不包含 `.pt` 权重。

PowerShell 校验 ZIP：

```powershell
$RepActualHash = (Get-FileHash .\repvgg_cifar10_v1.0_weights.zip -Algorithm SHA256).Hash.ToLower()
$RepExpectedHash = ((Get-Content .\SHA256SUMS.txt -Raw).Trim() -split '\s+')[0]
if ($RepActualHash -ne $RepExpectedHash) { throw 'ZIP checksum mismatch' }
```

压缩包内含 baseline、conv1、full 各 42/43/44 三个种子的最佳训练权重与部署权重，共 18 份，以及日志、配置、测试指标、逐样本预测和加载示例。无需解压即可执行下一步验证。只加载来源可信的权重；校验和用于检查完整性。

## 3. 无需重训：验证真实测试结果

先快速验证一份完整方法权重：

```powershell
.\.venv\Scripts\python.exe verify_release.py --weights .\repvgg_cifar10_v1.0_weights.zip --data .\data --download-data --variants full --seeds 42 --output .\verification\full42.json
```

程序会下载并校验 CIFAR-10 测试数据、核对包内文件哈希、严格加载训练和部署结构、重新融合并逐参数比较，然后在 **10,000 张真实测试图像**上重新计算准确率与 logits 误差。首次数据下载可能较慢。CIFAR-10 官方数据不可达时，可从可信来源取得官方 Python 版数据并放在 `data/cifar-10-batches-py/`；数据文件仍由 torchvision 校验，不能用替代图像冒充。

验证所有九次实验：

```powershell
.\.venv\Scripts\python.exe verify_release.py --weights .\repvgg_cifar10_v1.0_weights.zip --data .\data --download-data --output .\verification\all_nine.json
```

验证输出保存在新文件中，不覆盖论文原始结果；输出已存在时请指定新的路径。`recorded_accuracy_matches` 表示重新计算的准确率是否与发表记录相同。程序每次都运行真实推理，不使用已有指标代替计算。在原硬件上通常为分钟级任务，实际以运行结果为准。

## 4. 验证预期结果

| 方法 | seed=42 | seed=43 | seed=44 | 均值±样本标准差 |
|---|---:|---:|---:|---:|
| baseline | 87.87% | 88.43% | 88.18% | 88.16%±0.28% |
| conv1 | 88.21% | 88.40% | 88.03% | 88.21%±0.19% |
| full | 88.78% | 87.80% | 88.26% | 88.28%±0.49% |

所有九次实验的融合前后标签不一致数均为 0；最大绝对 logits 误差（跨运行取最大）为 `1.33514e-5`。浮点精度、后端、线程和硬件变化可能使误差及极接近决策边界的预测略有不同，检查时应保留实际输出。

## 5. 从头训练

下载官方 CIFAR-10 训练与测试数据：

```powershell
.\.venv\Scripts\python.exe -c "from torchvision.datasets import CIFAR10; CIFAR10('data', train=True, download=True); CIFAR10('data', train=False, download=True)"
```

以下命令对九个组合分别从头训练，使用 **retrain** 标签，避免与仓库中 `main_*` 历史实验记录冲突：

```powershell
foreach ($RepSeed in @(42,43,44)) {
    foreach ($RepVariant in @('baseline','conv1','full')) {
        .\.venv\Scripts\python.exe -u train.py --data .\data --variant $RepVariant --seed $RepSeed --epochs 60 --batch-size 128 --threads 4 --lr 0.05 --tag retrain
        if ($LASTEXITCODE -ne 0) { throw 'Training failed; check the output before continuing' }
    }
}
```

如果中断，对相同参数追加 `--resume`，从最近完整轮次继续。不要直接用无参数的 `run_all.ps1` 从 GitHub 全新克隆开始训练：它面向作者的本机目录，且已提交的 `main_*` 配置会与新运行冲突。

重训结束后，重新计算测试、转换与测速，例如：

```powershell
.\.venv\Scripts\python.exe evaluate_and_export.py --checkpoint .\runs\retrain_full_s42\best.pt --data .\data --threads 4 --repeats 30
```

对其他八个组合使用相应检查点路径。新结果保存在 `results/retrain_*/`，原始结果保留在 `results/main_*/`。不要直接用 `run_evaluation.ps1` 验证重训：它面向原始 `main_*` 全套运行。

## 6. 复现范围与限制

- **权重验证**：可直接验证已有模型的测试准确率、融合等价性与加载行为，不需要等待训练。
- **训练复现**：代码、固定划分、超参数、三个种子与每轮日志齐全，可从头重跑；跨平台、后端和版本不保证所有权重逐位相同。
- **速度复现**：只能在说明硬件、精度、线程、batch、预热和计时范围后比较。另一台电脑的绝对延迟与加速比不要求相同。原实验是 Ryzen 7 5800H、CPU FP32、4 线程，计时不含数据加载与预处理。
- **结论边界**：full 的平均测试准确率仅高于 baseline 0.12 个百分点，三个种子不支持显著性结论。本文的主要实验证据是有限误差下的输出一致性与本机推理加速。
- **论文交付**：图表和最终论文尚需完成。仓库和权重公开并不等于论文已经完成；提交前应把真实作者、单位、论文 PDF、图表、代码链接与 Release 链接补齐。
