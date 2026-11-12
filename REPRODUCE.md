# 从 GitHub 复现结果

本文题目：**基于结构重参数化的VGG式卷积神经网络图像分类方法研究**。

代码：https://github.com/qq-email-23/repvgg-cifar10-reproduction

权重：https://github.com/qq-email-23/repvgg-cifar10-reproduction/releases/tag/v1.0

原始训练与评估代码提交：`4053d74bcfdb784e902c592b69ce703df46dec04`。后续提交补充外部验证入口与文档，保持 `model.py`、`train.py`、`common.py` 不变。原始实验配置中的本地绝对路径仅是历史记录，验证程序不使用这些路径。

**已执行的复现检查**：在独立克隆目录中，使用 Release 对应 ZIP 对全部九个模型重新运行真实测试集推理。九次准确率均与原记录一致，所有融合前后标签不一致数均为 0，报告见 [results/reproduction_check.json](results/reproduction_check.json)。该检查复用了作者本机 Python 环境和已有数据；尚未验证另一台电脑的全新依赖安装与数据下载，不把它称为跨设备复现。v1.0 Release 已公开，包含权重 ZIP 和校验文件。

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

GitHub 自动提供的 Source code 压缩包不包含 `.pt` 权重，请下载 Release 的权重附件。

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

安装好环境后，可以使用一键脚本下载数据并训练全部九个组合（3 种方法 × 3 个种子）：

```powershell
.\run_all.ps1 -DownloadData
```

默认读取仓库中的 `.venv\Scripts\python.exe`，数据保存在 `data/`，训练 60 轮、batch=128、学习率 0.05、4 个 CPU 线程。新运行默认命名为 `retrain_*`，与论文原始 `main_*` 实验记录区分。已下载数据时不必加 `-DownloadData`。

只检查环境、已有数据完整性及运行目录，不开始训练：

```powershell
.\run_all.ps1 -CheckOnly
```

该检查需要先取得数据；`-CheckOnly` 不会下载数据。使用已有 Python 和数据目录时，可分别传入 `-Python '解释器完整路径' -Data '数据父目录完整路径'`。自定义路径相对于运行命令时的当前目录解析，默认路径相对于仓库解析。

如果 PowerShell 提示禁止执行脚本，可仅在当前终端允许本地脚本后重试：

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

中断后使用相同命令重跑，脚本自动从已有 `last.pt` 继续；续训需要保持原来的轮数、线程等训练参数一致。需要进行另一组独立实验时，指定新的 `-Tag retrain2`；`main` 标签保留给论文原始实验。

全部训练完成后，一键执行测试、转换、导出及测速：

```powershell
.\run_evaluation.ps1 -CheckOnly
.\run_evaluation.ps1
```

评估脚本在开始前检查全部九个运行是否训练完成。新指标、部署权重和逐样本预测分别保存在 `results/retrain_*/`。原论文汇总表不自动改写，比较新实验时读取对应的新 `metrics.json`。已存在的结果目录会阻止重复评估，避免覆盖；若上次评估中断，先保留并移走该次生成的结果目录，再重试。使用自定义训练标签、种子或方法时，评估必须传入相同的 `-Tag`、`-Seeds`、`-Variants`，自定义 Python、数据目录也需一致。

例如只重训并评估 full、seed=42（这是单模型演示，不是完整九组实验）：

```powershell
.\run_all.ps1 -Variants full -Seeds 42 -Tag demo -DownloadData
.\run_evaluation.ps1 -Variants full -Seeds 42 -Tag demo
```

也可以不用 PowerShell 脚本，直接调用 Python：

```powershell
.\.venv\Scripts\python.exe -u train.py --data .\data --variant full --seed 42 --epochs 60 --batch-size 128 --threads 4 --lr 0.05 --tag manual
.\.venv\Scripts\python.exe evaluate_and_export.py --checkpoint .\runs\manual_full_s42\best.pt --data .\data --threads 4 --repeats 30
```

直接调用 Python 续训时追加 `--resume`。其他八个组合替换 `--variant` 和 `--seed`。非 Windows 系统请使用此方式，并替换为相应的 Python 解释器路径。

## 6. 复现范围与限制

- **权重验证**：可直接验证已有模型的测试准确率、融合等价性与加载行为，不需要等待训练。
- **训练复现**：代码、固定划分、超参数、三个种子与每轮日志齐全，可从头重跑；跨平台、后端和版本不保证所有权重逐位相同。
- **速度复现**：只能在说明硬件、精度、线程、batch、预热和计时范围后比较。另一台电脑的绝对延迟与加速比不要求相同。原实验是 Ryzen 7 5800H、CPU FP32、4 线程，计时不含数据加载与预处理。
- **结论边界**：full 的平均测试准确率仅高于 baseline 0.12 个百分点，三个种子不支持显著性结论。本文的主要实验证据是有限误差下的输出一致性与本机推理加速。
