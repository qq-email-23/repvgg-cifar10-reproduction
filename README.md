# RepVGG 课程复现：从实验到完整论文

**外部读者请先看 [REPRODUCE.md](REPRODUCE.md)**：提供从全新克隆、安装环境、下载数据和 Release 权重，到验证全部九个模型及从头训练的完整命令，不依赖作者电脑的本地目录。

代码：[GitHub 仓库](https://github.com/qq-email-23/repvgg-cifar10-reproduction)。权重：[v1.0 Release](https://github.com/qq-email-23/repvgg-cifar10-reproduction/releases/tag/v1.0)

截止日期：2026 年 11 月 15 日（北京时间）。当前电脑：Ryzen 7 5800H、约 14 GB 可用物理内存、AMD 核显；本项目使用 CPU FP32。不承诺预先达到某个准确率或固定加速比。

**当前实验状态（11 月 11 日）**：三种模型×三个随机种子均已完成 60 轮训练，且全部完成官方 10,000 张测试图像的最终评估、融合一致性检查与 CPU 测速。测试准确率（均值±样本标准差）：baseline **88.16%±0.28%**，conv1 **88.21%±0.19%**，full **88.28%±0.49%**。九次运行转换前后标签变化均为 0，跨运行最大 logits 绝对误差约 **1.33514×10⁻⁵**。详见 [完整实验汇总](results/RESULTS_SUMMARY.md)。full 平均高出 baseline 0.12 个百分点，不据此声称显著提高。

## 1. 先明确你究竟在复现什么

已确定题目：**基于结构重参数化的VGG式卷积神经网络图像分类方法研究**。研究内容为 RepVGG 思想在 CIFAR-10 上的复现与实验分析。

本项目复现 RepVGG 的多分支训练与等价融合思想，使用适配 32×32 图像的小网络。不是原论文 ImageNet 实验的完整复现，也不是原版 RepVGG-A0，不能写成提出了一种全新的模型。

原论文：[CVPR 2021 官方页面](https://openaccess.thecvf.com/content/CVPR2021/html/Ding_RepVGG_Making_VGG-Style_ConvNets_Great_Again_CVPR_2021_paper.html)。官方代码：[DingXiaoH/RepVGG](https://github.com/DingXiaoH/RepVGG)。方法来自原论文，本项目是独立的教学实现。

你要回答三个问题：

1. 相同部署结构下，多分支训练能否提高分类准确率？完整 RepVGG 与单分支基线比较。
2. 多分支结构能否折叠为单个 3×3 卷积，同时保持数值一致？在真实测试集上比较融合前后的 logits、预测标签与准确率。
3. 折叠能否减少本机推理耗时？比较同一权重融合前后的延迟与吞吐量。

准确率与加速结论以文首列出的真实结果和结果表为准。数值误差约 1e-5 是可能的量级，并非必须达到的统一阈值；绝对误差受到 logits 尺度和浮点精度影响。不能仅凭“误差非零”否定数学等价。

## 2. 实验协议（训练前固定）

| 项目 | 设置 |
|---|---|
| 数据 | CIFAR-10 官方训练集 50,000；官方测试集 10,000 |
| 训练/验证 | 从官方训练集按类别固定划分 45,000/5,000，每类 4,500/500 |
| 划分随机种子 | 2026，保存到 configs/split_seed2026.json |
| 训练随机种子 | 先 42；有时间增加 43、44，三个变体都用同一组种子 |
| 输入 | 3×32×32 |
| 结构 | 四个 stage，通道 16/32/64/128，块数 1/2/2/2；后三个 stage 首块 stride=2 |
| 分类头 | 全局平均池化 + 128→10 全连接 |
| 激活 | 分支相加后 ReLU |
| 数据增强 | padding=4 的随机裁剪、随机水平翻转 |
| 归一化 | mean=(0.4914,0.4822,0.4465)，std=(0.2470,0.2435,0.2616) |
| 优化器 | SGD，lr=0.05，momentum=0.9，weight_decay=5e-4 |
| 训练 | batch=128，60 epochs，余弦学习率，CPU，4 线程 |
| 模型选择 | 仅按验证集准确率选择 best.pt；相同时保留较早轮次 |
| 最终测试 | 协议和模型选择固定后，在官方测试集评估 |

| 变体 | 训练结构 | 用途 |
|---|---|---|
| baseline | 3×3 Conv-BN | 普通 VGG 风格基线，同时也是“仅 3×3”消融 |
| conv1 | 3×3 Conv-BN + 1×1 Conv-BN | 消融 1×1 分支贡献 |
| full | 3×3 Conv-BN + 1×1 Conv-BN + 合法位置的恒等 BN | 完整 RepVGG 思想 |

**恒等分支仅在输入输出通道相等且 stride=1 时存在。** 三者部署后的卷积拓扑相同；基线与完整 RepVGG 部署后的速度相近是合理现象。不能将“多分支训练提高准确率”与“转换比普通 VGG 更快”混为一谈。对所有变体统一融合 BN 后再比较部署参数量。

## 3. 现在怎样运行（PowerShell）

本机已有的 Python 环境可复用，不修改旧项目。新项目的代码、日志、权重全部位于本目录；数据以只读方式使用已有 CIFAR-10 文件。

```powershell
Set-Location 'D:\图像处理与视觉\repvgg-course'
$RepPython = 'D:\图像处理与视觉\fasternet-course-restart\.venv\Scripts\python.exe'
$RepData = 'D:\图像处理与视觉\fasternet-reproduction\data'
```

检查融合和合成输入训练步耗时（不产生分类结论）：

```powershell
& $RepPython -u check_and_profile.py --steps 8
```

正式训练第一组全部三个模型，会自动续训已存在的同名检查点：

```powershell
.\run_all.ps1 -Python $RepPython -Data $RepData -Seeds 42
```

也可以直接执行 `.\run_all.ps1`，脚本会从当前项目所在目录自动定位相邻的 Python 环境和数据。脚本使用纯 ASCII 源码，兼容 Windows PowerShell 5.1，避免中文路径在无 BOM 的 UTF-8 脚本中被误读。仅检查环境而不启动训练可运行 `.\run_all.ps1 -CheckOnly`。正式运行后每完成一整轮训练与验证才打印一次指标，中间短暂无输出属于正常情况。

先跑一轮测得真实耗时：

```powershell
& $RepPython -u train.py --data $RepData --variant full --epochs 60 --threads 4 --tag main --stop-after 1
```

如果该轮已经运行，下一次必须使用 `--resume` 或运行 run_all.ps1，不能覆盖旧实验。`--stop-after 1` 只暂停，不改变 60 轮学习率计划。可用 Ctrl+C 停止，恢复时从最近完整轮次的检查点继续。保持接通电源，关闭自动睡眠，避免其他训练同时占用 CPU。正常关闭电脑后需要手动重新启动命令。

根据实际速度扩展到三个种子：

```powershell
.\run_all.ps1 -Python $RepPython -Data $RepData -Seeds 42,43,44
```

单个模型恢复示例：

```powershell
& $RepPython -u train.py --data $RepData --variant full --epochs 60 --threads 4 --tag main --resume
```

**不要在已有运行上修改 epochs、学习率、模型代码或训练代码。** 续训程序会检查关键配置和源文件哈希。如确需改变方案，使用新的 tag 重跑所有对照，并在论文中如实描述。

训练完成且固定实验协议后，依次进行最终测试与导出：

九个模型全部完成后，推荐直接使用以下命令，依次完成全部测试、等价性检查、部署权重导出与测速，再生成汇总表。已完成的评估会按权重哈希检查后跳过，避免覆盖已有结果。训练和其他高负载程序应先停止。使用固定的三个种子，不按测试结果选择或删除运行。

```powershell
.\run_evaluation.ps1
```

最终汇总位于 `results/RESULTS_SUMMARY.md`，逐次结果、三种子统计和延迟表分别为 `all_runs.csv`、`accuracy_summary.csv`、`all_timing.csv`。还会核对逐样本预测与记录的准确率，并验证九份部署权重可严格加载。每次评估的详细输出位于 `logs/eval_运行名.log`。以下是单独运行一个种子的等效低层命令，已经运行完整评估后无需再次执行：

```powershell
foreach ($RepVariant in @('baseline','conv1','full')) {
    & $RepPython evaluate_and_export.py --data $RepData --checkpoint "runs\main_${RepVariant}_s42\best.pt" --threads 4
    if ($LASTEXITCODE -ne 0) { throw '评估失败，请检查输出' }
}
```

多种子时将同样操作应用于 s43、s44。测试集不参与选轮次、调参或挑随机种子。`best.pt` 包含验证集选出的权重，`last.pt` 包含恢复训练状态，`results/运行名/deploy.pt` 为融合后的权重。只加载来源可信的 PyTorch 检查点。

权重加载示例（训练与部署结构不能混用）：

```python
import torch
from model import SmallRepVGG

# 训练结构，示例为完整方法；baseline 则使用相应路径与配置。
checkpoint = torch.load('runs/main_full_s42/best.pt', map_location='cpu', weights_only=False)
model = SmallRepVGG(branches=checkpoint['config']['variant'])
model.load_state_dict(checkpoint['model'])
model.eval()

# 部署结构。
checkpoint = torch.load('results/main_full_s42/deploy.pt', map_location='cpu', weights_only=False)
deployed = SmallRepVGG(branches=checkpoint['config']['variant'], deploy=True)
deployed.load_state_dict(checkpoint['model'])
deployed.eval()
with torch.inference_mode():
    logits = deployed(torch.randn(1, 3, 32, 32))
```

迁移环境时可新建 venv 后安装 requirements.txt，另行下载官方 CIFAR-10；不要将本地绝对路径照搬到其他电脑。绘图依赖目前可在正式绘图阶段安装。

## 4. 每一步产物是什么

| 产物 | 位置 | 用途 |
|---|---|---|
| 融合自检与合成输入耗时 | results/preflight.json | 代码可行性检查，不能当作训练结果 |
| 划分索引 | configs/split_seed2026.json | 验证数据划分可复现 |
| 实验配置及源文件哈希 | runs/运行名/config.json | 记录方法、种子、环境和代码版本 |
| 每轮训练与验证日志 | runs/运行名/history.csv | 绘制训练曲线 |
| 状态 | runs/运行名/status.json | 判断是否完成 60 轮 |
| 训练结构权重 | runs/运行名/best.pt、last.pt | 交付基线、提出方法与恢复训练 |
| 测试指标和延迟原始记录 | results/运行名/metrics.json | 所有最终表格的数据来源 |
| 逐样本预测 | results/运行名/test_predictions.npz | 混淆矩阵、错误案例、配对统计 |
| 部署权重 | results/运行名/deploy.pt | 独立验证融合后推理 |
| 全部运行的最终汇总 | results/RESULTS_SUMMARY.md | 主结果、等价性、延迟及限制 |
| 论文表格源数据 | results/all_runs.csv、accuracy_summary.csv、all_timing.csv | 保留全部种子和原始统计 |

## 5. 最少做哪些图表

1. 图 1：训练多分支结构与部署单分支结构图，标出 BN、求和、ReLU 和融合关系。
2. 图 2：三种方法的验证准确率曲线，横轴 epoch，明确这些是验证集而非测试集结果。
3. 表 1：数据集划分与训练超参数。
4. 表 2：三种方法的测试准确率、验证最优轮次、训练/部署参数量和训练耗时。
5. 表 3：真实测试集上的最大绝对 logits 误差、相对 L2 误差、预测不一致数量、融合前后准确率。
6. 图 3/表 4：batch=1 和 128 时的转换前后延迟中位数、四分位数和加速比。
7. 可选图 4：基线和完整方法的混淆矩阵，结合每类召回率讨论。

三种子时报告均值±样本标准差（ddof=1），并列出每个种子结果。单种子时明确写“单次运行”，不声称统计显著。测速重复 30 组用于延迟统计，不能当作 30 次独立训练。测速固定 CPU、FP32、线程数、batch、输入尺寸、预热和计时范围；程序采用交错随机顺序降低漂移，但本机温度与后台进程仍可能影响结果。

## 6. 如何写出“充分分析”

围绕引言中的三个问题逐一回答，先列数据，再解释，最后限定适用范围。

- full 比 baseline 高：报告百分点差值；多种子看改进是否一致。不能直接把原论文机制解释当成本实验已经证明的因果关系。
- full 没有提高：如实报告，讨论小网络、训练预算、小图像适配与随机波动；这不影响融合等价性实验成立。
- conv1 与 full 比较：说明恒等分支只出现在部分块，差异可能来自优化路径；单种子不宜作强因果结论。
- 融合前后误差：报告实际数值、相对误差及标签是否变化；BatchNorm 必须用 eval 模式的运行统计量。
- 融合后快：结合分支、BN 和逐元素求和减少解释；不要只凭参数量推断速度。
- batch=1 加速更明显或不明显：结合算子调度、线程开销和缓存解释，并承认这里测的是纯前向 CPU 延迟。
- 部署结构相同：基线和 full 的部署参数量应相同，不应声称重参数化使 full 部署模型比这个基线更小。

## 7. 从今天到截止前

- **11 月 10 日**：确定协议，跑融合检查、完整一轮测速，完成 seed=42 的三模型训练。
- **11 月 11 日**：检查训练日志和验证曲线，时间允许则完成 43、44 两个种子的对照。若发现程序错误，修复后统一重跑受影响实验。
- **11 月 12 日**：固定实验设置，最终测试、融合、测速、导出权重，制作图表。主线完成后再考虑 CIFAR-100；添加它需要所有相应对照，不能只跑一个模型。
- **11 月 13 日**：按 paper_outline.md 写完整中文正文和英文摘要，所有数值对应日志。
- **11 月 14 日**：完善讨论、参考文献、GitHub README 与权重发布，从新目录验证加载权重和复现命令。
- **11 月 15 日**：检查中英文信息、图表引用、单位、GitHub 链接与权重可下载性，按老师具体时间提交。尚未提供具体截止时刻，请以老师通知为准。

## 8. GitHub 交付（代码 + 权重必须都能访问）

1. 注册/登录 github.com，新建名为 repvgg-cifar10-reproduction 的公开仓库，不上传 CIFAR 数据集和虚拟环境。
2. 在本项目目录执行下列命令。先将尖括号里的姓名、邮箱、用户名替换成自己的信息：

```powershell
git init
git config user.name '<你的姓名>'
git config user.email '<你的邮箱>'
git add .
git commit -m 'Add reproducible RepVGG CIFAR-10 experiments'
git branch -M main
git remote add origin https://github.com/<你的用户名>/repvgg-cifar10-reproduction.git
git push -u origin main
```

首次推送按 Git 的浏览器提示登录。若提示已有仓库或 remote，先查看 `git status` 和 `git remote -v`，不要盲目重复初始化或覆盖远程。

3. 检查 `.gitignore`：本项目故意排除了 `.pt`、数据、虚拟环境和压缩包。训练配置、CSV、指标 JSON 应随代码提交。
4. 在 GitHub 仓库的 **Releases → Draft a new release** 创建 v1.0，上传 `baseline`、`full` 的 best.pt 及部署权重，建议同时上传 conv1。不同运行的 best.pt 应改名或分别打包，防止混淆；保留种子、变体和训练/部署标识。大文件不要直接放普通 Git 提交。
5. README 加入真实结果表、完整命令、硬件、权重下载链接和 SHA256；发布时以本机 `Get-FileHash -Algorithm SHA256` 或 metrics.json 校验。
6. 论文加“代码与权重可用性”段落，填入真实仓库 URL 和 Release URL；在未登录 GitHub 的浏览器窗口确认都能打开。

GitHub 代码仓库已公开。权重压缩包已在本机生成并校验，需在 Release 页面完成附件上传及 Publish release；公开下载状态以 Release 页面为准。论文定稿时填写作者、单位、邮箱，并确认学校对 AI 辅助写作的要求。
