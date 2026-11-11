# 论文写作骨架（未完成实验前不能作为最终论文提交）

## 中文题目
基于结构重参数化的VGG式卷积神经网络图像分类方法研究

作者：待填写真实姓名

单位：待填写学校、学院、城市、邮编

摘要：按“背景与目的—方法—实验设置—真实结果—结论”写 250～400 字。先完成实验再填性能数值。必须说明采用适配 CIFAR-10 的小型网络。若只做一个种子，不写显著提高或稳定优于。

关键词：图像分类；结构重参数化；RepVGG；CIFAR-10；模型部署

## English Title
Image Classification with VGG-Style Convolutional Neural Networks Based on Structural Re-parameterization

Authors: 待填写姓名拼音

Affiliation: 待填写学校与学院的正式英文名称

Abstract: 与最终中文摘要在任务、设置、数值和结论上严格对应，不能先预写 improved accuracy 或 faster inference。

Keywords: image classification; structural re-parameterization; RepVGG; CIFAR-10; model deployment

## 1 引言

说明计算机视觉分类任务与部署需求；多分支训练的潜在优化优势及推理开销。引出 RepVGG 的结构重参数化。明确本文是复现研究，目标是检验：分类效果、转换一致性和本机推理性能。将本文工作表述为实验贡献，而不是原创算法。

建议最后一段列出：构建统一部署拓扑的基线与分支消融；在真实数据上检验等价性；在指定 CPU 上测量延迟并分析收益与限制。所有结论须等待实验，不预设三个问题都成立。

## 2 相关工作

2.1 VGG 风格卷积网络：规整的 3×3 卷积堆叠与部署特征。

2.2 多分支与残差训练：讨论恒等路径，说明本文 BN 恒等分支与普通恒等映射的关系。

2.3 结构重参数化：以 RepVGG 为核心，说明训练结构与部署结构分离；与本文的小图像适配区分。

## 3 方法

### 3.1 任务与网络结构

输入 32×32 RGB 图像，输出 10 类 logits。给出 stage 表：32×32×16、16×16×32、8×8×64、4×4×128；块数分别 1、2、2、2。全局平均池化后线性分类。不把该模型命名为官方 RepVGG-A0。

### 3.2 训练块与消融

一般块：y = ReLU(BN₃(Conv₃(x)) + BN₁(Conv₁(x)) + BN_id(x))。

说明分支可选、恒等分支的尺寸限制、相加后只有一次 ReLU。baseline 也是 Conv-BN-ReLU，不能省略基线 BN 造成不公平对照。

### 3.3 卷积与 BatchNorm 融合

对输出通道 o，s_o = γ_o / sqrt(σ²_o + ε)。

通用公式：W'_o = s_o W_o，b'_o = β_o + s_o(b_o − μ_o)。本实现训练卷积无偏置，故 b_o=0。

将 1×1 核补零到 3×3 的中心；将恒等分支表示为中心位置为单位矩阵的 3×3 卷积核，再融合其 BN。

最终 W_deploy = W'₃ + pad(W'₁) + W'_id，b_deploy = b'₃ + b'₁ + b'_id。

强调条件：各分支相加之前为线性/仿射映射；BN 使用 eval 模式的固定运行统计；输入输出尺寸与填充对齐。实数运算下等价，浮点实现可能有舍入差异。

### 3.4 损失和训练

给出交叉熵公式与 SGD、余弦学习率设置，引用表 1。说明验证集选择模型，测试集独立评估。

## 4 实验与分析

### 4.1 数据、环境与协议

写明 CIFAR-10 的来源、数量、类别、训练/验证划分、增强、归一化；记录实际 CPU、PyTorch、线程、精度、种子、轮数。表 1 给设置，避免把计划值写成完成值。

### 4.2 主结果

表 2：方法、测试准确率、部署参数量、训练总耗时；可加每个种子。图 2：验证曲线。不要将验证集最高值当作测试准确率；不把训练准确率当作泛化能力。

### 4.3 分支消融

基线→增加 1×1→增加合法位置的恒等 BN。报告百分点变化。对提升、不提升、波动都作如实解释；多种子结合均值与标准差。

### 4.4 结构等价性

表 3：10,000 张真实测试图的最大绝对 logits 误差、相对 L2、预测不一致数量、融合前后准确率。可加 FP64 合成自检作为实现验证，但与训练后的真实测试结果明确分开。数值检查与代数推导共同支撑等价性，有限样本不能单独证明普遍数学命题。

### 4.5 推理性能

batch=1/128，固定 4 CPU 线程、FP32，预分配合成输入，inference_mode，20 次预热；30 组，每组 10 次前向；交错随机次序。表 4 给中位数、四分位数、吞吐量，加速比=转换前中位延迟/转换后中位延迟。计时不含数据读取和预处理，不据此声称端到端系统加速。

### 4.6 讨论与局限

逐条对应引言目标；讨论准确率与部署结构的关系、分支开销、CPU 特性、误差尺度。明确小网络、单数据集、单硬件、有限种子、非 ImageNet 原实验的范围。可用混淆矩阵说明哪些类别受益或退化；结论不得仅凭少量图片主观归因。

## 5 结论

用最终真实数字回应目标。给出本实验支持和未支持的结论，不扩大到未测试的 GPU、其他模型或数据集。下一步可以是 CIFAR-100、更大模型、GPU 与移动端测试。

## 代码与权重可用性

GitHub 仓库：待创建后填写真实链接。

权重 Release：待上传后填写真实链接，至少包括 baseline 与 full 的训练权重，建议附 full 部署权重、全部配置和日志。注明对应实验、种子、选中 epoch 与 SHA256。

## 参考文献准备

以下为起始文献，正文按实际使用顺序编号；不要引用未阅读的文献。除 RepVGG 官方元数据已核验外，其余条目在定稿前应再次按学校格式核对。数据集技术报告与 ImageNet/VGG 的数据和任务背景要分清。

1. Ding X, Zhang X, Ma N, Han J, Ding G, Sun J. RepVGG: Making VGG-Style ConvNets Great Again. CVPR, 2021: 13733–13742. https://openaccess.thecvf.com/content/CVPR2021/html/Ding_RepVGG_Making_VGG-Style_ConvNets_Great_Again_CVPR_2021_paper.html
2. Simonyan K, Zisserman A. Very Deep Convolutional Networks for Large-Scale Image Recognition. ICLR, 2015. https://arxiv.org/abs/1409.1556
3. Ioffe S, Szegedy C. Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift. ICML, 2015. https://proceedings.mlr.press/v37/ioffe15.html
4. He K, Zhang X, Ren S, Sun J. Deep Residual Learning for Image Recognition. CVPR, 2016. https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html
5. Krizhevsky A. Learning Multiple Layers of Features from Tiny Images. Technical report, University of Toronto, 2009. https://www.cs.toronto.edu/~kriz/cifar.html

## 提交前逐项核对

- 中英文题目、真实作者、单位、摘要、关键词齐全。
- 引言、相关工作、方法、实验、讨论、结论、参考文献齐全。
- 每个图表有编号、标题、单位，并在正文引用。
- 表格数字能在日志或指标 JSON 找到来源，所有单位一致。
- 没有待填写、编造数据、虚构 GitHub URL 或未完成实验的结论。
- baseline 和 full 的权重可从论文给出的链接下载，并成功加载。
- 说明复现范围、随机种子数量和限制；按照老师模板排版。
