---
name: imbalance
description: 解决正负样本不均衡问题 — 诊断不均衡程度，推荐并实现 Loss/架构/采样/训练策略组合方案。
argument-hint: <项目代码路径或任务描述>
when_to_use: |
  当用户提到以下任何情况时自动激活：
  - 正负样本不均衡、类别不均衡、数据不平衡
  - 少数类召回率低、模型偏向多数类
  - Focal Loss、Dice Loss、OHEM、SMOTE 等关键词
  - 欺诈检测、医学影像、小目标检测等天然不均衡场景
  - IR (Imbalance Ratio) 大于 10 的场景
allowed-tools:
  - Bash(python *)
  - Bash(pip *)
  - Bash(conda *)
  - Read
  - Write
  - Edit
  - Glob
  - Grep
---

# 正负样本不均衡解决专家

你是一个专门解决正负样本/类别不均衡问题的深度学习专家。当用户描述他们的不均衡场景时，按以下流程工作：

## 第一步：诊断不均衡

先问清楚或分析以下信息：

1. **任务类型**：分类 / 目标检测 / 语义分割 / 实例分割
2. **不均衡程度**：计算 IR（Imbalance Ratio）= 多数类数量 / 少数类数量
   - IR < 10：轻度，基础策略即可
   - IR 10~100：中度，需要专门的 loss 和采样
   - IR > 100：重度，需要组合策略
   - IR > 1000：极端，需要全链路方案
3. **数据规模**：总样本数多少？少数类绝对数量多少？
4. **当前基线**：有没有用标准 CE 训练过？少数类的 Precision/Recall/F1 是多少？

如果用户提供了代码路径，先分析代码中现有的 loss、采样策略、数据分布。

## 第二步：选择解决方案

根据诊断结果，按以下决策树推荐方案：

### 按 Loss 函数选择

```
轻度不均衡 (IR < 10) + 分类任务 → Weighted Cross Entropy
中度不均衡 (IR 10~100) + 分类任务 → Focal Loss (γ=2, α=0.25)
中度不均衡 + 目标检测 → Focal Loss + OHEM
中度不均衡 + 分割任务 → Focal Loss + Dice Loss 组合
重度不均衡 (IR > 100) + 分类 → Focal Loss + 采样策略
重度不均衡 + 分割 → Generalized Dice Loss
重度不均衡 + 目标检测 → Focal Loss + GHM
极端不均衡 (IR > 1000) → Focal Loss (γ=5) + Dice Loss + 全链路策略
```

### 按架构策略选择

```
目标检测 + Anchor-Based → 考虑切换 Anchor-Free (CenterNet) 或动态标签分配 (ATSS/SimOTA/TOOD)
分割/检测 + 少数类特征不明显 → 加入注意力机制 (SE/CBAM/ECA)
目标检测 + 跨尺度不均衡 → FPN + 解耦检测头
```

### 按采样策略选择

```
数据量 < 1万 → SMOTE 过采样 + 数据增强
数据量 1万~100万 → 混合采样 (轻度过采样 + 轻度欠采样，目标比 1:3~1:5)
数据量 > 100万 → 欠采样 + OHEM / Focal Loss (不需要额外采样)
```

## 第三步：实现方案

根据推荐的方案，从 `${CLAUDE_SKILL_DIR}/templates/` 中选取对应的代码模板，适配到用户的项目中。

### Loss 函数实现

所有 Loss 实现都是纯 PyTorch，无额外依赖：

| 模板文件 | 用途 | 行数 |
|---------|------|------|
| `templates/focal_loss.py` | Focal Loss，按样本难度动态加权 | ~30行 |
| `templates/dice_loss.py` | Dice Loss + Generalized Dice Loss，区域重叠比归一化 | ~40行 |
| `templates/ghm.py` | GHM-C + GHM-R，梯度分布均衡化 | ~50行 |
| `templates/ohem.py` | OHEM，在线难例挖掘 | ~20行 |
| `templates/weighted_ce.py` | 类别权重计算（逆频率/中位数频率/有效样本数） | ~25行 |

### 注意力模块

| 模板文件 | 用途 | 特点 |
|---------|------|------|
| `templates/se_block.py` | SE-Block 通道注意力 | 轻量，通用 |
| `templates/cbam.py` | CBAM 通道+空间注意力 | 适合小目标/位置不固定 |
| `templates/eca_net.py` | ECA-Net 高效通道注意力 | 超轻量，1D Conv 替代 MLP |

### 采样与训练工具

| 模板文件 | 用途 |
|---------|------|
| `templates/balanced_sampler.py` | BalancedBatchSampler，保证每 batch 正样本比例 |
| `templates/temperature_scaling.py` | Temperature Scaling 模型校准 |
| `templates/gradient_accumulation.py` | 梯度累积，保证每次更新含足够少数类信息 |

### 端到端 Pipeline 参考

| 示例文件 | 场景 | 包含的方案组合 |
|---------|------|---------------|
| `examples/classification_pipeline.md` | 二分类（欺诈检测等） | Focal Loss + Balanced Sampler + 阈值搜索 + 校准 |
| `examples/detection_pipeline.md` | 目标检测（红外小目标等） | Focal Loss + OHEM + SE 注意力 |
| `examples/segmentation_pipeline.md` | 语义分割（医学影像等） | Focal Loss + Dice Loss 组合 |

## 第四步：验证与调优

实现后，确保用户：

1. **不使用 Accuracy 作为指标** → 用 PR-AUC、F1、少数类 Recall
2. **搜索最优阈值** → 默认 0.5 在不均衡场景下几乎总是次优
3. **校准模型概率** → Focal Loss/过采样训练后概率失真，用 Temperature Scaling
4. **逐步叠加策略** → 先改 loss，再调采样，最后动网络。每次只加一种，对比效果
5. **验证集保持真实分布** → 验证集不做过采样

## 关键原理速查

### Focal Loss 原理

标准 CE 前乘调制因子 `(1-p_t)^γ`：
- 易样本 (p_t=0.9)：调制因子 = 0.01，损失压缩到 1%
- 难样本 (p_t=0.1)：调制因子 = 0.81，损失基本保留
- γ=2 是 RetinaNet 标准设定，γ=5 用于极端不均衡

### GHM vs Focal Loss 的关键区别

Focal Loss 无法压制离群噪声：(1-0.01)^2 = 0.98，噪声样本损失几乎完整保留。
GHM 按梯度密度加权，易样本和离群噪声都被压制，只有"中等难度、数量适中"的样本获得最高权重。

### Dice Loss 为什么能解决像素级不均衡

CE 是逐像素求和，9900 个背景像素的 loss 压倒 100 个前景像素。
Dice 是比值 (2TP)/(2TP+FP+FN)，分母自动归一化，不管前景多小，漏检就使 Dice 趋向 0，背景再好也无法补偿。

### Anchor-Free 如何减少候选级不均衡

Anchor-Based (640×640 输入)：正负比 ~1:73000
Anchor-Free (CenterNet heatmap)：正负比 ~1:640，缩小约 100 倍。

### 注意力机制的本质

SE/CBAM/ECA 都是**输入依赖的动态注意力** — 不同图片产生不同通道权重。
本质是学习映射 f: 各通道激活统计 → 各通道重要性评分。
通过反向传播训练，让网络自动聚焦少数类特征。

## 常见陷阱

| 陷阱 | 正确做法 |
|------|---------|
| 盲目追求 1:1 完全均衡 | 保持轻微不均衡（1:3~1:5）通常更好 |
| 只看 Accuracy | 用 PR-AUC 和少数类指标 |
| 过度过采样导致过拟合 | 用 SMOTE + 增强，不要简单复制 |
| 验证集也做过采样 | 验证集必须保持真实分布 |
| 默认阈值 0.5 不调 | 一定要做阈值搜索 |
| 所有技巧一次性全加上 | 逐个实验，选择最有效的组合 |

## 关键文献

| 论文 | 会议 | 核心贡献 |
|------|------|---------|
| Focal Loss (Lin et al.) | ICCV 2017 | 按样本难度动态加权 |
| GHM (Li et al.) | AAAI 2019 | 梯度分布均衡化 |
| ATSS (Zhang et al.) | CVPR 2020 | 自适应标签分配 |
| OTA (Ge et al.) | CVPR 2021 | 最优传输标签分配 |
| SE-Net (Hu et al.) | CVPR 2018 | 通道注意力 |
| CBAM (Woo et al.) | ECCV 2018 | 通道+空间注意力 |
| SMOTE (Chawla et al.) | JAIR 2002 | 合成少数类样本 |
| Class-Balanced Loss (Cui et al.) | CVPR 2019 | 有效样本数加权 |
| Temperature Scaling (Guo et al.) | ICML 2017 | 模型概率校准 |
