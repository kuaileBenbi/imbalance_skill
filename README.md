# imbalance-skill

一个 Claude Code skill，专门用于诊断和解决深度学习中的正负样本/类别不均衡问题。

## 这是什么

这不是一个 Python 库，而是一个 **Claude Code Agent 的领域技能**。安装后，你的 Claude Code 会获得正负样本不均衡方面的专家能力：

- 自动诊断你的数据不均衡程度
- 根据任务类型和不均衡程度推荐最佳解决方案组合
- 直接在代码中实现对应的 Loss 函数、注意力模块、采样策略
- 提供完整的训练 pipeline 和调参建议

**支持的方案覆盖三层**：

| 层面 | 解决什么 | 提供什么 |
|------|---------|---------|
| Loss 层 | 让少数类获得足够的梯度信号 | Focal Loss, Dice Loss, GHM, OHEM, Weighted CE |
| 架构层 | 减少不均衡的根源 | SE-Block, CBAM, ECA-Net 注意力模块 |
| 训练策略层 | 确保训练过程公平 | 均衡采样、梯度累积、两阶段训练、阈值搜索、模型校准 |

## 快速使用

### 方式一：直接安装到项目（推荐）

```bash
# 克隆到你的项目目录下
cd your-project
git clone https://github.com/your-username/imbalance-skill.git .claude/skills/imbalance

# 或者手动创建目录
mkdir -p .claude/skills/imbalance
# 然后把 skills/imbalance/ 下的文件复制进去
```

### 方式二：全局安装（所有项目可用）

```bash
# 克隆到用户级 skills 目录
git clone https://github.com/your-username/imbalance-skill.git ~/.claude/skills/imbalance
```

### 方式三：作为插件安装

```bash
# 克隆到本地
git clone https://github.com/your-username/imbalance-skill.git
cd imbalance-skill

# 用 --add-dir 加载（临时）
claude --add-dir /path/to/imbalance-skill
```

## 使用方法

安装后在 Claude Code 中输入：

```
/imbalance 我的欺诈检测数据集正样本只有 0.1%，怎么处理？
```

```
/imbalance 帮我看看这个检测项目的正负样本不均衡问题
```

Agent 会自动：
1. 诊断不均衡程度（计算 IR）
2. 根据任务类型推荐 Loss/架构/采样组合
3. 从代码模板中选取并适配到你的项目
4. 提供评估指标和调参建议

## 项目结构

```
imbalance-skill/
├── .claude-plugin/
│   └── plugin.json              # 插件配置
├── skills/
│   └── imbalance/
│       ├── SKILL.md             # 核心技能定义（知识 + 决策树 + 实现指南）
│       ├── templates/           # 可直接使用的 PyTorch 代码模板
│       │   ├── focal_loss.py    # Focal Loss
│       │   ├── dice_loss.py     # Dice Loss + Generalized Dice Loss
│       │   ├── ghm.py           # GHM-C + GHM-R
│       │   ├── ohem.py          # OHEM 在线难例挖掘
│       │   ├── weighted_ce.py   # 类别权重计算
│       │   ├── se_block.py      # SE 通道注意力
│       │   ├── cbam.py          # CBAM 通道+空间注意力
│       │   ├── eca_net.py       # ECA-Net 轻量注意力
│       │   ├── balanced_sampler.py   # 均衡批次采样
│       │   ├── temperature_scaling.py # Temperature Scaling 校准
│       │   └── gradient_accumulation.py # 梯度累积
│       └── examples/            # 端到端 Pipeline 示例
│           ├── classification_pipeline.md  # 二分类 pipeline
│           ├── detection_pipeline.md       # 目标检测 pipeline
│           └── segmentation_pipeline.md    # 语义分割 pipeline
├── LICENSE
└── README.md
```

## 覆盖的解决方案

### Loss 函数

| 方案 | 适用场景 | gamma/参数 |
|------|---------|-----------|
| Focal Loss | 中度~极端不均衡 | γ=2(标准), γ=5(极端) |
| Dice Loss | 分割的像素级不均衡 | 比值归一化 |
| GHM | 同时压制易样本和噪声 | 梯度密度加权 |
| OHEM | 检测的候选级不均衡 | 保留 top-K% 难样本 |
| Weighted CE | 轻度不均衡 | 逆频率/有效样本数 |

### 架构改进

| 方案 | 机制 | 特点 |
|------|------|------|
| SE-Block | 通道注意力 | 轻量通用 |
| CBAM | 通道+空间注意力 | 小目标场景 |
| ECA-Net | 1D Conv 注意力 | 超轻量 |

### 训练策略

| 方案 | 作用 |
|------|------|
| BalancedBatchSampler | 每 batch 保证正样本比例 |
| 梯度累积 | 等效大 batch，包含足够少数类 |
| Temperature Scaling | 校准训练后的概率输出 |
| 阈值搜索 | 找到最优决策阈值（替代默认 0.5） |
| 两阶段训练 | 先均衡数据学特征，后真实分布微调 |

## 参考文献

| 论文 | 会议 | 核心贡献 |
|------|------|---------|
| Focal Loss (Lin et al.) | ICCV 2017 | 按样本难度动态加权 |
| GHM (Li et al.) | AAAI 2019 | 梯度分布均衡化 |
| ATSS (Zhang et al.) | CVPR 2020 | 自适应标签分配 |
| SE-Net (Hu et al.) | CVPR 2018 | 通道注意力 |
| CBAM (Woo et al.) | ECCV 2018 | 通道+空间注意力 |
| SMOTE (Chawla et al.) | JAIR 2002 | 合成少数类样本 |
| Temperature Scaling (Guo et al.) | ICML 2017 | 模型概率校准 |

## License

MIT
