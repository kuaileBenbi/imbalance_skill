# 语义分割不均衡 Pipeline（医学影像 / 红外分割等）

## 适用场景
- 像素级不均衡，前景占 0.1% ~ 10%
- 语义分割 / 实例分割
- 重度像素级不均衡

## 推荐方案组合

```
Loss: Focal Loss + Dice Loss 组合 (最常见且有效)
  - Focal Loss: 微观视角，按像素难度加权
  - Dice Loss: 宏观视角，按区域重叠比评估
多类: Generalized Dice Loss
训练: 两阶段 (先均衡数据，后真实分布)
评估: Dice Score + mIoU + 少数类 IoU
部署: 类别级独立阈值
```

## 代码示例

```python
import torch
import torch.nn as nn

# === 1. Loss: Focal Loss + Dice Loss 组合 ===
# 复制 templates/focal_loss.py 和 templates/dice_loss.py
from focal_loss import FocalLoss
from dice_loss import DiceLoss, GeneralizedDiceLoss

focal = FocalLoss(alpha=0.25, gamma=2.0)
dice = DiceLoss(smooth=1.0)

def combined_loss(logits, targets, focal_weight=1.0, dice_weight=1.0):
    """
    Focal Loss 处理像素级难度，Dice Loss 处理区域级不均衡。
    两者互补是分割任务的标准组合。
    """
    return (focal_weight * focal(logits, targets)
            + dice_weight * dice(logits, targets))

# === 2. 训练循环 ===
model = UNet(in_channels=1, num_classes=1)  # 替换为你的模型
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

model.train()
for epoch in range(num_epochs):
    for images, masks in train_loader:
        logits = model(images)
        loss = combined_loss(logits, masks, focal_weight=1.0, dice_weight=1.0)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 监控 Dice Score
        with torch.no_grad():
            preds = torch.sigmoid(logits)
            intersection = (preds * masks).sum()
            dice_score = (2 * intersection) / (preds.sum() + masks.sum() + 1e-8)

# === 3. 多类分割: Generalized Dice Loss ===
gdice = GeneralizedDiceLoss(smooth=1.0)

# targets 需要是 one-hot: (N, C, H, W)
# loss = gdice(logits, targets_onehot)

# === 4. 两阶段训练 ===
# 阶段一: 用均衡数据 (对少数类过采样/增强) 训练至收敛
#   - 让模型学到少数类的基本特征
#   - Loss: CE 或 Focal Loss
#   - LR: 1e-3

# 阶段二: 用原始不均衡数据微调
#   - 让模型适应真实分布
#   - Loss: Focal + Dice
#   - LR: 1e-4 (降低 10 倍)
```

## 增强策略

```python
from torchvision import transforms

# 少数类专用更强增强
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomVerticalFlip(0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.Normalize(mean=[...], std=[...]),
])

# Mixup/CutMix 是 batch 级增强，在训练循环里做:
# from timm.data.mixup import Mixup
# mixup = Mixup(mixup_alpha=0.8, cutmix_alpha=1.0, prob=1.0)
# images, masks = mixup(images, masks)  # 注意: 混合后的 mask 也变了
```

## 调参建议

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| focal_weight | 1.0 | Focal Loss 权重 |
| dice_weight | 1.0 | Dice Loss 权重 |
| dice smooth | 1.0 | 防除零 |
| 两阶段 LR 比 | 10:1 | 阶段二学习率降低 10 倍 |
| 增强强度 | 少数类更强 | 类别差异化增强 |

## 常见问题

**Q: Dice Loss 训练不稳定？**
A: 配合 Focal Loss 使用。纯 Dice Loss 在极端不均衡下梯度可能不稳定，组合 Focal Loss 可以稳定训练。

**Q: Generalized Dice Loss 中某些类权重爆炸？**
A: 增大 smooth 参数，或对权重做 clamp。极小的类可能 weight = 1/(sum^2) 过大。
