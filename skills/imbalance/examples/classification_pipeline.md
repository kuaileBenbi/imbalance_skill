# 二分类不均衡 Pipeline（欺诈检测 / 医疗诊断等）

## 适用场景
- 正负比例 1:10 ~ 1:1000
- 二分类任务
- 中度到重度不均衡

## 推荐方案组合

```
Loss: Focal Loss (γ=2, α=0.25)
采样: BalancedBatchSampler (每 batch 保证正样本比例)
训练: 梯度累积 (steps=4)
评估: PR-AUC + F1 + 少数类 Recall
部署: Temperature Scaling 校准 + F1 最优阈值搜索
```

## 代码示例

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# === 1. Loss: Focal Loss ===
# 复制 templates/focal_loss.py 中的 FocalLoss 类
from focal_loss import FocalLoss

criterion = FocalLoss(alpha=0.25, gamma=2.0)

# === 2. 采样: BalancedBatchSampler ===
# 复制 templates/balanced_sampler.py
from balanced_sampler import BalancedBatchSampler

sampler = BalancedBatchSampler(
    labels=label_list,       # 全部标签
    batch_size=64,
    pos_per_batch=16,        # 25% 正样本
)
train_loader = DataLoader(dataset, batch_sampler=sampler)

# === 3. 训练循环 (带梯度累积) ===
# 复制 templates/gradient_accumulation.py
from gradient_accumulation import GradientAccumulator

accumulator = GradientAccumulator(model, optimizer, steps=4)

model.train()
for epoch in range(num_epochs):
    for images, targets in train_loader:
        logits = model(images).squeeze(-1)
        loss = criterion(logits, targets.float()) / accumulator.steps
        loss.backward()
        accumulator.step()

# === 4. 阈值搜索 ===
from sklearn.metrics import precision_recall_curve
import numpy as np

model.eval()
with torch.no_grad():
    probs = torch.sigmoid(model(all_features))

precisions, recalls, thresholds = precision_recall_curve(
    all_labels.numpy(), probs.numpy()
)
f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-8)
best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]
print(f"Optimal threshold: {best_threshold:.4f}, F1: {f1_scores[best_idx]:.4f}")

# === 5. 模型校准 ===
# 复制 templates/temperature_scaling.py
from temperature_scaling import calibrate_model

temp = calibrate_model(model, val_loader, device)

# 推理时:
# calibrated_logits = temp(model(images))
# probs = torch.softmax(calibrated_logits, dim=1)
```

## 调参建议

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| Focal alpha | 0.25 ~ 0.5 | 正样本越少 alpha 越大 |
| Focal gamma | 2.0 | 标准设定，极端不均衡用 5.0 |
| batch pos比例 | 25% ~ 50% | 不必 1:1 |
| 梯度累积步数 | 4 ~ 8 | 等效 batch 足够大即可 |
| 学习率 | 1e-3 (Adam) | 配合 Focal Loss 通常稳定 |
