# 目标检测不均衡 Pipeline（红外小目标 / 稀有目标检测等）

## 适用场景
- 正负比例 1:1000 ~ 1:100000（Anchor-Based）
- 目标检测任务
- 重度候选级不均衡

## 推荐方案组合

```
Loss: Focal Loss (分类) + GIoU/CIoU Loss (回归)
难例挖掘: OHEM
架构: FPN + 解耦检测头 + SE/CBAM 注意力
标签分配: SimOTA (YOLOv8 默认) 或 ATSS (mmdetection)
数据增强: Mosaic + Copy-Paste
评估: mAP@0.5:0.95 + 各类 AP
部署: Soft-NMS + 类别级独立阈值
```

## 代码示例

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# === 1. Loss: Focal Loss + OHEM ===
# 复制 templates/focal_loss.py 和 templates/ohem.py
from focal_loss import FocalLoss
from ohem import OHEMLoss

focal = FocalLoss(alpha=0.25, gamma=2.0, reduction='none')

def detection_loss(cls_logits, reg_preds, cls_targets, reg_targets):
    # 展平: (B, A, H, W) -> (N,)
    cls_flat = cls_logits.permute(0, 2, 3, 1).reshape(-1)
    tgt_flat = cls_targets.permute(0, 2, 3, 1).reshape(-1)

    # 分类 loss: Focal Loss (per-sample)
    cls_loss_per_sample = focal(cls_flat, tgt_flat)

    # OHEM: 只保留最难的 25%
    sorted_loss, _ = torch.sort(cls_loss_per_sample, descending=True)
    keep = max(1, int(len(sorted_loss) * 0.25))
    cls_loss = sorted_loss[:keep].mean()

    # 回归 loss: 只在正样本上计算
    pos_mask = tgt_flat > 0.5
    if pos_mask.sum() > 0:
        reg_flat = reg_preds.permute(0, 2, 3, 1).reshape(-1, 4)
        tgt_reg = reg_targets.permute(0, 2, 3, 1).reshape(-1, 4)
        reg_loss = F.smooth_l1_loss(reg_flat[pos_mask], tgt_reg[pos_mask])
    else:
        reg_loss = torch.tensor(0.0, device=cls_logits.device)

    return cls_loss + reg_loss

# === 2. 检测头 + SE 注意力 ===
# 复制 templates/se_block.py
from se_block import SEBlock

class DetectionHead(nn.Module):
    def __init__(self, in_channels=128, num_anchors=3):
        super().__init__()
        self.backbone_neck = nn.Sequential(
            nn.Conv2d(in_channels, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.se = SEBlock(128, reduction=16)
        self.cls_head = nn.Conv2d(128, num_anchors, 1)
        self.reg_head = nn.Conv2d(128, num_anchors * 4, 1)

    def forward(self, x):
        feat = self.backbone_neck(x)
        feat = self.se(feat)
        return self.cls_head(feat), self.reg_head(feat)
```

## 框架集成

### YOLOv8 (ultralytics) — SimOTA 默认支持

```yaml
# config.yaml
model: yolov8n.pt
data: custom_dataset.yaml
epochs: 300
loss:
  cls_loss: FocalLoss     # 默认已是
  box_loss: CIoULoss       # 默认已是
mosaic: 1.0                # 开启 Mosaic 增强
copy_paste: 0.1            # Copy-Paste 概率
```

### mmdetection — ATSS 配置

```python
# config.py
model = dict(
    type='ATSS',
    bbox_head=dict(
        type='ATSSHead',
        num_classes=1,                    # 单类检测
        anchor_generator=dict(
            strides=[8, 16, 32, 64, 128],
            scales=[8],
            ratios=[1.0],                 # 正方形 anchor
        ),
        loss_cls=dict(type='FocalLoss', gamma=2.0, alpha=0.25),
        loss_bbox=dict(type='GIoULoss'),
    ),
)
```

## 调参建议

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| OHEM ratio | 0.25 | 保留最难的 25% |
| SE reduction | 16 | 通道降维比 |
| Anchor scales | [8] | 红外小目标用小 anchor |
| Mosaic | 1.0 | 训练全程开启 |
| Copy-Paste | 0.1 | 少数类目标粘贴 |
