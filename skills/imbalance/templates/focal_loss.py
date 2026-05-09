"""
Focal Loss — 按样本难度动态加权，让模型聚焦难分类样本。

替换标准 CrossEntropy 即可，不需要改网络结构。

FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

参数选择：
  gamma=1 → 温和聚焦，轻度不均衡
  gamma=2 → 标准设定，中度不均衡 (IR 10~1000)
  gamma=5 → 强聚焦，极端不均衡
  alpha=0.25 ~ 0.75，根据正负比例调整

出处: Lin et al., "Focal Loss for Dense Object Detection" (ICCV 2017)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        Args:
            logits: 模型原始输出 (before sigmoid). Shape: (N,) or (N, C)
            targets: 标签. Shape: (N,) or (N, C), values in [0, 1]
        """
        ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p_t = torch.exp(-ce_loss)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        loss = focal_weight * ce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss
