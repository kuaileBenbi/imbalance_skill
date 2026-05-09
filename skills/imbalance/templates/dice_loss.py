"""
Dice Loss — 基于区域重叠比，天然对类别比例不敏感。

Dice = 2|X ∩ Y| / (|X| + |Y|)
Dice Loss = 1 - Dice

适用于语义分割的类别不均衡。CE 逐像素求和会被大类主导，
Dice 是比值，分母自动归一化，漏检小目标直接使 Dice 趋向 0。

常用组合: L = L_Focal + lambda * L_Dice

模板包含：
  - DiceLoss: 二值分割
  - GeneralizedDiceLoss: 多类分割，带类别权重
"""

import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0, reduction='mean'):
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        Args:
            logits: 模型输出 (before sigmoid). Shape: (N, 1, H, W) or (N, H, W)
            targets: 标签 mask. Shape: (N, 1, H, W) or (N, H, W), values in [0, 1]
        """
        preds = torch.sigmoid(logits)
        spatial_dims = list(range(2, preds.ndim))
        intersection = (preds * targets).sum(dim=spatial_dims)
        union = preds.sum(dim=spatial_dims) + targets.sum(dim=spatial_dims)
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        loss = 1.0 - dice

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class GeneralizedDiceLoss(nn.Module):
    """多类分割的广义 Dice Loss。

    用类别权重 1/(sum^2) 自动平衡各类贡献。
    出处: Sudre et al., DLMIA 2017
    """

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        """
        Args:
            logits: 模型输出. Shape: (N, C, H, W)
            targets: One-hot 标签. Shape: (N, C, H, W), values in [0, 1]
        """
        preds = torch.sigmoid(logits)
        spatial_dims = list(range(2, preds.ndim))

        class_sums = targets.sum(dim=spatial_dims)
        weights = 1.0 / (class_sums ** 2 + self.smooth)

        intersection = (preds * targets).sum(dim=spatial_dims)
        union = preds.sum(dim=spatial_dims) + targets.sum(dim=spatial_dims)

        numerator = (weights * intersection).sum(dim=1)
        denominator = (weights * union).sum(dim=1)

        dice = (2.0 * numerator + self.smooth) / (denominator + self.smooth)
        return (1.0 - dice).mean()
