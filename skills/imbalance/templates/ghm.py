"""
GHM (Gradient Harmonized Mechanism) — 按梯度密度加权，同时压制易样本和离群噪声。

与 Focal Loss 的关键区别：Focal Loss 无法压制离群噪声 (极端难样本)，
GHM 通过统计梯度分布，让"中等难度、数量适中"的样本获得最高权重。

权重 w_i = 1 / GD(g_i)，其中 g_i = 1 - p_t 是梯度范数，GD 是梯度密度。

出处: Li et al., "Gradient Harmonized Single-stage Detector" (AAAI 2019)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GHMC(nn.Module):
    """GHM-C: 分类用的梯度均衡化。"""

    def __init__(self, bins=100, momentum=0.0):
        super().__init__()
        self.bins = bins
        self.momentum = momentum
        self.register_buffer('edges', torch.linspace(0, 1, bins + 1))
        self.register_buffer('acc_sum', torch.zeros(bins))

    def forward(self, logits, targets):
        """
        Args:
            logits: 模型输出 (before sigmoid). Shape: (N,) or (N, C)
            targets: 标签. Shape: (N,) or (N, C), values in [0, 1]
        """
        probs = torch.sigmoid(logits)
        p_t = targets * probs + (1 - targets) * (1 - probs)
        g = 1.0 - p_t  # 梯度范数

        weights = torch.zeros_like(g)
        total = max(g.numel(), 1)

        for i in range(self.bins):
            inds = (g >= self.edges[i]) & (g < self.edges[i + 1])
            if self.momentum > 0:
                self.acc_sum[i] = (
                    self.momentum * self.acc_sum[i]
                    + (1 - self.momentum) * inds.sum().float()
                )
                density = self.acc_sum[i] / total
            else:
                density = inds.sum().float() / total

            if density > 0:
                weights[inds] = 1.0 / density

        weights = weights * (total / (weights.sum() + 1e-6))

        ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        return (weights * ce_loss).mean()


class GHMR(nn.Module):
    """GHM-R: 回归用的梯度均衡化 (smooth L1)。"""

    def __init__(self, bins=100, momentum=0.0, mu=0.02):
        super().__init__()
        self.bins = bins
        self.momentum = momentum
        self.mu = mu
        self.register_buffer('edges', torch.linspace(0, 1, bins + 1))
        self.register_buffer('acc_sum', torch.zeros(bins))

    def forward(self, pred, target):
        diff = pred - target
        abs_diff = diff.abs()
        is_asl = abs_diff > self.mu
        g = torch.where(is_asl, torch.ones_like(abs_diff), abs_diff / self.mu)

        weights = torch.zeros_like(g)
        total = max(g.numel(), 1)

        for i in range(self.bins):
            inds = (g >= self.edges[i]) & (g < self.edges[i + 1])
            if self.momentum > 0:
                self.acc_sum[i] = (
                    self.momentum * self.acc_sum[i]
                    + (1 - self.momentum) * inds.sum().float()
                )
                density = self.acc_sum[i] / total
            else:
                density = inds.sum().float() / total
            if density > 0:
                weights[inds] = 1.0 / density

        weights = weights * (total / (weights.sum() + 1e-6))
        loss = torch.where(is_asl, abs_diff - 0.5 * self.mu, 0.5 * diff ** 2 / self.mu)
        return (weights * loss).mean()
