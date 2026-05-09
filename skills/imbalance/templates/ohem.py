"""
OHEM (Online Hard Example Mining) — 只用 loss 最大的样本做梯度回传。

流程: 前向传播 → 按 loss 排序 → 取 top-K% → 只用这些样本回传。

适合目标检测中大量背景 anchor 淹没少数前景的场景。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class OHEMLoss(nn.Module):
    def __init__(self, ratio=0.25, loss_func=None):
        """
        Args:
            ratio: 保留最难的 top ratio 样本. Default: 0.25
            loss_func: 自定义 loss 函数，默认用 cross_entropy
        """
        super().__init__()
        self.ratio = ratio
        self.loss_func = loss_func

    def forward(self, logits, targets):
        if self.loss_func is not None:
            losses = self.loss_func(logits, targets)
        else:
            losses = F.cross_entropy(logits, targets, reduction='none')

        sorted_losses, _ = torch.sort(losses, descending=True)
        keep = max(1, int(len(sorted_losses) * self.ratio))
        return sorted_losses[:keep].mean()
