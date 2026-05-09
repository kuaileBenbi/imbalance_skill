"""
ECA-Net (Efficient Channel Attention) — 超轻量通道注意力。

用 1D Conv 替代 SE 的 MLP 瓶颈结构，参数更少但效果相当。
核大小自适应计算: k = |log2(C) / gamma + b / gamma|

出处: Wang et al., "ECA-Net: Efficient Channel Attention" (CVPR 2020)
"""

import math
import torch
import torch.nn as nn


class ECABlock(nn.Module):
    def __init__(self, channels, gamma=2, b=1):
        super().__init__()
        kernel_size = int(abs((math.log2(channels) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        kernel_size = max(kernel_size, 3)

        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x):
        b, c, _, _ = x.size()
        s = self.squeeze(x).view(b, 1, c)
        s = torch.sigmoid(self.conv(s)).view(b, c, 1, 1)
        return x * s
