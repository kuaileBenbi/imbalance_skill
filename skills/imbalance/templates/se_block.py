"""
SE-Block (Squeeze-and-Excitation) — 通道注意力，让网络自动聚焦少数类特征。

流程: GAP → FC 降维 → ReLU → FC 升维 → Sigmoid → 逐通道缩放

用法: 插入到 ResNet 残差块的卷积之后、残差相加之前。

出处: Hu et al., "Squeeze-and-Excitation Networks" (CVPR 2018)
"""

import torch
import torch.nn as nn


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(channels // reduction, 1)
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        s = self.squeeze(x).view(b, c)
        s = self.excitation(s).view(b, c, 1, 1)
        return x * s


# ====== 插入到 ResNet 的示例 ======

class SEResBlock(nn.Module):
    """在残差块中加入 SE 注意力。"""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se = SEBlock(out_channels, reduction=16)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        identity = x
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)  # 在残差相加前应用 SE
        if self.downsample:
            identity = self.downsample(x)
        return torch.relu(out + identity)
