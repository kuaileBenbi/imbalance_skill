"""
CBAM (Convolutional Block Attention Module) — 通道 + 空间双重注意力。

先通道注意力（关注"哪些通道重要"），再空间注意力（关注"哪些位置重要"）。
适合小目标/少数类位置不固定的场景，比 SE 多了空间维度。

出处: Woo et al., "CBAM: Convolutional Block Attention Module" (ECCV 2018)
"""

import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(channels // reduction, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        avg_out = self.mlp(self.avg_pool(x).view(b, c))
        max_out = self.mlp(self.max_pool(x).view(b, c))
        s = torch.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        return x * s


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)

    def forward(self, x):
        avg_out = x.mean(dim=1, keepdim=True)
        max_out = x.max(dim=1, keepdim=True)[0]
        s = torch.cat([avg_out, max_out], dim=1)
        s = torch.sigmoid(self.conv(s))
        return x * s


class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, spatial_kernel=7):
        super().__init__()
        self.channel_att = ChannelAttention(channels, reduction)
        self.spatial_att = SpatialAttention(spatial_kernel)

    def forward(self, x):
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x
