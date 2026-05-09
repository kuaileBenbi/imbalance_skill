"""
类别权重计算工具 — 用于 Weighted Cross Entropy。

三种权重策略：
  - inverse_freq: w_i = N / (C * n_i)，逆频率
  - median_freq: w_i = median(freq) / freq_i，中位数频率
  - effective_num: w_i = (1-beta) / (1-beta^n_i)，有效样本数法 (Class-Balanced Loss)

用法:
    weights = compute_class_weights([10000, 100], method='inverse_freq')
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))

出处: Cui et al., "Class-Balanced Loss Based on Effective Number of Samples" (CVPR 2019)
"""

import torch


def compute_class_weights(class_counts, method='inverse_freq', beta=0.9999):
    """
    Args:
        class_counts: 各类样本数列表, e.g. [10000, 100]
        method: 'inverse_freq' | 'median_freq' | 'effective_num'
        beta: effective_num 方法的参数

    Returns:
        Tensor of class weights
    """
    counts = torch.tensor(class_counts, dtype=torch.float32)
    total = counts.sum()
    num_classes = len(class_counts)

    if method == 'inverse_freq':
        weights = total / (num_classes * counts)
    elif method == 'median_freq':
        freqs = counts / total
        median_freq = freqs.median()
        weights = median_freq / freqs
    elif method == 'effective_num':
        effective_num = 1.0 - beta ** counts
        weights = (1.0 - beta) / effective_num
    else:
        raise ValueError(f"Unknown method: {method}")

    return weights / weights.sum() * num_classes
