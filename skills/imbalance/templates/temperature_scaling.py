"""
Temperature Scaling — 不均衡训练后的模型概率校准。

Focal Loss / 过采样训练后模型输出概率失真，需要校准使 P(正确) ≈ 模型置信度。
只学一个标量 T: calibrated_logits = logits / T，T 通常在 1.0 ~ 3.0 之间。

用法:
    temp = calibrate_model(trained_model, val_loader, device)
    # 推理时:
    calibrated_logits = temp(model_logits)

出处: Guo et al., "On Calibration of Modern Neural Networks" (ICML 2017)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemperatureScaling(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, logits):
        return logits / self.temperature


def calibrate_model(model, val_loader, device, max_iter=50):
    """在验证集上学习最优温度 T。不修改模型本身。"""
    model.eval()
    temp_scaling = TemperatureScaling().to(device)
    optimizer = torch.optim.LBFGS(
        [temp_scaling.temperature], lr=0.01, max_iter=max_iter
    )

    # 预先计算所有 logits
    all_logits, all_targets = [], []
    with torch.no_grad():
        for images, targets in val_loader:
            all_logits.append(model(images.to(device)))
            all_targets.append(targets.to(device))
    all_logits = torch.cat(all_logits)
    all_targets = torch.cat(all_targets)

    def closure():
        optimizer.zero_grad()
        loss = F.cross_entropy(temp_scaling(all_logits), all_targets)
        loss.backward()
        return loss

    optimizer.step(closure)
    print(f"Optimal temperature: {temp_scaling.temperature.item():.4f}")
    return temp_scaling
