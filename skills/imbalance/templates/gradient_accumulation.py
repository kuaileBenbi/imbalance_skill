"""
梯度累积 — 当每个 batch 少数类样本过少时，累积多个 batch 的梯度再更新。

等效 batch_size = accumulation_steps × actual_batch_size
确保每次参数更新包含足够的少数类信息。

用法:
    accumulator = GradientAccumulator(model, optimizer, steps=4)
    for batch in dataloader:
        loss = model(batch) / accumulator.steps
        loss.backward()
        accumulator.step()
"""

import torch.nn as nn


class GradientAccumulator:
    def __init__(self, model, optimizer, steps=4):
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        self._count = 0

    def step(self):
        """每次 backward 后调用。累积够了 steps 次才执行 optimizer.step()。"""
        self._count += 1
        if self._count % self.steps == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()
