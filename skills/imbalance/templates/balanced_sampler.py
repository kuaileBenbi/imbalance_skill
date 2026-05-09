"""
BalancedBatchSampler — 保证每个训练 batch 包含固定的正样本比例。

随机采样在严重不均衡下会导致很多 batch 完全没有正样本，
这个 sampler 强制每 batch 至少包含指定数量的正样本。

用法:
    sampler = BalancedBatchSampler(labels, batch_size=64, pos_per_batch=16)
    loader = DataLoader(dataset, batch_sampler=sampler)
"""

import random
from torch.utils.data import Sampler


class BalancedBatchSampler(Sampler):
    def __init__(self, labels, batch_size, pos_per_batch, shuffle=True):
        """
        Args:
            labels: 数据集标签列表
            batch_size: 总 batch 大小
            pos_per_batch: 每个 batch 中正样本数量
            shuffle: 是否打乱
        """
        self.batch_size = batch_size
        self.pos_per_batch = min(pos_per_batch, batch_size)
        self.neg_per_batch = batch_size - self.pos_per_batch
        self.shuffle = shuffle

        self.pos_indices = [i for i, l in enumerate(labels) if l == 1]
        self.neg_indices = [i for i, l in enumerate(labels) if l == 0]

        if not self.pos_indices or not self.neg_indices:
            raise ValueError("正负样本都需要存在")

        self.num_batches = min(
            len(self.pos_indices) // max(self.pos_per_batch, 1),
            len(self.neg_indices) // max(self.neg_per_batch, 1),
        )
        self.num_batches = max(self.num_batches, 1)

    def __iter__(self):
        pos_pool = list(self.pos_indices)
        neg_pool = list(self.neg_indices)
        if self.shuffle:
            random.shuffle(pos_pool)
            random.shuffle(neg_pool)

        for _ in range(self.num_batches):
            batch = []
            for _ in range(self.pos_per_batch):
                batch.append(pos_pool[random.randint(0, len(pos_pool) - 1)])
            for _ in range(self.neg_per_batch):
                batch.append(neg_pool[random.randint(0, len(neg_pool) - 1)])
            if self.shuffle:
                random.shuffle(batch)
            yield batch

    def __len__(self):
        return self.num_batches
