# coding:utf-8
import os
import sys

import torch
from torch.distributions import Categorical

# 定义概率分布
probs = torch.tensor([0.1, 0.4, 0.5])

# 创建 Categorical 分布
dist = Categorical(probs)

# 从分布中采样
sample = dist.sample()
print(sample)

# 计算 log_prob
log_prob = dist.log_prob(sample)
print(log_prob)



if __name__ == "__main__":
    pass
