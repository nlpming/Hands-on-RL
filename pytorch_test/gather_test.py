# coding:utf-8
import os
import sys
import torch

# 创建输入张量
input = torch.tensor([[1, 2], [3, 4], [5, 6]])

# 创建索引张量
index = torch.tensor([[0, 0], [1, 0]])

# 在 dim=0 维度上聚集值
result = torch.gather(input, 0, index)
print(result)
# tensor([[1, 2],
#         [3, 2]])

# result = torch.gather(input, 1, index)
# print(result)
# tensor([[1, 1],
#         [4, 3]])

if __name__ == "__main__":
    pass
