# coding:utf-8
import os
import sys
import torch

# 创建一个 1x12 的张量
x = torch.arange(12)

# 使用 view 改变形状为 3x4
y = x.view(3, 4)
z = x.reshape([3, 4])

print("Original tensor:")
print(x)
print("Reshaped tensor:")
print(y)
print(z)

if __name__ == "__main__":
    pass
