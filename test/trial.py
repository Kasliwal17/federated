import torch

from torch import nn


x = torch.randn(1, 3, 224, 224)
y = torch.randn(1, 3, 224, 224)
##Calculate mean of two tensors
z = (x + y) / 2
print(z.shape)