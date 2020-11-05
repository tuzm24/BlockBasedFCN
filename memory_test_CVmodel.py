import torchvision
import torch
from torchvision.models import densenet121

def calcTensorSize(tensor):
    return torch.numel(tensor) * 4

a = torch.randn(1,2,3,4,5)
print(calcTensorSize(a))
