import torch
import numpy as np
from torchvision import transforms


class RandomRotation(transforms):
    def __init__(self, degrees, p=1, **kwargs):
        super().__init__(degrees, **kwargs)
        self.p = p

    def forward(self, img):
        if torch.rand(1).item() < self.p:
            return super().forward(img)
        return img
