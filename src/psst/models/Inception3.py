from torch import Tensor
from torch.nn import AdaptiveAvgPool2d, Flatten, Linear, MaxPool2d, Module, Sequential

from ._InceptionSupport import *


class Inception3(Module):
    """Inception-block-based neural network for training on 2D images."""

    __name__ = "Inception3"

    def __init__(self):
        super().__init__()

        self.stack = Sequential(
            BasicConv2d(1, 32, kernel_size=3, stride=2),
            BasicConv2d(32, 32, kernel_size=3),
            BasicConv2d(32, 64, kernel_size=3, padding=1),
            MaxPool2d(kernel_size=3, stride=2),
            BasicConv2d(64, 80, kernel_size=1),
            BasicConv2d(80, 192, kernel_size=3),
            MaxPool2d(kernel_size=3, stride=2),
            InceptionA(192, pool_features=32),
            InceptionA(256, pool_features=64),
            InceptionA(288, pool_features=64),
            InceptionB(288),
            InceptionC(768, channels_7x7=128),
            InceptionC(768, channels_7x7=160),
            InceptionC(768, channels_7x7=160),
            InceptionC(768, channels_7x7=192),
            InceptionD(768),
            InceptionE(1280),
            InceptionE(2048),
            AdaptiveAvgPool2d((1, 1)),
            Flatten(1),
            Linear(2048, 1),
            Flatten(0),
        )

    def _transform_input(self, x: Tensor) -> Tensor:
        if x.ndim == 3:
            x = x.view(x.shape[0], 1, x.shape[1], x.shape[2])
        return x

    def forward(self, x: Tensor) -> Tensor:
        x = self._transform_input(x)
        return self.stack(x)
