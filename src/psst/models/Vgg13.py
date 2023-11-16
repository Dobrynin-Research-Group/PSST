from torch import Tensor
from torch.nn import Conv2d, Flatten, Linear, MaxPool2d, Module, ReLU, Sequential


class Vgg13(Module):
    """Very deep convolutional network for large-scale visual recognition of 2D images
    with 13 weight layers (https://www.robots.ox.ac.uk/~vgg/research/very_deep/).
    """

    __name__ = "Vgg13"

    def __init__(self):
        super().__init__()

        self.conv_stack = Sequential(
            # Convolutional layers
            Conv2d(1, 64, 3, padding=1),  # 224, 224, 64
            ReLU(),
            Conv2d(64, 64, 3, padding=1),  # 224, 224, 64
            ReLU(),
            MaxPool2d(2),  # 112, 112, 64
            Conv2d(64, 128, 3, padding=1),  # 112, 112, 128
            ReLU(),
            Conv2d(128, 128, 3, padding=1),  # 112, 112, 128
            ReLU(),
            MaxPool2d(2),  # 56, 56, 128
            Conv2d(128, 256, 3, padding=1),  # 56, 56, 256
            ReLU(),
            Conv2d(256, 256, 3, padding=1),  # 56, 56, 256
            ReLU(),
            MaxPool2d(2),  # 28, 28, 256
            Conv2d(256, 512, 3, padding=1),  # 28, 28, 512
            ReLU(),
            Conv2d(512, 512, 3, padding=1),  # 28, 28, 512
            ReLU(),
            MaxPool2d(2),  # 14, 14, 512
            Conv2d(512, 512, 3, padding=1),  # 14, 14, 512
            ReLU(),
            Conv2d(512, 512, 3, padding=1),  # 14, 14, 512
            ReLU(),
            MaxPool2d(2),  # 7, 7, 512
            # Fully connected layers
            Flatten(),
            Linear(25088, 4096),
            ReLU(),
            Linear(4096, 4096),
            ReLU(),
            Linear(4096, 1000),
            ReLU(),
            Linear(1000, 1),
        )

    def _transform_input(self, x: Tensor) -> Tensor:
        if x.ndim == 3:
            x = x.view(x.shape[0], 1, *x.shape[1:])
        return x

    def forward(self, x: Tensor) -> Tensor:
        x = self._transform_input(x)
        return self.conv_stack(x)
