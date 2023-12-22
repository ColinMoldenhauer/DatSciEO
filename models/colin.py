from typing import Type, Union
import numpy as np
import torch
import torch.nn as nn

from torchvision.models.resnet import BasicBlock, Bottleneck, ResNet, conv1x1, resnet50


class TreeClassifConvNet(nn.Module):
    def __init__(self, n_classes=10, width=5, height=5, depth=30, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.n_classes = n_classes
        self.width = width
        self.height = height
        self.depth = depth
    
        self.model = nn.Sequential(
            nn.Conv2d(depth, depth//2, kernel_size=3, padding=1),       # e.g. 5x5x30 -> 5x5x15
            nn.ReLU(),
            nn.Conv2d(depth//2, depth//4, kernel_size=3, padding=1),    # e.g. 5x5x15 -> 5x5x7
            nn.ReLU(),
            nn.Conv2d(depth//4, 5, kernel_size=3, padding=1),           # e.g. 5x5x7 -> 5x5x5
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(5*5*5, n_classes)
        )
        

    def forward(self, x):
        return self.model(x)


def make_residual_layer(
        block: Type[Union[BasicBlock, Bottleneck]],
        inplanes: int,
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
        groups: int = 1,
        base_width: int = 64
    ) -> nn.Sequential:
        """
        Adapted from `torchvison.models.resnet.ResNet().make_residual_layer()`.
        Creates a layer of chained residual building blocks.

        block: base object
        inplanes: depth of input
        planes: depth of output
        blocks: number of blocks in sequence
        """

        norm_layer = nn.BatchNorm2d
        downsample = None
        dilation = 1
        previous_dilation = 1
        if dilate:
            dilation *= stride
            stride = 1
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                inplanes, planes, stride, downsample, groups, base_width, previous_dilation, norm_layer
            )
        )
        inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    inplanes,
                    planes,
                    groups=groups,
                    base_width=base_width,
                    dilation=dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)


class TreeClassifResNet50(nn.Module):
    """
    A ResNet inspired architecture (https://arxiv.org/pdf/1512.03385.pdf).
    Changed the stride to 1, such that the spatial dimensions are not reduced.
    """
    def __init__(self, n_classes=10, width=5, height=5, depth=30, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.n_classes = n_classes
        self.width = width
        self.height = height
        self.depth = depth

        block = Bottleneck
        self.model = nn.Sequential(
            make_residual_layer(block, depth, 64, 3),
            make_residual_layer(block, 256, 128, 4, stride=1),      # resnet uses stride=2, which shrinks image
            make_residual_layer(block, 512, 256, 6, stride=1),      # with a spatial extent of 5x5, we can't afford
            make_residual_layer(block, 1024, 512, 3, stride=1),     # shrinking
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512 * block.expansion, n_classes)
        )
        

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    # input size:  N, C_in,  H_in,  W_in
    # output size: N, C_out, H_out, W_out
    # H_out = (H_in + 2*pad - dil * (kernel-1) - 1) / stride  +  1

    input = torch.randn(1, 30, 5, 5)

    m1 = TreeClassifConvNet()
    m2 = TreeClassifResNet50(10, 5, 5, 30)

    
    print("Number of Parameters:")
    print(f"\tTreeClassifConvNet  {sum(p.numel() for p in m1.parameters()):>10d}")
    print(f"\tTreeClassifResNet50 {sum(p.numel() for p in m2.parameters()):>10d}")

    print("in", input.shape)
    print("m1", m1(input).shape)
    print("m2", m2(input).shape)

