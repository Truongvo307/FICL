"""CNNs trained from scratch. Each maps images [B, 3, H, W] to features [B, feature_dim]; the linear head
is added by network.build_network().

    TinyCNN         two 3x3 conv layers with batch norm, 32 features (pipeline checks only)
    ResNet18Cifar   ResNet-18 with the CIFAR stem: 3x3 conv with stride 1 and no max pool, then four stages
                    of two basic blocks (64, 128, 256, 512 channels), global average pool, 512 features
    ResNet32        the CIFAR ResNet-32 of He et al. (2016): 3x3 conv with 16 channels, then three stages of
                    five basic blocks (16, 32, 64 channels), parameter-free shortcuts (subsampling and zero
                    channels), global average pool, 64 features

Convolutions are initialised with kaiming_normal_(mode="fan_out", nonlinearity="relu"), batch norm with
weight 1 and bias 0 (as torchvision's ResNets). The caller seeds the generator. Global pooling is a mean
over the spatial positions (not nn.AdaptiveAvgPool2d, whose CUDA backward is not deterministic).
"""
from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F


def _init(module: nn.Module) -> None:
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)


def _conv3x3(cin: int, cout: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(cin, cout, kernel_size=3, stride=stride, padding=1, bias=False)


class GlobalAvgPool(nn.Module):
    """[B, C, H, W] -> [B, C], the mean over positions."""

    def forward(self, x):
        return x.mean(dim=(2, 3))


class TinyCNN(nn.Module):
    feature_dim = 32

    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(
            _conv3x3(3, 16), nn.BatchNorm2d(16), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            _conv3x3(16, 32), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            GlobalAvgPool())
        _init(self)

    def forward(self, x):
        return self.body(x)


class BasicBlock(nn.Module):
    """Two 3x3 convolutions with batch norm; projection shortcut (1x1 conv + batch norm) when the shape changes."""

    def __init__(self, cin: int, cout: int, stride: int = 1):
        super().__init__()
        self.conv1, self.bn1 = _conv3x3(cin, cout, stride), nn.BatchNorm2d(cout)
        self.conv2, self.bn2 = _conv3x3(cout, cout), nn.BatchNorm2d(cout)
        self.shortcut = nn.Sequential()
        if stride != 1 or cin != cout:
            self.shortcut = nn.Sequential(nn.Conv2d(cin, cout, kernel_size=1, stride=stride, bias=False),
                                          nn.BatchNorm2d(cout))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


class ResNet18Cifar(nn.Module):
    feature_dim = 512

    def __init__(self):
        super().__init__()
        self.conv1, self.bn1 = _conv3x3(3, 64), nn.BatchNorm2d(64)
        layers, cin = [], 64
        for cout, stride in ((64, 1), (128, 2), (256, 2), (512, 2)):
            layers.append(nn.Sequential(BasicBlock(cin, cout, stride), BasicBlock(cout, cout, 1)))
            cin = cout
        self.layer1, self.layer2, self.layer3, self.layer4 = layers
        self.pool = GlobalAvgPool()
        _init(self)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return self.pool(x)


class ShortcutA(nn.Module):
    """Parameter-free shortcut: keep every stride-th pixel, add zero channels on both sides."""

    def __init__(self, cin: int, cout: int, stride: int):
        super().__init__()
        self.stride, extra = int(stride), int(cout - cin)
        self.pad_front, self.pad_back = extra // 2, extra - extra // 2

    def forward(self, x):
        if self.stride != 1:
            x = x[:, :, ::self.stride, ::self.stride]
        if self.pad_front or self.pad_back:
            x = F.pad(x, (0, 0, 0, 0, self.pad_front, self.pad_back))
        return x


class BlockA(nn.Module):
    """Basic block of the CIFAR ResNets with the parameter-free shortcut."""

    def __init__(self, cin: int, cout: int, stride: int = 1):
        super().__init__()
        self.conv1, self.bn1 = _conv3x3(cin, cout, stride), nn.BatchNorm2d(cout)
        self.conv2, self.bn2 = _conv3x3(cout, cout), nn.BatchNorm2d(cout)
        self.shortcut = ShortcutA(cin, cout, stride) if (stride != 1 or cin != cout) else nn.Identity()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


class ResNet32(nn.Module):
    feature_dim = 64

    def __init__(self, blocks_per_stage: int = 5):
        super().__init__()
        self.conv1, self.bn1 = _conv3x3(3, 16), nn.BatchNorm2d(16)
        stages, cin = [], 16
        for cout, stride in ((16, 1), (32, 2), (64, 2)):
            blocks = [BlockA(cin, cout, stride)] + [BlockA(cout, cout, 1) for _ in range(blocks_per_stage - 1)]
            stages.append(nn.Sequential(*blocks))
            cin = cout
        self.stage1, self.stage2, self.stage3 = stages
        self.pool = GlobalAvgPool()
        _init(self)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.stage3(self.stage2(self.stage1(x)))
        return self.pool(x)


CNNS = {"tiny_cnn": TinyCNN, "resnet18_cifar": ResNet18Cifar, "resnet32": ResNet32}
