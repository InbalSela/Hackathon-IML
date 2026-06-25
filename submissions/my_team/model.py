import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block.
    Learns a weight per channel — amplifies important channels, suppresses noise.

      1. Squeeze:    global avg pool -> one value per channel
      2. Excitation: small FC bottleneck -> 0-1 importance score per channel
      3. Scale:      multiply each channel by its score
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        bottleneck = max(channels // reduction, 4)
        self.squeeze    = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, bottleneck, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(bottleneck, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        s = self.squeeze(x).view(b, c)
        e = self.excitation(s).view(b, c, 1, 1)
        return x * e


class ResBlock(nn.Module):
    """
    Residual block with SE attention (SE-ResNet style).

    Structure:
      Conv(3x3) -> BN -> ReLU -> Conv(3x3) -> BN -> SE -> (+shortcut) -> ReLU

    The shortcut skips directly from input to output, so gradients always
    have a direct path back — this allows much deeper networks to train well.
    When stride > 1 or channels change, a 1x1 conv aligns the shortcut dimensions.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)
        self.se    = SEBlock(out_channels)

        # Shortcut: identity if dimensions match, 1x1 conv otherwise
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)            # channel attention before skip addition
        out = out + self.shortcut(x)  # residual addition
        return F.relu(out)


class ModelArchitecture(nn.Module):
    """
    SE-ResNet-18: ResNet-18 architecture with Squeeze-and-Excitation blocks.

    Spatial progression (input 3 x 224 x 224):
      Stem:   Conv(7x7, stride=2) -> BN -> ReLU -> MaxPool(3,2)  :  64 x 56 x 56
      Layer1: 2 x ResBlock(64,  64,  stride=1)                   :  64 x 56 x 56
      Layer2: 2 x ResBlock(64,  128, stride=2)                   : 128 x 28 x 28
      Layer3: 2 x ResBlock(128, 256, stride=2)                   : 256 x 14 x 14
      Layer4: 2 x ResBlock(256, 512, stride=2)                   : 512 x  7 x  7
      GlobalAvgPool -> flatten(512) -> FC(512 -> num_classes)

    Why ResNet over plain CNN:
      - Skip connections let gradients flow directly to early layers
      - Deeper network (8 conv blocks) without vanishing gradient problem
      - Consistently outperforms plain CNNs at the same depth
    """

    def __init__(self, num_classes: int = 20):
        super().__init__()

        # Stem: aggressive first downsampling 224x224 -> 56x56
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # -> 56x56
        )

        # Four residual stages — each doubles channels, halves spatial size
        self.layer1 = self._make_layer(64,  64,  num_blocks=2, stride=1)
        self.layer2 = self._make_layer(64,  128, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, num_blocks=2, stride=2)
        self.layer4 = self._make_layer(256, 512, num_blocks=2, stride=2)

        self.pool = nn.AdaptiveAvgPool2d(1)  # 512 x 7 x 7 -> 512 x 1 x 1
        self.fc   = nn.Linear(512, num_classes)

    def _make_layer(self, in_channels: int, out_channels: int,
                    num_blocks: int, stride: int) -> nn.Sequential:
        layers = [ResBlock(in_channels, out_channels, stride=stride)]
        for _ in range(1, num_blocks):
            layers.append(ResBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)
