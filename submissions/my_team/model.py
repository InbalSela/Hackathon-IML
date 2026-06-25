import torch
import torch.nn as nn


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block.

    Learns a weight for each channel:
      1. Squeeze:    global avg pool collapses spatial dims -> one value per channel
      2. Excitation: small FC bottleneck outputs a 0-1 importance score per channel
      3. Scale:      multiply each channel by its score
                     -> important channels amplified, noisy channels suppressed
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        bottleneck = max(channels // reduction, 4)  # never go below 4 units
        self.squeeze    = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, bottleneck, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(bottleneck, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        s = self.squeeze(x).view(b, c)          # [b, c]
        e = self.excitation(s).view(b, c, 1, 1) # [b, c, 1, 1]
        return x * e                             # channel-wise scaling


class ModelArchitecture(nn.Module):
    """
    6-block AlexNet-inspired CNN with Squeeze-and-Excitation channel attention.

    Spatial progression (input 3 x 224 x 224):
      Block 1: Conv(11x11, stride=4) -> BN -> ReLU -> MaxPool(3,2) -> SE -> Dropout  :  64 x 27 x 27
      Block 2: Conv(5x5)             -> BN -> ReLU -> MaxPool(3,2) -> SE -> Dropout  : 192 x 13 x 13
      Block 3: Conv(3x3)             -> BN -> ReLU                -> SE -> Dropout  : 384 x 13 x 13
      Block 4: Conv(3x3)             -> BN -> ReLU -> MaxPool(2,2) -> SE -> Dropout  : 384 x  6 x  6
      Block 5: Conv(3x3)             -> BN -> ReLU                -> SE -> Dropout  : 256 x  6 x  6
      Block 6: Conv(3x3)             -> BN -> ReLU -> MaxPool(2,2) -> SE -> Dropout  : 256 x  3 x  3
      GlobalAvgPool -> flatten(256) -> FC1(1024) -> ReLU -> Dropout -> FC2(num_classes)
    """

    def __init__(self, num_classes: int = 20, dropout: float = 0.5):
        super().__init__()

        # Block 1: 224x224 -> 27x27
        # Large 11x11 kernel captures coarse textures and color patterns
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            SEBlock(64),
            nn.Dropout2d(p=dropout),
        )

        # Block 2: 27x27 -> 13x13
        # 5x5 kernel captures mid-level patterns (shapes, object parts)
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 192, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            SEBlock(192),
            nn.Dropout2d(p=dropout),
        )

        # Block 3: 13x13 -> 13x13  (no pooling — adds depth without shrinking)
        self.block3 = nn.Sequential(
            nn.Conv2d(192, 384, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            SEBlock(384),
            nn.Dropout2d(p=dropout),
        )

        # Block 4: 13x13 -> 6x6
        self.block4 = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            SEBlock(384),
            nn.Dropout2d(p=dropout),
        )

        # Block 5: 6x6 -> 6x6  (no pooling — adds depth without shrinking)
        self.block5 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            SEBlock(256),
            nn.Dropout2d(p=dropout),
        )

        # Block 6: 6x6 -> 3x3
        self.block6 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            SEBlock(256),
            nn.Dropout2d(p=dropout),
        )

        # Global average pool: collapses 3x3 -> 1x1, works on any input size
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Fully connected classifier
        self.fc1 = nn.Sequential(
            nn.Linear(256, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
        )
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: float tensor [batch_size, 3, 224, 224]
        Returns:
            logits [batch_size, num_classes]
        """
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return self.fc2(x)
