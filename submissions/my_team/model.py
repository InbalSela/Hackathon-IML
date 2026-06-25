import torch
import torch.nn as nn


class ModelArchitecture(nn.Module):
    """
    Modern AlexNet-inspired CNN trained from scratch.

    Architecture overview (input: 3 x 224 x 224):

      Block 1: large 11x11 kernel to capture coarse features
               Conv -> BN -> ReLU -> MaxPool -> Dropout
               224x224 -> 27x27

      Block 2: 5x5 kernel for mid-level features
               Conv -> BN -> ReLU -> MaxPool -> Dropout
               27x27 -> 13x13

      Block 3: 3x3 kernel for fine-grained features
               Conv -> BN -> ReLU -> MaxPool -> Dropout
               13x13 -> 6x6

      Block 4: 3x3 kernel, deeper representation
               Conv -> BN -> ReLU -> MaxPool -> Dropout
               6x6 -> 3x3

      AdaptiveAvgPool -> flatten -> FC1 -> ReLU -> Dropout -> FC2

    Improvements over original AlexNet:
      - BatchNorm after every conv (AlexNet used LRN, which is obsolete)
      - Dropout 0.2 after every block (AlexNet only used it in FC layers)
      - bias=False on conv layers paired with BatchNorm (BN absorbs the bias)
    """

    def __init__(self, num_classes: int = 20):
        super().__init__()

        # --- Convolutional blocks ---

        # Block 1: 3x224x224 -> 64x27x27
        # Large 11x11 kernel with stride 4 rapidly reduces spatial size
        # while capturing large-scale patterns (edges, textures)
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # 56x56 -> 27x27
            nn.Dropout2d(p=0.2),
        )

        # Block 2: 64x27x27 -> 192x13x13
        # 5x5 kernel captures mid-level patterns (shapes, object parts)
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 192, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # 27x27 -> 13x13
            nn.Dropout2d(p=0.2),
        )

        # Block 3: 192x13x13 -> 384x6x6
        # 3x3 kernels for fine-grained feature extraction
        self.block3 = nn.Sequential(
            nn.Conv2d(192, 384, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 13x13 -> 6x6
            nn.Dropout2d(p=0.2),
        )

        # Block 4: 384x6x6 -> 256x3x3
        # Deeper representation before global pooling
        self.block4 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 6x6 -> 3x3
            nn.Dropout2d(p=0.2),
        )

        # Global average pool: collapses each feature map to a single value
        # avoids MPS divisibility issues and reduces overfitting
        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # 256x3x3 -> 256x1x1

        # --- Fully connected layers ---
        # FC1: 256 -> 1024
        self.fc1 = nn.Sequential(
            nn.Linear(256, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
        )

        # FC2: 1024 -> num_classes (outputs raw logits, no softmax)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: float tensor of shape [batch_size, 3, 224, 224]

        Returns:
            logits of shape [batch_size, num_classes]
        """
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)  # [batch, 256*4*4]
        x = self.fc1(x)
        return self.fc2(x)
