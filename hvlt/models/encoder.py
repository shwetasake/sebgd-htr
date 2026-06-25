"""
Stage III: Hierarchical Vision Encoding
- Gated CNN Pre-filter using Gated Linear Units (GLUs)
- Swin-Transformer backbone: 4-stage hierarchy
  layer distribution [2, 2, 18, 2], heads [4, 8, 16, 32]
  Input size: 64x256 (resized from 32x128 for Swin compatibility)

Stage IV: Artifact Classification Gate (ACG)
- 3 conv layers + 1 FC, all GLU-gated
- BCE auxiliary loss with lambda=0.1
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm import create_model


# ─── Gated CNN Pre-filter ──────────────────────────────────────────────────────

class GatedConv2d(nn.Module):
    """Single GLU-gated convolutional layer."""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels * 2,
                              kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn   = nn.BatchNorm2d(out_channels * 2)

    def forward(self, x):
        h = self.bn(self.conv(x))
        values, gate = h.chunk(2, dim=1)
        return values * torch.sigmoid(gate)


class GatedCNNPrefilter(nn.Module):
    """
    Gated CNN front-end (Stage III).
    Suppresses non-ink channels before Swin encoder.
    Also upsamples from (32,128) → (64,256) for Swin.
    """

    def __init__(self, in_channels=3, out_channels=64):
        super().__init__()
        self.layers = nn.Sequential(
            GatedConv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            GatedConv2d(32, 48, kernel_size=3, stride=1, padding=1),
            GatedConv2d(48, out_channels, kernel_size=3, stride=1, padding=1),
        )
        self.project = nn.Conv2d(out_channels, 3, kernel_size=1)

    def forward(self, x):
        # Upsample to 64x256 so Swin patch-embed works cleanly
        # (Swin patch size is 4x4, so input must be divisible by 4*window=28 minimum)
        x = F.interpolate(x, size=(64, 256), mode="bilinear", align_corners=False)
        filtered = self.layers(x)       # (B, 64, 64, 256)
        return self.project(filtered)   # (B, 3, 64, 256)


# ─── Swin Transformer Encoder ─────────────────────────────────────────────────

class SwinEncoder(nn.Module):
    """
    Swin-Transformer backbone as per paper:
    - 4-stage hierarchy, depths [2,2,18,2], heads [4,8,16,32], C=96
    - Input: (B, 3, 64, 256)
    - Output: (B, H', W', 768) — stage 4 features
    """

    def __init__(self, pretrained=True):
        super().__init__()

        self.swin = create_model(
            "swin_small_patch4_window7_224",
            pretrained=pretrained,
            features_only=True,
            img_size=(64, 256),        # our actual input size
            strict_img_size=False,     # disable hard size assertion
        )
        self.out_channels = 768

    def forward(self, x):
        """
        Args:  x: (B, 3, 64, 256)
        Returns: (B, H', W', 768)
        """
        feats = self.swin(x)
        return feats[-1]   # last stage: (B, H', W', 768)


# ─── Artifact Classification Gate (ACG) ───────────────────────────────────────

class ACG(nn.Module):
    """
    Artifact Classification Gate (Stage IV) — Table III from paper.

    Conv1: 32 filters, 3x3/1, GLU
    Conv2: 64 filters, 3x3/2, GLU  (stride-2 downsampling)
    Conv3: 128 filters, 3x3/1, GLU
    Linear: 256, ReLU
    Output: 1, Sigmoid

    Dropout=0.3 after Conv3. BCE aux loss weight lambda=0.1.
    """

    def __init__(self, in_channels=768, dropout=0.3):
        super().__init__()

        self.input_proj = nn.Conv2d(in_channels, 32, kernel_size=1)
        self.conv1   = GatedConv2d(32,  32,  kernel_size=3, stride=1, padding=1)
        self.conv2   = GatedConv2d(32,  64,  kernel_size=3, stride=2, padding=1)
        self.conv3   = GatedConv2d(64,  128, kernel_size=3, stride=1, padding=1)
        self.dropout = nn.Dropout(dropout)
        self.pool    = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        """
        Args:    x: (B, H', W', 768) Swin output
        Returns: gate: (B, 1)
        """
        x = x.permute(0, 3, 1, 2).contiguous()  # → (B, 768, H', W')
        x = self.input_proj(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.dropout(x)
        x = self.pool(x)       # (B, 128, 1, 1)
        return self.fc(x)      # (B, 1)


# ─── Combined Encoder ──────────────────────────────────────────────────────────

class HierarchicalVisionEncoder(nn.Module):
    """
    Combines Stage III (Gated CNN + Swin) and Stage IV (ACG).
    Returns visual features and ACG gate score.
    """

    def __init__(self, pretrained=True, acg_dropout=0.3):
        super().__init__()
        self.gated_cnn = GatedCNNPrefilter(in_channels=3, out_channels=64)
        self.swin      = SwinEncoder(pretrained=pretrained)
        self.acg       = ACG(in_channels=768, dropout=acg_dropout)

    def forward(self, x):
        """
        Args:    x: (B, 3, H, W)
        Returns: feats (B, H', W', 768), gate (B, 1)
        """
        filtered = self.gated_cnn(x)         # (B, 3, 64, 256)
        feats    = self.swin(filtered)        # (B, H', W', 768)
        gate     = self.acg(feats)           # (B, 1)
        return feats, gate