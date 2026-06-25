"""
Stage II: TPS-STN Geometric Rectification
- K=16 fiducial control points (as per paper Table IV)
- Thin Plate Spline interpolation, fully differentiable
- Learned jointly with recognition objective
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class TPSGridGenerator(nn.Module):
    """
    Generates sampling grid using Thin Plate Spline interpolation
    from K fiducial control points.
    """

    def __init__(self, out_h: int, out_w: int, num_fiducial: int = 16):
        super().__init__()
        self.out_h = out_h
        self.out_w = out_w
        self.F = num_fiducial

        # Build target control points on a grid
        ctrl_pts = self._build_control_points(num_fiducial)  # (F, 2)
        self.register_buffer("ctrl_pts_target", ctrl_pts)

        # Precompute inverse TPS kernel for target grid
        # This avoids recomputing per-batch for the canonical grid
        inv_delta_C = self._compute_inv_delta_C(ctrl_pts)       # (F+3, F+3)
        self.register_buffer("inv_delta_C", inv_delta_C)

        # Build the output sampling grid (H*W, 2)
        grid = self._build_output_grid(out_h, out_w)            # (H*W, 2)
        self.register_buffer("grid", grid)

        # Precompute P_hat for the output grid
        P_hat = self._compute_P_hat(ctrl_pts, grid)             # (H*W, F+3)
        self.register_buffer("P_hat", P_hat)

    def _build_control_points(self, num_fiducial: int) -> torch.Tensor:
        """Arrange F control points evenly on a unit square boundary."""
        F = num_fiducial
        half = F // 2
        ctrl = []
        # Top row: left to right
        for i in range(half):
            ctrl.append([-1.0 + 2.0 * i / (half - 1), -1.0])
        # Bottom row: left to right
        for i in range(half):
            ctrl.append([-1.0 + 2.0 * i / (half - 1), 1.0])
        return torch.tensor(ctrl, dtype=torch.float32)  # (F, 2)

    def _compute_inv_delta_C(self, ctrl: torch.Tensor) -> torch.Tensor:
        """Compute (F+3, F+3) inverse matrix for TPS."""
        F = self.F
        # Kernel matrix K: U(r) = r^2 * log(r^2)
        dist = torch.cdist(ctrl, ctrl)  # (F, F)
        K = dist ** 2 * (dist ** 2 + 1e-8).log()
        K = K + torch.eye(F) * 1e-3    # regularize

        # Build delta_C: [[K, P], [P^T, 0]]
        ones = torch.ones(F, 1)
        P = torch.cat([ones, ctrl], dim=1)  # (F, 3)

        top    = torch.cat([K, P], dim=1)                         # (F, F+3)
        bottom = torch.cat([P.t(), torch.zeros(3, 3)], dim=1)     # (3, F+3)
        delta_C = torch.cat([top, bottom], dim=0)                  # (F+3, F+3)

        return torch.inverse(delta_C)

    def _build_output_grid(self, H: int, W: int) -> torch.Tensor:
        """Build normalized output grid (H*W, 2) in [-1, 1]."""
        xs = torch.linspace(-1.0, 1.0, W)
        ys = torch.linspace(-1.0, 1.0, H)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        grid = torch.stack([gx, gy], dim=-1).reshape(-1, 2)  # (H*W, 2)
        return grid

    def _compute_P_hat(self, ctrl: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        """Compute (N, F+3) matrix for query points."""
        N = grid.shape[0]
        F = self.F
        dist = torch.cdist(grid, ctrl)  # (N, F)
        U = dist ** 2 * (dist ** 2 + 1e-8).log()
        ones = torch.ones(N, 1)
        P_hat = torch.cat([ones, grid, U], dim=1)  # (N, F+3) — wait, order is [1, x, y, U]
        # Correct order to match delta_C: [K | P] → solve for [w | a]
        P_hat = torch.cat([U, ones, grid], dim=1)  # (N, F+3)
        return P_hat

    def forward(self, source_ctrl_pts: torch.Tensor) -> torch.Tensor:
        """
        Args:
            source_ctrl_pts: (B, F, 2) predicted control points in source image
        Returns:
            sampling_grid: (B, H, W, 2) for F.grid_sample
        """
        B = source_ctrl_pts.shape[0]

        # For each image, solve TPS coefficients
        # target -> source mapping
        # coeffs = inv_delta_C @ [source_ctrl_pts; zeros_3x2]
        batch_ctrl = source_ctrl_pts  # (B, F, 2)
        zeros = torch.zeros(B, 3, 2, device=source_ctrl_pts.device)
        rhs = torch.cat([batch_ctrl, zeros], dim=1)  # (B, F+3, 2)

        # coeffs: (B, F+3, 2)
        inv_delta = self.inv_delta_C.unsqueeze(0).expand(B, -1, -1)  # (B, F+3, F+3)
        coeffs = torch.bmm(inv_delta, rhs)  # (B, F+3, 2)

        # Apply to output grid: P_hat (H*W, F+3) @ coeffs (B, F+3, 2)
        P_hat = self.P_hat.unsqueeze(0).expand(B, -1, -1)  # (B, HW, F+3)
        grid = torch.bmm(P_hat, coeffs)  # (B, HW, 2)

        return grid.reshape(B, self.out_h, self.out_w, 2)


class TPSSTN(nn.Module):
    """
    Thin Plate Spline Spatial Transformer Network.
    Stage II of HVLT: Geometric Rectification.
    
    - Localization network predicts K=16 control points
    - TPS warp is differentiable → learned end-to-end
    """

    def __init__(
        self,
        num_fiducial: int = 16,
        img_h: int = 32,
        img_w: int = 128,
    ):
        super().__init__()
        self.num_fiducial = num_fiducial
        self.img_h = img_h
        self.img_w = img_w

        # Localization network: predicts F control point (x,y) offsets
        self.localization = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                        # H/2, W/2

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                        # H/4, W/4

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 8)),              # 1 x 8 spatial

            nn.Flatten(),                              # 128 * 8 = 1024
        )

        self.fc_loc = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_fiducial * 2),         # F * 2 → (x, y) for each point
        )

        # Initialize control points to identity (uniform grid)
        ctrl_pts = self._init_ctrl_pts()
        self.fc_loc[-1].weight.data.zero_()
        self.fc_loc[-1].bias.data.copy_(ctrl_pts.view(-1))

        # TPS grid generator
        self.tps_grid = TPSGridGenerator(img_h, img_w, num_fiducial)

    def _init_ctrl_pts(self) -> torch.Tensor:
        """Initialize to canonical control point positions."""
        F = self.num_fiducial
        half = F // 2
        pts = []
        for i in range(half):
            pts.append([-1.0 + 2.0 * i / (half - 1), -1.0])
        for i in range(half):
            pts.append([-1.0 + 2.0 * i / (half - 1), 1.0])
        return torch.tensor(pts, dtype=torch.float32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) input image
        Returns:
            rectified: (B, 3, H, W) geometrically corrected image
        """
        B = x.shape[0]

        # Predict control points
        feats = self.localization(x)               # (B, 1024)
        ctrl_pts = self.fc_loc(feats)              # (B, F*2)
        ctrl_pts = torch.tanh(ctrl_pts)            # clamp to [-1, 1]
        ctrl_pts = ctrl_pts.view(B, self.num_fiducial, 2)

        # Compute sampling grid via TPS
        grid = self.tps_grid(ctrl_pts)             # (B, H, W, 2)

        # Sample from input image
        rectified = F.grid_sample(
            x, grid,
            align_corners=True,
            mode="bilinear",
            padding_mode="border",
        )
        return rectified
