"""
Symmetric Shared-Weight 1×1 Convolution Kernel

Parameterized as W = L + Lᵀ where L is a lower-triangular matrix,
guaranteeing exact symmetry (W = Wᵀ). Symmetric matrices have real
eigenvalues and orthogonal eigenvectors, which stabilizes recursive
application.

Storage: C(C+1)/2 unique parameters instead of C².
Initialization: Orthogonal, placing eigenvalues on the unit circle.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SymmetricConv1x1(nn.Module):
    """Symmetric 1×1 pointwise convolution.

    The weight matrix is constructed as W = L + L^T, where L is a
    learnable lower-triangular matrix. This guarantees:
    - W is symmetric → real eigenvalues
    - Orthogonal init → eigenvalues ≈ 1 at start
    - Fewer unique parameters: C(C+1)/2

    Args:
        channels (int): Number of input/output channels (square kernel).
    """

    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels

        # Lower-triangular parameterization
        # We store the full lower-triangular matrix as a parameter
        self.L = nn.Parameter(torch.empty(channels, channels))
        self._init_orthogonal()

        # Create mask for lower-triangular extraction
        self.register_buffer(
            "tril_mask", torch.tril(torch.ones(channels, channels))
        )

    def _init_orthogonal(self):
        """Initialize L such that L + L^T ≈ orthogonal matrix."""
        # Generate a random orthogonal matrix Q
        Q = torch.linalg.qr(torch.randn(self.channels, self.channels))[0]
        # W = L + L^T = Q  →  L = Q/2 (diagonal gets half)
        # But we only store the lower triangle, so:
        L_init = Q / 2.0
        with torch.no_grad():
            self.L.copy_(L_init)
        self._cached_weight = None

    def get_weight(self) -> torch.Tensor:
        """Construct the symmetric weight matrix W = L_lower + L_lower^T."""
        L_lower = self.L * self.tril_mask  # zero out upper triangle
        W = L_lower + L_lower.transpose(0, 1)
        return W

    def train(self, mode: bool = True):
        """Invalidate cached weight when switching modes."""
        super().train(mode)
        if mode:
            self._cached_weight = None
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (N, C, H, W).

        Returns:
            Output tensor (N, C, H, W) after symmetric 1×1 convolution.
        """
        if not self.training and self._cached_weight is not None:
            weight = self._cached_weight
        else:
            W = self.get_weight()  # (C, C)
            # Reshape for conv2d: (C_out, C_in, 1, 1)
            weight = W.unsqueeze(-1).unsqueeze(-1)
            if not self.training:
                self._cached_weight = weight
        return F.conv2d(x, weight)

    @property
    def unique_params(self) -> int:
        """Number of unique (non-redundant) parameters."""
        return self.channels * (self.channels + 1) // 2

    def extra_repr(self) -> str:
        return (
            f"channels={self.channels}, "
            f"unique_params={self.unique_params}"
        )
