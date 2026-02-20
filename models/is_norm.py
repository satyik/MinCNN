"""
Iteration-Specific Normalization (IS-Norm)

Each recursion step t gets its own affine parameters (γ_t, β_t),
while sharing the running statistics (mean/variance) computation.
This allows the network to "shift gears" at each recursion step
without the full parameter cost of separate BN layers.

Parameter cost: 2 × T × C  (negligible)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ISNorm(nn.Module):
    """Iteration-Specific Batch Normalization.

    Maintains T sets of learnable affine parameters (gamma, beta),
    one per recursion iteration, while performing standard batch
    normalization for mean/variance computation.

    Uses F.batch_norm for fused, memory-efficient normalization.

    Args:
        num_features (int): Number of channels C.
        num_iterations (int): Maximum recursion depth T.
        eps (float): Epsilon for numerical stability.
        momentum (float): Momentum for running stats.
    """

    def __init__(self, num_features: int, num_iterations: int,
                 eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.num_features = num_features
        self.num_iterations = num_iterations
        self.eps = eps
        self.momentum = momentum

        # Use separate BatchNorm2d for each iteration to maintain separate running stats.
        # This prevents "concept drift" across iterations from ruining the running mean/var.
        # Parameter count is identical (T * 2 * C), but memory for buffers increases.
        self.bns = nn.ModuleList([
            nn.BatchNorm2d(num_features, eps=eps, momentum=momentum)
            for _ in range(num_iterations)
        ])

    def forward(self, x: torch.Tensor, t: int) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (N, C, H, W).
            t: Current recursion iteration index (0-indexed, must be < T).

        Returns:
            Normalized and affine-transformed tensor.
        """
        # Select the specific BN layer for this iteration
        return self.bns[t](x)

    def extra_repr(self) -> str:
        return (
            f"num_features={self.num_features}, "
            f"num_iterations={self.num_iterations}, "
            f"eps={self.eps}, momentum={self.momentum}"
        )
