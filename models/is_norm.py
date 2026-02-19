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

        # Running statistics (shared across all iterations)
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))
        self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))

        # Per-iteration affine parameters: γ_t and β_t
        # Shape: (T, C) — each row is one iteration's parameters
        self.gamma = nn.Parameter(torch.ones(num_iterations, num_features))
        self.beta = nn.Parameter(torch.zeros(num_iterations, num_features))

    def forward(self, x: torch.Tensor, t: int) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (N, C, H, W).
            t: Current recursion iteration index (0-indexed, must be < T).

        Returns:
            Normalized and affine-transformed tensor of same shape.
        """
        assert 0 <= t < self.num_iterations, (
            f"Iteration index {t} out of range [0, {self.num_iterations})"
        )

        if self.training:
            self.num_batches_tracked += 1

        # Use fused F.batch_norm — handles mean/var computation in C++
        # Pass iteration-specific affine params directly
        return F.batch_norm(
            x,
            self.running_mean,
            self.running_var,
            weight=self.gamma[t],     # γ_t
            bias=self.beta[t],        # β_t
            training=self.training,
            momentum=self.momentum,
            eps=self.eps,
        )

    def extra_repr(self) -> str:
        return (
            f"num_features={self.num_features}, "
            f"num_iterations={self.num_iterations}, "
            f"eps={self.eps}, momentum={self.momentum}"
        )
