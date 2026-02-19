"""
Entropy-Gated Dynamic Dilation

Uses local variance as a differentiable proxy for Shannon entropy.
Produces a 3-way soft gate that weights convolution outputs at
dilation rates d ∈ {1, 2, 4}.

During training:  Gumbel-Softmax relaxation for gradient flow.
During inference: hard argmax for deterministic selection.

Key equation:
    V(i,j) = AvgPool(X²) - (AvgPool(X))²
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EntropyGate(nn.Module):
    """Entropy-Gated Dynamic Dilation module.

    Computes local variance of the feature map as an entropy proxy,
    then produces soft weights for three dilation paths (d=1, d=2, d=4).

    Args:
        channels (int): Number of input channels.
        pool_size (int): Kernel size for local variance computation.
        num_dilations (int): Number of dilation paths (default 3).
        tau (float): Initial Gumbel-Softmax temperature.
    """

    def __init__(self, channels: int, pool_size: int = 3,
                 num_dilations: int = 3, tau: float = 1.0):
        super().__init__()
        self.channels = channels
        self.pool_size = pool_size
        self.num_dilations = num_dilations
        self.tau = tau

        # Average pooling for variance computation
        self.avg_pool = nn.AvgPool2d(
            kernel_size=pool_size,
            stride=1,
            padding=pool_size // 2,
            count_include_pad=False,
        )

        # Lightweight decision head: 1×1 conv → 3-way logits
        self.decision_head = nn.Sequential(
            nn.Conv2d(channels, channels // 4, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, num_dilations, kernel_size=1, bias=True),
        )

        # Global average pool to produce per-sample gate (not per-pixel)
        # This avoids branch divergence on ARM CPUs (tile-based strategy)
        self.global_pool = nn.AdaptiveAvgPool2d(1)

    def compute_local_variance(self, x: torch.Tensor) -> torch.Tensor:
        """Compute local variance: V = E[X²] - (E[X])².

        Memory-optimized: computes E[X] first, then reuses it.
        """
        ex = self.avg_pool(x)                      # E[X]
        e_x2 = self.avg_pool(x.square())           # E[X²]  (.square() avoids x*x alloc)
        variance = (e_x2 - ex.square()).clamp_(min=0.0)  # in-place clamp
        return variance

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input feature map (N, C, H, W).

        Returns:
            gate_weights: Soft gate weights (N, num_dilations, 1, 1),
                          summing to 1 along dim=1.
        """
        # Compute local variance as entropy proxy
        variance = self.compute_local_variance(x)  # (N, C, H, W)

        # Decision head: variance → dilation logits
        logits = self.decision_head(variance)       # (N, 3, H, W)
        del variance  # free memory early

        # Global average pool to tile-level decision
        logits = self.global_pool(logits)            # (N, 3, 1, 1)
        logits = logits.squeeze(-1).squeeze(-1)      # (N, 3)

        if self.training:
            # Gumbel-Softmax for differentiable sampling
            gate_weights = F.gumbel_softmax(logits, tau=self.tau, hard=False, dim=-1)
        else:
            # Hard argmax for inference efficiency
            gate_weights = F.one_hot(
                logits.argmax(dim=-1), num_classes=self.num_dilations
            ).float()

        # Reshape for broadcasting: (N, 3) → (N, 3, 1, 1)
        return gate_weights.unsqueeze(-1).unsqueeze(-1)

    def set_tau(self, tau: float):
        """Update Gumbel-Softmax temperature (for annealing)."""
        self.tau = tau

    def extra_repr(self) -> str:
        return (
            f"channels={self.channels}, pool_size={self.pool_size}, "
            f"num_dilations={self.num_dilations}, tau={self.tau}"
        )
