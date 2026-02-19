"""
EGRR Block — Entropy-Gated Recursive Residual Block

The core building block combining all three mechanisms:
1. Symmetric Shared-Weight Expansion (1×1 pointwise)
2. Entropy-Gated Dynamic Dilation (depthwise convolution)
3. Iteration-Specific Normalization (IS-Norm)

Recursive update rule:
    h_t = h_{t-1} + Activation(IS-Norm_t(DWConv(W_shared * h_{t-1})))

The block replaces the standard Inverted Residual by "folding" depth
into the temporal dimension via weight sharing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint

from .is_norm import ISNorm
from .entropy_gate import EntropyGate
from .symmetric_conv import SymmetricConv1x1


class SharedDepthwiseConv(nn.Module):
    """Shared depthwise convolution with multiple dilation rates.

    The same weight kernel is used for all dilation rates.
    The entropy gate selects the appropriate dilation.

    Args:
        channels (int): Number of channels (groups = channels).
        kernel_size (int): Spatial kernel size (default 3).
        dilation_rates (list[int]): Available dilation rates.
    """

    def __init__(self, channels: int, kernel_size: int = 3,
                 dilation_rates: list = None):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.dilation_rates = dilation_rates or [1, 2, 4]

        # Single shared depthwise weight — reused for all dilation rates
        self.weight = nn.Parameter(
            torch.randn(channels, 1, kernel_size, kernel_size) * 0.02
        )
        self.bias = nn.Parameter(torch.zeros(channels))

    def forward(self, x: torch.Tensor, gate_weights: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input (N, C, H, W).
            gate_weights: Soft dilation weights (N, num_dilations, 1, 1).

        Returns:
            Weighted sum of dilated convolutions (N, C, H, W).

        Memory-optimized: accumulates weighted sum in-place instead of
        stacking a (N, num_dilations, C, H, W) tensor.
        At inference, skips dilations with zero weight (one-hot gate).
        """
        result = None
        for i, d in enumerate(self.dilation_rates):
            w = gate_weights[:, i:i+1]  # (N, 1, 1, 1)

            # At inference, skip dilations with zero gate weight
            if not self.training and w.sum().item() == 0:
                continue

            padding = d * (self.kernel_size // 2)
            out = F.conv2d(
                x, self.weight, self.bias,
                stride=1, padding=padding, dilation=d,
                groups=self.channels,
            )

            if result is None:
                result = out * w
            else:
                result = result + out * w

        return result


class EGRRBlock(nn.Module):
    """Entropy-Gated Recursive Residual Block.

    Applies a recursive loop of T iterations, each performing:
        h_t = h_{t-1} + ReLU6(IS-Norm_t(DWConv_gated(SymConv(h_{t-1}))))

    When in_channels ≠ out_channels, a lightweight projection handles
    the channel transition. Stride-2 downsampling uses average pooling.

    Args:
        in_channels (int): Input channel count.
        out_channels (int): Output channel count.
        num_iterations (int): Recursion depth T.
        stride (int): Spatial stride (1 or 2).
        kernel_size (int): Depthwise kernel size.
        dilation_rates (list[int]): Dilation rates for entropy gating.
        pool_size (int): Pool size for entropy computation.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_iterations: int = 4,
        stride: int = 1,
        kernel_size: int = 3,
        dilation_rates: list = None,
        pool_size: int = 3,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_iterations = num_iterations
        self.stride = stride
        self.use_projection = (in_channels != out_channels) or (stride != 1)

        # ── Channel projection (if dimensions change) ──
        if self.use_projection:
            self.projection = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
            if stride == 2:
                self.downsample = nn.AvgPool2d(kernel_size=2, stride=2)
            else:
                self.downsample = nn.Identity()
        else:
            self.projection = nn.Identity()
            self.downsample = nn.Identity()

        # ── Mechanism 1: Symmetric 1×1 Convolution (shared across T) ──
        self.sym_conv = SymmetricConv1x1(out_channels)

        # ── Mechanism 2: Entropy-Gated Depthwise Conv ──
        self.entropy_gate = EntropyGate(
            channels=out_channels,
            pool_size=pool_size,
            num_dilations=len(dilation_rates or [1, 2, 4]),
        )
        self.shared_dw_conv = SharedDepthwiseConv(
            channels=out_channels,
            kernel_size=kernel_size,
            dilation_rates=dilation_rates,
        )

        # ── Mechanism 3: Iteration-Specific Normalization ──
        self.is_norm = ISNorm(
            num_features=out_channels,
            num_iterations=num_iterations,
        )

        # ── Activation ──
        self.activation = nn.ReLU6(inplace=True)

        # Active recursion depth (for depth warm-up)
        self._active_iterations = num_iterations

        # Gradient checkpointing flag (set by EGRRNet)
        self._use_gradient_checkpointing = False

    @property
    def active_iterations(self) -> int:
        return self._active_iterations

    @active_iterations.setter
    def active_iterations(self, value: int):
        self._active_iterations = min(value, self.num_iterations)

    def _recursive_step(self, h: torch.Tensor, t: int) -> torch.Tensor:
        """Single recursive iteration (factored out for gradient checkpointing)."""
        # 1. Symmetric 1×1 conv (shared weights)
        z = self.sym_conv(h)

        # 2. Entropy-gated depthwise conv
        gate_weights = self.entropy_gate(h)
        z = self.shared_dw_conv(z, gate_weights)

        # 3. IS-Norm + Activation
        z = self.is_norm(z, t)
        z = self.activation(z)

        # 4. Residual connection
        return h + z

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (N, C_in, H, W).

        Returns:
            Output tensor (N, C_out, H', W') where H' = H/stride.
        """
        # Handle channel/spatial transition
        if self.stride == 2:
            x = self.downsample(x)

        h = self.projection(x)  # (N, C_out, H', W')

        # ── Recursive loop ──
        T = self._active_iterations
        use_checkpoint = (
            self.training
            and self._use_gradient_checkpointing
            and T > 1
        )

        for t in range(T):
            if use_checkpoint:
                # Gradient checkpointing: discard intermediates, recompute in backward
                h = grad_checkpoint(
                    self._recursive_step, h, t,
                    use_reentrant=False,
                )
            else:
                h = self._recursive_step(h, t)

        return h

    def extra_repr(self) -> str:
        return (
            f"in_channels={self.in_channels}, "
            f"out_channels={self.out_channels}, "
            f"num_iterations={self.num_iterations}, "
            f"stride={self.stride}"
        )
