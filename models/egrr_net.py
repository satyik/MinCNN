"""
EGRR Network — Complete macro-architecture for CIFAR-100.

Pyramidal structure: decreasing resolution, increasing channels.
Each stage uses EGRR blocks with recursive weight sharing.

Target: >70% Top-1 accuracy with ~420K parameters.
"""

import torch
import torch.nn as nn

from .egrr_block import EGRRBlock


class EGRRNet(nn.Module):
    """Entropy-Gated Recursive Residual Network.

    Args:
        num_classes (int): Number of output classes.
        stages (list): List of (base_channels, T, stride) tuples.
        stem_channels (int): Channels for the stem convolution.
        width_mult (float): Width multiplier for channel scaling.
        dilation_rates (list[int]): Dilation rates for entropy gating.
    """

    def __init__(
        self,
        num_classes: int = 100,
        stages: list = None,
        stem_channels: int = 32,
        width_mult: float = 1.5,
        dilation_rates: list = None,
    ):
        super().__init__()
        self.num_classes = num_classes

        if dilation_rates is None:
            dilation_rates = [1, 2, 4]

        if stages is None:
            # Default: paper Section 6.1 architecture
            stages = [
                (32, 3, 1),   # Stage 1: 32×32 → 32×32
                (64, 4, 2),   # Stage 2: 32×32 → 16×16
                (64, 4, 1),   # Stage 3: 16×16 → 16×16
                (128, 5, 2),  # Stage 4: 16×16 → 8×8
                (128, 5, 1),  # Stage 5: 8×8 → 8×8
                (128, 5, 1),  # Stage 6: 8×8 → 8×8
                (256, 6, 2),  # Stage 7: 8×8 → 4×4
                (256, 6, 1),  # Stage 8: 4×4 → 4×4
            ]

        # ── Stem: standard 3×3 conv ──
        self.stem = nn.Sequential(
            nn.Conv2d(3, stem_channels, kernel_size=3, stride=1,
                      padding=1, bias=False),
            nn.BatchNorm2d(stem_channels),
            nn.ReLU6(inplace=True),
        )

        # ── EGRR Stages ──
        self.stages = nn.ModuleList()
        in_ch = stem_channels
        for base_ch, T, stride in stages:
            out_ch = self._scale_channels(base_ch, width_mult)
            block = EGRRBlock(
                in_channels=in_ch,
                out_channels=out_ch,
                num_iterations=T,
                stride=stride,
                dilation_rates=dilation_rates,
            )
            self.stages.append(block)
            in_ch = out_ch

        # ── Classification Head ──
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.1),
            nn.Linear(in_ch, num_classes),
        )

        # Store last channel count for external use
        self.last_channels = in_ch

    @staticmethod
    def _scale_channels(base_c: int, width_mult: float) -> int:
        """Apply width multiplier and round to nearest multiple of 8."""
        return max(8, int(round(base_c * width_mult / 8) * 8))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input image batch (N, 3, 32, 32).

        Returns:
            Class logits (N, num_classes).
        """
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        x = self.head(x)
        return x

    def set_active_iterations(self, max_t: int):
        """Set maximum recursion depth across all blocks (for depth warm-up).

        Args:
            max_t: Maximum allowed iterations. Each block is capped at
                   min(max_t, block.num_iterations).
        """
        for stage in self.stages:
            stage.active_iterations = max_t

    def set_gumbel_tau(self, tau: float):
        """Set Gumbel-Softmax temperature across all entropy gates.

        Args:
            tau: Temperature value (annealed during training).
        """
        for stage in self.stages:
            stage.entropy_gate.set_tau(tau)

    def set_gradient_checkpointing(self, enable: bool = True):
        """Enable/disable gradient checkpointing across all EGRR blocks.

        Args:
            enable: If True, recursive loops use checkpointing to save memory.
        """
        for stage in self.stages:
            stage._use_gradient_checkpointing = enable

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def parameter_breakdown(self) -> dict:
        """Detailed parameter count by component."""
        breakdown = {
            "stem": sum(p.numel() for p in self.stem.parameters()),
            "head": sum(p.numel() for p in self.head.parameters()),
            "stages": {},
            "total": self.count_parameters(),
        }
        for i, stage in enumerate(self.stages):
            stage_params = {
                "sym_conv": sum(p.numel() for p in stage.sym_conv.parameters()),
                "entropy_gate": sum(
                    p.numel() for p in stage.entropy_gate.parameters()
                ),
                "shared_dw_conv": sum(
                    p.numel() for p in stage.shared_dw_conv.parameters()
                ),
                "is_norm": sum(p.numel() for p in stage.is_norm.parameters()),
                "projection": sum(
                    p.numel() for p in stage.projection.parameters()
                ) if stage.use_projection else 0,
            }
            stage_params["subtotal"] = sum(stage_params.values())
            breakdown["stages"][f"stage_{i + 1}"] = stage_params
        return breakdown

    def extra_repr(self) -> str:
        return (
            f"num_classes={self.num_classes}, "
            f"total_params={self.count_parameters():,}"
        )
