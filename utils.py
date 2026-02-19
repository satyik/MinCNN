"""
Utility functions for EGRR training.

- Orthogonal weight initialization
- Gumbel-Softmax helpers
- AverageMeter for metrics
- Cutout augmentation
- Depth warm-up scheduler
"""

import math
import torch
import torch.nn as nn
import numpy as np


# ──────────────────────────────────────────────
# Weight Initialization
# ──────────────────────────────────────────────

def init_weights(model: nn.Module):
    """Apply orthogonal initialization to convolutions, constant to BN."""
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                    nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                    nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    # SymmetricConv1x1 handles its own orthogonal init


# ──────────────────────────────────────────────
# Depth Warm-Up Scheduler
# ──────────────────────────────────────────────

def get_active_iterations(epoch: int, warmup_start: int = 0,
                          warmup_end: int = 20,
                          max_T: int = 10) -> int:
    """Compute the active recursion depth for warm-up.

    Linearly scales from T=1 at warmup_start to T=max_T at warmup_end.
    """
    if epoch <= warmup_start:
        return 1
    elif epoch >= warmup_end:
        return max_T
    else:
        progress = (epoch - warmup_start) / (warmup_end - warmup_start)
        return max(1, int(math.ceil(progress * max_T)))


# ──────────────────────────────────────────────
# Gumbel-Softmax Temperature Annealing
# ──────────────────────────────────────────────

def get_gumbel_tau(epoch: int, total_epochs: int,
                   tau_start: float = 1.0,
                   tau_end: float = 0.1) -> float:
    """Exponential decay of Gumbel-Softmax temperature."""
    progress = min(1.0, epoch / max(1, total_epochs))
    tau = tau_start * (tau_end / tau_start) ** progress
    return max(tau_end, tau)


# ──────────────────────────────────────────────
# Training Utilities
# ──────────────────────────────────────────────

class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output: torch.Tensor, target: torch.Tensor,
             topk: tuple = (1, 5)) -> list:
    """Compute Top-k accuracy for the given predictions and targets."""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        results = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            results.append(correct_k.mul_(100.0 / batch_size).item())
        return results


# ──────────────────────────────────────────────
# Cutout Augmentation
# ──────────────────────────────────────────────

class Cutout:
    """Randomly mask out a square patch from the image.

    Args:
        length (int): Side length of the square patch.
    """

    def __init__(self, length: int):
        self.length = length

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img: Tensor image of shape (C, H, W).

        Returns:
            Image with a random square patch zeroed out.
        """
        if self.length <= 0:
            return img

        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)

        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1:y2, x1:x2] = 0.0
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        return img * mask


# ──────────────────────────────────────────────
# Learning Rate Schedule
# ──────────────────────────────────────────────

def cosine_lr(optimizer, epoch: int, total_epochs: int,
              lr_max: float, lr_min: float = 0.0,
              warmup_epochs: int = 5):
    """Cosine annealing learning rate with linear warmup."""
    if epoch < warmup_epochs:
        lr = lr_max * (epoch + 1) / warmup_epochs
    else:
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * progress))

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr
