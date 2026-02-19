"""
EGRR Configuration — All hyperparameters and architecture settings.

Entropy-Gated Recursive Residual Network for CIFAR-100.
Target: >70% Top-1 accuracy with <0.5M parameters.

Model Variants:
    - "base":  ~486K params, 38 virtual layers  (original)
    - "deep":  ~500K params, 76 virtual layers  (2× recursion depth)
"""

import os

# ──────────────────────────────────────────────
# Model Variant (select one: "base" or "deep")
# ──────────────────────────────────────────────
MODEL_VARIANT = "deep"

# ──────────────────────────────────────────────
# Architecture
# ──────────────────────────────────────────────
NUM_CLASSES = 100
STEM_CHANNELS = 32

# ── EGRR-Base: ~486K params, virtual depth ≈ 38 ──
STAGES_BASE = [
    # (channels, T, stride)
    (32,  3, 1),   # Stage 1: 32×32 → 32×32
    (64,  4, 2),   # Stage 2: 32×32 → 16×16
    (64,  4, 1),   # Stage 3: 16×16
    (128, 5, 2),   # Stage 4: 16×16 → 8×8
    (128, 5, 1),   # Stage 5: 8×8
    (128, 5, 1),   # Stage 6: 8×8
    (256, 6, 2),   # Stage 7: 8×8 → 4×4
    (256, 6, 1),   # Stage 8: 4×4
]
WIDTH_MULT_BASE = 1.22

# ── EGRR-Deep: ~500K params, virtual depth ≈ 76 (2× recursion) ──
# Doubles all T values — adds only ~14K params (all IS-Norm)
# This gives ResNet-76-equivalent abstraction depth
STAGES_DEEP = [
    # (channels, T, stride)
    (32,  6,  1),  # Stage 1: 32×32 → 32×32   (T: 3→6)
    (64,  8,  2),  # Stage 2: 32×32 → 16×16   (T: 4→8)
    (64,  8,  1),  # Stage 3: 16×16            (T: 4→8)
    (128, 10, 2),  # Stage 4: 16×16 → 8×8     (T: 5→10)
    (128, 10, 1),  # Stage 5: 8×8              (T: 5→10)
    (128, 10, 1),  # Stage 6: 8×8              (T: 5→10)
    (256, 12, 2),  # Stage 7: 8×8 → 4×4       (T: 6→12)
    (256, 12, 1),  # Stage 8: 4×4              (T: 6→12)
]
WIDTH_MULT_DEEP = 1.22

# ── Resolved config (based on MODEL_VARIANT) ──
if MODEL_VARIANT == "deep":
    STAGES = STAGES_DEEP
    WIDTH_MULT = WIDTH_MULT_DEEP
else:
    STAGES = STAGES_BASE
    WIDTH_MULT = WIDTH_MULT_BASE

# ──────────────────────────────────────────────
# Entropy Gate
# ──────────────────────────────────────────────
DILATION_RATES = [1, 2, 4]
ENTROPY_POOL_SIZE = 3          # Kernel size of AvgPool for local variance
GUMBEL_TAU_START = 1.0         # Initial Gumbel-Softmax temperature
GUMBEL_TAU_END = 0.1           # Final temperature (annealed during training)

# ──────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────
BATCH_SIZE = 128
EPOCHS = 250 if MODEL_VARIANT == "deep" else 200  # Deep needs more epochs
LEARNING_RATE = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
LABEL_SMOOTHING = 0.1

# Learning rate schedule (cosine annealing)
LR_MIN = 0.0
LR_WARMUP_EPOCHS = 5

# Memory optimization
USE_AMP = True                 # Automatic Mixed Precision (CUDA only)
USE_GRADIENT_CHECKPOINTING = True  # Trade compute for memory in recursive loops

# Depth warm-up schedule
# Deep model needs longer warm-up since T values are 2× higher
DEPTH_WARMUP_START_EPOCH = 0
DEPTH_WARMUP_END_EPOCH = 35 if MODEL_VARIANT == "deep" else 20

# ──────────────────────────────────────────────
# Data Augmentation
# ──────────────────────────────────────────────
CUTOUT_LENGTH = 8              # Cutout patch size (0 to disable)
USE_AUTOAUGMENT = True         # Use CIFAR AutoAugment policy

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
DATA_DIR = os.getenv("DATA_DIR", "./data")
CHECKPOINT_DIR = os.getenv("CHECKPOINT_DIR", "./output/checkpoints")
LOG_DIR = os.getenv("LOG_DIR", "./output/logs")

# ──────────────────────────────────────────────
# Device
# ──────────────────────────────────────────────
DEVICE = "auto"  # "auto" | "cpu" | "cuda" | "mps"


def get_device():
    """Resolve device string to torch device."""
    import torch
    if DEVICE == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    return torch.device(DEVICE)


def scaled_channels(base_c):
    """Apply width multiplier and round to nearest multiple of 8."""
    return max(8, int(round(base_c * WIDTH_MULT / 8) * 8))


def get_variant_info():
    """Return a summary of the active model variant."""
    virtual_depth = sum(T for _, T, _ in STAGES)
    return {
        "variant": MODEL_VARIANT,
        "stages": len(STAGES),
        "virtual_depth": virtual_depth,
        "width_mult": WIDTH_MULT,
        "depth_warmup_end": DEPTH_WARMUP_END_EPOCH,
        "epochs": EPOCHS,
    }
