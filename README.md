# EGRR — Entropy-Gated Recursive Residual Network

A parameter-efficient deep learning architecture for **CIFAR-100** classification, targeting **>70% Top-1 accuracy** with **<0.5M parameters**.

## Core Innovations

| Mechanism | What It Does | Benefit |
|-----------|-------------|---------|
| **Symmetric Shared-Weight Expansion** | Recursive 1×1 conv with W = L + L^T | Folds depth into time, halves param storage |
| **Entropy-Gated Dynamic Dilation** | Local variance → soft gate → d ∈ {1,2,4} | Focuses compute on high-information regions |
| **Iteration-Specific Normalization** | Per-step (γ_t, β_t) affine parameters | Stabilizes recursive gradient flow |

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run tests (verify architecture correctness)
python -m pytest tests/test_model.py -v

# Train on CIFAR-100
python train.py

# Evaluate a checkpoint
python evaluate.py --checkpoint checkpoints/best.pth
```

## Architecture

```
Input (3, 32, 32)
    │
    ▼
┌─────────┐
│  Stem   │  3×3 Conv → BN → ReLU6  (32 channels)
└────┬────┘
     ▼
┌──────────────┐
│ EGRR Stage 1 │  48ch, T=3, stride=1  (32×32)
├──────────────┤
│ EGRR Stage 2 │  96ch, T=4, stride=2  (16×16)
├──────────────┤
│ EGRR Stage 3 │  96ch, T=4, stride=1  (16×16)
├──────────────┤
│ EGRR Stage 4 │  192ch, T=5, stride=2  (8×8)
├──────────────┤
│ EGRR Stage 5 │  192ch, T=5, stride=1  (8×8)
├──────────────┤
│ EGRR Stage 6 │  192ch, T=5, stride=1  (8×8)
├──────────────┤
│ EGRR Stage 7 │  384ch, T=6, stride=2  (4×4)
├──────────────┤
│ EGRR Stage 8 │  384ch, T=6, stride=1  (4×4)
└──────┬───────┘
       ▼
┌──────────┐
│   Head   │  AvgPool → Dropout → Linear(384, 100)
└──────────┘
```

**Total: ~420K parameters, virtual depth ~50-60 layers.**

## Training Features

- **Depth Warm-Up**: T=1 → full T over first 20 epochs
- **Gumbel-Softmax Annealing**: τ decays from 1.0 → 0.1
- **Cosine LR** with linear warmup
- **AutoAugment** + Cutout + Label Smoothing
- **Gradient Clipping** for recursive stability

## Project Structure

```
MinCNN/
├── config.py              # All hyperparameters
├── models/
│   ├── is_norm.py         # Iteration-Specific Normalization
│   ├── entropy_gate.py    # Entropy-Gated Dynamic Dilation
│   ├── symmetric_conv.py  # Symmetric 1×1 Conv (W = L + Lᵀ)
│   ├── egrr_block.py      # EGRR Block (all 3 mechanisms)
│   └── egrr_net.py        # Full network
├── train.py               # Training pipeline
├── evaluate.py            # Evaluation + metrics
├── utils.py               # Helpers
└── tests/
    └── test_model.py      # Architecture verification
```
