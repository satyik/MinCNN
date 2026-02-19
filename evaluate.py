"""
EGRR Evaluation Script for CIFAR-100

Evaluates a trained checkpoint on the CIFAR-100 test set.
Reports:
- Top-1 and Top-5 accuracy
- Per-class accuracy breakdown
- Parameter count and FLOPs estimation
- Confusion matrix (saved as image)
"""

import os
import sys
import argparse
import json

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

import config
from models import EGRRNet
from utils import AverageMeter, accuracy


def get_test_loader(batch_size: int, data_dir: str, num_workers: int = 2,
                    use_cuda: bool = False):
    """Build CIFAR-100 test data loader."""
    mean = (0.5071, 0.4867, 0.4408)
    std = (0.2675, 0.2565, 0.2761)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    dataset = torchvision.datasets.CIFAR100(
        root=data_dir, train=False, download=True, transform=transform,
    )
    # pin_memory only benefits CUDA — causes MallocStackLogging warning on macOS
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=use_cuda,
    )
    return loader, dataset


def count_flops(model, input_size=(1, 3, 32, 32)):
    """Estimate FLOPs using a forward hook counter (rough estimate)."""
    flop_count = {"total": 0}

    def conv_hook(module, inp, out):
        # FLOPs ≈ 2 * K * K * C_in * C_out * H_out * W_out / groups
        weight = module.weight
        flops = (
            2 * weight.size(2) * weight.size(3)
            * weight.size(1) * weight.size(0)
            * out.size(2) * out.size(3)
        )
        if module.groups > 1:
            flops = flops // module.groups
        flop_count["total"] += flops

    def linear_hook(module, inp, out):
        flop_count["total"] += 2 * module.in_features * module.out_features

    hooks = []
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(linear_hook))

    device = next(model.parameters()).device
    dummy = torch.randn(*input_size).to(device)
    model.eval()
    with torch.no_grad():
        model(dummy)

    for h in hooks:
        h.remove()

    return flop_count["total"]


@torch.no_grad()
def evaluate_full(model, loader, device, num_classes=100):
    """Full evaluation with per-class accuracy and confusion matrix."""
    model.eval()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # Confusion matrix
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = model(images)
        acc1, acc5 = accuracy(logits, targets, topk=(1, 5))
        top1.update(acc1, images.size(0))
        top5.update(acc5, images.size(0))

        preds = logits.argmax(dim=1).cpu().numpy()
        tgts = targets.cpu().numpy()
        for p, t in zip(preds, tgts):
            confusion[t, p] += 1

    # Per-class accuracy
    per_class_acc = []
    for c in range(num_classes):
        total = confusion[c].sum()
        correct = confusion[c, c]
        per_class_acc.append(correct / max(total, 1) * 100.0)

    return {
        "top1": top1.avg,
        "top5": top5.avg,
        "per_class_acc": per_class_acc,
        "confusion_matrix": confusion,
    }


def main():
    parser = argparse.ArgumentParser(description="EGRR Evaluation")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint (.pth)")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--variant", type=str, default=None,
                        choices=["base", "deep"],
                        help="Model variant (auto-detected from checkpoint if saved)")
    args = parser.parse_args()

    device = config.get_device()
    print(f"Using device: {device}")

    # Load checkpoint first to detect variant
    checkpoint = torch.load(args.checkpoint, map_location=device,
                            weights_only=False)

    # Auto-detect model config from checkpoint, fallback to CLI, then config
    if "stages" in checkpoint and "width_mult" in checkpoint:
        # Checkpoint has embedded architecture info (preferred)
        stages = checkpoint["stages"]
        width_mult = checkpoint["width_mult"]
        variant_name = checkpoint.get("variant", "unknown")
        print(f"Auto-detected variant from checkpoint: {variant_name}")
    elif args.variant:
        # CLI override
        if args.variant == "base":
            stages = config.STAGES_BASE
            width_mult = config.WIDTH_MULT_BASE
        else:
            stages = config.STAGES_DEEP
            width_mult = config.WIDTH_MULT_DEEP
        variant_name = args.variant
        print(f"Using variant from --variant flag: {variant_name}")
    else:
        # Fall back to current config
        stages = config.STAGES
        width_mult = config.WIDTH_MULT
        variant_name = config.MODEL_VARIANT
        print(f"Using variant from config: {variant_name}")

    # ── Model ──
    model = EGRRNet(
        num_classes=config.NUM_CLASSES,
        stages=stages,
        stem_channels=config.STEM_CHANNELS,
        width_mult=width_mult,
        dilation_rates=config.DILATION_RATES,
    ).to(device)

    # Load weights
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")

    # ── Parameter count ──
    total_params = model.count_parameters()
    print(f"\n{'='*60}")
    print(f"EGRR Network — {total_params:,} parameters")
    print(f"{'='*60}")

    # ── FLOPs ──
    flops = count_flops(model)
    print(f"Estimated FLOPs: {flops:,} ({flops/1e6:.1f}M)")

    # ── Detailed breakdown ──
    breakdown = model.parameter_breakdown()
    print(f"\nParameter Breakdown:")
    print(f"  Stem:   {breakdown['stem']:>8,}")
    for name, stage_info in breakdown["stages"].items():
        details = ", ".join(
            f"{k}={v:,}" for k, v in stage_info.items() if k != "subtotal"
        )
        print(f"  {name}: {stage_info['subtotal']:>8,}  ({details})")
    print(f"  Head:   {breakdown['head']:>8,}")
    print(f"{'='*60}")

    # ── Data ──
    use_cuda = device.type == 'cuda'
    test_loader, test_dataset = get_test_loader(
        args.batch_size, config.DATA_DIR, args.workers,
        use_cuda=use_cuda,
    )

    # ── Evaluate ──
    results = evaluate_full(model, test_loader, device, config.NUM_CLASSES)

    print(f"\n{'='*60}")
    print(f"Results on CIFAR-100 Test Set")
    print(f"{'='*60}")
    print(f"  Top-1 Accuracy: {results['top1']:.2f}%")
    print(f"  Top-5 Accuracy: {results['top5']:.2f}%")
    print(f"  Parameters:     {total_params:,}")
    print(f"  FLOPs:          {flops/1e6:.1f}M")
    print(f"  APP Ratio:      {results['top1']/total_params*1000:.4f} (acc% per 1K params)")

    # ── Per-class accuracy stats ──
    pca = results["per_class_acc"]
    print(f"\nPer-Class Accuracy:")
    print(f"  Mean:   {np.mean(pca):.2f}%")
    print(f"  Std:    {np.std(pca):.2f}%")
    print(f"  Min:    {np.min(pca):.2f}% (class {np.argmin(pca)})")
    print(f"  Max:    {np.max(pca):.2f}% (class {np.argmax(pca)})")
    print(f"{'='*60}")

    # ── Save results ──
    results_path = os.path.join(config.CHECKPOINT_DIR, "eval_results.json")
    with open(results_path, "w") as f:
        json.dump({
            "top1": results["top1"],
            "top5": results["top5"],
            "parameters": total_params,
            "flops": flops,
            "per_class_acc": pca,
        }, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    # ── Save confusion matrix ──
    cm_path = os.path.join(config.CHECKPOINT_DIR, "confusion_matrix.npy")
    np.save(cm_path, results["confusion_matrix"])
    print(f"Confusion matrix saved to: {cm_path}")


if __name__ == "__main__":
    main()
