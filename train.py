"""
EGRR Training Pipeline for CIFAR-100

Features:
- Depth warm-up: gradually increases recursion depth T
- Gumbel-Softmax temperature annealing
- Cosine LR schedule with linear warmup
- Data augmentation: AutoAugment + Cutout + RandomCrop + HFlip
- Label smoothing cross-entropy
- Checkpoint saving + best model tracking
- TensorBoard logging
"""

import os
import sys
import time
import argparse

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

# Allow running from project root
sys.path.insert(0, os.path.dirname(__file__))

import config
from models import EGRRNet
from utils import (
    init_weights,
    get_active_iterations,
    get_gumbel_tau,
    AverageMeter,
    accuracy,
    Cutout,
    cosine_lr,
)


def get_dataloaders(batch_size: int, data_dir: str, num_workers: int = 2,
                    use_cuda: bool = False):
    """Build CIFAR-100 train/test data loaders with augmentation."""

    # ── Normalization stats for CIFAR-100 ──
    mean = (0.5071, 0.4867, 0.4408)
    std = (0.2675, 0.2565, 0.2761)

    # ── Training transforms ──
    train_transforms = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
    ]

    if config.USE_AUTOAUGMENT:
        train_transforms.append(
            transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10)
        )

    train_transforms.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    if config.CUTOUT_LENGTH > 0:
        train_transforms.append(Cutout(config.CUTOUT_LENGTH))

    train_transform = transforms.Compose(train_transforms)

    # ── Test transforms ──
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # ── Datasets ──
    train_dataset = torchvision.datasets.CIFAR100(
        root=data_dir, train=True, download=True,
        transform=train_transform,
    )
    test_dataset = torchvision.datasets.CIFAR100(
        root=data_dir, train=False, download=True,
        transform=test_transform,
    )

    # pin_memory only benefits CUDA — causes MallocStackLogging warning on macOS
    pin = use_cuda

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin, drop_last=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin,
    )

    return train_loader, test_loader


def train_one_epoch(model, loader, criterion, optimizer, device, epoch,
                    scaler=None, use_amp=False):
    """Train for one epoch, return loss and accuracy."""
    model.train()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    amp_device_type = device.type if device.type in ('cuda', 'cpu') else 'cpu'

    for batch_idx, (images, targets) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # Forward with optional AMP
        with torch.amp.autocast(device_type=amp_device_type, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, targets)

        # Backward — set_to_none=True avoids allocating zero tensors
        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        # Metrics
        acc1, acc5 = accuracy(logits.float(), targets, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1, images.size(0))
        top5.update(acc5, images.size(0))

        if (batch_idx + 1) % 50 == 0:
            print(
                f"  Batch [{batch_idx+1}/{len(loader)}]  "
                f"Loss: {losses.avg:.4f}  "
                f"Top-1: {top1.avg:.2f}%  "
                f"Top-5: {top5.avg:.2f}%"
            )

    return losses.avg, top1.avg, top5.avg


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluate on test set."""
    model.eval()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, targets)

        acc1, acc5 = accuracy(logits, targets, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1, images.size(0))
        top5.update(acc5, images.size(0))

    return losses.avg, top1.avg, top5.avg


def main():
    parser = argparse.ArgumentParser(description="EGRR Training on CIFAR-100")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    # ── Device ──
    device = config.get_device()
    use_cuda = device.type == 'cuda'
    print(f"Using device: {device}")

    # Enable cuDNN benchmark for CUDA
    if use_cuda:
        torch.backends.cudnn.benchmark = True

    # ── Data ──
    os.makedirs(config.DATA_DIR, exist_ok=True)
    train_loader, test_loader = get_dataloaders(
        args.batch_size, config.DATA_DIR, args.workers,
        use_cuda=use_cuda,
    )
    print(f"CIFAR-100: {len(train_loader.dataset)} train, "
          f"{len(test_loader.dataset)} test")

    # ── Model ──
    model = EGRRNet(
        num_classes=config.NUM_CLASSES,
        stages=config.STAGES,
        stem_channels=config.STEM_CHANNELS,
        width_mult=config.WIDTH_MULT,
        dilation_rates=config.DILATION_RATES,
    ).to(device)

    init_weights(model)

    total_params = model.count_parameters()
    variant_info = config.get_variant_info()
    max_T = max(T for _, T, _ in config.STAGES)
    print(f"\n{'='*60}")
    print(f"EGRR-{variant_info['variant'].capitalize()} — "
          f"{total_params:,} params, "
          f"virtual depth={variant_info['virtual_depth']}, "
          f"max T={max_T}")
    print(f"{'='*60}")

    # Print detailed breakdown
    breakdown = model.parameter_breakdown()
    print(f"  Stem:   {breakdown['stem']:>8,}")
    for name, stage_info in breakdown["stages"].items():
        print(f"  {name}: {stage_info['subtotal']:>8,}")
    print(f"  Head:   {breakdown['head']:>8,}")
    print(f"  Total:  {breakdown['total']:>8,}")
    print(f"{'='*60}\n")

    # ── Gradient Checkpointing ──
    if config.USE_GRADIENT_CHECKPOINTING:
        model.set_gradient_checkpointing(True)
        print("Gradient checkpointing: ENABLED")

    # ── Loss, Optimizer ──
    criterion = nn.CrossEntropyLoss(label_smoothing=config.LABEL_SMOOTHING)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=config.MOMENTUM,
        weight_decay=config.WEIGHT_DECAY,
        nesterov=True,
    )

    # ── Logging ──
    os.makedirs(config.LOG_DIR, exist_ok=True)
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    writer = SummaryWriter(log_dir=config.LOG_DIR)

    # ── Resume ──
    start_epoch = 0
    best_acc = 0.0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_acc = checkpoint.get("best_acc", 0.0)
        print(f"Resumed from epoch {start_epoch}, best_acc={best_acc:.2f}%")

    # ── AMP (Automatic Mixed Precision) ──
    use_amp = config.USE_AMP and use_cuda
    scaler = torch.amp.GradScaler('cuda') if use_amp else None
    if use_amp:
        print("Mixed precision (AMP): ENABLED")
    else:
        print("Mixed precision (AMP): DISABLED (requires CUDA)")

    # ── Training Loop ──
    print(f"Training for {args.epochs} epochs...")
    print(f"Depth warm-up: epochs {config.DEPTH_WARMUP_START_EPOCH}"
          f"-{config.DEPTH_WARMUP_END_EPOCH}\n")

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        # 1. Depth warm-up: gradually increase recursion depth
        active_T = get_active_iterations(
            epoch,
            config.DEPTH_WARMUP_START_EPOCH,
            config.DEPTH_WARMUP_END_EPOCH,
            max_T=max_T,
        )
        model.set_active_iterations(active_T)

        # 2. Gumbel-Softmax temperature annealing
        tau = get_gumbel_tau(
            epoch, args.epochs,
            config.GUMBEL_TAU_START, config.GUMBEL_TAU_END,
        )
        model.set_gumbel_tau(tau)

        # 3. Learning rate schedule
        lr = cosine_lr(
            optimizer, epoch, args.epochs,
            args.lr, config.LR_MIN, config.LR_WARMUP_EPOCHS,
        )

        # 4. Train
        train_loss, train_acc1, train_acc5 = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch,
            scaler=scaler, use_amp=use_amp,
        )

        # 5. Evaluate
        test_loss, test_acc1, test_acc5 = evaluate(
            model, test_loader, criterion, device
        )

        elapsed = time.time() - t0

        # ── Logging ──
        print(
            f"Epoch [{epoch+1}/{args.epochs}]  "
            f"T={active_T}  τ={tau:.3f}  lr={lr:.5f}  "
            f"Train: {train_acc1:.2f}%  "
            f"Test: {test_acc1:.2f}% (Top-5: {test_acc5:.2f}%)  "
            f"Loss: {test_loss:.4f}  "
            f"Time: {elapsed:.1f}s"
        )

        writer.add_scalar("Train/Loss", train_loss, epoch)
        writer.add_scalar("Train/Top1", train_acc1, epoch)
        writer.add_scalar("Test/Loss", test_loss, epoch)
        writer.add_scalar("Test/Top1", test_acc1, epoch)
        writer.add_scalar("Test/Top5", test_acc5, epoch)
        writer.add_scalar("Schedule/LR", lr, epoch)
        writer.add_scalar("Schedule/ActiveT", active_T, epoch)
        writer.add_scalar("Schedule/GumbelTau", tau, epoch)

        # ── Checkpoint ──
        is_best = test_acc1 > best_acc
        if is_best:
            best_acc = test_acc1

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_acc": best_acc,
            "test_acc1": test_acc1,
            "variant": config.MODEL_VARIANT,
            "stages": config.STAGES,
            "width_mult": config.WIDTH_MULT,
        }

        torch.save(
            checkpoint,
            os.path.join(config.CHECKPOINT_DIR, "last.pth"),
        )
        if is_best:
            torch.save(
                checkpoint,
                os.path.join(config.CHECKPOINT_DIR, "best.pth"),
            )
            print(f"  ★ New best accuracy: {best_acc:.2f}%")

    writer.close()
    print(f"\nTraining complete! Best Top-1 accuracy: {best_acc:.2f}%")
    print(f"Best model saved to: {config.CHECKPOINT_DIR}/best.pth")


if __name__ == "__main__":
    main()
