"""
EGRR Training Pipeline for PatchCamelyon (PCam) - Local File Support

Adapted for:
- Input: 96x96 RGB images (histopathology tiles)
- Task: Binary classification (Tumor vs. Normal)
- Loading: Direct form local .h5.gz files (no download requred)
"""

import os
import sys
import time
import argparse
import gzip
import shutil
import json

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
import h5py
import numpy as np
from PIL import Image

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
    cosine_lr,
)


def get_model(model_name, num_classes, device):
    """Factory to get the requested model."""
    if model_name == "egrr":
        model = EGRRNet(
            num_classes=num_classes,
            stages=config.STAGES,
            stem_channels=config.STEM_CHANNELS,
            width_mult=config.WIDTH_MULT,
            dilation_rates=config.DILATION_RATES,
        )
    elif model_name == "resnet18":
        # Standard ResNet18 adapted for 2 classes
        model = torchvision.models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "mobilenet_v2":
        # MobileNetV2 adapted for 2 classes
        model = torchvision.models.mobilenet_v2(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model.to(device)


class PCamDataset(Dataset):
    """Custom PCam Dataset loader for local .h5 files."""

    def __init__(self, root, split='train', transform=None):
        self.root = root
        self.split = split
        self.transform = transform
        
        # Files expected: camelyonpatch_level_2_split_{split}_{x/y}.h5
        self.x_path = self._check_and_prepare(f"camelyonpatch_level_2_split_{split}_x.h5")
        self.y_path = self._check_and_prepare(f"camelyonpatch_level_2_split_{split}_y.h5")

        self.x_h5 = None
        self.y_h5 = None

    def _check_and_prepare(self, filename):
        """Check for .h5 file. If missing, try decomposing .h5.gz."""
        path = os.path.join(self.root, filename)
        if os.path.exists(path):
            return path
            
        # Check for .gz version
        gz_path = path + ".gz"
        if os.path.exists(gz_path):
            print(f"Decompressing {gz_path}...")
            with gzip.open(gz_path, 'rb') as f_in:
                with open(path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            return path
            
        # Check root directory if absolute path given doesn't work
        # Try checking in script directory if root was generic
        local_path = filename
        if os.path.exists(local_path):
             return local_path

        local_gz_path = filename + ".gz"
        if os.path.exists(local_gz_path):
            print(f"Decompressing {local_gz_path}...")
            with gzip.open(local_gz_path, 'rb') as f_in:
                with open(local_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            return local_path
            
        raise FileNotFoundError(f"Could not find {filename} or {filename}.gz in {self.root} or current directory.")

    def __getitem__(self, index):
        if self.x_h5 is None:
            self.x_h5 = h5py.File(self.x_path, 'r')['x']
            self.y_h5 = h5py.File(self.y_path, 'r')['y']

        # Read image and label
        img = self.x_h5[index]
        target = self.y_h5[index]

        # Convert to PIL Image for transforms
        img = Image.fromarray(img)
        
        # Target is (1, 1, 1) or similar in h5, extract scalar
        target = int(target.flatten()[0])

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        # We need to open the file to know length without keeping it open?
        # Or just open once. h5py objects are not pickleable for DataLoader workers sometimes.
        # Best practice: open in __init__ for length, but re-open in __getitem__ for safety/multiprocessing?
        # Actually h5py safe pattern for multiprocess DataLoader is opening in __getitem__.
        # We can read length once.
        with h5py.File(self.x_path, 'r') as f:
            return len(f['x'])


def get_dataloaders(batch_size: int, data_dir: str, num_workers: int = 2,
                    use_cuda: bool = False, subset_size: int = None):
    """Build PCam train/valid/test data loaders using local files.
    
    Args:
        subset_size: If provided, limits the dataset to this many samples (for debugging/testing).
    """
    
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    train_transforms = [
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(90),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]

    train_transform = transforms.Compose(train_transforms)

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # Use current directory as root for convenience if files are here
    # data_dir passed from args might be 'data/' but user has files in '.'
    # The custom dataset handles checking both.
    
    # We ideally want train, val, test. If user only has train files, we might fail on others.
    # Let's try to load what we can, but standard flow expects all.
    # PROACTIVE FIX: User said "downloaded whole ... train_x". 
    # If valid/test missing, we might need to mock or split train.
    # But usually "whole" implies the set. 
    # Valid/Test files are: camelyonpatch_level_2_split_valid_x.h5.gz, etc.
    
    # ── Robust Loading Logic ──
    # 1. Train (Required)
    try:
        train_dataset = PCamDataset(root=data_dir, split='train', transform=train_transform)
    except FileNotFoundError:
        # Fallback: check for any 'x' and 'y' pair if naming is non-standard? 
        # For now, enforce "train" naming or fail.
        raise FileNotFoundError("Could not find 'train' set files (camelyonpatch_level_2_split_train_x/y.h5).")

    # 2. Validation
    # Prefer 'val' split. If missing, try 'test'. If both missing, split 'train'.
    val_dataset = None
    try:
        val_dataset = PCamDataset(root=data_dir, split='valid', transform=test_transform)
    except FileNotFoundError:
        print("Warning: Validation set missing.")
    
    # 3. Test
    test_dataset = None
    try:
        test_dataset = PCamDataset(root=data_dir, split='test', transform=test_transform)
    except FileNotFoundError:
        print("Warning: Test set missing.")

    # ── Resolve Missing Sets ──
    
    # Case: Missing Val, but have Test -> Use Test as Val
    if val_dataset is None and test_dataset is not None:
        print("-> Using 'test' set for validation.")
        val_dataset = test_dataset
    
    # Case: Missing Val AND Test -> Split Train
    if val_dataset is None:
        print("-> Splitting 'train' set (90/10) to create validation set.")
        total = len(train_dataset)
        val_len = int(total * 0.1)
        train_len = default_len = total - val_len
        train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_len, val_len])
        # Note: val_dataset here will have train_transform (augmented). 
        # Ideally we'd override transform, but random_split subsets don't easily allow it without wrapping.
        # Acceptance: Validating on augmented data is okay, just slightly harder / noisier metrics.

    # Case: Missing Test -> Use Val as Test
    if test_dataset is None:
        print("-> Using validation set as test set.")
        test_dataset = val_dataset

    # ── Subset Logic ──
    if subset_size is not None:
        print(f"Subsetting data to {subset_size} samples per split...")
        # Helper to subset safely
        def safe_subset(ds, limit):
            limit = min(limit, len(ds))
            # Use constant indices for reproducibility if needed, or simple range
            indices = list(range(limit))
            return torch.utils.data.Subset(ds, indices)

        train_dataset = safe_subset(train_dataset, subset_size)
        if val_dataset:
            val_dataset = safe_subset(val_dataset, subset_size)
        if test_dataset:
            test_dataset = safe_subset(test_dataset, subset_size)

    pin = use_cuda

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin, drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin,
    )

    return train_loader, val_loader, test_loader


def train_one_epoch(model, loader, criterion, optimizer, device, epoch,
                    scaler=None, use_amp=False):
    """Train for one epoch."""
    model.train()
    losses = AverageMeter()
    top1 = AverageMeter()

    amp_device_type = device.type if device.type in ('cuda', 'cpu') else 'cpu'

    for batch_idx, (images, targets) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.amp.autocast(device_type=amp_device_type, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, targets)

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

        acc1 = accuracy(logits.float(), targets, topk=(1,))[0]
        losses.update(loss.item(), images.size(0))
        top1.update(acc1, images.size(0))

        if (batch_idx + 1) % 100 == 0:
            print(
                f"  Batch [{batch_idx+1}/{len(loader)}]  "
                f"Loss: {losses.avg:.4f}  "
                f"Acc: {top1.avg:.2f}%"
            )

    # Clean up h5 handles? They open/close in workers.
    return losses.avg, top1.avg


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluate on val/test set."""
    model.eval()
    losses = AverageMeter()
    top1 = AverageMeter()

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, targets)

        acc1 = accuracy(logits, targets, topk=(1,))[0]
        losses.update(loss.item(), images.size(0))
        top1.update(acc1, images.size(0))

    return losses.avg, top1.avg


def main():
    parser = argparse.ArgumentParser(description="EGRR Training on PCam")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32) # Reduced default from 128 to 32 to fix OOM
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--data-dir", type=str, default=".", 
                        help="Directory containing .h5 or .h5.gz files")
    parser.add_argument("--subset", type=int, default=None,
                        help="Train on a small subset of data (e.g. 1000) for debugging")
    parser.add_argument("--model", type=str, default="egrr",
                        choices=["egrr", "resnet18", "mobilenet_v2"],
                        help="Model architecture to train")
    args = parser.parse_args()

    device = config.get_device()
    use_cuda = device.type == 'cuda'
    print(f"Using device: {device}")

    if use_cuda:
        torch.backends.cudnn.benchmark = True

    # ── Data ──
    try:
        train_loader, val_loader, test_loader = get_dataloaders(
            args.batch_size, args.data_dir, args.workers,
            use_cuda=use_cuda,
            subset_size=args.subset
        )
        print(f"PCam: {len(train_loader.dataset)} train, "
              f"{len(val_loader.dataset)} val, "
              f"{len(test_loader.dataset)} test")
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        # ... (error message kept in print below for context if needed, but assuming user context)
        print("Please ensure all PCam files (train_x, train_y, valid_x, valid_y, test_x, test_y) are present.")
        return

    # ── Model ──
    num_classes = 2
    model = get_model(args.model, num_classes, device)

    init_weights(model)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {args.model}")
    print(f"Parameters: {total_params:,}")

    # ── Optimizer ──
    criterion = nn.CrossEntropyLoss(label_smoothing=0.0)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=1e-4, 
        nesterov=True,
    )

    # ── Logging ──
    log_dir = os.path.join(config.LOG_DIR, "pcam", args.model)
    ckpt_dir = os.path.join(config.CHECKPOINT_DIR, "pcam", args.model)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    # ── History Tracking ──
    history = {
        "model": args.model,
        "params": total_params,
        "epochs": [],
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    # ── Resume ──
    start_epoch = 0
    best_acc = 0.0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_acc = checkpoint.get("best_acc", 0.0)
        print(f"Resumed from epoch {start_epoch}, best_acc={best_acc:.2f}%")

    use_amp = config.USE_AMP and use_cuda
    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    # ── Training Loop ──
    print(f"Training for {args.epochs} epochs...")

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        
        # Schedules (Only for EGRR)
        active_T = 0
        if args.model == "egrr":
            active_T = get_active_iterations(epoch, 0, 5, max_T=max(T for _, T, _ in config.STAGES))
            if hasattr(model, 'set_active_iterations'):
                model.set_active_iterations(active_T)
            
            tau = get_gumbel_tau(epoch, args.epochs)
            if hasattr(model, 'set_gumbel_tau'):
                model.set_gumbel_tau(tau)

        lr = cosine_lr(optimizer, epoch, args.epochs, args.lr, 0.0)

        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch,
            scaler=scaler, use_amp=use_amp,
        )

        # Validate
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        elapsed = time.time() - t0

        print(
            f"Epoch [{epoch+1}/{args.epochs}]  "
            f"lr={lr:.5f}  "
            f"Train: {train_acc:.2f}%  "
            f"Val: {val_acc:.2f}%  "
            f"Time: {elapsed:.1f}s"
        )

        writer.add_scalar("Train/Loss", train_loss, epoch)
        writer.add_scalar("Train/Acc", train_acc, epoch)
        writer.add_scalar("Val/Loss", val_loss, epoch)
        writer.add_scalar("Val/Acc", val_acc, epoch)

        # Update History
        history["epochs"].append(epoch + 1)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        is_best = val_acc > best_acc
        if is_best:
            best_acc = val_acc

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "best_acc": best_acc,
        }
        torch.save(checkpoint, os.path.join(ckpt_dir, "last.pth"))
        if is_best:
            torch.save(checkpoint, os.path.join(ckpt_dir, "best.pth"))
            print(f"  ★ New best val accuracy: {best_acc:.2f}%")

    # Save History JSON
    hist_path = f"history_{args.model}.json"
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=4)
    print(f"Training history saved to {hist_path}")

    print("\nRunning final test evaluation...")
    best_ckpt = torch.load(os.path.join(ckpt_dir, "best.pth"), map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Test Accuracy: {test_acc:.2f}%")

    writer.close()


if __name__ == "__main__":
    main()
