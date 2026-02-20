"""
Knowledge Distillation Training for PCam
==========================================

Two-phase training pipeline:

PHASE 1 — Train the TEACHER model
  A ResNet-50 (ImageNet-pretrained) is fine-tuned on PCam.
  The teacher is saved to disk after training.

PHASE 2 — Train the STUDENT with KD
  The tiny EGRR model is trained using a combined loss:

      L = α * CE(student_logits, y_hard)          ← hard label loss
        + (1-α) * T² * KL(σ(s/T), σ(t/T))        ← soft label loss (KD)

  where:
    s = student logits
    t = teacher logits (frozen, no_grad)
    T = temperature  (softens the probability distribution)
    α = hard-label weight (0 = pure KD, 1 = pure CE)

Usage:
------
  # Phase 1: Train teacher (saves to output/checkpoints/pcam/teacher/best.pth)
  python train_pcam_kd.py --phase teacher --epochs 20

  # Phase 2: Train student with KD (requires teacher checkpoint)
  python train_pcam_kd.py --phase student --epochs 35

  # Both phases sequentially:
  python train_pcam_kd.py --phase both --teacher-epochs 20 --student-epochs 35

References:
-----------
  Hinton et al. (2015), "Distilling the Knowledge in a Neural Network"
  https://arxiv.org/abs/1503.02531
"""

import os
import sys
import time
import argparse
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
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
    cosine_lr,
)

# Reuse PCamDataset and get_dataloaders from train_pcam.py
from train_pcam import PCamDataset, get_dataloaders


# ──────────────────────────────────────────────────────────────────────────────
# Models
# ──────────────────────────────────────────────────────────────────────────────

def get_teacher(num_classes: int, device: torch.device) -> nn.Module:
    """Build teacher: ResNet-50 with ImageNet pretraining for faster convergence."""
    model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
    # Replace classification head
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(device)


def get_student(num_classes: int, device: torch.device) -> nn.Module:
    """Build student: tiny EGRR model from config."""
    model = EGRRNet(
        num_classes=num_classes,
        stages=config.STAGES,
        stem_channels=config.STEM_CHANNELS,
        width_mult=config.WIDTH_MULT,
        dilation_rates=config.DILATION_RATES,
    )
    return model.to(device)


# ──────────────────────────────────────────────────────────────────────────────
# Knowledge Distillation Loss
# ──────────────────────────────────────────────────────────────────────────────

class KDLoss(nn.Module):
    """
    Hinton Knowledge Distillation Loss.

    Combines:
      - Hard-label cross-entropy with true labels
      - Soft-label KL divergence with teacher's temperature-scaled logits

    Args:
        temperature (float): Temperature T for softening distributions.
            Higher T → softer probability distributions → more information 
            transferred from teacher to student. Typical range: 2–8.
        alpha (float): Weight given to hard-label CE loss.
            (1 - alpha) is the weight for the KD soft-label loss.
            alpha=0.0 → pure KD, alpha=1.0 → pure CE, alpha=0.5 → balanced.
    """

    def __init__(self, temperature: float = 4.0, alpha: float = 0.5):
        super().__init__()
        self.T = temperature
        self.alpha = alpha
        self.ce = nn.CrossEntropyLoss()

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            student_logits: Raw logits from student  (N, C)
            teacher_logits: Raw logits from teacher  (N, C)
            targets:        Hard integer labels       (N,)

        Returns:
            total_loss, hard_loss, soft_loss
        """
        # -- Hard label loss (standard cross-entropy)
        hard_loss = self.ce(student_logits, targets)

        # -- Soft label loss (KL divergence at temperature T)
        # KL(student_soft || teacher_soft)
        # Multiply by T² to preserve gradient magnitudes (Hinton et al.)
        soft_student = F.log_softmax(student_logits / self.T, dim=-1)
        soft_teacher = F.softmax(teacher_logits / self.T, dim=-1)
        soft_loss = F.kl_div(soft_student, soft_teacher, reduction="batchmean") * (self.T ** 2)

        total_loss = self.alpha * hard_loss + (1.0 - self.alpha) * soft_loss
        return total_loss, hard_loss, soft_loss


# ──────────────────────────────────────────────────────────────────────────────
# Train / Eval Functions
# ──────────────────────────────────────────────────────────────────────────────

def train_one_epoch_teacher(model, loader, criterion, optimizer, device, epoch, scaler=None, use_amp=False):
    """Standard one-epoch training for the teacher (no KD)."""
    model.train()
    losses = AverageMeter()
    accs = AverageMeter()

    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)

        with torch.amp.autocast('cuda', enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, targets)

        optimizer.zero_grad(set_to_none=True)
        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        batch_acc = accuracy(logits, targets, topk=(1,))[0]
        losses.update(loss.item(), images.size(0))
        accs.update(batch_acc.item(), images.size(0))

    return losses.avg, accs.avg


def train_one_epoch_student(model, teacher, loader, kd_criterion, optimizer, device, epoch,
                             scaler=None, use_amp=False):
    """One-epoch training for the student using KD loss."""
    model.train()
    teacher.eval()  # Teacher always frozen in eval mode

    total_losses = AverageMeter()
    hard_losses = AverageMeter()
    soft_losses = AverageMeter()
    accs = AverageMeter()

    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)

        with torch.no_grad():
            # Teacher logits — never update teacher weights
            with torch.amp.autocast('cuda', enabled=use_amp):
                teacher_logits = teacher(images)

        with torch.amp.autocast('cuda', enabled=use_amp):
            student_logits = model(images)
            total_loss, hard_loss, soft_loss = kd_criterion(student_logits, teacher_logits, targets)

        optimizer.zero_grad(set_to_none=True)
        if scaler:
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        batch_acc = accuracy(student_logits, targets, topk=(1,))[0]
        total_losses.update(total_loss.item(), images.size(0))
        hard_losses.update(hard_loss.item(), images.size(0))
        soft_losses.update(soft_loss.item(), images.size(0))
        accs.update(batch_acc.item(), images.size(0))

    return total_losses.avg, hard_losses.avg, soft_losses.avg, accs.avg


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Standard evaluation loop."""
    model.eval()
    losses = AverageMeter()
    accs = AverageMeter()

    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        logits = model(images)
        loss = criterion(logits, targets)
        batch_acc = accuracy(logits, targets, topk=(1,))[0]
        losses.update(loss.item(), images.size(0))
        accs.update(batch_acc.item(), images.size(0))

    return losses.avg, accs.avg


# ──────────────────────────────────────────────────────────────────────────────
# Phase 1: Train Teacher
# ──────────────────────────────────────────────────────────────────────────────

def run_teacher_phase(args, device):
    """Fine-tune ResNet-50 teacher on PCam. Saves best checkpoint."""
    print("\n" + "="*60)
    print("PHASE 1: Training Teacher (ResNet-50 pretrained)")
    print("="*60)

    use_cuda = device.type == 'cuda'
    train_loader, val_loader, _ = get_dataloaders(
        args.batch_size, args.data_dir, args.workers, use_cuda=use_cuda, subset_size=args.subset
    )

    teacher = get_teacher(num_classes=2, device=device)
    total_params = sum(p.numel() for p in teacher.parameters())
    print(f"Teacher (ResNet-50): {total_params:,} parameters")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    # Use a smaller LR for pretrained weights
    optimizer = torch.optim.SGD(
        teacher.parameters(), lr=args.teacher_lr,
        momentum=0.9, weight_decay=1e-4, nesterov=True,
    )

    ckpt_dir = os.path.join(config.CHECKPOINT_DIR, "pcam", "teacher")
    log_dir  = os.path.join(config.LOG_DIR, "pcam", "teacher")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    use_amp = config.USE_AMP and use_cuda
    scaler  = torch.amp.GradScaler('cuda') if use_amp else None
    best_acc = 0.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(args.teacher_epochs):
        lr = cosine_lr(optimizer, epoch, args.teacher_epochs, args.teacher_lr,
                       lr_min=1e-5, warmup_epochs=3)

        t0 = time.time()
        train_loss, train_acc = train_one_epoch_teacher(
            teacher, train_loader, criterion, optimizer, device, epoch,
            scaler=scaler, use_amp=use_amp,
        )
        val_loss, val_acc = evaluate(teacher, val_loader, criterion, device)
        elapsed = time.time() - t0

        print(f"[T] Epoch {epoch+1:02d}/{args.teacher_epochs}  "
              f"lr={lr:.5f}  "
              f"train_loss={train_loss:.4f}  train_acc={train_acc:.1f}%  "
              f"val_loss={val_loss:.4f}  val_acc={val_acc:.1f}%  "
              f"({elapsed:.0f}s)")

        writer.add_scalars("Loss", {"train": train_loss, "val": val_loss}, epoch)
        writer.add_scalars("Acc",  {"train": train_acc,  "val": val_acc},  epoch)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            ckpt_path = os.path.join(ckpt_dir, "best.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": teacher.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_acc": best_acc,
            }, ckpt_path)
            print(f"  ★ New best teacher: {best_acc:.2f}% → saved to {ckpt_path}")

    writer.close()
    # Save history
    history_path = os.path.join(log_dir, "history_teacher.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n✓ Teacher training complete. Best val acc: {best_acc:.2f}%")
    return os.path.join(ckpt_dir, "best.pth")


# ──────────────────────────────────────────────────────────────────────────────
# Phase 2: Train Student with KD
# ──────────────────────────────────────────────────────────────────────────────

def run_student_phase(args, device, teacher_ckpt_path=None):
    """Train EGRR student using knowledge distillation from saved teacher."""
    print("\n" + "="*60)
    print("PHASE 2: Training Student (EGRR) with Knowledge Distillation")
    print(f"         Temperature T={args.temperature}, Alpha={args.alpha}")
    print("="*60)

    # -- Resolve teacher checkpoint
    if teacher_ckpt_path is None:
        teacher_ckpt_path = args.teacher_ckpt
    if teacher_ckpt_path is None:
        # Default path if not supplied
        teacher_ckpt_path = os.path.join(config.CHECKPOINT_DIR, "pcam", "teacher", "best.pth")

    if not os.path.exists(teacher_ckpt_path):
        raise FileNotFoundError(
            f"Teacher checkpoint not found at: {teacher_ckpt_path}\n"
            "Run Phase 1 first:  python train_pcam_kd.py --phase teacher"
        )

    # -- Load teacher (frozen)
    teacher = get_teacher(num_classes=2, device=device)
    ckpt = torch.load(teacher_ckpt_path, map_location=device)
    teacher.load_state_dict(ckpt["model_state_dict"])
    teacher.eval()
    # Freeze all teacher parameters — no gradients needed
    for p in teacher.parameters():
        p.requires_grad_(False)
    teacher_val_acc = ckpt.get("best_acc", "?")
    total_teacher_params = sum(p.numel() for p in teacher.parameters())
    print(f"✓ Teacher loaded from {teacher_ckpt_path}")
    print(f"  Teacher best val acc: {teacher_val_acc:.2f}%  ({total_teacher_params:,} params)")

    use_cuda = device.type == 'cuda'
    train_loader, val_loader, test_loader = get_dataloaders(
        args.batch_size, args.data_dir, args.workers, use_cuda=use_cuda, subset_size=args.subset
    )

    # -- Student
    student = get_student(num_classes=2, device=device)
    init_weights(student)
    total_student_params = sum(p.numel() for p in student.parameters())
    print(f"✓ Student (EGRR): {total_student_params:,} parameters")
    compression = total_teacher_params / total_student_params
    print(f"  Compression ratio: {compression:.1f}×")

    # -- Loss and optimizer
    kd_criterion = KDLoss(temperature=args.temperature, alpha=args.alpha)
    ce_criterion  = nn.CrossEntropyLoss()   # for eval metric only

    optimizer = torch.optim.SGD(
        student.parameters(), lr=args.lr,
        momentum=0.9, weight_decay=1e-4, nesterov=True,
    )

    ckpt_dir = os.path.join(config.CHECKPOINT_DIR, "pcam", "student_kd")
    log_dir  = os.path.join(config.LOG_DIR, "pcam", "student_kd")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    use_amp = config.USE_AMP and use_cuda
    scaler  = torch.amp.GradScaler('cuda') if use_amp else None
    best_acc = 0.0
    history = {
        "model": "egrr_kd",
        "teacher": "resnet50",
        "temperature": args.temperature,
        "alpha": args.alpha,
        "epochs": [],
        "train_loss": [], "train_acc": [],
        "val_loss":   [], "val_acc":   [],
        "hard_loss":  [], "soft_loss": [],
    }

    max_T = max(T for _, T, _ in config.STAGES)

    for epoch in range(args.student_epochs):
        t0 = time.time()

        # -- EGRR-specific schedule
        active_T = get_active_iterations(epoch, 0, 5, max_T=max_T)
        student.set_active_iterations(active_T)
        tau = get_gumbel_tau(epoch, args.student_epochs)
        student.set_gumbel_tau(tau)

        lr = cosine_lr(optimizer, epoch, args.student_epochs, args.lr,
                       lr_min=0.0, warmup_epochs=5)

        total_loss, hard_loss, soft_loss, train_acc = train_one_epoch_student(
            student, teacher, train_loader, kd_criterion, optimizer, device, epoch,
            scaler=scaler, use_amp=use_amp,
        )
        val_loss, val_acc = evaluate(student, val_loader, ce_criterion, device)
        elapsed = time.time() - t0

        print(f"[S] Epoch {epoch+1:02d}/{args.student_epochs}  "
              f"lr={lr:.5f}  "
              f"total={total_loss:.4f}  (hard={hard_loss:.4f} soft={soft_loss:.4f})  "
              f"train_acc={train_acc:.1f}%  "
              f"val_loss={val_loss:.4f}  val_acc={val_acc:.1f}%  "
              f"τ={tau:.3f}  T={active_T}  ({elapsed:.0f}s)")

        writer.add_scalars("Loss/student",  {"total": total_loss, "hard": hard_loss, "soft": soft_loss}, epoch)
        writer.add_scalars("Acc/student",   {"train": train_acc, "val": val_acc}, epoch)
        writer.add_scalar("LR/student",     lr, epoch)
        writer.add_scalar("GumbelTau",      tau, epoch)
        writer.add_scalar("ActiveT",        active_T, epoch)

        history["epochs"].append(epoch + 1)
        history["train_loss"].append(total_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["hard_loss"].append(hard_loss)
        history["soft_loss"].append(soft_loss)

        if val_acc > best_acc:
            best_acc = val_acc
            ckpt_path = os.path.join(ckpt_dir, "best.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": student.state_dict(),
                "best_acc": best_acc,
                "kd_config": {
                    "temperature": args.temperature,
                    "alpha": args.alpha,
                    "teacher_ckpt": teacher_ckpt_path,
                },
            }, ckpt_path)
            print(f"  ★ New best student: {best_acc:.2f}% → saved to {ckpt_path}")

    # -- Final test evaluation
    print("\nRunning final test evaluation...")
    test_loss, test_acc = evaluate(student, test_loader, ce_criterion, device)
    print(f"Test Accuracy: {test_acc:.2f}%  |  Test Loss: {test_loss:.4f}")

    writer.close()

    # Save history
    history_path = os.path.join(log_dir, "history_student_kd.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n✓ KD training complete.")
    print(f"  Teacher val acc:  {teacher_val_acc:.2f}%")
    print(f"  Student best val: {best_acc:.2f}%")
    print(f"  Student test acc: {test_acc:.2f}%")


# ──────────────────────────────────────────────────────────────────────────────
# Main Entry Point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Knowledge Distillation Training for PCam (Teacher=ResNet-50, Student=EGRR)"
    )

    # Phase control
    parser.add_argument(
        "--phase", type=str, default="both",
        choices=["teacher", "student", "both"],
        help=(
            "'teacher': train ResNet-50 teacher only. "
            "'student': train EGRR student with KD (requires teacher checkpoint). "
            "'both': run Phase 1 then Phase 2 sequentially."
        )
    )

    # Shared args
    parser.add_argument("--data-dir",   type=str, default=config.DATA_DIR)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers",    type=int, default=2)
    parser.add_argument("--subset",     type=int, default=None,
                        help="Limit dataset size for debugging")

    # Teacher-specific
    parser.add_argument("--teacher-epochs", type=int, default=20,
                        help="Number of epochs to train the teacher")
    parser.add_argument("--teacher-lr",     type=float, default=0.005,
                        help="Peak LR for teacher (lower than student — pretrained weights)")

    # Student-specific
    parser.add_argument("--student-epochs", type=int, default=35,
                        help="Number of epochs to train the student")
    parser.add_argument("--lr",             type=float, default=0.01,
                        help="Peak LR for student")
    parser.add_argument("--temperature",    type=float, default=4.0,
                        help="KD temperature T (2–8 typical). Higher = softer distributions.")
    parser.add_argument("--alpha",          type=float, default=0.5,
                        help="Weight for hard-label CE loss. (1-alpha) weight for KD soft loss.")
    parser.add_argument("--teacher-ckpt",   type=str, default=None,
                        help="Path to pre-trained teacher checkpoint (skip Phase 1).")

    args = parser.parse_args()

    device = config.get_device()
    use_cuda = device.type == 'cuda'
    print(f"Using device: {device}")
    if use_cuda:
        torch.backends.cudnn.benchmark = True

    teacher_ckpt = None

    if args.phase in ("teacher", "both"):
        teacher_ckpt = run_teacher_phase(args, device)

    if args.phase in ("student", "both"):
        run_student_phase(args, device, teacher_ckpt_path=teacher_ckpt)


if __name__ == "__main__":
    main()
