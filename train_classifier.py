"""
Train a ViT-B/16 classifier on the TurCoins dataset.
Fine-tunes an ImageNet-pretrained Vision Transformer with differential learning rates.
"""

import argparse
import os
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.models as models
from datasets import load_dataset as hf_load_dataset

from train import HFImageDataset
from utils import count_parameters, set_seed


CLASSIFIER_CONFIG = {
    # Model
    "model_name": "vit_b_16",
    "num_classes": 138,
    "img_size": 224,

    # Training
    "epochs": 50,
    "batch_size": 64,
    "num_workers": 4,

    # Optimizer
    "backbone_lr": 1e-5,
    "head_lr": 1e-3,
    "weight_decay": 0.05,

    # Scheduler
    "warmup_epochs": 5,
    "min_lr": 1e-7,

    # Regularization
    "label_smoothing": 0.1,

    # AMP
    "use_amp": True,

    # Misc
    "seed": 42,
    "grad_clip": 1.0,
}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_turcoins_dataloaders(config):
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(config["img_size"], scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(180),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(config["img_size"]),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    hf_ds = hf_load_dataset("hsyntemiz/turcoins")
    train_dataset = HFImageDataset(hf_ds["train"], transform=train_transform)
    test_dataset = HFImageDataset(hf_ds["test"], transform=test_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=True,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=True,
    )
    return train_loader, test_loader


def create_vit_classifier(config):
    weights = models.ViT_B_16_Weights.IMAGENET1K_V1
    model = models.vit_b_16(weights=weights)

    model.heads = nn.Sequential(
        nn.LayerNorm(768),
        nn.Dropout(0.1),
        nn.Linear(768, config["num_classes"]),
    )
    nn.init.trunc_normal_(model.heads[2].weight, std=0.02)
    nn.init.zeros_(model.heads[2].bias)

    return model


def create_optimizer(model, config):
    head_params = list(model.heads.parameters())
    head_param_ids = {id(p) for p in head_params}
    backbone_params = [p for p in model.parameters() if id(p) not in head_param_ids]

    param_groups = [
        {"params": backbone_params, "lr": config["backbone_lr"]},
        {"params": head_params, "lr": config["head_lr"]},
    ]
    return torch.optim.AdamW(param_groups, weight_decay=config["weight_decay"])


def create_scheduler(optimizer, config, steps_per_epoch):
    warmup_steps = config["warmup_epochs"] * steps_per_epoch
    total_steps = config["epochs"] * steps_per_epoch

    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps,
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps - warmup_steps, eta_min=config["min_lr"],
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps],
    )


def save_checkpoint(path, model, optimizer, scheduler, scaler, epoch, best_acc, config):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "best_accuracy": best_acc,
        "config": config,
    }, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None, scaler=None):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    if optimizer and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    if scaler and "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])
    return ckpt


@torch.no_grad()
def evaluate(model, test_loader, criterion, device, use_amp):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in test_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)

        total_loss += loss.item() * images.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


def train_classifier(output_dir, resume, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    set_seed(config["seed"])
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Data
    print("Loading TurCoins dataset...")
    train_loader, test_loader = get_turcoins_dataloaders(config)
    print(f"Train: {len(train_loader.dataset)} images | Test: {len(test_loader.dataset)} images")

    # Model
    print("Creating ViT-B/16 classifier...")
    model = create_vit_classifier(config)
    model = model.to(device)
    print(f"Parameters: {count_parameters(model):,}")

    # Optimizer, scheduler, scaler
    optimizer = create_optimizer(model, config)
    steps_per_epoch = len(train_loader)
    scheduler = create_scheduler(optimizer, config, steps_per_epoch)
    scaler = torch.amp.GradScaler(enabled=config["use_amp"])
    criterion = nn.CrossEntropyLoss(label_smoothing=config["label_smoothing"])

    start_epoch = 0
    best_acc = 0.0

    # Resume
    if resume:
        print(f"Resuming from {resume}")
        ckpt = load_checkpoint(resume, model, optimizer, scheduler, scaler)
        start_epoch = ckpt["epoch"] + 1
        best_acc = ckpt.get("best_accuracy", 0.0)
        model = model.to(device)
        print(f"Resumed at epoch {start_epoch}, best acc: {best_acc:.4f}")

    print(f"\nStarting training for {config['epochs']} epochs...\n")

    for epoch in range(start_epoch, config["epochs"]):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        t0 = time.time()

        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=config["use_amp"]):
                logits = model(images)
                loss = criterion(logits, labels)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_loss += loss.item() * images.size(0)
            epoch_correct += (logits.argmax(1) == labels).sum().item()
            epoch_total += images.size(0)

        train_loss = epoch_loss / epoch_total
        train_acc = epoch_correct / epoch_total
        elapsed = time.time() - t0

        # Evaluate
        test_loss, test_acc = evaluate(model, test_loader, criterion, device, config["use_amp"])

        backbone_lr = optimizer.param_groups[0]["lr"]
        head_lr = optimizer.param_groups[1]["lr"]
        print(
            f"Epoch {epoch+1:3d}/{config['epochs']} | {elapsed:.1f}s | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Test Loss: {test_loss:.4f} Acc: {test_acc:.4f} | "
            f"LR: {backbone_lr:.1e}/{head_lr:.1e}"
        )

        # Save best
        if test_acc > best_acc:
            best_acc = test_acc
            save_checkpoint(
                output_path / "best_model.pt",
                model, optimizer, scheduler, scaler, epoch, best_acc, config,
            )
            print(f"  -> New best accuracy: {best_acc:.4f}")

        # Periodic checkpoint
        if (epoch + 1) % 10 == 0:
            save_checkpoint(
                output_path / f"checkpoint_epoch{epoch+1}.pt",
                model, optimizer, scheduler, scaler, epoch, best_acc, config,
            )

    # Final checkpoint
    save_checkpoint(
        output_path / "checkpoint_final.pt",
        model, optimizer, scheduler, scaler, config["epochs"] - 1, best_acc, config,
    )
    print(f"\nTraining complete. Best test accuracy: {best_acc:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Train ViT classifier on TurCoins")
    parser.add_argument("--output_dir", type=str, default="./outputs/turcoins_classifier")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--backbone_lr", type=float, default=None)
    parser.add_argument("--head_lr", type=float, default=None)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--no_amp", action="store_true")
    args = parser.parse_args()

    config = CLASSIFIER_CONFIG.copy()
    if args.epochs is not None:
        config["epochs"] = args.epochs
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if args.backbone_lr is not None:
        config["backbone_lr"] = args.backbone_lr
    if args.head_lr is not None:
        config["head_lr"] = args.head_lr
    if args.no_amp:
        config["use_amp"] = False
    config["num_workers"] = args.num_workers
    config["seed"] = args.seed

    train_classifier(output_dir=args.output_dir, resume=args.resume, config=config)


if __name__ == "__main__":
    main()
