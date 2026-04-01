"""
Training script for the Disaster Damage Detection models.

Usage:
    python train.py --mode classification --backbone efficientnet_b0 --epochs 10
    python train.py --mode segmentation --epochs 15
"""
import os
import sys
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import IMAGE_SIZE, SEGMENTATION_SIZE, BATCH_SIZE, EPOCHS, LEARNING_RATE, CLASS_NAMES
from models.cnn_model import build_classifier
from models.unet_model import build_unet
from utils.preprocessing import get_classification_transforms, get_segmentation_transforms


# ═══════════════════════════════════════════════════════════════════════════════
# Dataset Classes
# ═══════════════════════════════════════════════════════════════════════════════

class DamageClassificationDataset(Dataset):
    """
    Expects folder structure:
        data/
          train/
            damaged/       *.png, *.jpg
            not_damaged/   *.png, *.jpg
    """

    def __init__(self, root_dir: str, transform=None):
        self.samples = []
        self.transform = transform

        for label_idx, cls_name in enumerate(["not_damaged", "damaged"]):
            cls_dir = os.path.join(root_dir, cls_name)
            if not os.path.isdir(cls_dir):
                print(f"⚠  Directory not found: {cls_dir}")
                continue
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith((".png", ".jpg", ".jpeg", ".tif")):
                    self.samples.append((os.path.join(cls_dir, fname), label_idx))

        print(f"  Loaded {len(self.samples)} images from {root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


class DamageSegmentationDataset(Dataset):
    """
    Expects folder structure:
        data/
          images/    *.png
          masks/     *.png   (binary: 0 = no damage, 255 = damaged)
    """

    def __init__(self, images_dir: str, masks_dir: str, transform=None):
        self.image_paths = sorted([
            os.path.join(images_dir, f)
            for f in os.listdir(images_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif"))
        ])
        self.mask_paths = sorted([
            os.path.join(masks_dir, f)
            for f in os.listdir(masks_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif"))
        ])
        self.transform = transform
        assert len(self.image_paths) == len(self.mask_paths), \
            f"Mismatch: {len(self.image_paths)} images vs {len(self.mask_paths)} masks"
        print(f"  Loaded {len(self.image_paths)} image-mask pairs")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = np.array(Image.open(self.image_paths[idx]).convert("RGB"))
        mask = np.array(Image.open(self.mask_paths[idx]).convert("L"))
        mask = (mask > 127).astype(np.float32)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"].unsqueeze(0)  # [1, H, W]

        return image, mask


# ═══════════════════════════════════════════════════════════════════════════════
# Synthetic Demo Dataset (for testing without real data)
# ═══════════════════════════════════════════════════════════════════════════════

class SyntheticDamageDataset(Dataset):
    """Generate random synthetic satellite-style images for demo/testing."""

    def __init__(self, num_samples: int = 200, img_size: int = IMAGE_SIZE, transform=None):
        self.num_samples = num_samples
        self.img_size = img_size
        self.transform = transform
        np.random.seed(42)
        self.labels = np.random.randint(0, 2, num_samples)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        label = self.labels[idx]
        img = np.random.randint(60, 200, (self.img_size, self.img_size, 3), dtype=np.uint8)

        if label == 1:  # damaged — add red/brown patches
            num_patches = np.random.randint(3, 8)
            for _ in range(num_patches):
                cx = np.random.randint(20, self.img_size - 20)
                cy = np.random.randint(20, self.img_size - 20)
                r = np.random.randint(10, 30)
                img[max(0, cy - r):cy + r, max(0, cx - r):cx + r, 0] = np.random.randint(150, 255)
                img[max(0, cy - r):cy + r, max(0, cx - r):cx + r, 1] = np.random.randint(40, 80)
                img[max(0, cy - r):cy + r, max(0, cx - r):cx + r, 2] = np.random.randint(30, 70)
        else:  # not damaged — add green patches
            num_patches = np.random.randint(3, 8)
            for _ in range(num_patches):
                cx = np.random.randint(20, self.img_size - 20)
                cy = np.random.randint(20, self.img_size - 20)
                r = np.random.randint(10, 30)
                img[max(0, cy - r):cy + r, max(0, cx - r):cx + r, 0] = np.random.randint(30, 70)
                img[max(0, cy - r):cy + r, max(0, cx - r):cx + r, 1] = np.random.randint(120, 200)
                img[max(0, cy - r):cy + r, max(0, cx - r):cx + r, 2] = np.random.randint(30, 70)

        pil_img = Image.fromarray(img)
        if self.transform:
            pil_img = self.transform(pil_img)
        return pil_img, label


# ═══════════════════════════════════════════════════════════════════════════════
# Training Functions
# ═══════════════════════════════════════════════════════════════════════════════

def train_classifier(args):
    """Train the CNN damage classifier."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🔧 Device: {device}")

    # Transforms
    train_tf = get_classification_transforms(train=True, img_size=args.img_size)
    val_tf = get_classification_transforms(train=False, img_size=args.img_size)

    # Dataset
    if args.data_dir and os.path.isdir(os.path.join(args.data_dir, "damaged")):
        print("📂 Loading real dataset...")
        full_ds = DamageClassificationDataset(args.data_dir, transform=train_tf)
    else:
        print("🎲 Using synthetic demo dataset (no real data found)...")
        full_ds = SyntheticDamageDataset(num_samples=args.num_samples, img_size=args.img_size, transform=train_tf)

    # Split
    val_size = int(0.2 * len(full_ds))
    train_size = len(full_ds) - val_size
    train_ds, val_ds = random_split(full_ds, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Model
    model = build_classifier(backbone=args.backbone, num_classes=len(CLASS_NAMES), freeze=True)
    model = model.to(device)
    print(f"🧠 Model: {args.backbone}")
    print(f"   Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop
    best_acc = 0.0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", ncols=80)
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct / total:.3f}")

        scheduler.step()
        train_loss = running_loss / total
        train_acc = correct / total
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)

        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, preds = outputs.max(1)
                val_correct += preds.eq(labels).sum().item()
                val_total += labels.size(0)

        val_loss /= max(val_total, 1)
        val_acc = val_correct / max(val_total, 1)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"  → Train Loss: {train_loss:.4f}  Acc: {train_acc:.3f}  |  "
              f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.3f}")

        # Save best
        if val_acc > best_acc:
            best_acc = val_acc
            os.makedirs(args.save_dir, exist_ok=True)
            save_path = os.path.join(args.save_dir, "best_classifier.pth")
            torch.save(model.state_dict(), save_path)
            print(f"  ✅ Saved best model (acc={best_acc:.3f}) → {save_path}")

    print(f"\n🏁 Training complete! Best val accuracy: {best_acc:.3f}")
    return model, history


def train_segmentation(args):
    """Train the U-Net segmentation model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🔧 Device: {device}")

    images_dir = os.path.join(args.data_dir, "images") if args.data_dir else ""
    masks_dir = os.path.join(args.data_dir, "masks") if args.data_dir else ""

    if args.data_dir and os.path.isdir(images_dir) and os.path.isdir(masks_dir):
        print("📂 Loading real segmentation dataset...")
        train_tf = get_segmentation_transforms(train=True, img_size=args.img_size)
        full_ds = DamageSegmentationDataset(images_dir, masks_dir, transform=train_tf)
    else:
        print("⚠  Segmentation requires images/ and masks/ directories.")
        print("   Please provide --data_dir pointing to a folder with images/ and masks/ subdirs.")
        print("   Exiting.")
        return None, None

    val_size = int(0.2 * len(full_ds))
    train_size = len(full_ds) - val_size
    train_ds, val_ds = random_split(full_ds, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = build_unet(num_classes=1, pretrained=True).to(device)
    print(f"🧠 U-Net Segmentation")
    print(f"   Total params: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    best_iou = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", ncols=80)
        for images, masks in pbar:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        # Validation IoU
        model.eval()
        total_iou = 0.0
        num_batches = 0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                preds = torch.sigmoid(model(images)) > 0.5
                intersection = (preds * masks).sum().float()
                union = (preds + masks).clamp(0, 1).sum().float()
                iou = (intersection / (union + 1e-8)).item()
                total_iou += iou
                num_batches += 1

        mean_iou = total_iou / max(num_batches, 1)
        print(f"  → Loss: {running_loss / len(train_loader):.4f}  |  Val IoU: {mean_iou:.3f}")

        if mean_iou > best_iou:
            best_iou = mean_iou
            os.makedirs(args.save_dir, exist_ok=True)
            save_path = os.path.join(args.save_dir, "best_unet.pth")
            torch.save(model.state_dict(), save_path)
            print(f"  ✅ Saved best model (IoU={best_iou:.3f}) → {save_path}")

    print(f"\n🏁 Training complete! Best val IoU: {best_iou:.3f}")
    return model, None


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Train Disaster Damage Detection Model")
    parser.add_argument("--mode", type=str, default="classification",
                        choices=["classification", "segmentation"],
                        help="Training mode")
    parser.add_argument("--backbone", type=str, default="efficientnet_b0",
                        help="Backbone model (classification only)")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Path to dataset directory")
    parser.add_argument("--save_dir", type=str, default="saved_models",
                        help="Path to save trained models")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--img_size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--num_samples", type=int, default=400,
                        help="Number of synthetic samples (demo mode)")
    args = parser.parse_args()

    print("=" * 60)
    print("  🛰️  Disaster Damage Detection — Training Pipeline")
    print("=" * 60)

    start = time.time()

    if args.mode == "classification":
        model, history = train_classifier(args)
    else:
        model, history = train_segmentation(args)

    elapsed = time.time() - start
    print(f"\n⏱  Total training time: {elapsed / 60:.1f} minutes")


if __name__ == "__main__":
    main()
