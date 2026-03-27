"""
train.py
--------
Unified training loop for multimodal medical diagnosis.

Modes:
    --mode cnn3d    : Train 3D CNN on volumes only
    --mode bert     : Fine-tune BERT on clinical reports only
    --mode fusion   : Train full multimodal fusion (requires pretrained encoders)

Usage:
    python src/train.py --mode fusion --epochs 30 --batch-size 4
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import json
import argparse
from pathlib import Path
from tqdm import tqdm

from models.cnn3d import build_cnn3d
from models.bert_encoder import BertClinicalEncoder, MultimodalFusion, MultimodalDiagnosticPipeline

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─── Dataset ──────────────────────────────────────────────────────────────────

class MultimodalMedicalDataset(Dataset):
    """
    Dataset returning (volume, input_ids, attention_mask, labels).

    Args:
        volume_dir:  Directory with preprocessed .npy CT volumes.
        text_dir:    Directory with tokenized text tensors.
        split:       'train' | 'val' | 'test'
    """

    def __init__(self, volume_dir: str, text_dir: str, split: str = "train"):
        self.volume_dir = Path(volume_dir) / split
        self.input_ids = torch.load(os.path.join(text_dir, split, "input_ids.pt"))
        self.attention_masks = torch.load(os.path.join(text_dir, split, "attention_masks.pt"))
        self.labels = torch.load(os.path.join(text_dir, split, "labels.pt"))

        self.volume_files = sorted(self.volume_dir.glob("*.npy"))
        assert len(self.volume_files) == len(self.input_ids), \
            f"Mismatch: {len(self.volume_files)} volumes vs {len(self.input_ids)} reports"

    def __len__(self):
        return len(self.volume_files)

    def __getitem__(self, idx):
        volume = np.load(self.volume_files[idx])
        volume = torch.tensor(volume, dtype=torch.float32).unsqueeze(0)  # (1, D, H, W)
        return {
            "volume":         volume,
            "input_ids":      self.input_ids[idx],
            "attention_mask": self.attention_masks[idx],
            "labels":         self.labels[idx],
        }


# ─── Training ─────────────────────────────────────────────────────────────────

class EarlyStopping:
    def __init__(self, patience: int = 7, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best = float("inf")
        self.counter = 0

    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best - self.min_delta:
            self.best = val_loss
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience


def train_epoch(model, loader, optimizer, criterion, mode: str) -> float:
    model.train()
    losses = []

    for batch in tqdm(loader, desc="Train", leave=False):
        optimizer.zero_grad()

        if mode == "cnn3d":
            volume = batch["volume"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            out = model(volume)
            loss = criterion(out["logits"], labels)

        elif mode == "bert":
            ids = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            emb = model(ids, mask)
            loss = criterion(emb, labels)   # model includes classifier head

        else:  # fusion
            volume = batch["volume"].to(DEVICE)
            ids = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            out = model(volume, ids, mask)
            loss = criterion(out["logits"], labels)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        losses.append(loss.item())

    return float(np.mean(losses))


@torch.no_grad()
def eval_epoch(model, loader, criterion, mode: str) -> tuple[float, float]:
    model.eval()
    losses, all_preds, all_labels = [], [], []

    for batch in tqdm(loader, desc="Val", leave=False):
        if mode == "cnn3d":
            volume = batch["volume"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            out = model(volume)
            logits = out["logits"]

        elif mode == "bert":
            ids = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            logits = model(ids, mask)

        else:
            volume = batch["volume"].to(DEVICE)
            ids = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            out = model(volume, ids, mask)
            logits = out["logits"]

        loss = criterion(logits, labels)
        losses.append(loss.item())

        preds = (torch.sigmoid(logits) > 0.5).float()
        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())

    # Macro F1
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    tp = ((all_preds == 1) & (all_labels == 1)).sum(0)
    fp = ((all_preds == 1) & (all_labels == 0)).sum(0)
    fn = ((all_preds == 0) & (all_labels == 1)).sum(0)
    f1 = (2 * tp / (2 * tp + fp + fn + 1e-8)).mean().item()

    return float(np.mean(losses)), f1


def run_training(args):
    print(f"Device: {DEVICE} | Mode: {args.mode} | Epochs: {args.epochs}")

    # Build model
    if args.mode == "cnn3d":
        model = build_cnn3d(n_classes=args.n_classes, with_segmentation=False).to(DEVICE)
    elif args.mode == "bert":
        model = BertClinicalEncoder(embed_dim=args.n_classes).to(DEVICE)
    else:
        cnn3d = build_cnn3d(n_classes=args.n_classes, with_segmentation=True)
        bert_enc = BertClinicalEncoder(embed_dim=256)
        fusion = MultimodalFusion(embed_dim=256, n_classes=args.n_classes)
        model = MultimodalDiagnosticPipeline(cnn3d, bert_enc, fusion).to(DEVICE)

    # Data
    train_ds = MultimodalMedicalDataset(args.volume_dir, args.text_dir, "train")
    val_ds = MultimodalMedicalDataset(args.volume_dir, args.text_dir, "val")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.BCEWithLogitsLoss()
    early_stopper = EarlyStopping(patience=7)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    history = {"train_loss": [], "val_loss": [], "val_f1": []}
    best_f1 = 0.0

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, args.mode)
        val_loss, val_f1 = eval_epoch(model, val_loader, criterion, args.mode)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_f1"].append(val_f1)

        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"train={train_loss:.4f} | val={val_loss:.4f} | F1={val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(),
                       os.path.join(args.checkpoint_dir, f"{args.mode}_best.pt"))
            print(f"  ✅ Best F1={best_f1:.4f} — saved")

        if early_stopper(val_loss):
            print(f"Early stopping at epoch {epoch}")
            break

    os.makedirs("outputs", exist_ok=True)
    with open(f"outputs/{args.mode}_history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nBest val F1: {best_f1:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["cnn3d", "bert", "fusion"], default="fusion")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--n-classes", type=int, default=5)
    parser.add_argument("--volume-dir", type=str, default="data/processed/volumes")
    parser.add_argument("--text-dir", type=str, default="data/processed/text")
    parser.add_argument("--checkpoint-dir", type=str, default="models")
    args = parser.parse_args()
    run_training(args)
