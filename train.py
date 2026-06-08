"""
train.py
--------
Training loop with:
  • Cross-entropy loss + label smoothing
  • OneCycleLR scheduler
  • Early stopping
  • Best-model checkpointing
  • Per-epoch metrics logging to CSV
"""

import os
import time
import csv
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from sklearn.metrics import accuracy_score, f1_score

from dataset import get_dataloaders
from model   import build_model

# ── Config dataclass (plain dict is fine too) ─────────────────────────────────

DEFAULT_CONFIG = {
    # Paths
    "dataset_root":     "dataset",       # ← change to your TESS root folder
    "checkpoint_dir":   "checkpoints",
    "log_csv":          "training_log.csv",

    # Model
    "model_type":       "cnn_lstm",      # "cnn_lstm" or "cnn"
    "lstm_hidden":      256,
    "lstm_layers":      2,
    "dropout":          0.4,

    # Training
    "epochs":           60,
    "batch_size":       32,
    "lr":               3e-4,
    "weight_decay":     1e-4,
    "label_smoothing":  0.1,
    "patience":         10,              # early-stop patience

    # Data
    "val_size":         0.15,
    "test_size":        0.15,
    "num_workers":      0,
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_checkpoint(state: dict, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: str, model: nn.Module, optimizer=None, scheduler=None):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    if optimizer  and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    if scheduler  and "scheduler_state" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state"])
    return ckpt.get("epoch", 0), ckpt.get("best_val_acc", 0.0)


# ── One epoch ─────────────────────────────────────────────────────────────────

def run_epoch(model, loader, criterion, optimizer, scheduler, device, train=True):
    model.train(train)
    total_loss, all_preds, all_labels = 0.0, [], []

    for X, y in loader:
        X, y = X.to(device), y.to(device)

        with torch.set_grad_enabled(train):
            logits = model(X)
            loss   = criterion(logits, y)

        if train:
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

        total_loss  += loss.item() * len(y)
        preds        = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(y.cpu().numpy())

    n       = len(all_labels)
    avg_loss = total_loss / n
    acc      = accuracy_score(all_labels, all_preds)
    f1       = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    return avg_loss, acc, f1


# ── Main training function ─────────────────────────────────────────────────────

def train(cfg: dict = None):
    cfg = {**DEFAULT_CONFIG, **(cfg or {})}

    device = get_device()
    print(f"[Train] Using device: {device}")

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader, le = get_dataloaders(
        root_dir     = cfg["dataset_root"],
        batch_size   = cfg["batch_size"],
        val_size     = cfg["val_size"],
        test_size    = cfg["test_size"],
        num_workers  = cfg["num_workers"],
    )
    num_classes = len(le.classes_)
    print(f"[Train] Classes ({num_classes}): {list(le.classes_)}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(
        model_type  = cfg["model_type"],
        num_classes = num_classes,
        lstm_hidden = cfg["lstm_hidden"],
        lstm_layers = cfg["lstm_layers"],
        dropout     = cfg["dropout"],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Train] Model: {cfg['model_type']}  |  Parameters: {n_params:,}")

    # ── Loss / Optimizer / Scheduler ──────────────────────────────────────────
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg["label_smoothing"])
    optimizer = AdamW(model.parameters(),
                      lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = OneCycleLR(
        optimizer,
        max_lr       = cfg["lr"],
        steps_per_epoch = len(train_loader),
        epochs       = cfg["epochs"],
        pct_start    = 0.3,
    )

    # ── CSV log ───────────────────────────────────────────────────────────────
    log_path = cfg["log_csv"]
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(
            ["epoch", "train_loss", "train_acc", "train_f1",
             "val_loss",   "val_acc",   "val_f1",   "lr", "elapsed_s"]
        )

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_acc    = 0.0
    patience_count  = 0
    best_ckpt_path  = os.path.join(cfg["checkpoint_dir"], "best_model.pt")

    for epoch in range(1, cfg["epochs"] + 1):
        t0 = time.time()

        tr_loss, tr_acc, tr_f1 = run_epoch(
            model, train_loader, criterion, optimizer, scheduler, device, train=True
        )
        vl_loss, vl_acc, vl_f1 = run_epoch(
            model, val_loader,   criterion, None,      None,      device, train=False
        )

        elapsed = time.time() - t0
        current_lr = optimizer.param_groups[0]["lr"]

        # Logging
        print(
            f"Epoch {epoch:03d}/{cfg['epochs']}  "
            f"TrainLoss={tr_loss:.4f} TrainAcc={tr_acc:.4f} TrainF1={tr_f1:.4f}  |  "
            f"ValLoss={vl_loss:.4f}  ValAcc={vl_acc:.4f}  ValF1={vl_f1:.4f}  "
            f"LR={current_lr:.2e}  t={elapsed:.1f}s"
        )
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch, tr_loss, tr_acc, tr_f1,
                vl_loss, vl_acc, vl_f1, current_lr, elapsed
            ])

        # Checkpoint
        if vl_acc > best_val_acc:
            best_val_acc   = vl_acc
            patience_count = 0
            save_checkpoint(
                {
                    "epoch":           epoch,
                    "model_state":     model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "best_val_acc":    best_val_acc,
                    "classes":         list(le.classes_),
                    "config":          cfg,
                },
                best_ckpt_path,
            )
            print(f"  ✓ New best val_acc={best_val_acc:.4f} – saved checkpoint")
        else:
            patience_count += 1
            if patience_count >= cfg["patience"]:
                print(f"[Train] Early stopping at epoch {epoch} "
                      f"(no improvement for {cfg['patience']} epochs)")
                break

    # ── Test evaluation ────────────────────────────────────────────────────────
    print("\n[Train] Loading best checkpoint for test evaluation …")
    load_checkpoint(best_ckpt_path, model)
    model.to(device)
    te_loss, te_acc, te_f1 = run_epoch(
        model, test_loader, criterion, None, None, device, train=False
    )
    print(f"[Test]  Loss={te_loss:.4f}  Acc={te_acc:.4f}  F1={te_f1:.4f}")

    return model, le, best_val_acc, (te_loss, te_acc, te_f1)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse, json

    parser = argparse.ArgumentParser(description="Train emotion recognition model")
    parser.add_argument("--dataset",    default="dataset",
                        help="Path to TESS dataset root folder")
    parser.add_argument("--epochs",     type=int,   default=60)
    parser.add_argument("--batch_size", type=int,   default=32)
    parser.add_argument("--lr",         type=float, default=3e-4)
    parser.add_argument("--model",      default="cnn_lstm",
                        choices=["cnn_lstm", "cnn"])
    parser.add_argument("--patience",   type=int, default=10)
    args = parser.parse_args()

    cfg = {
        **DEFAULT_CONFIG,
        "dataset_root": args.dataset,
        "epochs":       args.epochs,
        "batch_size":   args.batch_size,
        "lr":           args.lr,
        "model_type":   args.model,
        "patience":     args.patience,
    }
    train(cfg)