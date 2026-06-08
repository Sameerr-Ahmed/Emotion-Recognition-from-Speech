"""
visualization.py
----------------
Plotting utilities:
  1. Training curves (loss, accuracy, F1)
  2. Confusion matrix (absolute + normalised)
  3. Per-class F1 bar chart
  4. MFCC feature visualisation for a single audio file
  5. Attention weight heatmap
  6. t-SNE embedding of extracted features
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless-safe backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import csv

from sklearn.manifold import TSNE
from sklearn.metrics  import (
    confusion_matrix, classification_report, f1_score
)

# ── Shared style ───────────────────────────────────────────────────────────────
PALETTE   = "Blues"
FIG_DPI   = 150
SAVE_DIR  = "plots"

def _save(fig, name: str):
    os.makedirs(SAVE_DIR, exist_ok=True)
    path = os.path.join(SAVE_DIR, name)
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[Viz] Saved → {path}")
    return path


# ── 1. Training curves ─────────────────────────────────────────────────────────

def plot_training_curves(log_csv: str = "training_log.csv", save: bool = True):
    """Read the CSV written by train.py and plot loss / accuracy / F1."""
    epochs, tr_loss, vl_loss = [], [], []
    tr_acc, vl_acc           = [], []
    tr_f1,  vl_f1            = [], []

    with open(log_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            tr_loss.append(float(row["train_loss"]))
            vl_loss.append(float(row["val_loss"]))
            tr_acc.append(float(row["train_acc"]))
            vl_acc.append(float(row["val_acc"]))
            tr_f1.append(float(row["train_f1"]))
            vl_f1.append(float(row["val_f1"]))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Training History", fontsize=14, fontweight="bold")

    for ax, (tr, vl, title) in zip(axes, [
        (tr_loss, vl_loss, "Loss"),
        (tr_acc,  vl_acc,  "Accuracy"),
        (tr_f1,   vl_f1,   "Weighted F1"),
    ]):
        ax.plot(epochs, tr, label="Train",      color="#2196F3")
        ax.plot(epochs, vl, label="Validation", color="#F44336", linestyle="--")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    if save:
        return _save(fig, "training_curves.png")
    plt.show()


# ── 2. Confusion matrix ────────────────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred, class_names,
                          normalise: bool = True, save: bool = True):
    cm = confusion_matrix(y_true, y_pred)
    if normalise:
        cm_plot = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        fmt, vmax = ".2f", 1.0
    else:
        cm_plot = cm
        fmt, vmax = "d", None

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(cm_plot, annot=True, fmt=fmt,
                xticklabels=class_names, yticklabels=class_names,
                cmap=PALETTE, vmin=0, vmax=vmax, ax=ax,
                linewidths=0.5, linecolor="white")
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True",      fontsize=12)
    ax.set_title("Confusion Matrix" + (" (Normalised)" if normalise else ""),
                 fontsize=13, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    if save:
        suffix = "normalised" if normalise else "raw"
        return _save(fig, f"confusion_matrix_{suffix}.png")
    plt.show()


# ── 3. Per-class F1 ───────────────────────────────────────────────────────────

def plot_per_class_f1(y_true, y_pred, class_names, save: bool = True):
    f1s = f1_score(y_true, y_pred, average=None, zero_division=0)
    order = np.argsort(f1s)[::-1]

    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.bar([class_names[i] for i in order],
                  [f1s[i] for i in order],
                  color=sns.color_palette("Blues_d", len(class_names)))
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Emotion")
    ax.set_ylabel("F1 Score")
    ax.set_title("Per-Class F1 Score", fontweight="bold")
    for bar, val in zip(bars, [f1s[i] for i in order]):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01, f"{val:.2f}",
                ha="center", va="bottom", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    if save:
        return _save(fig, "per_class_f1.png")
    plt.show()


# ── 4. MFCC visualisation ──────────────────────────────────────────────────────

def plot_mfcc(audio_path: str, emotion_label: str = "", save: bool = True):
    """Show waveform + MFCC spectrogram for one audio file."""
    import librosa
    import librosa.display
    from dataset import SAMPLE_RATE, N_MFCC, HOP_LENGTH, N_FFT, DURATION

    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, duration=DURATION)
    mfcc  = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC,
                                   n_fft=N_FFT, hop_length=HOP_LENGTH)

    fig   = plt.figure(figsize=(12, 6))
    gs    = gridspec.GridSpec(2, 1, height_ratios=[1, 2])

    # Waveform
    ax0 = fig.add_subplot(gs[0])
    times = np.linspace(0, len(y) / sr, len(y))
    ax0.plot(times, y, color="#1565C0", linewidth=0.6)
    ax0.set_title(f"Waveform  –  emotion: {emotion_label}", fontweight="bold")
    ax0.set_xlabel("Time (s)")
    ax0.set_ylabel("Amplitude")
    ax0.grid(alpha=0.3)

    # MFCC
    ax1 = fig.add_subplot(gs[1])
    img = librosa.display.specshow(mfcc, sr=sr, hop_length=HOP_LENGTH,
                                    x_axis="time", ax=ax1, cmap="magma")
    fig.colorbar(img, ax=ax1, format="%+2.0f dB")
    ax1.set_title(f"MFCCs ({N_MFCC} coefficients)", fontweight="bold")

    plt.tight_layout()
    if save:
        fname = os.path.splitext(os.path.basename(audio_path))[0]
        return _save(fig, f"mfcc_{fname}.png")
    plt.show()


# ── 5. Attention heatmap ───────────────────────────────────────────────────────

def plot_attention(model, sample_tensor, class_names,
                   true_label: str = "", save: bool = True):
    """
    Visualise attention weights over time frames.
    sample_tensor: (1, 1, F, T) torch.Tensor
    """
    import torch
    model.eval()
    with torch.no_grad():
        logits, attn_w = model.get_attention_weights(sample_tensor)
    pred_idx  = logits.argmax(dim=1).item()
    pred_label = class_names[pred_idx]
    attn_w     = attn_w.squeeze().cpu().numpy()

    fig, ax = plt.subplots(figsize=(12, 2))
    ax.plot(attn_w, color="#7B1FA2", linewidth=1.5)
    ax.fill_between(range(len(attn_w)), attn_w, alpha=0.3, color="#CE93D8")
    ax.set_title(
        f"Attention weights  |  Predicted: {pred_label}  "
        + (f"(True: {true_label})" if true_label else ""),
        fontweight="bold"
    )
    ax.set_xlabel("Time frame")
    ax.set_ylabel("Weight")
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if save:
        return _save(fig, "attention_weights.png")
    plt.show()


# ── 6. t-SNE embedding ────────────────────────────────────────────────────────

def plot_tsne(features: np.ndarray, labels: np.ndarray,
              class_names, save: bool = True):
    """
    2-D t-SNE of the feature matrix.
    features: (N, D)
    labels:   (N,) integer class indices
    """
    print("[Viz] Computing t-SNE (may take a minute) …")
    tsne  = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000)
    emb   = tsne.fit_transform(features)

    palette = sns.color_palette("tab10", len(class_names))
    fig, ax = plt.subplots(figsize=(9, 7))

    for i, cname in enumerate(class_names):
        mask = labels == i
        ax.scatter(emb[mask, 0], emb[mask, 1],
                   label=cname, color=palette[i], alpha=0.6, s=18)

    ax.set_title("t-SNE of Audio Features", fontweight="bold", fontsize=13)
    ax.legend(loc="best", fontsize=8)
    ax.axis("off")
    plt.tight_layout()

    if save:
        return _save(fig, "tsne_embeddings.png")
    plt.show()


# ── Convenience: generate all post-training plots ────────────────────────────

def generate_all_evaluation_plots(y_true, y_pred, class_names,
                                   log_csv="training_log.csv"):
    print("\n[Viz] Generating evaluation plots …")
    if os.path.exists(log_csv):
        plot_training_curves(log_csv)
    plot_confusion_matrix(y_true, y_pred, class_names, normalise=True)
    plot_confusion_matrix(y_true, y_pred, class_names, normalise=False)
    plot_per_class_f1(y_true, y_pred, class_names)

    # Print text report too
    print("\n" + classification_report(y_true, y_pred,
                                        target_names=class_names, zero_division=0))
    print(f"[Viz] All plots saved to '{SAVE_DIR}/' directory")


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_csv", default="training_log.csv")
    args = parser.parse_args()
    plot_training_curves(args.log_csv)