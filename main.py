"""
main.py
-------
Unified entry point for the Emotion-Recognition-from-Speech project.

Modes
-----
  train      – full training run
  evaluate   – load best checkpoint, run test set, generate all plots
  predict    – predict emotion from a single .wav file
  visualize  – just re-generate plots from existing training_log.csv
  demo       – quick sanity-check (model forward pass, no real data needed)

Usage examples
--------------
  python main.py train   --dataset path/to/TESS
  python main.py evaluate --dataset path/to/TESS
  python main.py predict  --audio  path/to/file.wav
  python main.py visualize
  python main.py demo
"""

import os
import sys
import argparse
import numpy as np
import torch

# ── Silence librosa/numba progress bars ──────────────────────────────────────
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")

from dataset       import get_dataloaders, extract_features, load_audio, MAX_LEN
from model         import build_model
from train         import train as run_training, DEFAULT_CONFIG, get_device, load_checkpoint
from visualization import (
    generate_all_evaluation_plots,
    plot_mfcc,
    plot_attention,
    plot_training_curves,
)


# ── Shared helpers ────────────────────────────────────────────────────────────

def load_best_model(checkpoint_path: str, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg         = ckpt.get("config", DEFAULT_CONFIG)
    classes     = ckpt["classes"]
    num_classes = len(classes)

    model = build_model(
        model_type  = cfg.get("model_type", "cnn_lstm"),
        num_classes = num_classes,
        lstm_hidden = cfg.get("lstm_hidden", 256),
        lstm_layers = cfg.get("lstm_layers", 2),
        dropout     = cfg.get("dropout", 0.4),
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model, classes


# ── Mode: train ───────────────────────────────────────────────────────────────

def mode_train(args):
    cfg = {
        **DEFAULT_CONFIG,
        "dataset_root": args.dataset,
        "epochs":       args.epochs,
        "batch_size":   args.batch_size,
        "lr":           args.lr,
        "model_type":   args.model,
        "patience":     args.patience,
    }
    run_training(cfg)


# ── Mode: evaluate ────────────────────────────────────────────────────────────

def mode_evaluate(args):
    device   = get_device()
    ckpt_path = os.path.join(args.checkpoint_dir, "best_model.pt")

    if not os.path.exists(ckpt_path):
        print(f"[Error] Checkpoint not found: {ckpt_path}")
        print("  Run  python main.py train --dataset <path>  first.")
        sys.exit(1)

    print(f"[Evaluate] Loading checkpoint: {ckpt_path}")
    model, classes = load_best_model(ckpt_path, device)

    _, _, test_loader, le = get_dataloaders(
        root_dir    = args.dataset,
        batch_size  = 64,
        val_size    = 0.15,
        test_size   = 0.15,
        num_workers = 0,
    )

    all_preds, all_labels = [], []
    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(device)
            preds = model(X).argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.numpy())

    generate_all_evaluation_plots(
        y_true      = all_labels,
        y_pred      = all_preds,
        class_names = classes,
    )

    # Optionally: attention plot on one test sample
    try:
        sample_x, sample_y = next(iter(test_loader))
        sample_x = sample_x[[0]].to(device)
        true_lbl  = classes[sample_y[0].item()]
        plot_attention(model, sample_x, classes, true_label=true_lbl)
    except AttributeError:
        pass  # plain CNN model – no attention method


# ── Mode: predict ─────────────────────────────────────────────────────────────

def mode_predict(args):
    device    = get_device()
    ckpt_path = os.path.join(args.checkpoint_dir, "best_model.pt")

    if not os.path.exists(ckpt_path):
        print(f"[Error] Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    model, classes = load_best_model(ckpt_path, device)

    audio_path = args.audio
    if not os.path.exists(audio_path):
        print(f"[Error] Audio file not found: {audio_path}")
        sys.exit(1)

    print(f"[Predict] Processing: {audio_path}")
    y      = load_audio(audio_path)
    feat   = extract_features(y)                         # (53, MAX_LEN)
    tensor = torch.tensor(feat).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,53,T)

    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1).squeeze().cpu().numpy()

    pred_idx   = probs.argmax()
    pred_label = classes[pred_idx]
    confidence = probs[pred_idx] * 100

    print(f"\n{'─'*40}")
    print(f"  Predicted emotion : {pred_label.upper()}")
    print(f"  Confidence        : {confidence:.1f}%")
    print(f"{'─'*40}")
    print("  Full probabilities:")
    for i, (cls, p) in enumerate(zip(classes, probs)):
        bar = "█" * int(p * 30)
        print(f"    {cls:<12} {p*100:5.1f}%  {bar}")

    # Also show MFCC plot
    plot_mfcc(audio_path, emotion_label=pred_label)

    return pred_label, confidence


# ── Mode: visualize ───────────────────────────────────────────────────────────

def mode_visualize(args):
    if not os.path.exists(args.log_csv):
        print(f"[Error] log CSV not found: {args.log_csv}")
        print("  Run training first to generate it.")
        sys.exit(1)
    plot_training_curves(args.log_csv)
    print("[Visualize] Done.")


# ── Mode: demo ────────────────────────────────────────────────────────────────

def mode_demo(_args):
    """Smoke-test: build both models, run a random tensor, print shapes."""
    print("[Demo] Running forward-pass smoke test …")
    from dataset import MAX_LEN
    B, F, T = 2, 53, MAX_LEN
    x = torch.randn(B, 1, F, T)

    for mtype in ("cnn_lstm", "cnn"):
        m   = build_model(mtype, num_classes=7)
        out = m(x)
        n   = sum(p.numel() for p in m.parameters() if p.requires_grad)
        print(f"  [{mtype:10s}] input={tuple(x.shape)} → output={tuple(out.shape)}  "
              f"params={n:,}")

    print("[Demo] All OK ✓")


# ── CLI ───────────────────────────────────────────────────────────────────────

def build_parser():
    p = argparse.ArgumentParser(
        description="Emotion Recognition from Speech – main entry point"
    )
    sub = p.add_subparsers(dest="mode", required=True)

    # ── train ──
    t = sub.add_parser("train", help="Train the model")
    t.add_argument("--dataset",        default="dataset",
                   help="Path to TESS dataset root")
    t.add_argument("--checkpoint_dir", default="checkpoints")
    t.add_argument("--epochs",         type=int,   default=60)
    t.add_argument("--batch_size",     type=int,   default=32)
    t.add_argument("--lr",             type=float, default=3e-4)
    t.add_argument("--model",          default="cnn_lstm",
                   choices=["cnn_lstm", "cnn"])
    t.add_argument("--patience",       type=int,   default=10)

    # ── evaluate ──
    e = sub.add_parser("evaluate", help="Evaluate on test set and generate plots")
    e.add_argument("--dataset",        default="dataset")
    e.add_argument("--checkpoint_dir", default="checkpoints")

    # ── predict ──
    pr = sub.add_parser("predict", help="Predict emotion from a .wav file")
    pr.add_argument("--audio",          required=True, help="Path to .wav file")
    pr.add_argument("--checkpoint_dir", default="checkpoints")

    # ── visualize ──
    v = sub.add_parser("visualize", help="Re-generate training curves")
    v.add_argument("--log_csv", default="training_log.csv")

    # ── demo ──
    sub.add_parser("demo", help="Quick forward-pass sanity check")

    return p


def main():
    parser = build_parser()
    args   = parser.parse_args()

    dispatch = {
        "train":     mode_train,
        "evaluate":  mode_evaluate,
        "predict":   mode_predict,
        "visualize": mode_visualize,
        "demo":      mode_demo,
    }
    dispatch[args.mode](args)


if __name__ == "__main__":
    main()