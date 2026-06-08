"""
dataset.py
----------
Handles loading the TESS dataset, extracting audio features (MFCCs, chroma, mel),
and preparing train/val/test splits ready for PyTorch.
"""

import os
import re
import numpy as np
import librosa
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

# ── Constants ─────────────────────────────────────────────────────────────────
SAMPLE_RATE   = 22050
DURATION      = 3          # seconds – pad / truncate to this length
N_MFCC        = 40
N_MELS        = 128
HOP_LENGTH    = 512
N_FFT         = 2048
MAX_LEN       = int(SAMPLE_RATE * DURATION / HOP_LENGTH) + 1   # ~130 frames

# TESS emotion labels encoded in filenames  (e.g. OAF_angry, YAF_happy …)
TESS_EMOTIONS = {
    "angry":     "angry",
    "disgust":   "disgust",
    "fear":      "fear",
    "happy":     "happy",
    "neutral":   "neutral",
    "ps":        "surprise",   # "pleasant surprise" in TESS
    "sad":       "sad",
}

# ── Feature extraction ────────────────────────────────────────────────────────

def load_audio(path: str, sr: int = SAMPLE_RATE, duration: float = DURATION):
    """Load a wav file, resample, and fix to `duration` seconds."""
    y, _ = librosa.load(path, sr=sr, duration=duration)
    target_len = int(sr * duration)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]
    return y


def extract_features(y: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Extract a 2-D feature map of shape (n_features, MAX_LEN).

    Channels:
        0:39  – MFCCs
        40:43 – MFCC deltas
        44:47 – MFCC delta-deltas
        48:51 – Chroma STFT
        52     – RMS energy
        Total  = 53 feature rows
    """
    mfcc        = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC,
                                        n_fft=N_FFT, hop_length=HOP_LENGTH)
    mfcc_delta  = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
    chroma      = librosa.feature.chroma_stft(y=y, sr=sr,
                                               n_fft=N_FFT, hop_length=HOP_LENGTH)
    rms         = librosa.feature.rms(y=y, hop_length=HOP_LENGTH)

    features = np.vstack([mfcc, mfcc_delta, mfcc_delta2, chroma, rms])  # (53, T)

    # Pad / truncate time axis
    if features.shape[1] < MAX_LEN:
        pad = MAX_LEN - features.shape[1]
        features = np.pad(features, ((0, 0), (0, pad)))
    else:
        features = features[:, :MAX_LEN]

    return features.astype(np.float32)   # (53, MAX_LEN)


# ── TESS parser ───────────────────────────────────────────────────────────────

def parse_tess(root_dir: str):
    """
    Walk the TESS folder structure and return (file_paths, labels).

    Expected layout (either flat or one-level nested):
        root_dir/
            OAF_angry/  *.wav
            YAF_happy/  *.wav
            …
        OR
        root_dir/
            TESS Toronto emotional speech set data/
                OAF_angry/ …
    """
    paths, labels = [], []

    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if not fname.lower().endswith(".wav"):
                continue
            # emotion is the last underscore-separated token in the folder name
            folder = os.path.basename(dirpath).lower()
            parts  = folder.split("_")
            emotion_key = parts[-1] if parts else ""
            if emotion_key not in TESS_EMOTIONS:
                continue
            paths.append(os.path.join(dirpath, fname))
            labels.append(TESS_EMOTIONS[emotion_key])

    if not paths:
        raise FileNotFoundError(
            f"No TESS .wav files found under '{root_dir}'.\n"
            "Make sure the folder contains sub-directories like OAF_angry, YAF_happy, etc."
        )
    return paths, labels


# ── PyTorch Dataset ───────────────────────────────────────────────────────────

class TESSDataset(Dataset):
    def __init__(self, paths, labels, label_encoder, augment=False):
        self.paths         = paths
        self.labels        = label_encoder.transform(labels)
        self.label_encoder = label_encoder
        self.augment       = augment

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        y = load_audio(self.paths[idx])

        if self.augment:
            y = self._augment(y)

        feat = extract_features(y)                         # (53, MAX_LEN)
        feat = torch.tensor(feat).unsqueeze(0)             # (1, 53, MAX_LEN)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return feat, label

    # ── Simple augmentation ──────────────────────────────────────────────────
    @staticmethod
    def _augment(y: np.ndarray) -> np.ndarray:
        """Apply one of three random augmentations."""
        choice = np.random.randint(3)
        if choice == 0:
            # Time stretch  (0.9 – 1.1×)
            rate = np.random.uniform(0.9, 1.1)
            y = librosa.effects.time_stretch(y, rate=rate)
        elif choice == 1:
            # Pitch shift  (±2 semitones)
            steps = np.random.uniform(-2, 2)
            y = librosa.effects.pitch_shift(y, sr=SAMPLE_RATE, n_steps=steps)
        else:
            # Add white noise
            noise = np.random.randn(len(y)) * 0.005
            y = y + noise
        # Re-fix length after augmentation
        target_len = int(SAMPLE_RATE * DURATION)
        if len(y) < target_len:
            y = np.pad(y, (0, target_len - len(y)))
        else:
            y = y[:target_len]
        return y


# ── Public helper ─────────────────────────────────────────────────────────────

def get_dataloaders(
    root_dir: str,
    batch_size: int = 32,
    val_size: float = 0.15,
    test_size: float = 0.15,
    num_workers: int = 0,
    encoder_save_path: str = "label_encoder.pkl",
):
    """
    Build train / val / test DataLoaders from the TESS root directory.

    Returns
    -------
    train_loader, val_loader, test_loader, label_encoder
    """
    paths, labels = parse_tess(root_dir)
    print(f"[Dataset] Found {len(paths)} audio files | "
          f"Classes: {sorted(set(labels))}")

    # Encode string labels → integers
    le = LabelEncoder()
    le.fit(labels)
    joblib.dump(le, encoder_save_path)
    print(f"[Dataset] Label encoder saved → {encoder_save_path}")

    # Stratified splits
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        paths, labels, test_size=val_size + test_size,
        stratify=labels, random_state=42
    )
    rel_test = test_size / (val_size + test_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=rel_test,
        stratify=y_tmp, random_state=42
    )

    print(f"[Dataset] Train={len(X_train)} | Val={len(X_val)} | Test={len(X_test)}")

    train_ds = TESSDataset(X_train, y_train, le, augment=True)
    val_ds   = TESSDataset(X_val,   y_val,   le, augment=False)
    test_ds  = TESSDataset(X_test,  y_test,  le, augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=num_workers)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size,
                              shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader, le


# ── Quick smoke-test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else "dataset"
    tl, vl, tel, enc = get_dataloaders(root, batch_size=8)
    x, y = next(iter(tl))
    print(f"Batch shape: {x.shape} | Labels: {y}")
    print(f"Classes: {list(enc.classes_)}")