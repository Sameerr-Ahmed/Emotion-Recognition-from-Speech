"""
model.py
--------
Defines the EmotionCNNLSTM architecture:
    CNN  → extract local spectro-temporal patterns
    BiLSTM → capture temporal dynamics
    Attention → weight important frames
    FC   → classify into emotion classes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Squeeze-and-Excitation block (channel attention) ──────────────────────────
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc   = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w


# ── Temporal self-attention ────────────────────────────────────────────────────
class TemporalAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size * 2, 1)

    def forward(self, lstm_out):
        # lstm_out: (B, T, hidden*2)
        scores = self.attn(lstm_out)          # (B, T, 1)
        weights = torch.softmax(scores, dim=1)  # (B, T, 1)
        context = (lstm_out * weights).sum(dim=1)  # (B, hidden*2)
        return context, weights.squeeze(-1)


# ── Main Model ─────────────────────────────────────────────────────────────────
class EmotionCNNLSTM(nn.Module):
    """
    Input:  (B, 1, n_features, time)   e.g. (B, 1, 53, 130)
    Output: (B, num_classes)
    """

    def __init__(self, num_classes: int, n_features: int = 53,
                 lstm_hidden: int = 256, lstm_layers: int = 2,
                 dropout: float = 0.4):
        super().__init__()
        self.num_classes = num_classes

        # ── CNN encoder ──────────────────────────────────────────────────────
        self.cnn = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            SEBlock(32),
            nn.MaxPool2d(2, 2),          # (B, 32, F/2, T/2)
            nn.Dropout2d(0.2),

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            SEBlock(64),
            nn.MaxPool2d(2, 2),          # (B, 64, F/4, T/4)
            nn.Dropout2d(0.2),

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            SEBlock(128),
            nn.MaxPool2d((2, 1)),        # (B, 128, F/8, T/4)  keep time dim
            nn.Dropout2d(0.3),
        )

        # Compute CNN output time-steps and feature dim
        # After 3 MaxPool operations: F → F//8, T → T//4
                # Compute CNN output size by running a dummy tensor through the CNN
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_features, 130)  # 130 = MAX_LEN
            dummy_out = self.cnn(dummy)
            _, C, H, W = dummy_out.shape
            cnn_feat = C * H   # features per time step
            cnn_time = W
        print(f"[Model] CNN output: {cnn_feat} features per frame, {cnn_time} time frames")
        # ── BiLSTM ───────────────────────────────────────────────────────────
        self.lstm = nn.LSTM(
            input_size=cnn_feat,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        # ── Attention ────────────────────────────────────────────────────────
        self.attention = TemporalAttention(lstm_hidden)

        # ── Classifier ───────────────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        # x: (B, 1, F, T)
        x = self.cnn(x)                    # (B, 128, F', T')

        B, C, F, T = x.shape
        x = x.permute(0, 3, 1, 2)         # (B, T', C, F')
        x = x.reshape(B, T, C * F)        # (B, T', C*F') – one vector per frame

        x, _ = self.lstm(x)               # (B, T', 2*hidden)
        context, attn_w = self.attention(x)  # (B, 2*hidden)

        out = self.classifier(context)    # (B, num_classes)
        return out

    def get_attention_weights(self, x):
        """Return (logits, attention_weights) for visualisation."""
        x = self.cnn(x)
        B, C, F, T = x.shape
        x = x.permute(0, 3, 1, 2).reshape(B, T, C * F)
        x, _ = self.lstm(x)
        context, attn_w = self.attention(x)
        out = self.classifier(context)
        return out, attn_w


# ── Lightweight baseline (pure CNN) ──────────────────────────────────────────
class EmotionCNN(nn.Module):
    """Simpler CNN-only model useful for ablation / fast experiments."""

    def __init__(self, num_classes: int, n_features: int = 53, dropout: float = 0.4, **kwargs):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16, 256), nn.ReLU(True), nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ── Factory ───────────────────────────────────────────────────────────────────
def build_model(model_type: str, num_classes: int, **kwargs) -> nn.Module:
    if model_type == "cnn_lstm":
        return EmotionCNNLSTM(num_classes=num_classes, **kwargs)
    elif model_type == "cnn":
        return EmotionCNN(num_classes=num_classes, **kwargs)
    else:
        raise ValueError(f"Unknown model_type '{model_type}'. "
                         "Choose 'cnn_lstm' or 'cnn'.")


# ── Quick sanity-check ────────────────────────────────────────────────────────
if __name__ == "__main__":
    from dataset import MAX_LEN
    B, F, T = 4, 53, MAX_LEN
    x = torch.randn(B, 1, F, T)

    for mtype in ("cnn_lstm", "cnn"):
        m = build_model(mtype, num_classes=7)
        out = m(x)
        n_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
        print(f"[{mtype:10s}] output={out.shape}  params={n_params:,}")