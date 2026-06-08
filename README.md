# 🎵 Emotion Recognition from Speech

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)

## Overview

**Emotion Recognition from Speech** is an advanced deep learning system that automatically classifies human emotions from audio clips. The model analyzes speech patterns to detect six distinct emotions: **Angry**, **Happy**, **Sad**, **Fear**, **Disgust**, and **Neutral**.

### Key Features

- ✅ **State-of-the-art Architecture**: CNN-LSTM hybrid model with temporal attention mechanisms
- ✅ **High Accuracy**: 100% test accuracy on the TESS dataset
- ✅ **Real-time Inference**: Quick emotion predictions from 3-second audio clips
- ✅ **User-Friendly GUI**: Interactive desktop application for easy predictions
- ✅ **Comprehensive Feature Extraction**: MFCC, chroma, and energy features
- ✅ **Production-Ready**: Includes training, evaluation, and visualization tools

---

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Training & Evaluation](#training--evaluation)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

---

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- CUDA 11.8+ (optional, for GPU acceleration)

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/Sameerr-Ahmed/Emotion-Recognition-from-Speech.git
   cd Emotion-Recognition-from-Speech
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Quick Start

### Option 1: Predict Emotion from Audio File

```bash
python main.py predict --audio path/to/audio.wav
```

### Option 2: Launch GUI Application

```bash
python app.py
```

The GUI provides an intuitive interface to:
- Select audio files
- Display real-time emotion predictions
- View prediction confidence scores

### Option 3: Full Training Pipeline

```bash
python main.py train --dataset path/to/TESS
```

---

## 📁 Project Structure

```
Emotion-Recognition-from-Speech/
├── main.py                 # Unified entry point (train, evaluate, predict, demo)
├── model.py               # CNN-LSTM architecture with attention
├── train.py               # Training loop with early stopping & checkpointing
├── dataset.py             # Audio loading & feature extraction
├── app.py                 # GUI application (Tkinter)
├── visualization.py       # Plotting & evaluation metrics
├── requirements.txt       # Python dependencies
├── training_log.csv       # Training history & metrics
└── README.md              # This file
```

---

## 💻 Usage

### Command-Line Interface

The `main.py` script supports multiple modes:

#### 1. **Training Mode**
Train the model on your dataset:
```bash
python main.py train --dataset /path/to/TESS
```

#### 2. **Evaluation Mode**
Evaluate on test set and generate plots:
```bash
python main.py evaluate --dataset /path/to/TESS
```

#### 3. **Prediction Mode**
Predict emotion from a single audio file:
```bash
python main.py predict --audio /path/to/file.wav
```

#### 4. **Visualization Mode**
Generate plots from existing training logs:
```bash
python main.py visualize
```

#### 5. **Demo Mode**
Quick sanity check (no data required):
```bash
python main.py demo
```

### GUI Application

Launch the interactive desktop application:
```bash
python app.py
```

**Features:**
- 📂 File browser for audio selection
- 🎤 Real-time emotion prediction
- 📊 Confidence scores visualization
- 🎯 Six emotion classes supported

---

## 🧠 Model Architecture

### EmotionCNNLSTM Network

The model combines multiple deep learning components for robust emotion classification:

```
Input (Audio Spectrogram)
    ↓
[CNN Blocks] → Extract local spectro-temporal patterns
    ↓
[SE-Block] → Channel-wise attention
    ↓
[BiLSTM] → Capture temporal dynamics
    ↓
[Temporal Attention] → Weight important frames
    ↓
[Fully Connected] → Classification (6 emotions)
    ↓
Output (Emotion Probabilities)
```

### Key Components

| Component | Purpose |
|-----------|---------|
| **CNN Layers** | Extract local spectro-temporal features |
| **SE-Block** | Channel attention mechanism |
| **BiLSTM** | Bidirectional temporal modeling |
| **Attention** | Focus on important temporal regions |
| **Dropout** | Regularization to prevent overfitting |

### Feature Extraction

- **MFCC** (Mel-Frequency Cepstral Coefficients): Capture phonetic characteristics
- **Chroma Features**: Represent pitch class energy distribution
- **Energy Features**: Encode amplitude information

---

## 📊 Training & Evaluation

### Training Configuration

The model is trained with:
- **Optimizer**: AdamW with weight decay
- **Loss Function**: Cross-entropy with label smoothing
- **Scheduler**: OneCycleLR for dynamic learning rate
- **Early Stopping**: Prevents overfitting
- **Checkpointing**: Saves best model automatically

### Key Metrics

- **Accuracy**: Classification accuracy across all emotions
- **F1-Score**: Harmonic mean of precision and recall
- **Per-Class Performance**: Detailed breakdown for each emotion

### Evaluation Outputs

Training generates comprehensive visualizations:
- Training/validation loss curves
- Accuracy progression
- MFCC spectrograms
- Confusion matrices
- Attention weight distributions

---

## 📈 Results

### TESS Dataset Performance

| Metric | Score |
|--------|-------|
| **Test Accuracy** | 100% |
| **Weighted F1-Score** | 1.00 |
| **Training Speed** | GPU-accelerated |
| **Inference Time** | <50ms per sample |

### Emotion Distribution

The model successfully classifies:
- 🤬 **Angry**
- 😊 **Happy**
- 😢 **Sad**
- 😨 **Fear**
- 🤢 **Disgust**
- 😐 **Neutral**

---

## 🔧 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | 2.0+ | Deep learning framework |
| `torchaudio` | Latest | Audio processing |
| `librosa` | 0.10+ | Audio feature extraction |
| `numpy` | 1.21+ | Numerical computations |
| `scikit-learn` | 1.0+ | Metrics & evaluation |
| `matplotlib` | 3.4+ | Plotting |
| `seaborn` | 0.11+ | Statistical visualization |
| `soundfile` | Latest | Audio I/O |

---

## 🤝 Contributing

We welcome contributions! To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Areas for Enhancement

- [ ] Support for additional datasets
- [ ] Real-time streaming audio support
- [ ] Model compression for mobile deployment
- [ ] Additional emotion classes
- [ ] Multi-language support

---

## 📝 License

This project is licensed under the **MIT License** - see the LICENSE file for details.

---

## 📧 Contact

**Author**: Sameerr-Ahmed and Saif ur Rehman

For questions, issues, or suggestions, please open an issue on the [GitHub repository](https://github.com/Sameerr-Ahmed/Emotion-Recognition-from-Speech).

---

## 🙏 Acknowledgments

- **TESS Dataset**: Toronto Emotional Speech Set
- **PyTorch Team**: For the exceptional deep learning framework
- **Librosa**: For comprehensive audio analysis tools
- **Contributors**: All who helped improve this project

---

**⭐ If you find this project useful, please consider giving it a star!**
