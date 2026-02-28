# 🫀 Heart Sound Classification System

> A clinical-grade machine learning pipeline for automated heart sound analysis — from raw audio acquisition to deployable diagnostic models.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange?logo=scikit-learn)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.x-green)](https://xgboost.readthedocs.io)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow)](https://tensorflow.org)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Features Extracted](#features-extracted)
- [Model Architectures](#model-architectures)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Pipeline Walkthrough](#pipeline-walkthrough)
- [Project Structure](#project-structure)
- [Results & Metrics](#results--metrics)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Roadmap](#roadmap)

---

## Overview

Cardiovascular disease is a leading cause of global mortality. Early detection through automated auscultation offers a cost-effective, non-invasive screening pathway — especially in resource-limited clinical settings.

This project implements a complete **end-to-end ML pipeline** for binary heart sound classification (**Normal vs. Abnormal**) using the [PhysioNet/CinC Challenge 2016](https://physionet.org/content/challenge-2016/1.0.0/) dataset. The pipeline covers:

1. **Automated dataset acquisition** from PhysioNet with fallback support
2. **Clinical-grade audio preprocessing** — bandpass filtering, normalization, segmentation
3. **Multi-domain feature extraction** — MFCCs, spectral features, temporal patterns
4. **Multi-architecture model training** — Random Forest, XGBoost, and Neural Network
5. **Robust evaluation** — cross-validation, AUC-ROC, precision/recall, confusion matrix
6. **Deployment-ready model export** via `joblib`

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     HEART SOUND CLASSIFICATION PIPELINE             │
│                                                                     │
│  01_data_downloader.py                                              │
│  ┌─────────────────────────────────────────────────────┐           │
│  │  PhysioNet 2016    Pascal HSDB    Michigan HSDB      │           │
│  │  (3,126 .wav)       (fallback)      (fallback)       │           │
│  └──────────────────────────┬──────────────────────────┘           │
│                             ▼                                       │
│  02_preprocessing_features.py                                       │
│  ┌─────────────────────────────────────────────────────┐           │
│  │  Load → Resample (2kHz) → Bandpass Filter (40-800Hz)│           │
│  │  → Segment (5s, 50% overlap) → Quality Filter       │           │
│  │  → Feature Extraction:                              │           │
│  │      MFCCs (52) + Spectral (24) + Temporal (9)      │           │
│  │  → Augmentation (noise, stretch, pitch)             │           │
│  └──────────────────────────┬──────────────────────────┘           │
│                             ▼                                       │
│  03_model_training.py                                               │
│  ┌─────────────────────────────────────────────────────┐           │
│  │  StandardScaler → DB-aware Train/Test Split          │           │
│  │  ┌────────────┐  ┌──────────┐  ┌──────────────────┐ │           │
│  │  │Random      │  │XGBoost   │  │Neural Network    │ │           │
│  │  │Forest      │  │(primary) │  │(Dense+BN+Dropout)│ │           │
│  │  └────────────┘  └──────────┘  └──────────────────┘ │           │
│  │  → 5-fold CV → Eval (F1, AUC-ROC) → Export .pkl     │           │
│  └─────────────────────────────────────────────────────┘           │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Dataset

### PhysioNet/CinC Challenge 2016

| Property | Details |
|---|---|
| **Source** | [physionet.org/content/challenge-2016](https://physionet.org/content/challenge-2016/1.0.0/) |
| **Recordings** | 3,126 `.wav` files across 6 training sets (training-a through training-f) |
| **Sample Rate** | 2,000 Hz |
| **Duration** | 5–120 seconds per recording |
| **Labels** | Binary — `normal` / `abnormal` |
| **Total Size** | ~169 MB |

**Class Distribution (approximate):**
- Normal: ~2,500 recordings (~80%)
- Abnormal: ~626 recordings (~20%)

> ⚠️ **Class Imbalance:** A ~4:1 imbalance exists — the pipeline flags this automatically and applies stratified splitting to handle it.

**Alternative Download (Kaggle):**
```
https://www.kaggle.com/datasets/bjoernjostein/physionet-challenge-2016
```

---

## Features Extracted

The pipeline extracts **85 features** per audio segment across three domains:

### 1. MFCC Features (52 features)
Mel-Frequency Cepstral Coefficients capture the timbral characteristics of heart sounds.

| Feature Group | Count | Description |
|---|---|---|
| MFCC Mean | 13 | Average coefficients across time |
| MFCC Std | 13 | Variance across time frames |
| MFCC Delta | 13 | First derivative (velocity) |
| MFCC Delta² | 13 | Second derivative (acceleration) |

**Parameters:** `n_mfcc=13`, `n_fft=512`, `hop_length=256`, `n_mels=40`

### 2. Spectral Features (24 features)

| Feature | Description |
|---|---|
| Spectral Centroid | "Centre of mass" of the frequency spectrum |
| Spectral Rolloff | Frequency below which 85% of energy lies |
| Spectral Bandwidth | Width of the frequency band |
| Zero Crossing Rate | Rate of sign changes (noise indicator) |
| RMS Energy | Signal loudness/power |
| Spectral Contrast (×n) | Difference between spectral peaks and valleys |
| Chroma (×12) | Energy distribution across 12 pitch classes |

### 3. Temporal Features (9 features)

| Feature | Clinical Relevance |
|---|---|
| Mean, Std | Signal baseline and variability |
| Skewness, Kurtosis | Distribution shape (murmur detection) |
| Energy, Power | Overall signal strength |
| Heart Rate Estimate | Derived from peak detection |
| Envelope Mean/Std | S1/S2 sound boundary characteristics |

---

## Model Architectures

### Random Forest
```
n_estimators=200, max_depth=15, min_samples_split=5
```
Robust ensemble baseline with built-in feature importance ranking.

### XGBoost *(Primary / Recommended)*
```
n_estimators=200, max_depth=6, learning_rate=0.1
subsample=0.8, colsample_bytree=0.8
```
Gradient boosting with early stopping. Best balance of accuracy and interpretability for tabular audio features.

### Neural Network (TensorFlow/Keras)
```
Dense(128) → BN → Dropout(0.3)
Dense(64)  → BN → Dropout(0.3)
Dense(32)  → BN → Dropout(0.3)
Dense(1, sigmoid)
```
Optimized with `Adam`, `EarlyStopping` (patience=15), and `ReduceLROnPlateau`.

---

## Installation

### Prerequisites

- Python 3.8+
- pip

### Install Dependencies

```bash
git clone https://github.com/your-org/heart-sound-classification.git
cd heart-sound-classification
pip install -r requirements.txt
```

### `requirements.txt`
```
numpy>=1.21
pandas>=1.3
librosa>=0.10
scikit-learn>=1.1
xgboost>=1.7
scipy>=1.7
matplotlib>=3.4
seaborn>=0.11
requests>=2.28
joblib>=1.1
tensorflow>=2.10  # optional, for neural network
```

---

## Quick Start

```bash
# Step 1: Download dataset
python 01_data_downloader.py

# Step 2: Extract features (~10-30 min for full dataset)
python 02_preprocessing_features.py

# Step 3: Train and evaluate models
python 03_model_training.py
```

---

## Pipeline Walkthrough

### Step 1 — Data Download (`01_data_downloader.py`)

Attempts automatic download from PhysioNet. Falls back to synthetic data if network access fails.

```python
downloader = HeartSoundDatasetDownloader(base_dir="datasets")
downloader.download_sample_data()
```

**Outputs:**
```
datasets/
  physionet_2016/
    training-a/    ← .wav files + REFERENCE.csv
    training-b/
    ...
    training-f/
    REFERENCE.csv  ← master label file
```

If PhysioNet is unreachable, follow the on-screen manual download instructions pointing to Kaggle.

---

### Step 2 — Preprocessing & Feature Extraction (`02_preprocessing_features.py`)

```python
preprocessor = HeartSoundPreprocessor(
    target_sr=2000,
    segment_duration=5.0,
    overlap_ratio=0.5
)
extractor = FeatureExtractor(preprocessor)
features_df = extractor.process_physionet_dataset(
    base_dir=".",
    output_file="physionet_heart_features.csv",
    augment=True,
    max_files_per_db=100  # Remove for full dataset
)
```

**Processing Steps per file:**
1. Load & resample to 2 kHz
2. Apply DC offset removal
3. Butterworth bandpass filter (40–800 Hz)
4. Segment into 5-second windows with 50% overlap
5. Quality score assessment (SNR, clipping, silence ratio)
6. Feature extraction across all domains
7. Augmentation for high-quality segments (SNR > 0.7):
   - Gaussian noise injection
   - Time stretching (±10%)
   - Pitch shifting (±2 semitones)

**Output:** `physionet_heart_features.csv` with 85+ feature columns + metadata

---

### Step 3 — Model Training (`03_model_training.py`)

```python
classifier = HeartSoundClassifier(model_type='xgboost')
X_train, X_test, y_train, y_test = classifier.load_physionet_data(
    data_path="physionet_heart_features.csv",
    use_database_split=True  # Prevents cross-database data leakage
)
classifier.train_model(X_train, y_train)
metrics = classifier.evaluate_model(X_test, y_test)
classifier.save_model("physionet_heart_model_xgboost")
```

**Database-Aware Splitting:** The pipeline respects PhysioNet's database boundaries (training-a through -f), ensuring the test set comes from entirely separate recording sessions — preventing subtle data leakage.

**Outputs:**
- `physionet_heart_model_xgboost.pkl` — trained model + scaler + encoder
- `feature_importance_xgboost.png` — top 20 feature importances
- `model_training.log` — full training log

---

## Project Structure

```
heart-sound-classification/
│
├── 01_data_downloader.py          # Dataset acquisition & management
├── 02_preprocessing_features.py   # Audio preprocessing & feature engineering  
├── 03_model_training.py           # Model training, evaluation & export
│
├── datasets/                      # Downloaded raw audio (created at runtime)
│   └── physionet_2016/
│       ├── training-a/
│       └── ...
│
├── physionet_heart_features.csv   # Extracted features (created at runtime)
├── physionet_heart_model_*.pkl    # Saved models (created at runtime)
├── feature_importance_*.png       # Visualization outputs (created at runtime)
│
├── dataset_download.log           # Download logs
├── preprocessing.log              # Feature extraction logs
├── model_training.log             # Training logs
│
└── README.md
```

---

## Results & Metrics

Expected performance on the PhysioNet 2016 dataset with XGBoost (full dataset, database-aware split):

| Metric | Expected Range |
|---|---|
| **Accuracy** | 0.82 – 0.88 |
| **F1-Score** | 0.78 – 0.85 |
| **AUC-ROC** | 0.85 – 0.92 |
| **Recall (Sensitivity)** | 0.75 – 0.85 |
| **Precision (PPV)** | 0.78 – 0.87 |

> 📌 **Clinical Note:** Recall (sensitivity) is prioritized in cardiac screening — missing an abnormal case (false negative) is more clinically costly than a false positive. A recall ≥ 0.80 is the primary optimization target.

### Interpreting the Model Output
- `accuracy ≥ 0.85` → Excellent: suitable for clinical screening
- `accuracy ≥ 0.75` → Acceptable: viable with human review
- `recall ≥ 0.80` → Good sensitivity for abnormal detection
- `precision ≥ 0.80` → Low false alarm rate

---

## Configuration

Key parameters to tune by use case:

| Parameter | Location | Default | Notes |
|---|---|---|---|
| `target_sr` | `HeartSoundPreprocessor` | `2000` | Optimal for heart sounds (20-800 Hz) |
| `segment_duration` | `HeartSoundPreprocessor` | `5.0` | Seconds per segment |
| `overlap_ratio` | `HeartSoundPreprocessor` | `0.5` | 50% overlap |
| `n_mfcc` | `mfcc_params` | `13` | Increase to 20 for richer features |
| `n_estimators` | `model_configs` | `200` | Increase for better RF/XGBoost |
| `max_files_per_db` | `process_physionet_dataset` | `None` | Set to 100 for quick testing |
| `model_type` | `HeartSoundClassifier` | `'xgboost'` | `'random_forest'` / `'neural_network'` |

---

## Troubleshooting

**PhysioNet download fails:**
```
→ Use the Kaggle mirror: https://www.kaggle.com/datasets/bjoernjostein/physionet-challenge-2016
→ Extract ZIP files into datasets/physionet_2016/
→ Ensure REFERENCE.csv is present in each training-* folder
```

**`physionet_heart_features.csv` not found when running step 3:**
```
→ Run 02_preprocessing_features.py first
→ Ensure your working directory contains training-a/, training-b/, etc.
```

**Low recall on abnormal class:**
```
→ Remove the quality filter threshold (quality < 0.3) to include more abnormal samples
→ Apply class_weight='balanced' to RandomForestClassifier
→ Increase augmentation for abnormal class specifically
```

**TensorFlow import error:**
```
→ The neural network is optional — Random Forest and XGBoost work without TF
→ Install with: pip install tensorflow>=2.10
```

**Memory error during feature extraction:**
```
→ Set max_files_per_db=50 to reduce batch size
→ Process one training database at a time
```

---

## Roadmap

- [ ] Convolutional Neural Network on mel-spectrograms (raw 2D input)
- [ ] Heart cycle segmentation using Schmidt state machine
- [ ] Multi-class classification (murmur type detection)
- [ ] REST API endpoint for real-time inference
- [ ] ONNX export for edge/mobile deployment
- [ ] Integration with digital stethoscope hardware
- [ ] Explainability layer (SHAP values) for clinical interpretability

---

## Citation

If you use this pipeline in your research, please cite the original dataset:

```bibtex
@article{clifford2016classification,
  title={Classification of normal/abnormal heart sound recordings: The PhysioNet/Computing in Cardiology Challenge 2016},
  author={Clifford, Gari D and Liu, Chengyu and Moody, Benjamin and others},
  journal={Computing in Cardiology},
  volume={43},
  year={2016}
}
```

---

## License

This project is licensed under the MIT License. The PhysioNet/CinC 2016 dataset is subject to its own [PhysioNet Credentialed Access](https://physionet.org/content/challenge-2016/1.0.0/) terms.

---

<div align="center">

**Built for clinical impact. Designed for extensibility.**

*Questions or contributions? Open an issue or submit a PR.*

</div>
