# EEG-Based Person Identification using Hybrid CNN-RNN

A deep learning pipeline for subject identification from EEG (Electroencephalography) signals using the PhysioNet EEG Motor Movement/Imagery Dataset.

---

## Overview

This project implements an end-to-end system that processes raw EEG recordings, extracts meaningful temporal and spatial features, and classifies them by subject identity using a custom Hybrid CNN-RNN neural network. The architecture combines convolutional layers for spatial/spectral feature extraction with an LSTM layer for temporal sequence modeling.

---

## Project Structure

```
EEG_Project/
├── preprocessing.ipynb      # EEG loading, filtering, epoching, and normalization
├── model.ipynb              # Model definition, training, and evaluation
└── data/
    ├── files/               # Raw EDF files (organized by subject)
    │   └── S001/
    │       ├── S001R03.edf
    │       └── ...
    └── processed/           # Preprocessed NumPy arrays (output of preprocessing)
        ├── X_train.npy
        ├── y_train.npy
        ├── X_test.npy
        └── y_test.npy
```

---

## Dataset

**PhysioNet EEG Motor Movement/Imagery Dataset**

- **Subjects:** Up to 109 (configurable via `N_SUBJECTS`)
- **Channels:** 64 EEG electrodes (standard 10-05 montage)
- **Sampling Rate:** 160 Hz
- **Format:** European Data Format (`.edf`)
- **Runs Used:**
  - Training: Runs 3–8
  - Testing: Runs 9–14

Download the dataset from [PhysioNet](https://physionet.org/content/eegmmidb/1.0.0/) and place it under `EEG_Project/data/files/` following the `S{NNN}/S{NNN}R{NN}.edf` naming convention.

---

## Pipeline

### 1. Preprocessing (`preprocessing.ipynb`)

| Step | Details |
|---|---|
| Load raw EDF files | Per subject, concatenate multiple run files |
| Channel name cleanup | Strip trailing dots from channel names |
| Montage alignment | `standard_1005` montage via MNE |
| Bandpass filter | 1–40 Hz (FIR, `firwin` design) |
| Epoching | 4-second windows, 2-second overlap |
| Normalization | Z-score using training set statistics |
| Output | `X_train.npy`, `y_train.npy`, `X_test.npy`, `y_test.npy` |

### 2. Model (`model.ipynb`)

**Architecture: `HybridCNNRNN`**

```
Input: (Batch, 1, 64 channels, 640 time points)
  │
  ├─ Conv2d(1→32, kernel=(1,64), same)  →  BatchNorm → ReLU → AvgPool(1,4)
  │    [Temporal frequency filtering]
  │
  ├─ Conv2d(32→64, kernel=(64,1), valid) →  BatchNorm → ReLU → AvgPool(1,4)
  │    [Spatial channel mixing]
  │
  ├─ Reshape to (Batch, Time, 64)
  │
  ├─ LSTM(input=64, hidden=128, unidirectional)
  │    [Temporal sequence modeling]
  │
  ├─ Dropout(0.5)
  │
  └─ Linear(128 → num_classes)
```

**Training Configuration:**

| Hyperparameter | Value |
|---|---|
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Loss Function | CrossEntropyLoss |
| Batch Size | 64 |
| Epochs | 25 |
| Device | CUDA (GPU) |

---

## Requirements

```
python >= 3.10
mne >= 1.11.0
numpy
torch
scikit-learn
matplotlib
seaborn
```

Install dependencies:

```bash
pip install mne numpy torch scikit-learn matplotlib seaborn
```

> **Note:** This project was developed and tested on Google Colab with a T4 GPU. Drive paths are configured accordingly.

---

## Usage

### Step 1 — Preprocess the data

Open `preprocessing.ipynb` in Google Colab and configure:

```python
N_SUBJECTS = 10        # Number of subjects to process (max 109)
raw_data_dir = '/content/drive/MyDrive/EEG_Project/data/files/'
save_path    = '/content/drive/MyDrive/EEG_Project/data/processed/'
```

Run all cells. Preprocessed arrays will be saved to your Drive.

### Step 2 — Train and evaluate the model

Open `model.ipynb` and verify:

```python
data_path = '/content/drive/MyDrive/EEG_Project/data/processed/'
```

Run all cells. The notebook will:
- Load the preprocessed arrays
- Build the `HybridCNNRNN` model
- Train for 25 epochs with live accuracy reporting
- Output a classification report and confusion matrix

---

## Output & Evaluation

- **Per-epoch logging:** Training and test accuracy printed each epoch
- **Classification Report:** Precision, recall, F1-score per subject
- **Confusion Matrix:** Heatmap visualization (first 20 subjects shown for readability)

---

## Configuration Reference

| Variable | Location | Description |
|---|---|---|
| `N_SUBJECTS` | `preprocessing.ipynb` | Number of subjects to process |
| `TRAIN_RUNS` | `preprocessing.ipynb` | EDF run IDs for training |
| `TEST_RUNS` | `preprocessing.ipynb` | EDF run IDs for testing |
| `BATCH_SIZE` | `model.ipynb` | Mini-batch size |
| `num_classes` | `model.ipynb` | Auto-detected from `y_train` |

---

## Authors

Developed by **Mustafa Othman & Seifeldin ibrahim**

---

## License

This project is for research and educational purposes. The EEG dataset is provided by PhysioNet under the [Open Data Commons Attribution License](https://physionet.org/content/eegmmidb/1.0.0/).
