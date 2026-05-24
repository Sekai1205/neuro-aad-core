# 🧠 neuro-aad-core

![Python](https://img.shields.io/badge/Python-3.13-black?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-MPS_Enabled-black?style=flat-square&logo=pytorch)
![License](https://img.shields.io/badge/License-MIT-black?style=flat-square)

> **Auditory Attention Decoding (AAD) System**
> Featuring a biologically grounded backward mTRF (Multivariate Temporal Response Function) decoder for auditory attention detection from EEG.

## ⚡️ Core Architecture

This repository contains the core algorithms for a robust AAD system designed for hearing prostheses. The system relies on a backward modeling approach that reconstructs the audio envelope from multi-channel EEG data.

* **Backward Decoder:** Reconstructs the attended speech envelope from EEG channels using regularized Ridge Regression.
* **Feature Extraction:** Audio envelopes are extracted using a Gammatone filter bank modeling the human cochlea.
* **Optimized Preprocessing:** Data loading is strictly cached and both EEG and audio envelopes are bandpass filtered (1-8Hz) to isolate cortical tracking of speech.

## 📊 Performance Benchmark (KU Leuven Dataset)

We evaluated the system using the standard KU Leuven EEG dataset across all 16 subjects. The model correctly identifies the attended speaker by comparing the Pearson correlation of the reconstructed envelope against both audio tracks.

| Metric | Performance |
| :--- | :--- |
| **mTRF Correlation ($r$)** | **~ 0.064** (Subject 1) |
| **AAD Accuracy (10s window)** | **68.9% ± 5.6%** (Population Mean) |
| **Chance Level** | 50.0% |

*Evaluated using Leave-One-Trial-Out Cross-Validation (LOOCV).*

## 🛠️ Quick Start

### 1. Environment Setup
This project is managed via `uv` for ultra-fast Python environment resolution.

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt # or install standard scientific stack (scipy, numpy, librosa)
```

### 2. Run AAD Evaluation
Ensure the KU Leuven dataset (`S1.mat` - `S16.mat`) is placed in `dataset/KULeuven_dataset/eeg/`.

```bash
uv run mtrf/experiment_aad_mtrf.py
```

## 🔬 Future Roadmap
- [x] Phase 0: Validate Backward mTRF Decoder against Chance Level
- [x] Phase 1: Benchmark Across 16 Subjects (Achieved ~69% accuracy)
- [ ] Phase 2: Integration of Deep Learning Non-linear models (e.g., CNN/MAML) on top of verified pipeline
- [ ] Phase 3: Real-time C++ Loop integration for ultra-low latency rendering

Developed by Sekai1205. Engineered for human augmentation.