
# 🧠 neuro-aad-core

![Python](https://img.shields.io/badge/Python-3.13-black?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-MPS_Enabled-black?style=flat-square&logo=pytorch)
![C++](https://img.shields.io/badge/C++-17-black?style=flat-square&logo=c%2B%2B)
![License](https://img.shields.io/badge/License-MIT-black?style=flat-square)

> **Next-Generation Auditory Attention Detection (AAD) System**
> Featuring Model-Agnostic Meta-Learning (MAML) for rapid personalization and a C++ lock-free asynchronous audio rendering engine.

## ⚡️ Core Architecture

This repository contains the core algorithms and system architecture for a real-time, low-latency AAD system designed for next-generation hearing prostheses. The system is strictly divided into two asynchronous loops to guarantee audio rendering stability.

* **Slow Loop (Python/PyTorch):** EEG decoding and MAML-based rapid personalization.
* **Fast Loop (C++):** Lock-free, real-time audio filtering (< 10ms latency).

## 📊 Performance Benchmark (KU Leuven Dataset)

We evaluated the system using the standard KU Leuven EEG dataset (S1). The MAML approach successfully overcomes the inter-subject variability and trial-fingerprint overfitting often seen in baseline CNN models.

| Model Architecture | Calibration Time | Test Accuracy (Unseen Data) |
| :--- | :--- | :--- |
| Baseline 1D-CNN | None | 58.8% (Overfitted) |
| **Proposed MAML-CNN** | **~ 3 minutes (3 steps)** | **85.7%** 🚀 |

## 🛠️ Quick Start

### 1. Environment Setup (macOS / Apple Silicon)
This project is managed via `uv` for ultra-fast Python environment resolution.
```bash
uv venv
source .venv/bin/activate
uv pip install torch torchvision torchaudio scipy matplotlib numpy higher

2. Run MAML Evaluation
Ensure the KU Leuven dataset (S1.mat) is placed in data/KULeuven data set/.
uv run slow_loop/maml_aad_trainer.py

3. Run C++ Fast Loop (Lock-free Test)
clang++ -std=c++17 fast_loop/src/main_audio.cpp -o fast_loop_test
./fast_loop_test

🔬 Future Roadmap
• [x] Phase 0: Baseline CNN vs MAML Evaluation (Achieved >80% accuracy)
• [ ] Phase 1: Integration of MAML and C++ Fast Loop via IPC (Inter-Process Communication)
• [ ] Phase 2: PAMR (Postauricular Muscle Reflex) pilot study for alternative triggering
Developed by Sekai1205. Engineered for real-time human augmentation.

```bash
