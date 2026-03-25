# 🌌 Unified Latents (UL) with Learned Noise Schedule

> **Research Submission for QUEST Lab (IISc) Interview**
> This project implements and extends the *Unified Latents* generative framework, specifically focusing on optimizing the diffusion noise schedule through learned, sample-specific parameters.

## 🚀 Overview

The **Unified Latents (UL)** framework treats the entire generative pipeline—Encoder, Prior, and Decoder—as a joint diffusion system. This repository contains a complete implementation on CIFAR-10, featuring a novel research extension: the **Learned Noise Schedule**.

### 🧠 Core Concepts Learned
Through this project, we explored:
- **Joint Latent-Pixel Diffusion**: Training a diffusion prior to map noise to a latent manifold, while a conditional decoder maps those latents back to image space.
- **Log-SNR Parametrization**: Using $\lambda(t)$ (log Signal-to-Noise Ratio) to control the diffusion process, ensuring numerical stability across different scales.
- **Signal-Preserving Sampling**: Implementing a variance-preserving (VP) DDPMSampler that maintains latent manifold structure.

---

## 🔬 Novel Contribution: Learned Noise Schedule

Standard diffusion models use a fixed log-SNR schedule for all samples. Our extension introduces a **NoiseSchedulePredictor**, a lightweight CNN that analyzes latent features to predict an optimal per-sample max noise level ($\lambda_0 \in [2, 8]$).

### 🏆 Why it matters:
- **Adaptive Fidelity**: High-frequency images (e.g., textures) can be assigned higher SNR, while low-frequency images use a more aggressive noise schedule for better generative diversity.
- **Efficiency**: Allows the model to focus its capacity on the samples that are hardest to denoise.
- **Improved Convergence**: Predicted noise levels act as a sample-specific regularizer during the joint training phase.

---

## 🛠️ Implementation Details

### 1. Training Pipeline
The framework follows a robust two-stage strategy:
- **Stage 1 (Joint Training)**: Encoder, Prior, and Decoder are trained together to establish the latent manifold and the noise predictor.
- **Stage 2 (Prior Refinement)**: The Prior is refined using a weighted loss ($\sigma(t)$ weighting) to sharpen the generative mapping.

### 2. Architecture
- **Encoder**: 3-stage ResNet with `NoiseSchedulePredictor` extension.
- **Prior**: Transformer-based diffusion model operating on (128, 4, 4) latents.
- **Decoder**: Conditional Vision Transformer (ViT) architecture for high-fidelity pixel reconstruction.

---

## 📊 Getting Started

### Installation
```bash
pip install torch torchvision einops pyyaml matplotlib tqdm
```

### Training
```bash
# Stage 1: Joint training with the Learned Noise Extension
python3 train.py --stage 1 --use-learned-schedule

# Stage 2: Refined Prior training
python3 train.py --stage 2 --use-learned-schedule --ckpt outputs/ckpt_stage1_final.pt
```

### Analysis
Evaluate the learned noise distribution and visualize the schedule:
```bash
python3 analyze_extension.py --ckpt outputs/ckpt_stage2_final.pt
```

---

## 🎓 QUEST Lab Research Conclusion
This implementation demonstrates that the Unified Latents framework is highly adaptable. By introducing learned noise schedules, we achieved stable convergence and clear reconstructions, proving that **sample-adaptive diffusion parameters** are a viable path for improving latent-space generative models.

**Repository maintained by:** GiGiKoneti 🚀
