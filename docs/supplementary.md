# Supplementary Material

This document provides complete experimental details to ensure full reproducibility of the results presented in the main paper: *Understanding Token Specialization in Distilled Vision Transformers under Severe Data Imbalance*.

## 1. Experimental Setup & Hardware

All experiments were conducted to ensure that differences in Token Specialization emerged solely from dataset imbalance and Knowledge Distillation dynamics, rather than computational inconsistencies.

- **Hardware:** Experiments were run in a Kaggle environment utilizing Dual NVIDIA T4 GPUs (2x 16GB VRAM) for the architectural scaling runs, and single NVIDIA T4 GPUs for ablation studies.
- **Framework:** PyTorch 2.0+ with Automatic Mixed Precision (`torch.amp`) and `torch.backends.cudnn.benchmark=True` for optimized ViT throughput.
- **Runtime Limits:** Due to 12-hour session limits on Kaggle, our optimized training loops executed a compressed 300-epoch schedule in ~3.3 hours on single T4 GPUs.

## 2. Dataset Specifics (CIFAR-10-LT & CIFAR-100-LT)

We utilized the standard exponential imbalance function to construct the long-tailed datasets.

- **Imbalance Factor (IF):** Defined as $N_{max} / N_{min}$.
- **CIFAR-10-LT (IF=50):** 
  - $N_{max}$ (Class 0): 5,000 images
  - $N_{min}$ (Class 9): 100 images
- **CIFAR-100-LT (IF=50):**
  - $N_{max}$ (Class 0): 500 images
  - $N_{min}$ (Class 99): 10 images
- **Data Augmentation:**
  - RandomCrop(32, padding=4)
  - RandomHorizontalFlip()
  - Resize(224) (required for ViT-based architectures)
  - Standard Normalization
  - MixUp ($\alpha=0.8$) applied for early epochs, disabled when Deferred Reweighting (DRW) begins.

## 3. Hyperparameters

### 3.1 Network Training (Student)
- **Architectures:** `deit_tiny_patch16_224`, `deit_small_patch16_224`, `deit_base_patch16_224`
- **Optimizer:** AdamW
- **Base Learning Rate:** 1e-3 (Single GPU Ablations) / 2e-3 (Dual GPU Scaling)
- **Weight Decay:** 0.05
- **Scheduler:** CosineAnnealingLR ($T_{max} = Epochs$)
- **Epochs:** 300
- **DRW Epoch:** 240 (Deferred Reweighting begins at epoch 240, equivalent to the 80% mark of the schedule)
- **Batch Size:** 256
- **Global Seed:** 42 (Seed 100 utilized during variance ablations)

### 3.2 Teacher Architectures
The CNN teacher models were initialized from standard pretrained weights and their final FC layers were re-initialized to match the target class count (10 or 100).
- **Teacher for Tiny/Small:** `resnext50_32x4d`
- **Teacher for Base:** `resnet101`

### 3.3 Neural Entropy Router
- **Input Features:** $X \in \mathbb{R}^4$ (Confidence\_CLS, Confidence\_DIST, Entropy\_CLS, Entropy\_DIST)
- **Hidden Layers:** Single hidden layer (16 units) with ReLU activation.
- **Output:** Scalar $\alpha \in (0, 1)$ via Sigmoid activation.
- **Optimizer:** Adam
- **Learning Rate:** 0.01
- **Epochs:** 500 (Self-supervised over test-set logits)

## 4. Oracle Alpha Grid Search Details

The Oracle search swept $\alpha \in [0, 1]$ in increments of 0.05 to find the mathematical upper-bound of Token Fusion. For a given class $c$, the optimal $\alpha_c$ was chosen as:
$$ \alpha_c = \arg\max_{\alpha} \frac{1}{N_c} \sum_{i \in c} \mathbb{I} [ \arg\max ( \alpha L_{cls}^i + (1-\alpha)L_{dist}^i ) == c ] $$
Where $L_{cls}^i$ and $L_{dist}^i$ are the logits for sample $i$. This produced the discrete step-function that our Neural Entropy Router successfully learned to approximate.

## 5. The No-KD Causal Baseline
For Phase 6 (No-KD), the exact `deit_tiny_patch16_224` architecture (with both tokens) was instantiated. The teacher was entirely removed. The loss function was modified from the DeiT Knowledge Distillation formulation:
$$ L_{KD} = L_{CE}(L_{cls}, y) + L_{CE}(L_{dist}, \arg\max(L_{teacher})) $$
To a symmetric ground-truth supervision:
$$ L_{No-KD} = L_{CE}(L_{cls}, y) + L_{CE}(L_{dist}, y) $$
This experiment conclusively demonstrated the collapse of token specialization.
