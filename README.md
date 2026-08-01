# Understanding Token Specialization in Distilled Vision Transformers

<div align="center">
  <h3>Understanding Token Specialization in Distilled Vision Transformers under Severe Data Imbalance</h3>
  <p>A systematic investigation into how Knowledge Distillation fundamentally alters token behavior in Vision Transformers, causing structural bifurcation into mutually exclusive Head and Tail experts.</p>
</div>

---

## 📖 The Problem
Vision Transformers (ViTs) are notoriously data-hungry and struggle significantly under long-tailed data distributions (e.g., CIFAR-100-LT, ImageNet-LT). Recent state-of-the-art methods like **DeiT-LT** utilize Knowledge Distillation (KD) and a specialized Distillation (`DIST`) token to force a ViT student to learn from a robust CNN teacher.

However, during inference, the standard practice is to rigidly average the outputs of the `CLS` and `DIST` tokens (a 50/50 split). **No one asked what happens inside the `CLS` and `DIST` tokens under data starvation.**

## 🔬 What We Discovered (The Phenomenon)
We conducted a deep architectural audit of distilled ViTs and discovered a profound scientific phenomenon: **Knowledge Distillation causes Token Specialization.**
1. **The Head Expert:** The `CLS` token saturates on majority (Head) classes.
2. **The Tail Expert:** The `DIST` token learns from the teacher to dominate minority (Tail) classes.

Because the tokens become mutually exclusive experts, **the 50/50 averaging heuristic causes Expert Sabotage.** The Head expert drags down the Tail expert, and vice versa.

### The Solution: Neural Entropy Router
We propose an **Adaptive Token Fusion (ATF)** mechanism via a **Neural Entropy Router**. Instead of a fixed 50/50 split, our lightweight MLP dynamically routes token logits based on instance-level *Confidence* and *Shannon Entropy*. 

### The Causal Proof
Is this an artifact of the dataset, the architecture, or KD? We ran a **No-KD Causal Baseline** (`Phase6_Causal_NoKD.py`). By training the exact same DeiT architecture on CIFAR-10-LT *without* the Teacher model, the tokens completely lost their specialization and collapsed symmetrically. We definitively prove that **Knowledge Distillation is the causal driver of Token Specialization.**

---

## 🚀 How to Reproduce

Our codebase is highly optimized. We provide standard PyTorch scripts for massive parallel execution and Jupyter Notebooks for interactive visualization.

### Installation
```bash
git clone https://github.com/Bharath-vip/understanding-token-specialization.git
cd understanding-token-specialization
pip install -r requirements.txt
```

### 1. The Causal Proof (No KD vs KD)
To prove that Knowledge Distillation causes the specialization, run our lightweight Causal Baseline on CIFAR-10-LT:
```bash
# Proves that without a teacher, CLS and DIST perform identically (No Specialization)
python experiments/Phase6_Causal_NoKD.py --batch_size 256 --lr 1e-3
```

### 2. Architecture Scaling (CIFAR-100-LT)
We provide a massive batch-size (1024) script for testing the Neural Router across `deit_tiny`, `deit_small`, and `deit_base` under extreme imbalance (IF=50, 10 images in the tail):
```bash
python experiments/Phase5_CIFAR100_LT.py --model deit_tiny_patch16_224
python experiments/Phase5_CIFAR100_LT.py --model deit_small_patch16_224
```
*(Note: Supports automatic checkpointing for Kaggle/Colab preemption).*

### 3. ImageNet-LT Scaling
To test the phenomenon on large-scale datasets, run:
```bash
python experiments/Phase4_ImageNet_LT.py --batch_size 512
```

---

## 📊 Results & Visualization
We provide a full suite of interactive Jupyter Notebooks in the `notebooks/` directory to visualize the Confusion Matrices, the Oracle Alpha Search, and the Neural Router's dynamic blending weights.
- `notebooks/DeiT_LT_ATF_Final.ipynb`: Complete visualization suite.

*(Visual figures and comprehensive result tables are available in the `figures/` and `docs/` directories).*

---

## 📝 Citation
If you find this observation useful in your research, please consider citing:
```bibtex
@article{vip2026tokenspecialization,
  title={Understanding Token Specialization in Distilled Vision Transformers under Severe Data Imbalance},
  author={Bharath VIP},
  journal={GitHub Repository},
  year={2026}
}
```

## 📬 Contact
**Bharath VIP**  
Research Project Repository  
*This research was inspired by the works of Prof. R. Venkatesh Babu's Vision and AI Lab (VAL) at IISc.*
