# Understanding Token Specialization in Distilled Vision Transformers under Severe Data Imbalance

**Abstract**
Vision Transformers (ViTs) are notoriously data-hungry and struggle significantly under long-tailed data distributions. Recent methods (e.g., DeiT-LT) employ Knowledge Distillation (KD) and a specialized Distillation (`DIST`) token to transfer robust representations from a CNN teacher to a ViT student. During inference, standard practice involves a static 50/50 averaging of the `CLS` and `DIST` tokens. In this paper, we conduct a deep structural audit of this heuristic. We uncover a profound scientific phenomenon: Knowledge Distillation fundamentally breaks the architectural symmetry of the ViT tokens under severe data scarcity, forcing the `CLS` and `DIST` tokens to bifurcate into mutually exclusive Head and Tail experts. We demonstrate that the 50/50 heuristic actively suppresses model performance by causing expert sabotage. To mitigate this, we introduce the Neural Entropy Router, a lightweight, class-agnostic routing mechanism that dynamically fuses tokens based on instance-level confidence and entropy. Finally, through a rigorous no-teacher causal ablation, we definitively prove that Knowledge Distillation—not architectural quirks or dataset imbalance alone—is the root cause of this Token Specialization.

## 1. Introduction
* **The Problem:** ViTs fail on Long-Tailed datasets.
* **The State of the Art:** DeiT-LT uses KD to solve this.
* **The Gap:** The inference phase relies on a rigid, unprincipled 50/50 token averaging heuristic.
* **Our Contribution:** 
    1. We empirically expose the Token Specialization phenomenon (CLS = Head Expert, DIST = Tail Expert).
    2. We propose the Neural Entropy Router to dynamically fuse tokens.
    3. We provide a causal proof linking Knowledge Distillation to token bifurcation.

## 2. Background and Related Work
* **Vision Transformers (ViTs) and DeiT:** Overview of the architecture and the introduction of the `DIST` token.
* **Long-Tailed Learning:** Strategies for handling extreme imbalance (e.g., Deferred Reweighting).
* **Knowledge Distillation in Imbalanced Settings:** How CNN teachers transfer "dark knowledge" to ViT students.

## 3. The Phenomenon: Expert Sabotage
* **Baseline Reproduction:** Setup on CIFAR-10-LT.
* **Confusion Matrix Analysis:** Visual evidence of the structural bifurcation (CLS saturating on Head, DIST saturating on Tail).
* **The Cost of Averaging:** How forced 50/50 averaging drags down the performance of both experts.

## 4. Methodology: Adaptive Token Fusion
* **The Oracle Search:** An empirical grid-search sweeping fusion weight $\alpha$ across classes, proving the existence of a mathematically perfect step-function routing policy.
* **Neural Entropy Router:** 
    * Input features: Softmax Confidence and Shannon Entropy from both tokens.
    * Architecture: A lightweight Multi-Layer Perceptron (MLP) generating a dynamic fusion weight $\alpha$.
    * Training: Self-supervised using the ground-truth labels on the validation set.

## 5. Experiments and Results
* **Setup:** CIFAR-10-LT (Imbalance Factors 100, 50, 10).
* **Router Performance:** The Neural Router perfectly mimics the Oracle step-function, recovering the suppressed performance across all severity levels.
* **Dynamic Dominance:** Analyzing how the routing policy fundamentally changes as dataset severity increases (e.g., heavy reliance on DIST at IF=100, balanced reliance at IF=10).

## 6. Causal Analysis: The No-KD Baseline
* **The Counter-Hypothesis:** Is token specialization an inherent property of the DeiT architecture or the long-tailed dataset?
* **The Experiment:** Training the DeiT architecture (with both tokens) on CIFAR-10-LT *without* the Teacher model.
* **Results:** The `DIST` token completely loses its global expertise. Both tokens collapse symmetrically on the Tail, and the Neural Router converges to a random 50/50 baseline.
* **Conclusion:** Knowledge Distillation is definitively proven as the causal driver of Token Specialization.

## 7. Conclusion
* Summary of findings.
* Implications for future Vision Transformer designs and Knowledge Distillation architectures.
* Future work (Scaling to ImageNet-LT, varying teacher architectures).
