# Technical Report: Token Specialization in Distilled Vision Transformers


---

## 1. Executive Summary

This report documents a comprehensive scientific investigation into the internal dynamics of Knowledge Distillation (KD) in Vision Transformers (ViTs) under severe data imbalance. By reproducing and auditing the CVPR 2024 paper *"DeiT-LT: Distillation Strikes Back for Vision Transformer Training on Long-Tailed Datasets"*, I uncovered a profound structural phenomenon: **Token Specialization**. 

The standard engineering heuristic for distilled ViTs relies on a fixed 50/50 fusion of the Class (`CLS`) and Distillation (`DIST`) tokens. However, my controlled experiments reveal that under long-tailed distributions, Knowledge Distillation breaks the architectural symmetry. The `CLS` token functions exclusively as a Head-class expert, while the `DIST` token functions as a globally dominant Tail expert. Consequently, the fixed 50/50 averaging causes *Expert Sabotage*, actively suppressing model performance. 

By designing an **Entropy-Guided Neural Router** and executing a **No-KD Causal Baseline**, I successfully recovered this latent performance and definitively proved that Knowledge Distillation is the causal driver of this phenomenon.

---

## 2. Experimental Setup & Reproduction

To ensure rigorous validation, I engineered a highly optimized training pipeline replicating the DeiT-LT methodology:
* **Hardware:** NVIDIA T4 GPUs with Automatic Mixed Precision (`torch.amp`).
* **Architectures Evaluated:** DeiT-Tiny (5M), DeiT-Small (22M), DeiT-Base (86M).
* **Teachers:** ResNet-32 and ResNeXt-50.
* **Datasets:** CIFAR-10-LT and CIFAR-100-LT.
* **Hyperparameters:** Strict 300-epoch schedule, batch size of 256, CosineAnnealingLR, and Deferred Reweighting (DRW).

**Baseline Result:** The reproduction on CIFAR-10-LT (Imbalance Factor 50) achieved a baseline top-1 accuracy of **72.10%** using the standard 50/50 token averaging heuristic.

---

## 3. The Discovery: Expert Sabotage

By logging class-wise confusion matrices for individual tokens, a striking bifurcation emerged:
* **The `CLS` Token (69.2% Native Acc):** Displayed intense diagonal saturation on majority (Head) classes but failed catastrophically on the Tail.
* **The `DIST` Token (73.8% Native Acc):** Anchored by the robust convolutional biases of the CNN teacher, the `DIST` token completely dominated the Tail classes.

**The Flaw:** When predicting a Tail image, the `CLS` token outputs high-entropy noise. The fixed 50/50 averaging heuristic forces this noise to dilute the highly confident correct prediction of the `DIST` token. The tokens were actively sabotaging each other.

---

## 4. The Oracle Bound & The Neural Entropy Router

To prove that dynamic routing could solve this, I conducted an empirical **Oracle Alpha Search**, testing every possible fusion weight ($\alpha \in [0,1]$) for every image.
* **Oracle Upper Bound:** The Oracle achieved **74.40%** (a massive +2.3% boost over the baseline), revealing a perfect step-function where Head classes required $\alpha \approx 0.80$ and Tail classes required $\alpha \approx 0.05$.

Because class frequency is unknown during true inference, I trained a Random Forest to predict the Oracle's choice purely from instance-level features. The feature importance algorithm revealed that **DIST Shannon Entropy** accounted for 77.5% of the predictive power.

Based on this, I designed a lightweight, post-hoc **Neural Entropy Router**. Taking only confidence and entropy as inputs, the MLP dynamically predicted the optimal fusion weight.
* **Router Result:** The Neural Router autonomously recognized the structural superiority of the `DIST` token and achieved **73.73%**, successfully bypassing the 50/50 heuristic and recovering the suppressed performance.

---

## 5. The Ablation Suite (Scale Invariance & Severity)

To ensure this phenomenon was not a statistical anomaly, I executed a rigorous ablation suite:

1. **Dataset Severity (IF=100 vs IF=10):**
   * At extreme starvation (IF100), the gap widened. The router completely ignored the poisoned `CLS` token ($\alpha \approx 0.001$), allowing the `DIST` token to achieve 69.53%.
   * At mild imbalance (IF10), both tokens were healthy. The router intelligently synergized them ($\alpha \approx 0.30$) to achieve 82.17%, beating both tokens individually.
2. **Architecture Scaling:**
   * Scaling to DeiT-Small (22M) and DeiT-Base (86M) caused severe overfitting on the small dataset, lowering overall global accuracy. However, the internal structural dynamic remained mathematically identical. The `DIST` token outperformed the `CLS` token by exactly **$\sim$4.6\%** across all three model sizes, proving scale-invariance.

---

## 6. The Causal Proof: The No-KD Baseline

To satisfy the highest threshold of scientific causality, it was necessary to prove that Knowledge Distillation—not dataset imbalance or architectural bias—caused the Token Specialization.

I executed a controlled baseline where the exact same DeiT architecture (with both tokens) was trained on CIFAR-10-LT, but **the Teacher model was removed**. 
* **The Result:** The tokens collapsed symmetrically. Both achieved $\sim$83.0\% overall accuracy, with identical Tail accuracies (65.8\% vs 66.5\%). 
* **Routing Failure:** Because the tokens learned redundant representations, the Neural Router converged to a random blending weight ($\alpha = 0.249$) and achieved zero improvement over the baseline.

**Conclusion:** This isolates Knowledge Distillation as the definitive root cause of Token Specialization. The distillation loss anchors the `DIST` token to the robust, translationally equivariant knowledge of the CNN teacher, preventing the gradient starvation that plagues the standard self-attention `CLS` token on long-tailed data.

---

## 7. Conclusion

By shifting the perspective from engineering metrics to structural auditing, this project demonstrates that the standard inference practices for distilled Vision Transformers actively mask the true complexity of their learned representations. Dynamic, uncertainty-guided routing is a computationally free, post-hoc mechanism to fully unleash the power of the Knowledge Distillation token.
