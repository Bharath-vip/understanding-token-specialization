# Presentation Outline (15 Slides)

*Speaker Notes: This presentation leverages the empirical data from the `ABLATION_LOG.md` to rigorously prove the Token Specialization phenomenon.*

---

### **Slide 1: Title Slide**
- **Title:** Understanding Token Specialization in Distilled Vision Transformers under Severe Data Imbalance
- **Presenter:** Bharath
- **Visuals:** Clean, minimal. ViT architecture diagram.

### **Slide 2: The Problem**
- **Core Issue:** Vision Transformers scale well but lack convolutional inductive biases, making them notoriously data-hungry. They fail catastrophically on long-tailed (highly imbalanced) datasets like CIFAR-100-LT or ImageNet-LT.
- **The Challenge:** Real-world data is inherently long-tailed. Models must learn the Tail without overfitting the Head.

### **Slide 3: The State-of-the-Art (DeiT-LT)**
- **The Solution:** Knowledge Distillation (KD). A robust CNN Teacher (e.g., ResNet-32) supervises a ViT Student (e.g., DeiT-Tiny).
- **The Mechanism:** Introduction of the Distillation (`DIST`) token alongside the Class (`CLS`) token.
- **The Inference Heuristic:** To predict a class, the authors rigidly average the logits: a static 50/50 fusion of `CLS` and `DIST`.

### **Slide 4: The Unexplored Question**
- **The Gap:** *Nobody asked what happens inside the `CLS` and `DIST` tokens under data starvation.* 
- **The Goal:** Are these tokens learning redundant features? Or does the 50/50 heuristic mask a deeper structural dynamic?

### **Slide 5: The Hypothesis**
- **Our Hypothesis:** Under severe imbalance, Knowledge Distillation fundamentally breaks the architectural symmetry of the network, forcing the tokens to bifurcate into mutually exclusive experts. 

### **Slide 6: The Evidence (Expert Sabotage)**
- **Observation:** (Include **Figure 3 - Confusion Matrices**). 
- **Finding:** At CIFAR-10-LT (IF=50), the `CLS` token functions as a Head Expert, saturating on majority classes. The `DIST` token functions as a Tail Expert.
- **The Consequence:** The 50/50 averaging heuristic causes "Expert Sabotage"—the Head expert drags down the Tail expert's predictions, and vice versa.

### **Slide 7: The Oracle**
- **The Experiment:** We empirically swept the fusion weight $\alpha$ (from 0.0 to 1.0) for every single sample to find the theoretical upper bound.
- **Visual:** (Include **Figure 4 - Oracle Curve**).
- **Result:** A mathematically perfect step-function exists! Head classes require $\alpha \approx 0.80$, while Tail classes require $\alpha \approx 0.05$. The Oracle bound was **74.40%** (vs the 72.10% baseline).

### **Slide 8: The Solution (Neural Entropy Router)**
- **The Method:** An Adaptive Token Fusion (ATF) mechanism.
- **How it works:** A lightweight MLP dynamically predicts the optimal fusion weight using instance-level *Confidence* and *Shannon Entropy*.
- **The Result:** The Neural Router achieved **73.73%**, nearly closing the massive Oracle Gap (+1.63% absolute boost over baseline) without retraining the backbone.

### **Slide 9: Ablation Suite - Severity (IF100 & IF10)**
- **IF100 (Extreme):** The gap between experts widened massively. Natively, `DIST` hit 69.5% while `CLS` collapsed to 63.8% (a 5.7% gap). The Router perfectly adapted, ignoring the CLS token ($\alpha \approx 0.001$) to achieve 69.53%.
- **IF10 (Mild):** The dataset is healthier. `CLS` (81.0%) caught up to `DIST` (81.5%). The Router intelligently recognized this and *blended* them ($\alpha \approx 0.30$), achieving a synergistic **82.17%**, beating both tokens individually!

### **Slide 10: Ablation Suite - Architecture Scaling**
- **Testing the Limits:** Does this hold on larger models? We tested DeiT-Small (22M params) and DeiT-Base (86M params).
- **The Overfitting Paradox:** Both larger models overfit the small dataset, resulting in lower global accuracy than DeiT-Tiny.
- **Scale-Invariant Dominance:** Despite overfitting, the structural gap remained perfectly mathematically identical. The `DIST` token outperformed the `CLS` token by **~4.6%** across Tiny, Small, and Base.

### **Slide 11: The Causal Proof (No-KD)**
- **The Counter-Argument:** Is this specialization caused by the dataset or by Knowledge Distillation?
- **The Experiment:** We trained the exact same architecture *without* the Teacher.
- **Result:** The tokens collapsed symmetrically. Both `CLS` and `DIST` achieved ~83.0%, with identical Tail accuracies (65.8% vs 66.5%). 
- **Conclusion:** Knowledge Distillation is definitively the causal driver of Token Specialization.

### **Slide 12: Discussion**
- **Why does DIST dominate the Tail?** 
- **Explanation:** The CNN teacher possesses strong translational equivariance, making it inherently more robust to data starvation. The `DIST` token anchors itself to this robust knowledge, while the `CLS` token (guided only by standard cross-entropy) overfits to the Head.

### **Slide 13: The Predictive Power of Entropy**
- **The 447 Images:** When analyzing where static class-frequency fusion failed but the Oracle succeeded, a Random Forest revealed the truth.
- **Finding:** Class frequency only accounted for a fraction of the predictive power. **DIST Entropy accounted for 77.5% of the predictive importance.** Uncertainty is the key to dynamic routing.

### **Slide 14: Limitations & Future Work**
- **Limitations:** Our routing mechanism operates purely at the logit level post-hoc during inference.
- **Future Work:** Can we explicitly encourage this Token Specialization during the *representation learning phase*? Exploring this phenomenon on massive datasets like ImageNet-LT.

### **Slide 15: Conclusion**
- **What did we learn?** The rigid 50/50 fusion heuristic masks a profound internal dynamic in distilled Vision Transformers. By dynamically routing based on entropy, we uncover the structural supremacy of the Knowledge Distillation token.
- **Questions?**
- GitHub: `https://github.com/Bharath/understanding-token-specialization`
