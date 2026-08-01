# Presentation Outline (15 Slides)

This presentation is designed to concisely and powerfully communicate our discovery to Prof. R. Venkatesh Babu and the Vision and AI Lab. The focus is on the scientific narrative: Observation $\to$ Proof.

---

### **Slide 1: Title Slide**
- **Title:** Understanding Token Specialization in Distilled Vision Transformers under Severe Data Imbalance
- **Presenter:** Bharath VIP
- **Visuals:** Clean, minimal. Perhaps a small icon of a ViT architecture.

### **Slide 2: The Problem**
- **Core Issue:** Vision Transformers lack inductive biases (like convolutions) and fail catastrophically when trained on long-tailed (highly imbalanced) datasets.
- **Why it matters:** Real-world data is inherently long-tailed. We need models that don't just memorize the majority classes.

### **Slide 3: The State-of-the-Art (DeiT-LT)**
- **The Solution:** Knowledge Distillation (KD). A robust CNN Teacher supervises a ViT Student.
- **The Mechanism:** Introduction of the Distillation (`DIST`) token alongside the Class (`CLS`) token.
- **The Inference Heuristic:** Rigidly average the predictions (50/50).

### **Slide 4: The Unexplored Question**
- **The Gap:** *Nobody asked what happens inside the `CLS` and `DIST` tokens under data starvation.* 
- **The Goal:** To look inside the black box. Are these tokens learning redundant features, or is something else happening?

### **Slide 5: The Hypothesis**
- **Our Hypothesis:** Under severe imbalance, Knowledge Distillation fundamentally breaks the architectural symmetry of the network, forcing the tokens to bifurcate into mutually exclusive experts.

### **Slide 6: The Evidence (Expert Sabotage)**
- **Observation:** (Include **Figure 3 - Confusion Matrices** here). 
- **Finding:** 
  - `CLS` acts exclusively as a Head Expert.
  - `DIST` acts exclusively as a Tail Expert.
- **The Consequence:** The 50/50 averaging heuristic causes "Expert Sabotage"—the Head expert drags down the Tail expert's predictions, and vice versa.

### **Slide 7: The Oracle**
- **The Experiment:** We empirically swept the fusion weight $\alpha$ for every sample to find the theoretical upper bound.
- **Visual:** (Include **Figure 4 - Oracle Curve** here).
- **Result:** A mathematically perfect step-function exists! Head classes need $\alpha \approx 0.8$, Tail classes need $\alpha \approx 0.05$.

### **Slide 8: The Solution (Neural Entropy Router)**
- **The Method:** An Adaptive Token Fusion (ATF) mechanism.
- **How it works:** A lightweight MLP that dynamically predicts the optimal fusion weight based on the token's instance-level *Confidence* and *Shannon Entropy*.
- **Visual:** (Include **Figure 1 - Pipeline** here).

### **Slide 9: Ablation & Scaling**
- **Testing the Limits:** We scaled the imbalance factor (IF=10, 50, 100) and the architecture size (Tiny, Small, Base).
- **Visual:** (Include **Figure 7 - Ablation Summary** here).
- **Finding:** The Router flawlessly matches the Oracle upper bound across all severities.

### **Slide 10: The Causal Proof (No-KD)**
- **The Counter-Argument:** Is this specialization caused by the dataset or by Knowledge Distillation?
- **The Experiment:** We trained the exact same architecture *without* the Teacher.
- **Result:** The `DIST` token completely loses its Tail expertise. Tokens collapse symmetrically.
- **Conclusion:** Knowledge Distillation is definitively the causal driver of Token Specialization.

### **Slide 11: Discussion**
- **Why does DIST dominate the Tail?** 
- **Explanation:** The CNN teacher possesses strong translational equivariance, making it inherently more robust to data starvation. The `DIST` token anchors itself to this robust knowledge, while the `CLS` token (guided only by standard cross-entropy) overfits to the Head.

### **Slide 12: Limitations**
- Our routing mechanism operates purely at the logit level during inference. 
- It relies on entropy heuristics, which can sometimes be miscalibrated on out-of-distribution (OOD) data.

### **Slide 13: Future Work**
- Can we explicitly encourage this Token Specialization during the *representation learning phase* rather than just managing it at inference?
- Scaling this detailed causal audit to ImageNet-LT.

### **Slide 14: Conclusion**
- **What did we learn?** The rigid 50/50 fusion heuristic masks a profound internal dynamic in distilled Vision Transformers. By embracing and dynamically routing between specialized tokens, we unlock significant latent performance.

### **Slide 15: Questions & Repository**
- "Thank you for your time."
- Link to the GitHub repository: `https://github.com/Bharath-vip/understanding-token-specialization`
