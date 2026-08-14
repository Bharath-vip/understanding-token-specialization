# Dynamic Entropy Routing for Distilled Vision Transformers

![Problem Overview](Problem%20Overview.svg)

## 📌 Project Overview
This repository contains the code, data, and findings from an independent structural audit of inference dynamics in distilled Vision Transformers (ViTs) under severe data starvation.

Recent state-of-the-art approaches (such as DeiT-LT) successfully mitigate long-tailed data imbalances by leveraging Knowledge Distillation (KD). By applying tail-focused distillation, these methods intentionally train the Distillation (`DIST`) token to act as a minority-class expert, while the standard Class (`CLS`) token remains a majority-class expert. 

However, during inference, these architectures rely on a rigid heuristic: a static 50/50 averaging of the `CLS` and `DIST` token predictions. **In this work, we mathematically and empirically demonstrate that this fixed heuristic is fundamentally flawed.**

**Key Contributions:**
1. **Exposing Expert Sabotage:** We prove that while the tokens successfully specialize during training, fixed 50/50 fusion actively suppresses performance by forcing the high-entropy noise of the non-expert token to dilute the confident prediction of the true expert.
2. **Entropy-Guided Routing:** We demonstrate that instance-level uncertainty (Shannon Entropy) accounts for 77.5\% of predictive power required to bypass the baseline heuristic. We introduce the Neural Entropy Router to dynamically route tokens at inference, recovering massive latent performance.
3. **Universality of the Oracle Bound:** We utilize SciPy L-BFGS-B, PyTorch Adam, and Differential Evolution to prove the optimal routing geometry is mathematically independent of the optimization algorithm.
4. **Causal Proof:** Through a No-KD baseline, we provide strong empirical evidence that Knowledge Distillation—not inherent architectural bias—is the causal driver of this specialization.

---

## 📂 Repository Structure

The project is structured logically around the core experiments rather than file types, allowing researchers to explore the specific code, notebooks, and resulting data for each stage of the journey.

### `docs/`
Contains the formal scientific documentation.
* `paper.pdf` & `paper.tex`: The massive, 15-page CVPR-formatted journal manuscript detailing the mathematical frameworks, methodology, optimizer studies, and causal arguments.
* `PROJECT_JOURNEY.md`: A highly detailed developer log tracking the evolution of the research, optimizations, and ablation tests.
* `The_Story_of_Token_Specialization.md`: A highly engaging, narrative-driven story of the entire research journey, from the initial hypothesis to the final discoveries.
* `Technical_Report.md`: A highly readable, two-page executive summary of the project.
* `supplementary.md`: Detailed hyperparameters, hardware specifications, and exact reproduction environments.

### `experiments/`
Contains the monolithic scripts, interactive Jupyter Notebooks, raw logs, and visual figures (Confusion Matrices, Oracle Curves) grouped by their scientific objective.
* **`01_Baseline_Reproduction/`**: The fundamental reproduction of DeiT-LT on CIFAR-10-LT exposing the "Expert Sabotage".
* **`02_Adaptive_Token_Fusion/`**: The empirical Oracle Alpha search, the Multi-Optimizer Universality Study, and the implementation of the Neural Entropy Router.
* **`03_Causal_Proof_NoKD/`**: The definitive test proving that removing the Teacher model eliminates the token specialization.
* **`04_Architecture_Scaling/`**: Extensive ablation testing across DeiT-Small (22M), DeiT-Base (86M), and varying imbalance factors to prove scale-invariance.

---

## 📖 Methodology & Research Journey
For a visual overview of how this investigation progressed from an initial observation to a definitive causal proof, see the Research Journey flowchart:

![Research Journey](Research%20Journey.svg)

---

## 📬 Contact & Citation

**Bharath**  
Final Year B.Tech | Artificial Intelligence and Data Science  
*This independent research study was inspired by the CVPR 2024 works of Prof. R. Venkatesh Babu's Vision and AI Lab (VAL) at IISc.*

```bibtex
@article{bharath2026dynamicentropyrouting,
  title={Dynamic Entropy Routing for Distilled Vision Transformers under Severe Data Imbalance},
  author={Bharath},
  journal={GitHub Repository},
  year={2026}
}
```
