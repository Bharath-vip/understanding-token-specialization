# Understanding Token Specialization in Distilled Vision Transformers

![Problem Overview](Problem%20Overview.svg)

## 📌 Project Overview
This repository contains the code, data, and findings from an independent structural audit of Knowledge Distillation (KD) in Vision Transformers (ViTs), specifically focusing on long-tailed data distributions. 

The standard inference heuristic for distilled ViTs relies on a rigid 50/50 averaging of the Class (`CLS`) and Distillation (`DIST`) tokens. Through rigorous experimentation, we uncover that under severe data starvation, these tokens do not learn redundant features; rather, they bifurcate into mutually exclusive experts. 

**Key Findings:**
1. **Token Bifurcation:** The `CLS` token functions as a Head-class expert, while the `DIST` token dominates the Tail.
2. **Expert Sabotage:** The fixed 50/50 inference heuristic actively suppresses model performance by forcing the Head expert to drag down the Tail expert's predictions, and vice versa.
3. **Entropy-Guided Routing:** We demonstrate that instance-level uncertainty (Shannon Entropy) can successfully act as a class-agnostic routing mechanism to recover this suppressed performance.
4. **Causal Proof:** Through a No-KD baseline, we provide strong empirical evidence that Knowledge Distillation—not inherent architectural bias—is the causal driver of this specialization.

---

## 📂 Repository Structure

The project is structured logically around the core experiments rather than file types, allowing researchers to explore the specific code, notebooks, and resulting data for each stage of the journey.

### `docs/`
Contains the formal scientific documentation.
* `paper.pdf`: The complete, CVPR-formatted research paper detailing the methodology, results, and causal arguments.
* `Technical_Report.md`: A highly readable, two-page executive summary of the project.
* `supplementary.md`: Detailed hyperparameters, hardware specifications, and exact reproduction environments.

### `experiments/`
Contains the monolithic scripts, interactive Jupyter Notebooks, raw logs, and visual figures (Confusion Matrices, Oracle Curves) grouped by their scientific objective.
* **`01_Baseline_Reproduction/`**: The fundamental reproduction of DeiT-LT on CIFAR-10-LT exposing the token bifurcation.
* **`02_Adaptive_Token_Fusion/`**: The empirical Oracle Alpha search and the implementation of the Neural Entropy Router.
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
@article{bharath2026tokenspecialization,
  title={Understanding Token Specialization in Distilled Vision Transformers under Severe Data Imbalance},
  author={Bharath},
  journal={GitHub Repository},
  year={2026}
}
```
