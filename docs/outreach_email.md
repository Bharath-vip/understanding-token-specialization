# Outreach Email Draft

*Note: Do NOT ask for an internship in the first email. As ChatGPT pointed out, you want to lead with undeniable scientific value.*

---

**Subject:** Causal Proof of Token Specialization in Distilled ViTs (Inspired by VAL)

Dear Prof. Venkatesh Babu,

My name is Bharath, and I have been closely following the research emerging from the Vision and AI Lab (VAL), particularly your lab's work on long-tailed distributions and dataset distillation. 

Inspired by VAL's rigorous approach to vision architectures, I recently conducted a deep structural audit into how Vision Transformers behave under severe data starvation. Specifically, I investigated the inference heuristics of distilled architectures like DeiT-LT.

I discovered that the standard practice of statically averaging the `CLS` and `DIST` tokens (50/50 fusion) masks a profound internal dynamic: **under severe imbalance, Knowledge Distillation breaks the architectural symmetry of the network, forcing the tokens to bifurcate into mutually exclusive Head and Tail experts.**

Because the fixed 50/50 averaging actively causes "expert sabotage," I engineered an Adaptive Token Fusion (ATF) mechanism via a Neural Entropy Router that dynamically routes predictions based on instance-level confidence and Shannon entropy. Furthermore, I ran a strict No-KD causal ablation, proving definitively that Knowledge Distillation—not dataset imbalance alone—is the root cause of this bifurcation.

I have formalized these findings into a CVPR-style paper and a fully reproducible repository:
*   **Paper Draft & Supplementary Material:** [Link to PDF]
*   **GitHub Repository (Optimized Codebase & Interactive Notebooks):** https://github.com/Bharath-vip/understanding-token-specialization
*   **Presentation Summary:** [Link to Slides]

Given your lab's expertise in these architectures, I would be incredibly honored if you or your PhD students had a few moments to review this phenomenon. I believe this insight into internal Token Specialization could open new pathways for how we design robust representations in ViTs.

Thank you for your time, and for the continuous inspiration VAL provides to the computer vision community.

Best regards,

Bharath VIP
[Link to your LinkedIn/Portfolio]
