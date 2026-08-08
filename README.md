# P2-ETF-R# P2-RBF-KAN-TRANSFORMER

**RBF-KAN-Transformer — Hybrid Neural Architecture for Financial Forecasting**

Part of the **P2Quant Engine Suite** · P2SAMAPA

---

## What This Engine Does

This engine implements a **hybrid neural architecture** combining:

- **Radial Basis Functions (RBF)** for localized pattern recognition
- **Kolmogorov-Arnold Networks (KAN)** for interpretable non-linear feature learning
- **Transformer** for global context and attention

### Theory

**RBF Layer:**
- Gaussian kernels for localized pattern detection
- Learns centers that capture market regimes
- Excellent for non-stationary financial data

**KAN Layer:**
- Based on Kolmogorov-Arnold representation theorem
- Learnable spline functions instead of fixed activations
- More interpretable than standard MLPs

**Transformer:**
- Self-attention for global context
- Captures long-range dependencies
- Handles variable-length sequences

---

## Key Metrics

| Metric | What it tells you |
|--------|-------------------|
| **z-score** | Cross-sectional ranking of forecast strength |
| **Forecast** | Predicted future return |
| **Loss** | Training loss (lower = better) |
| **N Epochs** | Number of training epochs |

---

## Windows

| Window | Purpose |
|--------|---------|
| 63d | Short-term forecasting |
| 126d | Medium-term forecasting |
| 252d | Core signal (primary) |
| 504d | Long-term forecasting |

---

## Interpretation

| z-score | Action | Meaning |
|---------|--------|---------|
| **> 0.05** | BUY | Positive forecast |
| **-0.05 to 0.05** | HOLD | Neutral forecast |
| **< -0.05** | SELL | Negative forecast |

---

## Setup

```bash
git clone https://github.com/P2SAMAPA/P2-RBF-KAN-TRANSFORMER
cd P2-RBF-KAN-TRANSFORMER
pip install -r requirements.txt

export HF_TOKEN=hf_...
python trainer.py

streamlit run streamlit_app.py
GitHub Actions
Runs automatically at 00:30 UTC Monday–Saturday.

Required secret: HF_TOKEN

References
Liu, Z., et al. (2024). KAN: Kolmogorov-Arnold Networks. arXiv:2404.19756.

Vaswani, A., et al. (2017). Attention Is All You Need. NeurIPS.

Powell, M. J. D. (1987). Radial Basis Functions for Multivariable Interpolation. IMA Conference on Algorithms for the Approximation of Functions and Data.BF-KAN-TRANSFORMER
