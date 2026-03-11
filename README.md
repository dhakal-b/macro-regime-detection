# Macroeconomic Regime Detection & Systematic Asset Allocation
A Gaussian Hidden Markov Model trained on FRED macroeconomic indicators to detect hidden economic regimes and systematically rotate across asset classes.

---

## Abstract
Using 240 months of data from December 2004 to December 2024, we identify three distinct economic regimes — Expansion, Recovery, and Tightening — and analyze the performance of SPY, QQQ, TLT, and GLD within each regime. A regime-switching allocation strategy is backtested against SPY buy-and-hold and a 60/40 portfolio, with full out-of-sample validation on unseen 2019–2024 data.

---

## Results

| Metric | Regime Strategy | SPY Buy & Hold | 60/40 Portfolio |
|---|---|---|---|
| Annualized Return | **13.78%** | 10.28% | 7.97% |
| Annualized Volatility | **12.42%** | 15.19% | 10.22% |
| Sharpe Ratio | **0.948** | 0.545 | 0.584 |
| Max Drawdown | **-25.12%** | -50.78% | -28.40% |
| Calmar Ratio | **0.549** | 0.202 | 0.281 |

**Out-of-sample (2019–2024 — never seen during training):**

| Metric | Regime Strategy | SPY Buy & Hold | 60/40 Portfolio |
|---|---|---|---|
| Annualized Return | **17.73%** | 17.14% | 9.21% |
| Sharpe Ratio | **1.172** | 0.855 | 0.529 |
| Max Drawdown | **-19.22%** | -23.97% | -26.22% |

> OOS Sharpe of 1.172 is higher than in-sample Sharpe of 0.948 — a model that was overfitted would perform worse on unseen data, not better.

---

## Methodology

**Data** — 6 FRED macroeconomic indicators (CPI, unemployment, Fed Funds Rate, 10Y & 2Y Treasury yields, industrial production, consumer sentiment) + 4 ETFs (SPY, QQQ, TLT, GLD) at monthly frequency.

**Features** — First-difference transforms on trending indicators, yield spread (10Y minus 2Y) as recession signal, all features standardized with z-score normalization.

**Model** — Gaussian HMM, 3 hidden states, diagonal covariance, 2000 iterations. Seed 33 selected through systematic search as the only convergent solution with all states above 20 observations.

**Regimes** — States interpreted post-hoc against known historical episodes. Correctly identified GFC 2008, post-GFC recovery, 2015–2019 expansion, COVID 2020, and 2022 rate hike cycle with no supervised labels.

**Allocation** — Weights derived directly from per-regime Sharpe ratios. 0.1% transaction cost per regime change. Minimum 10% floor on all assets at all times.

**Validation** — Train on 2005–2018, test on 2019–2024. Model retrained from scratch on training data only.

---

## Stack
`Python` `hmmlearn` `FRED API` `yfinance` `scikit-learn` `pandas` `numpy` `matplotlib` `seaborn` `plotly`
