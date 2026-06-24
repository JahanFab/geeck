# Options Greeks Predictor

Train ML models to forecast how delta, gamma, and vega evolve under real market conditions (volatility smile) versus the flat-vol Black-Scholes assumption.

---

## Overview

Black-Scholes computes greeks assuming a single constant volatility across all strikes and maturities. In practice, implied volatility varies with strike (the smile) and maturity (the term structure). This project quantifies how much that matters and whether ML can capture the gap.

---

## How It Works

1. **Simulate options data** — 8,000 samples with a parameterized IV smile:
   - `IV(k) = base_vol + skew·k + convexity·k²` where `k = log(S/K)`
   - Negative skew (OTM puts more expensive than OTM calls)
   - Vol-of-vol noise layered on top
2. **Compute "true" greeks** at the market-implied vol (smile-adjusted)
3. **Compute "naive" greeks** at flat base vol — what a standard BS pricer would report
4. **Train three ML models** to predict true greeks from option features
5. **Compare** ML accuracy against the flat-vol BS baseline

---

## Features Used

| Feature | Description |
|---------|-------------|
| `S`, `K`, `T`, `r` | Option inputs |
| `moneyness` | S/K |
| `log_moneyness` | log(S/K) |
| `sqrt_T` | √T |
| `is_call` | 1 = call, 0 = put |
| `bs_delta/gamma/vega` | Flat-vol BS greeks (used as priors) |

---

## Models

| Model | Description |
|-------|-------------|
| **BS Baseline** | Flat-vol Black-Scholes (naive pricer) |
| **GBM** | Gradient Boosting (200 trees, depth 5) |
| **RF** | Random Forest (150 trees, depth 10) |
| **MLP** | Neural net (128→64→32, ReLU) |

---

## Results

| Greek | Model | R² | MAE |
|-------|-------|----|-----|
| Delta | BS Baseline | 0.9995 | 0.00710 |
| Delta | GBM | **0.9996** | **0.00691** |
| Delta | RF | 0.9995 | 0.00716 |
| Gamma | RF | **0.9828** | 0.00027 |
| Gamma | BS Baseline | 0.9819 | 0.00026 |
| Vega | BS Baseline | 0.9969 | 0.01130 |
| Vega | GBM | 0.9968 | 0.01381 |

Top feature for delta: `bs_delta` (importance 0.9998) — ML learns a smile correction on top of BS, not a full replacement.

---

## Key Insight

The flat-vol BS baseline already performs well (R² > 0.99) because the IV smile in this simulation produces small absolute deviations. The ML models add the most value in the tails (deep ITM/OTM) where the smile effect is largest. In real markets with steep skew regimes (e.g. equity index options), the gap widens significantly.

---

## Output Plots

| File | Description |
|------|-------------|
| `options_smile_effect.png` | BS vs market greeks across moneyness |
| `options_model_comparison.png` | R², MAE, RMSE by model and greek |
| `options_delta_predictions.png` | Predicted vs actual delta scatter |
| `options_gamma_predictions.png` | Predicted vs actual gamma scatter |
| `options_vega_predictions.png` | Predicted vs actual vega scatter |

---

## Usage

```bash
pip install numpy pandas scikit-learn matplotlib scipy
python options_greeks_predictor.py
```

---

## Dependencies

```
numpy
pandas
scikit-learn
matplotlib
scipy
```
