"""
Options Greeks Predictor
Trains ML models to forecast how delta, gamma, and vega evolve under
different market conditions vs. Black-Scholes closed-form assumptions.
Demonstrates where BS breaks down and how neural nets capture the gaps.
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import norm
from scipy.optimize import brentq
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from dataclasses import dataclass
from typing import Tuple, Dict, List


# Black-Scholes Closed Form 

def bs_d1(S, K, T, r, sigma):
    return (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))


def bs_d2(S, K, T, r, sigma):
    return bs_d1(S, K, T, r, sigma) - sigma * np.sqrt(T)


def bs_price(S, K, T, r, sigma, option_type="call"):
    d1 = bs_d1(S, K, T, r, sigma)
    d2 = bs_d2(S, K, T, r, sigma)
    if option_type == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def bs_delta(S, K, T, r, sigma, option_type="call"):
    d1 = bs_d1(S, K, T, r, sigma)
    if option_type == "call":
        return norm.cdf(d1)
    return norm.cdf(d1) - 1


def bs_gamma(S, K, T, r, sigma):
    d1 = bs_d1(S, K, T, r, sigma)
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))


def bs_vega(S, K, T, r, sigma):
    d1 = bs_d1(S, K, T, r, sigma)
    return S * norm.pdf(d1) * np.sqrt(T) / 100   # per 1% vol move


def bs_theta(S, K, T, r, sigma, option_type="call"):
    d1 = bs_d1(S, K, T, r, sigma)
    d2 = bs_d2(S, K, T, r, sigma)
    term1 = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
    if option_type == "call":
        return (term1 - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
    return (term1 + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365


#  Realistic Option Data Simulator 
# Adds real-world deviations from BS:
#   - Volatility smile (IV varies with moneyness)
#   - Stochastic volatility effects (vol-of-vol)
#   - Jump diffusion price noise
#   - Skew (OTM puts have higher IV than OTM calls)


def implied_vol_smile(moneyness: np.ndarray, base_vol: float,
                      skew: float = -0.05, convexity: float = 0.15) -> np.ndarray:
    """Parameterised IV smile: quadratic in log-moneyness with negative skew."""
    log_m = np.log(moneyness)  # log(S/K)
    return base_vol + skew * log_m + convexity * log_m**2


def generate_options_dataset(n_samples: int = 8000, seed: int = 42) -> pd.DataFrame:
    """
    Simulate a realistic options dataset with market microstructure effects.
    Returns option features + 'true' greeks (which deviate from flat-vol BS).
    """
    rng = np.random.default_rng(seed)
    rows = []

    for _ in range(n_samples):
        S     = rng.uniform(50, 500)
        K     = S * rng.uniform(0.7, 1.3)       # moneyness range
        T     = rng.uniform(0.02, 2.0)           # 1 week to 2 years
        r     = rng.uniform(0.01, 0.07)
        base_vol = rng.uniform(0.10, 0.80)
        option_type = rng.choice(["call", "put"])

        moneyness = S / K
        # Market IV follows a smile + vol-of-vol noise
        iv = implied_vol_smile(np.array([moneyness]), base_vol)[0]
        iv = max(0.05, iv + rng.normal(0, 0.02))   # vol-of-vol noise

        # "True" greeks computed at the market-implied vol (not flat bs)
        try:
            true_delta = bs_delta(S, K, T, r, iv, option_type)
            true_gamma = bs_gamma(S, K, T, r, iv)
            true_vega  = bs_vega(S, K, T, r, iv)
            true_theta = bs_theta(S, K, T, r, iv, option_type)
            true_price = bs_price(S, K, T, r, iv, option_type)
        except Exception:
            continue




        # BS greeks computed at the *base* (flat) vol — what a naive pricer would give
        try:
            bs_delta_flat = bs_delta(S, K, T, r, base_vol, option_type)
            bs_gamma_flat = bs_gamma(S, K, T, r, base_vol)
            bs_vega_flat  = bs_vega(S, K, T, r, base_vol)
        except Exception:
            continue

        if not all(np.isfinite([true_delta, true_gamma, true_vega, true_theta, true_price])):
            continue

        # Input features for the ML model
        rows.append({
            "S": S,
            "K": K,
            "T": T,
            "r": r,
            "base_vol": base_vol,
            "moneyness": moneyness,
            "log_moneyness": np.log(moneyness),
            "sqrt_T": np.sqrt(T),
            "is_call": int(option_type == "call"),
            # What flat-vol BS predicts
            "bs_delta": bs_delta_flat,
            "bs_gamma": bs_gamma_flat,
            "bs_vega": bs_vega_flat,
            # Ground truth (market-implied-vol greeks)
            "true_delta": true_delta,
            "true_gamma": true_gamma,
            "true_vega": true_vega,
            "true_theta": true_theta,
            "market_iv": iv,
            "iv_spread": iv - base_vol,     # smile adjustment
        })

    return pd.DataFrame(rows)




#  Model Training 

FEATURES = [
    "S", "K", "T", "r", "base_vol", "moneyness", "log_moneyness",
    "sqrt_T", "is_call", "bs_delta", "bs_gamma", "bs_vega",
]


def train_greek_models(df: pd.DataFrame) -> Dict:
    """Train GBM, RF, and MLP models to predict each greek."""
    df = df.reset_index(drop=True)
    X = df[FEATURES].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    n = len(df)
    train_idx, test_idx = train_test_split(np.arange(n), test_size=0.2, random_state=42)
    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
    df_test = df.iloc[test_idx]

    results = {}
    for greek in ["delta", "gamma", "vega"]:
        y = df[f"true_{greek}"].values
        y_train, y_test = y[train_idx], y[test_idx]

        models = {
            "GBM":  GradientBoostingRegressor(n_estimators=200, max_depth=5,
                                               learning_rate=0.05, random_state=42),
            "RF":   RandomForestRegressor(n_estimators=150, max_depth=10,
                                           random_state=42, n_jobs=-1),
            "MLP":  MLPRegressor(hidden_layer_sizes=(128, 64, 32), max_iter=500,
                                  random_state=42, early_stopping=True),
        }

        greek_results = {"scaler": scaler, "df_test": df_test, "y_test": y_test}
        for name, model in models.items():
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            greek_results[name] = {
                "model": model,
                "pred": pred,
                "mae": mean_absolute_error(y_test, pred),
                "rmse": np.sqrt(mean_squared_error(y_test, pred)),
                "r2":  r2_score(y_test, pred),
            }
        # Baseline: flat-vol BS greek (same test order)
        bs_col = f"bs_{greek}"
        if bs_col in df_test.columns:
            bs_pred = df_test[bs_col].values
            greek_results["BS_baseline"] = {
                "pred": bs_pred,
                "mae":  mean_absolute_error(y_test, bs_pred),
                "rmse": np.sqrt(mean_squared_error(y_test, bs_pred)),
                "r2":   r2_score(y_test, bs_pred),
            }
        results[greek] = greek_results

    return results



# Feature Importance 
def get_feature_importance(results: Dict, greek: str) -> pd.Series:
    gbm = results[greek]["GBM"]["model"]
    return pd.Series(gbm.feature_importances_, index=FEATURES).sort_values(ascending=False)




# Visualization 
def plot_greek_predictions(results: Dict, greek: str, save_path: str = None):
    y_test = results[greek]["y_test"]
    models_to_plot = [k for k in ["GBM", "RF", "MLP", "BS_baseline"] if k in results[greek]]

    n = len(models_to_plot)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    fig.suptitle(f"Predicted vs Actual {greek.capitalize()} — Test Set",
                 fontsize=13, fontweight="bold")

    for ax, name in zip(axes if n > 1 else [axes], models_to_plot):
        pred = results[greek][name]["pred"]
        metrics = results[greek][name]
        ax.scatter(y_test, pred, alpha=0.2, s=5, color="#3498db")
        mn = min(y_test.min(), pred.min())
        mx = max(y_test.max(), pred.max())
        ax.plot([mn, mx], [mn, mx], "r--", linewidth=1.2)
        ax.set_xlabel(f"Actual {greek}")
        ax.set_ylabel(f"Predicted {greek}")
        ax.set_title(f"{name}\nR²={metrics['r2']:.3f}  MAE={metrics['mae']:.4f}")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Plot saved → {save_path}")
    plt.show()


def plot_smile_effect(df: pd.DataFrame, save_path: str = None):
    """Show how ML captures the vol smile correction to delta."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Vol Smile Effect on Greeks: BS vs Market", fontsize=13, fontweight="bold")

    calls = df[df["is_call"] == 1].copy()
    calls = calls[calls["T"].between(0.1, 0.5)]    # ~3-6 month options

    bins = pd.cut(calls["log_moneyness"], bins=20)
    grouped = calls.groupby(bins, observed=True)

    log_m_mid = grouped["log_moneyness"].mean()

    for ax, greek in zip(axes, ["delta", "gamma", "vega"]):
        bs_col = f"bs_{greek}" if greek != "theta" else None
        true_col = f"true_{greek}"

        ax.plot(log_m_mid, grouped[true_col].mean(),
                label="Market (smile IV)", color="#e74c3c", linewidth=2)
        if bs_col and bs_col in calls.columns:
            ax.plot(log_m_mid, grouped[bs_col].mean(),
                    label="Black-Scholes (flat vol)", color="#3498db",
                    linewidth=2, linestyle="--")
        ax.fill_between(
            log_m_mid,
            grouped[true_col].mean() - grouped[true_col].std(),
            grouped[true_col].mean() + grouped[true_col].std(),
            alpha=0.15, color="#e74c3c"
        )
        ax.set_xlabel("Log Moneyness (log S/K)")
        ax.set_ylabel(greek.capitalize())
        ax.set_title(f"{greek.capitalize()} vs Moneyness")
        ax.legend(fontsize=8)
        ax.axvline(0, color="black", lw=0.7, linestyle=":")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Plot saved → {save_path}")
    plt.show()







def plot_model_comparison(results: Dict, save_path: str = None):
    greeks = list(results.keys())
    model_names = [k for k in ["BS_baseline", "GBM", "RF", "MLP"] if k in results[greeks[0]]]

    metrics = ["r2", "mae", "rmse"]
    fig, axes = plt.subplots(len(greeks), len(metrics), figsize=(14, 4 * len(greeks)))
    fig.suptitle("Model Performance Comparison by Greek", fontsize=13, fontweight="bold")

    colors = {"BS_baseline": "#95a5a6", "GBM": "#2ecc71", "RF": "#3498db", "MLP": "#e74c3c"}

    for gi, greek in enumerate(greeks):
        for mi, metric in enumerate(metrics):
            ax = axes[gi][mi]
            vals = [results[greek][m][metric] for m in model_names if m in results[greek]]
            bar_colors = [colors[m] for m in model_names if m in results[greek]]
            ax.bar(model_names[:len(vals)], vals, color=bar_colors, alpha=0.85)
            ax.set_title(f"{greek.capitalize()} — {metric.upper()}", fontsize=9)
            ax.tick_params(axis="x", labelsize=7)
            if metric == "r2":
                ax.set_ylim(0, 1)
                ax.axhline(1.0, color="black", lw=0.7, linestyle="--", alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Plot saved → {save_path}")
    plt.show()




#  Main 

def run(n_samples: int = 8000, plot: bool = True):
    print("\n" + "="*60)
    print("  Options Greeks Predictor — ML vs Black-Scholes")
    print("="*60 + "\n")

    print(f"► Generating synthetic options dataset ({n_samples:,} samples) …")
    df = generate_options_dataset(n_samples=n_samples)
    print(f"  {len(df):,} valid samples generated")
    print(f"  Moneyness range: {df['moneyness'].min():.2f} – {df['moneyness'].max():.2f}")
    print(f"  IV spread (smile adj) mean: {df['iv_spread'].mean():.4f}, "
          f"std: {df['iv_spread'].std():.4f}")

    print("\n► Training models for delta, gamma, vega …")
    results = train_greek_models(df)

    print("\n► Performance Summary (Test Set)")
    print(f"{'Greek':<8} {'Model':<14} {'R2':>7} {'MAE':>10} {'RMSE':>10}")
    print("─" * 55)
    for greek in ["delta", "gamma", "vega"]:
        for name in ["BS_baseline", "GBM", "RF", "MLP"]:
            if name not in results[greek]:
                continue
            m = results[greek][name]
            print(f"{greek:<8} {name:<14} {m['r2']:>7.4f} {m['mae']:>10.5f} {m['rmse']:>10.5f}")
        print()

    print("► Feature Importances (GBM for Delta):")
    fi = get_feature_importance(results, "delta")
    for feat, imp in fi.head(6).items():
        print(f"  {feat:20s}: {imp:.4f}")

    if plot:
        print("\n► Generating plots …")
        plot_smile_effect(df, save_path="options_smile_effect.png")
        plot_model_comparison(results, save_path="options_model_comparison.png")
        for greek in ["delta", "gamma", "vega"]:
            plot_greek_predictions(results, greek,
                                   save_path=f"options_{greek}_predictions.png")

    return results, df


if __name__ == "__main__":
    run(n_samples=8000, plot=True)
