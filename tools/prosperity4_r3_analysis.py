"""
IMC Prosperity 4 – Round 3 Analysis Tool
==========================================
Products in this round:
  - HYDROGEL_PACK       : stable ~10000, market-make focus
  - VELVETFRUIT_EXTRACT : underlying asset for VEV options (~5250)
  - VEV_4000 … VEV_6500 : call options on VELVETFRUIT_EXTRACT

USAGE
-----
  1. Place this file in the same folder as the 6 CSV files, OR
     edit DATA_DIR below to point at the CSVs.
  2. Run:   python prosperity4_r3_analysis.py
  3. An  r3_analysis/  folder is created with 9 PNG charts + iv_summary.csv.
  4. Attach all PNGs + the CSV to Claude for strategy recommendations.

DEPENDENCIES
------------
  pip install pandas numpy scipy matplotlib
"""

# ─────────────────────────────────────────────────────────
#  CONFIG  –  edit these if needed
# ─────────────────────────────────────────────────────────
DATA_DIR   = "."          # folder with the 6 round-3 CSV files
OUTPUT_DIR = "r3_analysis"

TOTAL_DAYS    = 3          # days 0, 1, 2
TICKS_PER_DAY = 1_000_000

# How many ticks to skip when computing IV (higher = faster but less resolution)
# 100 means ~10 000 rows per product per day — runs in ~1-2 min on most laptops
IV_SUBSAMPLE = 100

STRIKES = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
TRADEABLE_STRIKES = [5000, 5100, 5200, 5300, 5400, 5500]   # near-ATM options

STRIKE_COLORS = {
    4000: "#1f77b4", 4500: "#ff7f0e", 5000: "#2ca02c",
    5100: "#d62728", 5200: "#9467bd", 5300: "#8c564b",
    5400: "#e377c2", 5500: "#7f7f7f", 6000: "#bcbd22", 6500: "#17becf"
}
# ─────────────────────────────────────────────────────────

import os, warnings, time
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
os.makedirs(OUTPUT_DIR, exist_ok=True)

t0 = time.time()


# ════════════════════════════════════════════════════════
#  BLACK-SCHOLES HELPERS
# ════════════════════════════════════════════════════════

def bs_call(S, K, T, sigma, r=0.0):
    if T <= 0 or sigma <= 0 or S <= 0:
        return float(max(S - K, 0.0))
    d1 = (np.log(S / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return float(S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))

def bs_delta(S, K, T, sigma):
    if T <= 0 or sigma <= 0 or S <= 0:
        return 1.0 if S > K else 0.0
    d1 = (np.log(S / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    return float(norm.cdf(d1))

def bs_gamma(S, K, T, sigma):
    if T <= 0 or sigma <= 0 or S <= 0:
        return 0.0
    d1 = (np.log(S / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    return float(norm.pdf(d1) / (S * sigma * np.sqrt(T)))

def bs_vega(S, K, T, sigma):
    if T <= 0 or sigma <= 0 or S <= 0:
        return 0.0
    d1 = (np.log(S / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    return float(S * norm.pdf(d1) * np.sqrt(T))

def implied_vol(market_price, S, K, T):
    """Compute IV via Brent.  Returns NaN on failure."""
    intrinsic = max(S - K, 0.0)
    if market_price <= intrinsic + 0.5 or T <= 0 or S <= 0:
        return np.nan
    def obj(sigma):
        return bs_call(S, K, T, sigma) - market_price
    try:
        lo, hi = 0.001, 8.0
        if obj(lo) * obj(hi) > 0:
            return np.nan
        return brentq(obj, lo, hi, xtol=1e-7, maxiter=100)
    except Exception:
        return np.nan

def tte_years(day, timestamp):
    """Fraction of total round remaining (options expire at end of last day)."""
    ticks_remaining = (TOTAL_DAYS - day) * TICKS_PER_DAY - timestamp
    total_ticks     = TOTAL_DAYS * TICKS_PER_DAY
    return max(ticks_remaining / total_ticks, 1e-9)


# ════════════════════════════════════════════════════════
#  LOAD DATA
# ════════════════════════════════════════════════════════

def load_all():
    pf, tf = [], []
    for day in range(TOTAL_DAYS):
        p = pd.read_csv(f"{DATA_DIR}/prices_round_3_day_{day}.csv", sep=";")
        p["day"] = day
        p["global_ts"] = day * TICKS_PER_DAY + p["timestamp"]
        pf.append(p)
        t = pd.read_csv(f"{DATA_DIR}/trades_round_3_day_{day}.csv", sep=";")
        t["day"] = day
        t["global_ts"] = day * TICKS_PER_DAY + t["timestamp"]
        tf.append(t)
    return pd.concat(pf, ignore_index=True), pd.concat(tf, ignore_index=True)

print("Loading CSVs …")
prices, trades = load_all()

hydro   = prices[prices["product"] == "HYDROGEL_PACK"].copy()
vfe     = prices[prices["product"] == "VELVETFRUIT_EXTRACT"].copy()
vev_all = prices[prices["product"].str.startswith("VEV_")].copy()
vev_all["strike"] = vev_all["product"].str.replace("VEV_", "").astype(int)

# Build fast underlying price series for merging
vfe_sorted = vfe[["global_ts", "mid_price"]].sort_values("global_ts").set_index("global_ts")

def get_S(global_ts):
    idx = vfe_sorted.index.searchsorted(global_ts, side="right") - 1
    idx = max(0, min(idx, len(vfe_sorted) - 1))
    return vfe_sorted.iloc[idx]["mid_price"]


# ════════════════════════════════════════════════════════
#  COMPUTE IMPLIED VOLATILITY (sub-sampled)
# ════════════════════════════════════════════════════════

print(f"Computing IV (sub-sample every {IV_SUBSAMPLE} ticks) – may take 1-2 min …")

vev_sub = vev_all.iloc[::IV_SUBSAMPLE].copy().reset_index(drop=True)
vev_sub["S"]   = vev_sub["global_ts"].map(get_S)
vev_sub["TTE"] = vev_sub.apply(lambda r: tte_years(r["day"], r["timestamp"]), axis=1)

ivs, deltas, gammas, vegas, bs_prices = [], [], [], [], []
for _, row in vev_sub.iterrows():
    iv = implied_vol(row["mid_price"], row["S"], row["strike"], row["TTE"])
    ivs.append(iv)
    if np.isnan(iv):
        deltas.append(np.nan); gammas.append(np.nan)
        vegas.append(np.nan);  bs_prices.append(np.nan)
    else:
        bs_prices.append(bs_call(row["S"], row["strike"], row["TTE"], iv))
        deltas.append(bs_delta(row["S"], row["strike"], row["TTE"], iv))
        gammas.append(bs_gamma(row["S"], row["strike"], row["TTE"], iv))
        vegas.append(bs_vega(row["S"], row["strike"], row["TTE"], iv))

vev_sub["IV"]       = ivs
vev_sub["BS_price"] = bs_prices
vev_sub["delta"]    = deltas
vev_sub["gamma"]    = gammas
vev_sub["vega"]     = vegas
vev_sub["price_diff"] = vev_sub["mid_price"] - vev_sub["BS_price"]

print(f"  IV done ({time.time()-t0:.0f}s)")

# ── Fit quadratic smile per day ──────────────────────────
smile_params = {}
for day in range(TOTAL_DAYS):
    d = vev_sub[(vev_sub["day"] == day)].dropna(subset=["IV"])
    d = d[(d["IV"] > 0.01) & (d["IV"] < 5.0)]
    med = d.groupby("strike")["IV"].median().reset_index()
    if len(med) >= 3:
        smile_params[day] = np.polyfit(med["strike"], med["IV"], 2)

# Compute mispricing against smile
def fitted_iv(strike, day):
    if day not in smile_params:
        return np.nan
    v = np.polyval(smile_params[day], strike)
    return v if v > 0 else np.nan

vev_sub["fitted_IV"] = vev_sub.apply(lambda r: fitted_iv(r["strike"], r["day"]), axis=1)
vev_sub["fitted_BS"] = vev_sub.apply(
    lambda r: bs_call(r["S"], r["strike"], r["TTE"], r["fitted_IV"])
    if not np.isnan(r.get("fitted_IV", np.nan)) and r["fitted_IV"] > 0 else np.nan, axis=1
)
vev_sub["misprice"] = vev_sub["mid_price"] - vev_sub["fitted_BS"]


# ════════════════════════════════════════════════════════
#  HELPER
# ════════════════════════════════════════════════════════

def savefig(name):
    path = os.path.join(OUTPUT_DIR, name)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"  → {path}")


# ════════════════════════════════════════════════════════
#  CHART 1 – HYDROGEL_PACK overview
# ════════════════════════════════════════════════════════
print("\n[1/9] HYDROGEL_PACK overview")
fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
fig.suptitle("HYDROGEL_PACK – Market Overview (all days)", fontsize=14, fontweight="bold")

ax = axes[0]
for day in range(TOTAL_DAYS):
    d = hydro[hydro["day"] == day]
    ax.plot(d["global_ts"] / 1e6, d["mid_price"], label=f"Day {day}", lw=0.7)
ax.set_ylabel("Mid Price"); ax.legend(fontsize=8); ax.set_title("Mid Price")

ax = axes[1]
for day in range(TOTAL_DAYS):
    d = hydro[hydro["day"] == day].copy()
    d["spread"] = d["ask_price_1"] - d["bid_price_1"]
    ax.plot(d["global_ts"] / 1e6, d["spread"], label=f"Day {day}", lw=0.7)
ax.set_ylabel("Spread"); ax.legend(fontsize=8); ax.set_title("Best Bid-Ask Spread")

ax = axes[2]
for day in range(TOTAL_DAYS):
    d = hydro[hydro["day"] == day].copy()
    ax.plot(d["global_ts"] / 1e6, d["mid_price"] - 10000, label=f"Day {day}", lw=0.7)
ax.axhline(0, color="k", lw=0.5, ls="--")
ax.set_ylabel("Mid − 10,000"); ax.set_xlabel("Global Timestamp (×10⁶)")
ax.legend(fontsize=8); ax.set_title("Deviation from 10,000 (fair value)")

plt.tight_layout(); savefig("01_hydrogel_pack_overview.png")


# ════════════════════════════════════════════════════════
#  CHART 2 – HYDROGEL_PACK statistics
# ════════════════════════════════════════════════════════
print("[2/9] HYDROGEL_PACK statistics")
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
fig.suptitle("HYDROGEL_PACK – Statistical Analysis", fontsize=13, fontweight="bold")

mp = hydro["mid_price"]
axes[0].hist(mp, bins=60, color="steelblue", edgecolor="white")
axes[0].axvline(mp.mean(), color="red", ls="--", label=f"mean={mp.mean():.1f}")
axes[0].set_title("Mid-Price Distribution"); axes[0].legend()

ret = mp.diff().dropna()
axes[1].hist(ret, bins=60, color="coral", edgecolor="white")
axes[1].axvline(0, color="k", ls="--"); axes[1].set_title("Tick Returns (ΔMid)")

from pandas.plotting import autocorrelation_plot
autocorrelation_plot(ret.head(5000), ax=axes[2])
axes[2].set_title("Autocorrelation of Returns"); axes[2].set_xlim(0, 100)

plt.tight_layout(); savefig("02_hydrogel_pack_stats.png")


# ════════════════════════════════════════════════════════
#  CHART 3 – VELVETFRUIT_EXTRACT overview
# ════════════════════════════════════════════════════════
print("[3/9] VELVETFRUIT_EXTRACT overview")
fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
fig.suptitle("VELVETFRUIT_EXTRACT – Market Overview (all days)", fontsize=14, fontweight="bold")

ax = axes[0]
for day in range(TOTAL_DAYS):
    d = vfe[vfe["day"] == day]
    ax.plot(d["global_ts"] / 1e6, d["mid_price"], label=f"Day {day}", lw=0.7)
ax.set_ylabel("Mid Price"); ax.legend(fontsize=8); ax.set_title("Mid Price (underlying for VEV options)")

ax = axes[1]
for day in range(TOTAL_DAYS):
    d = vfe[vfe["day"] == day].copy()
    d["spread"] = d["ask_price_1"] - d["bid_price_1"]
    ax.plot(d["global_ts"] / 1e6, d["spread"], label=f"Day {day}", lw=0.7)
ax.set_ylabel("Spread"); ax.legend(fontsize=8); ax.set_title("Best Bid-Ask Spread")

ax = axes[2]
for day in range(TOTAL_DAYS):
    d = vfe[vfe["day"] == day].copy()
    d["rvol"] = d["mid_price"].diff().rolling(200).std() * np.sqrt(1e6)
    ax.plot(d["global_ts"] / 1e6, d["rvol"], label=f"Day {day}", lw=0.7)
ax.set_ylabel("Realised Vol (annualised proxy)"); ax.set_xlabel("Global Timestamp (×10⁶)")
ax.legend(fontsize=8); ax.set_title("Rolling Realised Volatility (200-tick window)")

plt.tight_layout(); savefig("03_velvetfruit_extract_overview.png")


# ════════════════════════════════════════════════════════
#  CHART 4 – VEV option mid-prices
# ════════════════════════════════════════════════════════
print("[4/9] VEV option mid-prices")
fig, axes = plt.subplots(2, 5, figsize=(20, 8))
fig.suptitle("VEV Options – Mid Price History (all days)", fontsize=14, fontweight="bold")
axes = axes.flatten()

for i, strike in enumerate(STRIKES):
    ax = axes[i]
    d = vev_all[vev_all["strike"] == strike]
    for day in range(TOTAL_DAYS):
        dd = d[d["day"] == day]
        ax.plot(dd["global_ts"] / 1e6, dd["mid_price"], label=f"Day {day}", lw=0.6, color=f"C{day}")
    ax.set_title(f"VEV_{strike}", fontsize=10)
    ax.set_xlabel("TS (×10⁶)", fontsize=8); ax.set_ylabel("Mid", fontsize=8)
    ax.legend(fontsize=7)

plt.tight_layout(); savefig("04_vev_midprices.png")


# ════════════════════════════════════════════════════════
#  CHART 5 – IV Smile per day
# ════════════════════════════════════════════════════════
print("[5/9] IV Smile per day")
fig, axes = plt.subplots(1, TOTAL_DAYS, figsize=(16, 5), sharey=True)
fig.suptitle("Implied Volatility Smile by Day\n(dots = IV samples, line = quadratic fit, red dots = median IV per strike)",
             fontsize=12, fontweight="bold")

for day in range(TOTAL_DAYS):
    ax = axes[day]
    d = vev_sub[vev_sub["day"] == day].dropna(subset=["IV"])
    d = d[(d["IV"] > 0.01) & (d["IV"] < 5.0)]

    sc = ax.scatter(d["strike"], d["IV"], c=d["global_ts"], cmap="viridis",
                    s=3, alpha=0.3, linewidths=0)
    med = d.groupby("strike")["IV"].median().reset_index()
    ax.scatter(med["strike"], med["IV"], color="red", s=50, zorder=5, label="Median IV")

    if day in smile_params:
        xs = np.linspace(STRIKES[0], STRIKES[-1], 300)
        ax.plot(xs, np.polyval(smile_params[day], xs), "r-", lw=2, label="Quad fit")

    ax.set_title(f"Day {day}"); ax.set_xlabel("Strike")
    if day == 0: ax.set_ylabel("Implied Volatility")
    ax.legend(fontsize=8)
    plt.colorbar(sc, ax=ax, label="Global TS")

plt.tight_layout(); savefig("05_iv_smile_per_day.png")


# ════════════════════════════════════════════════════════
#  CHART 6 – IV over time per strike
# ════════════════════════════════════════════════════════
print("[6/9] IV over time per strike")
fig, axes = plt.subplots(2, 3, figsize=(16, 8))
fig.suptitle("Implied Volatility Over Time (near-ATM strikes)", fontsize=13, fontweight="bold")
axes = axes.flatten()

for i, strike in enumerate(TRADEABLE_STRIKES):
    ax = axes[i]
    d = vev_sub[(vev_sub["strike"] == strike)].dropna(subset=["IV"])
    d = d[(d["IV"] > 0.01) & (d["IV"] < 5.0)]
    for day in range(TOTAL_DAYS):
        dd = d[d["day"] == day]
        # Rolling mean to smooth
        ax.plot(dd["global_ts"] / 1e6, dd["IV"].rolling(5, center=True).mean(),
                label=f"Day {day}", lw=1.0, color=f"C{day}")
    ax.set_title(f"VEV_{strike}", fontsize=10)
    ax.set_xlabel("TS (×10⁶)", fontsize=8); ax.set_ylabel("IV", fontsize=8)
    ax.legend(fontsize=7)

plt.tight_layout(); savefig("06_iv_over_time.png")


# ════════════════════════════════════════════════════════
#  CHART 7 – Mispricing (market vs fitted smile)
# ════════════════════════════════════════════════════════
print("[7/9] Mispricing vs fitted smile")
fig, axes = plt.subplots(2, 3, figsize=(16, 8))
fig.suptitle("Option Mispricing: Market Price − Fitted-Smile BS Price\n"
             "(positive = market expensive vs smile → sell; negative → buy)",
             fontsize=12, fontweight="bold")
axes = axes.flatten()

for i, strike in enumerate(TRADEABLE_STRIKES):
    ax = axes[i]
    d = vev_sub[vev_sub["strike"] == strike].dropna(subset=["misprice"])
    for day in range(TOTAL_DAYS):
        dd = d[d["day"] == day]
        ax.plot(dd["global_ts"] / 1e6,
                dd["misprice"].rolling(5, center=True).mean(),
                label=f"Day {day}", lw=0.9, color=f"C{day}")
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_title(f"VEV_{strike}", fontsize=10)
    ax.set_xlabel("TS (×10⁶)", fontsize=8); ax.set_ylabel("Misprice", fontsize=8)
    ax.legend(fontsize=7)

plt.tight_layout(); savefig("07_mispricing.png")


# ════════════════════════════════════════════════════════
#  CHART 8 – Greeks over time
# ════════════════════════════════════════════════════════
print("[8/9] Greeks over time")
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle("Option Greeks Over Time", fontsize=13, fontweight="bold")

# Delta
ax = axes[0, 0]
for strike in TRADEABLE_STRIKES:
    d = vev_sub[vev_sub["strike"] == strike].dropna(subset=["delta"])
    ax.plot(d["global_ts"] / 1e6, d["delta"],
            label=f"K={strike}", lw=0.7, color=STRIKE_COLORS.get(strike))
ax.set_title("Delta (Δ)"); ax.set_ylabel("Delta"); ax.legend(fontsize=7)

# Gamma
ax = axes[0, 1]
for strike in TRADEABLE_STRIKES:
    d = vev_sub[vev_sub["strike"] == strike].dropna(subset=["gamma"])
    ax.plot(d["global_ts"] / 1e6, d["gamma"],
            label=f"K={strike}", lw=0.7, color=STRIKE_COLORS.get(strike))
ax.set_title("Gamma (Γ)"); ax.set_ylabel("Gamma"); ax.legend(fontsize=7)

# Vega
ax = axes[1, 0]
for strike in TRADEABLE_STRIKES:
    d = vev_sub[vev_sub["strike"] == strike].dropna(subset=["vega"])
    ax.plot(d["global_ts"] / 1e6, d["vega"],
            label=f"K={strike}", lw=0.7, color=STRIKE_COLORS.get(strike))
ax.set_title("Vega (ν)"); ax.set_ylabel("Vega"); ax.legend(fontsize=7)

# IV decay vs TTE (theta proxy)
ax = axes[1, 1]
for day in range(TOTAL_DAYS):
    d = vev_sub[(vev_sub["day"] == day) & (vev_sub["strike"] == 5200)].dropna(subset=["IV"])
    d = d[(d["IV"] > 0.01) & (d["IV"] < 5.0)]
    ax.scatter(d["TTE"], d["IV"], s=2, alpha=0.4, label=f"Day {day} K=5200", color=f"C{day}")
ax.set_title("IV vs TTE (ATM strike=5200, theta decay)"); ax.set_xlabel("TTE (fraction of round)")
ax.set_ylabel("IV"); ax.legend(fontsize=7)

plt.tight_layout(); savefig("08_greeks.png")


# ════════════════════════════════════════════════════════
#  CHART 9 – Trades analysis
# ════════════════════════════════════════════════════════
print("[9/9] Trades analysis")
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle("Trade Data – Bot Behaviour & Patterns", fontsize=13, fontweight="bold")

# VFE trades
ax = axes[0, 0]
vfe_t = trades[trades["symbol"] == "VELVETFRUIT_EXTRACT"]
for day in range(TOTAL_DAYS):
    dd = vfe_t[vfe_t["day"] == day]
    ax.scatter(dd["global_ts"] / 1e6, dd["price"], s=dd["quantity"] * 3,
               alpha=0.5, label=f"Day {day}", color=f"C{day}")
ax.set_title("VFE Trades (size ∝ qty)"); ax.set_xlabel("TS (×10⁶)"); ax.set_ylabel("Price")
ax.legend(fontsize=8)

# HYDROGEL trades
ax = axes[0, 1]
hp_t = trades[trades["symbol"] == "HYDROGEL_PACK"]
for day in range(TOTAL_DAYS):
    dd = hp_t[hp_t["day"] == day]
    ax.scatter(dd["global_ts"] / 1e6, dd["price"], s=dd["quantity"] * 3,
               alpha=0.5, label=f"Day {day}", color=f"C{day}")
ax.set_title("HYDROGEL_PACK Trades"); ax.set_xlabel("TS (×10⁶)"); ax.set_ylabel("Price")
ax.legend(fontsize=8)

# VEV trade volume per strike
ax = axes[1, 0]
vev_t = trades[trades["symbol"].str.startswith("VEV_", na=False)].copy()
vev_t["strike"] = vev_t["symbol"].str.replace("VEV_", "").astype(int)
cnt = vev_t.groupby(["day", "strike"])["quantity"].sum().unstack("day").fillna(0)
cnt.plot(kind="bar", ax=ax, width=0.7)
ax.set_title("Total Traded Volume per VEV Strike & Day")
ax.set_xlabel("Strike"); ax.set_ylabel("Total Qty"); ax.legend(title="Day", fontsize=8)

# Trade price vs BS price per strike scatter
ax = axes[1, 1]
merged_tv = pd.merge_asof(
    vev_t.sort_values("global_ts"),
    vev_sub[["global_ts", "strike", "BS_price"]].dropna().sort_values("global_ts"),
    on="global_ts", by="strike", direction="nearest"
)
merged_tv["trade_vs_bs"] = merged_tv["price"] - merged_tv["BS_price"]
for day in range(TOTAL_DAYS):
    dd = merged_tv[merged_tv["day"] == day]
    ax.scatter(dd["strike"], dd["trade_vs_bs"], alpha=0.5,
               label=f"Day {day}", color=f"C{day}", s=18)
ax.axhline(0, color="k", lw=0.7, ls="--")
ax.set_title("Actual Trade Price − BS Price"); ax.set_xlabel("Strike"); ax.set_ylabel("Trade − BS")
ax.legend(fontsize=8)

plt.tight_layout(); savefig("09_trades_analysis.png")


# ════════════════════════════════════════════════════════
#  BONUS CHART – IV Heatmap (strike × time)
# ════════════════════════════════════════════════════════
print("[+] IV heatmap by strike × time")
fig, axes = plt.subplots(1, TOTAL_DAYS, figsize=(18, 5), sharey=True)
fig.suptitle("IV Heatmap: Strike vs Time (darker = higher IV)", fontsize=13, fontweight="bold")

for day in range(TOTAL_DAYS):
    ax = axes[day]
    d = vev_sub[(vev_sub["day"] == day)].dropna(subset=["IV"])
    d = d[(d["IV"] > 0.01) & (d["IV"] < 5.0)]

    # Pivot: rows = strike, cols = timestamp buckets
    d["ts_bucket"] = pd.cut(d["global_ts"], bins=50, labels=False)
    piv = d.groupby(["strike", "ts_bucket"])["IV"].mean().unstack("ts_bucket")
    im = ax.imshow(piv.values, aspect="auto", cmap="plasma",
                   extent=[0, 50, STRIKES[-1], STRIKES[0]], origin="upper")
    ax.set_title(f"Day {day}"); ax.set_xlabel("Time bucket (0=start, 50=end)")
    if day == 0: ax.set_ylabel("Strike")
    ax.set_yticks(STRIKES); ax.set_yticklabels(STRIKES, fontsize=7)
    plt.colorbar(im, ax=ax, label="IV")

plt.tight_layout(); savefig("10_iv_heatmap.png")


# ════════════════════════════════════════════════════════
#  BONUS CHART – Smile stability (IV per strike across days)
# ════════════════════════════════════════════════════════
print("[+] Smile stability across days")
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_title("IV Smile Comparison Across Days (median per strike)", fontsize=13, fontweight="bold")

for day in range(TOTAL_DAYS):
    d = vev_sub[(vev_sub["day"] == day)].dropna(subset=["IV"])
    d = d[(d["IV"] > 0.01) & (d["IV"] < 5.0)]
    med = d.groupby("strike")["IV"].median()
    ax.plot(med.index, med.values, "o-", label=f"Day {day}", lw=2, markersize=6)
    if day in smile_params:
        xs = np.linspace(STRIKES[0], STRIKES[-1], 200)
        ax.plot(xs, np.polyval(smile_params[day], xs), ls="--", color=f"C{day}", lw=1, alpha=0.6)

ax.set_xlabel("Strike"); ax.set_ylabel("Implied Volatility")
ax.legend(title="Day"); ax.grid(True, alpha=0.3)
plt.tight_layout(); savefig("11_smile_stability.png")


# ════════════════════════════════════════════════════════
#  SUMMARY CSV
# ════════════════════════════════════════════════════════
print("\nWriting summary CSV …")
rows = []
for day in range(TOTAL_DAYS):
    for strike in STRIKES:
        d = vev_sub[(vev_sub["day"] == day) & (vev_sub["strike"] == strike)].dropna(subset=["IV"])
        d = d[(d["IV"] > 0.01) & (d["IV"] < 5.0)]
        if len(d) == 0:
            continue
        fit_iv = fitted_iv(strike, day)
        rows.append({
            "day": day, "strike": strike,
            "IV_mean":      round(d["IV"].mean(), 5),
            "IV_std":       round(d["IV"].std(), 5),
            "IV_min":       round(d["IV"].min(), 5),
            "IV_max":       round(d["IV"].max(), 5),
            "fitted_IV":    round(fit_iv, 5) if not np.isnan(fit_iv) else "",
            "IV_vs_fit":    round(d["IV"].mean() - fit_iv, 5) if not np.isnan(fit_iv) else "",
            "delta_mean":   round(d["delta"].mean(), 4),
            "gamma_mean":   round(d["gamma"].mean(), 6),
            "vega_mean":    round(d["vega"].mean(), 4),
            "misprice_mean":round(vev_sub[(vev_sub["day"]==day) & (vev_sub["strike"]==strike)]["misprice"].mean(), 3),
            "n_samples":    len(d),
        })

summary = pd.DataFrame(rows)
summary_path = os.path.join(OUTPUT_DIR, "iv_summary.csv")
summary.to_csv(summary_path, index=False)
print(f"  → {summary_path}")

# ════════════════════════════════════════════════════════
#  PRINT CONSOLE SUMMARY
# ════════════════════════════════════════════════════════
print("\n" + "═" * 65)
print("QUICK SUMMARY")
print("═" * 65)

hp_mp = hydro["mid_price"]
vfe_mp = vfe["mid_price"]
print(f"\nHYDROGEL_PACK   mean={hp_mp.mean():.2f}  std={hp_mp.std():.2f}  "
      f"range=[{hp_mp.min():.1f}, {hp_mp.max():.1f}]")
print(f"VELVETFRUIT_EXT mean={vfe_mp.mean():.2f}  std={vfe_mp.std():.2f}  "
      f"range=[{vfe_mp.min():.1f}, {vfe_mp.max():.1f}]")

print("\nImplied Vol (mean across all days):")
piv = summary.groupby("strike")[["IV_mean", "IV_std", "delta_mean"]].mean().round(4)
print(piv.to_string())

print("\nSmile fit: IV = a·K² + b·K + c  per day:")
for day, c in smile_params.items():
    print(f"  Day {day}: {c[0]:.3e}·K² + {c[1]:.3e}·K + {c[2]:.5f}")

print(f"\nElapsed: {time.time()-t0:.0f}s")
print(f"Output:  ./{OUTPUT_DIR}/")
print("═" * 65)
print("\n✅  DONE. Upload all PNG files + iv_summary.csv to Claude.")
