#!/usr/bin/env python3
"""
IMC Prosperity 4 – Master Round Analysis Tool  (combined v3)
=============================================================
Merges the best of both analyzer versions into one comprehensive tool.

Produces (in --output dir):
  00_summary.txt              ← paste into Claude first
  00_parameter_report.txt     ← recommended code parameters
  01_delta1_overview.png
  02_delta1_ema_deviation.png
  03_vol_smile.png
  04_iv_price_deviations.png
  05_option_autocorr.png
  06_delta_surface.png
  07_fair_vs_market.png
  08_iv_level_trends.png
  09_mark_volume_heatmap.png
  10_mark_timing.png
  11_mark_signals.png
  12_mark_hydro.png           ← top-Mark overlay on HYDROGEL_PACK
  12_mark_velv.png            ← top-Mark overlay on VELVETFRUIT_EXTRACT
  13_spread_summary.png
  mark_volume_summary.csv
  mark_signal_strength.csv
  spread_liquidity_summary.csv

──────────────────────────────────────────────────────────────
QUICK START – two usage modes
──────────────────────────────────────────────────────────────

MODE A  (simple – put CSVs in one folder):
  python analyzer.py --data-dir ./my_csvs --round 4

MODE B  (explicit – name each file):
  python analyzer.py \\
      --prices prices_round_4_day_1.csv prices_round_4_day_2.csv prices_round_4_day_3.csv \\
      --trades trades_round_4_day_1.csv trades_round_4_day_2.csv trades_round_4_day_3.csv \\
      --round 4 --output ./r4_output

COMMON OPTIONS:
  --sample N        IV stride (default 20; lower = more accurate but slower)
  --ema-window N    EMA window for mean-reversion signal (default 20)
  --forward-ticks N Mark signal horizon (default 5)
  --no-marks        Skip all Mark-trader analysis
  --no-options      Skip options / smile analysis
  --quiet           Print only warnings + final report

CSV FORMAT (auto-detected ; or ,):
  Prices: day;timestamp;product;bid_price_1;bid_volume_1;...;mid_price;profit_and_loss
  Trades: day;timestamp;buyer;seller;symbol;currency;price;quantity

CURRENCY: XIRECS  (IMC Prosperity 4)
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
matplotlib.use("Agg")
matplotlib.rcParams.update({
    "figure.dpi": 120,
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# ════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ════════════════════════════════════════════════════════════════════════════

CURRENCY         = "XIRECS"
PRODUCTS_DELTA1  = ["HYDROGEL_PACK", "VELVETFRUIT_EXTRACT"]
UNDERLYING       = "VELVETFRUIT_EXTRACT"
DAY_PALETTE      = ["#4477AA", "#EE6677", "#228833", "#CCBB44", "#66CCEE"]
TICKS_PER_DAY    = 1_000_000

STRIKES_MAP: dict[str, int] = {
    "VEV_4000": 4000, "VEV_4500": 4500,
    "VEV_5000": 5000, "VEV_5100": 5100,
    "VEV_5200": 5200, "VEV_5300": 5300,
    "VEV_5400": 5400, "VEV_5500": 5500,
    "VEV_6000": 6000, "VEV_6500": 6500,
}
STRIKES       = sorted(STRIKES_MAP.values())
VEV_PRODUCTS  = [f"VEV_{k}" for k in STRIKES]
VEV_COLORS    = plt.cm.plasma(np.linspace(0.1, 0.9, len(STRIKES)))

# TTE in calendar days by round and historical day label
TTE_DAYS_PER_ROUND: dict[int, dict[int, int]] = {
    3: {0: 8, 1: 7, 2: 6},
    4: {1: 4, 2: 3, 3: 2},
    5: {1: 1, 2: 0},
}
TRADING_DAYS_PER_YEAR = 252   # consistent with r4 analyzer

# ════════════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ════════════════════════════════════════════════════════════════════════════

def _detect_sep(filepath: str) -> str:
    with open(filepath) as f:
        first = f.readline()
    return "," if ("," in first and ";" not in first) else ";"


def load_prices(filepath: str, day: int | None = None) -> pd.DataFrame:
    sep = _detect_sep(filepath)
    df  = pd.read_csv(filepath, sep=sep)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    if "timestamp" not in df.columns and "time" in df.columns:
        df = df.rename(columns={"time": "timestamp"})
    if "product" not in df.columns and "symbol" in df.columns:
        df = df.rename(columns={"symbol": "product"})
    if day is not None:
        df["day"] = day
    elif "day" not in df.columns:
        df["day"] = 1
    if "mid_price" not in df.columns:
        bp = next((c for c in df.columns if "bid" in c and "price" in c and "1" in c), None)
        ap = next((c for c in df.columns if "ask" in c and "price" in c and "1" in c), None)
        if bp and ap:
            df["mid_price"] = (df[bp] + df[ap]) / 2.0
        else:
            raise ValueError(f"Cannot find 'mid_price' column in {filepath}")
    df["product"]   = df["product"].str.strip().str.upper()
    df["global_ts"] = (df["day"] - 1) * TICKS_PER_DAY + df["timestamp"]
    for c in ["bid_price_1", "ask_price_1", "bid_volume_1", "ask_volume_1",
              "bid_price_2", "ask_price_2", "bid_volume_2", "ask_volume_2",
              "mid_price", "profit_and_loss"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "bid_price_1" in df.columns and "ask_price_1" in df.columns:
        df["spread"] = df["ask_price_1"] - df["bid_price_1"]
    return df


def load_trades(filepath: str, day: int | None = None) -> pd.DataFrame:
    sep = _detect_sep(filepath)
    df  = pd.read_csv(filepath, sep=sep)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    if "symbol" not in df.columns and "product" in df.columns:
        df = df.rename(columns={"product": "symbol"})
    if "symbol" not in df.columns:
        raise ValueError(f"Cannot find 'symbol' column in {filepath}")
    if day is not None:
        df["day"] = day
    elif "day" not in df.columns:
        df["day"] = 1
    for col in ("buyer", "seller"):
        df[col] = df[col].fillna("").astype(str).str.strip() if col in df.columns else ""
    df["symbol"]    = df["symbol"].str.strip().str.upper()
    df["price"]     = pd.to_numeric(df["price"],    errors="coerce")
    df["quantity"]  = pd.to_numeric(df["quantity"], errors="coerce")
    df["global_ts"] = (df["day"] - 1) * TICKS_PER_DAY + df["timestamp"]
    return df


def load_many(filepaths: list[str], loader_fn, base_day: int = 1) -> pd.DataFrame:
    frames = [loader_fn(fp, day=base_day + i) for i, fp in enumerate(filepaths)]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def discover_csvs(data_dir: str, round_num: int) -> tuple[list[str], list[str]]:
    """Auto-discover price and trade CSVs in data_dir for a given round."""
    price_files, trade_files = [], []
    for fname in sorted(os.listdir(data_dir)):
        path = os.path.join(data_dir, fname)
        if f"round_{round_num}" in fname or f"round{round_num}" in fname:
            if "price" in fname:
                price_files.append(path)
            elif "trade" in fname:
                trade_files.append(path)
    return price_files, trade_files


def mid_series(prices_df: pd.DataFrame, product: str) -> pd.DataFrame:
    mask = prices_df["product"] == product.upper()
    return (
        prices_df[mask][["day", "timestamp", "global_ts", "mid_price"]]
        .dropna(subset=["mid_price"])
        .sort_values(["day", "timestamp"])
        .reset_index(drop=True)
    )


def spread_series(prices_df: pd.DataFrame, product: str) -> pd.DataFrame:
    mask = prices_df["product"] == product.upper()
    sub  = prices_df[mask].copy()
    if "spread" in sub.columns:
        return sub[["day", "timestamp", "spread"]].dropna()
    bp = next((c for c in sub.columns if "bid" in c and "price" in c and "1" in c), None)
    ap = next((c for c in sub.columns if "ask" in c and "price" in c and "1" in c), None)
    if bp and ap:
        sub["spread"] = sub[ap] - sub[bp]
        return sub[["day", "timestamp", "spread"]].dropna()
    return pd.DataFrame(columns=["day", "timestamp", "spread"])


def acf_series(s: pd.Series, max_lag: int = 30) -> pd.DataFrame:
    """Return a DataFrame with columns [lag, acf]."""
    s = s.dropna()
    return pd.DataFrame(
        [(lag, s.autocorr(lag=lag)) for lag in range(1, max_lag + 1)],
        columns=["lag", "acf"],
    )


# ════════════════════════════════════════════════════════════════════════════
#  BLACK-SCHOLES (pure-Python, no scipy needed for IV computation)
# ════════════════════════════════════════════════════════════════════════════

def _ncdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / 1.4142135623730951))


def bs_call(S: float, K: float, T: float, sigma: float) -> float:
    if T <= 1e-8 or sigma <= 1e-8:
        return max(0.0, S - K)
    sq = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * sigma ** 2 * T) / sq
    return S * _ncdf(d1) - K * _ncdf(d1 - sq)


def bs_delta(S: float, K: float, T: float, sigma: float) -> float:
    if T <= 1e-8 or sigma <= 1e-8:
        return 1.0 if S > K else 0.0
    d1 = (math.log(S / K) + 0.5 * sigma ** 2 * T) / (sigma * math.sqrt(T))
    return _ncdf(d1)


def solve_iv(S: float, K: float, T: float, price: float,
             lo: float = 0.005, hi: float = 3.5, n: int = 80) -> float | None:
    """Binary-search implied volatility.  Returns None if unsolvable."""
    intrinsic = max(0.0, S - K)
    if price < intrinsic + 0.05 or T <= 1e-8 or S <= 0:
        return None
    if bs_call(S, K, T, lo) > price:
        return None
    if bs_call(S, K, T, hi) < price:
        return None
    for _ in range(n):
        mid = (lo + hi) * 0.5
        if bs_call(S, K, T, mid) < price:
            lo = mid
        else:
            hi = mid
    iv = (lo + hi) * 0.5
    return iv if 0.015 < iv < 3.0 else None


# ════════════════════════════════════════════════════════════════════════════
#  IV SURFACE  (v2 approach: proper per-tick merge, stored flat DataFrame)
# ════════════════════════════════════════════════════════════════════════════

def build_iv_surface(prices_df: pd.DataFrame, tte_map: dict[int, float],
                     iv_stride: int = 20) -> pd.DataFrame:
    """
    For every (day, timestamp) snapshot containing VELVETFRUIT_EXTRACT,
    solve implied vol for each VEV strike and return a flat DataFrame with:
      day, timestamp, global_ts, sym, K, S, T, iv, moneyness, market_mid,
      delta, fair_price
    iv_stride: sample every N unique timestamps per day (speed vs accuracy).
    """
    under = (
        prices_df[prices_df["product"] == UNDERLYING][["day", "timestamp", "mid_price"]]
        .rename(columns={"mid_price": "S"})
        .dropna()
    )

    rows: list[dict] = []
    for sym, K in STRIKES_MAP.items():
        opt = (
            prices_df[prices_df["product"] == sym][["day", "timestamp", "mid_price"]]
            .rename(columns={"mid_price": "market_mid"})
            .dropna()
        )
        # Sub-sample by stride per day
        if iv_stride > 1:
            sampled = []
            for d, g in opt.groupby("day"):
                ts_vals = g["timestamp"].unique()
                ts_sel  = ts_vals[::iv_stride]
                sampled.append(g[g["timestamp"].isin(ts_sel)])
            opt = pd.concat(sampled, ignore_index=True) if sampled else opt

        merged = opt.merge(under, on=["day", "timestamp"], how="inner")

        for row in merged.itertuples(index=False):
            d   = int(row.day)
            T   = tte_map.get(d, 4.0 / TRADING_DAYS_PER_YEAR)
            S   = float(row.S)
            mkt = float(row.market_mid)
            if S <= 0 or mkt <= 0:
                continue
            iv = solve_iv(S, K, T, mkt)
            if iv is None:
                continue
            rows.append({
                "day":        d,
                "timestamp":  int(row.timestamp),
                "global_ts":  (d - 1) * TICKS_PER_DAY + int(row.timestamp),
                "sym":        sym,
                "K":          K,
                "S":          S,
                "T":          T,
                "iv":         iv,
                "moneyness":  math.log(S / K),
                "market_mid": mkt,
                "delta":      bs_delta(S, K, T, iv),
                "fair_price": bs_call(S, K, T, iv),
            })

    return pd.DataFrame(rows)


def fit_smile_deviations(iv_df: pd.DataFrame) -> pd.DataFrame:
    """
    Per (day, timestamp) tick, fit a quadratic smile IV = f(moneyness).
    Adds: fitted_iv, iv_deviation, smile_price, price_deviation.
    Requires ≥3 strikes per tick for a valid fit.
    """
    out_rows: list[dict] = []
    for (d, ts), grp in iv_df.groupby(["day", "timestamp"]):
        nan_ext = {"fitted_iv": float("nan"), "iv_deviation": float("nan"),
                   "smile_price": float("nan"), "price_deviation": float("nan")}
        if len(grp) < 3:
            for row in grp.itertuples(index=False):
                out_rows.append({**row._asdict(), **nan_ext})
            continue
        m_arr  = grp["moneyness"].values
        iv_arr = grp["iv"].values
        try:
            coeffs = np.polyfit(m_arr, iv_arr, 2)
        except np.linalg.LinAlgError:
            coeffs = np.array([0.0, 0.0, np.mean(iv_arr)])
        for row in grp.itertuples(index=False):
            fitted = max(0.01, float(np.polyval(coeffs, row.moneyness)))
            sp     = bs_call(row.S, row.K, row.T, fitted)
            out_rows.append({
                **row._asdict(),
                "fitted_iv":      fitted,
                "iv_deviation":   row.iv - fitted,
                "smile_price":    sp,
                "price_deviation": row.market_mid - sp,
            })
    return pd.DataFrame(out_rows)


# ════════════════════════════════════════════════════════════════════════════
#  MARK / TRADER SIGNAL ANALYSIS
# ════════════════════════════════════════════════════════════════════════════

def compute_mark_volume_table(trades_df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a tidy DataFrame [mark, product, buy, sell, net]
    for every counterparty (not just those starting with 'Mark').
    Also saves mark_volume_summary.csv.
    """
    all_marks = sorted(
        set(trades_df["buyer"].dropna().unique()) |
        set(trades_df["seller"].dropna().unique())
    )
    rows = []
    for mark in all_marks:
        if not mark:
            continue
        ab  = (trades_df[trades_df["buyer"]  == mark]
               .groupby("symbol")["quantity"].sum().rename("buy"))
        as_ = (trades_df[trades_df["seller"] == mark]
               .groupby("symbol")["quantity"].sum().rename("sell"))
        m = pd.concat([ab, as_], axis=1).fillna(0)
        m["net"]  = m["buy"] - m["sell"]
        m["mark"] = mark
        rows.append(m.reset_index().rename(columns={"symbol": "product"}))
    if not rows:
        return pd.DataFrame(columns=["mark", "product", "buy", "sell", "net"])
    return pd.concat(rows, ignore_index=True)


def compute_mark_forward_returns(
    trades_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    forward_ticks: int = 5,
    products: list[str] | None = None,
) -> pd.DataFrame:
    """
    For each Mark trade, compute the signed forward return (bps) forward_ticks later.
    Convention: positive t-stat = following this Mark is profitable.
    """
    if products is None:
        products = PRODUCTS_DELTA1

    all_rows: list[dict] = []
    for prod in products:
        mp = mid_series(prices_df, prod).reset_index(drop=True)
        ts_index: dict[tuple[int, int], int] = {
            (int(r.day), int(r.timestamp)): i for i, r in mp.iterrows()
        }
        mid_arr    = mp["mid_price"].values
        prod_trades = trades_df[trades_df["symbol"] == prod.upper()]

        for row in prod_trades.itertuples(index=False):
            key = (int(row.day), int(row.timestamp))
            idx = ts_index.get(key)
            if idx is None or idx + forward_ticks >= len(mid_arr):
                continue
            ref_price  = float(row.price)
            fwd_mid    = float(mid_arr[idx + forward_ticks])
            fwd_bps    = (fwd_mid - ref_price) / ref_price * 10_000

            for participant, side in [(row.buyer, "buy"), (row.seller, "sell")]:
                if not str(participant):
                    continue
                signed = fwd_bps if side == "buy" else -fwd_bps
                all_rows.append({
                    "product":   prod,
                    "mark":      participant,
                    "side":      side,
                    "fwd_ret_bps": signed,
                    "price":     ref_price,
                    "day":       row.day,
                    "timestamp": row.timestamp,
                })

    if not all_rows:
        return pd.DataFrame()

    df      = pd.DataFrame(all_rows)
    grouped = (
        df.groupby(["product", "mark", "side"])["fwd_ret_bps"]
        .agg(avg="mean", std="std", n="count")
        .reset_index()
    )
    grouped["t_stat"] = grouped.apply(
        lambda r: r["avg"] / (r["std"] / math.sqrt(max(r["n"], 1))) if r["std"] > 0 else 0.0,
        axis=1,
    )
    return grouped.sort_values("t_stat", key=abs, ascending=False)


# ════════════════════════════════════════════════════════════════════════════
#  CHART HELPERS
# ════════════════════════════════════════════════════════════════════════════

def _save(fig: plt.Figure, path: str, name: str):
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  ✓ {name}")


def _out(out: str, fname: str) -> str:
    return os.path.join(out, fname)


# ════════════════════════════════════════════════════════════════════════════
#  CHART 01 – Delta-1 overview (mid price + spread + inset ACF)
# ════════════════════════════════════════════════════════════════════════════

def plot_01_delta1_overview(prices_df: pd.DataFrame, out: str):
    fig, axes = plt.subplots(3, 2, figsize=(16, 13))
    fig.suptitle("Delta-1 Overview – HYDROGEL_PACK & VELVETFRUIT_EXTRACT",
                 fontsize=13, fontweight="bold")

    for col, prod in enumerate(PRODUCTS_DELTA1):
        mp = mid_series(prices_df, prod)
        sp = spread_series(prices_df, prod)
        color = DAY_PALETTE[col]

        # Row 0 – mid price
        ax = axes[0, col]
        for d, g in mp.groupby("day"):
            ax.plot(g["timestamp"], g["mid_price"], lw=0.5,
                    color=DAY_PALETTE[d % len(DAY_PALETTE)], label=f"Day {d}")
        ax.set_title(f"{prod} – Mid Price")
        ax.set_xlabel("Timestamp"); ax.set_ylabel(f"Price ({CURRENCY})")
        ax.legend(fontsize=7)

        # Row 1 – spread
        ax = axes[1, col]
        if not sp.empty:
            for d, g in sp.groupby("day"):
                ax.plot(g["timestamp"], g["spread"], lw=0.35, alpha=0.55,
                        color=DAY_PALETTE[d % len(DAY_PALETTE)])
            mn = sp["spread"].mean()
            ax.axhline(mn, color="red", ls="--", lw=1.3, label=f"Mean={mn:.2f}")
        ax.set_title(f"{prod} – Bid-Ask Spread")
        ax.set_xlabel("Timestamp"); ax.set_ylabel(f"Spread ({CURRENCY})")
        ax.legend(fontsize=7)

        # Row 2 – ACF of returns
        ax = axes[2, col]
        ret = mp["mid_price"].pct_change().dropna()
        acf_df = acf_series(ret, 30)
        ci_ = 1.96 / math.sqrt(max(len(ret), 1))
        ax.bar(acf_df["lag"], acf_df["acf"],
               color=[("#CC3333" if v < -ci_ else color) for v in acf_df["acf"]], alpha=0.8)
        ax.axhline(ci_,  color="red", ls="--", lw=0.8)
        ax.axhline(-ci_, color="red", ls="--", lw=0.8)
        ax.axhline(0,    color="black", lw=0.5)
        acf1 = ret.autocorr(lag=1)
        ax.set_title(f"{prod} – Return ACF  (lag-1={acf1:.4f})")
        ax.set_xlabel("Lag"); ax.set_ylabel("ACF")

    _save(fig, _out(out, "01_delta1_overview.png"), "01_delta1_overview.png")


# ════════════════════════════════════════════════════════════════════════════
#  CHART 02 – EMA deviation / mean-reversion thresholds
# ════════════════════════════════════════════════════════════════════════════

def plot_02_ema_deviation(prices_df: pd.DataFrame, out: str, ema_win: int = 20) -> dict:
    fig, axes = plt.subplots(2, 2, figsize=(16, 9))
    fig.suptitle("EMA Deviation – Mean-Reversion Signal", fontsize=13, fontweight="bold")
    thresholds: dict[str, float] = {}

    for col, prod in enumerate(PRODUCTS_DELTA1):
        mp = mid_series(prices_df, prod)
        mp["ema20"]  = mp["mid_price"].ewm(span=20,  adjust=False).mean()
        mp["ema100"] = mp["mid_price"].ewm(span=100, adjust=False).mean()
        mp["dev20"]  = mp["mid_price"] - mp["ema20"]
        p90 = mp["dev20"].quantile(0.90)
        p10 = mp["dev20"].quantile(0.10)
        thresholds[prod] = abs(p90)

        ax = axes[0, col]
        for d, g in mp.groupby("day"):
            ax.plot(g["timestamp"], g["mid_price"], lw=0.4,
                    color=DAY_PALETTE[d % len(DAY_PALETTE)], alpha=0.7, label=f"D{d}")
        ax.plot(mp["timestamp"], mp["ema20"],  lw=1.0, color="#EE6677", label="EMA-20")
        ax.plot(mp["timestamp"], mp["ema100"], lw=1.0, ls="--", color="#CCBB44", label="EMA-100")
        ax.set_title(f"{prod} – Price + EMAs")
        ax.set_ylabel(f"Price ({CURRENCY})"); ax.legend(fontsize=6)

        ax = axes[1, col]
        for d, g in mp.groupby("day"):
            ax.fill_between(g["timestamp"], g["dev20"], 0,
                            alpha=0.35, color=DAY_PALETTE[d % len(DAY_PALETTE)])
        ax.axhline(p90, color="red",   ls="--", lw=1.3, label=f"p90={p90:+.2f}")
        ax.axhline(p10, color="green", ls="--", lw=1.3, label=f"p10={p10:+.2f}")
        ax.axhline(0,   color="black", lw=0.7)
        ax.set_title(f"{prod} – Deviation from EMA-{ema_win}  (MR threshold ≈ ±{abs(p90):.1f})")
        ax.set_ylabel(f"Deviation ({CURRENCY})"); ax.legend(fontsize=7)

    _save(fig, _out(out, "02_delta1_ema_deviation.png"), "02_delta1_ema_deviation.png")
    return thresholds


# ════════════════════════════════════════════════════════════════════════════
#  CHART 03 – Volatility smile per day
# ════════════════════════════════════════════════════════════════════════════

def plot_03_vol_smile(iv_df: pd.DataFrame, out: str):
    days = sorted(iv_df["day"].unique())
    fig, axes = plt.subplots(1, len(days), figsize=(6 * len(days), 5), sharey=True)
    if len(days) == 1:
        axes = [axes]
    fig.suptitle("Volatility Smile – IV vs ln(S/K) by Day", fontsize=13, fontweight="bold")

    for ax, d in zip(axes, days):
        g = iv_df[iv_df["day"] == d].dropna(subset=["iv", "moneyness"])
        ax.scatter(g["moneyness"], g["iv"], alpha=0.15, s=8, color="#4477AA", label="IV obs")
        m_rng = np.linspace(g["moneyness"].min(), g["moneyness"].max(), 300)
        try:
            c  = np.polyfit(g["moneyness"].values, g["iv"].values, 2)
            yt = np.polyval(c, g["moneyness"].values)
            r2 = 1 - np.sum((g["iv"].values - yt) ** 2) / np.sum(
                (g["iv"].values - g["iv"].mean()) ** 2)
            ax.plot(m_rng, np.polyval(c, m_rng), "r-", lw=2, label=f"Parabola (R²={r2:.3f})")
        except Exception:
            pass
        tte_d = g["T"].iloc[0] * TRADING_DAYS_PER_YEAR if not g.empty else "?"
        ax.set_title(f"Day {d}  TTE={tte_d:.0f}d" if isinstance(tte_d, float) else f"Day {d}")
        ax.set_xlabel("ln(S/K)"); ax.set_ylabel("IV"); ax.legend(fontsize=8)

    _save(fig, _out(out, "03_vol_smile.png"), "03_vol_smile.png")


# ════════════════════════════════════════════════════════════════════════════
#  CHART 04 – Price deviation from IV-smile fair value
# ════════════════════════════════════════════════════════════════════════════

def plot_04_iv_price_deviations(smile_df: pd.DataFrame, out: str):
    syms = sorted(smile_df["sym"].unique(), key=lambda s: STRIKES_MAP.get(s, 0))
    days = sorted(smile_df["day"].unique())
    nrows, ncols = len(syms), len(days)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 2.5 * nrows))
    if nrows == 1: axes = np.array([axes])
    if ncols == 1: axes = np.array([[r] for r in axes])
    fig.suptitle("Price Deviation from IV-Fair Value\n"
                 "(+= market ABOVE smile = overpriced;  −= underpriced)",
                 fontsize=12, fontweight="bold")

    print("  Persistent price deviations from smile:")
    for ri, sym in enumerate(syms):
        for ci, d in enumerate(days):
            ax  = axes[ri][ci]
            sub = (smile_df[(smile_df["sym"] == sym) & (smile_df["day"] == d)]
                   .sort_values("timestamp")
                   .dropna(subset=["price_deviation"]))
            if sub.empty:
                ax.set_visible(False); continue
            devs = sub["price_deviation"].values
            ts   = sub["timestamp"].values
            ax.fill_between(ts, devs, 0, where=(devs > 0), color="#EE6677", alpha=0.55)
            ax.fill_between(ts, devs, 0, where=(devs <= 0), color="#66CC99", alpha=0.55)
            ax.plot(ts, devs, lw=0.4, color="#334455")
            ax.axhline(0, color="black", lw=0.6)
            ax.set_title(f"{sym} D{d}", fontsize=7)
            if ci == 0: ax.set_ylabel("Dev", fontsize=6)
            mean_dev = float(np.mean(devs))
            bias = "OVER" if mean_dev > 0.5 else "UNDER" if mean_dev < -0.5 else "neutral"
            if bias != "neutral":
                print(f"    {sym} Day{d}: mean={mean_dev:+.2f} → {bias}PRICED")

    _save(fig, _out(out, "04_iv_price_deviations.png"), "04_iv_price_deviations.png")


# ════════════════════════════════════════════════════════════════════════════
#  CHART 05 – Option return ACF (scalping signal)
# ════════════════════════════════════════════════════════════════════════════

def plot_05_option_autocorr(prices_df: pd.DataFrame, smile_df: pd.DataFrame,
                             out: str, max_lag: int = 20) -> dict:
    """
    Uses raw mid-prices from prices_df for all VEV products (not just ones
    with valid IV), giving full coverage. Falls back to smile_df if needed.
    """
    # Gather which symbols have data
    all_syms = sorted(STRIKES_MAP.keys(), key=lambda s: STRIKES_MAP[s])
    avail_syms = [s for s in all_syms
                  if (prices_df["product"] == s).any() or
                     (not smile_df.empty and s in smile_df["sym"].unique())]

    ncols = 5
    nrows = math.ceil(len(avail_syms) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.2 * nrows))
    axes = np.array(axes).flatten()
    fig.suptitle("VEV Option Return ACF\n"
                 "(red bars = statistically significant; negative lag-1 = mean-reverting)",
                 fontsize=12, fontweight="bold")

    lag1_acfs: dict[str, float] = {}
    for idx, sym in enumerate(avail_syms):
        ax = axes[idx]
        # Prefer raw prices for full resolution
        sub = prices_df[prices_df["product"] == sym].sort_values("global_ts")
        if sub.empty and not smile_df.empty:
            sub_iv = smile_df[smile_df["sym"] == sym].sort_values(["day", "timestamp"])
            prices_arr = sub_iv["market_mid"].values
        else:
            prices_arr = sub["mid_price"].values

        if len(prices_arr) < max_lag + 5:
            ax.set_title(sym, fontsize=8); continue

        rets = np.diff(prices_arr) / np.maximum(np.abs(prices_arr[:-1]), 0.01)
        ci   = 1.96 / math.sqrt(max(len(rets), 1))
        acf  = []
        for lag in range(1, max_lag + 1):
            c = np.corrcoef(rets[:-lag], rets[lag:])[0, 1] if len(rets) > lag else 0.0
            acf.append(0.0 if np.isnan(c) else c)

        lag1 = acf[0] if acf else 0.0
        lag1_acfs[sym] = lag1
        colors = ["#CC3333" if v < -ci else "#4477AA" for v in acf]
        ax.bar(range(1, len(acf) + 1), acf, color=colors, width=0.7)
        ax.axhline(ci,  color="red", ls="--", lw=0.8)
        ax.axhline(-ci, color="red", ls="--", lw=0.8)
        ax.axhline(0,   color="black", lw=0.5)
        ax.set_title(f"{sym}\nlag-1={lag1:+.3f}", fontsize=8)
        ax.set_ylim(-0.5, 0.3)

    for i in range(len(avail_syms), len(axes)):
        axes[i].set_visible(False)

    _save(fig, _out(out, "05_option_autocorr.png"), "05_option_autocorr.png")

    print("  Option lag-1 ACF (negative = mean-reverting = scalping opportunity):")
    for s, v in sorted(lag1_acfs.items(), key=lambda x: x[1]):
        note = " ← STRONG SCALP" if v < -0.15 else (" ← moderate" if v < -0.05 else "")
        print(f"    {s}: {v:+.4f}{note}")
    return lag1_acfs


# ════════════════════════════════════════════════════════════════════════════
#  CHART 06 – Delta surface
# ════════════════════════════════════════════════════════════════════════════

def plot_06_delta_surface(iv_df: pd.DataFrame, out: str):
    days     = sorted(iv_df["day"].unique())
    sym_order = sorted(STRIKES_MAP.keys(), key=lambda s: STRIKES_MAP[s])
    cmap     = plt.cm.plasma(np.linspace(0.05, 0.95, len(STRIKES_MAP)))

    fig, axes = plt.subplots(1, len(days), figsize=(6 * len(days), 5))
    if len(days) == 1: axes = [axes]
    fig.suptitle("VEV Option Deltas vs Time", fontsize=13, fontweight="bold")

    for ax, d in zip(axes, days):
        dg = iv_df[iv_df["day"] == d]
        for sym, col in zip(sym_order, cmap):
            sg = dg[dg["sym"] == sym].sort_values("timestamp")
            if sg.empty: continue
            ax.plot(sg["timestamp"].values, sg["delta"].values,
                    lw=0.8, color=col, label=sym.replace("VEV_", ""), alpha=0.9)
        tte_d = dg["T"].iloc[0] * TRADING_DAYS_PER_YEAR if not dg.empty else "?"
        ax.set_title(f"Day {d} – TTE={tte_d:.0f}d" if isinstance(tte_d, float) else f"Day {d}")
        ax.axhline(0.5, color="black", ls="--", lw=0.8, alpha=0.5)
        ax.set_xlabel("Timestamp"); ax.set_ylabel("Delta")
        ax.legend(fontsize=6, ncol=2); ax.set_ylim(0, 1.05)

    _save(fig, _out(out, "06_delta_surface.png"), "06_delta_surface.png")


# ════════════════════════════════════════════════════════════════════════════
#  CHART 07 – BS Fair vs Market Mid
# ════════════════════════════════════════════════════════════════════════════

def plot_07_fair_vs_market(smile_df: pd.DataFrame, out: str):
    focus = [s for s in ["VEV_5000", "VEV_5200", "VEV_5400", "VEV_5500", "VEV_6000"]
             if s in smile_df["sym"].unique()]
    if not focus:
        focus = sorted(smile_df["sym"].unique(), key=lambda s: STRIKES_MAP.get(s, 0))[:5]
    days  = sorted(smile_df["day"].unique())
    nrows, ncols = len(focus), len(days)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 2.8 * nrows))
    if nrows == 1: axes = np.array([axes])
    if ncols == 1: axes = np.array([[r] for r in axes])
    fig.suptitle("VEV – BS Fair (smile) vs Market Mid", fontsize=13, fontweight="bold")

    for ri, sym in enumerate(focus):
        for ci, d in enumerate(days):
            ax  = axes[ri][ci]
            sub = (smile_df[(smile_df["sym"] == sym) & (smile_df["day"] == d)]
                   .sort_values("timestamp")
                   .dropna(subset=["smile_price"]))
            if sub.empty: ax.set_visible(False); continue
            ax.plot(sub["timestamp"], sub["market_mid"], lw=0.7, color="#4477AA", label="Market")
            ax.plot(sub["timestamp"], sub["smile_price"], lw=0.7, ls="--",
                    color="#EE6677", label="BS Fair")
            ax.fill_between(sub["timestamp"], sub["market_mid"], sub["smile_price"],
                            alpha=0.2, color="#AA44BB")
            ax.set_title(f"{sym} D{d}", fontsize=8)
            if ci == 0: ax.set_ylabel(f"Price ({CURRENCY})", fontsize=7)
            if ri == 0 and ci == 0: ax.legend(fontsize=6)

    _save(fig, _out(out, "07_fair_vs_market.png"), "07_fair_vs_market.png")


# ════════════════════════════════════════════════════════════════════════════
#  CHART 08 – IV level trends over time per strike
# ════════════════════════════════════════════════════════════════════════════

def plot_08_iv_level_trends(iv_df: pd.DataFrame, out: str):
    syms      = sorted(STRIKES_MAP.keys(), key=lambda s: STRIKES_MAP[s])
    ncols     = 5
    nrows     = math.ceil(len(syms) / ncols)
    day_colors = {1: "#4477AA", 2: "#EE6677", 3: "#228833", 4: "#CCBB44"}

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.5 * nrows))
    axes = np.array(axes).flatten()
    fig.suptitle("IV Level Over Time per Strike", fontsize=12, fontweight="bold")

    for idx, sym in enumerate(syms):
        ax = axes[idx]
        for d, col in day_colors.items():
            sg = iv_df[(iv_df["sym"] == sym) & (iv_df["day"] == d)].sort_values("timestamp")
            if sg.empty: continue
            ax.plot(sg["timestamp"], sg["iv"], lw=0.7, color=col, alpha=0.85, label=f"D{d}")
        ax.set_title(sym, fontsize=9)
        ax.set_xlabel("Timestamp", fontsize=7); ax.set_ylabel("IV", fontsize=7)
        ax.legend(fontsize=6)

    for i in range(len(syms), len(axes)):
        axes[i].set_visible(False)
    _save(fig, _out(out, "08_iv_level_trends.png"), "08_iv_level_trends.png")


# ════════════════════════════════════════════════════════════════════════════
#  CHART 09 – Mark volume heatmap (buy / sell / net)
# ════════════════════════════════════════════════════════════════════════════

def plot_09_mark_heatmap(mark_vol: pd.DataFrame, out: str):
    if mark_vol.empty:
        print("  ⚠  No Mark volume data – skipping heatmap"); return

    all_marks = sorted(mark_vol["mark"].unique())
    all_prods = (PRODUCTS_DELTA1 +
                 sorted([s for s in STRIKES_MAP.keys()
                         if s in mark_vol["product"].unique()],
                        key=lambda s: STRIKES_MAP[s]))

    def _mat(col):
        return np.array([[mark_vol[(mark_vol["mark"] == m) & (mark_vol["product"] == p)][col]
                          .sum() for p in all_prods] for m in all_marks])

    buy_arr  = _mat("buy")
    sell_arr = _mat("sell")
    net_arr  = buy_arr - sell_arr

    # Drop rows/cols that are all-zero
    active_row = (buy_arr + sell_arr).sum(axis=1) > 0
    active_col = (buy_arr + sell_arr).sum(axis=0) > 0
    buy_arr  = buy_arr[active_row][:, active_col]
    sell_arr = sell_arr[active_row][:, active_col]
    net_arr  = net_arr[active_row][:, active_col]
    row_labels = [all_marks[i] for i, ok in enumerate(active_row) if ok]
    col_labels = [all_prods[i] for i, ok in enumerate(active_col) if ok]
    col_short  = [c.replace("VELVETFRUIT_EXTRACT", "VFE").replace("HYDROGEL_PACK", "HYDRO")
                  for c in col_labels]

    if not row_labels:
        print("  ⚠  No active Mark traders – skipping heatmap"); return

    h   = max(5, len(row_labels) * 0.42 + 2)
    fig, axes = plt.subplots(1, 3, figsize=(22, h))
    fig.suptitle("Mark Activity – Volume Heatmaps", fontsize=13, fontweight="bold")

    for ax, data, title, cmap in zip(
        axes, [buy_arr, sell_arr, net_arr], ["Buy", "Sell", "Net (Buy−Sell)"],
        ["Blues", "Reds", "RdYlGn"]
    ):
        vmax = max(abs(data).max(), 1)
        kwargs = dict(vmin=-vmax, vmax=vmax) if cmap == "RdYlGn" else {}
        im = ax.imshow(data, aspect="auto", cmap=cmap, **kwargs)
        ax.set_xticks(range(len(col_short)))
        ax.set_xticklabels(col_short, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels(row_labels, fontsize=7)
        ax.set_title(title)
        plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02)

    _save(fig, _out(out, "09_mark_volume_heatmap.png"), "09_mark_volume_heatmap.png")


# ════════════════════════════════════════════════════════════════════════════
#  CHART 10 – Mark timing (timestamp distribution per day)
# ════════════════════════════════════════════════════════════════════════════

def plot_10_mark_timing(trades_df: pd.DataFrame, out: str, nbins: int = 60):
    days      = sorted(trades_df["day"].unique())
    all_marks = sorted(
        set(trades_df["buyer"].dropna().unique()) |
        set(trades_df["seller"].dropna().unique())
    )
    # Top 8 by total trade count
    top_marks = sorted(
        all_marks,
        key=lambda m: len(trades_df[(trades_df["buyer"] == m) | (trades_df["seller"] == m)]),
        reverse=True,
    )[:8]

    fig, axes = plt.subplots(1, len(days), figsize=(6 * len(days), 5))
    if len(days) == 1: axes = [axes]
    fig.suptitle("Mark Trade Timestamp Distribution", fontsize=13, fontweight="bold")

    for ax, d in zip(axes, days):
        dt = trades_df[trades_df["day"] == d]
        for i, mark in enumerate(top_marks):
            ts = pd.concat([
                dt[dt["buyer"]  == mark]["timestamp"],
                dt[dt["seller"] == mark]["timestamp"],
            ])
            if len(ts) > 5:
                ax.hist(ts, bins=nbins, alpha=0.4,
                        color=DAY_PALETTE[i % len(DAY_PALETTE)], label=mark)
        ax.set_title(f"Day {d}")
        ax.set_xlabel("Timestamp"); ax.set_ylabel("# Trades")
        if d == days[0]: ax.legend(fontsize=6, ncol=2)

    _save(fig, _out(out, "10_mark_timing.png"), "10_mark_timing.png")


# ════════════════════════════════════════════════════════════════════════════
#  CHART 11 – Mark forward-return signals
# ════════════════════════════════════════════════════════════════════════════

def plot_11_mark_signals(signals_df: pd.DataFrame, out: str):
    if signals_df.empty:
        print("  ⚠  signals_df empty – skipping mark signals chart"); return

    fig, axes = plt.subplots(1, len(PRODUCTS_DELTA1), figsize=(14, 7))
    fig.suptitle("Mark Signals – Forward Return (bps)", fontsize=13, fontweight="bold")

    for ax, prod in zip(axes, PRODUCTS_DELTA1):
        sub = signals_df[(signals_df["product"] == prod) & (signals_df["n"] >= 5)].copy()
        if sub.empty: ax.set_title(prod); continue
        sub = sub.sort_values("t_stat")
        labels = sub.apply(lambda r: f"{r['mark']} / {r['side']}", axis=1).values
        vals   = sub["avg"].values
        colors = ["#228833" if v > 0 else "#CC3333" for v in vals]
        ax.barh(range(len(sub)), vals, color=colors, alpha=0.85)
        ax.set_yticks(range(len(sub)))
        ax.set_yticklabels(labels, fontsize=7)
        for i, row in enumerate(sub.itertuples()):
            ax.text(0.005, i, f"t={row.t_stat:.1f}  n={int(row.n)}",
                    va="center", ha="left", fontsize=6, color="black",
                    transform=ax.get_yaxis_transform())
        ax.axvline(0, color="black", lw=0.9)
        ax.set_xlabel("Avg fwd return (bps)")
        ax.set_title(prod.replace("_", " "))

    _save(fig, _out(out, "11_mark_signals.png"), "11_mark_signals.png")


# ════════════════════════════════════════════════════════════════════════════
#  CHART 12 – Per-Mark overlay on individual product
# ════════════════════════════════════════════════════════════════════════════

def plot_12_mark_overlay(trades_df: pd.DataFrame, prices_df: pd.DataFrame,
                          product: str, out: str, fname: str, n_marks: int = 6):
    mp = mid_series(prices_df, product)
    pt = trades_df[trades_df["symbol"] == product.upper()]
    all_marks = sorted(
        set(pt["buyer"].unique()) | set(pt["seller"].unique()),
        key=lambda m: len(pt[(pt["buyer"] == m) | (pt["seller"] == m)]),
        reverse=True,
    )
    top_marks = [m for m in all_marks if m][:n_marks]
    if not top_marks:
        print(f"  ⚠  No trades for {product} – skipping overlay"); return

    ncols = 2
    nrows = math.ceil(len(top_marks) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 4 * nrows))
    axes = np.array(axes).flatten()
    fig.suptitle(f"{product.replace('_', ' ')} – Top-{len(top_marks)} Mark Overlay",
                 fontsize=12, fontweight="bold")

    for idx, mark in enumerate(top_marks):
        ax = axes[idx]
        for d, g in mp.groupby("day"):
            ax.plot(g["timestamp"], g["mid_price"], lw=0.35, color="#AAAAAA", alpha=0.7)
        mt    = pt[(pt["buyer"] == mark) | (pt["seller"] == mark)]
        buys  = mt[mt["buyer"]  == mark]
        sells = mt[mt["seller"] == mark]
        if not buys.empty:
            ax.scatter(buys["timestamp"],  buys["price"],
                       marker="^", s=20, color="#228833", zorder=5,
                       label=f"Buy ({len(buys)})")
        if not sells.empty:
            ax.scatter(sells["timestamp"], sells["price"],
                       marker="v", s=20, color="#CC3333", zorder=5,
                       label=f"Sell ({len(sells)})")
        ax.set_title(mark, fontsize=9)
        ax.legend(fontsize=7)
        ax.set_xlabel("Timestamp"); ax.set_ylabel(f"Price ({CURRENCY})")

    for i in range(len(top_marks), len(axes)):
        axes[i].set_visible(False)
    _save(fig, _out(out, f"{fname}.png"), f"{fname}.png")


# ════════════════════════════════════════════════════════════════════════════
#  CHART 13 – Spread & liquidity summary
# ════════════════════════════════════════════════════════════════════════════

def plot_13_spread_summary(prices_df: pd.DataFrame, out: str) -> pd.DataFrame:
    all_prods = PRODUCTS_DELTA1 + sorted(STRIKES_MAP.keys(), key=lambda s: STRIKES_MAP[s])
    rows = []
    for prod in all_prods:
        sp = spread_series(prices_df, prod)
        sub = prices_df[prices_df["product"] == prod]
        entry: dict = {
            "product":         prod,
            "mean_spread":     sp["spread"].mean()   if not sp.empty else float("nan"),
            "median_spread":   sp["spread"].median() if not sp.empty else float("nan"),
            "min_spread":      sp["spread"].min()    if not sp.empty else float("nan"),
            "max_spread":      sp["spread"].max()    if not sp.empty else float("nan"),
            "mean_mid":        sub["mid_price"].mean(),
            "std_mid":         sub["mid_price"].std(),
        }
        if "bid_volume_1" in sub.columns:
            entry["mean_bid_vol1"] = sub["bid_volume_1"].mean()
            entry["mean_ask_vol1"] = sub["ask_volume_1"].mean() if "ask_volume_1" in sub.columns else float("nan")
        rows.append(entry)

    df = pd.DataFrame(rows)
    df.to_csv(_out(out, "spread_liquidity_summary.csv"), index=False)
    print("  ✓ spread_liquidity_summary.csv")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Spread & Liquidity", fontsize=13, fontweight="bold")

    d1_mask  = df["product"].isin(PRODUCTS_DELTA1)
    vev_mask = df["product"].isin(STRIKES_MAP.keys())
    d1       = df[d1_mask]
    vev      = df[vev_mask]

    ax = axes[0]; ax.set_title("Delta-1 – Mean Spread")
    for i, (_, row) in enumerate(d1.iterrows()):
        ax.bar(row["product"], row["mean_spread"],
               color=["#4477AA", "#228833"][i % 2])
        if not math.isnan(row.get("min_spread", float("nan"))):
            ax.errorbar(row["product"], row["mean_spread"],
                        yerr=[[row["mean_spread"] - row["min_spread"]],
                              [row["max_spread"]  - row["mean_spread"]]],
                        fmt="none", color="black", capsize=5)
    ax.set_ylabel("Spread (ticks)")
    ax.set_xticklabels(d1["product"], rotation=15)

    ax = axes[1]; ax.set_title("VEV Options – Mean Spread")
    ax.bar(vev["product"], vev["mean_spread"], color="#6699CC", alpha=0.85)
    for i, (_, row) in enumerate(vev.iterrows()):
        if not math.isnan(row.get("min_spread", float("nan"))):
            ax.errorbar(i, row["mean_spread"],
                        yerr=[[row["mean_spread"] - row["min_spread"]],
                              [row["max_spread"]  - row["mean_spread"]]],
                        fmt="none", color="black", capsize=4)
    ax.set_xticks(range(len(vev)))
    ax.set_xticklabels(vev["product"], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Spread (ticks)")

    _save(fig, _out(out, "13_spread_summary.png"), "13_spread_summary.png")
    return df


# ════════════════════════════════════════════════════════════════════════════
#  COMBINED REPORT  (parameter report + summary)
# ════════════════════════════════════════════════════════════════════════════

def write_reports(
    prices_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    iv_df: pd.DataFrame,
    smile_df: pd.DataFrame,
    signals_df: pd.DataFrame,
    mark_vol: pd.DataFrame,
    spread_df: pd.DataFrame,
    ema_thresholds: dict,
    lag1_acfs: dict,
    out: str,
    round_num: int,
):
    SEP = "=" * 70

    # ── 00_summary.txt  (paste into Claude) ──────────────────────────────────
    L = [SEP, f"IMC Prosperity 4 – Round {round_num} Analysis Summary  ({CURRENCY})", SEP]

    L.append("\n── DELTA-1 ──")
    for prod in PRODUCTS_DELTA1:
        mp   = mid_series(prices_df, prod)
        ret  = mp["mid_price"].pct_change().dropna()
        acf1 = ret.autocorr(lag=1)
        spr  = spread_df[spread_df["product"] == prod]["mean_spread"].values
        thresh = ema_thresholds.get(prod, float("nan"))
        L += [
            f"\n  {prod}",
            f"    Mean price:        {mp['mid_price'].mean():.2f} {CURRENCY}",
            f"    Std price:         {mp['mid_price'].std():.4f}",
            f"    Return ACF(1):     {acf1:.4f}  {'◄ MEAN-REVERT' if acf1 < -0.05 else ''}",
            f"    Mean spread:       {spr[0]:.2f}" if len(spr) else "    Mean spread:  N/A",
            f"    EMA-20 MR thresh:  ±{thresh:.2f}",
        ]

    L.append("\n── VEV IV SUMMARY ──")
    if not iv_df.empty:
        for sym in sorted(STRIKES_MAP.keys(), key=lambda s: STRIKES_MAP[s]):
            sub = iv_df[iv_df["sym"] == sym]["iv"].dropna()
            if sub.empty:
                L.append(f"  {sym}: no valid IV")
            else:
                L.append(f"  {sym}: mean={sub.mean():.3f}  std={sub.std():.3f}"
                         f"  range=[{sub.min():.3f}, {sub.max():.3f}]")
    else:
        L.append("  (IV surface empty – check VELVETFRUIT_EXTRACT is present)")

    L.append("\n── MARK / TRADER SUMMARY ──")
    if not mark_vol.empty:
        top = (mark_vol.groupby("mark")[["buy", "sell"]].sum()
               .sum(axis=1).nlargest(15))
        for mark, vol in top.items():
            L.append(f"  {mark:<22}  vol={vol:.0f}")
    else:
        L.append("  (No trade data or no named traders)")

    if not signals_df.empty:
        L.append("\n── TOP MARK SIGNALS ──")
        for prod in PRODUCTS_DELTA1:
            L.append(f"  {prod}:")
            sig_sub = signals_df[signals_df["product"] == prod].head(8)
            for _, r in sig_sub.iterrows():
                L.append(f"    {r['mark']:15s} {r['side']:5s}  "
                         f"avg={r['avg']:+.1f}bps  t={r['t_stat']:.2f}  n={int(r['n'])}")

    L.append("\n── OPTION ACF SCALPING SIGNALS ──")
    for sym, v in sorted(lag1_acfs.items(), key=lambda x: x[1]):
        note = " ← STRONG SCALP" if v < -0.15 else (" ← moderate" if v < -0.05 else "")
        L.append(f"  {sym}: lag-1 ACF = {v:+.4f}{note}")

    L += [
        "\n── FILES GENERATED ──",
        *[f"  {f}" for f in sorted(os.listdir(out))],
        f"\n── NEXT STEP ──",
        "Upload all PNGs + this 00_summary.txt to Claude.",
        "Ask: 'Here are my Round 4 analysis results – build the final strategy and trader.py'",
        "",
        SEP,
    ]

    summary_text = "\n".join(L)
    path = _out(out, "00_summary.txt")
    with open(path, "w") as f:
        f.write(summary_text)
    print("\n" + summary_text)
    print(f"\n  ✓ 00_summary.txt")

    # ── 00_parameter_report.txt  (recommended code params) ────────────────
    P = [SEP, f"IMC Prosperity 4 – Round {round_num} Parameter Report  ({CURRENCY})", SEP, ""]

    P.append("── DELTA-1 PARAMETERS ──")
    for prod in PRODUCTS_DELTA1:
        mp     = mid_series(prices_df, prod)
        sp     = spread_series(prices_df, prod)
        acf1   = mp["mid_price"].pct_change().dropna().autocorr(lag=1)
        thresh = ema_thresholds.get(prod, 5.0)
        mspr   = sp["spread"].mean() if not sp.empty else float("nan")
        P += [
            f"  {prod}",
            f"    Return ACF(1):              {acf1:.4f}  {'◄ MEAN-REVERT' if acf1 < -0.05 else ''}",
            f"    EMA-20 MR threshold (p90):  ±{thresh:.2f}",
            f"    → take_edge (≈40% of p90):  {max(1, round(thresh * 0.40))} {CURRENCY}",
            f"    → halfspread (≈55% of spr): {max(2, round(mspr * 0.55))} {CURRENCY}",
            "",
        ]

    if not iv_df.empty:
        P.append("── VEV IV SUMMARY ──")
        for sym in sorted(STRIKES_MAP.keys(), key=lambda s: STRIKES_MAP[s]):
            sub = iv_df[iv_df["sym"] == sym]["iv"].dropna()
            if sub.empty: continue
            P.append(f"  {sym}: mean={sub.mean():.3f}  std={sub.std():.3f}"
                     f"  range=[{sub.min():.3f}, {sub.max():.3f}]")
        P.append("")

    if not smile_df.empty:
        P.append("── PRICE DEVIATION FROM SMILE (+= overpriced → sell; −= underpriced → buy) ──")
        for sym in sorted(STRIKES_MAP.keys(), key=lambda s: STRIKES_MAP[s]):
            sub = smile_df[smile_df["sym"] == sym]["price_deviation"].dropna()
            if sub.empty: continue
            mn, sd = sub.mean(), sub.std()
            bias = "→ SELL BIAS" if mn > 0.5 else ("→ BUY BIAS" if mn < -0.5 else "→ NEUTRAL")
            P.append(f"  {sym}: mean_dev={mn:+.3f}  std={sd:.3f}  {bias}")
        P.append("")

        dev_abs = smile_df["iv_deviation"].dropna().abs()
        if not dev_abs.empty:
            p70, p85 = dev_abs.quantile(0.70), dev_abs.quantile(0.85)
            P += [
                "── SMILE TRADING THRESHOLDS ──",
                f"  Medium signal (p70):  IV_DEV_THRESHOLD = {p70:.4f}",
                f"  High   signal (p85):  IV_DEV_THRESHOLD = {p85:.4f}",
                f"  Recommended:          IV_DEV_THRESHOLD = {p70:.4f}",
                f"  MIN_EDGE:             1.0 {CURRENCY}",
                "",
            ]

    if lag1_acfs:
        P.append("── OPTION ACF SCALPING SIGNALS ──")
        for sym, v in sorted(lag1_acfs.items(), key=lambda x: x[1]):
            note = " ← STRONG SCALP" if v < -0.15 else (" ← moderate" if v < -0.05 else "")
            P.append(f"  {sym}: lag-1 ACF = {v:+.4f}{note}")
        P.append("")

    if not signals_df.empty:
        P.append("── TOP MARK SIGNALS (|t| ≥ 2.0, n ≥ 5) ──")
        top = (signals_df[(signals_df["t_stat"].abs() >= 2.0) & (signals_df["n"] >= 5)]
               .sort_values("t_stat", key=abs, ascending=False))
        for _, r in top.iterrows():
            P.append(f"  {r['product']}: {r['mark']} {r['side'].upper():5s}  "
                     f"avg={r['avg']:+.1f}bps  t={r['t_stat']:.2f}  n={int(r['n'])}")
        P += ["",
              "  INTERPRETATION:",
              "    avg > 0 and t > 2.0 → following this Mark/side is profitable (copy them)",
              "    avg < 0 and t < −2.0 → FADE this Mark/side (take opposite side)",
              "    |t| < 2.0 → not statistically significant, ignore",
              ""]

    P.append(SEP)
    param_text = "\n".join(P)
    ppath = _out(out, "00_parameter_report.txt")
    with open(ppath, "w") as f:
        f.write(param_text)
    print(f"\n  ✓ 00_parameter_report.txt")


# ════════════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="IMC Prosperity 4 – Master Round Analysis Tool (combined v3)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    # ── Input modes ──
    ap.add_argument("--data-dir", default=None,
                    help="Folder with price/trade CSVs (auto-detected by round). "
                         "Alternative to --prices / --trades.")
    ap.add_argument("--prices", nargs="+", default=[],
                    help="Price CSV file(s) in day order (explicit mode).")
    ap.add_argument("--trades", nargs="+", default=[],
                    help="Trade CSV file(s) in day order (explicit mode).")
    # ── Round / day config ──
    ap.add_argument("--round",     type=int, default=4, dest="round_num",
                    help="Competition round (3, 4, or 5). Default: 4")
    ap.add_argument("--start-day", type=int, default=1,
                    help="Day label for first file (default: 1).")
    # ── Analysis options ──
    ap.add_argument("--sample",        type=int, default=20,
                    help="IV stride: sample every N timestamps (default 20; "
                         "lower = more accurate but slower).")
    ap.add_argument("--ema-window",    type=int, default=20,
                    help="EMA window for mean-reversion analysis (default 20).")
    ap.add_argument("--forward-ticks", type=int, default=5,
                    help="Mark signal forward-return horizon in ticks (default 5).")
    ap.add_argument("--no-marks",   action="store_true", help="Skip all Mark analysis.")
    ap.add_argument("--no-options", action="store_true", help="Skip options / IV analysis.")
    # ── Output ──
    ap.add_argument("--output", default="./analysis_output",
                    help="Output directory (created if absent). Default: ./analysis_output")
    ap.add_argument("--quiet", "-q", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # ── Resolve TTE map ──
    tte_raw  = TTE_DAYS_PER_ROUND.get(args.round_num, {1: 4, 2: 3, 3: 2})
    keys     = sorted(tte_raw.keys())
    tte_map  = {args.start_day + (k - keys[0]): tte_raw[k] / TRADING_DAYS_PER_YEAR
                for k in keys}

    print(f"\n{'─'*62}")
    print(f"  IMC Prosperity 4 – Master Analyzer v3  |  Round {args.round_num}")
    print(f"  Output: {os.path.abspath(args.output)}")
    print(f"  Currency: {CURRENCY}  |  TTE map: {tte_map}")
    print(f"{'─'*62}\n")

    # ── Resolve file lists ──
    price_files = args.prices
    trade_files = args.trades
    if args.data_dir:
        pf, tf = discover_csvs(args.data_dir, args.round_num)
        if not price_files:
            price_files = pf
        if not trade_files:
            trade_files = tf
        print(f"  Auto-discovered in {args.data_dir}:")
        print(f"    prices: {price_files}")
        print(f"    trades: {trade_files}\n")

    if not price_files:
        sys.exit("ERROR: No price files found. Use --prices or --data-dir.")

    # ── Load ──
    print("▶ Loading price CSVs …")
    prices_df = load_many(price_files, load_prices, base_day=args.start_day)
    if prices_df.empty:
        sys.exit("ERROR: No price data loaded.")
    print(f"  {len(prices_df):,} rows | {prices_df['day'].nunique()} day(s) | "
          f"{prices_df['product'].nunique()} products\n")

    trades_df = pd.DataFrame()
    if trade_files and not args.no_marks:
        print("▶ Loading trade CSVs …")
        trades_df = load_many(trade_files, load_trades, base_day=args.start_day)
        print(f"  {len(trades_df):,} trades | "
              f"{trades_df['buyer'].nunique() + trades_df['seller'].nunique()} unique traders\n")

    # ── Delta-1 ──
    print("▶ [01/13] Delta-1 overview …")
    plot_01_delta1_overview(prices_df, args.output)

    print("▶ [02/13] EMA mean-reversion thresholds …")
    ema_thresholds = plot_02_ema_deviation(prices_df, args.output, ema_win=args.ema_window)

    # ── Options ──
    iv_df = pd.DataFrame(); smile_df = pd.DataFrame(); lag1_acfs: dict[str, float] = {}

    if not args.no_options:
        print(f"\n▶ [03/13] Building IV surface (stride={args.sample}) …")
        iv_df = build_iv_surface(prices_df, tte_map, iv_stride=args.sample)
        if iv_df.empty:
            print("  ⚠  IV surface empty. Check VELVETFRUIT_EXTRACT + VEV_* exist.")
        else:
            print(f"  {len(iv_df):,} IV observations across {iv_df['sym'].nunique()} strikes")
            print("▶ [04/13] Fitting smile deviations …")
            smile_df = fit_smile_deviations(iv_df)
            print("▶ [05/13] Vol smile plot …")
            plot_03_vol_smile(iv_df, args.output)
            print("▶ [06/13] IV price deviation plots …")
            plot_04_iv_price_deviations(smile_df, args.output)
            print("▶ [07/13] Option return ACF …")
            lag1_acfs = plot_05_option_autocorr(prices_df, smile_df, args.output)
            print("▶ [08/13] Delta surface …")
            plot_06_delta_surface(iv_df, args.output)
            print("▶ [09/13] Fair vs market …")
            plot_07_fair_vs_market(smile_df, args.output)
            print("▶ [10/13] IV level trends …")
            plot_08_iv_level_trends(iv_df, args.output)
    else:
        print("  (Options analysis skipped via --no-options)\n")
        for n in range(3, 11):
            print(f"  ✓ chart {n:02d} skipped")

    # ── Mark analysis ──
    mark_vol    = pd.DataFrame()
    signals_df  = pd.DataFrame()

    if not trades_df.empty and not args.no_marks:
        print("\n▶ [11/13] Mark volume table + heatmap …")
        mark_vol = compute_mark_volume_table(trades_df)
        mark_vol.to_csv(_out(args.output, "mark_volume_summary.csv"), index=False)
        print("  ✓ mark_volume_summary.csv")
        plot_09_mark_heatmap(mark_vol, args.output)

        print("▶ [12/13] Mark timing …")
        plot_10_mark_timing(trades_df, args.output)

        print("▶ [12b/13] Mark forward-return signals …")
        signals_df = compute_mark_forward_returns(
            trades_df, prices_df, forward_ticks=args.forward_ticks
        )
        if not signals_df.empty:
            signals_df.to_csv(_out(args.output, "mark_signal_strength.csv"), index=False)
            print("  ✓ mark_signal_strength.csv")
        plot_11_mark_signals(signals_df, args.output)

        print("▶ [12c/13] Mark overlay plots …")
        for prod, fname in [("HYDROGEL_PACK",        "12_mark_hydro"),
                             ("VELVETFRUIT_EXTRACT",  "12_mark_velv")]:
            plot_12_mark_overlay(trades_df, prices_df, prod, args.output, fname)
    else:
        print("  (Mark analysis skipped)\n")

    # ── Spread summary ──
    print("▶ [13/13] Spread & liquidity summary …")
    spread_df = plot_13_spread_summary(prices_df, args.output)

    # ── Reports ──
    print("\n▶ Writing reports (00_summary.txt + 00_parameter_report.txt) …\n")
    write_reports(
        prices_df, trades_df, iv_df, smile_df, signals_df,
        mark_vol, spread_df, ema_thresholds, lag1_acfs,
        args.output, args.round_num,
    )

    print(f"\n{'─'*62}")
    print(f"  ✓  All outputs in: {os.path.abspath(args.output)}/")
    print(f"{'─'*62}\n")
    print("✅  Done. Upload all PNGs + 00_summary.txt to Claude.")


if __name__ == "__main__":
    main()
