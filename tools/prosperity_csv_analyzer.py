#!/usr/bin/env python3
"""
IMC Prosperity 4 — Historical CSV Analyzer
===========================================
Analyzes price and trade history CSVs from any round and produces a
compact, copy-pasteable report optimised for pasting into Claude.

Usage:
    python prosperity_csv_analyzer.py --prices prices_round_3_day_*.csv \
                                       --trades trades_round_3_day_*.csv

Or auto-discover files in a folder:
    python prosperity_csv_analyzer.py --dir ./round3_data

Output is written to stdout AND saved to  prosperity_analysis_report.txt
"""

import sys, re, math, glob, argparse
from pathlib import Path
from collections import defaultdict
from statistics import mean, stdev, median

try:
    import pandas as pd
    import numpy as np
except ImportError:
    print("ERROR: pandas and numpy are required.  Run:  pip install pandas numpy")
    sys.exit(1)


# ── Constants ─────────────────────────────────────────────────────────────────

VOUCHER_RE = re.compile(r"^VEV_(\d+)$")

POSITION_LIMITS = {
    "HYDROGEL_PACK":       200,
    "VELVETFRUIT_EXTRACT": 200,
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def hr(title="", width=72):
    if title:
        return "\n" + "=" * width + f"\n  {title}\n" + "=" * width
    return "\n" + "=" * width


def fmt(val, decimals=2):
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return "N/A"
    if isinstance(val, float):
        return f"{val:,.{decimals}f}"
    return str(val)


def pct(val):
    return fmt(val * 100, 1) + "%" if val is not None else "N/A"


def safe_stdev(lst):
    return stdev(lst) if len(lst) >= 2 else 0.0


def returns(series):
    """First-difference returns of a list."""
    return [series[i] - series[i-1] for i in range(1, len(series))]


def sharpe(series):
    """Annualised-style Sharpe from a PnL series."""
    r = returns(series)
    if len(r) < 2:
        return 0.0
    mu, sd = mean(r), safe_stdev(r)
    return (mu / sd * math.sqrt(len(r))) if sd > 0 else 0.0


def max_drawdown(series):
    peak, dd = series[0], 0.0
    for v in series:
        peak = max(peak, v)
        dd = max(dd, peak - v)
    return dd


def autocorr1(series):
    """Lag-1 autocorrelation of a series."""
    if len(series) < 3:
        return float("nan")
    n = len(series)
    mu = mean(series)
    num = sum((series[i] - mu) * (series[i-1] - mu) for i in range(1, n))
    den = sum((v - mu) ** 2 for v in series)
    return num / den if den > 0 else float("nan")


def hurst(series, max_lag=20):
    """
    Hurst exponent estimate via R/S analysis.
    H < 0.5 → mean-reverting, H ≈ 0.5 → random walk, H > 0.5 → trending.
    """
    if len(series) < max_lag * 2:
        return float("nan")
    lags = range(2, min(max_lag, len(series) // 2))
    rs_vals = []
    for lag in lags:
        chunks = [series[i:i+lag] for i in range(0, len(series)-lag, lag)]
        rs_chunk = []
        for chunk in chunks:
            if len(chunk) < 2:
                continue
            mu = mean(chunk)
            dev = [v - mu for v in chunk]
            cum = [sum(dev[:i+1]) for i in range(len(dev))]
            r_range = max(cum) - min(cum)
            s = safe_stdev(chunk)
            if s > 0:
                rs_chunk.append(r_range / s)
        if rs_chunk:
            rs_vals.append((math.log(lag), math.log(mean(rs_chunk))))
    if len(rs_vals) < 2:
        return float("nan")
    xs = [v[0] for v in rs_vals]
    ys = [v[1] for v in rs_vals]
    mu_x, mu_y = mean(xs), mean(ys)
    num = sum((xs[i]-mu_x)*(ys[i]-mu_y) for i in range(len(xs)))
    den = sum((x-mu_x)**2 for x in xs)
    return num / den if den > 0 else float("nan")


def implied_vol_bs(S, K, T, V):
    """Black-Scholes implied vol via bisection (call, r=0)."""
    from math import log, sqrt, erf
    def cdf(x): return 0.5 * (1.0 + erf(x / sqrt(2.0)))
    def bs(sigma):
        if sigma <= 0 or T <= 0: return 0.0
        d1 = (log(S/K) + 0.5*sigma**2*T) / (sigma*sqrt(T))
        return S*cdf(d1) - K*cdf(d1 - sigma*sqrt(T))
    intrinsic = max(0.0, S - K)
    if V <= intrinsic: return float("nan")
    lo, hi = 1e-4, 5.0
    for _ in range(50):
        m = (lo + hi) / 2
        (lo if bs(m) < V else hi).__class__  # dummy
        if bs(m) < V:
            lo = m
        else:
            hi = m
    iv = (lo + hi) / 2
    return iv if 0.001 < iv < 4.9 else float("nan")


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_prices(paths):
    frames = []
    for p in sorted(paths):
        df = pd.read_csv(p, sep=";", low_memory=False)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df.columns = [c.strip() for c in df.columns]
    num_cols = [c for c in df.columns if c not in ("product",)]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df.sort_values(["day", "timestamp"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def load_trades(paths):
    frames = []
    for p in sorted(paths):
        df = pd.read_csv(p, sep=";", low_memory=False)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df.columns = [c.strip() for c in df.columns]
    for c in ["timestamp", "price", "quantity"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# ── Per-product price analysis ────────────────────────────────────────────────

def analyse_prices(prices_df):
    products = sorted(prices_df["product"].dropna().unique())
    results = {}

    for prod in products:
        sub = prices_df[prices_df["product"] == prod].copy()
        mids = sub["mid_price"].dropna().tolist()
        if not mids:
            continue

        spread_col = sub["ask_price_1"].fillna(0) - sub["bid_price_1"].fillna(0)
        spreads = spread_col[spread_col > 0].tolist()

        bid1v = sub["bid_volume_1"].dropna().tolist()
        ask1v = sub["ask_volume_1"].dropna().tolist()

        # Book depth — sum available volume across all 3 levels
        def total_vol(side):
            cols = [f"{side}_volume_{i}" for i in range(1, 4)]
            avail = [c for c in cols if c in sub.columns]
            return sub[avail].fillna(0).sum(axis=1).tolist()

        bid_depth = total_vol("bid")
        ask_depth = total_vol("ask")

        ret = returns(mids)
        h = hurst(mids)
        ac = autocorr1(mids)

        # PnL if present
        pnls = sub["profit_and_loss"].dropna().tolist()

        results[prod] = {
            "ticks":           len(mids),
            "days":            sorted(sub["day"].dropna().unique().tolist()),
            # Price
            "mid_mean":        mean(mids),
            "mid_min":         min(mids),
            "mid_max":         max(mids),
            "mid_range":       max(mids) - min(mids),
            "mid_stdev":       safe_stdev(mids),
            "mid_final":       mids[-1],
            # Returns
            "ret_mean":        mean(ret) if ret else 0.0,
            "ret_stdev":       safe_stdev(ret) if len(ret) >= 2 else 0.0,
            "ret_skew":        _skew(ret) if len(ret) >= 3 else 0.0,
            "ret_kurtosis":    _kurtosis(ret) if len(ret) >= 4 else 0.0,
            "autocorr_lag1":   ac,
            "hurst_exp":       h,
            # Book
            "avg_spread":      mean(spreads) if spreads else None,
            "min_spread":      min(spreads) if spreads else None,
            "avg_bid1_vol":    mean(bid1v) if bid1v else None,
            "avg_ask1_vol":    mean(ask1v) if ask1v else None,
            "avg_bid_depth":   mean(bid_depth) if bid_depth else None,
            "avg_ask_depth":   mean(ask_depth) if ask_depth else None,
            # PnL (oracle — from price CSV)
            "pnl_final":       pnls[-1] if pnls else None,
            "pnl_peak":        max(pnls) if pnls else None,
            "pnl_trough":      min(pnls) if pnls else None,
            "pnl_max_dd":      max_drawdown(pnls) if len(pnls) >= 2 else None,
            "pnl_sharpe":      sharpe(pnls) if len(pnls) >= 2 else None,
        }

    return results


def _skew(lst):
    if len(lst) < 3: return 0.0
    mu = mean(lst); sd = safe_stdev(lst)
    if sd == 0: return 0.0
    return mean(((v - mu)/sd)**3 for v in lst)


def _kurtosis(lst):
    if len(lst) < 4: return 0.0
    mu = mean(lst); sd = safe_stdev(lst)
    if sd == 0: return 0.0
    return mean(((v - mu)/sd)**4 for v in lst) - 3.0  # excess kurtosis


# ── Per-product trade analysis ────────────────────────────────────────────────

def analyse_trades(trades_df, prices_df):
    if trades_df.empty:
        return {}

    sym_col = "symbol" if "symbol" in trades_df.columns else "product"
    results = {}

    for prod in sorted(trades_df[sym_col].dropna().unique()):
        t = trades_df[trades_df[sym_col] == prod].copy()
        prices = t["price"].dropna().tolist()
        qtys   = t["quantity"].dropna().tolist()
        if not prices:
            continue

        # VWAP
        vwap = (sum(p*q for p,q in zip(prices,qtys)) / sum(qtys)) if sum(qtys) else None

        # Trade-size distribution
        sizes = [abs(q) for q in qtys]

        # Price impact proxy: std of per-trade price changes
        p_changes = [abs(prices[i]-prices[i-1]) for i in range(1, len(prices))]

        results[prod] = {
            "total_trades":   len(t),
            "total_volume":   int(sum(sizes)),
            "avg_trade_size": mean(sizes) if sizes else None,
            "max_trade_size": max(sizes) if sizes else None,
            "vwap":           vwap,
            "price_min":      min(prices),
            "price_max":      max(prices),
            "price_stdev":    safe_stdev(prices),
            "avg_price_move": mean(p_changes) if p_changes else None,
        }

    return results


# ── Voucher / IV surface ──────────────────────────────────────────────────────

def analyse_iv_surface(prices_df):
    """Compute per-strike implied vol stats across all ticks."""
    vouchers = {c: int(VOUCHER_RE.match(c).group(1))
                for c in prices_df["product"].unique()
                if VOUCHER_RE.match(str(c))}
    if not vouchers:
        return {}

    # Build aligned snapshots
    by_ts = defaultdict(dict)
    for _, row in prices_df.iterrows():
        key = (row["day"], row["timestamp"])
        by_ts[key][row["product"]] = row["mid_price"]

    sorted_keys = sorted(by_ts.keys())
    n = len(sorted_keys)

    # Time-to-expiry: assume 5 trading days total, linear decay
    iv_by_strike = defaultdict(list)

    for idx, key in enumerate(sorted_keys):
        snap = by_ts[key]
        S = snap.get("VELVETFRUIT_EXTRACT")
        if not S or S <= 0:
            continue
        progress = idx / max(n - 1, 1)
        T = max((5.0 / 365.0) * (1.0 - progress), 1e-6)

        for sym, K in vouchers.items():
            V = snap.get(sym)
            if not V or V <= 0 or V < max(0.0, S - K):
                continue
            iv = implied_vol_bs(S, K, T, V)
            if iv and not math.isnan(iv):
                iv_by_strike[sym].append(iv)

    out = {}
    for sym, ivs in iv_by_strike.items():
        if not ivs:
            continue
        K = vouchers[sym]
        out[sym] = {
            "strike":    K,
            "avg_iv":    mean(ivs),
            "min_iv":    min(ivs),
            "max_iv":    max(ivs),
            "iv_stdev":  safe_stdev(ivs),
            "n_samples": len(ivs),
        }
    return out


# ── Mean-reversion / momentum regime analysis ─────────────────────────────────

def analyse_regime(price_results):
    """
    Per-product regime classification based on Hurst + autocorrelation.
    Useful for deciding market-making vs directional strategy.
    """
    out = {}
    for prod, r in price_results.items():
        h  = r["hurst_exp"]
        ac = r["autocorr_lag1"]
        if math.isnan(h):
            regime = "unknown"
        elif h < 0.45:
            regime = "mean-reverting  (market-make / fade moves)"
        elif h > 0.55:
            regime = "trending        (momentum / follow breakouts)"
        else:
            regime = "random-walk     (spread-capture / neutral)"
        out[prod] = {
            "hurst":    h,
            "autocorr": ac,
            "regime":   regime,
        }
    return out


# ── Cross-product correlation ─────────────────────────────────────────────────

def analyse_correlations(prices_df):
    """Return correlation matrix of mid-price returns across products."""
    products = sorted(prices_df["product"].dropna().unique())
    series = {}
    for prod in products:
        sub = prices_df[prices_df["product"] == prod][["day","timestamp","mid_price"]].dropna()
        sub = sub.set_index(["day","timestamp"])["mid_price"]
        series[prod] = sub

    # Align on common index
    combined = pd.DataFrame(series).dropna()
    if combined.empty or combined.shape[1] < 2:
        return None, []

    ret_df = combined.diff().dropna()
    corr = ret_df.corr()
    return corr, products


# ── Intraday pattern (early / mid / late) ────────────────────────────────────

def analyse_intraday(prices_df):
    """PnL accrual rate split into thirds of each day."""
    out = {}
    for prod in sorted(prices_df["product"].dropna().unique()):
        sub = prices_df[prices_df["product"] == prod].copy()
        sub = sub.sort_values(["day","timestamp"])
        pnls = sub["profit_and_loss"].dropna().tolist()
        if len(pnls) < 6:
            continue
        n = len(pnls)
        t1, t2 = n//3, 2*n//3
        out[prod] = {
            "early": round(pnls[t1] - pnls[0], 2),
            "mid":   round(pnls[t2] - pnls[t1], 2),
            "late":  round(pnls[-1] - pnls[t2], 2),
            "total": round(pnls[-1] - pnls[0], 2),
        }
    return out


# ── Report builder ────────────────────────────────────────────────────────────

def build_report(price_results, trade_results, iv_surface,
                 regime, corr_matrix, products, intraday,
                 round_tag):
    lines = []
    W = lines.append

    W(f"IMC PROSPERITY — HISTORICAL DATA ANALYSIS REPORT")
    W(f"Round/dataset : {round_tag}")
    W(f"Products found: {', '.join(sorted(price_results.keys()))}")
    W("")

    # ── 1. PRICE OVERVIEW ─────────────────────────────────────────────────────
    W(hr("1. PRICE OVERVIEW"))
    hdr = f"  {'Product':<26} {'Mean':>9} {'Min':>9} {'Max':>9} {'Range':>9} {'StDev':>8} {'Ticks':>6}"
    W(hdr)
    W("  " + "-"*78)
    for prod in sorted(price_results):
        r = price_results[prod]
        W(f"  {prod:<26} {fmt(r['mid_mean']):>9} {fmt(r['mid_min']):>9} "
          f"{fmt(r['mid_max']):>9} {fmt(r['mid_range']):>9} {fmt(r['mid_stdev']):>8} "
          f"{r['ticks']:>6}")

    # ── 2. RETURN STATISTICS ──────────────────────────────────────────────────
    W(hr("2. RETURN STATISTICS  (tick-to-tick mid-price changes)"))
    hdr = f"  {'Product':<26} {'RetMean':>9} {'RetStdev':>9} {'Skew':>7} {'ExKurt':>7} {'AutoCorr':>9}"
    W(hdr)
    W("  " + "-"*72)
    for prod in sorted(price_results):
        r = price_results[prod]
        W(f"  {prod:<26} {fmt(r['ret_mean'],4):>9} {fmt(r['ret_stdev'],4):>9} "
          f"{fmt(r['ret_skew'],3):>7} {fmt(r['ret_kurtosis'],3):>7} "
          f"{fmt(r['autocorr_lag1'],3):>9}")

    # ── 3. REGIME / MEAN-REVERSION ────────────────────────────────────────────
    W(hr("3. REGIME CLASSIFICATION  (Hurst exponent + autocorrelation)"))
    W(f"  H < 0.45 → mean-reverting   H ≈ 0.5 → random walk   H > 0.55 → trending")
    W("")
    hdr = f"  {'Product':<26} {'Hurst':>7} {'AutoCorr':>9}   Regime"
    W(hdr)
    W("  " + "-"*75)
    for prod in sorted(regime):
        r = regime[prod]
        W(f"  {prod:<26} {fmt(r['hurst'],3):>7} {fmt(r['autocorr'],3):>9}   {r['regime']}")

    # ── 4. ORDER BOOK QUALITY ─────────────────────────────────────────────────
    W(hr("4. ORDER BOOK QUALITY"))
    hdr = f"  {'Product':<26} {'AvgSpread':>10} {'MinSpread':>10} {'AvgBid1V':>9} {'AvgAsk1V':>9} {'AvgBidDepth':>12} {'AvgAskDepth':>12}"
    W(hdr)
    W("  " + "-"*90)
    for prod in sorted(price_results):
        r = price_results[prod]
        W(f"  {prod:<26} {fmt(r['avg_spread'] or 0):>10} {fmt(r['min_spread'] or 0):>10} "
          f"{fmt(r['avg_bid1_vol'] or 0,1):>9} {fmt(r['avg_ask1_vol'] or 0,1):>9} "
          f"{fmt(r['avg_bid_depth'] or 0,1):>12} {fmt(r['avg_ask_depth'] or 0,1):>12}")

    # ── 5. MARKET TRADE METRICS ───────────────────────────────────────────────
    W(hr("5. MARKET TRADE METRICS  (historical market trades, no submission filter)"))
    if trade_results:
        hdr = f"  {'Product':<26} {'Trades':>7} {'TotalVol':>9} {'AvgSize':>8} {'MaxSize':>8} {'VWAP':>10} {'PriceStdev':>11}"
        W(hdr)
        W("  " + "-"*82)
        for prod in sorted(trade_results):
            r = trade_results[prod]
            W(f"  {prod:<26} {r['total_trades']:>7} {r['total_volume']:>9} "
              f"{fmt(r['avg_trade_size'],1):>8} {int(r['max_trade_size']):>8} "
              f"{fmt(r['vwap']):>10} {fmt(r['price_stdev']):>11}")
    else:
        W("  (no trade files loaded)")

    # ── 6. PnL METRICS (oracle) ───────────────────────────────────────────────
    W(hr("6. PnL METRICS  (from price CSV oracle column)"))
    has_pnl = [p for p in sorted(price_results) if price_results[p]["pnl_final"] is not None]
    if has_pnl:
        total_final = sum(price_results[p]["pnl_final"] for p in has_pnl)
        W(f"  TOTAL final PnL across all products: {fmt(total_final)}")
        W("")
        hdr = f"  {'Product':<26} {'FinalPnL':>10} {'PeakPnL':>10} {'TroughPnL':>10} {'MaxDD':>10} {'Sharpe':>8}"
        W(hdr)
        W("  " + "-"*76)
        for prod in sorted(has_pnl, key=lambda p: price_results[p]["pnl_final"], reverse=True):
            r = price_results[prod]
            W(f"  {prod:<26} {fmt(r['pnl_final']):>10} {fmt(r['pnl_peak']):>10} "
              f"{fmt(r['pnl_trough']):>10} {fmt(r['pnl_max_dd']):>10} "
              f"{fmt(r['pnl_sharpe'],3):>8}")
    else:
        W("  (profit_and_loss column all zero or missing — oracle data unavailable)")

    # ── 7. INTRADAY PnL ATTRIBUTION ───────────────────────────────────────────
    W(hr("7. INTRADAY PnL ATTRIBUTION  (early / mid / late thirds of session)"))
    if intraday:
        hdr = f"  {'Product':<26} {'Early':>10} {'Mid':>10} {'Late':>10} {'Total':>10}"
        W(hdr)
        W("  " + "-"*66)
        for prod in sorted(intraday, key=lambda p: intraday[p]["total"], reverse=True):
            a = intraday[prod]
            W(f"  {prod:<26} {fmt(a['early']):>10} {fmt(a['mid']):>10} "
              f"{fmt(a['late']):>10} {fmt(a['total']):>10}")
    else:
        W("  (no PnL data for intraday breakdown)")

    # ── 8. VOUCHER IV SURFACE ─────────────────────────────────────────────────
    W(hr("8. VOUCHER IMPLIED VOLATILITY SURFACE"))
    if iv_surface:
        hdr = f"  {'Symbol':<14} {'Strike':>7} {'AvgIV':>8} {'MinIV':>8} {'MaxIV':>8} {'IVStDev':>8} {'N':>6}"
        W(hdr)
        W("  " + "-"*62)
        for sym in sorted(iv_surface, key=lambda s: iv_surface[s]["strike"]):
            iv = iv_surface[sym]
            W(f"  {sym:<14} {iv['strike']:>7} {fmt(iv['avg_iv'],4):>8} "
              f"{fmt(iv['min_iv'],4):>8} {fmt(iv['max_iv'],4):>8} "
              f"{fmt(iv['iv_stdev'],4):>8} {iv['n_samples']:>6}")

        # IV smile summary
        atm_approx = 5255  # approximate VELVETFRUIT mid
        atm_sym = min(iv_surface, key=lambda s: abs(iv_surface[s]["strike"] - atm_approx))
        avg_ivs = {s: iv_surface[s]["avg_iv"] for s in iv_surface}
        iv_vals = sorted(avg_ivs.values())
        W(f"\n  ATM proxy (K≈{atm_approx}): {atm_sym}  avg IV = {fmt(avg_ivs.get(atm_sym,'?'),4)}")
        if len(iv_vals) > 1:
            W(f"  IV smile range:  {fmt(iv_vals[0],4)} – {fmt(iv_vals[-1],4)}  "
              f"(spread = {fmt(iv_vals[-1]-iv_vals[0],4)})")
        W(f"\n  Per-strike avg IVs (sorted by strike):")
        for sym in sorted(iv_surface, key=lambda s: iv_surface[s]["strike"]):
            W(f"    {sym:<14}  K={iv_surface[sym]['strike']:>5}   IV={fmt(iv_surface[sym]['avg_iv'],4)}")
    else:
        W("  (no VEV_* products found — skip)")

    # ── 9. CROSS-PRODUCT CORRELATIONS ─────────────────────────────────────────
    W(hr("9. CROSS-PRODUCT RETURN CORRELATIONS"))
    if corr_matrix is not None and not corr_matrix.empty:
        cols = list(corr_matrix.columns)
        col_w = max(len(c) for c in cols)
        header = " " * (col_w + 4) + "  ".join(f"{c[:8]:>8}" for c in cols)
        W("  " + header)
        for row_prod in cols:
            row_str = f"  {row_prod:<{col_w}}"
            for col_prod in cols:
                val = corr_matrix.loc[row_prod, col_prod]
                row_str += f"  {fmt(val,2):>8}"
            W(row_str)

        # Flag strong correlations (|r| > 0.6, non-diagonal)
        strong = []
        for i, p1 in enumerate(cols):
            for j, p2 in enumerate(cols):
                if j <= i: continue
                val = corr_matrix.loc[p1, p2]
                if abs(val) > 0.6:
                    strong.append((p1, p2, val))
        if strong:
            W("\n  Strong correlations (|r| > 0.6):")
            for p1, p2, val in sorted(strong, key=lambda x: -abs(x[2])):
                W(f"    {p1}  ↔  {p2}   r = {fmt(val,3)}")
    else:
        W("  (insufficient data for correlation)")

    # ── 10. STRATEGY HINTS ────────────────────────────────────────────────────
    W(hr("10. STRATEGY HINTS  (auto-generated from above metrics)"))
    W("")
    for prod in sorted(price_results):
        pr = price_results[prod]
        rg = regime.get(prod, {})
        tr = trade_results.get(prod, {})
        hints = []

        h_val = rg.get("hurst", float("nan"))
        ac_val = rg.get("autocorr", float("nan"))

        if not math.isnan(h_val):
            if h_val < 0.45:
                hints.append("Strong mean-reversion signal — market-making/fade strategy preferred.")
            elif h_val > 0.55:
                hints.append("Trending behaviour — consider momentum / breakout entry logic.")

        if not math.isnan(ac_val):
            if ac_val < -0.15:
                hints.append(f"Negative autocorr ({fmt(ac_val,3)}) — tick-level mean-reversion is profitable; "
                              "fade large single-tick moves.")
            elif ac_val > 0.15:
                hints.append(f"Positive autocorr ({fmt(ac_val,3)}) — price has short-term momentum; "
                              "enter in direction of last move.")

        sp = pr.get("avg_spread")
        if sp is not None and sp < 1.5:
            hints.append(f"Very tight avg spread ({fmt(sp,2)}) — thin edge per unit; "
                         "size and fill rate critical.")
        elif sp is not None and sp > 10:
            hints.append(f"Wide avg spread ({fmt(sp,2)}) — large edge available; "
                         "aggressive market-making viable.")

        if tr.get("avg_trade_size") and tr["avg_trade_size"] > 20:
            hints.append(f"Large avg market trade size ({fmt(tr['avg_trade_size'],1)}) — "
                         "watch for informed order flow.")

        skew = pr.get("ret_skew", 0.0)
        kurt = pr.get("ret_kurtosis", 0.0)
        if abs(skew) > 0.5:
            direction = "right-skewed (positive tail risk)" if skew > 0 else "left-skewed (negative tail risk)"
            hints.append(f"Returns are {direction} (skew={fmt(skew,2)}) — "
                         "adjust stop/take-profit asymmetrically.")
        if kurt > 1.0:
            hints.append(f"Fat tails (excess kurtosis={fmt(kurt,2)}) — "
                         "large price jumps more common than Gaussian; widen risk limits.")

        if hints:
            W(f"  [{prod}]")
            for h in hints:
                W(f"    • {h}")
            W("")

    W(hr("END OF REPORT"))
    W("  Paste this entire output into Claude for strategy analysis.")
    W("=" * 72)

    return "\n".join(lines)


# ── CLI ───────────────────────────────────────────────────────────────────────

def discover_files(folder):
    folder = Path(folder)
    price_files = sorted(folder.glob("prices_*.csv"))
    trade_files = sorted(folder.glob("trades_*.csv"))
    return [str(p) for p in price_files], [str(p) for p in trade_files]


def main():
    parser = argparse.ArgumentParser(
        description="IMC Prosperity historical CSV analyzer — produces a compact report for Claude."
    )
    parser.add_argument("--prices", nargs="*", default=[],
                        help="Price CSV files (glob patterns OK)")
    parser.add_argument("--trades", nargs="*", default=[],
                        help="Trade CSV files (glob patterns OK)")
    parser.add_argument("--dir", default=None,
                        help="Auto-discover prices_*.csv and trades_*.csv in this folder")
    parser.add_argument("--round", default="auto", dest="round_tag",
                        help="Label for the report (e.g. 'Round 3')")
    parser.add_argument("--out", default="prosperity_analysis_report.txt",
                        help="Output file path (default: prosperity_analysis_report.txt)")
    args = parser.parse_args()

    price_paths, trade_paths = [], []

    if args.dir:
        p, t = discover_files(args.dir)
        price_paths.extend(p)
        trade_paths.extend(t)

    # Expand any globs passed on the command line (needed on Windows)
    for pat in (args.prices or []):
        price_paths.extend(glob.glob(pat) or [pat])
    for pat in (args.trades or []):
        trade_paths.extend(glob.glob(pat) or [pat])

    # Deduplicate and verify
    price_paths = sorted(set(price_paths))
    trade_paths = sorted(set(trade_paths))

    if not price_paths:
        print("ERROR: No price CSV files found.  Use --prices or --dir.")
        parser.print_help()
        sys.exit(1)

    print(f"Loading price files : {price_paths}", file=sys.stderr)
    print(f"Loading trade files : {trade_paths or ['(none)']}", file=sys.stderr)

    # Infer round tag from filenames
    round_tag = args.round_tag
    if round_tag == "auto":
        m = re.search(r"round_(\d+)", " ".join(price_paths))
        round_tag = f"Round {m.group(1)}" if m else "unknown"

    prices_df = load_prices(price_paths)
    trades_df = load_trades(trade_paths) if trade_paths else pd.DataFrame()

    print(f"  Price rows loaded : {len(prices_df):,}", file=sys.stderr)
    print(f"  Trade rows loaded : {len(trades_df):,}", file=sys.stderr)

    print("Computing metrics...", file=sys.stderr)
    price_results = analyse_prices(prices_df)
    trade_results = analyse_trades(trades_df, prices_df)
    iv_surface    = analyse_iv_surface(prices_df)
    regime        = analyse_regime(price_results)
    corr_matrix, prods = analyse_correlations(prices_df)
    intraday      = analyse_intraday(prices_df)

    report = build_report(
        price_results, trade_results, iv_surface,
        regime, corr_matrix, prods, intraday,
        round_tag,
    )

    print(report)

    out_path = Path(args.out)
    out_path.write_text(report, encoding="utf-8")
    print(f"\nReport saved to: {out_path.resolve()}", file=sys.stderr)


if __name__ == "__main__":
    main()
