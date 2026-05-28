#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════╗
║    IMC Prosperity 4 — Round 5 Overfitting Detector                     ║
║                                                                          ║
║  WHAT THIS IS                                                            ║
║  A backtester tells you "this parameter made money on this data."        ║
║  This tool asks a harder question: "does this signal appear on EVERY     ║
║  day independently, or only on one of them?"                             ║
║  You don't need to trust PnL simulation to trust day-by-day             ║
║  consistency. Consistency is a property of the market, not your model.  ║
║                                                                          ║
║  HOW IT WORKS                                                            ║
║  Treats each CSV day as a held-out fold. For every product it computes   ║
║  ACF(1), spread, EMA-threshold and win-rate independently on each day,  ║
║  then measures how stable those values are across days.                  ║
║                                                                          ║
║    • Overfit risk 0–30  → signal is consistent, safe to trade           ║
║    • Overfit risk 30–60 → signal is mixed, trade with caution           ║
║    • Overfit risk 60+   → signal is noise, do NOT tune params to this   ║
║                                                                          ║
║  QUICK START                                                             ║
║    python r5_overfit_detector.py \\                                      ║
║        prices_round_5_day_2.csv \\                                       ║
║        prices_round_5_day_3.csv \\                                       ║
║        prices_round_5_day_4.csv                                          ║
║                                                                          ║
║    --out DIR      output folder (default: r5_overfit/)                  ║
║    --ema N        EMA window (default: 20)                               ║
║    --thresh-pct N  entry threshold percentile (default: 90)             ║
║    --exit-pct N    exit threshold percentile (default: 25)              ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

import sys, os, math, argparse
from collections import defaultdict
from statistics import mean, stdev

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    HAS_MPL = True
except ImportError:
    print("[WARN] pip install matplotlib numpy pandas  — charts disabled")
    HAS_MPL = False

# ═══════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════

POSITION_LIMIT = 10
EMA_WIN        = 20

GROUPS = {
    "GALAXY_SOUNDS": ["GALAXY_SOUNDS_DARK_MATTER","GALAXY_SOUNDS_BLACK_HOLES",
                      "GALAXY_SOUNDS_PLANETARY_RINGS","GALAXY_SOUNDS_SOLAR_WINDS",
                      "GALAXY_SOUNDS_SOLAR_FLAMES"],
    "SLEEP_POD":     ["SLEEP_POD_SUEDE","SLEEP_POD_LAMB_WOOL","SLEEP_POD_POLYESTER",
                      "SLEEP_POD_NYLON","SLEEP_POD_COTTON"],
    "MICROCHIP":     ["MICROCHIP_CIRCLE","MICROCHIP_OVAL","MICROCHIP_SQUARE",
                      "MICROCHIP_RECTANGLE","MICROCHIP_TRIANGLE"],
    "PEBBLES":       ["PEBBLES_XS","PEBBLES_S","PEBBLES_M","PEBBLES_L","PEBBLES_XL"],
    "ROBOT":         ["ROBOT_VACUUMING","ROBOT_MOPPING","ROBOT_DISHES",
                      "ROBOT_LAUNDRY","ROBOT_IRONING"],
    "UV_VISOR":      ["UV_VISOR_YELLOW","UV_VISOR_AMBER","UV_VISOR_ORANGE",
                      "UV_VISOR_RED","UV_VISOR_MAGENTA"],
    "TRANSLATOR":    ["TRANSLATOR_SPACE_GRAY","TRANSLATOR_ASTRO_BLACK",
                      "TRANSLATOR_ECLIPSE_CHARCOAL","TRANSLATOR_GRAPHITE_MIST",
                      "TRANSLATOR_VOID_BLUE"],
    "PANEL":         ["PANEL_1X2","PANEL_2X2","PANEL_1X4","PANEL_2X4","PANEL_4X4"],
    "OXYGEN_SHAKE":  ["OXYGEN_SHAKE_MORNING_BREATH","OXYGEN_SHAKE_EVENING_BREATH",
                      "OXYGEN_SHAKE_MINT","OXYGEN_SHAKE_CHOCOLATE","OXYGEN_SHAKE_GARLIC"],
    "SNACKPACK":     ["SNACKPACK_CHOCOLATE","SNACKPACK_VANILLA","SNACKPACK_PISTACHIO",
                      "SNACKPACK_STRAWBERRY","SNACKPACK_RASPBERRY"],
}
ALL_PRODUCTS = [p for prods in GROUPS.values() for p in prods]

GROUP_COLORS = {
    "GALAXY_SOUNDS":"#4477AA","SLEEP_POD":"#EE6677","MICROCHIP":"#228833",
    "PEBBLES":"#CCBB44","ROBOT":"#66CCEE","UV_VISOR":"#AA3377",
    "TRANSLATOR":"#888888","PANEL":"#EE8833","OXYGEN_SHAKE":"#44BB99",
    "SNACKPACK":"#9966CC",
}

def product_group(p):
    for g, prods in GROUPS.items():
        if p in prods: return g
    return "OTHER"

def short(p):
    parts = p.split("_")
    return "_".join(parts[-2:]) if len(parts) >= 3 else p


# ═══════════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════

def load_csv(path, day_label):
    df = pd.read_csv(path, sep=";")
    df.columns = [c.strip().lower() for c in df.columns]
    if "product" not in df.columns and "symbol" in df.columns:
        df = df.rename(columns={"symbol": "product"})
    df["product"] = df["product"].str.strip().str.upper()
    df["day"] = day_label
    for col in ["bid_price_1","ask_price_1","mid_price"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "mid_price" not in df.columns and "bid_price_1" in df.columns:
        df["mid_price"] = (df["bid_price_1"] + df["ask_price_1"]) / 2.0
    if "bid_price_1" in df.columns and "ask_price_1" in df.columns:
        df["spread"] = df["ask_price_1"] - df["bid_price_1"]
    return df


# ═══════════════════════════════════════════════════════════════════════════
#  PER-DAY SIGNAL COMPUTATION
# ═══════════════════════════════════════════════════════════════════════════

def _acf1(series):
    """Lag-1 autocorrelation of price changes."""
    chg = np.diff(series)
    if len(chg) < 10: return float("nan")
    c = np.corrcoef(chg[:-1], chg[1:])[0, 1]
    return float(c) if not np.isnan(c) else float("nan")


def _regime(acf, ci):
    if np.isnan(acf): return "UNKNOWN"
    if acf < -ci:     return "MEAN_REVERT"
    if acf >  ci:     return "TRENDING"
    return "RANDOM_WALK"


def simulate_ema_strategy(mid_prices, entry_pct, exit_pct, ema_win=EMA_WIN):
    """
    Entry: |deviation from EMA| > p{entry_pct}.
    Exit:  |deviation| < p{exit_pct}.
    Direction: sell when above EMA (expensive), buy when below (cheap).
    Returns: win_rate, n_trades, trade_pnls list
    """
    if len(mid_prices) < ema_win + 10:
        return float("nan"), 0, []

    ema   = pd.Series(mid_prices).ewm(span=ema_win, adjust=False).mean().values
    dev   = mid_prices - ema
    entry_thresh = float(np.percentile(np.abs(dev[ema_win:]), entry_pct))
    exit_thresh  = float(np.percentile(np.abs(dev[ema_win:]), exit_pct))

    pos, entry_price = 0, 0.0
    trade_pnls = []

    for i in range(ema_win, len(mid_prices) - 1):
        price = mid_prices[i]
        d     = dev[i]

        # Entry
        if pos == 0:
            if d > entry_thresh and pos > -POSITION_LIMIT:   # expensive → short
                pos = -1; entry_price = price
            elif d < -entry_thresh and pos < POSITION_LIMIT: # cheap → long
                pos = 1;  entry_price = price

        # Exit
        elif abs(d) < exit_thresh:
            trade_pnl = pos * (price - entry_price)
            trade_pnls.append(trade_pnl)
            pos = 0; entry_price = 0.0

    # Close any open position at last price
    if pos != 0:
        trade_pnl = pos * (mid_prices[-1] - entry_price)
        trade_pnls.append(trade_pnl)

    if not trade_pnls:
        return float("nan"), 0, []

    wins     = sum(1 for p in trade_pnls if p > 0)
    win_rate = wins / len(trade_pnls)
    return win_rate, len(trade_pnls), trade_pnls


def compute_day_signals(df, day_label, entry_pct, exit_pct):
    """
    For a single day's price CSV, compute per-product signals.
    Returns dict: product -> {acf1, regime, mean_spread, ema_thresh_p90,
                              win_rate, n_trades, total_pnl, mid_std}
    """
    results = {}
    for prod in ALL_PRODUCTS:
        sub = df[df["product"] == prod].sort_values("timestamp")
        mid = sub["mid_price"].dropna().values
        if len(mid) < EMA_WIN + 20:
            continue

        acf = _acf1(mid)
        ci  = 1.96 / math.sqrt(len(mid))

        spread_vals = sub["spread"].dropna().values if "spread" in sub.columns else np.array([])
        mean_spr    = float(np.mean(spread_vals)) if len(spread_vals) > 0 else float("nan")

        ema    = pd.Series(mid).ewm(span=EMA_WIN, adjust=False).mean().values
        dev    = mid - ema
        thresh = float(np.percentile(np.abs(dev[EMA_WIN:]), 90))
        mid_std = float(np.std(np.diff(mid)))

        wr, n_trades, trade_pnls = simulate_ema_strategy(mid, entry_pct, exit_pct)
        total_pnl = sum(trade_pnls) if trade_pnls else 0.0

        results[prod] = {
            "day":         day_label,
            "acf1":        acf,
            "ci95":        ci,
            "regime":      _regime(acf, ci),
            "mean_spread": mean_spr,
            "ema_thresh":  thresh,
            "mid_std":     mid_std,
            "win_rate":    wr,
            "n_trades":    n_trades,
            "total_pnl":   total_pnl,
        }
    return results


# ═══════════════════════════════════════════════════════════════════════════
#  CROSS-DAY CONSISTENCY (OVERFIT DETECTION)
# ═══════════════════════════════════════════════════════════════════════════

def _cv(vals):
    """Coefficient of variation: std / |mean|. Lower = more stable."""
    clean = [v for v in vals if not math.isnan(v)]
    if len(clean) < 2: return float("nan")
    m = mean(clean)
    s = stdev(clean)
    return s / max(abs(m), 1e-9)


def compute_overfit_scores(all_day_signals: dict[int, dict]) -> dict:
    """
    all_day_signals: {day_label: {product: signal_dict}}

    Returns per-product consistency report:
      overfit_risk      0–100 (higher = more likely to be noise)
      signal_stability  dict of per-metric CV across days
      regime_consistent bool (same regime label every day)
      win_rate_stable   bool (win rate consistent ± 10%)
      recommended_action  plain-English instruction
    """
    days    = sorted(all_day_signals.keys())
    n_days  = len(days)

    result = {}
    for prod in ALL_PRODUCTS:
        # Gather per-day values
        acf_vals  = [all_day_signals[d].get(prod, {}).get("acf1",        float("nan")) for d in days]
        spr_vals  = [all_day_signals[d].get(prod, {}).get("mean_spread", float("nan")) for d in days]
        thr_vals  = [all_day_signals[d].get(prod, {}).get("ema_thresh",  float("nan")) for d in days]
        wr_vals   = [all_day_signals[d].get(prod, {}).get("win_rate",    float("nan")) for d in days]
        pnl_vals  = [all_day_signals[d].get(prod, {}).get("total_pnl",   0.0)          for d in days]
        regimes   = [all_day_signals[d].get(prod, {}).get("regime",      "UNKNOWN")    for d in days]

        # Skip products with no data
        valid_acf = [v for v in acf_vals if not math.isnan(v)]
        if not valid_acf:
            continue

        # ── Consistency metrics ────────────────────────────────────────────
        acf_cv  = _cv(acf_vals)
        spr_cv  = _cv(spr_vals)
        thr_cv  = _cv(thr_vals)
        wr_cv   = _cv(wr_vals)

        regime_consistent = len({r for r in regimes if r != "UNKNOWN"}) == 1

        valid_wr = [v for v in wr_vals if not math.isnan(v)]
        if len(valid_wr) >= 2:
            wr_range        = max(valid_wr) - min(valid_wr)
            win_rate_stable = wr_range < 0.12    # < 12pp swing across days
        else:
            wr_range        = float("nan")
            win_rate_stable = False

        pnl_positive_days = sum(1 for p in pnl_vals if p > 0)
        pnl_sign_consistent = (pnl_positive_days == n_days or
                                pnl_positive_days == 0)

        # ── Overfit risk score ─────────────────────────────────────────────
        # Each component 0–100, weighted average
        # ACF stability: most important — if regime flips, you have nothing
        acf_risk  = min(100, (acf_cv or 5.0) * 20)          # weight 35%
        spr_risk  = min(100, (spr_cv or 0.5) * 60)          # weight 20%
        thr_risk  = min(100, (thr_cv or 0.5) * 60)          # weight 20%
        reg_risk  = 0 if regime_consistent else 30           # weight 15%
        wr_risk   = min(100, (wr_cv or 0.5) * 80)           # weight 10%

        overfit_risk = (
            acf_risk  * 0.35 +
            spr_risk  * 0.20 +
            thr_risk  * 0.20 +
            reg_risk  * 0.15 +
            wr_risk   * 0.10
        )

        # ── Mean stats across days ─────────────────────────────────────────
        mean_acf  = mean(valid_acf)
        mean_wr   = mean(valid_wr) if valid_wr else float("nan")
        mean_spr  = mean([v for v in spr_vals if not math.isnan(v)] or [float("nan")])
        mean_thr  = mean([v for v in thr_vals if not math.isnan(v)] or [float("nan")])
        final_regime = (regimes[0] if regime_consistent
                        else ("MEAN_REVERT" if mean_acf < -0.06
                              else "TRENDING" if mean_acf > 0.06
                              else "RANDOM_WALK"))

        # ── Action recommendation ──────────────────────────────────────────
        action = _recommend(overfit_risk, final_regime, mean_wr,
                            win_rate_stable, pnl_sign_consistent, mean_thr, mean_spr)

        # Per-day detail for report
        per_day = {}
        for d in days:
            per_day[d] = all_day_signals[d].get(prod, {})

        result[prod] = {
            "group":             product_group(prod),
            "overfit_risk":      round(overfit_risk, 1),
            "risk_label":        _risk_label(overfit_risk),
            "mean_acf":          mean_acf,
            "acf_cv":            acf_cv,
            "spr_cv":            spr_cv,
            "thr_cv":            thr_cv,
            "wr_cv":             wr_cv,
            "mean_wr":           mean_wr,
            "wr_range":          wr_range,
            "win_rate_stable":   win_rate_stable,
            "regime":            final_regime,
            "regime_consistent": regime_consistent,
            "regimes_per_day":   regimes,
            "pnl_sign_consistent": pnl_sign_consistent,
            "pnl_positive_days": pnl_positive_days,
            "mean_spread":       mean_spr,
            "mean_thresh":       mean_thr,
            "recommended_action": action,
            "per_day":           per_day,
            "acf_per_day":       acf_vals,
            "wr_per_day":        wr_vals,
            "pnl_per_day":       pnl_vals,
        }

    return dict(sorted(result.items(), key=lambda x: x[1]["overfit_risk"]))


def _risk_label(risk):
    if risk < 25:  return "🟢 LOW   — safe to trade"
    if risk < 50:  return "🟡 MEDIUM — trade cautiously"
    if risk < 70:  return "🟠 HIGH  — signal is noisy"
    return              "🔴 EXTREME — do not tune params to this"


def _recommend(risk, regime, wr, wr_stable, pnl_ok, thresh, spread):
    if risk < 25:
        if regime == "MEAN_REVERT" and not math.isnan(wr) and wr > 0.75:
            return (f"STRONG EDGE. EMA mean-reversion entry at ±{thresh:.1f}. "
                    f"Win rate stable at ~{wr*100:.0f}% across all days. "
                    f"This is real signal — tune params here.")
        if regime == "RANDOM_WALK":
            return (f"CONSISTENT RANDOM WALK. Market-making is correct. "
                    f"Spread {spread:.1f} is stable — post quotes within it. "
                    f"Do NOT tune MR params — there is no MR signal.")
        return (f"CONSISTENT signal. Risk {risk:.0f}. "
                f"Regime={regime}. WR={wr*100:.0f}% if applicable.")
    if risk < 50:
        return (f"MODERATE. Signal exists but is noisy across days. "
                f"Use conservative params — do not over-fit to p90 threshold. "
                f"Test threshold at p75 and p85 as well.")
    if risk < 70:
        return (f"HIGH OVERFIT RISK. ACF or spread inconsistent across days. "
                f"If you tune params to one day, they will break on others. "
                f"Consider skipping or using very wide/safe params only.")
    return (f"EXTREME NOISE. Regime flips or metrics wildly inconsistent. "
            f"Any parameter you derive from this product's history is unreliable. "
            f"Skip this product entirely.")


# ═══════════════════════════════════════════════════════════════════════════
#  CHARTS
# ═══════════════════════════════════════════════════════════════════════════

OUTPUT_DIR = "r5_overfit"

def _save(fig, fname):
    path = os.path.join(OUTPUT_DIR, fname)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    print(f"  ✓ {fname}")


def chart_01_overfit_ranking(scores: dict):
    """Full overfit risk ranking — all 50 products."""
    prods  = list(scores.keys())   # already sorted by risk asc
    risks  = [scores[p]["overfit_risk"] for p in prods]
    colors = [GROUP_COLORS.get(scores[p]["group"], "#999") for p in prods]
    labels = [short(p) for p in prods]

    fig, ax = plt.subplots(figsize=(14, max(10, len(prods) * 0.38)))
    bars = ax.barh(labels, risks, color=colors, alpha=0.85)
    ax.axvline(25, color="#228833", ls="--", lw=1.2, label="25 — Safe threshold")
    ax.axvline(50, color="#CCBB44", ls="--", lw=1.2, label="50 — Caution")
    ax.axvline(70, color="#E53935", ls="--", lw=1.2, label="70 — Do not tune")
    ax.set_xlabel("Overfit Risk Score  (0 = perfectly consistent, 100 = pure noise)")
    ax.set_title("Overfitting Risk per Product\n"
                 "Lower = signal is stable across days = safe to build strategy around",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=8)

    for bar, p in zip(bars, prods):
        s = scores[p]
        ann = f"  {s['regime'][:2]}  WR={s['mean_wr']*100:.0f}%" if not math.isnan(s["mean_wr"]) else f"  {s['regime'][:2]}"
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                ann, va="center", fontsize=5.5)

    from matplotlib.patches import Patch
    patches = [Patch(facecolor=c, label=g) for g, c in GROUP_COLORS.items()]
    ax.legend(handles=patches + [
        plt.Line2D([0],[0], color="#228833", ls="--", label="Safe (25)"),
        plt.Line2D([0],[0], color="#CCBB44", ls="--", label="Caution (50)"),
        plt.Line2D([0],[0], color="#E53935", ls="--", label="Do not tune (70)"),
    ], fontsize=6, loc="lower right")

    _save(fig, "01_overfit_risk_ranking.png")


def chart_02_per_day_acf(scores: dict, days: list):
    """ACF(1) per product per day — shows where signal flips."""
    # Focus on products with any MR signal
    mr_prods = [p for p, s in scores.items()
                if abs(s["mean_acf"]) > 0.04 or s["overfit_risk"] < 30]
    mr_prods = mr_prods[:24]
    if not mr_prods: return

    ncols = 4
    nrows = math.ceil(len(mr_prods) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 3.2*nrows), squeeze=False)
    axes = axes.flatten()
    fig.suptitle("ACF(1) Per Day — Is the Regime Consistent?\n"
                 "(bars all same side = consistent = real signal; mixed = noise)",
                 fontsize=12, fontweight="bold")
    day_colors = ["#4477AA", "#EE6677", "#228833"]

    for i, prod in enumerate(mr_prods):
        ax  = axes[i]
        s   = scores[prod]
        col = GROUP_COLORS.get(s["group"], "steelblue")
        acf_vals = s["acf_per_day"]
        x = range(len(days))
        bar_colors = [day_colors[j % len(day_colors)] for j in range(len(days))]
        ax.bar(x, acf_vals, color=bar_colors, alpha=0.85)
        ax.axhline(0,    color="black",  lw=0.8)
        ax.axhline(-0.06, color="green",  ls=":", lw=0.8, label="MR threshold")
        ax.axhline( 0.06, color="orange", ls=":", lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([f"Day{d}" for d in days], fontsize=7)
        regime_flag = "✓ STABLE" if s["regime_consistent"] else "✗ FLIPS"
        ax.set_title(f"{short(prod)}\n"
                     f"risk={s['overfit_risk']:.0f}  {regime_flag}  [{s['regime'][:2]}]",
                     fontsize=7, color=col)
        ax.set_ylabel("ACF(1)", fontsize=6)
        ax.set_ylim(-0.5, 0.5)
        if i == 0:
            ax.legend(fontsize=5)

    for j in range(len(mr_prods), len(axes)):
        axes[j].set_visible(False)

    _save(fig, "02_acf_per_day.png")


def chart_03_winrate_stability(scores: dict, days: list):
    """Win rate per day per product — stable WR = real edge."""
    active = {p: s for p, s in scores.items()
              if not math.isnan(s["mean_wr"]) and s["mean_wr"] > 0}
    if not active: return

    prods   = sorted(active.keys(), key=lambda p: active[p]["overfit_risk"])[:30]
    ncols   = 5
    nrows   = math.ceil(len(prods) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 3.5*nrows), squeeze=False)
    axes = axes.flatten()
    fig.suptitle("Win Rate Per Day (EMA Mean-Reversion Strategy)\n"
                 "Stable bars (all similar height) = real edge; Wildly varying = overfit",
                 fontsize=12, fontweight="bold")
    day_colors = ["#4477AA", "#EE6677", "#228833"]

    for i, prod in enumerate(prods):
        ax  = axes[i]
        s   = active[prod]
        col = GROUP_COLORS.get(s["group"], "steelblue")
        wr_vals = s["wr_per_day"]
        x = range(len(days))
        ax.bar(x, [v*100 if not math.isnan(v) else 0 for v in wr_vals],
               color=day_colors[:len(days)], alpha=0.85)
        ax.axhline(50,  color="black",  lw=0.8, ls="--")
        ax.axhline(75,  color="green",  lw=0.8, ls=":",  label="75% good")
        ax.set_xticks(x)
        ax.set_xticklabels([f"Day{d}" for d in days], fontsize=7)
        stable_flag = "✓ STABLE" if s["win_rate_stable"] else "✗ VARIABLE"
        ax.set_title(f"{short(prod)}\n"
                     f"risk={s['overfit_risk']:.0f}  {stable_flag}  "
                     f"avg={s['mean_wr']*100:.0f}%",
                     fontsize=7, color=col)
        ax.set_ylabel("Win Rate %", fontsize=6)
        ax.set_ylim(0, 110)
        if i == 0: ax.legend(fontsize=5)

    for j in range(len(prods), len(axes)):
        axes[j].set_visible(False)

    _save(fig, "03_winrate_per_day.png")


def chart_04_consistency_scatter(scores: dict):
    """Scatter: overfit risk vs mean ACF. Bottom-left = sweet spot."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("Strategy Safety Map\n"
                 "Bottom-left = consistent + mean-reverting = best products to trade",
                 fontsize=12, fontweight="bold")

    # Left: risk vs ACF
    ax = axes[0]
    for p, s in scores.items():
        if math.isnan(s["mean_acf"]): continue
        col = GROUP_COLORS.get(s["group"], "#999")
        ax.scatter(s["overfit_risk"], s["mean_acf"],
                   color=col, s=55, alpha=0.8, zorder=5)
        if s["overfit_risk"] < 30 or abs(s["mean_acf"]) > 0.08:
            ax.annotate(short(p), (s["overfit_risk"], s["mean_acf"]),
                        fontsize=5, ha="left", va="bottom")
    ax.axhline(-0.06, color="green",  ls="--", lw=1.0,
               label="MR threshold (−0.06)")
    ax.axhline( 0.06, color="orange", ls="--", lw=1.0,
               label="Trend threshold (+0.06)")
    ax.axvline(25, color="grey",   ls=":", lw=1.0, label="Safe zone (risk<25)")
    ax.axvline(50, color="#E53935", ls=":", lw=1.0, label="Danger zone (risk>50)")
    ax.set_xlabel("Overfit Risk  (lower = more consistent across days)")
    ax.set_ylabel("Mean ACF(1)  (negative = mean-reverting)")
    ax.set_title("Overfit Risk vs ACF\n"
                 "Bottom-left quadrant = safe MR products")
    ax.legend(fontsize=7)

    # Shade the sweet spot
    ax.axhspan(-0.5, -0.06, xmin=0, xmax=0.25/ax.get_xlim()[1]
               if ax.get_xlim()[1] > 0 else 0.25, alpha=0.08, color="green")

    # Right: CV components stacked (shows which dimension drives risk)
    ax2 = axes[1]
    prods_sorted = sorted(scores.keys(), key=lambda p: scores[p]["overfit_risk"])[:25]
    labels = [short(p) for p in prods_sorted]
    acf_cvs = [min(scores[p]["acf_cv"] * 35, 100) for p in prods_sorted]
    spr_cvs = [min(scores[p]["spr_cv"] * 20, 100) for p in prods_sorted]
    thr_cvs = [min(scores[p]["thr_cv"] * 20, 100) for p in prods_sorted]
    x = range(len(prods_sorted))
    ax2.bar(x, acf_cvs, label="ACF instability (35%)", color="#4477AA", alpha=0.85)
    ax2.bar(x, spr_cvs, bottom=acf_cvs, label="Spread instability (20%)",
            color="#EE6677", alpha=0.85)
    bottom2 = [a+b for a,b in zip(acf_cvs, spr_cvs)]
    ax2.bar(x, thr_cvs, bottom=bottom2, label="Threshold instability (20%)",
            color="#228833", alpha=0.85)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=45, ha="right", fontsize=6)
    ax2.set_ylabel("Risk contribution (stacked)")
    ax2.set_title("What Is Driving Overfit Risk?\n(top 25 safest products)")
    ax2.legend(fontsize=7)

    _save(fig, "04_consistency_scatter.png")


def chart_05_group_heatmap(scores: dict, days: list):
    """Heatmap: group × day for ACF value and win rate."""
    if not HAS_MPL: return
    fig, axes = plt.subplots(2, 5, figsize=(22, 8))
    fig.suptitle("Per-Group Consistency Heatmap\n"
                 "Each cell = product×day ACF. Uniform color = consistent = safe.",
                 fontsize=12, fontweight="bold")
    axes = axes.flatten()

    for idx, (gname, members) in enumerate(GROUPS.items()):
        ax = axes[idx]
        col = GROUP_COLORS.get(gname, "#666")
        # matrix: rows=products, cols=days
        mat = []
        row_labels = []
        for prod in members:
            if prod not in scores: continue
            row = [scores[prod]["per_day"].get(d, {}).get("acf1", float("nan"))
                   for d in days]
            mat.append(row)
            row_labels.append(short(prod))
        if not mat:
            ax.set_title(gname, fontsize=8, color=col); continue
        mat_arr = np.array(mat, dtype=float)
        im = ax.imshow(mat_arr, cmap="RdYlGn", vmin=-0.3, vmax=0.3, aspect="auto")
        ax.set_xticks(range(len(days)))
        ax.set_xticklabels([f"D{d}" for d in days], fontsize=8)
        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels(row_labels, fontsize=8)
        for r in range(len(row_labels)):
            for c in range(len(days)):
                val = mat_arr[r, c]
                if not np.isnan(val):
                    ax.text(c, r, f"{val:+.2f}", ha="center", va="center",
                            fontsize=6,
                            color="white" if abs(val) > 0.15 else "black")
        ax.set_title(gname, fontsize=9, fontweight="bold", color=col)
        plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)

    _save(fig, "05_group_consistency_heatmap.png")


def chart_06_threshold_stability(scores: dict, days: list):
    """For top safe MR products: EMA threshold per day — how stable is your entry level?"""
    mr_safe = [p for p, s in scores.items()
               if s["regime"] == "MEAN_REVERT" and s["overfit_risk"] < 40]
    if not mr_safe: return

    fig, axes = plt.subplots(1, max(1, len(mr_safe)), figsize=(5*max(1,len(mr_safe)), 5))
    fig.suptitle("EMA Threshold Stability for Mean-Reverting Products\n"
                 "(consistent bars = threshold from CSV data will hold in final sim)",
                 fontsize=11, fontweight="bold")
    if len(mr_safe) == 1: axes = [axes]

    day_colors = ["#4477AA", "#EE6677", "#228833"]
    for i, prod in enumerate(mr_safe):
        ax  = axes[i]
        s   = scores[prod]
        thr = [s["per_day"].get(d, {}).get("ema_thresh", float("nan")) for d in days]
        x   = range(len(days))
        ax.bar(x, [t if not math.isnan(t) else 0 for t in thr],
               color=day_colors[:len(days)], alpha=0.85)
        ax.axhline(s["mean_thresh"], color="black", lw=1.2, ls="--",
                   label=f"Mean={s['mean_thresh']:.1f}")
        ax.set_xticks(x)
        ax.set_xticklabels([f"Day{d}" for d in days], fontsize=8)
        ax.set_title(f"{short(prod)}\n"
                     f"CV={s['thr_cv']:.3f}  risk={s['overfit_risk']:.0f}",
                     fontsize=8, color=GROUP_COLORS.get(s["group"],"steelblue"))
        ax.set_ylabel("EMA Threshold (XIRECS)", fontsize=7)
        ax.legend(fontsize=7)

    _save(fig, "06_threshold_stability.png")


# ═══════════════════════════════════════════════════════════════════════════
#  REPORT
# ═══════════════════════════════════════════════════════════════════════════

def write_report(scores: dict, days: list, out_dir: str):
    SEP  = "═" * 72
    SEP2 = "─" * 72
    lines = [
        SEP,
        "  IMC Prosperity 4 — Round 5 Overfitting Detector Report",
        f"  Days analysed: {days}  |  Products: {len(scores)}",
        SEP, "",
        "  HOW TO READ THIS REPORT",
        "  ─────────────────────────────────────────────────────────────────",
        "  Overfit Risk 0–25  : 🟢 Signal is consistent across all days.",
        "                       Safe to build strategy around these metrics.",
        "                       The parameter you derive will hold in the final sim.",
        "  Overfit Risk 25–50 : 🟡 Signal exists but is noisy.",
        "                       Use conservative/wide parameters — do not",
        "                       over-tune to the exact p90 threshold.",
        "  Overfit Risk 50–70 : 🟠 High noise. Regime or spread flips across days.",
        "                       If you tune parameters, they will break.",
        "  Overfit Risk 70+   : 🔴 Pure noise. Skip this product.",
        "",
    ]

    # Section 1: Safe products to trade
    safe = [(p, s) for p, s in scores.items() if s["overfit_risk"] < 30]
    lines += [SEP2,
              f"  SECTION 1 — SAFE TO TRADE  ({len(safe)} products, risk < 30)",
              SEP2,
              f"  {'Product':<42} {'Risk':>5} {'Regime':>12} {'MeanWR%':>8} "
              f"{'ACF':>7} {'Spread':>8}  Action"]
    lines.append("  " + "─" * 100)
    for p, s in safe:
        wr_str = f"{s['mean_wr']*100:.0f}%" if not math.isnan(s["mean_wr"]) else "  n/a"
        lines.append(
            f"  {p:<42} {s['overfit_risk']:>5.1f} {s['regime']:>12} {wr_str:>8} "
            f"{s['mean_acf']:>+7.3f} {s['mean_spread']:>8.1f}  {s['recommended_action'][:60]}"
        )
    lines.append("")

    # Section 2: Caution products
    caution = [(p, s) for p, s in scores.items() if 30 <= s["overfit_risk"] < 60]
    lines += [SEP2,
              f"  SECTION 2 — TRADE WITH CAUTION  ({len(caution)} products, risk 30–60)",
              SEP2]
    for p, s in caution:
        lines.append(f"  {p:<42} risk={s['overfit_risk']:>5.1f}  {s['recommended_action'][:70]}")
    lines.append("")

    # Section 3: Do not touch
    danger = [(p, s) for p, s in scores.items() if s["overfit_risk"] >= 60]
    lines += [SEP2,
              f"  SECTION 3 — DO NOT TUNE PARAMS TO THESE  ({len(danger)} products, risk ≥ 60)",
              SEP2]
    for p, s in danger:
        lines.append(f"  {p:<42} risk={s['overfit_risk']:>5.1f}  regime flips: {s['regimes_per_day']}")
    lines.append("")

    # Section 4: Per-day detail for top products
    lines += [SEP2, "  SECTION 4 — PER-DAY DETAIL (safe + MR products)", SEP2]
    mr_safe = [(p, s) for p, s in scores.items()
               if s["overfit_risk"] < 40 and s["regime"] == "MEAN_REVERT"]
    for p, s in mr_safe:
        lines.append(f"\n  {p}  [risk={s['overfit_risk']:.1f}]")
        lines.append(f"    {'Day':<8} {'ACF1':>8} {'Regime':>12} {'WinRate%':>10} "
                     f"{'N_trades':>9} {'EMA_thresh':>11}")
        for d in days:
            pd_ = s["per_day"].get(d, {})
            acf = pd_.get("acf1", float("nan"))
            reg = pd_.get("regime", "?")
            wr  = pd_.get("win_rate", float("nan"))
            nt  = pd_.get("n_trades", 0)
            th  = pd_.get("ema_thresh", float("nan"))
            lines.append(
                f"    Day {d:<5} {acf:>+8.4f} {reg:>12} "
                f"{wr*100:>9.1f}% {nt:>9}  {th:>11.1f}"
                if not math.isnan(acf) else
                f"    Day {d:<5}  (no data)"
            )
        lines.append(f"    → {s['recommended_action']}")

    lines += ["", SEP,
              "  FILES GENERATED",
              *[f"    {f}" for f in sorted(os.listdir(out_dir))],
              "", SEP]

    text = "\n".join(lines)
    path = os.path.join(out_dir, "OVERFIT_REPORT.txt")
    with open(path, "w") as f:
        f.write(text)
    print("\n" + text)
    print(f"\n  ✓ OVERFIT_REPORT.txt")
    return text


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="IMC Prosperity 4 — Round 5 Overfitting Detector",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("csvfiles", nargs="+",
                    help="Price CSV files in day order (e.g. day2.csv day3.csv day4.csv)")
    ap.add_argument("--out",        default="r5_overfit",
                    help="Output directory (default: r5_overfit/)")
    ap.add_argument("--ema",        type=int, default=20,
                    help="EMA window (default: 20)")
    ap.add_argument("--thresh-pct", type=int, default=90,
                    help="Entry threshold percentile (default: 90)")
    ap.add_argument("--exit-pct",   type=int, default=25,
                    help="Exit threshold percentile (default: 25)")
    args = ap.parse_args()

    global OUTPUT_DIR, EMA_WIN
    OUTPUT_DIR = args.out
    EMA_WIN    = args.ema
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"\n{'─'*64}")
    print(f"  IMC Prosperity 4 — Round 5 Overfitting Detector")
    print(f"  Days: {len(args.csvfiles)}  |  EMA: {args.ema}  |  "
          f"Entry p{args.thresh_pct}  Exit p{args.exit_pct}")
    print(f"  Output: {os.path.abspath(OUTPUT_DIR)}")
    print(f"{'─'*64}\n")

    if len(args.csvfiles) < 2:
        sys.exit("ERROR: Provide at least 2 price CSVs to compare across days.")

    # ── Load ──────────────────────────────────────────────────────────────
    print("▶ Loading CSVs …")
    all_day_signals = {}
    day_labels = []
    for i, path in enumerate(args.csvfiles):
        day = i + 2  # Round 5 days are labelled 2, 3, 4
        print(f"  Day {day}: {path}")
        df = load_csv(path, day)
        signals = compute_day_signals(df, day, args.thresh_pct, args.exit_pct)
        all_day_signals[day] = signals
        day_labels.append(day)
        print(f"  → {len(signals)} products with data")

    # ── Score ─────────────────────────────────────────────────────────────
    print("\n▶ Computing cross-day consistency scores …")
    scores = compute_overfit_scores(all_day_signals)
    print(f"  {len(scores)} products scored")

    safe    = sum(1 for s in scores.values() if s["overfit_risk"] < 30)
    caution = sum(1 for s in scores.values() if 30 <= s["overfit_risk"] < 60)
    danger  = sum(1 for s in scores.values() if s["overfit_risk"] >= 60)
    print(f"  🟢 Safe ({safe})  🟡 Caution ({caution})  🔴 Danger ({danger})")

    # ── Charts ────────────────────────────────────────────────────────────
    if HAS_MPL:
        print("\n▶ Generating charts …")
        chart_01_overfit_ranking(scores)
        chart_02_per_day_acf(scores, day_labels)
        chart_03_winrate_stability(scores, day_labels)
        chart_04_consistency_scatter(scores)
        chart_05_group_heatmap(scores, day_labels)
        chart_06_threshold_stability(scores, day_labels)

    # ── Report ────────────────────────────────────────────────────────────
    print("\n▶ Writing report …\n")
    write_report(scores, day_labels, OUTPUT_DIR)

    print(f"\n{'─'*64}")
    print(f"  ✓ All outputs in: {os.path.abspath(OUTPUT_DIR)}/")
    print(f"{'─'*64}")
    print("\n✅ Done.")
    print("   Paste OVERFIT_REPORT.txt + charts into Claude.")
    print("   Only tune parameters for 🟢 LOW risk products.")


if __name__ == "__main__":
    main()
