#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║        IMC Prosperity 4 — Round 4 Options Analyzer  (r4_options_analyzer)  ║
║                                                                              ║
║  Three tools in one file:                                                    ║
║                                                                              ║
║  1. LOG ANALYZER      — parse your .log file, analyse VEV options,          ║
║                         IV smile, per-trade edge, and Mark counterparties   ║
║                                                                              ║
║  2. EXOTIC PRICER     — price the AETHER_CRYSTAL manual-trading products    ║
║                         (Chooser, Binary Put, Knock-Out Put, Vanillas)       ║
║                         using Monte Carlo + Black-Scholes                    ║
║                                                                              ║
║  3. MARK PROFILER     — given a log file, rank every Mark counterparty       ║
║                         by volume, timing regularity, and predictability     ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  QUICK START                                                                 ║
║                                                                              ║
║  # Analyse your log + generate charts:                                       ║
║    python r4_options_analyzer.py log  <your_result.log>                     ║
║                                                                              ║
║  # Price exotic options interactively:                                       ║
║    python r4_options_analyzer.py price                                       ║
║                                                                              ║
║  # Mark counterparty deep-dive:                                              ║
║    python r4_options_analyzer.py marks  <your_result.log>                   ║
║                                                                              ║
║  Dependencies:  pip install matplotlib numpy scipy                           ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import sys, re, json, math, os, time, random
from collections import defaultdict
from statistics import mean, stdev, median

# ── Optional deps ──────────────────────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import numpy as np
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("[WARN] matplotlib/numpy not found.  pip install matplotlib numpy scipy")

try:
    from scipy.stats import norm as sp_norm
    from scipy.optimize import brentq
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

VOUCHER_STRIKES = {
    "VEV_4000": 4000, "VEV_4500": 4500,
    "VEV_5000": 5000, "VEV_5100": 5100,
    "VEV_5200": 5200, "VEV_5300": 5300,
    "VEV_5400": 5400, "VEV_5500": 5500,
    "VEV_6000": 6000, "VEV_6500": 6500,
}
POSITION_LIMITS = {
    "HYDROGEL_PACK": 200,
    "VELVETFRUIT_EXTRACT": 200,
    **{k: 150 for k in VOUCHER_STRIKES},
}
CURRENCY       = "XIRECS"
TOTAL_DAYS     = 3
TICKS_PER_DAY  = 1_000_000
OUTPUT_DIR     = "r4_charts"
MY_ID          = "SUBMISSION"

# AETHER_CRYSTAL simulation parameters (from Round 4 wiki)
AC_SIGMA       = 2.51          # 251% annualized vol
AC_STEPS_PER_DAY = 4
AC_TRADING_DAYS  = 252
AC_DT          = 1.0 / (AC_TRADING_DAYS * AC_STEPS_PER_DAY)  # length of one step in years
WEEK_DAYS      = 5             # 1 week = 5 trading days
WEEK_STEPS     = WEEK_DAYS * AC_STEPS_PER_DAY  # steps per week

# ══════════════════════════════════════════════════════════════════════════════
#  MATHS HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_call(S: float, K: float, T: float, sigma: float, r: float = 0.0) -> float:
    """Black-Scholes European call price."""
    if T <= 0 or sigma <= 0 or S <= 0:
        return max(0.0, S - K)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * math.exp(-0.0 * T) * _cdf(d1) - K * math.exp(-r * T) * _cdf(d2)


def bs_put(S: float, K: float, T: float, sigma: float, r: float = 0.0) -> float:
    return bs_call(S, K, T, sigma, r) - S + K * math.exp(-r * T)


def bs_delta(S: float, K: float, T: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0 or S <= 0:
        return 1.0 if S > K else 0.0
    d1 = (math.log(S / K) + 0.5 * sigma ** 2 * T) / (sigma * math.sqrt(T))
    return _cdf(d1)


def bs_gamma(S: float, K: float, T: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0 or S <= 0:
        return 0.0
    d1 = (math.log(S / K) + 0.5 * sigma ** 2 * T) / (sigma * math.sqrt(T))
    pdf = math.exp(-0.5 * d1 ** 2) / math.sqrt(2 * math.pi)
    return pdf / (S * sigma * math.sqrt(T))


def bs_vega(S: float, K: float, T: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0 or S <= 0:
        return 0.0
    d1 = (math.log(S / K) + 0.5 * sigma ** 2 * T) / (sigma * math.sqrt(T))
    pdf = math.exp(-0.5 * d1 ** 2) / math.sqrt(2 * math.pi)
    return S * math.sqrt(T) * pdf


def solve_iv(S: float, K: float, T: float, market_price: float,
             lo: float = 0.01, hi: float = 4.0, iters: int = 60) -> float | None:
    """Binary search for implied vol; returns None if outside range or no time value."""
    intrinsic = max(0.0, S - K)
    if market_price <= intrinsic + 0.01:
        return None
    try:
        if HAS_SCIPY:
            def f(v):
                return bs_call(S, K, T, v) - market_price
            if f(lo) * f(hi) > 0:
                return None
            iv = brentq(f, lo, hi, xtol=1e-6, maxiter=iters)
        else:
            for _ in range(iters):
                mid = (lo + hi) / 2
                if bs_call(S, K, T, mid) < market_price:
                    lo = mid
                else:
                    hi = mid
            iv = (lo + hi) / 2
        return iv if 0.05 < iv < 3.5 else None
    except Exception:
        return None


def fit_smile_parabola(moneyness_list: list[float], iv_list: list[float]):
    """Fit  IV = a*m^2 + b*m + c  to (moneyness, iv) pairs.
    Returns (coeffs, fitted_fn) where moneyness = log(K/S)."""
    if len(moneyness_list) < 3:
        return None, None
    if HAS_MPL:
        coeffs = np.polyfit(moneyness_list, iv_list, 2)
        def fitted(m):
            return float(np.polyval(coeffs, m))
        return coeffs, fitted
    return None, None


# ══════════════════════════════════════════════════════════════════════════════
#  LOG PARSING  (handles JSON envelope, JSONL, and plain CSV section formats)
# ══════════════════════════════════════════════════════════════════════════════

def load_and_parse(filepath: str):
    print(f"\n>>> Loading: {filepath}")
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        raw = f.read().strip()
    lines = [l for l in raw.splitlines() if l.strip()]
    print(f"  {len(raw):,} bytes  |  {len(lines):,} non-empty lines")

    # JSON envelope
    if raw.startswith("{"):
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            obj = _regex_extract(raw)
        act_csv    = _get(obj, "activitiesLog", "activities_log")
        trades_raw = _get(obj, "tradesLog", "trades_log", "tradeHistory", "trade_history")
        sandbox    = _get(obj, "sandboxLog", "lambdaLog", "logs")
        if isinstance(sandbox, list):
            sandbox = "\n".join(str(e.get("lambdaLog") or e.get("sandboxLog") or e) for e in sandbox)
        activities = _parse_activities(act_csv)
        trades     = (_parse_trades_list(trades_raw) if isinstance(trades_raw, list)
                      else _parse_trades_csv(str(trades_raw)))
        return activities, trades, str(sandbox or "")

    # JSONL
    if lines and lines[0].startswith("{"):
        activities, trades = [], []
        for line in lines:
            a, t = _parse_jsonl_line(line)
            activities.extend(a); trades.extend(t)
        return activities, trades, ""

    # Plain section text
    return _parse_section_text(raw), [], ""


def _get(obj, *keys):
    for k in keys:
        v = obj.get(k)
        if v is not None:
            return v
    return ""


def _regex_extract(raw: str) -> dict:
    obj = {}
    for field in ["activitiesLog", "tradesLog", "lambdaLog", "sandboxLog"]:
        m = re.search(rf'"{field}"\s*:\s*"((?:[^"\\]|\\.)*)"', raw)
        if m:
            obj[field] = m.group(1).replace("\\n", "\n").replace("\\t", "\t")
    return obj


def _parse_activities(csv_string: str) -> list:
    rows = []
    if not csv_string:
        return rows
    for line in str(csv_string).splitlines():
        line = line.strip()
        if not line or re.match(r'^day', line, re.I):
            continue
        parts = line.split(";")
        if len(parts) < 17:
            continue
        try:
            def flt(s):
                s = s.strip(); return float(s) if s else None
            rows.append({
                "day":       int(parts[0]),
                "timestamp": int(parts[1]),
                "product":   parts[2].strip(),
                "bid1_p": flt(parts[3]),  "bid1_v": flt(parts[4]),
                "bid2_p": flt(parts[5]),  "bid2_v": flt(parts[6]),
                "bid3_p": flt(parts[7]),  "bid3_v": flt(parts[8]),
                "ask1_p": flt(parts[9]),  "ask1_v": flt(parts[10]),
                "ask2_p": flt(parts[11]), "ask2_v": flt(parts[12]),
                "ask3_p": flt(parts[13]), "ask3_v": flt(parts[14]),
                "mid_price": flt(parts[15]),
                "pnl":       flt(parts[16]),
            })
        except (ValueError, IndexError):
            continue
    return rows


def _parse_trades_csv(csv_string: str) -> list:
    trades = []
    if not csv_string:
        return trades
    for line in csv_string.splitlines():
        line = line.strip()
        if not line or re.match(r'^timestamp', line, re.I):
            continue
        parts = line.split(";")
        if len(parts) < 7:
            continue
        try:
            trades.append({
                "timestamp": int(parts[0]),
                "buyer":     parts[1].strip(),
                "seller":    parts[2].strip(),
                "symbol":    parts[3].strip(),
                "price":     float(parts[5]),
                "quantity":  int(parts[6]),
            })
        except (ValueError, IndexError):
            continue
    return trades


def _parse_trades_list(trade_list: list) -> list:
    trades = []
    for e in (trade_list or []):
        if not isinstance(e, dict):
            continue
        try:
            trades.append({
                "timestamp": int(e.get("timestamp", 0)),
                "buyer":     str(e.get("buyer", "") or ""),
                "seller":    str(e.get("seller", "") or ""),
                "symbol":    str(e.get("symbol", "") or ""),
                "price":     float(e["price"]),
                "quantity":  int(e["quantity"]),
            })
        except (KeyError, ValueError, TypeError):
            continue
    return trades


def _parse_jsonl_line(line: str):
    activities, trades = [], []
    try:
        obj   = json.loads(line)
        state = obj.get("state") or {}
        if isinstance(state, str):
            try: state = json.loads(state)
            except: state = {}
        ts  = state.get("timestamp", 0)
        day = ts // TICKS_PER_DAY
        obs = state.get("observations", {}) or {}
        for sym, depth in (state.get("order_depths") or {}).items():
            bids = depth.get("buy_orders", {})
            asks = depth.get("sell_orders", {})
            all_bids = sorted([int(p) for p in bids], reverse=True)
            all_asks = sorted([int(p) for p in asks])
            mid = None
            if all_bids and all_asks:
                mid = (all_bids[0] + all_asks[0]) / 2.0
            pnl = None
            activities.append({
                "day": day, "timestamp": ts, "product": sym,
                "bid1_p": all_bids[0] if len(all_bids) > 0 else None,
                "bid2_p": all_bids[1] if len(all_bids) > 1 else None,
                "bid3_p": all_bids[2] if len(all_bids) > 2 else None,
                "ask1_p": all_asks[0] if len(all_asks) > 0 else None,
                "ask2_p": all_asks[1] if len(all_asks) > 1 else None,
                "ask3_p": all_asks[2] if len(all_asks) > 2 else None,
                "bid1_v": None, "bid2_v": None, "bid3_v": None,
                "ask1_v": None, "ask2_v": None, "ask3_v": None,
                "mid_price": mid, "pnl": pnl,
            })
        for sym, ts_trades in (state.get("market_trades") or {}).items():
            for t in (ts_trades or []):
                try:
                    trades.append({
                        "timestamp": ts,
                        "buyer":     str(t.get("buyer", "") or ""),
                        "seller":    str(t.get("seller", "") or ""),
                        "symbol":    sym,
                        "price":     float(t["price"]),
                        "quantity":  int(t["quantity"]),
                    })
                except Exception:
                    pass
    except Exception:
        pass
    return activities, trades


def _parse_section_text(raw: str) -> list:
    activities = []
    in_activities = False
    for line in raw.splitlines():
        if re.search(r'Activities log|activitiesLog', line, re.I):
            in_activities = True
            continue
        if in_activities and re.match(r'^\s*[A-Z]', line) and ";" not in line:
            in_activities = False
        if in_activities:
            activities.extend(_parse_activities(line))
    return activities


# ══════════════════════════════════════════════════════════════════════════════
#  ANALYTICS — LOG MODE
# ══════════════════════════════════════════════════════════════════════════════

def compute_pnl_metrics(activities: list) -> dict:
    by_prod = defaultdict(list)
    for r in activities:
        if r["pnl"] is not None:
            by_prod[r["product"]].append((r["timestamp"], r["pnl"]))

    metrics = {}
    for prod, ts_pnl in by_prod.items():
        ts_pnl.sort()
        pnls = [p for _, p in ts_pnl]
        if not pnls:
            continue
        mids = [r["mid_price"] for r in activities
                if r["product"] == prod and r["mid_price"] is not None]
        metrics[prod] = {
            "final_pnl":   pnls[-1],
            "initial_pnl": pnls[0],
            "net_pnl":     pnls[-1] - pnls[0],
            "peak_pnl":    max(pnls),
            "trough_pnl":  min(pnls),
            "max_drawdown": max(pnls) - min(pnls[pnls.index(max(pnls)):] or [max(pnls)]),
            "sharpe":      _sharpe(pnls),
            "mid_range":   [min(mids), max(mids)] if mids else [None, None],
            "mid_std":     stdev(mids) if len(mids) > 1 else 0.0,
            "ticks":       len(pnls),
        }
    return metrics


def _sharpe(pnls: list[float]) -> float:
    if len(pnls) < 2:
        return 0.0
    rets = [pnls[i] - pnls[i - 1] for i in range(1, len(pnls))]
    m    = mean(rets)
    s    = stdev(rets)
    return (m / s) * math.sqrt(len(rets)) if s > 0 else 0.0


def compute_spread_analysis(activities: list) -> dict:
    by_prod = defaultdict(list)
    for r in activities:
        b, a = r.get("bid1_p"), r.get("ask1_p")
        if b is not None and a is not None and a > b:
            by_prod[r["product"]].append(a - b)
    return {prod: {
        "avg": mean(spreads),
        "min": min(spreads),
        "max": max(spreads),
        "median": median(spreads),
        "n": len(spreads),
    } for prod, spreads in by_prod.items() if spreads}


def compute_iv_surface(activities: list) -> tuple[dict, dict]:
    """
    Returns:
      iv_surface[day][K] = list of {"ts", "S", "iv", "moneyness", "delta"}
      smile[day]         = {"moneyness": [...], "iv": [...], "fitted_fn": fn}
    """
    # Build underlying time-series
    vfe_mids = {}
    for r in activities:
        if r["product"] == "VELVETFRUIT_EXTRACT" and r["mid_price"] is not None:
            vfe_mids[r["timestamp"]] = r["mid_price"]

    iv_surface = defaultdict(lambda: defaultdict(list))

    for sym, K in VOUCHER_STRIKES.items():
        for r in activities:
            if r["product"] != sym or r["mid_price"] is None:
                continue
            ts = r["timestamp"]
            day = r["day"]
            S = vfe_mids.get(ts)
            if S is None:
                # nearest
                candidates = [t for t in vfe_mids if abs(t - ts) < 5000]
                if candidates:
                    S = vfe_mids[min(candidates, key=lambda t: abs(t - ts))]
            if S is None:
                continue
            day_frac = ts / TICKS_PER_DAY
            progress = min(1.0, day_frac / TOTAL_DAYS)
            T = max(1e-6, (5.0 / 365.0) - progress * (1.0 / 365.0))
            iv = solve_iv(S, K, T, r["mid_price"])
            if iv is None:
                continue
            moneyness = math.log(K / S) if S > 0 else 0.0
            delta = bs_delta(S, K, T, iv)
            iv_surface[day][K].append({
                "ts":        ts,
                "S":         S,
                "iv":        iv,
                "moneyness": moneyness,
                "delta":     delta,
                "mid":       r["mid_price"],
                "T":         T,
            })

    # Build daily smiles
    smile = {}
    for day, by_K in iv_surface.items():
        moneyness_all, iv_all = [], []
        for K, pts in by_K.items():
            if pts:
                avg_m  = mean(p["moneyness"] for p in pts)
                avg_iv = mean(p["iv"] for p in pts)
                moneyness_all.append(avg_m)
                iv_all.append(avg_iv)
        coeffs, fitted_fn = fit_smile_parabola(moneyness_all, iv_all)
        smile[day] = {
            "moneyness":  moneyness_all,
            "iv":         iv_all,
            "coeffs":     coeffs.tolist() if HAS_MPL and coeffs is not None else None,
            "fitted_fn":  fitted_fn,
        }

    return dict(iv_surface), dict(smile)


def compute_iv_deviations(iv_surface: dict, smile: dict) -> dict:
    """Per observation: iv_residual = observed_iv - fitted_iv from smile."""
    deviations = defaultdict(list)  # key = (day, K)
    for day, by_K in iv_surface.items():
        fn = smile.get(day, {}).get("fitted_fn")
        for K, pts in by_K.items():
            for p in pts:
                fitted = fn(p["moneyness"]) if fn else p["iv"]
                residual = p["iv"] - fitted
                deviations[(day, K)].append({
                    "ts":       p["ts"],
                    "iv":       p["iv"],
                    "fitted":   fitted,
                    "residual": residual,
                    "S":        p["S"],
                    "T":        p["T"],
                })
    return dict(deviations)


def compute_option_edge(activities: list, trades: list, iv_surface: dict, smile: dict) -> list:
    """
    For each trade where we were buyer or seller, compute the theoretical BS price
    and the edge we captured.  Returns list of trade-edge records.
    """
    # Build fast mid lookups
    vfe_mids_by_ts = {}
    for r in activities:
        if r["product"] == "VELVETFRUIT_EXTRACT" and r["mid_price"] is not None:
            vfe_mids_by_ts[r["timestamp"]] = r["mid_price"]

    # Build per-strike per-day avg IV
    avg_iv_lookup = {}
    for day, by_K in iv_surface.items():
        for K, pts in by_K.items():
            if pts:
                avg_iv_lookup[(day, K)] = mean(p["iv"] for p in pts)

    edge_records = []
    for t in trades:
        sym = t["symbol"]
        if sym not in VOUCHER_STRIKES:
            continue
        K   = VOUCHER_STRIKES[sym]
        ts  = t["timestamp"]
        day = ts // TICKS_PER_DAY
        we_bought = (t["buyer"] == MY_ID or t["buyer"] == "")
        we_sold   = (t["seller"] == MY_ID or t["seller"] == "")
        if not we_bought and not we_sold:
            continue

        # Get underlying
        S = vfe_mids_by_ts.get(ts)
        if S is None:
            candidates = [x for x in vfe_mids_by_ts if abs(x - ts) < 10000]
            if candidates:
                S = vfe_mids_by_ts[min(candidates, key=lambda x: abs(x - ts))]
        if S is None:
            continue

        progress = min(1.0, (ts / TICKS_PER_DAY) / TOTAL_DAYS)
        T = max(1e-6, (5.0 / 365.0) - progress * (1.0 / 365.0))

        sigma = avg_iv_lookup.get((day, K))
        if sigma is None:
            sigma = 0.265  # fallback from R3 analysis
        fair     = bs_call(S, K, T, sigma)
        px       = t["price"]
        edge     = (fair - px) if we_bought else (px - fair)
        delta    = bs_delta(S, K, T, sigma)

        edge_records.append({
            "ts":        ts,
            "day":       day,
            "symbol":    sym,
            "K":         K,
            "S":         S,
            "T":         T,
            "sigma":     sigma,
            "fair":      fair,
            "px":        px,
            "qty":       t["quantity"],
            "side":      "BUY" if we_bought else "SELL",
            "edge":      edge,
            "pnl_contrib": edge * t["quantity"],
            "delta":     delta,
        })
    return edge_records


# ══════════════════════════════════════════════════════════════════════════════
#  MARK COUNTERPARTY ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def compute_mark_profiles(trades: list) -> dict:
    """
    Profile each Mark-* counterparty.
    Returns dict  mark_id -> profile dict.
    """
    profiles = defaultdict(lambda: {
        "total_trades": 0,
        "total_volume": 0,
        "buy_qty":      0,
        "sell_qty":     0,
        "by_product":   defaultdict(lambda: {
            "trades": 0, "buy_qty": 0, "sell_qty": 0,
            "prices": [], "timestamps": [],
        }),
        "timestamps":   [],
        "prices":       [],
    })

    for t in trades:
        for role, other_role in [("buyer", "seller"), ("seller", "buyer")]:
            tid = t.get(role, "")
            if not tid or tid == MY_ID:
                continue
            p = profiles[tid]
            p["total_trades"]  += 1
            p["total_volume"]  += t["quantity"]
            p["timestamps"].append(t["timestamp"])
            p["prices"].append(t["price"])
            if role == "buyer":
                p["buy_qty"] += t["quantity"]
            else:
                p["sell_qty"] += t["quantity"]

            bp = p["by_product"][t["symbol"]]
            bp["trades"] += 1
            bp["timestamps"].append(t["timestamp"])
            bp["prices"].append(t["price"])
            if role == "buyer":
                bp["buy_qty"] += t["quantity"]
            else:
                bp["sell_qty"] += t["quantity"]

    # Post-process
    result = {}
    for tid, p in profiles.items():
        ts = sorted(p["timestamps"])
        intervals = [ts[i] - ts[i - 1] for i in range(1, len(ts))]
        avg_interval   = mean(intervals) if intervals else None
        std_interval   = stdev(intervals) if len(intervals) > 1 else None
        regularity     = (1.0 - std_interval / avg_interval
                          if avg_interval and std_interval is not None and avg_interval > 0
                          else None)
        net_qty        = p["buy_qty"] - p["sell_qty"]
        direction      = ("NET BUYER" if net_qty > p["total_volume"] * 0.15
                          else "NET SELLER" if net_qty < -p["total_volume"] * 0.15
                          else "BALANCED")
        # Classify role
        role_label = _classify_mark_role(avg_interval, regularity, direction,
                                         p["total_volume"], p["by_product"])
        result[tid] = {
            "total_trades":    p["total_trades"],
            "total_volume":    p["total_volume"],
            "buy_qty":         p["buy_qty"],
            "sell_qty":        p["sell_qty"],
            "net_qty":         net_qty,
            "direction":       direction,
            "products":        sorted(p["by_product"].keys()),
            "num_products":    len(p["by_product"]),
            "avg_interval_ts": avg_interval,
            "std_interval_ts": std_interval,
            "regularity_score": regularity,   # 1.0 = perfectly regular, 0 = random
            "role_guess":      role_label,
            "first_ts":        ts[0] if ts else None,
            "last_ts":         ts[-1] if ts else None,
            "by_product":      {sym: dict(d) for sym, d in p["by_product"].items()},
        }
    return result


def _classify_mark_role(avg_interval, regularity, direction, volume, by_product) -> str:
    """
    Heuristic classifier:
    - Market Maker: balanced, regular, many products, tight intervals
    - Liquidity Taker: directional, irregular, short bursts
    - Large Clumsier Bot: high volume, irregular, predictable arrival
    - Trend Follower: directional across time
    """
    prods = len(by_product)
    if avg_interval is None:
        return "UNKNOWN"
    if regularity is not None and regularity > 0.7 and direction == "BALANCED":
        return "MARKET_MAKER (regular, balanced)"
    if direction != "BALANCED" and (regularity is None or regularity < 0.5):
        if volume > 5000:
            return "LARGE_DIRECTIONAL_BOT (potential price mover)"
        return "LIQUIDITY_TAKER (directional, irregular)"
    if regularity is not None and regularity > 0.6 and direction != "BALANCED":
        return "SCHEDULED_DIRECTIONAL (predictable timing + bias)"
    if prods == 1:
        return "SINGLE_PRODUCT_SPECIALIST"
    return "MIXED_PATTERN"


# ══════════════════════════════════════════════════════════════════════════════
#  EXOTIC OPTIONS PRICER  (AETHER_CRYSTAL  —  Round 4 Manual Challenge)
# ══════════════════════════════════════════════════════════════════════════════
#
#  GBM parameters (from wiki):
#    drift (risk-neutral) = 0
#    sigma = 251%  annualized
#    252 trading days/year, 4 steps per day  →  dt = 1/(252*4) years
#
#  Products:
#    Vanilla call / put   — 2-week (10 td) or 3-week (15 td) expiry
#    Chooser              — total expiry 3 weeks; choose at 2 weeks
#    Binary Put           — pays fixed amount if S_T < K; otherwise 0
#    Knock-Out Put        — regular put, but worthless if S ever < barrier
# ══════════════════════════════════════════════════════════════════════════════

def simulate_ac_paths(S0: float, n_sims: int, total_steps: int,
                      sigma: float = AC_SIGMA, dt: float = AC_DT,
                      seed: int = 42) -> "np.ndarray":
    """Returns array shape (n_sims, total_steps+1) of price paths."""
    rng = np.random.default_rng(seed)
    z   = rng.standard_normal((n_sims, total_steps))
    log_increments = (0.0 - 0.5 * sigma ** 2) * dt + sigma * math.sqrt(dt) * z
    log_paths      = np.cumsum(log_increments, axis=1)
    log_paths      = np.hstack([np.zeros((n_sims, 1)), log_paths])
    return S0 * np.exp(log_paths)


def price_vanilla(S0: float, K: float, weeks: int, option_type: str = "call",
                  n_sims: int = 50_000) -> dict:
    """Price a vanilla call or put with BS and MC."""
    T = weeks * WEEK_DAYS / AC_TRADING_DAYS
    if option_type.lower() == "call":
        bs_price = bs_call(S0, K, T, AC_SIGMA)
    else:
        bs_price = bs_put(S0, K, T, AC_SIGMA)
    total_steps = weeks * WEEK_STEPS
    paths = simulate_ac_paths(S0, n_sims, total_steps)
    S_T   = paths[:, -1]
    if option_type.lower() == "call":
        payoffs = np.maximum(S_T - K, 0.0)
    else:
        payoffs = np.maximum(K - S_T, 0.0)
    mc_price = float(np.mean(payoffs))
    mc_se    = float(np.std(payoffs) / math.sqrt(n_sims))
    return {
        "product":   f"Vanilla {option_type.upper()} {weeks}W",
        "S0":        S0, "K": K, "weeks": weeks, "T_years": T,
        "bs_price":  bs_price,
        "mc_price":  mc_price,
        "mc_se":     mc_se,
        "delta":     bs_delta(S0, K, T, AC_SIGMA),
        "gamma":     bs_gamma(S0, K, T, AC_SIGMA),
        "vega":      bs_vega(S0, K, T, AC_SIGMA),
    }


def price_chooser(S0: float, K: float,
                  choose_weeks: int = 2, total_weeks: int = 3,
                  n_sims: int = 50_000) -> dict:
    """
    Chooser option: after choose_weeks the holder picks call or put for the
    remaining (total_weeks - choose_weeks) weeks.

    Analytical: C_chooser = C_vanilla(T_total) + P_vanilla(T_choose, K*e^{-r*(T_total-T_choose)})
    where r=0 here, so = C_vanilla(T_total) + P_vanilla(T_choose, K)

    We also price by MC for verification.
    """
    T_total  = total_weeks  * WEEK_DAYS / AC_TRADING_DAYS
    T_choose = choose_weeks * WEEK_DAYS / AC_TRADING_DAYS
    T_remain = T_total - T_choose

    # Analytical decomposition (Rubinstein 1991): chooser = call(T_total) + put(T_choose)
    # for ATM / zero-rate case; generalise below
    anal_price = bs_call(S0, K, T_total, AC_SIGMA) + bs_put(S0, K, T_choose, AC_SIGMA)

    # MC
    steps_choose = choose_weeks * WEEK_STEPS
    steps_remain = (total_weeks - choose_weeks) * WEEK_STEPS
    steps_total  = total_weeks * WEEK_STEPS

    paths_all = simulate_ac_paths(S0, n_sims, steps_total)
    S_choose  = paths_all[:, steps_choose]

    # At choice point: holder chooses call or put for remaining life
    # Value of call: BS_call(S_choose, K, T_remain, sigma)
    # Value of put:  BS_put(S_choose, K, T_remain, sigma)
    mc_payoffs = np.array([
        max(bs_call(float(s), K, T_remain, AC_SIGMA),
            bs_put(float(s),  K, T_remain, AC_SIGMA))
        for s in S_choose
    ])
    mc_price = float(np.mean(mc_payoffs))
    mc_se    = float(np.std(mc_payoffs) / math.sqrt(n_sims))

    return {
        "product":       "Chooser Option",
        "S0":            S0, "K": K,
        "choose_weeks":  choose_weeks, "total_weeks": total_weeks,
        "anal_price":    anal_price,
        "mc_price":      mc_price,
        "mc_se":         mc_se,
        "interpretation": (
            f"Phase 1 (0→{choose_weeks}w): acts like straddle. "
            f"Phase 2 ({choose_weeks}→{total_weeks}w): acts like call or put. "
            f"Hedge: long call(T={total_weeks}w) + long put(T={choose_weeks}w) "
            f"to replicate Phase 1 exposure."
        ),
    }


def price_binary_put(S0: float, K: float, payout: float, weeks: int,
                     n_sims: int = 50_000) -> dict:
    """Binary (cash-or-nothing) put: pays `payout` if S_T < K, else 0."""
    T = weeks * WEEK_DAYS / AC_TRADING_DAYS
    # BS analytical: payout * N(-d2)
    if T > 0:
        d2 = (math.log(S0 / K) + (-0.5 * AC_SIGMA ** 2) * T) / (AC_SIGMA * math.sqrt(T))
        anal_price = payout * _cdf(-d2)
    else:
        anal_price = payout if S0 < K else 0.0

    # MC
    steps = weeks * WEEK_STEPS
    paths = simulate_ac_paths(S0, n_sims, steps)
    S_T   = paths[:, -1]
    mc_price = float(payout * np.mean(S_T < K))
    mc_se    = float(math.sqrt(mc_price * (payout - mc_price) / n_sims))

    return {
        "product":    "Binary Put",
        "S0":         S0, "K": K, "payout": payout, "weeks": weeks,
        "anal_price": anal_price,
        "mc_price":   mc_price,
        "mc_se":      mc_se,
        "prob_ITM":   float(np.mean(S_T < K)),
    }


def price_knockout_put(S0: float, K: float, barrier: float, weeks: int,
                       n_sims: int = 50_000) -> dict:
    """
    Down-and-out put: regular put that becomes worthless if price ever
    touches or crosses below `barrier` before expiry.
    """
    T     = weeks * WEEK_DAYS / AC_TRADING_DAYS
    steps = weeks * WEEK_STEPS
    paths = simulate_ac_paths(S0, n_sims, steps)

    # Check if barrier ever breached along path
    min_prices = np.min(paths, axis=1)
    knocked_out = min_prices <= barrier
    S_T    = paths[:, -1]
    payoffs = np.where(knocked_out, 0.0, np.maximum(K - S_T, 0.0))

    mc_price    = float(np.mean(payoffs))
    mc_se       = float(np.std(payoffs) / math.sqrt(n_sims))
    vanilla_put = bs_put(S0, K, T, AC_SIGMA)
    ko_discount = vanilla_put - mc_price if mc_price < vanilla_put else 0.0

    return {
        "product":      "Knock-Out Put",
        "S0":           S0, "K": K, "barrier": barrier, "weeks": weeks,
        "vanilla_put":  vanilla_put,
        "mc_price":     mc_price,
        "mc_se":        mc_se,
        "knockout_prob": float(np.mean(knocked_out)),
        "ko_discount":  ko_discount,
        "note": (
            "Cheaper than vanilla put but caps upside: you lose the option "
            "if the spot dips to the barrier even briefly."
        ),
    }


def print_price_result(res: dict):
    """Pretty-print a pricing result dict."""
    print(f"\n  ┌─ {res['product']} {'─'*(50 - len(res['product']))}┐")
    skip = {"product", "interpretation", "note", "by_product"}
    for k, v in res.items():
        if k in skip:
            continue
        if isinstance(v, float):
            print(f"  │  {k:<22} : {v:>14.4f}")
        elif isinstance(v, str):
            print(f"  │  {k:<22} : {v}")
        else:
            print(f"  │  {k:<22} : {v}")
    for extra_k in ["interpretation", "note"]:
        if extra_k in res:
            print(f"  │")
            for line in res[extra_k].split(". "):
                if line.strip():
                    print(f"  │  ℹ  {line.strip()}.")
    print(f"  └{'─'*55}┘")


# ══════════════════════════════════════════════════════════════════════════════
#  CHARTS
# ══════════════════════════════════════════════════════════════════════════════

def _save(fig, name: str):
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✔  {path}")


def chart_pnl_curves(pnl_m: dict, activities: list):
    if not HAS_MPL or not pnl_m:
        return
    by_prod = defaultdict(list)
    for r in activities:
        if r["pnl"] is not None:
            by_prod[r["product"]].append((r["timestamp"], r["pnl"]))

    n   = len(pnl_m)
    cols = 3
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(18, rows * 3.5))
    axes = axes.flatten()
    colours = plt.cm.tab10.colors

    for i, (prod, m) in enumerate(sorted(pnl_m.items())):
        ax  = axes[i]
        pts = sorted(by_prod.get(prod, []))
        if pts:
            ts_  = [p[0] for p in pts]
            pnls = [p[1] for p in pts]
            ax.plot(ts_, pnls, color=colours[i % 10], linewidth=0.8)
            ax.fill_between(ts_, pnls, 0,
                            where=[p >= 0 for p in pnls],
                            alpha=0.15, color="green")
            ax.fill_between(ts_, pnls, 0,
                            where=[p < 0 for p in pnls],
                            alpha=0.15, color="red")
        ax.axhline(0, color="grey", linewidth=0.5, linestyle="--")
        ax.set_title(f"{prod}\n(net={m['net_pnl']:+,.0f} {CURRENCY}, "
                     f"dd={m['max_drawdown']:,.0f})", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("PnL Curves per Product", fontweight="bold", fontsize=13)
    fig.tight_layout()
    _save(fig, "01_pnl_curves.png")


def chart_iv_smile(iv_surface: dict, smile: dict):
    if not HAS_MPL or not iv_surface:
        return
    n   = len(iv_surface)
    fig, axes = plt.subplots(1, max(n, 1), figsize=(6 * max(n, 1), 4.5), squeeze=False)

    for col, day in enumerate(sorted(iv_surface.keys())):
        ax   = axes[0][col]
        by_K = iv_surface[day]
        for K, pts in sorted(by_K.items()):
            if not pts:
                continue
            ms  = [p["moneyness"] for p in pts]
            ivs = [p["iv"] for p in pts]
            ax.scatter(ms, ivs, s=6, alpha=0.4, label=f"K={K}")
        fn = smile.get(day, {}).get("fitted_fn")
        if fn:
            m_grid = np.linspace(
                min(p["moneyness"] for K, pts in by_K.items() for p in pts),
                max(p["moneyness"] for K, pts in by_K.items() for p in pts),
                200
            )
            ax.plot(m_grid, [fn(m) for m in m_grid], "k--", linewidth=1.5,
                    label="Fitted parabola")
        ax.set_title(f"IV Smile — Day {day}")
        ax.set_xlabel("Moneyness  log(K/S)")
        ax.set_ylabel("Implied Volatility")
        ax.legend(fontsize=6, ncol=2)
        ax.grid(alpha=0.3)

    fig.suptitle("Volatility Smile", fontweight="bold")
    fig.tight_layout()
    _save(fig, "02_iv_smile.png")


def chart_iv_deviations(deviations: dict):
    if not HAS_MPL or not deviations:
        return
    keys = sorted(deviations.keys())
    n    = len(keys)
    cols = 3
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(18, rows * 3), squeeze=False)
    axes_flat = axes.flatten()

    for i, key in enumerate(keys):
        day, K = key
        pts   = sorted(deviations[key], key=lambda x: x["ts"])
        if not pts:
            continue
        ts_   = [p["ts"] for p in pts]
        resid = [p["residual"] for p in pts]
        ax    = axes_flat[i]
        ax.plot(ts_, resid, linewidth=0.8, color="steelblue")
        ax.axhline(0, color="grey", linewidth=0.5, linestyle="--")
        mu  = mean(resid)
        sd  = stdev(resid) if len(resid) > 1 else 0
        ax.axhline(mu + sd, color="orange", linewidth=0.7, linestyle=":")
        ax.axhline(mu - sd, color="orange", linewidth=0.7, linestyle=":")
        ax.set_title(f"K={K} Day {day}  μ={mu:+.3f}  σ={sd:.3f}", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_ylabel("IV residual", fontsize=7)

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle("IV Deviations from Fitted Smile (scalping signal)", fontweight="bold")
    fig.tight_layout()
    _save(fig, "03_iv_deviations.png")


def chart_edge_per_trade(edge_records: list):
    if not HAS_MPL or not edge_records:
        return
    by_sym = defaultdict(list)
    for r in edge_records:
        by_sym[r["symbol"]].append(r)

    n    = len(by_sym)
    cols = 3
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(18, rows * 3.5), squeeze=False)
    axes_flat = axes.flatten()

    for i, (sym, recs) in enumerate(sorted(by_sym.items())):
        ax   = axes_flat[i]
        recs = sorted(recs, key=lambda x: x["ts"])
        buys = [r for r in recs if r["side"] == "BUY"]
        sells = [r for r in recs if r["side"] == "SELL"]

        if buys:
            ax.scatter([r["ts"] for r in buys], [r["edge"] for r in buys],
                       color="green", s=20, alpha=0.8, label="BUY edge")
        if sells:
            ax.scatter([r["ts"] for r in sells], [r["edge"] for r in sells],
                       color="red", s=20, alpha=0.8, label="SELL edge")
        ax.axhline(0, color="grey", linewidth=0.5, linestyle="--")
        total_pnl = sum(r["pnl_contrib"] for r in recs)
        ax.set_title(f"{sym}  |  trades={len(recs)}  total_PnL_contrib={total_pnl:+.0f}",
                     fontsize=8)
        ax.set_ylabel(f"Edge ({CURRENCY})", fontsize=7)
        ax.legend(fontsize=6)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=7)

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle("Per-Trade Edge  (BS fair − execution price)", fontweight="bold")
    fig.tight_layout()
    _save(fig, "04_trade_edge.png")


def chart_mark_profiles(mark_profiles: dict):
    if not HAS_MPL or not mark_profiles:
        return
    # Volume bar chart + direction heatmap
    top_marks = sorted(mark_profiles.items(),
                       key=lambda x: x[1]["total_volume"], reverse=True)[:20]
    names  = [m[0] for m in top_marks]
    vols   = [m[1]["total_volume"] for m in top_marks]
    buys   = [m[1]["buy_qty"] for m in top_marks]
    sells  = [m[1]["sell_qty"] for m in top_marks]
    reg    = [m[1]["regularity_score"] or 0 for m in top_marks]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9))

    x = np.arange(len(names))
    ax1.bar(x - 0.2, buys, 0.4, label="Buy qty", color="green", alpha=0.8)
    ax1.bar(x + 0.2, sells, 0.4, label="Sell qty", color="red", alpha=0.8)
    ax1.set_xticks(x); ax1.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
    ax1.set_ylabel("Volume"); ax1.legend()
    ax1.set_title("Mark Counterparty Volume (top 20)")
    ax1.grid(axis="y", alpha=0.3)

    ax2.bar(x, reg, color="steelblue", alpha=0.8)
    ax2.set_xticks(x); ax2.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
    ax2.set_ylabel("Regularity score (1=clock-like, 0=random)")
    ax2.set_ylim(0, 1.1)
    ax2.set_title("Timing Regularity (high = predictable arrival timing)")
    ax2.axhline(0.7, color="orange", linestyle="--", linewidth=1, label="0.7 threshold")
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    _save(fig, "05_mark_profiles.png")


def chart_net_delta(iv_surface: dict, activities: list, pnl_m: dict):
    """Plot the net delta exposure of the option book over time."""
    if not HAS_MPL:
        return
    # We approximate delta from mid-prices and IV surface
    vfe_mids = {}
    for r in activities:
        if r["product"] == "VELVETFRUIT_EXTRACT" and r["mid_price"] is not None:
            vfe_mids[r["timestamp"]] = r["mid_price"]

    # Build per-timestamp avg IVs
    all_ts = sorted(vfe_mids.keys())
    if not all_ts:
        return

    # Collect VEV positions over time (approximate from mid_price slope)
    # We'll just chart delta vs time for the available IV data
    fig, ax = plt.subplots(figsize=(14, 4))
    colours = plt.cm.tab10.colors
    for ci, (sym, K) in enumerate(VOUCHER_STRIKES.items()):
        day_pts = []
        for day, by_K in iv_surface.items():
            for pt in by_K.get(K, []):
                day_pts.append((pt["ts"], pt["delta"]))
        if not day_pts:
            continue
        day_pts.sort()
        ts_   = [p[0] for p in day_pts]
        delts = [p[1] for p in day_pts]
        ax.plot(ts_, delts, linewidth=0.8, label=sym, color=colours[ci % 10])

    ax.axhline(0.5, color="black", linewidth=0.7, linestyle="--", label="ATM delta=0.5")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Delta (per unit)")
    ax.set_title("Option Delta per Strike over Time")
    ax.legend(fontsize=7, ncol=5)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, "06_option_delta.png")


def chart_spread_quality(spread_a: dict):
    if not HAS_MPL or not spread_a:
        return
    prods  = sorted(spread_a.keys())
    avgs   = [spread_a[p]["avg"] for p in prods]
    mins_  = [spread_a[p]["min"] for p in prods]
    maxs_  = [spread_a[p]["max"] for p in prods]

    fig, ax = plt.subplots(figsize=(14, 5))
    x  = np.arange(len(prods))
    ax.bar(x, avgs, color="steelblue", label="Avg spread")
    ax.errorbar(x, avgs,
                yerr=[[a - mn for a, mn in zip(avgs, mins_)],
                      [mx - a for a, mx in zip(avgs, maxs_)]],
                fmt="none", color="black", capsize=5, label="Min-Max range")
    ax.set_xticks(x); ax.set_xticklabels(prods, rotation=45, ha="right")
    ax.set_ylabel(f"Spread ({CURRENCY})")
    ax.set_title("Order Book Spread Quality (L1 Bid-Ask)")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, "07_spread_quality.png")


def chart_exotic_payoffs(S0: float = 100.0):
    """Visualise payoff profiles for the manual trading products."""
    if not HAS_MPL:
        return
    S_range = np.linspace(30, 200, 400)
    K  = S0
    barrier = 0.7 * K

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Chooser at expiry (approximated as straddle payoff at T_choose)
    chooser = np.maximum(S_range - K, 0) + np.maximum(K - S_range, 0)
    axes[0].plot(S_range, chooser, label="Chooser at choice time (≈ straddle)")
    axes[0].axvline(K, color="grey", linestyle="--")
    axes[0].set_title("Chooser Option  (Phase 1 payoff approximation)")
    axes[0].set_xlabel("S"); axes[0].set_ylabel("Value"); axes[0].legend(); axes[0].grid(alpha=0.3)

    # Binary put
    binary = np.where(S_range < K, 1.0, 0.0) * 100
    axes[1].step(S_range, binary, label="Binary Put payout=100", color="orange")
    axes[1].axvline(K, color="grey", linestyle="--", label=f"K={K}")
    axes[1].set_title("Binary Put")
    axes[1].set_xlabel("S"); axes[1].set_ylabel("Payoff"); axes[1].legend(); axes[1].grid(alpha=0.3)

    # Knock-out put vs vanilla put
    vanilla_put = np.maximum(K - S_range, 0)
    # Show KO put as zero below barrier
    ko_put = np.where(S_range > barrier, np.maximum(K - S_range, 0), 0.0)
    axes[2].plot(S_range, vanilla_put, "--", label="Vanilla Put", color="blue")
    axes[2].plot(S_range, ko_put, label="KO Put (given no breach)", color="red")
    axes[2].axvline(barrier, color="orange", linestyle=":", label=f"Barrier={barrier:.0f}")
    axes[2].axvline(K, color="grey", linestyle="--", label=f"K={K}")
    axes[2].set_title("Knock-Out Put vs Vanilla Put")
    axes[2].set_xlabel("S"); axes[2].legend(fontsize=7); axes[2].grid(alpha=0.3)

    fig.suptitle("Exotic Option Payoff Profiles  (AETHER_CRYSTAL)", fontweight="bold")
    fig.tight_layout()
    _save(fig, "08_exotic_payoffs.png")


# ══════════════════════════════════════════════════════════════════════════════
#  REPORT
# ══════════════════════════════════════════════════════════════════════════════

def sep(title: str):
    w = 72
    print(f"\n{'═'*w}")
    print(f"  {title}")
    print(f"{'═'*w}")


def flag(cond: bool, msg: str):
    if cond:
        print(f"  ⚠️  {msg}")


def print_log_report(activities, trades, pnl_m, spread_a, iv_surface,
                     smile, deviations, edge_records, mark_profiles):
    sep("SUMMARY")
    products = sorted(pnl_m.keys())
    total    = sum(m["net_pnl"] for m in pnl_m.values())
    print(f"  Total Net PnL  : {total:+,.2f} {CURRENCY}")
    print(f"  Products       : {products}")
    print(f"  Activity rows  : {len(activities):,}")
    print(f"  Trade rows     : {len(trades):,}")

    sep("PnL BY PRODUCT")
    print(f"  {'Product':<25} {'Net PnL':>12} {'Peak':>10} {'Drawdown':>12} {'Sharpe':>8}")
    print("  " + "-"*72)
    for prod in sorted(pnl_m, key=lambda p: pnl_m[p]["net_pnl"], reverse=True):
        m = pnl_m[prod]
        print(f"  {prod:<25} {m['net_pnl']:>+12,.1f} {m['peak_pnl']:>10,.1f} "
              f"{m['max_drawdown']:>12,.1f} {m['sharpe']:>8.3f}")

    sep("IV SURFACE SUMMARY")
    for day in sorted(iv_surface.keys()):
        print(f"\n  Day {day}:")
        print(f"  {'Strike':<8} {'AvgIV':>8} {'AvgDelta':>10} {'N_pts':>7} {'FittedIV':>10} {'Resid':>8}")
        print("  " + "-"*56)
        fn = smile.get(day, {}).get("fitted_fn")
        for K in sorted(iv_surface[day].keys()):
            pts = iv_surface[day][K]
            if not pts:
                continue
            avg_iv    = mean(p["iv"]       for p in pts)
            avg_delta = mean(p["delta"]    for p in pts)
            avg_m     = mean(p["moneyness"]for p in pts)
            fitted    = fn(avg_m) if fn else float("nan")
            resid     = avg_iv - fitted if fn else float("nan")
            print(f"  {K:<8} {avg_iv:>8.4f} {avg_delta:>10.4f} {len(pts):>7} "
                  f"{fitted:>10.4f} {resid:>+8.4f}")

    sep("OPTIONS EDGE ANALYSIS  (our trades vs BS fair)")
    if edge_records:
        by_sym = defaultdict(list)
        for r in edge_records:
            by_sym[r["symbol"]].append(r)
        print(f"  {'Symbol':<12} {'Trades':>7} {'Avg Edge':>10} {'Total PnL':>12} {'Win%':>7}")
        print("  " + "-"*52)
        for sym in sorted(by_sym):
            recs   = by_sym[sym]
            avg_e  = mean(r["edge"] for r in recs)
            tot_p  = sum(r["pnl_contrib"] for r in recs)
            win    = sum(1 for r in recs if r["edge"] > 0) / len(recs) * 100
            flag(avg_e < 0, f"{sym}: negative average edge — we are consistently the price taker!")
            print(f"  {sym:<12} {len(recs):>7} {avg_e:>+10.3f} {tot_p:>+12.1f} {win:>6.0f}%")
    else:
        print("  (No own-trade data found — buyer/seller IDs not populated in log)")

    sep("MARK COUNTERPARTY PROFILES  (Round 4 key insight)")
    if mark_profiles:
        top = sorted(mark_profiles.items(),
                     key=lambda x: x[1]["total_volume"], reverse=True)[:20]
        print(f"\n  {'Trader':<22} {'Vol':>8} {'Net':>8} {'Direction':<14} {'Regularity':>11} {'Role Guess'}")
        print("  " + "-"*90)
        for name, p in top:
            reg = f"{p['regularity_score']:.2f}" if p["regularity_score"] is not None else "  n/a"
            print(f"  {name:<22} {p['total_volume']:>8} {p['net_qty']:>+8} "
                  f"{p['direction']:<14} {reg:>11}  {p['role_guess']}")
        print(f"\n  💡 Round 4 strategy tip:")
        print(f"     High-regularity Marks = likely market makers; fade their quotes.")
        print(f"     Directional Marks (NET BUYER/SELLER) = may signal trend; trade with them.")
        print(f"     Large-volume irregular Marks = may move price; position ahead of them.")
    else:
        print("  (No counterparty data — buyer/seller IDs were None in this log)")

    sep("SPREAD QUALITY")
    if spread_a:
        print(f"  {'Product':<25} {'AvgSpread':>10} {'Min':>6} {'Max':>6} {'Median':>8}")
        print("  " + "-"*60)
        for prod in sorted(spread_a):
            s = spread_a[prod]
            print(f"  {prod:<25} {s['avg']:>10.2f} {s['min']:>6.1f} "
                  f"{s['max']:>6.1f} {s['median']:>8.2f}")

    sep("STRATEGY FLAGS")
    vev_pnls = {p: pnl_m[p] for p in pnl_m if p.startswith("VEV_")}
    total_vev = sum(m["net_pnl"] for m in vev_pnls.values())
    flag(total_vev < 0, f"Net LOSS on VEV options portfolio: {total_vev:+,.0f} {CURRENCY}")
    for prod, m in pnl_m.items():
        flag(m["max_drawdown"] > abs(m["net_pnl"]) * 3,
             f"{prod}: drawdown ({m['max_drawdown']:.0f}) is >3x net PnL — high path risk!")
    for sym, s in spread_a.items():
        if sym.startswith("VEV_") and s["avg"] < 2:
            print(f"  ✅  {sym}: tight spread ({s['avg']:.1f}) — IV scalping viable")
        elif sym.startswith("VEV_") and s["avg"] > 15:
            flag(True, f"{sym}: very wide spread ({s['avg']:.1f}) — hard to scalp profitably")

    sep("END — attach charts from ./{OUTPUT_DIR}/  to Claude for further advice")


def print_marks_report(mark_profiles: dict):
    sep("MARK COUNTERPARTY DEEP DIVE")
    if not mark_profiles:
        print("  (No data)")
        return
    for name, p in sorted(mark_profiles.items(),
                           key=lambda x: x[1]["total_volume"], reverse=True):
        print(f"\n  ── {name}  (role: {p['role_guess']}) ──")
        print(f"     Volume      : {p['total_volume']:,}  (buy={p['buy_qty']:,} sell={p['sell_qty']:,})")
        print(f"     Direction   : {p['direction']}  (net={p['net_qty']:+,})")
        reg = f"{p['regularity_score']:.3f}" if p["regularity_score"] is not None else "n/a"
        print(f"     Regularity  : {reg}  (avg interval: "
              f"{p['avg_interval_ts']:,.0f} ts)" if p["avg_interval_ts"] else "")
        print(f"     Products    : {', '.join(p['products'][:8])}")
        if p["by_product"]:
            print(f"     {'Product':<20} {'Buy':>7} {'Sell':>7} {'Net':>7}")
            for sym, d in sorted(p["by_product"].items()):
                net = d["buy_qty"] - d["sell_qty"]
                print(f"       {sym:<20} {d['buy_qty']:>7} {d['sell_qty']:>7} {net:>+7}")


# ══════════════════════════════════════════════════════════════════════════════
#  INTERACTIVE PRICER SESSION
# ══════════════════════════════════════════════════════════════════════════════

def run_interactive_pricer():
    print("\n" + "═"*65)
    print("  AETHER_CRYSTAL Exotic Options Pricer")
    print(f"  σ={AC_SIGMA*100:.0f}%  |  252 td/yr  |  4 steps/td")
    print("═"*65)

    try:
        S0_str = input("\n  Enter current AETHER_CRYSTAL spot price (e.g. 100): ").strip()
        S0     = float(S0_str)
    except ValueError:
        S0 = 100.0
        print(f"  Using default S0={S0}")

    n_sims = 100_000
    print(f"\n  Running {n_sims:,} MC simulations …  (may take ~10s)")

    products_to_price = []

    # Vanillas
    for weeks in [2, 3]:
        for opt_type in ["call", "put"]:
            products_to_price.append(("vanilla", {"S0": S0, "K": S0, "weeks": weeks,
                                                   "option_type": opt_type, "n_sims": n_sims}))

    # Chooser (3w expiry, choose at 2w)
    products_to_price.append(("chooser", {"S0": S0, "K": S0,
                                           "choose_weeks": 2, "total_weeks": 3,
                                           "n_sims": n_sims}))

    # Binary put: ask user for payout and strike
    try:
        K_bin    = float(input(f"\n  Binary Put — enter strike (Enter for ATM={S0}): ").strip() or S0)
        payout   = float(input(f"  Binary Put — enter payout amount (Enter for 100): ").strip() or 100)
        weeks_b  = int(input(f"  Binary Put — expiry weeks (2 or 3, Enter for 3): ").strip() or 3)
    except ValueError:
        K_bin, payout, weeks_b = S0, 100.0, 3
    products_to_price.append(("binary_put", {"S0": S0, "K": K_bin,
                                              "payout": payout, "weeks": weeks_b,
                                              "n_sims": n_sims}))

    # Knock-out put
    try:
        K_ko      = float(input(f"\n  KO Put — enter strike (Enter for ATM={S0}): ").strip() or S0)
        barrier   = float(input(f"  KO Put — enter barrier (Enter for 80% of K={0.8*K_ko:.1f}): ")
                          .strip() or 0.8 * K_ko)
        weeks_ko  = int(input(f"  KO Put — expiry weeks (2 or 3, Enter for 3): ").strip() or 3)
    except ValueError:
        K_ko, barrier, weeks_ko = S0, 0.8 * S0, 3
    products_to_price.append(("knockout_put", {"S0": S0, "K": K_ko,
                                                "barrier": barrier, "weeks": weeks_ko,
                                                "n_sims": n_sims}))

    print("\n" + "─"*65)
    print("  RESULTS")
    print("─"*65)

    results = []
    for kind, kwargs in products_to_price:
        if kind == "vanilla":
            res = price_vanilla(**kwargs)
        elif kind == "chooser":
            res = price_chooser(**kwargs)
        elif kind == "binary_put":
            res = price_binary_put(**kwargs)
        elif kind == "knockout_put":
            res = price_knockout_put(**kwargs)
        print_price_result(res)
        results.append(res)

    # Hedge suggestion
    chooser_res = next((r for r in results if r["product"] == "Chooser Option"), None)
    if chooser_res:
        print(f"\n  ┌─ HEDGE SUGGESTION FOR CHOOSER {'─'*30}┐")
        c3 = price_vanilla(S0, S0, 3, "call", 10_000)
        p2 = price_vanilla(S0, S0, 2, "put",  10_000)
        print(f"  │  Replicate chooser via:                               │")
        print(f"  │    Long 1 Vanilla Call 3W  ≈ {c3['bs_price']:>8.4f} {CURRENCY}            │")
        print(f"  │    Long 1 Vanilla Put  2W  ≈ {p2['bs_price']:>8.4f} {CURRENCY}            │")
        print(f"  │    Synthetic total         ≈ {c3['bs_price']+p2['bs_price']:>8.4f} {CURRENCY}            │")
        print(f"  │    Chooser MC price        ≈ {chooser_res['mc_price']:>8.4f} {CURRENCY}            │")
        print(f"  └{'─'*55}┘")

    # Chart payoffs
    if HAS_MPL:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        chart_exotic_payoffs(S0)
        print(f"\n  Payoff diagram → ./{OUTPUT_DIR}/08_exotic_payoffs.png")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def run_log_mode(filepath: str):
    """Full log analysis: IV surface, edge, Marks, charts."""
    t0 = time.time()
    activities, trades, _sandbox = load_and_parse(filepath)
    products = sorted({r["product"] for r in activities})
    print(f"  Activity rows  : {len(activities):,}")
    print(f"  Trade rows     : {len(trades):,}")
    print(f"  Products found : {products}")

    if not activities:
        print("\n⚠️  No activity data found — check log format.")
        sys.exit(1)

    print("\nComputing analytics …")
    pnl_m        = compute_pnl_metrics(activities)
    spread_a     = compute_spread_analysis(activities)
    iv_surf, smile = compute_iv_surface(activities)
    deviations   = compute_iv_deviations(iv_surf, smile)
    edge_recs    = compute_option_edge(activities, trades, iv_surf, smile)
    mark_prof    = compute_mark_profiles(trades)

    print_log_report(activities, trades, pnl_m, spread_a,
                     iv_surf, smile, deviations, edge_recs, mark_prof)

    if HAS_MPL:
        print(f"\nGenerating charts → ./{OUTPUT_DIR}/")
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        chart_pnl_curves(pnl_m, activities)
        chart_iv_smile(iv_surf, smile)
        chart_iv_deviations(deviations)
        chart_edge_per_trade(edge_recs)
        chart_mark_profiles(mark_prof)
        chart_net_delta(iv_surf, activities, pnl_m)
        chart_spread_quality(spread_a)
        chart_exotic_payoffs()
        print(f"  All charts saved in ./{OUTPUT_DIR}/")
    else:
        print("\n[SKIP] Install matplotlib + numpy for charts")

    # Save JSON summary
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    summary = {
        "products": products,
        "total_pnl": sum(m["net_pnl"] for m in pnl_m.values()),
        "pnl_by_product": {p: {k: v for k, v in m.items() if not isinstance(v, list)}
                           for p, m in pnl_m.items()},
        "mark_count": len(mark_prof),
        "top_marks": [
            {"id": n, "volume": p["total_volume"], "role": p["role_guess"],
             "direction": p["direction"], "regularity": p["regularity_score"]}
            for n, p in sorted(mark_prof.items(),
                               key=lambda x: x[1]["total_volume"], reverse=True)[:15]
        ],
    }
    jp = os.path.join(OUTPUT_DIR, "r4_summary.json")
    with open(jp, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSummary JSON → {jp}")
    print(f"Elapsed       : {time.time()-t0:.1f}s")


def run_marks_mode(filepath: str):
    _acts, trades, _ = load_and_parse(filepath)
    mark_prof = compute_mark_profiles(trades)
    print_marks_report(mark_prof)
    if HAS_MPL:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        chart_mark_profiles(mark_prof)


def usage():
    print(__doc__)
    print("Usage:")
    print("  python r4_options_analyzer.py  log    <result.log>")
    print("  python r4_options_analyzer.py  price")
    print("  python r4_options_analyzer.py  marks  <result.log>")


def main():
    args = sys.argv[1:]
    if not args:
        usage()
        sys.exit(0)

    mode = args[0].lower()

    if mode == "log":
        if len(args) < 2:
            print("  ⚠️  Please provide a log file path.")
            usage()
            sys.exit(1)
        run_log_mode(args[1])

    elif mode == "price":
        if not HAS_MPL:
            print("  ⚠️  numpy required for Monte Carlo.  pip install numpy matplotlib scipy")
            sys.exit(1)
        run_interactive_pricer()

    elif mode == "marks":
        if len(args) < 2:
            print("  ⚠️  Please provide a log file path.")
            usage()
            sys.exit(1)
        run_marks_mode(args[1])

    else:
        usage()
        sys.exit(1)


if __name__ == "__main__":
    main()
