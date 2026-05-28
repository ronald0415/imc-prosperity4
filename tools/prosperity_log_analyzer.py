#!/usr/bin/env python3
"""
IMC Prosperity 4 - Log Analyzer
Handles the actual submission format:
  {"submissionId":"...","activitiesLog":"day;timestamp;...\\n...","lambdaLog":"..."}

Usage: python prosperity_log_analyzer.py <your_submission.log>
"""

import sys, re, json, math
from collections import defaultdict
from statistics import mean, stdev

VOUCHER_STRIKES = {
    "VEV_4000": 4000, "VEV_4500": 4500, "VEV_5000": 5000,
    "VEV_5100": 5100, "VEV_5200": 5200, "VEV_5300": 5300,
    "VEV_5400": 5400, "VEV_5500": 5500, "VEV_6000": 6000,
    "VEV_6500": 6500,
}
POSITION_LIMITS = {
    "HYDROGEL_PACK": 200, "VELVETFRUIT_EXTRACT": 200,
    **{k: 300 for k in VOUCHER_STRIKES},
}

# ─── TOP-LEVEL LOADER ──────────────────────────────────────────────────────────

def load_and_parse(filepath):
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        raw = f.read().strip()

    # ── Print debug info ──────────────────────────────────────────────────────
    lines = [l for l in raw.splitlines() if l.strip()]
    print(f"\n--- DEBUG: first 3 non-empty lines (truncated to 200 chars) ---")
    for i, l in enumerate(lines[:3]):
        print(f"  [{i}] {l[:200]}")
    print(f"  Total non-empty lines: {len(lines)}")
    print("----------------------------------------------------------------")

    # ── Single top-level JSON object (actual IMC format) ─────────────────────
    # {"submissionId":..., "activitiesLog":"...", "lambdaLog":"...", ...}
    if raw.startswith("{"):
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError as e:
            # File might be too big for one parse — try grabbing fields with regex
            print(f"  [WARN] Full JSON parse failed ({e}), trying field extraction...")
            obj = extract_fields_regex(raw)

        print(f"  [FORMAT] Single JSON envelope")
        print(f"  Top-level keys found: {list(obj.keys())[:10]}")

        activities_csv = (obj.get("activitiesLog") or
                          obj.get("activities_log") or
                          obj.get("activitieslog") or "")
        trades_raw     = (obj.get("tradesLog") or
                          obj.get("trades_log") or
                          obj.get("tradeHistory") or
                          obj.get("trade_history") or "")
        sandbox_log    = (obj.get("sandboxLog") or
                          obj.get("lambdaLog") or
                          obj.get("logs") or "")

        # sandboxLog may be a list of per-tick log objects — stringify it
        if isinstance(sandbox_log, list):
            sandbox_log = "\n".join(
                str(entry.get("lambdaLog") or entry.get("sandboxLog") or entry)
                for entry in sandbox_log
            )

        activities = parse_activities_csv(activities_csv)

        # tradeHistory may be a pre-parsed list of dicts rather than a CSV string
        if isinstance(trades_raw, list):
            trades = parse_trades_list(trades_raw)
        else:
            trades = parse_trades_csv(trades_raw)

        return activities, trades, sandbox_log

    # ── Multiple JSON objects, one per line ───────────────────────────────────
    if lines and lines[0].startswith("{"):
        print("  [FORMAT] JSONL (one JSON object per line)")
        activities, trades = [], []
        for line in lines:
            a, t = parse_single_state_json(line)
            activities.extend(a)
            trades.extend(t)
        return activities, trades, ""

    # ── Section-based plain text ──────────────────────────────────────────────
    print("  [FORMAT] Attempting section-based text parse...")
    return parse_section_text(raw), [], ""


def extract_fields_regex(raw):
    """Fallback: extract known string fields via regex when JSON parse fails."""
    obj = {}
    for field in ["activitiesLog", "tradesLog", "lambdaLog", "sandboxLog", "submissionId"]:
        m = re.search(rf'"{field}"\s*:\s*"((?:[^"\\]|\\.)*)"', raw)
        if m:
            obj[field] = m.group(1).replace("\\n", "\n").replace("\\t", "\t")
    return obj


# ─── CSV PARSERS ───────────────────────────────────────────────────────────────

def parse_activities_csv(csv_string):
    """
    Columns (semicolon-separated):
    day;timestamp;product;
    bid_price_1;bid_volume_1;bid_price_2;bid_volume_2;bid_price_3;bid_volume_3;
    ask_price_1;ask_volume_1;ask_price_2;ask_volume_2;ask_price_3;ask_volume_3;
    mid_price;profit_and_loss
    """
    activities = []
    if not csv_string:
        return activities

    for line in csv_string.splitlines():
        line = line.strip()
        if not line or re.match(r'day', line, re.I):
            continue
        parts = line.split(";")
        if len(parts) < 17:
            continue
        try:
            def flt(s):
                s = s.strip()
                return float(s) if s else None

            row = {
                "day":       int(parts[0]),
                "timestamp": int(parts[1]),
                "product":   parts[2].strip(),
                "bid1_p":    flt(parts[3]),
                "bid1_v":    flt(parts[4]),
                "bid2_p":    flt(parts[5]),
                "bid2_v":    flt(parts[6]),
                "bid3_p":    flt(parts[7]),
                "bid3_v":    flt(parts[8]),
                "ask1_p":    flt(parts[9]),
                "ask1_v":    flt(parts[10]),
                "ask2_p":    flt(parts[11]),
                "ask2_v":    flt(parts[12]),
                "ask3_p":    flt(parts[13]),
                "ask3_v":    flt(parts[14]),
                "mid_price": flt(parts[15]),
                "pnl":       flt(parts[16]),
            }
            activities.append(row)
        except (ValueError, IndexError):
            continue

    return activities


def parse_trades_csv(csv_string):
    """
    Columns: timestamp;buyer;seller;symbol;currency;price;quantity
    """
    trades = []
    if not csv_string:
        return trades
    for line in csv_string.splitlines():
        line = line.strip()
        if not line or re.match(r'timestamp', line, re.I):
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


def parse_trades_list(trade_list):
    """
    Parse tradeHistory when it's already a list of dicts (actual IMC format).
    Expected keys per entry: timestamp, buyer, seller, symbol, currency, price, quantity
    """
    trades = []
    for entry in trade_list:
        if not isinstance(entry, dict):
            continue
        try:
            trades.append({
                "timestamp": int(entry.get("timestamp", 0)),
                "buyer":     str(entry.get("buyer",  "") or ""),
                "seller":    str(entry.get("seller", "") or ""),
                "symbol":    str(entry.get("symbol", "") or ""),
                "price":     float(entry["price"]),
                "quantity":  int(entry["quantity"]),
            })
        except (KeyError, ValueError, TypeError):
            continue
    return trades


def parse_single_state_json(line):
    """Parse one JSON state object (JSONL format fallback)."""
    activities, trades = [], []
    try:
        obj   = json.loads(line)
        state = obj.get("state") or {}
        if isinstance(state, str):
            try: state = json.loads(state)
            except: state = {}

        ts  = state.get("timestamp", 0)
        day = obj.get("day", 0)
        order_depths = state.get("order_depths") or {}

        for product, depth in order_depths.items():
            if not isinstance(depth, dict): continue
            buys  = depth.get("buy_orders")  or {}
            sells = depth.get("sell_orders") or {}
            bid_prices = sorted([int(k) for k in buys.keys()],  reverse=True)
            ask_prices = sorted([int(k) for k in sells.keys()])
            bb = bid_prices[0] if bid_prices else None
            ba = ask_prices[0] if ask_prices else None
            mid = (bb + ba) / 2.0 if (bb and ba) else (bb or ba)
            activities.append({
                "day": day, "timestamp": ts, "product": product,
                "bid1_p": bb, "bid1_v": buys.get(str(bb)) if bb else None,
                "ask1_p": ba, "ask1_v": abs(sells.get(str(ba), 0)) if ba else None,
                "mid_price": mid, "pnl": None,
            })
    except:
        pass
    return activities, trades


def parse_section_text(raw):
    activities = []
    in_activities = False
    for line in raw.splitlines():
        l = line.strip()
        if re.match(r'activities\s*log', l, re.I): in_activities = True;  continue
        if re.match(r'(sandbox|trade)',  l, re.I): in_activities = False; continue
        if in_activities and l:
            activities.extend(parse_activities_csv(l))
    return activities


# ─── ANALYTICS ─────────────────────────────────────────────────────────────────

def cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def compute_pnl_metrics(activities):
    by_product = defaultdict(list)
    for row in activities:
        by_product[row["product"]].append(row)

    summary = {}
    for prod, rows in by_product.items():
        rows.sort(key=lambda r: (r["day"], r["timestamp"]))
        pnls = [r["pnl"] for r in rows if r["pnl"] is not None]
        mids = [r["mid_price"] for r in rows if r["mid_price"] is not None]
        if not pnls: continue

        peak, max_dd = pnls[0], 0.0
        for p in pnls:
            if p > peak: peak = p
            max_dd = max(max_dd, peak - p)

        changes = [pnls[i] - pnls[i-1] for i in range(1, len(pnls))]
        sharpe = 0.0
        if len(changes) > 1:
            mu, sd = mean(changes), stdev(changes)
            sharpe = (mu / sd * math.sqrt(len(changes))) if sd > 0 else 0.0

        spreads = [r["ask1_p"] - r["bid1_p"]
                   for r in rows if r.get("bid1_p") and r.get("ask1_p")]

        summary[prod] = {
            "final_pnl":         round(pnls[-1], 2),
            "peak_pnl":          round(max(pnls), 2),
            "trough_pnl":        round(min(pnls), 2),
            "max_drawdown":      round(max_dd, 2),
            "annualised_sharpe": round(sharpe, 3),
            "avg_spread":        round(mean(spreads), 3) if spreads else None,
            "mid_range":         (round(min(mids), 2), round(max(mids), 2)) if mids else None,
            "ticks":             len(rows),
        }
    return summary


def compute_trade_metrics(trades, my_id="SUBMISSION"):
    by_product = defaultdict(list)
    for t in trades:
        if t["buyer"] == my_id or t["seller"] == my_id:
            by_product[t["symbol"]].append(t)

    result = {}
    for prod, ts in by_product.items():
        buys  = [t for t in ts if t["buyer"]  == my_id]
        sells = [t for t in ts if t["seller"] == my_id]
        bq = sum(t["quantity"] for t in buys)
        sq = sum(t["quantity"] for t in sells)
        bvwap = sum(t["price"]*t["quantity"] for t in buys)  / bq if bq else None
        svwap = sum(t["price"]*t["quantity"] for t in sells) / sq if sq else None
        prices = [t["price"] for t in ts]
        result[prod] = {
            "total_trades":    len(ts),
            "buy_trades":      len(buys),
            "sell_trades":     len(sells),
            "buy_qty":         bq,
            "sell_qty":        sq,
            "net_qty":         bq - sq,
            "total_volume":    bq + sq,
            "buy_vwap":        round(bvwap, 3) if bvwap else None,
            "sell_vwap":       round(svwap, 3) if svwap else None,
            "spread_captured": round(svwap - bvwap, 3) if (bvwap and svwap) else None,
            "price_range":     (round(min(prices), 2), round(max(prices), 2)) if prices else None,
        }
    return result


def compute_position_from_pnl(activities):
    by_product = defaultdict(list)
    for row in activities:
        by_product[row["product"]].append(row)

    result = {}
    for prod, rows in by_product.items():
        rows.sort(key=lambda r: (r["day"], r["timestamp"]))
        pnls = [r["pnl"] for r in rows if r["pnl"] is not None]
        mids = [r["mid_price"] for r in rows if r["mid_price"] is not None]
        if len(pnls) < 2 or len(mids) < 2: continue
        implied = []
        for i in range(1, min(len(pnls), len(mids))):
            dp, dm = pnls[i] - pnls[i-1], mids[i] - mids[i-1]
            if abs(dm) > 0.05:
                implied.append(dp / dm)
        if not implied: continue
        lim = POSITION_LIMITS.get(prod, 999)
        result[prod] = {
            "avg_implied_pos":   round(mean(implied), 2),
            "max_long_implied":  round(max(implied),  2),
            "max_short_implied": round(min(implied),  2),
            "position_limit":    lim,
            "avg_utilisation_%": round(100 * abs(mean(implied)) / lim, 1),
        }
    return result


def compute_iv_surface(activities):
    by_ts = defaultdict(dict)
    for row in activities:
        by_ts[(row["day"], row["timestamp"])][row["product"]] = row.get("mid_price")

    iv_by_strike = defaultdict(list)
    sorted_ts = sorted(by_ts.keys())
    n = len(sorted_ts)
    for idx, key in enumerate(sorted_ts):
        snap = by_ts[key]
        S = snap.get("VELVETFRUIT_EXTRACT")
        if not S or S <= 0: continue
        progress = idx / max(n - 1, 1)
        T = (5.0 / 365.0) - progress * (1.0 / 365.0)
        if T <= 0: continue
        for sym, K in VOUCHER_STRIKES.items():
            V = snap.get(sym)
            if not V or V <= 0 or V < max(0.0, S - K): continue
            lo, hi = 0.01, 2.0
            for _ in range(20):
                m = (lo + hi) / 2.0
                try:
                    d1 = (math.log(S / K) + 0.5 * m**2 * T) / (m * math.sqrt(T))
                    p  = S * cdf(d1) - K * cdf(d1 - m * math.sqrt(T))
                    lo, hi = (m, hi) if p < V else (lo, m)
                except: break
            iv = (lo + hi) / 2.0
            if 0.01 < iv < 1.5:
                iv_by_strike[sym].append(iv)

    return {
        sym: {
            "avg_iv":    round(mean(ivs), 4),
            "min_iv":    round(min(ivs),  4),
            "max_iv":    round(max(ivs),  4),
            "iv_stdev":  round(stdev(ivs), 4) if len(ivs) > 1 else 0.0,
            "n_samples": len(ivs),
        }
        for sym, ivs in iv_by_strike.items() if ivs
    }


def compute_pnl_attribution(activities):
    """Break down PnL into segments: early / mid / late third of simulation."""
    by_product = defaultdict(list)
    for row in activities:
        by_product[row["product"]].append(row)

    result = {}
    for prod, rows in by_product.items():
        rows.sort(key=lambda r: (r["day"], r["timestamp"]))
        pnls = [r["pnl"] for r in rows if r["pnl"] is not None]
        if len(pnls) < 3: continue
        n = len(pnls)
        t1, t2 = n // 3, 2 * n // 3
        result[prod] = {
            "early_pnl_change":  round(pnls[t1]  - pnls[0],   2),
            "mid_pnl_change":    round(pnls[t2]  - pnls[t1],  2),
            "late_pnl_change":   round(pnls[-1]  - pnls[t2],  2),
            "total":             round(pnls[-1]  - pnls[0],   2),
        }
    return result


# ─── REPORT HELPERS ────────────────────────────────────────────────────────────

def sep(title=""):
    print("\n" + "=" * 72)
    if title: print(f"  {title}")
    print("=" * 72)

def pdict(d, indent=4):
    pad = " " * indent
    for k, v in d.items():
        if isinstance(v, dict):
            print(f"{pad}{k}:")
            pdict(v, indent + 4)
        else:
            print(f"{pad}{k:<42} {v}")


# ─── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print("Usage: python prosperity_log_analyzer.py <logfile.log>")
        sys.exit(1)

    filepath = sys.argv[1]
    print(f"\n>>> Analyzing: {filepath}")

    activities, trades, sandbox = load_and_parse(filepath)

    products = sorted({r["product"] for r in activities})
    print(f"\n    Activity rows   : {len(activities)}")
    print(f"    Trade rows      : {len(trades)}")
    print(f"    Products found  : {products}")

    if not activities:
        print("\n  ⚠️  No activity data parsed.")
        print("  ➜  Run:  python -c \"import json; d=json.load(open('yourfile.log')); print(list(d.keys()))\"")
        print("     and paste the output into Claude to identify the correct field names.")
        return

    pnl_m   = compute_pnl_metrics(activities)
    trade_m = compute_trade_metrics(trades)
    pos_m   = compute_position_from_pnl(activities)
    iv_surf = compute_iv_surface(activities)
    attr    = compute_pnl_attribution(activities)
    total   = sum(v["final_pnl"] for v in pnl_m.values())

    # ── Summary table ─────────────────────────────────────────────────────────
    sep("OVERALL SUMMARY")
    print(f"  {'Total Final PnL:':<44} {total:,.2f}")
    print(f"\n  {'Product':<32} {'FinalPnL':>11} {'PeakPnL':>10} "
          f"{'MaxDD':>9} {'Sharpe':>8} {'Spread':>7}")
    print("  " + "-" * 79)
    for p in sorted(pnl_m, key=lambda x: pnl_m[x]["final_pnl"], reverse=True):
        m  = pnl_m[p]
        sp = m.get("avg_spread") or 0
        print(f"  {p:<32} {m['final_pnl']:>11,.1f} {m['peak_pnl']:>10,.1f} "
              f"{m['max_drawdown']:>9,.1f} {m['annualised_sharpe']:>8.3f} {sp:>7.2f}")

    # ── PnL attribution ───────────────────────────────────────────────────────
    sep("PnL ATTRIBUTION  (early / mid / late thirds of simulation)")
    print(f"\n  {'Product':<32} {'Early':>10} {'Mid':>10} {'Late':>10} {'Total':>10}")
    print("  " + "-" * 64)
    for p in sorted(attr, key=lambda x: attr[x]["total"], reverse=True):
        a = attr[p]
        print(f"  {p:<32} {a['early_pnl_change']:>10,.1f} {a['mid_pnl_change']:>10,.1f} "
              f"{a['late_pnl_change']:>10,.1f} {a['total']:>10,.1f}")

    # ── Per-product details ────────────────────────────────────────────────────
    sep("PER-PRODUCT PnL DETAILS")
    for p in sorted(pnl_m, key=lambda x: pnl_m[x]["final_pnl"], reverse=True):
        print(f"\n  [{p}]")
        pdict(pnl_m[p])

    # ── Trades ────────────────────────────────────────────────────────────────
    sep("TRADE EXECUTION METRICS")
    if trade_m:
        for p in sorted(trade_m):
            print(f"\n  [{p}]")
            pdict(trade_m[p])
    else:
        print("  (tradesLog field empty or not found — PnL data above is still valid)")
        print("  Tip: check 'tradesLog' key exists in the JSON envelope.")

    # ── Position utilisation ──────────────────────────────────────────────────
    sep("IMPLIED POSITION UTILISATION  (ΔPnL / ΔMid proxy)")
    if pos_m:
        for p in sorted(pos_m):
            print(f"\n  [{p}]")
            pdict(pos_m[p])
    else:
        print("  (Could not compute — PnL column may be zero/missing)")

    # ── IV surface ────────────────────────────────────────────────────────────
    sep("VOUCHER IMPLIED VOLATILITY SURFACE")
    if iv_surf:
        print(f"\n  {'Strike':<15} {'AvgIV':>8} {'MinIV':>8} "
              f"{'MaxIV':>8} {'StDev':>8} {'N':>6}")
        print("  " + "-" * 57)
        for sym in sorted(iv_surf, key=lambda s: VOUCHER_STRIKES.get(s, 0)):
            iv = iv_surf[sym]
            print(f"  {sym:<15} {iv['avg_iv']:>8.4f} {iv['min_iv']:>8.4f} "
                  f"{iv['max_iv']:>8.4f} {iv['iv_stdev']:>8.4f} {iv['n_samples']:>6}")

        avg_ivs = {s: iv_surf[s]["avg_iv"] for s in iv_surf}
        if avg_ivs:
            atm_sym = min(avg_ivs,
                          key=lambda s: abs(VOUCHER_STRIKES[s] - 5255))
            print(f"\n  Most-ATM strike (near S≈5255): {atm_sym} "
                  f"→ avg IV = {avg_ivs.get(atm_sym, 'N/A')}")
            iv_vals = sorted(avg_ivs.values())
            if len(iv_vals) > 1:
                print(f"  IV range across strikes: "
                      f"{iv_vals[0]:.4f} – {iv_vals[-1]:.4f}  "
                      f"(spread = {iv_vals[-1]-iv_vals[0]:.4f})")
    else:
        print("  (No IV data — VEV products or extract mid prices not found)")

    # ── Extract-specific diagnostics ──────────────────────────────────────────
    sep("VELVETFRUIT_EXTRACT — STRATEGY DIAGNOSTICS")
    vex_pnl = pnl_m.get("VELVETFRUIT_EXTRACT", {})
    vex_tr  = trade_m.get("VELVETFRUIT_EXTRACT", {})
    vex_pos = pos_m.get("VELVETFRUIT_EXTRACT", {})

    if vex_pnl:
        sc  = vex_tr.get("spread_captured")
        ns  = vex_pnl.get("avg_spread")
        eff = round(100 * sc / ns, 1) if (sc and ns and ns > 0) else "N/A"
        print(f"\n  Natural avg book spread       : {ns}")
        print(f"  Our buy VWAP                  : {vex_tr.get('buy_vwap')}")
        print(f"  Our sell VWAP                 : {vex_tr.get('sell_vwap')}")
        print(f"  Spread captured (sell-buy)    : {sc}")
        print(f"  Spread capture efficiency     : {eff}%")
        print(f"  Final PnL (extract only)      : {vex_pnl.get('final_pnl')}")
        print(f"  Max drawdown (extract only)   : {vex_pnl.get('max_drawdown')}")
        print(f"  Avg position utilisation      : {vex_pos.get('avg_utilisation_%')}%")
        print(f"  Mid-price range               : {vex_pnl.get('mid_range')}")

        issues = []
        if isinstance(sc, float) and sc < 0.5:
            issues.append("Near-zero spread capture — buying/selling at similar prices.")
        if isinstance(sc, float) and sc < 0:
            issues.append("NEGATIVE spread capture — on average buying high and selling low!")
        if vex_pos.get("avg_utilisation_%", 0) > 55:
            issues.append("High utilisation — little capacity left for voucher delta hedging.")
        dd = vex_pnl.get("max_drawdown", 0)
        fp = vex_pnl.get("final_pnl", 1)
        if dd > 0 and abs(fp) > 0 and dd > abs(fp) * 1.5:
            issues.append("MaxDrawdown >> FinalPnL — large adverse inventory swings.")

        if issues:
            print()
            for iss in issues:
                print(f"  ⚠️  {iss}")
    else:
        print("  (VELVETFRUIT_EXTRACT not found in activities data)")

    sep("END OF REPORT — paste full output to Claude for analysis")
    print()


if __name__ == "__main__":
    main()
