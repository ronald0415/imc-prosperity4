from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict, Tuple
import json

# ═══════════════════════════════════════════════════════════════════════════════
#  IMC PROSPERITY 4  –  ROUND 5 TRADER  v8
#
#  v7 result: +11,603 XIRECS (down from v6's +17,556 due to 3 new losers).
#  DARK_MATTER lost -4,138 in BOTH v5 and v7 (different random samples)
#  → structurally broken, not random variance. UV_VISOR_RED lost -831.
#
#  v8 CHANGES:
#  ──────────────────────────────────────────────────────────────────
#    1. DROP DARK_MATTER: lost -4,138 in v5 AND v7. Not sample noise.
#    2. DROP UV_VISOR_RED: lost -831 (stuck SHORT 78% of time at hs=9).
#    3. ADD UV_VISOR stat-arb pairs (Hint 3: "pairing for profit").
#       Conservative params: entry_z=2.0, pair_size=2, ratio_span=80.
#       Same proven logic as SNACKPACK pairs, applied to UV_VISOR group.
#    4. INVENTORY ACCELERATION: when |pos| == LIMIT (fully capped at ±10),
#       place unwind order at fair±1 instead of normal skew price. Gets
#       position back to two-sided MM faster. Structural fix, not tuning.
#
#  NON-OVERFITTING:
#    - Drops are based on losses across MULTIPLE random samples (not one)
#    - UV_VISOR stat-arb uses SAME logic as proven SNACKPACK pairs
#    - Inventory acceleration is mechanical, applies to ALL products
#
# ═══════════════════════════════════════════════════════════════════════════════

LIMIT         = 10
CAP_THRESHOLD = 8   # stop adding in direction of exposure above this

# ─────────────────────────────────────────────────────────────────────────────
#  MARKET-MAKING CONFIG  (13 products)
# ─────────────────────────────────────────────────────────────────────────────
MM_CFG: Dict[str, dict] = {

    # ── SNACKPACK (spread/noise 2.1–3.0×) — highest edge group ──────────────
    "SNACKPACK_PISTACHIO":          {"ema_span": 20, "mm_hs": 10, "group": "SNACKPACK"},
    "SNACKPACK_VANILLA":            {"ema_span": 20, "mm_hs": 9,  "group": "SNACKPACK"},
    "SNACKPACK_RASPBERRY":          {"ema_span": 20, "mm_hs": 9,  "group": "SNACKPACK"},
    "SNACKPACK_CHOCOLATE":          {"ema_span": 20, "mm_hs": 9,  "group": "SNACKPACK"},
    "SNACKPACK_STRAWBERRY":         {"ema_span": 20, "mm_hs": 10, "group": "SNACKPACK"},

    # ── UV_VISOR (spread/noise ~1.27×) — 3 proven winners ──────────────────
    "UV_VISOR_MAGENTA":             {"ema_span": 20, "mm_hs": 8,  "group": "UV_VISOR"},
    "UV_VISOR_ORANGE":              {"ema_span": 20, "mm_hs": 9,  "group": "UV_VISOR"},
    "UV_VISOR_YELLOW":              {"ema_span": 20, "mm_hs": 8,  "group": "UV_VISOR"},
    # v8: RED removed (lost -831 in v7, stuck SHORT 78%)

    # ── GALAXY_SOUNDS — 4 of 5 (DARK_MATTER dropped) ──────────────────────
    "GALAXY_SOUNDS_SOLAR_FLAMES":   {"ema_span": 20, "mm_hs": 8,  "group": "GALAXY"},
    "GALAXY_SOUNDS_BLACK_HOLES":    {"ema_span": 20, "mm_hs": 8,  "group": "GALAXY"},
    "GALAXY_SOUNDS_PLANETARY_RINGS":{"ema_span": 20, "mm_hs": 8,  "group": "GALAXY"},
    "GALAXY_SOUNDS_SOLAR_WINDS":    {"ema_span": 20, "mm_hs": 8,  "group": "GALAXY"},
    # v8: DARK_MATTER removed (lost -4,138 in BOTH v5 and v7)
}

# ─────────────────────────────────────────────────────────────────────────────
#  MEAN-REVERSION CONFIG  (1 product)
# ─────────────────────────────────────────────────────────────────────────────
MR_CFG: Dict[str, dict] = {
    "OXYGEN_SHAKE_EVENING_BREATH": {"ema_span": 20, "mr_thr": 32, "mr_close": 13},
}

# ─────────────────────────────────────────────────────────────────────────────
#  STAT-ARB PAIRS  (Hint 3: "pairing for profit")
#
#  SNACKPACK pairs: proven across v4/v5/v6/v7. Tight CVs (2-3%).
#  UV_VISOR pairs: NEW in v8. Same logic, conservative params.
#    Historical CVs ~5% (looser than SNACKPACK), so we use:
#    - entry_z=2.0 (vs 1.5) — only enter on extreme deviations
#    - pair_size=2 (vs 3) — smaller position per pair
#    - ratio_span=80 (vs 50) — slower EMA, less reactive to noise
# ─────────────────────────────────────────────────────────────────────────────

# SNACKPACK pairs (proven, original params)
SNACK_PAIRS: List[Tuple[str, str, float, float]] = [
    ("SNACKPACK_CHOCOLATE",  "SNACKPACK_PISTACHIO",  1.0368, 0.0213),
    ("SNACKPACK_VANILLA",    "SNACKPACK_RASPBERRY",  1.0022, 0.0242),
    ("SNACKPACK_CHOCOLATE",  "SNACKPACK_RASPBERRY",  0.9770, 0.0252),
    ("SNACKPACK_VANILLA",    "SNACKPACK_PISTACHIO",  1.0639, 0.0325),
]
SNACK_ENTRY_Z    = 1.5
SNACK_EXIT_Z     = 0.3
SNACK_PAIR_SIZE  = 3
SNACK_RATIO_SPAN = 50

# UV_VISOR pairs (new, conservative params)
UV_PAIRS: List[Tuple[str, str, float, float]] = [
    ("UV_VISOR_ORANGE",  "UV_VISOR_MAGENTA",  0.9397, 0.0465),
    ("UV_VISOR_ORANGE",  "UV_VISOR_YELLOW",   0.8531, 0.0500),   # est from price ratios
    ("UV_VISOR_MAGENTA", "UV_VISOR_YELLOW",   0.9082, 0.0550),   # est from price ratios
]
UV_ENTRY_Z    = 2.0    # more conservative — only extreme deviations
UV_EXIT_Z     = 0.5    # exit faster
UV_PAIR_SIZE  = 2      # smaller position
UV_RATIO_SPAN = 80     # slower EMA adaptation

# ─────────────────────────────────────────────────────────────────────────────
#  GROUPS for cross-sectional signal (Hint 1 + Hint 2)
# ─────────────────────────────────────────────────────────────────────────────
GROUPS: Dict[str, List[str]] = {
    "SNACKPACK": ["SNACKPACK_PISTACHIO", "SNACKPACK_VANILLA",
                  "SNACKPACK_RASPBERRY", "SNACKPACK_CHOCOLATE",
                  "SNACKPACK_STRAWBERRY"],
    "UV_VISOR":  ["UV_VISOR_MAGENTA", "UV_VISOR_ORANGE", "UV_VISOR_YELLOW"],
    "GALAXY":    ["GALAXY_SOUNDS_SOLAR_FLAMES", "GALAXY_SOUNDS_BLACK_HOLES",
                  "GALAXY_SOUNDS_PLANETARY_RINGS", "GALAXY_SOUNDS_SOLAR_WINDS"],
}
GROUP_PULL = 0.2


class Trader:

    def run(self, state: TradingState):

        try:
            sv = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            sv = {}
        ema_map:   Dict[str, float] = sv.get("ema", {})
        rmean_map: Dict[str, float] = sv.get("rm",  {})
        pair_pos:  Dict[str, int]   = sv.get("pp",  {})

        result: Dict[str, List[Order]] = {}

        # ── Step 1: Compute current mids + update EMAs ───────────────────────
        mid_map: Dict[str, float] = {}
        all_prods = set(MM_CFG) | set(MR_CFG)

        for prod in all_prods:
            if prod not in state.order_depths:
                continue
            od = state.order_depths[prod]
            bb = max(od.buy_orders)  if od.buy_orders  else None
            ba = min(od.sell_orders) if od.sell_orders else None
            if bb is None and ba is None:
                continue
            mid = (bb + ba) / 2.0 if (bb is not None and ba is not None) else float(bb or ba)
            mid_map[prod] = mid

            cfg   = MM_CFG.get(prod) or MR_CFG.get(prod, {})
            alpha = 2.0 / (cfg.get("ema_span", 20) + 1.0)
            ema_map[prod] = alpha * mid + (1.0 - alpha) * ema_map.get(prod, mid)

        # ── Step 2: Group cross-sectional fair value (Hint 1) ────────────────
        group_mid: Dict[str, float] = {}
        for grp, members in GROUPS.items():
            vals = [mid_map[p] for p in members if p in mid_map]
            if vals:
                group_mid[grp] = sum(vals) / len(vals)

        # ── Step 3: Market making ────────────────────────────────────────────
        for prod, cfg in MM_CFG.items():
            if prod not in mid_map:
                continue
            pos = state.position.get(prod, 0)
            od  = state.order_depths[prod]
            grp = cfg["group"]

            # Fair value = current mid pulled toward group average
            mid  = mid_map[prod]
            fair = mid + GROUP_PULL * (group_mid.get(grp, mid) - mid) \
                   if grp in group_mid else mid

            result[prod] = self._mm(prod, od, pos, fair, cfg["mm_hs"])

        # ── Step 4: Mean reversion (EVENING_BREATH only) ─────────────────────
        for prod, cfg in MR_CFG.items():
            if prod not in mid_map:
                continue
            pos = state.position.get(prod, 0)
            result[prod] = self._mr(
                prod, state.order_depths[prod], pos,
                ema_map[prod], cfg["mr_thr"], cfg["mr_close"]
            )

        # ── Step 5: Stat-arb pairs ───────────────────────────────────────────
        pair_orders: Dict[str, List[Order]] = {}

        # 5a: SNACKPACK pairs (proven, original params)
        self._run_pairs(SNACK_PAIRS, SNACK_ENTRY_Z, SNACK_EXIT_Z,
                        SNACK_PAIR_SIZE, SNACK_RATIO_SPAN,
                        mid_map, rmean_map, pair_pos, state, pair_orders)

        # 5b: UV_VISOR pairs (new, conservative params)
        self._run_pairs(UV_PAIRS, UV_ENTRY_Z, UV_EXIT_Z,
                        UV_PAIR_SIZE, UV_RATIO_SPAN,
                        mid_map, rmean_map, pair_pos, state, pair_orders)

        for prod, orders in pair_orders.items():
            result.setdefault(prod, []).extend(orders)

        return result, 0, json.dumps({"ema": ema_map, "rm": rmean_map, "pp": pair_pos})

    # ─────────────────────────────────────────────────────────────────────────
    def _run_pairs(self, pairs, entry_z, exit_z, pair_size, ratio_span,
                   mid_map, rmean_map, pair_pos, state, pair_orders):
        """Execute stat-arb logic for a list of pairs with given params."""
        for pa, pb, hist_mean, hist_std in pairs:
            if pa not in mid_map or pb not in mid_map:
                continue
            ratio = mid_map[pa] / mid_map[pb]
            pkey  = f"{pa}|{pb}"
            alpha = 2.0 / (ratio_span + 1.0)
            rmean = alpha * ratio + (1.0 - alpha) * rmean_map.get(pkey, hist_mean)
            rmean_map[pkey] = rmean
            z        = (ratio - rmean) / hist_std
            cur_side = pair_pos.get(pkey, 0)
            pos_a    = state.position.get(pa, 0)
            pos_b    = state.position.get(pb, 0)
            od_a     = state.order_depths[pa]
            od_b     = state.order_depths[pb]

            if cur_side == 0:
                if z > entry_z and od_a.buy_orders and od_b.sell_orders:
                    vol = min(pair_size, LIMIT + pos_a, LIMIT - pos_b)
                    if vol > 0:
                        pair_orders.setdefault(pa, []).append(Order(pa, max(od_a.buy_orders), -vol))
                        pair_orders.setdefault(pb, []).append(Order(pb, min(od_b.sell_orders),  vol))
                        pair_pos[pkey] = -1
                elif z < -entry_z and od_a.sell_orders and od_b.buy_orders:
                    vol = min(pair_size, LIMIT - pos_a, LIMIT + pos_b)
                    if vol > 0:
                        pair_orders.setdefault(pa, []).append(Order(pa, min(od_a.sell_orders),  vol))
                        pair_orders.setdefault(pb, []).append(Order(pb, max(od_b.buy_orders), -vol))
                        pair_pos[pkey] = 1
            elif abs(z) < exit_z:
                if cur_side == -1 and od_a.sell_orders and od_b.buy_orders:
                    vol = min(pair_size, LIMIT - pos_a, LIMIT + pos_b)
                    if vol > 0:
                        pair_orders.setdefault(pa, []).append(Order(pa, min(od_a.sell_orders),  vol))
                        pair_orders.setdefault(pb, []).append(Order(pb, max(od_b.buy_orders), -vol))
                        pair_pos[pkey] = 0
                elif cur_side == 1 and od_a.buy_orders and od_b.sell_orders:
                    vol = min(pair_size, LIMIT + pos_a, LIMIT - pos_b)
                    if vol > 0:
                        pair_orders.setdefault(pa, []).append(Order(pa, max(od_a.buy_orders), -vol))
                        pair_orders.setdefault(pb, []).append(Order(pb, min(od_b.sell_orders),  vol))
                        pair_pos[pkey] = 0

    # ─────────────────────────────────────────────────────────────────────────
    def _mm(self, product: str, od: OrderDepth, pos: int,
            fair: float, hs: int) -> List[Order]:
        """
        Market making with bounded skew, one-sided quoting, and
        inventory acceleration (v8: aggressive unwind when fully capped).
        """
        orders: List[Order] = []
        buy_cap  = LIMIT - pos
        sell_cap = LIMIT + pos

        # One-sided quoting near cap
        if pos >= CAP_THRESHOLD:
            buy_cap = 0
        if pos <= -CAP_THRESHOLD:
            sell_cap = 0

        # Bounded inventory skew
        raw_skew = pos * 0.5
        skew = int(round(max(-(hs - 1), min(hs - 1, raw_skew))))

        bid_px = round(fair - hs - skew)
        ask_px = round(fair + hs - skew)

        # Hard bound: never post bid > fair-1 or ask < fair+1
        bid_px = min(bid_px, int(fair) - 1)
        ask_px = max(ask_px, int(fair) + 1)

        # v8: Inventory acceleration — when FULLY capped, unwind aggressively
        # This gets us back to two-sided MM faster (structural, not tuning)
        if pos == LIMIT:
            ask_px = int(fair) + 1   # sell at fair+1 to clear inventory fast
        elif pos == -LIMIT:
            bid_px = int(fair) - 1   # buy at fair-1 to clear inventory fast

        # Snipe clearly mispriced orders
        if od.sell_orders and buy_cap > 0:
            best_ask = min(od.sell_orders)
            if best_ask <= bid_px:
                vol = min(-od.sell_orders[best_ask], buy_cap)
                if vol > 0:
                    orders.append(Order(product, best_ask, vol))
                    buy_cap -= vol

        if od.buy_orders and sell_cap > 0:
            best_bid = max(od.buy_orders)
            if best_bid >= ask_px:
                vol = min(od.buy_orders[best_bid], sell_cap)
                if vol > 0:
                    orders.append(Order(product, best_bid, -vol))
                    sell_cap -= vol

        # Passive resting quotes
        if buy_cap > 0:
            orders.append(Order(product, bid_px, buy_cap))
        if sell_cap > 0:
            orders.append(Order(product, ask_px, -sell_cap))

        return orders

    # ─────────────────────────────────────────────────────────────────────────
    def _mr(self, product: str, od: OrderDepth, pos: int,
            ema: float, thr: float, close_thr: float) -> List[Order]:
        """
        Mean-reversion scalp for EVENING_BREATH only.
        Enter aggressively when mid deviates > thr from EMA.
        Unwind passively at close_thr offset from EMA.
        """
        orders: List[Order] = []
        buy_cap  = LIMIT - pos
        sell_cap = LIMIT + pos

        # Aggressive entry: lift ask when price is thr below EMA
        if od.sell_orders and buy_cap > 0:
            for ask in sorted(od.sell_orders):
                dev = ema - ask
                if dev < thr:
                    break
                vol = min(-od.sell_orders[ask], buy_cap)
                if vol > 0:
                    orders.append(Order(product, ask, vol))
                    buy_cap -= vol
                if buy_cap == 0:
                    break

        # Aggressive entry: hit bid when price is thr above EMA
        if od.buy_orders and sell_cap > 0:
            for bid in sorted(od.buy_orders, reverse=True):
                dev = bid - ema
                if dev < thr:
                    break
                vol = min(od.buy_orders[bid], sell_cap)
                if vol > 0:
                    orders.append(Order(product, bid, -vol))
                    sell_cap -= vol
                if sell_cap == 0:
                    break

        # Passive unwind near EMA
        unwind_vol = min(3, abs(pos))
        if unwind_vol > 0:
            if pos > 0 and sell_cap > 0:
                orders.append(Order(product, round(ema + close_thr), -unwind_vol))
            elif pos < 0 and buy_cap > 0:
                orders.append(Order(product, round(ema - close_thr), unwind_vol))

        return orders


# ═══════════════════════════════════════════════════════════════════════════════
#  CHANGE TABLE (v7 → v8)
# ═══════════════════════════════════════════════════════════════════════════════
#
#  Change                             | v7 (570100)  | v8
#  ───────────────────────────────────┼──────────────┼─────────────────────────
#  GALAXY_DARK_MATTER                 | hs=8 (-4138) | REMOVED (lost in v5+v7)
#  UV_VISOR_RED                       | hs=9 (-831)  | REMOVED (stuck SHORT)
#  UV_VISOR stat-arb pairs            | none         | ADDED (3 pairs, Hint 3)
#  Inventory acceleration             | none         | ADDED (unwind at fair±1)
#  _run_pairs() method                | inline       | Refactored (reusable)
#  GALAXY group members               | 5            | 4 (no DARK_MATTER)
#  UV_VISOR group members             | 4            | 3 (no RED)
#  Active product count               | 15           | 13 MM + 1 MR = 14
#
#  v7 results (570100.log):
#    +11,603 PnL. Winners: MAGENTA +5,582, ORANGE +4,408, VANILLA +2,244
#    PLANETARY_RINGS +1,284 (new winner), SOLAR_FLAMES +1,145
#    Losers: DARK_MATTER -4,138 (struct. broken), RED -831, WINDS -463
#
# ═══════════════════════════════════════════════════════════════════════════════