"""
IMC Prosperity 4 – Round 4 Trader
===================================

FIXES ADDED IN THIS VERSION
────────────────────────────
FIX-1  Smile convergence gate
       MM_STRIKES are not quoted until smile_fit_count >= MIN_SMILE_FITS (20).
       Before that the smile coefficients carry too much seed error.
       VEV_5400 sigma error of 0.02 → 2.42 tick price error → bots hammer
       our mis-positioned bid 1111 times. Gate eliminates this entirely.

FIX-2  Dynamic MM halfspread
       halfspread starts at 6 (covering ±5.6 tick model uncertainty for 5400)
       and tightens linearly to 2 as smile_fit_count reaches 50.
       Formula: halfspread = max(2, int(6 * (1 - calibration)))
       At fit=0: halfspread=6, at fit=20: halfspread=3.6, at fit=50: halfspread=2.

FIX-3  Sanity check before MM quoting
       If |bs_fair - market_mid| / market_mid > 30%, model is clearly wrong.
       Skip MM for that strike that tick. Catches catastrophic seed errors.

ORIGINAL FIXES (from 493282.py)
────────────────────────────────
BUG-1  Polyfit coefficient ordering fixed (descending [a2, a1, a0])
BUG-2  HYDROGEL pure adaptive EMA, no stale anchor
BUG-3  Single price-edge gate (removed double iv_dev condition)

STRATEGY UPGRADES (retained)
─────────────────────────────
U-1  Per-strike structural IV bias
U-2  Market-making on tight-spread options (now gated + dynamic spread)
U-3  Weighted smile fitting
U-4  Decayed Mark flow signals
U-5  HYDROGEL mean-reversion overlay
"""

from datamodel import OrderDepth, TradingState, Order
import json
import math


class Trader:
    # ------------------------------------------------------------------ #
    #  CONFIGURATION                                                       #
    # ------------------------------------------------------------------ #
    LIMITS = {
        "HYDROGEL_PACK": 200, "VELVETFRUIT_EXTRACT": 200,
        **{k: 300 for k in [
            "VEV_4000", "VEV_4500", "VEV_5000", "VEV_5100", "VEV_5200",
            "VEV_5300", "VEV_5400", "VEV_5500", "VEV_6000", "VEV_6500",
        ]},
    }

    STRIKES = {
        "VEV_4000": 4000, "VEV_4500": 4500, "VEV_5000": 5000,
        "VEV_5100": 5100, "VEV_5200": 5200, "VEV_5300": 5300,
        "VEV_5400": 5400, "VEV_5500": 5500, "VEV_6000": 6000,
        "VEV_6500": 6500,
    }

    # U-1: per-strike IV bias derived from observed deviation chart
    IV_BIAS = {
        "VEV_4000":  -0.025,  # was 0.000  — noisy, set to mu to stop random buys
        "VEV_4500":  +0.068,  # was -0.040 — WRONG SIGN, market above smile → raise fair
        "VEV_5000":  -0.024,  # was +0.015 — was generating false buys (+1.56 tick error)
        "VEV_5100":  -0.013,  # was +0.008 — was generating false buys (+2.50 tick error)
        "VEV_5200":  0.00,  # was 0.000  — nearly correct already ✓
        "VEV_5300":  +0.00,  # was 0.000  — nearly correct already ✓
        "VEV_5400":  -0.015,  # was +0.010 — WRONG SIGN, was +2.66 tick overestimate
        "VEV_5500":  -0.016,  # was +0.010 — WRONG SIGN, was +0.97 tick overestimate
        "VEV_6000":  +0.029,  # was -0.020 — WRONG SIGN, was underestimating
        "VEV_6500":  -0.003,  # was 0.000  — fine
    }
    SCALP_DISABLED = {"VEV_4000","VEV_5400", "VEV_5500","VEV_4500","VEV_5100"}
    MIN_EDGE       = 0.5
    DELTA_LIMIT    = 60.0
    HEDGE_FRACTION = 0.80

    # MM_STRIKES: only quote passively here (tight spread ≤ 1.5 avg)
    MM_STRIKES = {"VEV_5400", "VEV_5500", "VEV_6000", "VEV_6500"}
    MM_SIZE    = 5

    # FIX-1: minimum smile fits before MM quoting begins
    # At alpha=0.20, after 20 fits the seed contributes only 1.2% of the EMA
    MIN_SMILE_FITS = 20

    # FIX-2: dynamic halfspread bounds
    # Starts wide (covers ±5.6 tick model uncertainty on VEV_5400)
    # Tightens to MM_HS_MIN as smile converges over MM_HS_DECAY fits
    MM_HS_START = 6
    MM_HS_MIN   = 2
    MM_HS_DECAY = 50   # fits to reach MM_HS_MIN

    # FIX-3: sanity check — skip MM if BS fair deviates > this fraction from market mid
    MM_MAX_MODEL_ERROR = 0.30   # 30%

    # ------------------------------------------------------------------ #
    #  STATE                                                               #
    # ------------------------------------------------------------------ #
    def load_data(self, trader_data: str) -> dict:
        defaults = {
            "hydro_hist":      [],
            "hydro_ema":       None,
            "vev_ema":         5262.0,
            "last_ts":         -1,
            # Smile coefficients [a2, a1, a0] descending order
            # Seeded from R3/R4 smile analysis: ATM IV ≈ 0.27, curvature ≈ 5
            "smile_coeffs":    [5.0, -0.10, 0.27],
            "smile_fit_count": 0,
            "mark_flow":       {},
        }
        if trader_data:
            try:
                loaded = json.loads(trader_data)
                defaults.update(loaded)
            except Exception:
                pass
        return defaults

    def dump_data(self, d: dict) -> str:
        if len(d.get("mark_flow", {})) > 200:
            d["mark_flow"] = {}
        return json.dumps(d, separators=(",", ":"))

    # ------------------------------------------------------------------ #
    #  OPTIONS MATHS                                                       #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _cdf(x: float) -> float:
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    def bs_price(self, S: float, K: float, T: float, sigma: float) -> float:
        if T <= 0 or sigma <= 0:
            return max(0.0, S - K)
        d1 = (math.log(S / K) + 0.5 * sigma**2 * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        return S * self._cdf(d1) - K * self._cdf(d2)

    def bs_delta(self, S: float, K: float, T: float, sigma: float) -> float:
        if T <= 0 or sigma <= 0:
            return 1.0 if S > K else 0.0
        d1 = (math.log(S / K) + 0.5 * sigma**2 * T) / (sigma * math.sqrt(T))
        return self._cdf(d1)

    def solve_iv(self, S: float, K: float, T: float, price: float,
                 lo: float = 0.01, hi: float = 3.0, iters: int = 60) -> float | None:
        intrinsic = max(0.0, S - K)
        if price <= intrinsic + 0.05:
            return None
        for _ in range(iters):
            mid = (lo + hi) / 2.0
            if self.bs_price(S, K, T, mid) < price:
                lo = mid
            else:
                hi = mid
        iv = (lo + hi) / 2.0
        return iv if 0.02 < iv < 2.5 else None

    # ------------------------------------------------------------------ #
    #  VOLATILITY SMILE FITTING                                            #
    # ------------------------------------------------------------------ #
    def fit_smile(self, S: float, T: float, depths: dict, data: dict) -> dict:
        """
        Weighted quadratic fit across all strikes.
        Returns dict: sym -> adjusted smile IV (including IV_BIAS).
        """
        m_vals, iv_vals, w_vals = [], [], []

        for sym, K in self.STRIKES.items():
            depth = depths.get(sym)
            if not depth:
                continue
            mid = self._mid(depth)
            if mid is None:
                continue
            intrinsic = max(0.0, S - K)
            if mid - intrinsic < 0.5:
                continue
            iv = self.solve_iv(S, K, T, mid)
            if iv is None:
                continue
            m = math.log(K / S)
            m_vals.append(m)
            iv_vals.append(iv)
            # U-3: downweight deep ITM/OTM (noisier IV observations)
            w_vals.append(math.exp(-2.0 * abs(m)))

        if len(m_vals) >= 3:
            try:
                coeffs = self._wpolyfit2(m_vals, iv_vals, w_vals)
                if coeffs is None:
                    raise ValueError("singular")
                alpha = 0.20
                prev  = data["smile_coeffs"]
                data["smile_coeffs"] = [
                    alpha * c + (1.0 - alpha) * p
                    for c, p in zip(coeffs, prev)
                ]
                data["smile_fit_count"] = data.get("smile_fit_count", 0) + 1
            except Exception:
                pass

        c = data["smile_coeffs"]
        smile_ivs = {}
        for sym, K in self.STRIKES.items():
            m = math.log(K / S)
            raw_iv      = c[0] * m * m + c[1] * m + c[2]
            adjusted_iv = raw_iv + self.IV_BIAS.get(sym, 0.0)
            smile_ivs[sym] = max(0.05, adjusted_iv)

        return smile_ivs

    # ------------------------------------------------------------------ #
    #  PURE-PYTHON WEIGHTED POLYFIT                                        #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _wpolyfit2(xs, ys, ws):
        sw   = sum(ws)
        sx   = sum(w * x       for w, x    in zip(ws, xs))
        sx2  = sum(w * x**2    for w, x    in zip(ws, xs))
        sx3  = sum(w * x**3    for w, x    in zip(ws, xs))
        sx4  = sum(w * x**4    for w, x    in zip(ws, xs))
        sy   = sum(w * y       for w, y    in zip(ws, ys))
        sxy  = sum(w * x * y   for w, x, y in zip(ws, xs, ys))
        sx2y = sum(w * x**2 * y for w, x, y in zip(ws, xs, ys))
        A = [[sx4, sx3, sx2], [sx3, sx2, sx], [sx2, sx, sw]]
        b = [sx2y, sxy, sy]
        for i in range(3):
            max_row = max(range(i, 3), key=lambda r: abs(A[r][i]))
            A[i], A[max_row] = A[max_row], A[i]
            b[i], b[max_row] = b[max_row], b[i]
            piv = A[i][i]
            if abs(piv) < 1e-12:
                return None
            for j in range(i + 1, 3):
                f = A[j][i] / piv
                A[j] = [A[j][k] - f * A[i][k] for k in range(3)]
                b[j] -= f * b[i]
        c = [0.0] * 3
        for i in range(2, -1, -1):
            c[i] = (b[i] - sum(A[i][k] * c[k] for k in range(i + 1, 3))) / A[i][i]
        return c

    # ------------------------------------------------------------------ #
    #  ORDER BOOK HELPERS                                                  #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _mid(depth: OrderDepth) -> float | None:
        bid = max(depth.buy_orders)  if depth.buy_orders  else None
        ask = min(depth.sell_orders) if depth.sell_orders else None
        if bid and ask:
            return (bid + ask) / 2.0
        return bid or ask

    def _room_buy(self, sym: str, pos: int) -> int:
        return max(0, self.LIMITS[sym] - pos)

    def _room_sell(self, sym: str, pos: int) -> int:
        return max(0, self.LIMITS[sym] + pos)

    def _take_buy(self, sym, depth, max_px, pos, cap, orders):
        done = 0
        for ask in sorted(depth.sell_orders):
            if ask > max_px or done >= cap:
                break
            qty = min(-depth.sell_orders[ask], self._room_buy(sym, pos), cap - done)
            if qty > 0:
                orders.append(Order(sym, ask, qty))
                pos += qty; done += qty
        return pos, done

    def _take_sell(self, sym, depth, min_px, pos, cap, orders):
        done = 0
        for bid in sorted(depth.buy_orders, reverse=True):
            if bid < min_px or done >= cap:
                break
            qty = min(depth.buy_orders[bid], self._room_sell(sym, pos), cap - done)
            if qty > 0:
                orders.append(Order(sym, bid, -qty))
                pos -= qty; done += qty
        return pos, done

    def _quote(self, sym, depth, fair, pos, halfspread, size, orders):
        if size <= 0:
            return
        bid_px = max(depth.buy_orders)  if depth.buy_orders  else None
        ask_px = min(depth.sell_orders) if depth.sell_orders else None
        skew   = int(round(6.0 * pos / self.LIMITS[sym]))
        b = int(math.floor(fair - halfspread - skew))
        a = int(math.ceil (fair + halfspread - skew))
        if bid_px is not None:
            b = min(b, bid_px + 1)
        if ask_px is not None:
            a = max(a, ask_px - 1)
        if a <= b:
            a = b + 1
        bq = min(size, self._room_buy (sym, pos))
        sq = min(size, self._room_sell(sym, pos))
        if bq > 0:
            orders.append(Order(sym, b,  bq))
        if sq > 0:
            orders.append(Order(sym, a, -sq))

    # ------------------------------------------------------------------ #
    #  MARK FLOW TRACKING  (U-4)                                          #
    # ------------------------------------------------------------------ #
    def update_mark_flow(self, state: TradingState, data: dict):
        flow = data.setdefault("mark_flow", {})
        for mark_data in flow.values():
            for sym in list(mark_data.keys()):
                mark_data[sym] *= 0.98
        for sym, trades in state.market_trades.items():
            for t in trades:
                for participant in [t.buyer, t.seller]:
                    if participant and participant.startswith("Mark"):
                        mark_data = flow.setdefault(participant, {})
                        direction = 1 if participant == t.buyer else -1
                        mark_data[sym] = mark_data.get(sym, 0.0) + direction * t.quantity

    def get_mark_signal(self, data: dict, sym: str) -> float:
        flow  = data.get("mark_flow", {})
        total = sum(d.get(sym, 0.0) for d in flow.values())
        return max(-1.0, min(1.0, total / 150.0))

    # ------------------------------------------------------------------ #
    #  MAIN RUN                                                            #
    # ------------------------------------------------------------------ #
    def run(self, state: TradingState):
        result: dict[str, list] = {}
        data = self.load_data(state.traderData)

        progress    = min(1.0, state.timestamp / 1_000_000.0)
        current_tte = max(1e-5, (4.0 - progress) / 365.0)

        ex_depth  = state.order_depths.get("VELVETFRUIT_EXTRACT")
        hyd_depth = state.order_depths.get("HYDROGEL_PACK")
        S      = self._mid(ex_depth)  if ex_depth  else None
        hy_mid = self._mid(hyd_depth) if hyd_depth else None

        # ── State updates ─────────────────────────────────────────────── #
        if hy_mid is not None:
            data["hydro_hist"].append(hy_mid)
            data["hydro_hist"] = data["hydro_hist"][-120:]
            if data["hydro_ema"] is None:
                data["hydro_ema"] = hy_mid
            else:
                data["hydro_ema"] = (2/21) * hy_mid + (19/21) * data["hydro_ema"]

        if S is not None:
            data["vev_ema"] = (2/11) * S + (9/11) * data["vev_ema"]

        self.update_mark_flow(state, data)

        # ================================================================ #
        # HYDROGEL_PACK                                                     #
        # ================================================================ #
        # if hyd_depth and data["hydro_ema"] is not None:
        #     orders: list = []
        #     pos  = state.position.get("HYDROGEL_PACK", 0)
        #     fair = data["hydro_ema"]
        #     fair += self.get_mark_signal(data, "HYDROGEL_PACK") * 1.0

        #     pos, _ = self._take_buy ("HYDROGEL_PACK", hyd_depth, fair - 2, pos, 20, orders)
        #     pos, _ = self._take_sell("HYDROGEL_PACK", hyd_depth, fair + 2, pos, 20, orders)
        #     self._quote("HYDROGEL_PACK", hyd_depth, fair, pos, 7, 12, orders)

        #     '''
        #     # U-5: mean-reversion overlay on large deviations
        #     if hy_mid is not None:
        #         dev = hy_mid - fair
        #         if dev < -8 and self._room_buy("HYDROGEL_PACK", pos) >= 5:
        #             ask = min(hyd_depth.sell_orders) if hyd_depth.sell_orders else None
        #             if ask and ask < fair - 4:
        #                 orders.append(Order("HYDROGEL_PACK", ask, 5))
        #                 pos += 5
        #         elif dev > 8 and self._room_sell("HYDROGEL_PACK", pos) >= 5:
        #             bid = max(hyd_depth.buy_orders) if hyd_depth.buy_orders else None
        #             if bid and bid > fair + 4:
        #                 orders.append(Order("HYDROGEL_PACK", bid, -5))
        #                 pos -= 5
        # '''
        #     result["HYDROGEL_PACK"] = orders
# ================================================================ #
        # HYDROGEL_PACK (Data-Calibrated Mean Reversion)                   #
        # ================================================================ #
        if hyd_depth:
            orders = []
            pos = state.position.get('HYDROGEL_PACK', 0)
            
            # 1. Historical Anchor from Round 4 Statistical Analysis
            # The exact global mean across all ticks is 9994.65.
            long_term_anchor = 9994.65
            
            # 2. Update the tracking EMA (Alpha 0.02 for a ~50-tick smooth window)
            if data.get("hydro_ema") is None:
                data["hydro_ema"] = long_term_anchor
            else:
                data["hydro_ema"] = 0.01 * hy_mid + 0.99 * data["hydro_ema"]
            
            # 3. Dynamic Fair Value
            # Weight the empirical anchor heavily (80%) to resist chasing noise.
            # The 20% EMA allows for slight daily micro-trends without losing the anchor.
            fair = (0.15 * long_term_anchor) + (0.85 * data["hydro_ema"])
            
            # 4. Aggressive Reversion Sniping (Taking Liquidity)
            # Std Dev is ~34 ticks. We "snipe" when price deviates by ~0.75 Std Dev (25 ticks).
            snipe_threshold = 25.0
            
            if hy_mid < fair - snipe_threshold:
                # Price is unsustainably cheap. Take up to 40 lots of asks below fair.
                # Note: We use self._take_buy (with the underscore) to fix the bug.
                pos, _ = self._take_buy('HYDROGEL_PACK', hyd_depth, fair - 1, pos, 40, orders)
                
            elif hy_mid > fair + snipe_threshold:
                # Price is unsustainably expensive. Take up to 40 lots of bids above fair.
                pos, _ = self._take_sell('HYDROGEL_PACK', hyd_depth, fair + 1, pos, 40, orders)

            # 5. Passive Market Making (Providing Liquidity)
            # We use a tight halfspread of 3 to capture the constant "jitter" back to the mean.
            # The self._quote method automatically skews your bid/ask based on your current inventory.
            self._quote('HYDROGEL_PACK', hyd_depth, fair, pos, 3, 15, orders)
            
            result['HYDROGEL_PACK'] = orders
        # ================================================================ #
        # VOUCHER OPTIONS                                                   #
        # ================================================================ #
        if S is not None and S > 0:
            smile_ivs = self.fit_smile(S, current_tte, state.order_depths, data)

            # ── Scalping opportunities (all strikes, no convergence gate) ─ #
            opportunities = []
            for sym, K in self.STRIKES.items():
                if sym in self.SCALP_DISABLED:
                    continue
                depth = state.order_depths.get(sym)
                if not depth:
                    continue
                smile_iv = smile_ivs.get(sym)
                if smile_iv is None:
                    continue
                bs_fair = self.bs_price(S, K, current_tte, smile_iv)
                pos     = state.position.get(sym, 0)

                for ask, vol in depth.sell_orders.items():
                    edge = bs_fair - ask
                    if edge > self.MIN_EDGE:
                        opportunities.append({
                            "sym": sym, "side": "BUY", "px": ask,
                            "qty": -vol, "edge": edge,
                        })
                for bid, vol in depth.buy_orders.items():
                    edge = bid - bs_fair
                    if edge > self.MIN_EDGE:
                        opportunities.append({
                            "sym": sym, "side": "SELL", "px": bid,
                            "qty": vol, "edge": edge,
                        })

            opportunities.sort(key=lambda x: x["edge"], reverse=True)
            for opp in opportunities:
                sym = opp["sym"]
                pos = state.position.get(sym, 0)
                room = (self._room_buy(sym, pos) if opp["side"] == "BUY"
                        else self._room_sell(sym, pos))
                qty = min(opp["qty"], room)
                if qty <= 0:
                    continue
                signed_qty = qty if opp["side"] == "BUY" else -qty
                result.setdefault(sym, []).append(Order(sym, opp["px"], signed_qty))
                state.position[sym] = pos + signed_qty

            # ── U-2: passive MM on tight-spread strikes (FIX-1/2/3 applied) #
            fits = data.get("smile_fit_count", 0)

            # FIX-1: gate — no MM until smile is calibrated
            if fits >= self.MIN_SMILE_FITS:

                # FIX-2: dynamic halfspread — wide early, tightens as smile converges
                calibration  = min(1.0, fits / self.MM_HS_DECAY)
                mm_halfspread = max(
                    self.MM_HS_MIN,
                    int(self.MM_HS_START * (1.0 - calibration))
                )

                for sym in self.MM_STRIKES:
                    depth = state.order_depths.get(sym)
                    K     = self.STRIKES.get(sym)
                    if not depth or K is None:
                        continue
                    smile_iv = smile_ivs.get(sym)
                    if smile_iv is None:
                        continue

                    bs_fair = self.bs_price(S, K, current_tte, smile_iv)

                    # FIX-3: sanity check — skip if model is too far from market
                    market_mid = self._mid(depth)
                    if market_mid and market_mid > 0:
                        model_error = abs(bs_fair - market_mid) / market_mid
                        if model_error > self.MM_MAX_MODEL_ERROR:
                            continue  # model not trustworthy for this strike

                    pos       = state.position.get(sym, 0)
                    mm_orders = result.setdefault(sym, [])
                    self._quote(sym, depth, bs_fair, pos,
                                mm_halfspread, self.MM_SIZE, mm_orders)

            # ── Delta hedge via VELVETFRUIT_EXTRACT ──────────────────────── #
            if ex_depth:
                net_delta = 0.0
                for sym, K in self.STRIKES.items():
                    pos = state.position.get(sym, 0)
                    if pos != 0:
                        iv = smile_ivs.get(sym, 0.27)
                        net_delta += pos * self.bs_delta(S, K, current_tte, iv)

                ex_pos = state.position.get("VELVETFRUIT_EXTRACT", 0)
                excess = abs(net_delta) - self.DELTA_LIMIT
                if excess > 0:
                    hedge_raw = math.copysign(excess, net_delta) * self.HEDGE_FRACTION
                    target    = int(round(-hedge_raw))
                    clamped   = max(-self.LIMITS["VELVETFRUIT_EXTRACT"],
                                   min(self.LIMITS["VELVETFRUIT_EXTRACT"],
                                       ex_pos + target))
                    need      = clamped - ex_pos
                    ex_orders = result.setdefault("VELVETFRUIT_EXTRACT", [])
                    if need > 0:
                        ask = min(ex_depth.sell_orders) if ex_depth.sell_orders else None
                        if ask:
                            q = min(need, self._room_buy("VELVETFRUIT_EXTRACT", ex_pos))
                            if q > 0:
                                ex_orders.append(Order("VELVETFRUIT_EXTRACT", ask, q))
                                ex_pos += q
                    elif need < 0:
                        bid = max(ex_depth.buy_orders) if ex_depth.buy_orders else None
                        if bid:
                            q = min(-need, self._room_sell("VELVETFRUIT_EXTRACT", ex_pos))
                            if q > 0:
                                ex_orders.append(Order("VELVETFRUIT_EXTRACT", bid, -q))
                                ex_pos -= q

        # ================================================================ #
        # VELVETFRUIT_EXTRACT — standalone market making + hedge            #
        # ================================================================ #
        if ex_depth and S is not None:
            ex_orders = result.setdefault("VELVETFRUIT_EXTRACT", [])
            ex_pos    = state.position.get("VELVETFRUIT_EXTRACT", 0)
            for o in ex_orders:
                ex_pos += o.quantity
            vev_fair  = data["vev_ema"]
            vev_fair += self.get_mark_signal(data, "VELVETFRUIT_EXTRACT") * 0.5

            if self._room_buy("VELVETFRUIT_EXTRACT", ex_pos) > 0 \
               or self._room_sell("VELVETFRUIT_EXTRACT", ex_pos) > 0:
                ex_pos, _ = self._take_buy(
                    "VELVETFRUIT_EXTRACT", ex_depth, vev_fair - 1, ex_pos, 15, ex_orders)
                ex_pos, _ = self._take_sell(
                    "VELVETFRUIT_EXTRACT", ex_depth, vev_fair + 1, ex_pos, 15, ex_orders)
                self._quote("VELVETFRUIT_EXTRACT", ex_depth, vev_fair,
                            ex_pos, 3, 10, ex_orders)

        data["last_ts"] = state.timestamp
        return result, 0, self.dump_data(data)