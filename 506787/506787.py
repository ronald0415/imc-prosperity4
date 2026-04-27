"""
IMC Prosperity 4 – Round 4 Trader  (UPGRADED from 493282.py)
=============================================================

FIXES vs 493282.py
──────────────────
BUG-1  Polyfit coefficient ordering
       np.polyfit(x,y,2) → [a2, a1, a0] (DESCENDING).
       Old code initialised smile_coeffs = [0.30, 0.0, 0.50] as if ascending
       and evaluated c[2]*m^2 + c[1]*m + c[0] — swapping a2 and a0.
       After first EMA update ATM smile_IV rose to ~2.0 → every iv_dev
       exceeded IV_DEV_MAX → zero option trades for the entire round.
       FIX: use np.polyval(coeffs, m) which handles descending order
       correctly, and re-seed defaults in the same convention.

BUG-2  HYDROGEL 10% anchor at 9991 (stale Round-3 prior)
       Created large directional bets when R4 price diverged from 9991.
       FIX: pure adaptive EMA, seed from first observation.

BUG-3  Double condition kills real edges
       Required BOTH price_edge > MIN_EDGE AND abs(iv_dev) > threshold.
       When smile is noisy the two conditions rarely align.
       FIX: single price-edge gate; iv_dev used only to rank, not filter.

STRATEGY UPGRADES
─────────────────
U-1  Per-strike structural IV bias
     IV deviation chart shows persistent biases independent of smile fit:
       K=4500  μ=+0.068 → smile consistently under-estimates → sell signal
       K=5000  μ=-0.024 → smile over-estimates       → buy signal
       K=5100  μ=-0.013 → slight structural under-price
       K=5400  μ=-0.015 → structural under-price
       K=5500  μ=-0.016 → structural under-price
       K=6000  μ=+0.029 → structural over-price
     These are added as a fixed offset to the smile-implied fair value.

U-2  Market-making on tight-spread options
     VEV_5400, VEV_5500, VEV_6000, VEV_6500 all have avg spread ≤ 1.2.
     We now passively quote a 2-tick market in these options for small size,
     capturing spread in addition to IV-misprice scalping.

U-3  Weighted smile fitting
     Deep-ITM and deep-OTM IV observations are noisier.  Weight each
     point by 1/(1 + |moneyness|) so near-ATM strikes drive the fit.

U-4  Smarter Mark flow signals
     Decay cumulative flow by 0.98 each tick so stale positions don't
     permanently skew the signal.  Tilt only 1 XIREC on HYDROGEL, 0.5 on
     VEV-underlying — smaller than before, less directional risk.

U-5  HYDROGEL: cap-aware mean reversion
     Add a thin mean-reversion overlay when price deviates >8 from EMA.
"""

from datamodel import OrderDepth, TradingState, Order
import json
import math
# numpy not used – pure-Python polyfit/polyval replacements below


class Trader:
    # ------------------------------------------------------------------ #
    #  CONFIGURATION                                                       #
    # ------------------------------------------------------------------ #
    LIMITS = {
    "HYDROGEL_PACK": 200, "VELVETFRUIT_EXTRACT": 200,
    **{k: 300 for k in [
        "VEV_4000","VEV_4500","VEV_5000","VEV_5100","VEV_5200",
        "VEV_5300","VEV_5400","VEV_5500","VEV_6000","VEV_6500",
    ]},
}

    STRIKES = {
        "VEV_4000": 4000, "VEV_4500": 4500, "VEV_5000": 5000,
        "VEV_5100": 5100, "VEV_5200": 5200, "VEV_5300": 5300,
        "VEV_5400": 5400, "VEV_5500": 5500, "VEV_6000": 6000,
        "VEV_6500": 6500,
    }

    # U-1: per-strike IV bias derived from observed deviation chart
    # Positive = market consistently prices option ABOVE smile → we fade up (sell)
    # Negative = market consistently prices BELOW smile → we fade down (buy)
    # These are added to the smile fair IV before computing BS fair price.
    # Interpretation: adjust smile IV DOWN by this amount to get true fair IV
    # (so we adjust BS fair price accordingly).
    IV_BIAS = {
        "VEV_4000": 0.000,   # noisy, no clear signal (σ too high)
        "VEV_4500": -0.040,  # μ=+0.068 → market overpriced vs smile → our fair is lower → less likely to buy
        "VEV_5000": +0.015,  # μ=-0.024 → market underpriced vs smile → our fair is higher → more likely to buy
        "VEV_5100": +0.008,  # μ=-0.013
        "VEV_5200": 0.000,   # μ=-0.002, noise only
        "VEV_5300": 0.000,   # μ=+0.001, no signal
        "VEV_5400": +0.010,  # μ=-0.015
        "VEV_5500": +0.010,  # μ=-0.016
        "VEV_6000": -0.020,  # μ=+0.029 → overpriced → our fair is lower
        "VEV_6500": 0.000,   # μ=-0.003, noise only
    }

    # Price edge required to cross the spread (XIRECS).
    # Intentionally low — IV_BIAS and smile residuals already select quality trades.
    MIN_EDGE = 0.5

    # Maximum net-delta exposure before hedging kicks in (underlying units)
    DELTA_LIMIT = 60.0
    HEDGE_FRACTION = 0.80

    # Strikes where we also passively market-make (spread is ≤ 1.5 XIREC avg)
    MM_STRIKES = {"VEV_5400", "VEV_5500", "VEV_6000", "VEV_6500"}
    MM_SIZE = 5      # passive quote size per side
    MM_HALFSPREAD = 1  # 1 tick on each side

    # ------------------------------------------------------------------ #
    #  STATE                                                               #
    # ------------------------------------------------------------------ #
    def load_data(self, trader_data: str) -> dict:
        defaults = {
            "hydro_hist":   [],
            "hydro_ema":    None,      # seeded from first observation
            "vev_ema":      5262.0,
            "last_ts":      -1,
            # FIX BUG-1: coefficients in DESCENDING polyfit order [a2, a1, a0]
            # i.e. IV ≈ a2*m^2 + a1*m + a0.  Seeded from R3 smile analysis:
            # ATM IV ≈ 0.27, parabola curvature ≈ 5, slight asymmetry a1=-0.10
            "smile_coeffs": [5.0, -0.10, 0.27],
            "mark_flow":    {},        # {mark_id: {sym: net_qty}} — decayed each tick
            "smile_fit_count": 0,
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
        d1 = (math.log(S / K) + 0.5 * sigma ** 2 * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        return S * self._cdf(d1) - K * self._cdf(d2)

    def bs_delta(self, S: float, K: float, T: float, sigma: float) -> float:
        if T <= 0 or sigma <= 0:
            return 1.0 if S > K else 0.0
        d1 = (math.log(S / K) + 0.5 * sigma ** 2 * T) / (sigma * math.sqrt(T))
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
    #  VOLATILITY SMILE FITTING  (BUG-1 FIXED)                            #
    # ------------------------------------------------------------------ #
    def fit_smile(self, S: float, T: float, depths: dict, data: dict) -> dict:
        """
        Fit a weighted quadratic to (log(K/S), solved_IV) across all strikes.

        FIX BUG-1
        ---------
        np.polyfit returns coefficients in DESCENDING order: [a2, a1, a0].
        We now store and evaluate them the same way, using np.polyval which
        accepts descending-order coefficients natively.

        U-3: Weighted fit
        -----------------
        Points with |moneyness| > 0.15 (deep ITM/OTM) are noisier and
        down-weighted by w = exp(-2 * |m|).

        Returns dict: sym -> (smile_iv + IV_BIAS[sym])
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
            if mid - intrinsic < 0.5:          # too little time value
                continue
            iv = self.solve_iv(S, K, T, mid)
            if iv is None:
                continue
            m = math.log(K / S)
            m_vals.append(m)
            iv_vals.append(iv)
            # U-3: weight: near-ATM points dominate
            w_vals.append(math.exp(-2.0 * abs(m)))

        smile_ivs: dict[str, float] = {}

        if len(m_vals) >= 3:
            try:
                # FIX: weighted polyfit — np returns descending [a2, a1, a0]
                coeffs = self._wpolyfit2(m_vals, iv_vals, w_vals)
                if coeffs is None: raise ValueError('singular')
                # EMA-smooth: alpha=0.20 for stability
                alpha = 0.20
                prev  = data["smile_coeffs"]
                data["smile_coeffs"] = [
                    alpha * c + (1.0 - alpha) * p
                    for c, p in zip(coeffs, prev)
                ]
                data["smile_fit_count"] = data.get("smile_fit_count", 0) + 1
            except Exception:
                pass  # keep previous coefficients

        c = data["smile_coeffs"]   # [a2, a1, a0] in descending order

        for sym, K in self.STRIKES.items():
            m = math.log(K / S)
            # FIX BUG-1: np.polyval handles descending-order coefficients correctly
            raw_smile_iv = c[0]*m*m + c[1]*m + c[2]
            # U-1: apply per-strike structural bias
            adjusted_iv  = raw_smile_iv + self.IV_BIAS.get(sym, 0.0)
            smile_ivs[sym] = max(0.05, adjusted_iv)

        return smile_ivs

    # ------------------------------------------------------------------ #
    #  PURE-PYTHON WEIGHTED POLYFIT  (drop-in replacement for numpy)      #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _wpolyfit2(xs, ys, ws):
        """
        Weighted least-squares fit of a degree-2 polynomial.
        Returns coefficients [a2, a1, a0] (descending, same convention as
        np.polyfit), so the polynomial is evaluated as a2*x^2 + a1*x + a0.
        Returns None if the system is degenerate (caller should keep previous).
        """
        sw   = sum(ws)
        sx   = sum(w*x      for w, x   in zip(ws, xs))
        sx2  = sum(w*x**2   for w, x   in zip(ws, xs))
        sx3  = sum(w*x**3   for w, x   in zip(ws, xs))
        sx4  = sum(w*x**4   for w, x   in zip(ws, xs))
        sy   = sum(w*y      for w, y   in zip(ws, ys))
        sxy  = sum(w*x*y    for w, x, y in zip(ws, xs, ys))
        sx2y = sum(w*x**2*y for w, x, y in zip(ws, xs, ys))
        # 3x3 normal equations:  A c = b
        A = [[sx4, sx3, sx2], [sx3, sx2, sx], [sx2, sx, sw]]
        b = [sx2y, sxy, sy]
        # Gaussian elimination with partial pivoting
        for i in range(3):
            # Find pivot
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
        # Back-substitution
        c = [0.0] * 3
        for i in range(2, -1, -1):
            c[i] = (b[i] - sum(A[i][k] * c[k] for k in range(i + 1, 3))) / A[i][i]
        return c   # [a2, a1, a0]

    # ------------------------------------------------------------------ #
    #  ORDER BOOK HELPERS                                                  #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _mid(depth: OrderDepth) -> float | None:
        bid = max(depth.buy_orders) if depth.buy_orders else None
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
        bid_px = max(depth.buy_orders) if depth.buy_orders else None
        ask_px = min(depth.sell_orders) if depth.sell_orders else None
        skew   = int(round(6.0 * pos / self.LIMITS[sym]))
        b = int(math.floor(fair - halfspread - skew))
        a = int(math.ceil(fair + halfspread - skew))
        if bid_px is not None:
            b = min(b, bid_px + 1)
        if ask_px is not None:
            a = max(a, ask_px - 1)
        if a <= b:
            a = b + 1
        bq = min(size, self._room_buy(sym, pos))
        sq = min(size, self._room_sell(sym, pos))
        if bq > 0:
            orders.append(Order(sym, b, bq))
        if sq > 0:
            orders.append(Order(sym, a, -sq))

    # ------------------------------------------------------------------ #
    #  COUNTERPARTY TRACKING  (U-4: decayed signals)                      #
    # ------------------------------------------------------------------ #
    def update_mark_flow(self, state: TradingState, data: dict):
        """
        Track net signed quantity per Mark per symbol.
        U-4: decay existing flow by 0.98 each tick so old signals fade.
        """
        flow = data.setdefault("mark_flow", {})
        # Decay existing positions
        for mark_data in flow.values():
            for sym in list(mark_data.keys()):
                mark_data[sym] *= 0.98
        # Accumulate new trades from market_trades
        for sym, trades in state.market_trades.items():
            for t in trades:
                for participant in [t.buyer, t.seller]:
                    if participant and participant.startswith("Mark"):
                        mark_data = flow.setdefault(participant, {})
                        direction = 1 if participant == t.buyer else -1
                        mark_data[sym] = mark_data.get(sym, 0.0) + direction * t.quantity

    def get_mark_signal(self, data: dict, sym: str) -> float:
        """Aggregate decayed flow across all Marks for a symbol → [-1, +1]."""
        flow  = data.get("mark_flow", {})
        total = sum(d.get(sym, 0.0) for d in flow.values())
        cap   = 150.0
        return max(-1.0, min(1.0, total / cap))

    # ------------------------------------------------------------------ #
    #  MAIN RUN                                                            #
    # ------------------------------------------------------------------ #
    def run(self, state: TradingState):
        result: dict[str, list] = {}
        data = self.load_data(state.traderData)

        # Round 4: TTE starts at 4 days, decays over 10,000 ticks (1 round)
        progress    = min(1.0, state.timestamp / 1_000_000.0)
        current_tte = max(1e-5, (4.0 - progress) / 365.0)

        ex_depth  = state.order_depths.get("VELVETFRUIT_EXTRACT")
        hyd_depth = state.order_depths.get("HYDROGEL_PACK")
        S         = self._mid(ex_depth)  if ex_depth  else None
        hy_mid    = self._mid(hyd_depth) if hyd_depth else None

        # ---- State updates ------------------------------------------ #
        if hy_mid is not None:
            data["hydro_hist"].append(hy_mid)
            data["hydro_hist"] = data["hydro_hist"][-120:]
            # FIX BUG-2: seed EMA from first observation, then track
            if data["hydro_ema"] is None:
                data["hydro_ema"] = hy_mid
            else:
                alpha_h = 2 / (20 + 1)   # EMA-20 for smoother fair value
                data["hydro_ema"] = alpha_h * hy_mid + (1 - alpha_h) * data["hydro_ema"]

        if S is not None:
            alpha_v = 2 / 11
            data["vev_ema"] = alpha_v * S + (1 - alpha_v) * data["vev_ema"]

        self.update_mark_flow(state, data)

        # ================================================================ #
        # HYDROGEL_PACK  (BUG-2 fixed + U-5 mean reversion overlay)        #
        # ================================================================ #
        if hyd_depth and data["hydro_ema"] is not None:
            orders: list = []
            pos   = state.position.get("HYDROGEL_PACK", 0)
            fair  = data["hydro_ema"]

            # U-4: U-4 reduced tilt — ±1 XIREC max, decay already smooths signal
            mark_sig = self.get_mark_signal(data, "HYDROGEL_PACK")
            fair += mark_sig * 1.0      # reduced from 2.0 to 1.0

            # Standard market-making quotes
            pos, _ = self._take_buy( "HYDROGEL_PACK", hyd_depth, fair - 2, pos, 20, orders)
            pos, _ = self._take_sell("HYDROGEL_PACK", hyd_depth, fair + 2, pos, 20, orders)
            self._quote("HYDROGEL_PACK", hyd_depth, fair, pos, 7, 12, orders)

            # U-5: thin mean-reversion overlay — fade large deviations from EMA
            if hy_mid is not None:
                dev = hy_mid - fair
                if dev < -8 and self._room_buy("HYDROGEL_PACK", pos) >= 5:
                    ask = min(hyd_depth.sell_orders) if hyd_depth.sell_orders else None
                    if ask and ask < fair - 4:
                        orders.append(Order("HYDROGEL_PACK", ask, 5))
                        pos += 5
                elif dev > 8 and self._room_sell("HYDROGEL_PACK", pos) >= 5:
                    bid = max(hyd_depth.buy_orders) if hyd_depth.buy_orders else None
                    if bid and bid > fair + 4:
                        orders.append(Order("HYDROGEL_PACK", bid, -5))
                        pos -= 5

            result["HYDROGEL_PACK"] = orders

        # ================================================================ #
        # VOUCHER OPTIONS  (BUG-1 + BUG-3 fixed; U-1, U-2, U-3 applied)   #
        # ================================================================ #
        if S is not None and S > 0:
            # Step 1: fit smile (polyfit bug fixed inside fit_smile)
            smile_ivs = self.fit_smile(S, current_tte, state.order_depths, data)

            # Step 2: collect scalping opportunities
            # FIX BUG-3: only require price_edge > MIN_EDGE (no iv_dev filter)
            opportunities = []
            for sym, K in self.STRIKES.items():
                depth = state.order_depths.get(sym)
                if not depth:
                    continue

                smile_iv = smile_ivs.get(sym)
                if smile_iv is None:
                    continue

                bs_fair  = self.bs_price(S, K, current_tte, smile_iv)
                pos      = state.position.get(sym, 0)

                # BUY: ask is below our fair value
                for ask, vol in depth.sell_orders.items():
                    edge = bs_fair - ask
                    if edge > self.MIN_EDGE:
                        opportunities.append({
                            "sym": sym, "side": "BUY", "px": ask,
                            "qty": -vol, "edge": edge, "K": K,
                        })

                # SELL: bid is above our fair value
                for bid, vol in depth.buy_orders.items():
                    edge = bid - bs_fair
                    if edge > self.MIN_EDGE:
                        opportunities.append({
                            "sym": sym, "side": "SELL", "px": bid,
                            "qty": vol, "edge": edge, "K": K,
                        })

            # Sort best edge first, execute
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

            # U-2: passive market-making on tight-spread strikes
            for sym in self.MM_STRIKES:
                depth = state.order_depths.get(sym)
                K     = self.STRIKES.get(sym)
                if not depth or K is None:
                    continue
                smile_iv = smile_ivs.get(sym)
                if smile_iv is None:
                    continue
                bs_fair = self.bs_price(S, K, current_tte, smile_iv)
                pos     = state.position.get(sym, 0)
                mm_orders = result.setdefault(sym, [])
                self._quote(sym, depth, bs_fair, pos,
                            self.MM_HALFSPREAD, self.MM_SIZE, mm_orders)

            # ---- DELTA HEDGE via VELVETFRUIT_EXTRACT -------------------- #
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
        # VELVETFRUIT_EXTRACT standalone market-making                      #
        # ================================================================ #
        if ex_depth and S is not None:
            ex_orders = result.setdefault("VELVETFRUIT_EXTRACT", [])
            ex_pos    = state.position.get("VELVETFRUIT_EXTRACT", 0)
            for o in ex_orders:          # account for hedge orders already queued
                ex_pos += o.quantity
            vev_fair  = data["vev_ema"]
            # U-4: halved tilt on VEV-extract (0.5 instead of 1.5)
            vev_fair += self.get_mark_signal(data, "VELVETFRUIT_EXTRACT") * 0.5
            if self._room_buy("VELVETFRUIT_EXTRACT", ex_pos) > 0 \
               or self._room_sell("VELVETFRUIT_EXTRACT", ex_pos) > 0:
                ex_pos, _ = self._take_buy(
                    "VELVETFRUIT_EXTRACT", ex_depth, vev_fair - 1, ex_pos, 15, ex_orders)
                ex_pos, _ = self._take_sell(
                    "VELVETFRUIT_EXTRACT", ex_depth, vev_fair + 1, ex_pos, 15, ex_orders)
                self._quote("VELVETFRUIT_EXTRACT", ex_depth, vev_fair, ex_pos, 3, 10, ex_orders)

        data["last_ts"] = state.timestamp
        return result, 0, self.dump_data(data)