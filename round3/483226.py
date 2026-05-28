from datamodel import OrderDepth, TradingState, Order
import json
import math

class Trader:
    LIMITS = {
        "HYDROGEL_PACK": 200, "VELVETFRUIT_EXTRACT": 200,
        "VEV_4000": 150, "VEV_4500": 150, "VEV_5000": 150,
        "VEV_5100": 150, "VEV_5200": 150, "VEV_5400": 150,
        "VEV_5500": 150, "VEV_6000": 150, "VEV_6500": 150
    }

    STRIKES = {
        "VEV_4000": 4000, "VEV_4500": 4500, "VEV_5000": 5000,
        "VEV_5100": 5100, "VEV_5200": 5200, "VEV_5400": 5400,
        "VEV_5500": 5500, "VEV_6000": 6000, "VEV_6500": 6500
    }

    DELTA_LIMIT = 180.0
    MIN_EDGE = 1.5

    # Per-strike IV seeds derived from Round 3 Day 2 historical data
    # (inverting BS on every market mid-price observation, taking mean)
    IV_SEEDS = {
        'VEV_4000': 0.9355,  # deep ITM — high IV, high std, adapt quickly
        'VEV_4500': 0.5367,
        'VEV_5000': 0.2638,
        'VEV_5100': 0.2608,
        'VEV_5200': 0.2688,
        'VEV_5400': 0.2500,  # KEY: lower than 5200, NOT higher
        'VEV_5500': 0.2718,
        'VEV_6000': 0.4421,
        'VEV_6500': 0.6739,
    }

    def load_data(self, trader_data):
        d = {
            'iv_ema': dict(self.IV_SEEDS),  # per-strike IV EMAs
            'hydro_hist': [],
            'last_ts': -1,
            'vev_ema': 5262.0,  # seeded at actual Round 3 mean (not 5255)
        }
        if trader_data:
            try:
                x = json.loads(trader_data)
                d.update(x)
                # Handle old scalar iv_ema from previous versions
                if not isinstance(d['iv_ema'], dict):
                    d['iv_ema'] = dict(self.IV_SEEDS)
                # Back-fill any missing strikes
                for sym, seed in self.IV_SEEDS.items():
                    d['iv_ema'].setdefault(sym, seed)
            except:
                pass
        if 'hydro_hist' not in d:
            d['hydro_hist'] = []
        if 'last_ts' not in d:
            d['last_ts'] = -1
        return d

    def dump_data(self, d):
        return json.dumps(d, separators=(',', ':'))

    # --- OPTIONS MATH ---
    def cdf(self, x):
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    def get_bs_price(self, S, K, T, sigma):
        if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
            return max(0.0, S - K)
        d1 = (math.log(S / K) + 0.5 * sigma**2 * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        return S * self.cdf(d1) - K * self.cdf(d2)

    def get_bs_delta(self, S, K, T, sigma):
        if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
            return 1.0 if S > K else 0.0
        d1 = (math.log(S / K) + 0.5 * sigma**2 * T) / (sigma * math.sqrt(T))
        return self.cdf(d1)

    def mid_price(self, depth: OrderDepth):
        bid = max(depth.buy_orders.keys()) if depth.buy_orders else None
        ask = min(depth.sell_orders.keys()) if depth.sell_orders else None
        if bid is not None and ask is not None:
            return (bid + ask) / 2.0
        return bid if bid is not None else ask

    def solve_iv(self, S, K, T, market_price):
        # Wider bounds (0.01-2.0) and 50 iterations so deep ITM/OTM
        # strikes (VEV_4000 IV~0.94, VEV_6500 IV~0.67) don't clamp at 0.50.
        intrinsic = max(0.0, S - K)
        if market_price <= intrinsic + 0.01:
            return None  # no time value to solve
        low, high = 0.01, 2.0
        for _ in range(50):
            mid = (low + high) / 2.0
            if self.get_bs_price(S, K, T, mid) < market_price:
                low = mid
            else:
                high = mid
        iv = (low + high) / 2.0
        return iv if 0.05 < iv < 1.5 else None

    # --- HELPER FUNCTIONS ---
    def best_bid_ask(self, depth):
        bid = max(depth.buy_orders) if depth.buy_orders else None
        ask = min(depth.sell_orders) if depth.sell_orders else None
        return bid, ask

    def room_buy(self, product, pos):
        return max(0, self.LIMITS[product] - pos)

    def room_sell(self, product, pos):
        return max(0, self.LIMITS[product] + pos)

    def take_buy(self, product, depth, max_price, pos, qty_cap, orders):
        if qty_cap <= 0:
            return pos, 0
        done = 0
        for ask in sorted(depth.sell_orders):
            if ask > max_price:
                break
            avail = -depth.sell_orders[ask]
            qty = min(avail, self.room_buy(product, pos), qty_cap - done)
            if qty > 0:
                orders.append(Order(product, ask, qty))
                pos += qty
                done += qty
            if done >= qty_cap:
                break
        return pos, done

    def take_sell(self, product, depth, min_price, pos, qty_cap, orders):
        if qty_cap <= 0:
            return pos, 0
        done = 0
        for bid in sorted(depth.buy_orders, reverse=True):
            if bid < min_price:
                break
            avail = depth.buy_orders[bid]
            qty = min(avail, self.room_sell(product, pos), qty_cap - done)
            if qty > 0:
                orders.append(Order(product, bid, -qty))
                pos -= qty
                done += qty
            if done >= qty_cap:
                break
        return pos, done

    def quote(self, product, depth, fair, pos, halfspread, size, orders):
        if size <= 0:
            return
        bid, ask = self.best_bid_ask(depth)
        skew = int(round(8.0 * pos / self.LIMITS[product]))
        bpx = int(math.floor(fair - halfspread - skew))
        apx = int(math.ceil(fair + halfspread - skew))
        if bid is not None:
            bpx = min(bpx, bid + 1)
        if ask is not None:
            apx = max(apx, ask - 1)
        if apx <= bpx:
            apx = bpx + 1
        bq = min(size, self.room_buy(product, pos))
        sq = min(size, self.room_sell(product, pos))
        if bq > 0:
            orders.append(Order(product, bpx, bq))
        if sq > 0:
            orders.append(Order(product, apx, -sq))

    def run(self, state: TradingState):
        result = {}
        data = self.load_data(state.traderData)

        progress = min(1.0, state.timestamp / 1000000.0)
        current_tte = (5.0 / 365.0) - progress * (1.0 / 365.0)

        ex_depth = state.order_depths.get('VELVETFRUIT_EXTRACT')
        hyd = state.order_depths.get('HYDROGEL_PACK')

        ex_mid = self.mid_price(ex_depth) if ex_depth else None
        hy_mid = self.mid_price(hyd) if hyd else None

        # --- STATE TRACKING ---
        if state.timestamp != data['last_ts']:
            if hy_mid is not None:
                data['hydro_hist'].append(hy_mid)
                data['hydro_hist'] = data['hydro_hist'][-120:]
            data['tick_count'] = data.get('tick_count', 0) + 1
            data['last_ts'] = state.timestamp

        # =================================================================
        # HYDROGEL_PACK
        # =================================================================
        if hyd:
            orders = []
            pos = state.position.get('HYDROGEL_PACK', 0)
            fair = 9991.0
            window = 10
            if len(data['hydro_hist']) >= window:
                recent = sum(data['hydro_hist'][-window:]) / window
                fair = 9991.0 * 0.1+ 0.9 * recent
            pos, _ = self.take_buy('HYDROGEL_PACK', hyd, fair - 2, pos, 20, orders)
            pos, _ = self.take_sell('HYDROGEL_PACK', hyd, fair + 2, pos, 20, orders)
            self.quote('HYDROGEL_PACK', hyd, fair, pos, 8, 10, orders)
            result['HYDROGEL_PACK'] = orders

        # --- OPTIONAL ADAPTIVE-EMA HYDROGEL (enable in Round 4 after backtesting) ---
        # if hyd and hy_mid is not None:
        #     orders = []
        #     pos = state.position.get('HYDROGEL_PACK', 0)
        #     hist = data['hydro_hist']
        #     if len(hist) >= 5:
        #         mad = sum(abs(hist[i] - hist[i-1]) for i in range(-4, 0)) / 4
        #         # Higher volatility → faster EMA. Effective window ≈ 2/alpha - 1.
        #         # mad/20 targets alpha~0.15 at typical vol, alpha~0.40 if doubling.
        #         alpha = min(0.5, max(0.08, mad / 20.0))
        #     else:
        #         alpha = 0.15  # default ≈ window 12
        #     data['hydro_ema'] = alpha * hy_mid + (1 - alpha) * data.get('hydro_ema', 9990.81)
        #     fair = data['hydro_ema']
        #     pos, _ = self.take_buy('HYDROGEL_PACK', hyd, fair - 2, pos, 20, orders)
        #     pos, _ = self.take_sell('HYDROGEL_PACK', hyd, fair + 2, pos, 20, orders)
        #     self.quote('HYDROGEL_PACK', hyd, fair, pos, 8, 10, orders)
        #     result['HYDROGEL_PACK'] = orders

        # =================================================================
        # ================================================================
        # VOUCHERS — per-strike IV mean reversion
        # ================================================================
        # Each strike has its own IV EMA seeded from historical data.
        # This fixes the VEV_5400 issue: true IV=0.25, not 0.27.
        # Using wrong IV caused +2.66 tick systematic overprice → kept buying it.
        # solve_iv bounds widened (0.01-2.0, 50 iters) for deep ITM/OTM strikes.
        # No delta limit check — VEV is independent, not used for hedging.
        if ex_mid:
            opps = []

            for sym, K in self.STRIKES.items():
                depth = state.order_depths.get(sym)
                if not depth:
                    continue

                # Update this strike's own IV EMA
                v_mid = self.mid_price(depth)
                if v_mid is not None:
                    iv_solved = self.solve_iv(ex_mid, K, current_tte, v_mid)
                    if iv_solved is not None:
                        data['iv_ema'][sym] = 0.15 * iv_solved + 0.85 * data['iv_ema'][sym]

                # BS fair using THIS strike's own IV
                sigma_i = data['iv_ema'][sym]
                bs_fair = self.get_bs_price(ex_mid, K, current_tte, sigma_i)
                pos = state.position.get(sym, 0)

                buy_edge_req = self.MIN_EDGE if pos >= 0 else 0.0
                sell_edge_req = self.MIN_EDGE if pos <= 0 else 0.0

                for ask, vol in depth.sell_orders.items():
                    edge = bs_fair - ask
                    if edge > buy_edge_req:
                        opps.append({'sym': sym, 'px': ask, 'qty': -vol,
                                     'edge': edge, 'side': 'BUY'})
                for bid, vol in depth.buy_orders.items():
                    edge = bid - bs_fair
                    if edge > sell_edge_req:
                        opps.append({'sym': sym, 'px': bid, 'qty': vol,
                                     'edge': edge, 'side': 'SELL'})

            # Sort by raw edge (no delta weighting — not hedging)
            opps.sort(key=lambda x: x['edge'], reverse=True)

            for o in opps:
                sym = o['sym']
                pos = state.position.get(sym, 0)
                room = (self.room_buy(sym, pos) if o['side'] == 'BUY'
                        else self.room_sell(sym, pos))
                trade_qty = min(o['qty'], room)
                if trade_qty > 0:
                    signed_qty = trade_qty if o['side'] == 'BUY' else -trade_qty
                    orders = result.get(sym, [])
                    orders.append(Order(sym, o['px'], signed_qty))
                    result[sym] = orders
                    state.position[sym] = pos + signed_qty

        # ================================================================
        # VELVETFRUIT_EXTRACT — standalone EMA market making
        # ================================================================
        # Backtested findings:
        #   halfspread=6 with alpha=0.7: fair≈spot always, quotes outside spread
        #   100% of the time → 0 PnL. Clip logic places at bid+1/ask-1 but
        #   inventory never turns over because both sides fill equally.
        #   take_edge=4: fired 0 times in simulation.
        #   FIX: halfspread=3, alpha=2/11 (EMA-10), take_edge=1 → ~1640 PnL
        if ex_depth and ex_mid is not None:
            orders = []
            ex_pos = state.position.get('VELVETFRUIT_EXTRACT', 0)

            # EMA-10 tracks drift without overfitting
            alpha = 2 / (10 + 1)
            data['vev_ema'] = alpha * ex_mid + (1 - alpha) * data['vev_ema']
            vev_fair = data['vev_ema']

            # take_edge=1: buy when ask<=fair-1, sell when bid>=fair+1
            # Fires on genuine mispricings (~28x/day vs 0x with edge=4)
            pos, _ = self.take_buy('VELVETFRUIT_EXTRACT', ex_depth,
                                   vev_fair - 1, ex_pos, 20, orders)
            pos, _ = self.take_sell('VELVETFRUIT_EXTRACT', ex_depth,
                                    vev_fair + 1, pos, 20, orders)

            # halfspread=3: quotes sit inside the 5-tick market spread
            # quote() clips to bid+1/ask-1 and applies inventory skew
            self.quote('VELVETFRUIT_EXTRACT', ex_depth, vev_fair, pos, 3, 15, orders)

            result['VELVETFRUIT_EXTRACT'] = orders

        data['last_ts'] = state.timestamp
        return result, 0, self.dump_data(data)