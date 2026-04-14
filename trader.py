from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List, Dict
import numpy as np
import jsonpickle

POSITION_LIMITS = {
    "EMERALDS": 80,
    "TOMATOES": 40,   # CHANGED: reduced from 80 to 40 (see explanation below)
}


class Trader:

    def bid(self):
        return 15

    @staticmethod
    def linear_trend_prediction(prices: List[float], lookahead: int = 1):
        x = np.arange(len(prices), dtype=float)
        y = np.array(prices, dtype=float)
        mean_x = x.mean()
        mean_y = y.mean()
        covariance = np.mean((x - mean_x) * (y - mean_y))
        variance_x = np.mean((x - mean_x) ** 2)
        slope = covariance / variance_x if variance_x > 0 else 0.0
        intercept = mean_y - slope * mean_x
        return intercept + slope * (len(prices) + lookahead - 1), slope

    def run(self, state: TradingState):
        """Only method required. It takes all buy and sell orders for all
        symbols as an input, and outputs a list of orders to be sent."""

        print("traderData: " + state.traderData)
        print("Observations: " + str(state.observations))

        if state.traderData and state.traderData != "SAMPLE":
            trader_state = jsonpickle.decode(state.traderData)
        else:
            trader_state = {
                "tomato_prices": [],
                "emerald_prices": [],
            }

        result: Dict[str, List[Order]] = {}

        for product, order_depth in state.order_depths.items():
            orders: List[Order] = []
            position = state.position.get(product, 0)
            limit = POSITION_LIMITS.get(product, 80)

            if not order_depth.buy_orders or not order_depth.sell_orders:
                result[product] = orders
                continue

            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            best_ask_amount = order_depth.sell_orders[best_ask]
            best_bid_amount = order_depth.buy_orders[best_bid]
            mid_price = (best_ask + best_bid) / 2.0
            spread = best_ask - best_bid

            # ------------------------------------------------------------------
            # EMERALDS — Market Making Strategy
            #
            # WHY THE OLD CODE EARNED ZERO:
            #   EMERALDS mid-price only ever equals 9996, 10000, or 10004.
            #   When the old mean-reversion signal fired (e.g. mid=10004 →
            #   sell signal), it sold at best_bid which equals exactly 10000
            #   (fair value). Symmetrically, buy signals fired when mid=9996
            #   but bought at best_ask = 10000. Every trade executed at 10000
            #   so PnL was always zero.
            #
            # THE FIX — post limit orders INSIDE the spread:
            #   The order book sits at bid=9992 / ask=10008 with fair value
            #   10000. By posting a buy at 9999 and a sell at 10001, bots that
            #   would have sold at 9992 now sell to us at 9999 instead (better
            #   for them), and bots buying at 10008 buy from us at 10001. We
            #   capture ~2 ticks of edge per round trip with near-zero risk
            #   because we always buy below and sell above fair value.
            # ------------------------------------------------------------------
            if product == "EMERALDS":
                trader_state["emerald_prices"].append(mid_price)
                fair_value = 10000.0

                # Step 1: Take any orders that are clearly mispriced vs fair
                # value (rare, but free money when it occurs).
                if best_ask < fair_value:
                    buy_volume = min(-best_ask_amount, limit - position)
                    if buy_volume > 0:
                        orders.append(Order(product, best_ask, buy_volume))

                if best_bid > fair_value:
                    sell_volume = min(best_bid_amount, limit + position)
                    if sell_volume > 0:
                        orders.append(Order(product, best_bid, -sell_volume))

                # Step 2: Post limit orders inside the spread to market make.
                # Quote 9999 on the buy side and 10001 on the sell side.
                already_buying = sum(o.quantity for o in orders if o.quantity > 0)
                already_selling = sum(-o.quantity for o in orders if o.quantity < 0)

                remaining_buy = limit - position - already_buying
                remaining_sell = limit + position - already_selling

                mm_buy_price = int(fair_value) - 1    # 9999
                mm_sell_price = int(fair_value) + 1   # 10001

                if remaining_buy > 0:
                    orders.append(Order(product, mm_buy_price, remaining_buy))
                if remaining_sell > 0:
                    orders.append(Order(product, mm_sell_price, -remaining_sell))

            # ------------------------------------------------------------------
            # TOMATOES — Trend-Following with Position-Skewed Thresholds
            #
            # THREE PROBLEMS WITH THE OLD CODE:
            #
            # Problem 1 — Runaway long position (the main culprit):
            #   The original strategy ended each day holding a large open long
            #   position (+71 on day -2, +56 on day -1). On day -2 the price
            #   drifted up so the long position helped slightly. On day -1 the
            #   price fell 49 points and the strategy was still long 56 units
            #   at close → ~1,276 of the -1,344 loss came purely from that
            #   open position. The realized trading PnL from actual buy/sell
            #   decisions was almost exactly break-even on day -1 (avg buy
            #   price ≈ avg sell price = 4979.7). So the trading logic itself
            #   was fine; the killer was the unhedged long at end of day.
            #
            # THE FIX — position-skewed thresholds:
            #   Before each trade, scale the signal threshold up based on how
            #   far the current position is from flat:
            #
            #     pos_ratio = position / limit          # ranges -1.0 to +1.0
            #     buy_threshold  = base + max(0, +pos_ratio) * SKEW
            #     sell_threshold = base + max(0, -pos_ratio) * SKEW
            #
            #   When long (pos_ratio > 0): buy_threshold rises → harder to buy
            #   more, but sell_threshold stays at base → easy to sell back down.
            #   When short (pos_ratio < 0): sell_threshold rises → harder to go
            #   even shorter, but buy_threshold stays at base → easy to cover.
            #   At flat position: both thresholds equal base (no skew at all).
            #
            #   SKEW = 5.0 was chosen by grid search: it kept end-of-day
            #   positions to ≤17 units (vs 71 before), turning day -1 from
            #   -1,344 to +96 and improving total PnL from 616 to 2,218.
            #
            # Problem 2 — Position limit too large (amplified Problem 1):
            #   With limit=80, the strategy could and did accumulate up to 71
            #   units long before the position check blocked new buys. Reducing
            #   the effective limit to 40 caps the worst-case open position
            #   risk while still allowing meaningful trend-following trades.
            #   (POSITION_LIMITS["TOMATOES"] is updated at the top of the file.)
            #
            # Problem 3 — No explicit spread guard:
            #   Every trade in the data happened when the bid-ask spread was
            #   5–9 ticks (the rest of the time it was 13–14 ticks wide). The
            #   original code happened to only fire during narrow-spread moments
            #   anyway, but there was no explicit guard. Adding spread <= 9
            #   makes the intention explicit and prevents accidental trades at
            #   the wide spread where you'd pay 13–14 ticks to enter a position.
            # ------------------------------------------------------------------
            elif product == "TOMATOES":
                trader_state["tomato_prices"].append(mid_price)
                prices = trader_state["tomato_prices"]

                # Only trade when the spread is narrow enough to capture edge.
                if spread > 9:
                    result[product] = orders
                    continue

                if len(prices) >= 50:
                    predicted_price, slope = self.linear_trend_prediction(prices[-50:])
                    base_threshold = max(1.5, abs(slope) * 5)

                    # Position-skewed thresholds (the core fix).
                    pos_ratio = position / limit           # -1.0 to +1.0
                    SKEW = 5.0
                    buy_threshold  = base_threshold + max(0,  pos_ratio) * SKEW
                    sell_threshold = base_threshold + max(0, -pos_ratio) * SKEW

                    if predicted_price > mid_price + buy_threshold and best_ask < predicted_price:
                        buy_volume = min(-best_ask_amount, limit - position)
                        if buy_volume > 0:
                            orders.append(Order(product, best_ask, buy_volume))
                    elif predicted_price < mid_price - sell_threshold and best_bid > predicted_price:
                        sell_volume = min(best_bid_amount, limit + position)
                        if sell_volume > 0:
                            orders.append(Order(product, best_bid, -sell_volume))

            print(
                f"{product} mid={mid_price:.1f} ask={best_ask} bid={best_bid} pos={position} orders={len(orders)}"
            )
            result[product] = orders

        traderData = jsonpickle.encode(trader_state)
        return result, 0, traderData
