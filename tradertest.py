from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import jsonpickle

POSITION_LIMITS = {
    "EMERALDS": 80,
    "TOMATOES": 50,
}

class Trader:

    def run(self, state: TradingState):

        # --- Load persisted data ---
        if state.traderData and state.traderData not in ("", "SAMPLE"):
            trader_state = jsonpickle.decode(state.traderData)
        else:
            trader_state = {
                "tomato_prices": [],
                "emerald_prices": []
            }

        result = {}

        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []

            if len(order_depth.sell_orders) == 0 or len(order_depth.buy_orders) == 0:
                result[product] = orders
                continue

            position = state.position.get(product, 0)
            limit = POSITION_LIMITS.get(product, 20)

            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            mid_price = (best_ask + best_bid) / 2

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
            # -------------------------------------------------------
            elif product == "TOMATOES":
                trader_state["tomato_prices"].append(mid_price)
                prices = trader_state["tomato_prices"]
                WINDOW = 100
                SPREAD = 4

                if len(prices) < WINDOW:
                    result[product] = orders
                    continue

                fair_value = sum(prices[-WINDOW:]) / WINDOW

                buy_price  = round(fair_value) - SPREAD
                sell_price = round(fair_value) + SPREAD

                buy_volume  = limit - position
                sell_volume = limit + position

                if buy_volume > 0:
                    orders.append(Order(product, buy_price, buy_volume))
                if sell_volume > 0:
                    orders.append(Order(product, sell_price, -sell_volume))

            result[product] = orders

        traderData = jsonpickle.encode(trader_state)
        conversions = 0
        return result, conversions, traderData