from datamodel import OrderDepth, TradingState, Order
from typing import List
import jsonpickle
import math

# --- IMPROVED STRATEGY PARAMETERS ---
POSITION_LIMIT = 80

# PEPPER_ROOT: Highly trending, needs reactive tracking
ROOT_ALPHA = 0.2
ROOT_SPREAD = 3 

# OSMIUM: Mean-reverting but noisy, needs stable tracking
OSMIUM_ALPHA = 0.05
OSMIUM_SPREAD = 4

class Trader:

    def run(self, state: TradingState):
        # 1. State Management
        if state.traderData and state.traderData not in ("", "SAMPLE"):
            trader_state = jsonpickle.decode(state.traderData)
        else:
            trader_state = {"emas": {}}

        result = {}

        for product in ["ASH_COATED_OSMIUM", "INTARIAN_PEPPER_ROOT"]:
            order_depth: OrderDepth = state.order_depths.get(product, None)
            if not order_depth:
                continue

            orders: List[Order] = []
            pos = state.position.get(product, 0)
            
            # --- SAFETY CHECK: Ensure the order book is not empty ---
            if not order_depth.sell_orders or not order_depth.buy_orders:
                # If we have an existing EMA, we can still trade against the side that exists
                if product in trader_state["emas"]:
                    fair_value = trader_state["emas"][product]
                else:
                    # No data at all yet, skip this tick
                    continue
            else:
                # 2. Update Fair Value (EMA)
                best_ask = min(order_depth.sell_orders.keys())
                best_bid = max(order_depth.buy_orders.keys())
                mid_price = (best_ask + best_bid) / 2

                if product not in trader_state["emas"]:
                    trader_state["emas"][product] = mid_price
                
                alpha = ROOT_ALPHA if product == "INTARIAN_PEPPER_ROOT" else OSMIUM_ALPHA
                trader_state["emas"][product] = (alpha * mid_price) + ((1 - alpha) * trader_state["emas"][product])
                fair_value = trader_state["emas"][product]

            # 3. Market Taking (Aggressively snatching mispriced orders)
            # Buy from people selling too cheap (relative to our fair value)
            if order_depth.sell_orders:
                sorted_sell_orders = sorted(order_depth.sell_orders.items())
                for price, vol in sorted_sell_orders:
                    if price <= fair_value - 1 and pos < POSITION_LIMIT:
                        buy_vol = min(-vol, POSITION_LIMIT - pos)
                        orders.append(Order(product, price, buy_vol))
                        pos += buy_vol

            # Sell to people buying too expensive
            if order_depth.buy_orders:
                sorted_buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
                for price, vol in sorted_buy_orders:
                    if price >= fair_value + 1 and pos > -POSITION_LIMIT:
                        sell_vol = max(-vol, -POSITION_LIMIT - pos)
                        orders.append(Order(product, price, sell_vol))
                        pos -= abs(sell_vol)

            # 4. Market Making (Passive Quoting with Skew)
            # Calculate inventory skew to push position back to 0
            # If pos is +80, skew is -8 (lowers our prices to discourage buying/encourage selling)
            skew = -int(pos / 10)
            spread = ROOT_SPREAD if product == "INTARIAN_PEPPER_ROOT" else OSMIUM_SPREAD
            
            my_bid = math.floor(fair_value - spread + skew)
            my_ask = math.ceil(fair_value + spread + skew)

            # Pennying: Attempt to be at the top of the book if it's still profitable
            if order_depth.buy_orders:
                best_bid = max(order_depth.buy_orders.keys())
                my_bid = max(my_bid, best_bid + 1)
            
            if order_depth.sell_orders:
                best_ask = min(order_depth.sell_orders.keys())
                my_ask = min(my_ask, best_ask - 1)
            
            # Safety: Ensure we don't buy higher than we sell
            if my_bid >= my_ask:
                my_bid = my_ask - 1

            # 5. Send remaining inventory as limit orders
            if pos < POSITION_LIMIT:
                orders.append(Order(product, int(my_bid), POSITION_LIMIT - pos))
            if pos > -POSITION_LIMIT:
                orders.append(Order(product, int(my_ask), - (POSITION_LIMIT + pos)))

            result[product] = orders

        return result, 0, jsonpickle.encode(trader_state)