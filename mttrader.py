from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import jsonpickle
import math

POSITION_LIMITS = {"INTARIAN_PEPPER_ROOT": 80, "ASH_COATED_OSMIUM": 80}
OSMIUM_SKEW_FACTOR = 0.04
OSMIUM_BASE_SPREAD = 2
class Trader:
    def run(self, state: TradingState):
        if state.traderData:
            trader_state = jsonpickle.decode(state.traderData)
        else:
            # Initialize history and the 'Global Anchor' you mentioned
            trader_state = {"osmium_prices": [], "global_anchor": None}

        result = {}

        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []
            
            if not order_depth.sell_orders or not order_depth.buy_orders:
                result[product] = orders
                continue

            position = state.position.get(product, 0)
            limit = POSITION_LIMITS.get(product, 80)
            mid_price = (min(order_depth.sell_orders.keys()) + max(order_depth.buy_orders.keys())) / 2
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())

# --- 1. ASH_COATED_OSMIUM (ANCHORED MEAN REVERSION) ---
            if product == "ASH_COATED_OSMIUM":
                FAIR = 10000
                
                # Hint 3 & 4: simulate how much volume sits at each price level
                # then take everything available that gives us edge
                
                # TAKE aggressively when price is wrong (market taking)
                for ask_price, ask_vol in sorted(order_depth.sell_orders.items()):
                    if ask_price < FAIR:           # anything below fair value is free edge
                        buy_vol = min(-ask_vol, limit - position)
                        if buy_vol > 0:
                            orders.append(Order(product, ask_price, buy_vol))
                            position += buy_vol

                for bid_price, bid_vol in sorted(order_depth.buy_orders.items(), reverse=True):
                    if bid_price > FAIR:           # anything above fair value is free edge
                        sell_vol = min(bid_vol, limit + position)
                        if sell_vol > 0:
                            orders.append(Order(product, bid_price, -sell_vol))
                            position -= sell_vol

                # MAKE passively inside the spread for the rest
                skew = -int(round(position * OSMIUM_SKEW_FACTOR))
                buy_price  = min(FAIR - OSMIUM_BASE_SPREAD + skew, best_bid + 1)
                sell_price = max(FAIR + OSMIUM_BASE_SPREAD + skew, best_ask - 1)

                buy_volume  = limit - position
                sell_volume = limit + position
                if buy_volume > 0:
                    orders.append(Order(product, buy_price,  buy_volume))
                if sell_volume > 0:
                    orders.append(Order(product, sell_price, -sell_volume))

            # --- 2. PEPPER ROOT (STAYING WITH SUCCESSFUL TREND-RIDING) --- 
            elif product == "INTARIAN_PEPPER_ROOT":
                if position < limit:
                    asks = sorted(order_depth.sell_orders.items(), key=lambda x: x[0])
                    current_pos = position
                    for price, vol in asks:
                        buy_amount = min(abs(vol), limit - current_pos)
                        if buy_amount > 0:
                            orders.append(Order(product, price, buy_amount))
                            current_pos += buy_amount
                        if current_pos >= limit: break
            result[product] = orders

        return result, 0, jsonpickle.encode(trader_state)