from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import jsonpickle
import math

# Fixed position limits defined by the challenge rules
POSITION_LIMITS = {"INTARIAN_PEPPER_ROOT": 80, "ASH_COATED_OSMIUM": 80}

# --- OSMIUM STRATEGY PARAMETERS ---
# SMA Window: 50 ticks provides a smooth 'fair value' that ignores high-frequency noise.
OSMIUM_WINDOW = 50     
# Base Spread: 2 ticks ensures our quotes land INSIDE the market book spread (avg ~16).
# This gives us 'Queue Priority' so we are filled before other bots.
OSMIUM_BASE_SPREAD = 2    
# Skew Factor: Corrects inventory drift. At 80 units, price shifts by ~3 ticks to force fills.
OSMIUM_SKEW_FACTOR = 0.04 

class Trader:
    def run(self, state: TradingState):
        # State Persistence: Store rolling price history across trading ticks
        if state.traderData:
            trader_state = jsonpickle.decode(state.traderData)
        else:
            trader_state = {"osmium_prices": []}

        result = {}

        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []

            # Safety: Skip calculation if the order book is empty
            if not order_depth.sell_orders or not order_depth.buy_orders:
                result[product] = orders
                continue

            pos = state.position.get(product, 0)
            limit = POSITION_LIMITS.get(product, 80)
            
            # Mid-Price calculation serves as our anchor for 'fair value'
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            mid_price = (best_ask + best_bid) / 2

            # =================================================================
            # 1. PEPPER ROOT: Asymmetric Trend-Rider
            # Strategy: Root trends upward ~1000/day. We 'sweep' the asks to 
            # stay long (+80) and capture the massive trend PnL.
            # =================================================================
            if product == "INTARIAN_PEPPER_ROOT":
                if pos < limit:
                    # Sort asks from cheapest to most expensive for efficient filling
                    asks = sorted(order_depth.sell_orders.items(), key=lambda x: x[0])
                    current_pos = pos
                    for price, vol in asks:
                        buy_amount = min(abs(vol), limit - current_pos)
                        if buy_amount > 0:
                            orders.append(Order(product, price, buy_amount))
                            current_pos += buy_amount
                        if current_pos >= limit: break
            
            # =================================================================
            # 2. ASH_COATED_OSMIUM: Symmetric SMA Market Maker
            # Strategy: Price is mean-reverting. We quote a tight spread around
            # a rolling average to collect the spread 'churn'.
            # =================================================================
            elif product == "ASH_COATED_OSMIUM":
                # Maintain price history for the SMA calculation
                trader_state["osmium_prices"].append(mid_price)
                prices = trader_state["osmium_prices"]

                # Use min() to ensure we can trade from Tick 1 (no 'warm-up' blackout)
                actual_window = min(len(prices), OSMIUM_WINDOW)
                fair_value = sum(prices[-actual_window:]) / actual_window

                # Inventory Skew: Nudges buy/sell prices to keep position near 0
                skew = -int(round(pos * OSMIUM_SKEW_FACTOR))

                # Fixed 2-tick spread ensures we land inside the wide book spread.
                # We stay 1 tick ahead of the best bid/ask for queue priority.
                buy_price = min(round(fair_value) - OSMIUM_BASE_SPREAD + skew, best_ask - 1)
                sell_price = max(round(fair_value) + OSMIUM_BASE_SPREAD + skew, best_bid + 1)

                buy_vol = limit - pos
                sell_vol = limit + pos

                if buy_vol >