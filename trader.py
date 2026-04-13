from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import string
import numpy as np
import jsonpickle

POSITION_LIMITS = {
    "EMERALDS": 20,
    "TOMATOES": 20,
}
class Trader:

    def bid(self):
        return 15
    
    def run(self, state: TradingState):
        """Only method required. It takes all buy and sell orders for all
        symbols as an input, and outputs a list of orders to be sent."""

        # --- Load persisted data ---
        if state.traderData and state.traderData != "SAMPLE":
            trader_state = jsonpickle.decode(state.traderData)
        else:
            trader_state = {
                "emerald_prices": [],
                "tomato_prices": [],
                "tomato_z": 0
            }

        print("traderData: " + state.traderData)
        print("Observations: " + str(state.observations))

        # Orders to be placed on exchange matching engine
        result = {}
        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []
            position = state.position.get(product, 0)
            limit = POSITION_LIMITS.get(product, 20)

            if len(order_depth.sell_orders) == 0 or len(order_depth.buy_orders) == 0:
                result[product] = orders
                continue

            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            mid_price = (best_ask + best_bid) / 2
            
            if product == 'EMERALDS':

                acceptable_price = 10000
                #-ve in data model
                best_ask_amount = order_depth.sell_orders[best_ask] 
                #+ve in data model
                best_bid_amount = order_depth.buy_orders[best_bid]   

                #Buy if volume is selling below fair value
                if best_ask < acceptable_price:
                    buy_volume = min(-best_ask_amount, limit-position)
                    if buy_volume > 0:
                        print("BUY", buy_volume,"x", limit - position)
                        orders.append(Order(product,best_ask,buy_volume))

                if best_bid > acceptable_price:
                    sell_volume = min(best_bid_amount, limit + position)
                    if sell_volume > 0:
                        print("SELL", sell_volume,"x",best_bid)
                        orders.append(Order(product,best_bid,-sell_volume))

            elif product == "TOMATOES":
                trader_state["tomato_prices"].append(mid_price)
                WINDOW = 200
                Z_ENTRY = 1.5      # enter when price is this many std devs from trend
                Z_EXIT = 0.3       # exit when price returns this close to trend

                prices = trader_state["tomato_prices"][-WINDOW:]

                best_ask_amount = order_depth.sell_orders[best_ask]
                best_bid_amount = order_depth.buy_orders[best_bid]

                if len(prices) < 40:
                    result[product] = orders
                    continue

                x = np.arange(len(prices))
                slope, intercept = np.polyfit(x, prices, 1)
                predicted_price = slope * (len(prices) - 1) + intercept
                
                # Residuals over the window — for std dev calculation
                predicted_all = slope * x + intercept
                residuals = np.array(prices) - predicted_all
                residual_std = np.std(residuals)
                
                current_residual = mid_price - predicted_price
                z_score = current_residual / residual_std if residual_std > 0 else 0

                # Store z_score so we can check exit condition
                trader_state["tomato_z"] = z_score

                # Direction-neutral: trade based on z_score sign, not assumed trend direction
                if z_score < -Z_ENTRY:  # price well BELOW trend — BUY expecting reversion up
                    buy_volume = min(-best_ask_amount, limit - position)
                    if buy_volume > 0:
                        print("BUY", buy_volume, "x", best_ask)
                        orders.append(Order(product, best_ask, buy_volume))

                elif z_score > Z_ENTRY:  # price well ABOVE trend — SELL expecting reversion down
                    sell_volume = min(best_bid_amount, limit + position)
                    if sell_volume > 0:
                        print("SELL", sell_volume, "x", best_bid)
                        orders.append(Order(product, best_bid, -sell_volume))
                        # print("Acceptable price : " + str(acceptable_price))
                        # print("Buy Order depth : " + str(len(order_depth.buy_orders)) + ", Sell order depth : " + str(len(order_depth.sell_orders)))
                
                        result[product] = orders
    
        # String value holding Trader state data required. 
        # It will be delivered as TradingState.traderData on next execution.
        traderData = "SAMPLE" 
        
        #Persists state for next tick
        traderData = jsonpickle.encode(trader_state)
        # Sample conversion request. Check more details below. 
        conversions = 1
        return result, conversions, traderData
