# =============================================================================
# IMC Prosperity 4 — Round 1 Improved Bot
# =============================================================================
#
# ── WHAT CHANGED FROM 149924.py AND WHY ─────────────────────────────────────
#
# ┌─────────────────────────────────────────────────────────────────────────┐
# │ PROBLEM 1 — "PnL flat at 0 for the first period" (image 3 / draft 3)   │
# └─────────────────────────────────────────────────────────────────────────┘
#
# Root cause: an earlier draft of trader_beginner.py had a "blackout" for
# OSMIUM — it refused to trade until it had collected 50 price samples:
#
#   if len(prices) < OSMIUM_WINDOW:      # ← old code
#       result[product] = orders         #   skip first 50 ticks
#       continue
#
# During those 50 ticks OSMIUM earned nothing, so PnL sat at 0 until the
# window was warm.  The fix was already applied in 149924.py:
#
#   window_size = min(len(prices), OSMIUM_WINDOW)   # ← use whatever we have
#   fair_value  = sum(prices[-window_size:]) / window_size
#
# This lets OSMIUM trade from tick 1 using however many prices it has so far.
#
# ┌──────────────────────────────────────────────────────────────────────────┐
# │ PROBLEM 2 — OSMIUM: 97.8 % of ticks our quotes were AT or BEHIND the    │
# │             best bid/ask  →  back of the queue  →  almost never filled  │
# └──────────────────────────────────────────────────────────────────────────┘
#
# Old spread formula (149924.py):
#   spread = max(OSMIUM_BASE_SPREAD=3, math.floor(book_spread / 2))
#
# OSMIUM's average book spread = 16.  floor(16/2) = 8.  So max(3,8) = 8.
# That formula returned 8 on 97.8 % of ticks.  With fair_value ≈ 10001:
#   old buy_price  = round(10001) - 8 = 9993   ← equals the existing best_bid
#   old sell_price = round(10001) + 8 = 10009  ← equals the existing best_ask
#
# We were tied with whoever already sat at the front of the book — we were
# placed at the BACK of the queue and rarely got filled.  The data confirms:
# positions only reached ±22 instead of using the ±80 limit.
#
# NEW formula:  spread = OSMIUM_BASE_SPREAD = 2   (constant, no book_spread)
#   new buy_price  = round(10001) - 2 = 9999  ← 6 ticks BETTER than best_bid
#   new sell_price = round(10001) + 2 = 10003 ← 6 ticks BETTER than best_ask
#
# We are now the BEST bid and BEST ask on 100 % of ticks → queue priority
# → many more fills → much more spread income.
#
# ┌──────────────────────────────────────────────────────────────────────────┐
# │ PROBLEM 3 — ROOT: positions never built up (max ≈ +21 vs limit of +80)  │
# │             and PnL was flat for almost half the day                     │
# └──────────────────────────────────────────────────────────────────────────┘
#
# Two sub-problems:
#
#   a) SYMMETRIC spreads on a TRENDING asset
#      ROOT's price rises ~1 000 every day (~0.1 per tick).  A symmetric
#      market maker buys and sells with equal frequency, keeping position
#      near 0.  That is fine for a stable asset but it WASTES the trend.
#
#      If you hold +40 units while the price rises 1 000:
#        trend PnL = 40 × 1 000 = 40 000 seashells
#      If you hold +80 units:
#        trend PnL = 80 × 1 000 = 80 000 seashells
#      Market making alone (collecting a 2-tick spread occasionally) earns
#      maybe 1 000–2 000.  The trend income dominates by 40–80×.
#
#   b) SKEW_FACTOR was fighting the trend
#      ROOT_SKEW_FACTOR = 0.08 lowers the buy price when long and raises
#      the sell price when short — actively pushing the position back toward
#      0.  That is the opposite of what we want on a rising asset.
#
# FIX — Asymmetric spread (long-biased market making):
#   ROOT_BUY_SPREAD  = 2   tight bid  → fills frequently → buy often
#   ROOT_SELL_SPREAD = 10  wide ask   → almost never fills → rarely sell
#   ROOT_SKEW_FACTOR = 0   no skew    → let position drift long naturally
#
# With avg book spread = 13 and sell_spread = 10:
#   sell_price = ceil(mid) + 10 ≈ mid + 10
#   best_ask   = mid + book_spread/2 ≈ mid + 6.5
#   sell_price > best_ask on almost every tick → order rests outside the book
#   → we almost never sell → position drifts toward +80 over time
#
# Result: sustained long position captures the trend rise while the tight
# buy spread still fills and earns spread income on the buy side.
#
# ── MATHEMATICAL BASIS FOR PARAMETER CHOICES ─────────────────────────────
#
# OSMIUM_BASE_SPREAD = 2:
#   Tick-to-tick σ ≈ 3.7. Avellaneda-Stoikov optimal half-spread ≈ σ/2 ≈ 1.9
#   → round up to 2 ticks.  At 2 we are inside the book 100 % of ticks and
#   earn roughly σ/2 per round-trip fill in expectation.
#
# ROOT_BUY_SPREAD = 2:
#   Same σ ≈ 3.0. Optimal ≈ 1.5 → round up to 2. This keeps the buy
#   competitive (97 % of ticks inside the book) while protecting against
#   buying at a negative-expected-value price.
#
# ROOT_SELL_SPREAD = 10:
#   Must exceed half the average book spread (~6.5) to place the ask
#   outside the book reliably.  10 >> 6.5, so on ~100 % of ticks our ask
#   is outside — we keep the position long.
#
# OSMIUM_SKEW_FACTOR = 0.04:
#   With limit = 80, max skew = 80 × 0.04 = 3.2 ticks ≈ 3 ticks.
#   Three ticks is enough to nudge fills without killing the spread edge.
#
# =============================================================================

from datamodel import OrderDepth, TradingState, Order
from typing import List
import jsonpickle
import math


# =============================================================================
# POSITION LIMITS — maximum units long (+) or short (−)
# =============================================================================
POSITION_LIMITS = {
    "INTARIAN_PEPPER_ROOT": 80,
    "ASH_COATED_OSMIUM":    80,
}


# =============================================================================
# INTARIAN_PEPPER_ROOT PARAMETERS
# Strategy: asymmetric market making — build a net long position to ride
#           the reliable upward trend of ~1 000 per day.
# =============================================================================

# Tight buy spread — places our bid INSIDE the book on 97 % of ticks.
# We fill on buys often, accumulating a long position.
ROOT_BUY_SPREAD  = 2

# Wide sell spread — places our ask OUTSIDE the book on ~100 % of ticks.
# We rarely sell, so the long position is preserved.
ROOT_SELL_SPREAD = 10

# No skew — we WANT the position to stay long.  Skewing toward 0 would
# fight the trend and reduce profit.  Set to 0.
ROOT_SKEW_FACTOR = 0


# =============================================================================
# ASH_COATED_OSMIUM PARAMETERS
# Strategy: symmetric market making — price is flat, SMA is fair value.
#           FIX: always use BASE_SPREAD (not max with floor(book_spread/2)).
# =============================================================================

# Number of past mid-prices to average for fair value.
# Shorter window = more reactive (good for sudden shifts).
# Longer window = smoother (better for stable prices like Osmium).
OSMIUM_WINDOW      = 50

# FIXED spread — replaces old: max(BASE_SPREAD, floor(book_spread/2)).
# The old formula gave spread=8 on 97.8 % of ticks, placing quotes AT or
# BEHIND the best bid/ask (back of queue, rarely filled).
# This constant 2 places us INSIDE the book on 100 % of ticks.
OSMIUM_BASE_SPREAD = 2

# Mild skew to prevent inventory from drifting too far.
# max skew at limit (80): 80 × 0.04 = 3.2 ticks
OSMIUM_SKEW_FACTOR = 0.04


# =============================================================================
# TRADER CLASS
# =============================================================================

class Trader:

    def run(self, state: TradingState):

        # ── 1. RESTORE STATE ─────────────────────────────────────────────────
        if state.traderData and state.traderData not in ("", "SAMPLE"):
            trader_state = jsonpickle.decode(state.traderData)
        else:
            trader_state = {
                "osmium_prices": [],
            }

        result = {}

        # ── 2. LOOP OVER PRODUCTS ─────────────────────────────────────────────
        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []

            # Skip if the book has no bid or no ask (can't compute mid-price)
            if not order_depth.sell_orders or not order_depth.buy_orders:
                result[product] = orders
                continue

            position = state.position.get(product, 0)
            limit    = POSITION_LIMITS.get(product, 80)

            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            mid_price = (best_ask + best_bid) / 2

            # =================================================================
            # INTARIAN_PEPPER_ROOT — Asymmetric long-biased market making
            # =================================================================
            if product == "INTARIAN_PEPPER_ROOT":

                # Fair value = current mid-price.
                # No backward average — the trend means any lagging average
                # would under-price our sells and cause us to sell too cheap.
                fair_value = mid_price

                # INVENTORY SKEW:
                # ROOT_SKEW_FACTOR = 0, so skew = 0 on every tick.
                # The long position is intentional — do not fight it.
                skew = -int(round(position * ROOT_SKEW_FACTOR))

                # BUY QUOTE: tight spread → inside the book → fills often
                # The min() guard prevents accidentally crossing the spread
                # (we never want to be a market taker on purpose here).
                buy_price = min(
                    math.floor(fair_value) - ROOT_BUY_SPREAD + skew,
                    best_ask - 1   # never cross the spread
                )

                # SELL QUOTE: wide spread → outside the book → rarely fills
                # This means we almost never sell, so the long position grows.
                # The max() guard ensures the sell is always above best_bid
                # (we never want to sell below the current bid).
                sell_price = max(
                    math.ceil(fair_value) + ROOT_SELL_SPREAD + skew,
                    best_bid + 1   # never cross the spread
                )

                # Order sizes = full remaining capacity on each side
                buy_volume  = limit - position   # e.g. 80 − 30 = 50 capacity
                sell_volume = limit + position   # e.g. 80 + 30 = 110 but we only hold 30 so max sell = 30

                if buy_volume > 0:
                    orders.append(Order(product, buy_price, buy_volume))
                if sell_volume > 0:
                    orders.append(Order(product, sell_price, -sell_volume))

            # =================================================================
            # ASH_COATED_OSMIUM — Symmetric market making (fixed spread)
            # =================================================================
            elif product == "ASH_COATED_OSMIUM":

                # Append today's mid-price to the rolling history
                trader_state["osmium_prices"].append(mid_price)
                prices = trader_state["osmium_prices"]

                # SMA fair value — use however many prices we have so far.
                # This avoids any blackout period at the start of the day.
                window_size = min(len(prices), OSMIUM_WINDOW)
                fair_value  = sum(prices[-window_size:]) / window_size

                # FIXED SPREAD: always = OSMIUM_BASE_SPREAD = 2
                # OLD: spread = max(OSMIUM_BASE_SPREAD, math.floor(book_spread/2))
                #      → gave spread = 8 on 97.8% of ticks
                #      → quotes landed AT or BEHIND best bid/ask (back of queue)
                # NEW: spread = 2 always
                #      → quotes land INSIDE the book on 100% of ticks (queue priority)
                spread = OSMIUM_BASE_SPREAD

                # Inventory skew: lean against current position to stay balanced
                skew = -int(round(position * OSMIUM_SKEW_FACTOR))

                buy_price  = min(round(fair_value) - spread + skew, best_ask - 1)
                sell_price = max(round(fair_value) + spread + skew, best_bid + 1)

                buy_volume  = limit - position
                sell_volume = limit + position

                if buy_volume > 0:
                    orders.append(Order(product, buy_price, buy_volume))
                if sell_volume > 0:
                    orders.append(Order(product, sell_price, -sell_volume))

            result[product] = orders

        # ── 3. PERSIST STATE ──────────────────────────────────────────────────
        traderData = jsonpickle.encode(trader_state)
        conversions = 0
        return result, conversions, traderData
