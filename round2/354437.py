from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict
import jsonpickle
import math

POSITION_LIMITS = {
    "INTARIAN_PEPPER_ROOT": 80,
    "ASH_COATED_OSMIUM": 80,
}

ROOT_FAST_ALPHA = 0.18
ROOT_SLOW_ALPHA = 0.04
# Osmium: slow EMA + 10k anchor, quote then take. Tuned on ROUND_2 CSVs (rust_backtester, DYLD to Anaconda lib on macOS).
# Large random/grid sweeps topped out ~61.5k Osmium sum (3 days) — still below an informal 70k goal on this harness.
# half_spread=1 beat fixed 2/3; MM_SIZE >= ~16 plateaus (extra size rarely adds PnL here). Dual price tiers hurt in tests.
# For a stricter live take rule, try OSMIUM_TAKE_EDGE=1.
OSMIUM_ALPHA = 0.0804
OSMIUM_FAIR_ANCHOR_WEIGHT = 0.424
OSMIUM_FAIR_EMA_WEIGHT = 0.576
OSMIUM_SHAVE_MULT = 6
OSMIUM_MM_SIZE = 16
OSMIUM_TAKE_EDGE = 0
OSMIUM_MICRO_W = 0.035

# Round 2 Market Access Fee (MAF) bid.
# If you're in the top 50% of bids, you'll get +25% extra quotes but pay this fee.
MAF_BID = 3888


def _microprice_at_touch(depth: OrderDepth, best_bid: int, best_ask: int, mid: float) -> float:
    vb = abs(depth.buy_orders.get(best_bid, 0))
    va = abs(depth.sell_orders.get(best_ask, 0))
    denom = vb + va
    if denom <= 0:
        return mid
    return (best_bid * va + best_ask * vb) / denom


class Trader:
    def bid(self) -> int:
        return int(MAF_BID)

    def run(self, state: TradingState):
        if state.traderData:
            data = jsonpickle.decode(state.traderData)
        else:
            data = {
                "root_fast": None,
                "root_slow": None,
                "root_last_mid": None,
                "root_hold_mode": False,
                "osmium_ema": 10000.0,
            }

        result: Dict[str, List[Order]] = {}

        for product, depth in state.order_depths.items():
            orders: List[Order] = []
            pos = state.position.get(product, 0)
            limit = POSITION_LIMITS[product]

            if not depth.buy_orders or not depth.sell_orders:
                result[product] = orders
                continue

            best_bid = max(depth.buy_orders.keys())
            best_ask = min(depth.sell_orders.keys())
            mid = (best_bid + best_ask) / 2
            spread = best_ask - best_bid
            # Round 2 spreads are often wide; scale "edge" thresholds with spread.
            half_spread = max(1, int(round(spread / 2)))

            if product == "INTARIAN_PEPPER_ROOT":
                # Original (high-PnL) trend-following long bias.
                fast = data.get("root_fast")
                slow = data.get("root_slow")
                last_mid = data.get("root_last_mid")
                hold_mode = data.get("root_hold_mode", False)

                if fast is None:
                    fast = mid
                    slow = mid
                    last_mid = mid

                fast = ROOT_FAST_ALPHA * mid + (1 - ROOT_FAST_ALPHA) * fast
                slow = ROOT_SLOW_ALPHA * mid + (1 - ROOT_SLOW_ALPHA) * slow
                slope = fast - slow
                momentum = mid - last_mid
                fair = 0.65 * fast + 0.35 * slow

                # Once the trend is established, behave more like the successful hold-to-80 logic.
                if slope > 2 or momentum > 1:
                    hold_mode = True
                if slope < -4 and momentum < -2:
                    hold_mode = False

                current = pos

                # Step 1: Take asks, but not blindly. If trend is up, allow paying slightly above fair.
                if hold_mode and current < limit:
                    buy_cap_price = fair + 2
                    for ask, vol in sorted(depth.sell_orders.items()):
                        if current >= limit:
                            break
                        if ask <= buy_cap_price or ask <= best_ask:
                            qty = min(-vol, limit - current)
                            if qty > 0:
                                orders.append(Order(product, ask, qty))
                                current += qty
                        else:
                            break
                else:
                    target = 40 if slope >= 0 else 20
                    buy_cap_price = fair + 1
                    for ask, vol in sorted(depth.sell_orders.items()):
                        if current >= target:
                            break
                        if ask <= buy_cap_price:
                            qty = min(-vol, target - current)
                            if qty > 0:
                                orders.append(Order(product, ask, qty))
                                current += qty
                        else:
                            break

                # Step 2: Passive bid if we still want more and quote can look calm.
                desired = limit if hold_mode else (40 if slope >= 0 else 20)
                if current < desired:
                    bid_px = min(best_bid + 1, math.floor(fair))
                    bid_px = min(bid_px, best_ask - 1)
                    qty = min(desired - current, 20)
                    if qty > 0 and bid_px > 0:
                        orders.append(Order(product, int(bid_px), int(qty)))

                # Step 3: Only reduce if the trend weakens a lot and there is a rich bid.
                if current > 0 and (slope < -2 or momentum < -2):
                    for bid, vol in sorted(depth.buy_orders.items(), reverse=True):
                        if bid >= fair + 2:
                            qty = min(vol, current)
                            if qty > 0:
                                orders.append(Order(product, bid, -qty))
                                current -= qty
                        else:
                            break

                data["root_fast"] = fast
                data["root_slow"] = slow
                data["root_last_mid"] = mid
                data["root_hold_mode"] = hold_mode

            elif product == "ASH_COATED_OSMIUM":
                # Teammate-style: slow EMA fair, linear inventory shave, quote then opportunistic take.
                micro = _microprice_at_touch(depth, best_bid, best_ask, mid)
                ref = (1.0 - OSMIUM_MICRO_W) * mid + OSMIUM_MICRO_W * micro
                ema = data.get("osmium_ema", 10000.0)
                ema = OSMIUM_ALPHA * ref + (1 - OSMIUM_ALPHA) * ema
                fair = OSMIUM_FAIR_ANCHOR_WEIGHT * 10000.0 + OSMIUM_FAIR_EMA_WEIGHT * ema

                current = pos
                load_factor = current / limit
                shave = int(round(load_factor * OSMIUM_SHAVE_MULT))
                # Quote one tick closer than old default (effective half_spread 2); improves backtest Osmium here.
                half_spread = 1
                my_bid = min(best_bid + 1, math.floor(fair - half_spread) - shave)
                my_ask = max(best_ask - 1, math.ceil(fair + half_spread) - shave)

                if my_bid >= my_ask:
                    my_bid = my_ask - 1

                buy_cap = max(0, limit - current)
                sell_cap = max(0, limit + current)
                buy_size = min(OSMIUM_MM_SIZE, buy_cap)
                sell_size = min(OSMIUM_MM_SIZE, sell_cap)

                if buy_size > 0 and my_bid > 0:
                    orders.append(Order(product, int(my_bid), int(buy_size)))
                if sell_size > 0 and my_ask > 0:
                    orders.append(Order(product, int(my_ask), -int(sell_size)))

                buy_room = max(0, limit - current)
                sell_room = max(0, limit + current)
                for ask, vol in sorted(depth.sell_orders.items()):
                    if buy_room <= 0:
                        break
                    if ask <= fair - OSMIUM_TAKE_EDGE:
                        qty = min(-vol, buy_room)
                        if qty > 0:
                            orders.append(Order(product, ask, qty))
                            current += qty
                            buy_room -= qty
                    else:
                        break

                for bid, vol in sorted(depth.buy_orders.items(), reverse=True):
                    if sell_room <= 0:
                        break
                    if bid >= fair + OSMIUM_TAKE_EDGE:
                        qty = min(vol, sell_room)
                        if qty > 0:
                            orders.append(Order(product, bid, -qty))
                            current -= qty
                            sell_room -= qty
                    else:
                        break

                data["osmium_ema"] = ema

            result[product] = orders

        return result, 0, jsonpickle.encode(data)