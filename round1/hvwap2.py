import json
import jsonpickle
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import Any
import math

class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(self.to_json([
            self.compress_state(state, ""),
            self.compress_orders(orders),
            conversions,
            "",
            "",
        ]))

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
        max_item_length = (self.max_log_length - base_length) // 3

        print(self.to_json([
            self.compress_state(state, self.truncate(state.traderData, max_item_length)),
            self.compress_orders(orders),
            conversions,
            self.truncate(trader_data, max_item_length),
            self.truncate(self.logs, max_item_length),
        ]))

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing["symbol"], listing["product"], listing["denomination"]])

        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append([
                    trade.symbol,
                    trade.price,
                    trade.quantity,
                    trade.buyer,
                    trade.seller,
                    trade.timestamp,
                ])

        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sunlight,
                observation.humidity,
            ]

        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if len(value) <= max_length:
            return value

        return value[:max_length - 3] + "..."

logger = Logger()

class HistoricalVWAP:
    def __init__(self, bv=0, sv=0, bpv=0, spv=0):
        self.buy_volume = bv
        self.sell_volume = sv
        self.buy_price_volume = bpv
        self.sell_price_volume = spv

LIMIT = 20

class Trader:
    def calculate_vwap(self, orders, volume, price_volume):
        volume += sum(amount for _, amount in orders.items())
        price_volume += sum(price * amount for price, amount in orders.items())
        if volume == 0:
            return 0
        return (price_volume / volume, volume, price_volume)
    
    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        result = {}
        conversions = 0

        data = dict()

        if state.traderData == '':
            for product in state.order_depths:
                data[product] = HistoricalVWAP()
        else:
            data = jsonpickle.decode(state.traderData)

        for product in state.order_depths:
            position = state.position[product] if product in state.position else 0
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []

            sell_volume = data[product].sell_volume + sum(-amount for _, amount in order_depth.sell_orders.items())
            buy_volume = data[product].buy_volume + sum(amount for _, amount in order_depth.buy_orders.items())

            sell_price_volume = data[product].sell_price_volume + sum(price * -amount for price, amount in order_depth.sell_orders.items())
            buy_price_volume = data[product].buy_price_volume + sum(price * amount for price, amount in order_depth.buy_orders.items())

            buy_vwap = buy_price_volume/buy_volume if buy_volume != 0 else 0
            sell_vwap = sell_price_volume/sell_volume if sell_volume != 0 else 0

            vwap = (sell_vwap + buy_vwap) / 2 if buy_vwap and sell_vwap else max(sell_vwap, buy_vwap)

            # update historical
            data[product] = HistoricalVWAP(buy_volume, sell_volume, buy_price_volume, sell_price_volume)
            
            orders.append(Order(product, math.floor(vwap), LIMIT-position))

            orders.append(Order(product, math.ceil(vwap), -LIMIT-position))

            result[product] = orders

            traderData = jsonpickle.encode(data)

        logger.flush(state, result, conversions, traderData)
        return result, conversions, traderData