from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Optional


class Action(IntEnum):
    WAIT = 0
    ENTER_LONG = 1
    ENTER_SHORT = 2
    HOLD = 3
    EXIT = 4


class Side(IntEnum):
    LONG = 1
    SHORT = -1


@dataclass
class BookLevel:
    px: float
    sz: float
    n: int = 0


@dataclass
class L2Book:
    coin: str
    time_ms: int
    bids: list[BookLevel]
    asks: list[BookLevel]

    @property
    def best_bid(self) -> Optional[float]:
        return self.bids[0].px if self.bids else None

    @property
    def best_ask(self) -> Optional[float]:
        return self.asks[0].px if self.asks else None

    @property
    def mid(self) -> Optional[float]:
        return (self.best_bid + self.best_ask) / 2 if self.best_bid and self.best_ask else None

    @property
    def spread_bps(self) -> float:
        if not self.best_bid or not self.best_ask or not self.mid:
            return 0.0
        return (self.best_ask - self.best_bid) / self.mid * 10_000


@dataclass
class Position:
    side: Side
    entry_px: float
    qty: float
    entry_step: int
    entry_time_ms: int


@dataclass
class Transition:
    state: list[float]
    action: int
    reward: float
    next_state: list[float]
    next_mask: list[bool]
    done: bool
    ts_ms: int


DEFAULT_MARKETS = {
    "SP500": "xyz:SP500",
    "BTC": "BTC",
    "ETH": "ETH",
    "SOL": "SOL",
    "HYPE": "HYPE",
    "XRP": "XRP",
    "DOGE": "DOGE",
    "LINK": "LINK",
    "BNB": "BNB",
    "PAXG": "PAXG",
    "WTI": "xyz:CL",
    "Gold": "xyz:GOLD",
    "Silver": "xyz:SILVER",
    "Brent": "xyz:BRENTOIL",
    "EUR": "xyz:EUR",
    "JPY": "xyz:JPY",
}
