from __future__ import annotations

import time
from .models import Action, BookLevel, L2Book, Position, Side


class VirtualPerpBroker:
    def __init__(self, notional_usd: float = 1000.0, taker_fee_rate: float = 0.00045, max_hold_steps: int = 600):
        self.notional_usd = notional_usd
        self.taker_fee_rate = taker_fee_rate
        self.max_hold_steps = max_hold_steps
        self.cash_pnl = 0.0
        self.position: Position | None = None
        self.trades: list[dict] = []

    def mask(self) -> list[bool]:
        return [True, True, True, False, False] if self.position is None else [False, False, False, True, True]

    def equity(self, book: L2Book) -> float:
        return self.cash_pnl + self.unrealized(book)

    def step(self, action: Action, book: L2Book, step: int) -> dict:
        before = self.equity(book)
        forced = self.position is not None and step - self.position.entry_step >= self.max_hold_steps
        effective = Action.EXIT if forced and action == Action.HOLD else action
        reason = "noop"
        realized = 0.0
        now = int(time.time() * 1000)
        if effective == Action.WAIT:
            reason = "wait_flat" if self.position is None else "invalid_wait_holding"
        elif effective == Action.HOLD:
            reason = "hold" if self.position else "invalid_hold_flat"
        elif effective == Action.ENTER_LONG and self.position is None:
            fill = self._walk_notional(book.asks, self.notional_usd)
            if fill:
                px, qty = fill
                self.cash_pnl -= self.notional_usd * self.taker_fee_rate
                self.position = Position(Side.LONG, px, qty, step, now)
                reason = "enter_long"
                self.trades.append({"ts_ms": now, "event": reason, "px": px, "qty": qty, "pnl": 0})
        elif effective == Action.ENTER_SHORT and self.position is None:
            fill = self._walk_notional(book.bids, self.notional_usd)
            if fill:
                px, qty = fill
                self.cash_pnl -= self.notional_usd * self.taker_fee_rate
                self.position = Position(Side.SHORT, px, qty, step, now)
                reason = "enter_short"
                self.trades.append({"ts_ms": now, "event": reason, "px": px, "qty": qty, "pnl": 0})
        elif effective == Action.EXIT and self.position:
            pos = self.position
            px = self._walk_qty(book.bids if pos.side == Side.LONG else book.asks, pos.qty)
            if px:
                gross = (px - pos.entry_px) * pos.qty if pos.side == Side.LONG else (pos.entry_px - px) * pos.qty
                fee = px * pos.qty * self.taker_fee_rate
                realized = gross - fee
                self.cash_pnl += realized
                self.position = None
                reason = "forced_exit_max_hold" if forced else "exit"
                self.trades.append({"ts_ms": now, "event": reason, "px": px, "qty": pos.qty, "pnl": realized})
        after = self.equity(book)
        return {"action": effective.name, "reward": after - before, "equity": after, "realized": realized, "reason": reason}

    def unrealized(self, book: L2Book) -> float:
        if not self.position:
            return 0.0
        px = book.best_bid if self.position.side == Side.LONG else book.best_ask
        if not px:
            return 0.0
        return (px - self.position.entry_px) * self.position.qty if self.position.side == Side.LONG else (self.position.entry_px - px) * self.position.qty

    def _walk_notional(self, levels: list[BookLevel], notional: float) -> tuple[float, float] | None:
        remaining, qty, cost = notional, 0.0, 0.0
        for l in levels:
            take_qty = min(l.sz, remaining / l.px)
            qty += take_qty
            cost += take_qty * l.px
            remaining -= take_qty * l.px
            if remaining <= 1e-6 and qty > 0:
                return cost / qty, qty
        return None

    def _walk_qty(self, levels: list[BookLevel], qty: float) -> float | None:
        remaining, got, cost = qty, 0.0, 0.0
        for l in levels:
            take = min(l.sz, remaining)
            got += take
            cost += take * l.px
            remaining -= take
            if remaining <= 1e-9 and got > 0:
                return cost / got
        return None
