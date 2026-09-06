from __future__ import annotations

import math
from .models import L2Book, Position, Side


class L2FeatureBuilder:
    def __init__(self, depth: int = 10):
        self.depth = depth
        self.last_mid: float | None = None
        self.last_imb: float | None = None
        self.feature_size = 6 + depth * 6 + 4

    def build(self, book: L2Book, position: Position | None, step: int) -> list[float] | None:
        if len(book.bids) < self.depth or len(book.asks) < self.depth or book.mid is None:
            return None
        mid = book.mid
        bids = book.bids[: self.depth]
        asks = book.asks[: self.depth]
        bid_sum = max(sum(x.sz for x in bids), 1e-9)
        ask_sum = max(sum(x.sz for x in asks), 1e-9)
        imb = (bid_sum - ask_sum) / (bid_sum + ask_sum)
        micro = ((book.best_ask or mid) * bid_sum + (book.best_bid or mid) * ask_sum) / (bid_sum + ask_sum)
        mid_ret = math.log(mid / self.last_mid) if self.last_mid else 0.0
        imb_change = imb - self.last_imb if self.last_imb is not None else 0.0
        self.last_mid, self.last_imb = mid, imb

        out = [
            clip(book.spread_bps, -100, 100),
            clip(mid_ret * 10_000, -200, 200),
            imb,
            clip(imb_change, -2, 2),
            clip((micro - mid) / mid * 10_000, -200, 200),
            math.log1p(bid_sum + ask_sum),
        ]
        for b, a in zip(bids, asks):
            out.extend([
                clip((mid - b.px) / mid * 10_000, 0, 500), math.log1p(b.sz), math.log1p(b.n),
                clip((a.px - mid) / mid * 10_000, 0, 500), math.log1p(a.sz), math.log1p(a.n),
            ])
        side = 0.0 if position is None else float(position.side)
        age = 0.0 if position is None else clip((step - position.entry_step) / 600, 0, 10)
        unreal_bps = 0.0
        if position:
            sign = 1 if position.side == Side.LONG else -1
            unreal_bps = clip(sign * (mid - position.entry_px) / position.entry_px * 10_000, -1000, 1000)
        fresh = 1.0
        out.extend([side, age, unreal_bps, fresh])
        return [float(x) for x in out]

    def patch_position(self, state: list[float], position: Position | None, step: int) -> list[float]:
        """Return a copy of an already-built book feature vector with only the
        virtual-position fields changed. This avoids calling build() twice for
        the same book and corrupting one-step temporal features.
        """
        out = list(state)
        side = 0.0 if position is None else float(position.side)
        age = 0.0 if position is None else clip((step - position.entry_step) / 600, 0, 10)
        # Same-book post-action state cannot honestly know mark-to-market from a
        # later book. Keep the already-built book-flow fields intact and patch
        # only side/age here; future observations carry future unrealized move.
        out[-4] = float(side)
        out[-3] = float(age)
        out[-2] = 0.0 if position is None else out[-2]
        out[-1] = 1.0
        return out


def clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))
