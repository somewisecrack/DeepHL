"""Deterministic audit tests for the DeepHL virtual broker + RL reward accounting.

These are written as an *independent* check of behaviour: they assert what the
documented contract says should happen, not what the current code happens to do.
Run with:  cd backend && .venv/bin/python -m pytest tests -q
"""
from __future__ import annotations
import sys, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deephl.broker import VirtualPerpBroker
from deephl.models import Action, BookLevel, L2Book, Side

FEE = 0.00045  # HyperLiquid public base cross rate, verified live from /info userFees


def book(bid=100.0, ask=100.1, sz=1000.0, t=0, levels=10):
    """Flat synthetic book with `levels` rungs of `sz` on each side."""
    bids = [BookLevel(bid - i * 0.1, sz, 3) for i in range(levels)]
    asks = [BookLevel(ask + i * 0.1, sz, 3) for i in range(levels)]
    return L2Book("TEST", t, bids, asks)


def broker(fee=FEE, funding=0.0, notional=1000.0):
    b = VirtualPerpBroker(notional_usd=notional)
    b.set_costs(cross_fee_rate=fee, add_fee_rate=0.00015,
                funding_rate_hourly=funding, source="hyperliquid_info:test")
    return b


# ---------------------------------------------------------------- fill routing
def test_enter_long_walks_asks_and_enter_short_walks_bids():
    b = broker()
    b.step(Action.ENTER_LONG, book(), 1)
    assert b.position.side == Side.LONG
    assert math.isclose(b.position.entry_px, 100.1), "long must lift the ask"

    b2 = broker()
    b2.step(Action.ENTER_SHORT, book(), 1)
    assert math.isclose(b2.position.entry_px, 100.0), "short must hit the bid"


def test_exit_long_walks_bids_and_exit_short_walks_asks():
    b = broker(fee=0.0)
    b.step(Action.ENTER_LONG, book(), 1)
    b.step(Action.EXIT, book(), 2)
    assert b.trades[-1]["px"] == 100.0, "exiting a long must hit the bid"

    b2 = broker(fee=0.0)
    b2.step(Action.ENTER_SHORT, book(), 1)
    b2.step(Action.EXIT, book(), 2)
    assert b2.trades[-1]["px"] == 100.1, "exiting a short must lift the ask"


def test_partial_liquidity_refuses_the_fill():
    """A book that cannot absorb the full notional must not produce a fill."""
    thin = L2Book("TEST", 0, [BookLevel(100.0, 0.5, 1)], [BookLevel(100.1, 0.5, 1)])
    b = broker()
    r = b.step(Action.ENTER_LONG, thin, 1)
    assert b.position is None
    assert r["reason"] == "noop", "silent no-fill is reported as a generic noop"


def test_walk_crosses_multiple_levels_when_top_is_thin():
    thin_top = L2Book("TEST", 0,
                      [BookLevel(100.0, 1.0, 1), BookLevel(99.0, 100.0, 5)],
                      [BookLevel(100.1, 1.0, 1), BookLevel(101.0, 100.0, 5)])
    b = broker(fee=0.0)
    b.step(Action.ENTER_LONG, thin_top, 1)
    # 1 unit @100.1 = 100.1, remaining 899.9 @101.0 = 8.9099 units
    assert b.position.entry_px > 100.1, "VWAP must be worse than top of book"
    assert math.isclose(b.position.entry_px * b.position.qty, 1000.0, rel_tol=1e-9)


# ---------------------------------------------------------------- trade ledger
def test_profitable_long_ledger_is_exact():
    b = broker()
    b.step(Action.ENTER_LONG, book(bid=100.0, ask=100.1), 1)
    qty = b.position.qty                       # 1000/100.1
    b.step(Action.EXIT, book(bid=110.0, ask=110.1), 2)
    entry, exit_ = b.trades
    assert entry["pnl"] == -entry["fee"], "entry row must show the fee as a negative cost"
    assert math.isclose(entry["fee"], 1000.0 * FEE)
    gross = (110.0 - 100.1) * qty
    exit_fee = 110.0 * qty * FEE
    assert math.isclose(exit_["gross"], gross)
    assert math.isclose(exit_["fee"], exit_fee)
    assert math.isclose(exit_["pnl"], gross - exit_fee)
    assert math.isclose(b.cash_pnl, entry["pnl"] + exit_["pnl"]), "cash == sum of ledger rows"
    assert b.cash_pnl > 0


def test_losing_long_ledger_is_exact():
    b = broker()
    b.step(Action.ENTER_LONG, book(bid=100.0, ask=100.1), 1)
    qty = b.position.qty
    b.step(Action.EXIT, book(bid=99.0, ask=99.1), 2)
    gross = (99.0 - 100.1) * qty
    assert math.isclose(b.trades[-1]["gross"], gross)
    assert b.cash_pnl < 0
    assert math.isclose(b.cash_pnl, sum(t["pnl"] for t in b.trades))


def test_profitable_short_ledger_is_exact():
    b = broker()
    b.step(Action.ENTER_SHORT, book(bid=100.0, ask=100.1), 1)
    qty = b.position.qty                        # 1000/100.0
    b.step(Action.EXIT, book(bid=89.9, ask=90.0), 2)
    gross = (100.0 - 90.0) * qty
    assert math.isclose(b.trades[-1]["gross"], gross)
    assert b.cash_pnl > 0
    assert math.isclose(b.cash_pnl, sum(t["pnl"] for t in b.trades))


def test_losing_short_ledger_is_exact():
    b = broker()
    b.step(Action.ENTER_SHORT, book(bid=100.0, ask=100.1), 1)
    qty = b.position.qty
    b.step(Action.EXIT, book(bid=110.0, ask=110.1), 2)
    assert math.isclose(b.trades[-1]["gross"], (100.0 - 110.1) * qty)
    assert b.cash_pnl < 0
    assert math.isclose(b.cash_pnl, sum(t["pnl"] for t in b.trades))


# ---------------------------------------------------------------- fees
def test_round_trip_with_zero_spread_costs_exactly_two_fees():
    b = broker()
    flat = book(bid=100.0, ask=100.0)
    b.step(Action.ENTER_LONG, flat, 1)
    b.step(Action.EXIT, flat, 2)
    assert math.isclose(b.cash_pnl, -2 * 1000.0 * FEE, rel_tol=1e-9)


def test_zero_fee_zero_spread_round_trip_is_flat():
    b = broker(fee=0.0)
    flat = book(bid=100.0, ask=100.0)
    b.step(Action.ENTER_LONG, flat, 1)
    b.step(Action.EXIT, flat, 2)
    assert math.isclose(b.cash_pnl, 0.0, abs_tol=1e-12)


# ---------------------------------------------------------------- hold / MTM
def test_hold_step_reward_ignores_mark_to_market_within_a_single_book():
    """broker.step() alone cannot credit MTM: before/after use the same book."""
    b = broker(fee=0.0)
    b.step(Action.ENTER_LONG, book(bid=100.0, ask=100.0), 1)
    r = b.step(Action.HOLD, book(bid=110.0, ask=110.0), 2)
    assert r["reward"] == 0.0, "MTM must be supplied by the engine, not broker.step"


def test_hold_mark_to_market_positive_via_equity_delta():
    b = broker(fee=0.0)
    b.step(Action.ENTER_LONG, book(bid=100.0, ask=100.0), 1)
    qty = b.position.qty
    e0 = b.equity(book(bid=100.0, ask=100.0))
    e1 = b.equity(book(bid=101.0, ask=101.0))
    assert math.isclose(e1 - e0, 1.0 * qty), "rising book must credit a long"


def test_hold_mark_to_market_negative_via_equity_delta():
    b = broker(fee=0.0)
    b.step(Action.ENTER_SHORT, book(bid=100.0, ask=100.0), 1)
    qty = b.position.qty
    e0 = b.equity(book(bid=100.0, ask=100.0))
    e1 = b.equity(book(bid=101.0, ask=101.0))
    assert math.isclose(e1 - e0, -1.0 * qty), "rising book must debit a short"


def test_unrealized_uses_the_executable_side():
    b = broker(fee=0.0)
    bk = book(bid=100.0, ask=100.1)
    b.step(Action.ENTER_LONG, bk, 1)          # entered at ask 100.1
    # marking at best bid 100.0 => immediately negative by the spread
    assert b.unrealized(bk) < 0
    assert math.isclose(b.unrealized(bk), (100.0 - 100.1) * b.position.qty)


# ---------------------------------------------------------------- funding
def test_positive_funding_debits_longs_and_credits_shorts():
    for side, sign in ((Action.ENTER_LONG, -1), (Action.ENTER_SHORT, +1)):
        b = broker(fee=0.0, funding=0.0000125)   # HL base hourly rate
        b.step(side, book(bid=100.0, ask=100.0), 1)
        cash_after_entry = b.cash_pnl
        b.last_funding_ms -= 3_600_000           # pretend one hour elapsed
        b.step(Action.HOLD, book(bid=100.0, ask=100.0), 2)
        delta = b.cash_pnl - cash_after_entry
        expected = sign * 100.0 * b.position.qty * 0.0000125
        assert math.isclose(delta, expected, rel_tol=1e-9), (
            f"positive funding must be paid by longs, received by shorts ({side})")


def test_funding_is_not_charged_while_flat():
    b = broker(fee=0.0, funding=0.01)
    b.step(Action.WAIT, book(), 1)
    b.step(Action.WAIT, book(), 2)
    assert b.cash_pnl == 0.0


# ---------------------------------------------------------------- masking
def test_mask_matches_position_state():
    b = broker()
    assert b.mask() == [True, True, True, False, False]
    b.step(Action.ENTER_LONG, book(), 1)
    assert b.mask() == [False, False, False, True, True]


def test_forced_exit_after_max_hold():
    b = broker(fee=0.0)
    b.step(Action.ENTER_LONG, book(), 1)
    r = b.step(Action.HOLD, book(), 1 + b.max_hold_steps)
    assert r["reason"] == "forced_exit_max_hold"
    assert b.position is None


# ---------------------------------------------------------------- cost gating
def test_broker_refuses_to_claim_costs_loaded_without_hyperliquid_source():
    b = VirtualPerpBroker()
    assert b.costs_loaded is False
    b.set_costs(cross_fee_rate=0.0, add_fee_rate=0.0, funding_rate_hourly=0.0, source="default")
    assert b.costs_loaded is False, "non-HyperLiquid source must not count as loaded"
    b.set_costs(cross_fee_rate=FEE, add_fee_rate=0.0, funding_rate_hourly=0.0,
                source="hyperliquid_info:userFees")
    assert b.costs_loaded is True
