"""Audit of DeepHL reset semantics. No network."""
from __future__ import annotations
import sys, asyncio, tempfile, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deephl.engine import DeepHLEngine
from deephl.models import Action, BookLevel, L2Book, Transition


def book(bid=100.0, ask=100.1, t=1, sz=1000.0):
    return L2Book("TEST", t, [BookLevel(bid - i * .1, sz, 3) for i in range(10)],
                  [BookLevel(ask + i * .1, sz, 3) for i in range(10)])


def dirty(e):
    """Put the engine into a state where every resettable field is non-default."""
    e.broker.set_costs(cross_fee_rate=0.00045, add_fee_rate=0.00015,
                       funding_rate_hourly=0.0, source="hyperliquid_info:test")
    e.broker.step(Action.ENTER_LONG, book(), 1)
    e.broker.step(Action.EXIT, book(110.0, 110.1), 2)
    e.latest_book = book()
    e.book_updates = 123
    e.last_book_wall_ms = 999
    e.last_decision_book_time_ms = 55
    e.ws_status = "connected"
    e.step = 77
    e.last_equity = 12.5
    e.pending_state = [0.0] * e.features.feature_size
    e.pending_action = 3
    e.pending_done = True
    e.last_q = [1.0, 2.0, 3.0, 4.0, 5.0]
    e.last_result = {"action": "EXIT", "reward": 5.0, "equity": 5.0, "realized": 5.0, "reason": "exit"}
    s = [0.0] * e.features.feature_size
    for _ in range(10):
        e.replay.add(Transition(s, 0, 1.0, s, [True] * 5, False, 1))
    e.agent.updates = 500
    e.agent.epsilon = 0.05
    e.agent.save()


def test_reset_clears_every_documented_field_and_archives_state():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        e = DeepHLEngine(root)
        dirty(e)
        replay_path = root / "replay" / "SP500.jsonl"
        ckpt_path = root / "checkpoints" / "SP500.npz"
        assert replay_path.exists() and ckpt_path.exists()
        before_replay = replay_path.read_text()

        asyncio.run(e.reset_learning())

        # archived, not deleted
        arch_r = list((root / "replay").glob("SP500.archive-*.jsonl"))
        arch_c = list((root / "checkpoints").glob("SP500.archive-*.npz"))
        assert arch_r and arch_c, "reset must archive replay and checkpoint"
        assert arch_r[0].read_text() == before_replay

        # every visible/internal field cleared
        assert e.running is False
        assert e.ws_task is None and e.loop_task is None
        assert e.ws_status == "idle"
        assert e.book_updates == 0
        assert e.last_book_wall_ms == 0
        assert e.last_decision_book_time_ms == 0
        assert e.latest_book is None
        assert e.step == 0
        assert e.last_equity is None
        assert e.pending_state is None and e.pending_action is None
        assert e.pending_done is False
        assert e.last_q == [0.0] * 5
        assert e.broker.trades == []
        assert e.broker.cash_pnl == 0.0
        assert e.broker.position is None
        assert e.broker.unrealized(book()) == 0.0
        assert len(e.replay.data) == 0, "replay must not reload the archived file"
        assert e.agent.updates == 0
        assert e.agent.epsilon == 0.25

        snap = e.snapshot()
        assert snap["replay"] == 0 and snap["updates"] == 0
        assert snap["equity"] == 0 and snap["realizedPnl"] == 0.0
        assert snap["unrealizedPnl"] == 0.0
        assert snap["trades"] == [] and snap["bookUpdates"] == 0
        assert snap["closedTradeStats"]["count"] == 0
        assert snap["rewardStats"]["positive"] == 0


def test_reset_wipes_the_cost_source_so_training_cannot_resume_silently():
    """After reset the broker is fresh, so costs are 'not_loaded' until refetched."""
    with tempfile.TemporaryDirectory() as tmp:
        e = DeepHLEngine(Path(tmp))
        dirty(e)
        asyncio.run(e.reset_learning())
        assert e.broker.costs_loaded is False
        assert e.broker.cost_source == "not_loaded"
        assert e.broker.cross_fee_rate == 0.0


def test_market_switch_repoints_coin_replay_and_checkpoint():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        e = DeepHLEngine(root)
        assert e.market_label == "SP500" and e.coin == "xyz:SP500", "default must be xyz:SP500"
        e.market_label, e.coin = "BTC", "BTC"
        e.replay = type(e.replay)(root / "replay" / "BTC.jsonl")
        e.agent = type(e.agent)(e.features.feature_size, ckpt=root / "checkpoints" / "BTC.npz")
        assert e.replay.path.name == "BTC.jsonl"
        assert e.agent.ckpt.name == "BTC.npz"
        assert e.snapshot()["coin"] == "BTC"
