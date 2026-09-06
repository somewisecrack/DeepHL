"""Offline audit of DeepHL's transition alignment. No network: the WS/cost loaders
are bypassed and books are fed straight into the decision loop body."""
from __future__ import annotations
import sys, asyncio, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deephl.engine import DeepHLEngine
from deephl.models import Action, BookLevel, L2Book

FEE = 0.00045


def book(bid, ask, t, sz=1000.0, levels=10):
    return L2Book("TEST", t,
                  [BookLevel(bid - i * 0.1, sz, 3) for i in range(levels)],
                  [BookLevel(ask + i * 0.1, sz, 3) for i in range(levels)])


def make_engine(tmp):
    e = DeepHLEngine(Path(tmp))
    e.broker.set_costs(cross_fee_rate=FEE, add_fee_rate=0.00015,
                       funding_rate_hourly=0.0, source="hyperliquid_info:test")
    e.agent.epsilon = 0.0
    return e


async def drive(engine, books, forced_actions):
    """Run the decision-loop body once per book, forcing a chosen action."""
    it = iter(forced_actions)
    engine.agent.select = lambda s, m: next(it)
    engine.running = True
    for b in books:
        engine.latest_book = b
        task = asyncio.create_task(engine._decision_loop())
        await asyncio.sleep(1.15)
        engine.running = False
        task.cancel()
        try: await task
        except asyncio.CancelledError: pass
        engine.running = True
    engine.running = False


def run(books, actions):
    with tempfile.TemporaryDirectory() as tmp:
        e = make_engine(tmp)
        asyncio.run(drive(e, books, actions))
        return e, [t for t in e.replay.data]


def test_hold_mark_to_market_reaches_the_replay_buffer():
    """Enter long, then let the book rise while holding: the gain must appear
    as a positive reward on a HOLD transition."""
    books = [book(100.0, 100.1, 1), book(100.0, 100.1, 2),
             book(101.0, 101.1, 3), book(101.0, 101.1, 4)]
    acts = [Action.ENTER_LONG, Action.HOLD, Action.HOLD, Action.HOLD]
    e, ts = run(books, acts)
    holds = [t for t in ts if t.action == Action.HOLD]
    assert any(t.reward > 0 for t in holds), (
        f"HOLD must be credited with mark-to-market; got {[t.reward for t in holds]}")


def test_total_replay_reward_equals_total_equity_change():
    """No reward may be double counted or lost: the sum of all rewards in the
    buffer must equal the change in virtual equity over the run."""
    books = [book(100.0, 100.1, 1), book(101.0, 101.1, 2),
             book(102.0, 102.1, 3), book(102.0, 102.1, 4)]
    acts = [Action.ENTER_LONG, Action.HOLD, Action.EXIT, Action.WAIT]
    e, ts = run(books, acts)
    total_reward = sum(t.reward for t in ts)
    final_equity = e.broker.equity(books[-1])
    assert abs(total_reward - final_equity) < 1e-9, (
        f"reward sum {total_reward} != equity {final_equity} "
        f"(difference {total_reward - final_equity})")


def test_no_transition_trains_an_action_that_is_illegal_in_its_own_state():
    """Every stored (state, action) pair must be one the policy could actually
    have selected: entering is illegal while holding, holding is illegal flat."""
    books = [book(100.0, 100.1, 1), book(100.5, 100.6, 2),
             book(101.0, 101.1, 3), book(101.0, 101.1, 4)]
    acts = [Action.ENTER_LONG, Action.HOLD, Action.EXIT, Action.WAIT]
    e, ts = run(books, acts)
    bad = []
    for t in ts:
        side = t.state[-4]            # position side feature: 0 flat, +1 long, -1 short
        flat = side == 0.0
        legal = {0, 1, 2} if flat else {3, 4}
        if t.action not in legal:
            bad.append((t.action, "flat" if flat else "holding", t.reward))
    assert not bad, f"illegal (state, action) pairs written to replay: {bad}"


def test_state_and_next_state_keep_their_return_features():
    """mid_ret (index 1) and imb_change (index 3) must not be silently zeroed by
    rebuilding features twice from the same book."""
    books = [book(100.0, 100.1, 1), book(101.0, 101.1, 2),
             book(102.0, 102.1, 3), book(103.0, 103.1, 4)]
    acts = [Action.WAIT, Action.WAIT, Action.WAIT, Action.WAIT]
    e, ts = run(books, acts)
    # the book moves every step, so no stored vector should have a zero mid_ret
    zeroed = [(i, t.state[1], t.next_state[1]) for i, t in enumerate(ts)
              if t.state[1] == 0.0 or t.next_state[1] == 0.0]
    assert not zeroed, (
        f"mid_ret feature destroyed by double feature build in {len(zeroed)}/{len(ts)} "
        f"transitions: {zeroed}")
