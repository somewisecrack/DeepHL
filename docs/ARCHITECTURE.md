# DeepHL Architecture

DeepHL is a local full-stack RL lab for **virtual-only HyperLiquid L2 order-book RL**.

Status: correctness-tested MVP. It is ready for virtual experimentation, not for claims of profitability.

## Backend

- FastAPI API server
- HyperLiquid public L2 WebSocket ingestor
- HyperLiquid public `/info` cost/funding loader
- L2-only feature builder
- virtual perp broker
- masked Double-Q learner
- durable replay/checkpoints
- deterministic audit tests under `backend/tests/`

## Frontend

- React/Vite dashboard
- live state via Server-Sent Events
- market selector
- order book display
- policy monitor
- virtual trade ledger with gross/fee/net fields
- L2 heartbeat/update count
- reward and closed-trade diagnostics
- actual fee/funding source display

## Runtime contract

- Default market label is `SP500`; actual HyperLiquid coin is `xyz:SP500`.
- Market switch changes WebSocket coin, cost/funding lookup coin/dex, replay path, checkpoint path, and displayed coin.
- The decision loop acts only once per **fresh exchange L2 timestamp**.
- Duplicate/stale book timestamps do not produce training transitions.
- Reset hard-stops training, cancels WebSocket/decision tasks, archives active replay/checkpoint, and clears runtime counters, PnL, trades, latest book, Q-values, and replay.

## Reward/transition contract

Replay rows must be valid masked RL transitions:

```text
(state_t, action_t, reward_{t+1}, next_state_{t+1}, next_mask_{t+1}, done)
```

Key points:

- Entry/exit immediate execution cost is assigned to the selected entry/exit action.
- Between-book mark-to-market movement is assigned to a legal interval action: `HOLD` while positioned, `WAIT` while flat.
- No replay row should train an action that was illegal in its stored state.
- Feature generation avoids rebuilding temporal features twice on the same book; post-action state patches only virtual-position fields.

## Cost contract

- Fees are loaded from HyperLiquid public `/info` via `type: userFees`.
- Market funding is loaded from `/info` via `type: metaAndAssetCtxs`.
- The app refuses to train if selected-market costs cannot be resolved.
- No private key, wallet signing, or real trading endpoint is used.
- User-specific fee tiers are not assumed.

## Persistence contract

Per-market files:

```text
data/replay/<market>.jsonl
data/checkpoints/<market>.npz
```

Replay appends immediately. Checkpoints are atomic file replacements. Active files are archived on reset.

## Test contract

Run from `backend/`:

```bash
.venv/bin/python -m pytest tests -q
```

Current expected result: all audit tests pass.
