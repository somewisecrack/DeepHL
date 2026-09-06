# DeepHL

**DeepHL** is a local webapp for training a virtual-only reinforcement-learning agent on HyperLiquid perps using **only public L2 order-book data**.

Default market: `xyz:SP500`.

The app supports choosing other markets, but trains one active market at a time. Learning state is persisted on disk and survives backend/browser restarts unless the project/data directory is deleted.

![icon](frontend/public/icon.jpeg)

## What makes it more powerful than the phone MVP

- Desktop/server Python backend instead of phone-only Kotlin loop.
- Top-10 L2 features by default, ready for top-20.
- Durable replay store with much larger capacity.
- Per-market replay/checkpoint files.
- Web dashboard with live book, Q-values, PnL, trade ledger, replay size, epsilon, update count.
- Cleaner path to PyTorch Dueling Double DQN; current implementation uses a Numpy Double-Q fallback with the same API.

## Strict data rule

DeepHL v1 uses only:

```text
HyperLiquid public l2Book
bid/ask prices
bid/ask sizes
order counts
virtual position state
```

It does **not** use:

- candles
- technical indicators
- momentum/volume strategy rules
- trades tape
- funding/open interest
- private account data
- real orders

## Architecture

```text
React/Vite frontend
  ↓ REST + Server-Sent Events
FastAPI backend
  ↓
HyperLiquid L2 WebSocket ingestor
  ↓
L2FeatureBuilder → Masked RL policy → VirtualPerpBroker
  ↓
ReplayStore + model checkpoint persistence
```

## Persistence

Data lives under:

```text
data/
  replay/<market>.jsonl
  checkpoints/<market>.npz
  logs/
```

- Replay is appended on every transition.
- Checkpoints save every 100 updates and on stop.
- Replay compacts on stop.
- Reset Learning archives the current replay/checkpoint and starts clean.
- Market-specific learning state is restored on app restart.

This survives app/backend restarts and browser closes. It does not survive deleting the repo/data directory.

## Run locally

### Backend

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn deephl.app:app --reload --host 0.0.0.0 --port 18987
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Open:

```text
http://localhost:5173
```

## Actions

```text
WAIT
ENTER_LONG
ENTER_SHORT
HOLD
EXIT
```

Action masks:

```text
Flat:    WAIT, ENTER_LONG, ENTER_SHORT
Holding: HOLD, EXIT
```

## Reward

Reward is executable virtual equity change across decision ticks:

```text
reward_t = executable_virtual_equity_after_action_t - executable_virtual_equity_after_action_t_minus_1
```

This includes mark-to-market movement while holding, plus execution costs when entering/exiting.

Virtual entries/exits walk the visible L2 book and include taker fees.

## Safety

- No private keys.
- No wallet connection.
- No real order placement.
- No exchange trading endpoints.
- Virtual broker only.

## Roadmap

- Replace Numpy linear Double-Q fallback with PyTorch masked Dueling Double DQN.
- Add target network and n-step returns.
- Add stratified replay and action-balance diagnostics.
- Add baseline policies: always-wait, random-valid, frozen policy.
- Add checkpoint browser and export/import.
- Add top-20 book mode.
- Add crash-safe SQLite trade ledger.
