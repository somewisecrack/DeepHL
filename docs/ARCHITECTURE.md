# DeepHL Architecture

DeepHL is a local full-stack RL lab.

## Backend

- FastAPI API server
- HyperLiquid public L2 WebSocket ingestor
- L2-only feature builder
- virtual perp broker
- masked Double-Q learner
- durable replay/checkpoints

## Frontend

- React/Vite dashboard
- live state via Server-Sent Events
- market selector
- order book display
- policy monitor
- virtual trade ledger

## Persistence contract

Per-market files:

```text
data/replay/SP500.jsonl
data/checkpoints/SP500.npz
```

Replay appends immediately. Checkpoints are atomic file replacements.
