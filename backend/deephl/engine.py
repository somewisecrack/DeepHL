from __future__ import annotations

import asyncio, json, time, urllib.request
from pathlib import Path
import websockets
from .models import Action, BookLevel, DEFAULT_MARKETS, L2Book, Transition
from .features import L2FeatureBuilder
from .broker import VirtualPerpBroker
from .rl import DuelingDoubleDQN, ReplayStore


class DeepHLEngine:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.market_label = "SP500"
        self.coin = DEFAULT_MARKETS[self.market_label]
        self.depth = 10
        self.features = L2FeatureBuilder(self.depth)
        self.broker = VirtualPerpBroker()
        self.replay = ReplayStore(data_dir / "replay" / f"{self.market_label}.jsonl")
        self.agent = DuelingDoubleDQN(self.features.feature_size, ckpt=data_dir / "checkpoints" / f"{self.market_label}.npz")
        self.latest_book: L2Book | None = None
        self.book_updates = 0
        self.last_book_wall_ms = 0
        self.last_decision_book_time_ms = 0
        self.ws_status = "idle"
        self.running = False
        self.ws_task: asyncio.Task | None = None
        self.loop_task: asyncio.Task | None = None
        self.step = 0
        self.last_equity: float | None = None
        self.last_result = {"action":"WAIT","reward":0,"equity":0,"realized":0,"reason":"idle"}
        self.last_q = [0.0] * 5
        self.subscribers: set[asyncio.Queue] = set()

    def _load_hyperliquid_costs_sync(self) -> dict:
        def post(payload: dict):
            req = urllib.request.Request(
                "https://api.hyperliquid.xyz/info",
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                return json.loads(resp.read().decode())

        # Public endpoint. Without private keys/user auth, the only honest default is
        # HyperLiquid's public base user fee schedule, not an invented config value.
        fees = post({"type": "userFees", "user": "0x0000000000000000000000000000000000000000"})
        dex = self.coin.split(":", 1)[0] if ":" in self.coin else ""
        meta_payload = {"type": "metaAndAssetCtxs", **({"dex": dex} if dex else {})}
        meta, ctxs = post(meta_payload)
        funding = 0.0
        deployer_scale = 1.0
        for asset, ctx in zip(meta.get("universe", []), ctxs):
            if asset.get("name") == self.coin or asset.get("name") == self.coin.split(":")[-1]:
                funding = float(ctx.get("funding") or 0.0)
                deployer_scale = float(asset.get("deployerFeeScale") or 1.0)
                break
        base_cross = float(fees.get("userCrossRate") or fees.get("feeSchedule", {}).get("cross") or 0.0)
        base_add = float(fees.get("userAddRate") or fees.get("feeSchedule", {}).get("add") or 0.0)
        return {
            "cross_fee_rate": base_cross * deployer_scale,
            "add_fee_rate": base_add * deployer_scale,
            "funding_rate_hourly": funding,
            "source": f"hyperliquid_info:userFees+metaAndAssetCtxs coin={self.coin} deployerFeeScale={deployer_scale}",
        }

    async def refresh_costs(self):
        costs = await asyncio.to_thread(self._load_hyperliquid_costs_sync)
        self.broker.set_costs(**costs)
        await self._publish()

    async def start(self):
        if self.running: return
        await self.refresh_costs()
        if not self.broker.costs_loaded:
            raise RuntimeError("HyperLiquid costs were not loaded; refusing to train with synthetic costs")
        self.running = True
        self.ws_task = asyncio.create_task(self._ws_loop())
        self.loop_task = asyncio.create_task(self._decision_loop())

    async def stop(self):
        self.running = False
        for task in [self.ws_task, self.loop_task]:
            if task: task.cancel()
        self.agent.save()
        self.replay.compact()
        await self._publish()

    async def reset_learning(self):
        was = self.running
        if was:
            await self.stop()
        ts = int(time.time())
        for path in [self.data_dir / "replay" / f"{self.market_label}.jsonl", self.data_dir / "checkpoints" / f"{self.market_label}.npz"]:
            if path.exists():
                path.rename(path.with_name(f"{path.stem}.archive-{ts}{path.suffix}"))
        self.features = L2FeatureBuilder(self.depth)
        self.broker = VirtualPerpBroker()
        await self.refresh_costs()
        self.replay = ReplayStore(self.data_dir / "replay" / f"{self.market_label}.jsonl")
        self.agent = DuelingDoubleDQN(self.features.feature_size, ckpt=self.data_dir / "checkpoints" / f"{self.market_label}.npz")
        self.last_equity = None
        self.last_decision_book_time_ms = 0
        self.last_result = {"action":"WAIT","reward":0,"equity":0,"realized":0,"reason":"reset_learning"}
        if was:
            await self.start()
        await self._publish()

    async def set_market(self, label: str):
        if label not in DEFAULT_MARKETS: raise ValueError("unknown market")
        was = self.running
        if was: await self.stop()
        self.market_label, self.coin = label, DEFAULT_MARKETS[label]
        self.features = L2FeatureBuilder(self.depth)
        self.broker = VirtualPerpBroker()
        await self.refresh_costs()
        self.replay = ReplayStore(self.data_dir / "replay" / f"{label}.jsonl")
        self.agent = DuelingDoubleDQN(self.features.feature_size, ckpt=self.data_dir / "checkpoints" / f"{label}.npz")
        self.latest_book = None
        self.book_updates = 0
        self.last_book_wall_ms = 0
        self.last_decision_book_time_ms = 0
        self.ws_status = "idle"
        self.last_equity = None
        if was: await self.start()
        await self._publish()

    async def _ws_loop(self):
        while self.running:
            try:
                self.ws_status = "connecting"
                await self._publish()
                async with websockets.connect("wss://api.hyperliquid.xyz/ws", ping_interval=20) as ws:
                    self.ws_status = "connected"
                    await ws.send(json.dumps({"method":"subscribe","subscription":{"type":"l2Book","coin":self.coin}}))
                    async for msg in ws:
                        if not self.running: break
                        book = self._parse_book(msg)
                        if book:
                            self.latest_book = book
                            self.book_updates += 1
                            self.last_book_wall_ms = int(time.time() * 1000)
                            await self._publish()
            except Exception as e:
                self.ws_status = f"reconnecting: {type(e).__name__}"
                await self._publish()
                await asyncio.sleep(3)

    def _parse_book(self, msg: str) -> L2Book | None:
        obj = json.loads(msg)
        if obj.get("channel") != "l2Book": return None
        d = obj["data"]
        def side(arr): return [BookLevel(float(x["px"]), float(x["sz"]), int(x.get("n", 0))) for x in arr]
        return L2Book(d.get("coin", self.coin), int(d.get("time", time.time()*1000)), side(d["levels"][0]), side(d["levels"][1]))

    async def _decision_loop(self):
        while self.running:
            await asyncio.sleep(1)
            book = self.latest_book
            if not book or book.time_ms == self.last_decision_book_time_ms:
                continue
            self.last_decision_book_time_ms = book.time_ms
            self.step += 1
            state = self.features.build(book, self.broker.position, self.step)
            if state is None: continue
            mask = self.broker.mask()
            action_idx = self.agent.select(state, mask)
            previous_equity = self.last_equity if self.last_equity is not None else self.broker.equity(book)
            result = self.broker.step(Action(action_idx), book, self.step)
            current_equity = float(result["equity"])
            # Correct reward accounting: credit/debit mark-to-market movement since
            # the previous decision tick, plus any execution cost from the current action.
            # The old version compared before/after on the same book snapshot, so HOLD
            # rewards were always zero and the agent saw mostly fees/slippage only.
            result["reward"] = current_equity - previous_equity
            self.last_equity = current_equity
            next_state = self.features.build(book, self.broker.position, self.step) or state
            t = Transition(state, action_idx, float(result["reward"]), next_state, self.broker.mask(), self.broker.position is None and action_idx == Action.EXIT, int(time.time()*1000))
            self.replay.add(t)
            if len(self.replay.data) >= 64:
                self.agent.train(self.replay.sample(64))
            if self.agent.updates and self.agent.updates % 100 == 0: self.agent.save()
            self.last_result = result
            self.last_q = [float(x) for x in self.agent.q_values(state)]
            await self._publish()

    def snapshot(self) -> dict:
        book = self.latest_book
        pos = self.broker.position
        now_ms = int(time.time() * 1000)
        rewards = [float(t.reward) for t in self.replay.data]
        exits = [t for t in self.broker.trades if t.get("event") in {"exit", "forced_exit_max_hold"}]
        return {
            "running": self.running,
            "wsStatus": self.ws_status,
            "bookUpdates": self.book_updates,
            "bookExchangeTime": book.time_ms if book else 0,
            "bookAgeMs": max(0, now_ms - self.last_book_wall_ms) if self.last_book_wall_ms else 0,
            "market": self.market_label,
            "coin": self.coin,
            "markets": DEFAULT_MARKETS,
            "mid": book.mid if book else 0,
            "spreadBps": book.spread_bps if book else 0,
            "bids": [x.__dict__ for x in book.bids[:self.depth]] if book else [],
            "asks": [x.__dict__ for x in book.asks[:self.depth]] if book else [],
            "position": None if not pos else {"side": pos.side.name, "entryPx": pos.entry_px, "qty": pos.qty},
            "equity": self.last_result["equity"],
            "realizedPnl": self.broker.cash_pnl,
            "unrealizedPnl": self.broker.unrealized(book) if book else 0.0,
            "lastAction": self.last_result["action"],
            "lastReward": self.last_result["reward"],
            "reason": self.last_result["reason"],
            "replay": len(self.replay.data),
            "updates": self.agent.updates,
            "epsilon": self.agent.epsilon,
            "qValues": self.last_q,
            "costs": self.broker.costs(),
            "rewardStats": {
                "positive": sum(1 for r in rewards if r > 0),
                "negative": sum(1 for r in rewards if r < 0),
                "zero": sum(1 for r in rewards if r == 0),
                "max": max(rewards) if rewards else 0.0,
                "min": min(rewards) if rewards else 0.0,
            },
            "closedTradeStats": {
                "count": len(exits),
                "positive": sum(1 for t in exits if float(t.get("pnl", 0)) > 0),
                "max": max([float(t.get("pnl", 0)) for t in exits], default=0.0),
                "min": min([float(t.get("pnl", 0)) for t in exits], default=0.0),
            },
            "trades": self.broker.trades[-50:],
        }

    async def subscribe(self):
        q = asyncio.Queue(maxsize=5)
        self.subscribers.add(q)
        try:
            await q.put(self.snapshot())
            while True: yield await q.get()
        finally:
            self.subscribers.discard(q)

    async def _publish(self):
        snap = self.snapshot()
        for q in list(self.subscribers):
            if q.full():
                try: q.get_nowait()
                except Exception: pass
            await q.put(snap)
