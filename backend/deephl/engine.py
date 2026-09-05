from __future__ import annotations

import asyncio, json, time
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
        self.running = False
        self.ws_task: asyncio.Task | None = None
        self.loop_task: asyncio.Task | None = None
        self.step = 0
        self.last_state: list[float] | None = None
        self.last_action: int | None = None
        self.last_result = {"action":"WAIT","reward":0,"equity":0,"realized":0,"reason":"idle"}
        self.last_q = [0.0] * 5
        self.subscribers: set[asyncio.Queue] = set()

    async def start(self):
        if self.running: return
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

    async def set_market(self, label: str):
        if label not in DEFAULT_MARKETS: raise ValueError("unknown market")
        was = self.running
        if was: await self.stop()
        self.market_label, self.coin = label, DEFAULT_MARKETS[label]
        self.features = L2FeatureBuilder(self.depth)
        self.broker = VirtualPerpBroker()
        self.replay = ReplayStore(self.data_dir / "replay" / f"{label}.jsonl")
        self.agent = DuelingDoubleDQN(self.features.feature_size, ckpt=self.data_dir / "checkpoints" / f"{label}.npz")
        self.latest_book = None
        self.last_state = None
        self.last_action = None
        if was: await self.start()
        await self._publish()

    async def _ws_loop(self):
        while self.running:
            try:
                async with websockets.connect("wss://api.hyperliquid.xyz/ws", ping_interval=20) as ws:
                    await ws.send(json.dumps({"method":"subscribe","subscription":{"type":"l2Book","coin":self.coin}}))
                    async for msg in ws:
                        if not self.running: break
                        book = self._parse_book(msg)
                        if book: self.latest_book = book
            except Exception:
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
            if not book: continue
            self.step += 1
            state = self.features.build(book, self.broker.position, self.step)
            if state is None: continue
            mask = self.broker.mask()
            action_idx = self.agent.select(state, mask)
            result = self.broker.step(Action(action_idx), book, self.step)
            next_state = self.features.build(book, self.broker.position, self.step) or state
            if self.last_state is not None and self.last_action is not None:
                t = Transition(self.last_state, self.last_action, float(result["reward"]), next_state, self.broker.mask(), self.broker.position is None and action_idx == Action.EXIT, int(time.time()*1000))
                self.replay.add(t)
            if len(self.replay.data) >= 64:
                self.agent.train(self.replay.sample(64))
            if self.agent.updates and self.agent.updates % 100 == 0: self.agent.save()
            self.last_state, self.last_action = next_state, action_idx
            self.last_result = result
            self.last_q = [float(x) for x in self.agent.q_values(state)]
            await self._publish()

    def snapshot(self) -> dict:
        book = self.latest_book
        pos = self.broker.position
        return {
            "running": self.running,
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
            "lastAction": self.last_result["action"],
            "lastReward": self.last_result["reward"],
            "reason": self.last_result["reason"],
            "replay": len(self.replay.data),
            "updates": self.agent.updates,
            "epsilon": self.agent.epsilon,
            "qValues": self.last_q,
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
