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
        self.pending_state: list[float] | None = None
        self.pending_action: int | None = None
        self.pending_done: bool = False
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
        funding = None
        deployer_scale = None
        for asset, ctx in zip(meta.get("universe", []), ctxs):
            if asset.get("name") == self.coin or asset.get("name") == self.coin.split(":")[-1]:
                funding = float(ctx.get("funding") or 0.0)
                deployer_scale = float(asset.get("deployerFeeScale") or 1.0)
                break
        if funding is None or deployer_scale is None:
            raise RuntimeError(f"selected asset {self.coin} not found in HyperLiquid metaAndAssetCtxs")
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

    async def stop(self, *, save: bool = True):
        self.running = False
        tasks = [t for t in [self.ws_task, self.loop_task] if t]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self.ws_task = None
        self.loop_task = None
        self.ws_status = "idle"
        if save:
            self.agent.save()
            self.replay.compact()
        await self._publish()

    async def reset_learning(self):
        # Hard reset means: stop training, cancel WS/decision tasks, archive active
        # replay/checkpoint, and clear all visible runtime/training/trade counters.
        await self.stop(save=False)
        ts = int(time.time())
        for path in [self.data_dir / "replay" / f"{self.market_label}.jsonl", self.data_dir / "checkpoints" / f"{self.market_label}.npz"]:
            if path.exists():
                path.rename(path.with_name(f"{path.stem}.archive-{ts}{path.suffix}"))
        self.features = L2FeatureBuilder(self.depth)
        self.broker = VirtualPerpBroker()
        self.replay = ReplayStore(self.data_dir / "replay" / f"{self.market_label}.jsonl")
        self.agent = DuelingDoubleDQN(self.features.feature_size, ckpt=self.data_dir / "checkpoints" / f"{self.market_label}.npz")
        self.latest_book = None
        self.book_updates = 0
        self.last_book_wall_ms = 0
        self.last_decision_book_time_ms = 0
        self.ws_status = "idle"
        self.step = 0
        self.last_equity = None
        self.pending_state = None
        self.pending_action = None
        self.pending_done = False
        self.last_q = [0.0] * 5
        self.last_result = {"action":"WAIT","reward":0,"equity":0,"realized":0,"reason":"reset_learning"}
        await self._publish()

    async def set_market(self, label: str):
        if label not in DEFAULT_MARKETS: raise ValueError("unknown market")
        was = self.running
        old_label, old_coin = self.market_label, self.coin
        if was: await self.stop()
        try:
            self.market_label, self.coin = label, DEFAULT_MARKETS[label]
            new_broker = VirtualPerpBroker()
            old_broker = self.broker
            self.broker = new_broker
            await self.refresh_costs()
            self.features = L2FeatureBuilder(self.depth)
            self.replay = ReplayStore(self.data_dir / "replay" / f"{label}.jsonl")
            self.agent = DuelingDoubleDQN(self.features.feature_size, ckpt=self.data_dir / "checkpoints" / f"{label}.npz")
            self.latest_book = None
            self.book_updates = 0
            self.last_book_wall_ms = 0
            self.last_decision_book_time_ms = 0
            self.ws_status = "idle"
            self.step = 0
            self.last_equity = None
            self.pending_state = None
            self.pending_action = None
            self.pending_done = False
            self.last_result = {"action":"WAIT","reward":0,"equity":0,"realized":0,"reason":"market_changed"}
            self.last_q = [0.0] * 5
        except Exception:
            self.market_label, self.coin = old_label, old_coin
            self.broker = old_broker if 'old_broker' in locals() else self.broker
            raise
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
            state_before_action = self.features.build(book, self.broker.position, self.step)
            if state_before_action is None: continue
            first_decision = self.last_equity is None
            previous_equity = self.last_equity if self.last_equity is not None else self.broker.equity(book)

            # Close the previous between-book interval using a legal passive
            # action for that interval: HOLD while positioned, WAIT while flat.
            # Do not write ENTER/EXIT from states where those actions are masked.
            if self.pending_state is not None and self.pending_action is not None:
                interval_reward = self.broker.equity(book) - previous_equity
                t = Transition(self.pending_state, self.pending_action, float(interval_reward), state_before_action, self.broker.mask(), self.pending_done, int(time.time()*1000))
                self.replay.add(t)
                if len(self.replay.data) >= 64:
                    self.agent.train(self.replay.sample(64))
                self.last_result["reward"] = float(interval_reward)

            mask = self.broker.mask()
            action_idx = self.agent.select(state_before_action, mask)
            result = self.broker.step(Action(action_idx), book, self.step)
            current_equity = float(result["equity"])
            self.last_equity = current_equity
            state_after_action = self.features.patch_position(state_before_action, self.broker.position, self.step)

            # Immediate execution cost/slippage belongs to the selected action.
            if result["reason"].startswith("enter") or result["reason"] in {"exit", "forced_exit_max_hold"}:
                effective_idx = Action[result["action"]].value
                t = Transition(state_before_action, effective_idx, float(result["reward"]), state_after_action, self.broker.mask(), result["reason"] in {"exit", "forced_exit_max_hold"}, int(time.time()*1000))
                self.replay.add(t)
                if len(self.replay.data) >= 64:
                    self.agent.train(self.replay.sample(64))

            # The next between-book interval is legally represented by HOLD if
            # we are positioned after this action, otherwise WAIT.
            if first_decision and self.broker.position is None and result["reason"] == "wait_flat":
                self.pending_state = None
                self.pending_action = None
                self.pending_done = False
            else:
                self.pending_state = state_after_action
                self.pending_action = Action.HOLD.value if self.broker.position is not None else Action.WAIT.value
                self.pending_done = False
            if self.agent.updates and self.agent.updates % 100 == 0: self.agent.save()
            self.last_result = result
            self.last_q = [float(x) for x in self.agent.q_values(state_before_action)]
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
