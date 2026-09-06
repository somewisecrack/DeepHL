from __future__ import annotations

import json
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from .engine import DeepHLEngine

ROOT = Path(__file__).resolve().parents[2]
engine = DeepHLEngine(ROOT / "data")
app = FastAPI(title="DeepHL")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class MarketReq(BaseModel):
    market: str

@app.get("/api/state")
def state():
    return engine.snapshot()

@app.post("/api/start")
async def start():
    await engine.start()
    return engine.snapshot()

@app.post("/api/stop")
async def stop():
    await engine.stop()
    return engine.snapshot()

@app.post("/api/reset")
async def reset():
    await engine.reset_learning()
    return engine.snapshot()

@app.post("/api/market")
async def market(req: MarketReq):
    try:
        await engine.set_market(req.market)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return engine.snapshot()

@app.get("/api/events")
async def events():
    async def gen():
        async for snap in engine.subscribe():
            yield f"data: {json.dumps(snap)}\n\n"
    return StreamingResponse(gen(), media_type="text/event-stream")
