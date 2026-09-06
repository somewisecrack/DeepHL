import React, {useEffect, useMemo, useState} from 'react';
import {createRoot} from 'react-dom/client';
import {Activity, Brain, Database, Pause, Play, Radio, Zap} from 'lucide-react';
import './style.css';

const API = import.meta.env.VITE_API || 'http://127.0.0.1:18987';
const actions = ['WAIT','LONG','SHORT','HOLD','EXIT'];

function App(){
  const [s,setS]=useState(null);
  useEffect(()=>{ fetch(API+'/api/state').then(r=>r.json()).then(setS); const es=new EventSource(API+'/api/events'); es.onmessage=e=>setS(JSON.parse(e.data)); return()=>es.close();},[]);
  const pnlColor = (s?.equity||0)>=0 ? 'good':'bad';
  if(!s) return <div className="boot">DeepHL booting…</div>;
  const qmax = Math.max(1e-9,...s.qValues.map(x=>Math.abs(x)));
  return <main>
    <section className="hero card">
      <div className="brand"><img src="/icon.jpeg"/><div><h1>DeepHL</h1><p>L2-only reinforcement lab for HyperLiquid perps</p></div></div>
      <div className="controls">
        <button className="primary" onClick={()=>fetch(API+'/api/start',{method:'POST'}).then(r=>r.json()).then(setS)} disabled={s.running}><Play size={17}/> Start training</button>
        <button onClick={()=>fetch(API+'/api/stop',{method:'POST'}).then(r=>r.json()).then(setS)} disabled={!s.running}><Pause size={17}/> Stop</button>
        <button className="danger" onClick={()=>{ if(confirm('Hard reset: stop training, archive active replay/checkpoint, and clear all visible runtime stats?')) fetch(API+'/api/reset',{method:'POST'}).then(r=>r.json()).then(setS) }}><Zap size={17}/> Reset learning</button>
      </div>
      <div className="status"><span className={s.running?'dot live':'dot'} /> {s.running?'LIVE TRAINING':'PAUSED'} · {s.coin} · WS {s.wsStatus || 'idle'} · L2 #{s.bookUpdates || 0} · age {s.bookAgeMs || 0}ms</div>
    </section>

    <section className="grid top">
      <Metric icon={<Zap/>} label="Virtual PnL" value={`${fmtSigned(s.equity)} USDC`} tone={pnlColor}/>
      <Metric icon={<Activity/>} label="Last reward" value={fmtSigned(s.lastReward)} tone={(s.lastReward||0)>=0?'good':'bad'} sub={`+${s.rewardStats?.positive||0} / -${s.rewardStats?.negative||0}`}/>
      <Metric icon={<Database/>} label="Replay" value={s.replay.toLocaleString()} sub="durable JSONL"/>
      <Metric icon={<Radio/>} label="L2 updates" value={(s.bookUpdates||0).toLocaleString()} sub={`age ${s.bookAgeMs||0}ms`}/>
      <Metric icon={<Brain/>} label="Updates" value={s.updates.toLocaleString()} sub={`ε ${s.epsilon.toFixed(3)}`}/>
      <Metric icon={<Activity/>} label="HL taker fee" value={`${bps(s.costs?.crossFeeRate)} bps`} sub="from /info userFees"/>
    </section>

    <section className="card panelRow">
      <div><label>Market</label><select value={s.market} onChange={e=>fetch(API+'/api/market',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({market:e.target.value})}).then(r=>r.json()).then(setS)} disabled={s.running}>{Object.keys(s.markets).map(k=><option key={k}>{k}</option>)}</select></div>
      <div><label>Mid</label><strong>{num(s.mid)}</strong></div>
      <div><label>Spread</label><strong>{(s.spreadBps||0).toFixed(3)} bps</strong></div>
      <div><label>Book time</label><strong>{s.bookExchangeTime ? new Date(s.bookExchangeTime).toLocaleTimeString() : '—'}</strong></div>
      <div><label>Position</label><strong>{s.position?`${s.position.side} @ ${num(s.position.entryPx)}`:'flat'}</strong><small>unreal {fmtSigned(s.unrealizedPnl)}</small></div>
      <div><label>Last action</label><strong>{s.lastAction}</strong><small>{s.reason}</small></div>
      <div><label>Funding/hr</label><strong>{bps(s.costs?.fundingRateHourly)} bps</strong><small>{s.costs?.source || 'not loaded'}</small></div>
    </section>

    <section className="columns">
      <div className="card"><h2><Radio/> L2 book</h2><Book bids={s.bids} asks={s.asks}/></div>
      <div className="card"><h2><Brain/> Policy monitor</h2><p className="muted">Masked Double-Q learner. Inputs are only L2 book shape + virtual position state.</p><div className="qgrid">{s.qValues.map((q,i)=><div key={actions[i]} className="q"><span>{actions[i]}</span><b>{q.toFixed(4)}</b><i style={{width:`${Math.abs(q)/qmax*100}%`}} /></div>)}</div></div>
    </section>

    <section className="card"><h2>Virtual trade ledger</h2><p className="muted">Closed exits positive: {s.closedTradeStats?.positive||0}/{s.closedTradeStats?.count||0}. Table separates gross move, HyperLiquid fee, and net ledger PnL.</p><table><thead><tr><th>Event</th><th>Price</th><th>Qty</th><th>Gross</th><th>Fee</th><th>Net PnL</th></tr></thead><tbody>{[...s.trades].reverse().map((t,i)=><tr key={i}><td>{t.event}</td><td>{num(t.px)}</td><td>{Number(t.qty).toFixed(5)}</td><td className={(t.gross||0)>=0?'good':'bad'}>{fmtSigned(t.gross||0)}</td><td className="bad">-{Number(t.fee||0).toFixed(4)}</td><td className={t.pnl>=0?'good':'bad'}>{fmtSigned(t.pnl)}</td></tr>)}</tbody></table></section>
  </main>
}
function Metric({icon,label,value,sub,tone=''}){return <div className="metric card"><div>{icon}<span>{label}</span></div><b className={tone}>{value}</b>{sub&&<small>{sub}</small>}</div>}
function Book({bids,asks}){const max=Math.max(1,...bids.map(x=>x.sz),...asks.map(x=>x.sz));return <div className="book"><div><h3>Bids</h3>{bids.map((x,i)=><Level key={i} x={x} max={max} side="bid"/>)}</div><div><h3>Asks</h3>{asks.map((x,i)=><Level key={i} x={x} max={max} side="ask"/>)}</div></div>}
function Level({x,max,side}){return <div className={'level '+side}><i style={{width:`${x.sz/max*100}%`}}/><span>{num(x.px)}</span><b>{Number(x.sz).toFixed(2)}</b><em>{x.n}</em></div>}
function num(x){return Number(x||0).toLocaleString(undefined,{maximumFractionDigits:4})}
function fmtSigned(x){return `${x>=0?'+':''}${Number(x||0).toFixed(4)}`}
function bps(x){return (Number(x||0)*10000).toFixed(3)}
createRoot(document.getElementById('root')).render(<App/>);
