"""Live dashboard for the state-based racer. Reads runs/metrics.jsonl (one record per episode)
and serves auto-refreshing graphs. Runs locally -> open http://localhost:8000 directly.

  python -m racer_state.dashboard --port 8000
"""
from __future__ import annotations
import argparse, json, os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = "."

PAGE = r"""<!doctype html><html><head><meta charset="utf-8"><title>racer_state — training</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
 :root{--bg:#0d1117;--card:#161b22;--fg:#e6edf3;--mut:#8b949e;--acc:#3fb950;--bd:#30363d}
 *{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--fg);font:14px -apple-system,Segoe UI,Roboto,sans-serif}
 header{padding:14px 20px;border-bottom:1px solid var(--bd);display:flex;align-items:center;gap:16px;flex-wrap:wrap}
 h1{font-size:17px;margin:0} .sub{color:var(--mut);font-size:12px}
 .stats{display:flex;gap:10px;flex-wrap:wrap;margin-left:auto}
 .stat{background:var(--card);border:1px solid var(--bd);border-radius:8px;padding:6px 12px;min-width:76px}
 .stat .v{font-size:19px;font-weight:700} .stat .k{color:var(--mut);font-size:10px;text-transform:uppercase}
 .stat.big .v{color:var(--acc)}
 .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(360px,1fr));gap:12px;padding:16px}
 .card{background:var(--card);border:1px solid var(--bd);border-radius:10px;padding:10px 12px 4px}
 .card h3{margin:0 0 6px;font-size:12px;color:var(--mut);font-weight:600;text-transform:uppercase}
 .wrap{position:relative;height:200px}
</style></head><body>
<header><h1>🚁 racer_state <span class="sub" id="sub">connecting…</span></h1><div class="stats" id="stats"></div></header>
<div class="grid" id="grid"></div>
<script>
const CH={};
function card(id,t){const g=document.getElementById('grid');const c=document.createElement('div');
 c.className='card';c.innerHTML=`<h3>${t}</h3><div class="wrap"><canvas id="${id}"></canvas></div>`;g.appendChild(c);}
const DEFS=[
 ['mind','Min distance to gate 0 (m) vs episode  ▼ lower = better'],
 ['succ','Gate-0 success rate (rolling 20) vs episode'],
 ['rew','Episode reward vs episode'],
 ['gates','Gates passed vs episode'],
 ['steps','Episode length (steps) vs episode'],
 ['q','Mean Q vs episode'],
 ['closs','Critic loss vs episode'],
 ['aloss','Actor loss vs episode'],
 ['alpha','Entropy temperature α vs episode'],
 ['ent','Policy entropy vs episode'],
 ['reasons','Outcome mix (rolling 20): crash/stray/away/timeout'],
 ['ema','Reward EMA (drives adaptive pitch penalty)'],
 ['wpitch','Adaptive pitch-rate penalty weight  ▼ lower = looser pitch'],
 ['caps','Curriculum: rollyaw cap / thrust cap'],
 ['updms','SAC update time (ms per batch)'],
 ['mstep','Env step time (ms) — control loop'],
 ['tput','Throughput (steps/sec): flight vs incl. reset'],
];
DEFS.forEach(([id,t])=>card(id,t));
const GRID={color:'#20262d'},TICK={color:'#8b949e',font:{size:10}};
function mk(id,ds,st){const c=document.getElementById(id);CH[id]=new Chart(c,{type:'line',
 data:{labels:[],datasets:ds},options:{animation:false,responsive:true,maintainAspectRatio:false,
  plugins:{legend:{display:ds.length>1,labels:{color:'#8b949e',boxWidth:10,font:{size:10}}}},
  scales:{x:{grid:GRID,ticks:TICK},y:{grid:GRID,ticks:TICK,stacked:!!st}}}});}
function L(label,color,fill){return{label,data:[],borderColor:color,backgroundColor:color+'33',borderWidth:2,pointRadius:0,tension:.2,fill:!!fill};}
let ok=false;
function init(){
 mk('mind',[L('min dist','#3fb950',true)]);
 mk('succ',[L('success','#3fb950',true)]);
 mk('rew',[L('reward','#58a6ff')]);
 mk('gates',[L('gates','#d29922')]);
 mk('steps',[L('steps','#8b949e')]);
 mk('q',[L('Q','#58a6ff')]);
 mk('closs',[L('critic','#f85149')]);
 mk('aloss',[L('actor','#db61a2')]);
 mk('alpha',[L('alpha','#a371f7')]);
 mk('ent',[L('entropy','#39c5cf')]);
 mk('reasons',[L('crash','#f85149'),L('stray','#d29922'),L('away','#db61a2'),L('timeout','#58a6ff')],true);
 mk('ema',[L('reward EMA','#3fb950')]);
 mk('wpitch',[L('pitch penalty w','#a371f7')]);
 mk('caps',[L('rollyaw','#f85149'),L('thrust','#3fb950')]);
 mk('updms',[L('ms/update','#58a6ff')]);
 mk('mstep',[L('ms/step','#d29922')]);
 mk('tput',[L('flight','#3fb950'),L('incl. reset','#8b949e')]);
 ok=true;
}
function roll(arr,n,f){return arr.map((_,i)=>{const s=arr.slice(Math.max(0,i-n+1),i+1);return f(s);});}
async function tick(){
 let R;try{R=await(await fetch('data')).json();}catch(e){return;}
 const tr=R.filter(r=>!r.eval);
 if(!tr.length){document.getElementById('sub').textContent='waiting for first episode…';return;}
 if(!ok)init();
 const X=tr.map(r=>r.ep);const last=tr[tr.length-1];
 document.getElementById('sub').textContent=`ep ${last.ep} · ${tr.length} episodes · ${last.total_steps} steps · ${new Date().toLocaleTimeString()}`;
 const reached=tr.map(r=>r.reached_g0?1:0);
 const sr=roll(reached,20,s=>s.reduce((a,b)=>a+b,0)/s.length);
 document.getElementById('stats').innerHTML=[
  ['ep',last.ep],['succ20',(sr[sr.length-1]*100).toFixed(0)+'%'],['min d',last.min_dist?.toFixed(1)],
  ['gates',last.gates],['reward',last.reward?.toFixed(0)],['buf',last.total_steps],
  ['alpha',(last.alpha??0).toFixed(3)],['Q',(last.q??0).toFixed(1)]
 ].map(([k,v],i)=>`<div class="stat ${i<2?'big':''}"><div class="v">${v}</div><div class="k">${k}</div></div>`).join('');
 const S=(id,i,d)=>{CH[id].data.labels=X;CH[id].data.datasets[i].data=d;CH[id].update();};
 S('mind',0,tr.map(r=>r.min_dist));
 CH['succ'].data.labels=X;CH['succ'].data.datasets[0].data=sr;CH['succ'].update();
 S('rew',0,tr.map(r=>r.reward));
 S('gates',0,tr.map(r=>r.gates));
 S('steps',0,tr.map(r=>r.steps));
 S('q',0,tr.map(r=>r.q??null));
 S('closs',0,tr.map(r=>r.c_loss??null));
 S('aloss',0,tr.map(r=>r.a_loss??null));
 S('alpha',0,tr.map(r=>r.alpha??null));
 S('ent',0,tr.map(r=>r.entropy??null));
 const rr=(name)=>roll(tr.map(r=>r.reason===name?1:0),20,s=>s.reduce((a,b)=>a+b,0));
 CH['reasons'].data.labels=X;CH['reasons'].data.datasets[0].data=rr('crash');
 CH['reasons'].data.datasets[1].data=rr('stray');CH['reasons'].data.datasets[2].data=rr('away');
 CH['reasons'].data.datasets[3].data=rr('timeout');CH['reasons'].update();
 S('ema',0,tr.map(r=>r.reward_ema));
 S('wpitch',0,tr.map(r=>r.w_pitch));
 S('caps',0,tr.map(r=>r.rollyaw_cap));CH['caps'].data.datasets[1].data=tr.map(r=>r.thrust_cap);CH['caps'].update();
 S('updms',0,tr.map(r=>r.upd_ms));
 S('mstep',0,tr.map(r=>r.ms_per_step));
 const inclReset=tr.map((r,i)=> i>0 ? r.steps/Math.max(0.01, r.t - tr[i-1].t) : null);
 CH['tput'].data.labels=X;CH['tput'].data.datasets[0].data=tr.map(r=>r.steps_per_s);
 CH['tput'].data.datasets[1].data=inclReset;CH['tput'].update();
}
tick();setInterval(tick,2500);
</script></body></html>"""


class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_GET(self):
        if self.path.startswith("/data"):
            recs = []
            p = os.path.join(ROOT, "metrics.jsonl")
            if os.path.exists(p):
                with open(p, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try: recs.append(json.loads(line))
                            except json.JSONDecodeError: pass
            body = json.dumps(recs).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
        else:
            body = PAGE.encode()
            self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)


def main():
    global ROOT
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=r"C:/Users/satas/Downloads/AI-GP Simulator v1.0.3385-VQ1/PyAIPilotExample-v1/racer_state/runs")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    ROOT = args.root
    print(f"[dashboard] http://localhost:{args.port}  (metrics: {ROOT}/metrics.jsonl)", flush=True)
    ThreadingHTTPServer(("0.0.0.0", args.port), H).serve_forever()


if __name__ == "__main__":
    main()
