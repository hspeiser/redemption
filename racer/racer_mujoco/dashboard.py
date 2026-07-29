"""Live dashboard for the MuJoCo racer. Reads runs_mj/metrics.jsonl. Local -> http://localhost:8060

  python -m racer_mujoco.dashboard --port 8060
"""
from __future__ import annotations
import argparse, json, os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.join(os.path.dirname(__file__), "runs_mj")

PAGE = r"""<!doctype html><html><head><meta charset="utf-8"><title>racer_mujoco</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
 :root{--bg:#0d1117;--card:#161b22;--fg:#e6edf3;--mut:#8b949e;--acc:#3fb950;--bd:#30363d}
 *{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px -apple-system,Segoe UI,Roboto,sans-serif}
 header{padding:14px 20px;border-bottom:1px solid var(--bd);display:flex;gap:16px;align-items:center;flex-wrap:wrap}
 h1{font-size:17px;margin:0} .sub{color:var(--mut);font-size:12px}
 .stats{display:flex;gap:10px;flex-wrap:wrap;margin-left:auto}
 .stat{background:var(--card);border:1px solid var(--bd);border-radius:8px;padding:6px 12px;min-width:74px}
 .stat .v{font-size:19px;font-weight:700}.stat .k{color:var(--mut);font-size:10px;text-transform:uppercase}
 .stat.big .v{color:var(--acc)}
 .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(360px,1fr));gap:12px;padding:16px}
 .card{background:var(--card);border:1px solid var(--bd);border-radius:10px;padding:10px 12px 4px}
 .card h3{margin:0 0 6px;font-size:12px;color:var(--mut);font-weight:600;text-transform:uppercase}
 .wrap{position:relative;height:200px}
</style></head><body>
<header><h1>🐜 racer_mujoco <span class="sub" id="sub">connecting…</span></h1>
<label class="sub" style="user-select:none;cursor:pointer"><input type="checkbox" id="emaRew" checked> EMA reward</label>
<label class="sub" style="user-select:none;cursor:pointer"><input type="checkbox" id="emaG" checked> EMA gates
 <input type="number" id="emaK" value="0.05" min="0.01" max="1" step="0.01" title="EMA smoothing factor" style="width:52px;background:var(--card);color:var(--fg);border:1px solid var(--bd);border-radius:4px"></label>
<label class="sub" style="user-select:none">eval gate
 <select id="evalGate" style="background:var(--card);color:var(--fg);border:1px solid var(--bd);border-radius:4px">
  <option value="all" selected>all</option>
  <option value="0">g0</option><option value="1">g1</option><option value="2">g2</option>
  <option value="3">g3</option><option value="4">g4</option><option value="5">g5</option>
 </select></label>
<div class="stats" id="stats"></div></header>
<div class="grid" id="grid"></div>
<script>
const CH={};
const DEFS=[
 ['eval','GREEDY eval: gate-0 passes / 20  ▲ goal = 20'],
 ['pr','Train pass rate (rolling 100) %'],
 ['gmax','Gates passed per episode (1 = through g0, 2 = through g1, …)'],
 ['mind','Min distance to gate (m): train (EMA) + greedy eval  ▼ lower better'],
 ['rew','Episode reward'],
 ['sps','Throughput (env steps / sec)'],
 ['q','Mean Q'],
 ['closs','Critic loss'],
 ['alpha','Entropy temperature α'],
 ['reasons','Outcome mix (rolling 30): success/crash/away/stray'],
];
function card(id,t){const g=document.getElementById('grid');const c=document.createElement('div');
 c.className='card';c.innerHTML=`<h3>${t}</h3><div class="wrap"><canvas id="${id}"></canvas></div>`;g.appendChild(c);}
DEFS.forEach(([id,t])=>card(id,t));
const GRID={color:'#20262d'},TICK={color:'#8b949e',font:{size:10}};
function L(l,c,f){return{label:l,data:[],borderColor:c,backgroundColor:c+'33',borderWidth:2,pointRadius:0,tension:.2,fill:!!f};}
function mk(id,ds,st){CH[id]=new Chart(document.getElementById(id),{type:'line',data:{labels:[],datasets:ds},
 options:{animation:false,responsive:true,maintainAspectRatio:false,
  plugins:{legend:{display:ds.length>1,labels:{color:'#8b949e',boxWidth:10,font:{size:10}}}},
  scales:{x:{grid:GRID,ticks:TICK},y:{grid:GRID,ticks:TICK,stacked:!!st}}}});}
let ok=false;
function init(){
 mk('eval',[L('g0/20','#3fb950'),L('g1/20','#58a6ff'),L('g2/20','#a371f7'),
            L('g3/20','#d29922'),L('g4/20','#db61a2'),L('g5/20','#f85149')]);
 mk('pr',[L('pass%','#58a6ff',true)]);
 mk('gmax',[L('gates','#3fb950'),
   {label:'raw',data:[],borderColor:'rgba(63,185,80,0.18)',backgroundColor:'transparent',
    borderWidth:1,pointRadius:0,tension:.2}]);
 mk('mind',[{label:'train raw',data:[],borderColor:'rgba(210,153,34,0.15)',backgroundColor:'transparent',borderWidth:1,pointRadius:0,tension:.2},
            L('train EMA','#d29922'),
            {label:'greedy eval',data:[],borderColor:'#3fb950',backgroundColor:'transparent',borderWidth:2,pointRadius:2,tension:.2,spanGaps:true}]);
 mk('rew',[L('reward','#58a6ff'),
   {label:'raw',data:[],borderColor:'rgba(88,166,255,0.18)',backgroundColor:'transparent',
    borderWidth:1,pointRadius:0,tension:.2}]);
 mk('sps',[L('steps/s','#a371f7')]);mk('q',[L('Q','#58a6ff')]);
 mk('closs',[L('critic','#f85149')]);mk('alpha',[L('alpha','#a371f7')]);
 mk('reasons',[L('finish','#3fb950'),L('success','#2ea043'),L('crash','#f85149'),L('away','#db61a2'),L('stray','#d29922')],true);
 ok=true;
}
function roll(a,n,f){return a.map((_,i)=>f(a.slice(Math.max(0,i-n+1),i+1)));}
function ema(a,k){let m=null;return a.map(v=>{if(v==null||!isFinite(v))return m;m=(m==null)?v:k*v+(1-k)*m;return m;});}
async function tick(){
 let R;try{R=await(await fetch('data')).json();}catch(e){return;}
 if(!R.length){document.getElementById('sub').textContent='waiting for first episode…';return;}
 if(!ok)init();
 const X=R.map(r=>r.ep),last=R[R.length-1];
 document.getElementById('sub').textContent=`ep ${last.ep} · step ${last.step} · ${(last.steps_per_s||0).toFixed(0)} steps/s · ${new Date().toLocaleTimeString()}`;
 const ev=R.filter(r=>r.eval_g0!=null);
 document.getElementById('stats').innerHTML=[
  ['ep',last.ep],['eval',ev.length?ev[ev.length-1].eval_g0+'/20':'—'],
  ['pass%',(100*(last.pass_rate100||0)).toFixed(0)],['min d',(last.min_dist||0).toFixed(1)],
  ['steps/s',(last.steps_per_s||0).toFixed(0)],['buf',last.buf]
 ].map(([k,v],i)=>`<div class="stat ${i<2?'big':''}"><div class="v">${v}</div><div class="k">${k}</div></div>`).join('');
 const S=(id,d)=>{CH[id].data.labels=X;CH[id].data.datasets[0].data=d;CH[id].update();};
 const gsel=document.getElementById('evalGate').value;
 const gc=r=>r.eval_gates??[r.eval_g0,r.eval_g1];
 CH['eval'].data.labels=ev.map(r=>r.ep);
 for(let k=0;k<6;k++)CH['eval'].data.datasets[k].data=(gsel=='all'||+gsel==k)?ev.map(r=>gc(r)[k]??null):[];
 CH['eval'].update();
 S('pr',R.map(r=>100*(r.pass_rate100||0)));
 const rewRaw=R.map(r=>r.reward);
 const useE=document.getElementById('emaRew').checked,kk=+document.getElementById('emaK').value||0.05;
 const mdRaw=R.map(r=>r.min_dist);
 CH['mind'].data.labels=X;CH['mind'].data.datasets[0].data=mdRaw;
 CH['mind'].data.datasets[1].data=ema(mdRaw,kk);
 CH['mind'].data.datasets[2].data=R.map(r=>r.eval_mind??null);CH['mind'].update();
 CH['rew'].data.labels=X;CH['rew'].data.datasets[0].data=useE?ema(rewRaw,kk):rewRaw;
 CH['rew'].data.datasets[1].data=useE?rewRaw:[];CH['rew'].update();
 const gRaw=R.map(r=>r.gates??0);
 const useG=document.getElementById('emaG').checked;
 CH['gmax'].data.labels=X;CH['gmax'].data.datasets[0].data=useG?ema(gRaw,kk):gRaw;
 CH['gmax'].data.datasets[1].data=useG?gRaw:[];CH['gmax'].update();
 S('sps',R.map(r=>r.steps_per_s));S('q',R.map(r=>r.q??null));S('closs',R.map(r=>r.c_loss??null));S('alpha',R.map(r=>r.alpha??null));
 const rr=n=>roll(R.map(r=>r.reason===n?1:0),30,s=>s.reduce((a,b)=>a+b,0));
 CH['reasons'].data.labels=X;['finish','success','crash','away','stray'].forEach((n,i)=>CH['reasons'].data.datasets[i].data=rr(n));CH['reasons'].update();
}
tick();setInterval(tick,2000);
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
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--port", type=int, default=8060)
    args = ap.parse_args()
    ROOT = args.root
    print(f"[dashboard] http://localhost:{args.port}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", args.port), H).serve_forever()


if __name__ == "__main__":
    main()
