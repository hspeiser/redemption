"""VQ2 map editor — browser UI.

    .venv-train\\Scripts\\python.exe scripts\\vq2_map_web.py ^
        --trace data\\vq2_trace_003153.npz --map data\\vq2_map_current.json ^
        --out data\\vq2_map_human.json
    -> open http://localhost:8891

Zoom (wheel) / pan (drag with right button or hold space). Frame slider.
Wireframe + label overlays are OFF-by-default toggles. Corner-fit mode:
click the 4 HOLE corners (any order? NO - top-left, top-right,
bottom-right, bottom-left) of the selected gate on the zoomed image; the
gate's position+yaw are solved exactly from your clicks and the frame's
known camera pose.
"""

import argparse
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from aigp.vision.labels import load_calib  # noqa: E402

W, H = 640, 360
HOLE, PANEL = 0.75, 1.36
SQ = {"hole": np.array([[-HOLE, 0, -HOLE], [HOLE, 0, -HOLE],
                        [HOLE, 0, HOLE], [-HOLE, 0, HOLE]]),
      "panel": np.array([[-PANEL, 0, -PANEL], [PANEL, 0, -PANEL],
                         [PANEL, 0, PANEL], [-PANEL, 0, PANEL]])}
RX90 = np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]])

S = {}          # server state
LOCK = threading.Lock()


def gate_R(g):
    q = g["quat_wxyz"]
    return Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()


def gate_yaw(g):
    return float(Rotation.from_matrix(gate_R(g)).as_euler(
        "zyx", degrees=True)[0])


def set_yaw(g, yaw):
    q = Rotation.from_euler("z", yaw, degrees=True).as_quat()
    g["quat_wxyz"] = [float(q[3]), float(q[0]), float(q[1]), float(q[2])]


def frame_pose(i):
    p = S["pos"][i]
    qw, qx, qy, qz = S["quat"][i]
    R_wb = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
    return p, R_wb


def project_gates(i):
    fx, fy, cx, cy = S["K"]
    p_b, R_wb = frame_pose(i)
    R_cw = S["R_cb"] @ R_wb.T
    out = []
    for gi, g in enumerate(S["gates"]):
        Rg = gate_R(g)
        row = {"id": gi, "hole": [], "panel": [], "center": None}
        ok_any = False
        for kind in ("hole", "panel"):
            pts = []
            for c in SQ[kind]:
                Xc = R_cw @ (np.asarray(g["pos"]) + Rg @ c - p_b)
                if Xc[2] < 0.3:
                    pts = []
                    break
                u = fx * Xc[0] / Xc[2] + cx
                v = fy * Xc[1] / Xc[2] + cy
                if abs(u) > 6000 or abs(v) > 6000:
                    pts = []
                    break
                pts.append([round(float(u), 1), round(float(v), 1)])
            row[kind] = pts
            if pts:
                ok_any = True
        Xc = R_cw @ (np.asarray(g["pos"]) - p_b)
        if Xc[2] > 0.3:
            row["center"] = [round(float(fx * Xc[0] / Xc[2] + cx), 1),
                             round(float(fy * Xc[1] / Xc[2] + cy), 1)]
            row["dist"] = round(float(np.linalg.norm(
                np.asarray(g["pos"]) - p_b)), 1)
        if ok_any and row["center"] is not None and \
                -400 <= row["center"][0] <= W + 400 and \
                -300 <= row["center"][1] <= H + 300:
            out.append(row)
    return out


def nearest_gate(i, corners):
    """Which map gate did the user click? Nearest projected center to the
    click centroid — never trust the dropdown for attribution."""
    cc = np.mean(np.asarray(corners, float), axis=0)
    best, best_d = None, 1e9
    for row in project_gates(i):
        if row["center"] is None:
            continue
        d = float(np.hypot(row["center"][0] - cc[0],
                           row["center"][1] - cc[1]))
        if d < best_d:
            best, best_d = row["id"], d
    return best, best_d


def solve_from_clicks(i, gi, corners):
    """4 clicked HOLE corners (TL,TR,BR,BL in image) -> gate pos+yaw."""
    fx, fy, cx, cy = S["K"]
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    obj = np.ascontiguousarray(SQ["hole"] @ RX90.T)
    ip = np.ascontiguousarray(corners, np.float64).reshape(-1, 1, 2)
    try:
        n, rv, tv, errs = cv2.solvePnPGeneric(
            obj, ip, K, None, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    except cv2.error as e:
        return {"ok": False, "err": str(e)}
    p_b, R_wb = frame_pose(i)
    R_wc = R_wb @ S["R_cb"].T
    best = None
    for r0, t0 in zip(rv, tv):
        try:
            r0, t0 = cv2.solvePnPRefineLM(obj, ip, K, None, r0, t0)
        except cv2.error:
            continue
        pr, _ = cv2.projectPoints(obj, r0, t0, K, None)
        rms = float(np.sqrt(((pr - ip) ** 2).sum(axis=2).mean()))
        R0, _ = cv2.Rodrigues(r0)
        R_g2c = R0 @ RX90
        R_gw = R_wc @ R_g2c
        up_err = abs(float(R_gw[2, 2]) - 1.0)   # gate z should be world z
        score = rms + 5.0 * up_err
        if best is None or score < best[0]:
            best = (score, rms, t0.ravel(), R_gw)
    if best is None:
        return {"ok": False, "err": "no PnP solution"}
    _sc, rms, t_c, R_gw = best
    p_new = p_b + R_wc @ t_c
    yaw = float(np.degrees(np.arctan2(R_gw[1, 0], R_gw[0, 0])))
    g = S["gates"][gi]
    g["pos"] = [float(v) for v in p_new]
    set_yaw(g, yaw)
    S["dirty"].add(gi)
    return {"ok": True, "gate": gi, "rms": round(rms, 2),
            "pos": [round(v, 2) for v in g["pos"]], "yaw": round(yaw, 1)}


HTML = """<!doctype html><html><head><title>VQ2 map editor</title><style>
body{background:#101014;color:#ddd;font-family:system-ui,sans-serif;margin:0}
#top{padding:6px 10px;background:#1a1a22;display:flex;gap:10px;align-items:center;flex-wrap:wrap}
#top b{color:#f80} button{background:#2a2a34;color:#ddd;border:1px solid #444;
border-radius:4px;padding:4px 10px;cursor:pointer} button:hover{background:#3a3a46}
button.on{background:#0a5;color:#fff}
#cv{display:block;margin:0 auto;background:#000;cursor:crosshair}
#hud{padding:4px 10px;color:#8f8;font-family:monospace;white-space:pre}
input[type=range]{width:420px} select{background:#2a2a34;color:#ddd;border:1px solid #444}
.warn{color:#f66}.okp{color:#6f6}#fitmsg{color:#ff0}
</style></head><body>
<div id=top>
 <b>VQ2 map editor</b>
 <button id=prev>&#9664;</button>
 <input type=range id=slider min=0 max=0 value=0>
 <button id=next>&#9654;</button>
 <span id=ft></span>
 <label>gate <select id=gsel></select></label>
 <button id=tWire>wireframes</button>
 <button id=tAll>all gates</button>
 <button id=tLab>labels</button>
 <button id=fit>corner-fit mode</button>
 <span id=fitmsg></span>
 <button id=undo>reset gate</button>
 <button id=save>SAVE MAP</button>
 <span>nudge: arrows=NE/SW &nbsp; PgUp/Dn=height &nbsp; ,/.=yaw &nbsp; [step <select id=step>
 <option value=0.02>2cm</option><option value=0.1 selected>10cm</option><option value=0.5>50cm</option></select>]</span>
</div>
<canvas id=cv width=1280 height=720></canvas>
<div id=hud></div>
<script>
let N=0, i=0, sel=0, st=null, img=new Image();
let scale=2, ox=0, oy=0, wire=false, all=false, labs=false, fitMode=false, clicks=[];
let panning=false, px=0, py=0;
const cv=document.getElementById('cv'), ctx=cv.getContext('2d');
function T(u,v){return [u*scale+ox, v*scale+oy];}
function inv(x,y){return [(x-ox)/scale, (y-oy)/scale];}
async function meta(){const r=await fetch('/meta');const m=await r.json();N=m.n;
 slider.max=N-1; const gs=document.getElementById('gsel');
 for(let k=0;k<m.gates;k++){const o=document.createElement('option');o.value=k;o.text='g'+k;gs.add(o);} }
async function load(){
 st=await (await fetch('/state?i='+i)).json();
 img.onload=draw;
 img.src='/img?i='+i+'&r='+st.rev;
 document.getElementById('ft').textContent='f '+i+'/'+(N-1)+'  t='+st.t.toFixed(1)+'s';
 draw();
}
function draw(){
 ctx.fillStyle='#000';ctx.fillRect(0,0,cv.width,cv.height);
 ctx.imageSmoothingEnabled = scale<3;
 ctx.drawImage(img, ox, oy, 640*scale, 360*scale);
 if(st && wire){
  for(const g of st.gates){
   if(!all && g.id!=sel) continue;
   const col = g.id==sel ? '#ff0' : (st.dirty.includes(g.id)?'#0f5':'#999');
   ctx.strokeStyle=col; ctx.lineWidth=g.id==sel?2:1;
   for(const kind of ['hole','panel']){
    const q=g[kind]; if(!q||q.length!=4) continue;
    ctx.beginPath();
    for(let k=0;k<5;k++){const[x,y]=T(q[k%4][0],q[k%4][1]); k?ctx.lineTo(x,y):ctx.moveTo(x,y);}
    ctx.stroke();
   }
   if(labs && g.center){const[x,y]=T(g.center[0],g.center[1]);
    ctx.fillStyle=col;ctx.font='14px monospace';
    ctx.fillText('g'+g.id+' '+(g.dist||'?')+'m', x+4, y-4);}
  }
 }
 ctx.fillStyle='#f0f';
 for(const c of clicks){const[x,y]=T(c[0],c[1]);ctx.beginPath();ctx.arc(x,y,4,0,7);ctx.fill();}
 const trusted = st && st.sigma<0.12;
 document.getElementById('hud').textContent =
  st? ('sigma '+(st.sigma*100).toFixed(1)+'cm  '+(trusted?'[POSE TRUSTED - safe to align]':'[POSE ROUGH - do not align here]')
   +'   selected g'+sel+(st.sel_info?('  pos='+st.sel_info.pos+'  yaw='+st.sel_info.yaw):'')
   +'   edited: '+st.dirty.join(',')):'';
 document.getElementById('hud').className = trusted?'okp':'warn';
}
async function post(u,body){const r=await fetch(u,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});return r.json();}
document.getElementById('prev').onclick=()=>{i=Math.max(0,i-1);slider.value=i;load();};
document.getElementById('next').onclick=()=>{i=Math.min(N-1,i+1);slider.value=i;load();};
slider.oninput=()=>{i=+slider.value;load();};
gsel.onchange=()=>{sel=+gsel.value;post('/select',{g:sel}).then(load);};
function tog(btn,fn){btn.onclick=()=>{fn();btn.classList.toggle('on');draw();};}
tog(document.getElementById('tWire'),()=>wire=!wire);
tog(document.getElementById('tAll'),()=>all=!all);
tog(document.getElementById('tLab'),()=>labs=!labs);
document.getElementById('fit').onclick=function(){fitMode=!fitMode;clicks=[];
 this.classList.toggle('on');
 document.getElementById('fitmsg').textContent=fitMode?'click hole corners: TL,TR,BR,BL':'';draw();};
document.getElementById('undo').onclick=()=>post('/reset',{g:sel}).then(load);
document.getElementById('save').onclick=()=>post('/save',{}).then(r=>{
 document.getElementById('fitmsg').textContent='saved -> '+r.path;});
cv.addEventListener('wheel',e=>{e.preventDefault();
 const f=e.deltaY<0?1.2:1/1.2; const [u,v]=inv(e.offsetX,e.offsetY);
 scale=Math.min(24,Math.max(1,scale*f)); ox=e.offsetX-u*scale; oy=e.offsetY-v*scale; draw();});
cv.addEventListener('mousedown',e=>{
 if(e.button==2||e.shiftKey){panning=true;px=e.offsetX;py=e.offsetY;return;}
 if(e.button==0&&fitMode){const c=inv(e.offsetX,e.offsetY);clicks.push(c);draw();
  document.getElementById('fitmsg').textContent='corner '+clicks.length+'/4';
  if(clicks.length==4){post('/fit',{g:sel,i:i,corners:clicks}).then(r=>{
   clicks=[];fitMode=false;document.getElementById('fit').classList.remove('on');
   if(r.ok){sel=r.gate;gsel.value=sel;}
   document.getElementById('fitmsg').textContent=r.ok?('fitted GATE '+r.gate+'  rms '+r.rms+'px  pos '+r.pos+'  yaw '+r.yaw):('FIT FAILED '+r.err);
   load();});}
 }});
cv.addEventListener('mousemove',e=>{if(panning){ox+=e.offsetX-px;oy+=e.offsetY-py;px=e.offsetX;py=e.offsetY;draw();}});
cv.addEventListener('mouseup',()=>panning=false);
cv.addEventListener('contextmenu',e=>e.preventDefault());
document.addEventListener('keydown',e=>{
 const s=+document.getElementById('step').value;
 const m={'ArrowUp':[s,0,0,0],'ArrowDown':[-s,0,0,0],'ArrowLeft':[0,-s,0,0],'ArrowRight':[0,s,0,0],
  'PageUp':[0,0,-s,0],'PageDown':[0,0,s,0],',':[0,0,0,-2],'.':[0,0,0,2]};
 if(e.key=='a'){i=Math.max(0,i-1);slider.value=i;load();}
 else if(e.key=='d'){i=Math.min(N-1,i+1);slider.value=i;load();}
 else if(e.key=='w'){i=Math.max(0,i-30);slider.value=i;load();}
 else if(e.key=='e'){i=Math.min(N-1,i+30);slider.value=i;load();}
 else if(m[e.key]){e.preventDefault();const[dn,de,dd,dy]=m[e.key];
  post('/nudge',{g:sel,dn:dn,de:de,dd:dd,dyaw:dy}).then(load);}
});
meta().then(load);
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, obj, code=200):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/":
            b = HTML.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)
        elif u.path == "/meta":
            self._json({"n": len(S["paths"]), "gates": len(S["gates"])})
        elif u.path == "/img":
            i = int(parse_qs(u.query)["i"][0])
            try:
                b = Path(S["paths"][i]).read_bytes()
            except OSError:
                b = b""
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)
        elif u.path == "/state":
            i = int(parse_qs(u.query)["i"][0])
            with LOCK:
                g = S["gates"][S["sel"]]
                self._json({
                    "t": float(S["t"][i]),
                    "sigma": float(S["sigma"][i]),
                    "rev": S["rev"],
                    "gates": project_gates(i),
                    "dirty": sorted(S["dirty"]),
                    "sel_info": {
                        "pos": "(" + ",".join(f"{v:.2f}"
                                              for v in g["pos"]) + ")",
                        "yaw": f"{gate_yaw(g):.1f}"}})
        else:
            self._json({"err": "?"}, 404)

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n) or b"{}")
        u = urlparse(self.path)
        with LOCK:
            if u.path == "/select":
                S["sel"] = int(body["g"])
                self._json({"ok": True})
            elif u.path == "/nudge":
                g = S["gates"][int(body["g"])]
                g["pos"][0] += float(body.get("dn", 0))
                g["pos"][1] += float(body.get("de", 0))
                g["pos"][2] += float(body.get("dd", 0))
                if body.get("dyaw"):
                    set_yaw(g, gate_yaw(g) + float(body["dyaw"]))
                S["dirty"].add(int(body["g"]))
                S["rev"] += 1
                self._json({"ok": True})
            elif u.path == "/fit":
                gi, d_pick = nearest_gate(int(body["i"]), body["corners"])
                if gi is None:
                    self._json({"ok": False,
                                "err": "no map gate near clicks"})
                else:
                    r = solve_from_clicks(int(body["i"]), gi,
                                          body["corners"])
                    if r.get("ok"):
                        S["sel"] = gi
                    S["rev"] += 1
                    self._json(r)
            elif u.path == "/reset":
                gi = int(body["g"])
                S["gates"][gi] = json.loads(json.dumps(S["orig"][gi]))
                S["dirty"].discard(gi)
                S["rev"] += 1
                self._json({"ok": True})
            elif u.path == "/save":
                Path(S["out"]).write_text(json.dumps(
                    {"frame": "local spawn (human-aligned)",
                     "gates": S["gates"]}, indent=1))
                self._json({"ok": True, "path": str(S["out"])})
            else:
                self._json({"err": "?"}, 404)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--map", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--port", type=int, default=8891)
    args = ap.parse_args()

    calib = load_calib(REPO / "data/calib/calib.json")
    tr = np.load(args.trace, allow_pickle=False)
    m = json.loads(Path(args.map).read_text())
    S.update({
        "K": calib["K"], "R_cb": np.asarray(calib["R_cb"]),
        "t": tr["t"], "paths": tr["path"], "pos": tr["pos"],
        "quat": tr["quat"], "sigma": tr["sigma"],
        "gates": m["gates"],
        "orig": json.loads(json.dumps(m["gates"])),
        "sel": 0, "dirty": set(), "out": args.out, "rev": 0,
    })
    print(f"map editor: http://localhost:{args.port}  "
          f"({len(S['paths'])} frames, {len(S['gates'])} gates)")
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
