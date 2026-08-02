"""Low-overhead live dashboard for the exact VQ2 control localizer."""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np


_HTML = b"""<!doctype html>
<html><head><meta charset="utf-8"><title>VQ2 Localizer Live</title>
<style>
:root{color-scheme:dark}body{margin:0;background:#090b10;color:#dce3ef;
font:14px ui-monospace,Consolas,monospace}.top{display:flex;align-items:center;
gap:18px;padding:10px 16px;background:#111722;border-bottom:1px solid #273044}
h2{margin:0;color:#ff8a32}.ok{color:#5dff9a}.bad{color:#ff5d72}
.grid{display:grid;grid-template-columns:minmax(520px,1fr) minmax(520px,1fr);
gap:12px;padding:12px}.panel{background:#10151e;border:1px solid #273044;
border-radius:8px;padding:10px}
canvas{width:100%;height:330px;background:#080b10;border-radius:5px}
pre{white-space:pre-wrap;line-height:1.45;margin:0}
.wide{grid-column:1/-1}.title{color:#aab6ca;margin:0 0 8px 2px}
.legend{padding:8px 0;color:#aab6ca}.red{color:#ff5364}.green{color:#5dff9a}
.cyan{color:#4edbff}.amber{color:#ffca55}.magenta{color:#ff62d0}
</style></head><body>
<div class="top"><h2>VQ2 G0-G4 POC</h2>
<span id="health">waiting...</span><span id="gate"></span><span id="source"></span>
</div><div class="grid">
<div class="panel"><div class="title">LIVE MAP / EKF POSITION</div>
<canvas id="map" width="700" height="430"></canvas></div>
<div class="panel"><div class="title">LOWEST GATE-4 COMPLETION TIME VS RUN NUMBER</div>
<canvas id="times" width="700" height="430"></canvas></div>
<div class="panel wide"><pre id="stats">waiting...</pre></div>
</div><script>
const cv=document.getElementById('map'),ctx=cv.getContext('2d'),
tcv=document.getElementById('times'),tctx=tcv.getContext('2d');
function n(v,d=3){return Number.isFinite(v)?v.toFixed(d):'--'}
function draw(s){ctx.clearRect(0,0,cv.width,cv.height);let pts=s.map_positions||[];
if(!pts.length)return;let p=s.position||[0,0,0], all=pts.concat([p]);
let xs=all.map(v=>v[0]),ys=all.map(v=>v[1]),pad=38;
let minx=Math.min(...xs),maxx=Math.max(...xs),miny=Math.min(...ys),maxy=Math.max(...ys);
let sc=Math.min((cv.width-2*pad)/Math.max(maxx-minx,1),(cv.height-2*pad)/Math.max(maxy-miny,1));
let xy=q=>[pad+(q[0]-minx)*sc,cv.height-pad-(q[1]-miny)*sc];
ctx.strokeStyle='#3b465a';ctx.lineWidth=2;ctx.beginPath();pts.forEach((q,i)=>{
let z=xy(q);if(i)ctx.lineTo(...z);else ctx.moveTo(...z)});ctx.stroke();
pts.forEach((q,i)=>{let z=xy(q);ctx.fillStyle=i===s.target?'#ff8a32':'#73829d';
ctx.beginPath();ctx.arc(...z,i===s.target?7:4,0,7);ctx.fill();ctx.fillStyle='#cdd6e5';
ctx.fillText(String(i),z[0]+7,z[1]-7)});let z=xy(p);ctx.fillStyle='#5dff9a';
ctx.beginPath();ctx.arc(...z,7,0,7);ctx.fill();ctx.fillText('DRONE',z[0]+10,z[1]+4)}
function drawTimes(s){let rows=s.poc_results||[];tctx.clearRect(0,0,tcv.width,tcv.height);
let pad=48,w=tcv.width-2*pad,h=tcv.height-2*pad;
tctx.strokeStyle='#3b465a';tctx.lineWidth=1;tctx.strokeRect(pad,pad,w,h);
let thresholds=[8.7,8.27,7.787], times=rows.filter(r=>Number.isFinite(r.time_s));
let run=(r,i)=>Number.isFinite(r.run_number)?r.run_number:i;
let xmax=Math.max(10,...rows.map((r,i)=>run(r,i)+1)),ymax=10.3,ymin=7.3;
let X=x=>pad+x/xmax*w,Y=y=>pad+(ymax-y)/(ymax-ymin)*h;
thresholds.forEach((v,i)=>{tctx.strokeStyle=['#ffca55','#ff8a32','#ff5d72'][i];
tctx.setLineDash([6,5]);tctx.beginPath();tctx.moveTo(pad,Y(v));tctx.lineTo(pad+w,Y(v));
tctx.stroke();tctx.setLineDash([]);tctx.fillStyle=tctx.strokeStyle;tctx.fillText(v+'s',pad+4,Y(v)-5)});
tctx.fillStyle='#73829d';rows.forEach((r,i)=>{if(!Number.isFinite(r.time_s))
tctx.fillRect(X(run(r,i))-2,pad+h-4,4,4)});
tctx.fillStyle='#aab6ca';rows.forEach((r,i)=>{if(Number.isFinite(r.time_s)){
tctx.beginPath();tctx.arc(X(run(r,i)),Y(r.time_s),3,0,7);tctx.fill()}});
let best=Infinity;tctx.strokeStyle='#5dff9a';tctx.lineWidth=3;tctx.beginPath();let started=false;
rows.forEach((r,i)=>{if(Number.isFinite(r.time_s))best=Math.min(best,r.time_s);if(Number.isFinite(best)){
let x=X(run(r,i)),y=Y(best);if(started)tctx.lineTo(x,y);else{tctx.moveTo(x,y);started=true}}});tctx.stroke();
tctx.fillStyle='#cdd6e5';tctx.fillText('run',pad+w-25,pad+h+28);
tctx.save();tctx.translate(15,pad+100);tctx.rotate(-Math.PI/2);tctx.fillText('gate-4 time (s)',0,0);tctx.restore()}
async function tick(){try{let r=await fetch('/state.json?'+Date.now()),s=await r.json();
let healthy=s.timing_healthy!==false;document.getElementById('health').textContent=healthy?'TIMING HEALTHY':'TIMING BAD';
document.getElementById('health').className=healthy?'ok':'bad';
document.getElementById('gate').textContent='TARGET GATE '+(s.target??'--');
document.getElementById('source').textContent='VISION '+(s.localizer_source??'--');
document.getElementById('stats').textContent=
`episode / step       ${s.episode??'--'} / ${s.step??'--'}
probe arm            ${s.schedule_arm??s.probe_arm??'--'}
position XYZ         ${(s.position||[]).map(x=>n(x,2)).join(', ')}
velocity m/s         ${(s.velocity||[]).map(x=>n(x,2)).join(', ')}
speed                ${n(s.speed,2)} m/s
position sigma       ${n(s.position_sigma_m,3)} m
landmark age         ${n(s.visual_age_s,3)} s
camera packet age    ${n(s.camera_age_s,3)} s
IMU packet age       ${n(s.imu_age_s,3)} s
corners fused        ${s.corners_fused??0}
association matches  ${s.debug_match_count??0}
association radius   ${n(s.association_radius_px,1)} px
vision consensus     ${s.visual_consensus??'--'}
measurement sigma    ${n(s.measurement_sigma_px,2)} px
vision inference     ${n(s.vision_inference_ms,1)} ms
crop tracker         ${s.crop_tracker_enabled?'ON':'OFF'}
crop inference       ${n(s.crop_track_ms,1)} ms
crop fixes accepted  ${s.crop_track_updates??0}
control step         ${n(s.step_ms,1)} ms
sim transition       ${n(s.sim_step_s,4)} s
reference row        ${s.reference_row??'--'}
reward total         ${n(s.episode_reward,1)}
failure              ${s.failure??'none'}

POC attempts          ${(s.poc_results||[]).length}
POC last-10 success   ${s.poc_last10_success??0}/10
POC best gate-4 time ${n(s.poc_best_s,3)} s
threshold status      ${s.poc_threshold_status??'not started'}`;
draw(s);drawTimes(s)}catch(e){}setTimeout(tick,250)}tick();
</script></body></html>"""


class VQ2Dashboard:
    def __init__(self, localizer, map_positions, port: int = 8899,
                 history_path: str | Path | None = None,
                 poc_gate: int = 4, control_hz: float = 30.0) -> None:
        self.localizer = localizer
        self.map_positions = np.asarray(map_positions, float).tolist()
        self.port = int(port)
        self.history_path = Path(history_path) if history_path else None
        self.poc_gate = int(poc_gate)
        self.control_hz = float(control_hz)
        self._lock = threading.Lock()
        self._results: list[dict] = []
        if self.history_path is not None and self.history_path.exists():
            for line in self.history_path.read_text().splitlines():
                try:
                    row = json.loads(line)
                    row.setdefault("run_number", len(self._results))
                    self._results.append(row)
                except json.JSONDecodeError:
                    continue
        self._state: dict = {"map_positions": self.map_positions}
        self._add_results_state(self._state)
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    def update(self, payload: dict) -> None:
        _, debug = self.localizer.dashboard_snapshot()
        state = dict(payload)
        state.update({
            "map_positions": self.map_positions,
            "debug_match_count": len(debug.get("matches", [])),
            "association_radius_px": debug.get("association_radius_px"),
            "visual_consensus": debug.get("visual_consensus"),
            "measurement_sigma_px": debug.get("measurement_sigma_px"),
        })
        self._add_results_state(state)
        with self._lock:
            self._state = state

    def _add_results_state(self, state: dict) -> None:
        times = [r["time_s"] for r in self._results if r.get("time_s") is not None]
        best = min(times) if times else None
        recent = self._results[-10:]
        state.update({
            "poc_results": self._results,
            "poc_best_s": best,
            "poc_last10_success": sum(bool(r.get("success")) for r in recent),
            "poc_threshold_status": (
                "<7.787 ACHIEVED" if best is not None and best < 7.787 else
                "<8.27 ACHIEVED" if best is not None and best < 8.27 else
                "<8.7 ACHIEVED" if best is not None and best < 8.7 else
                "baseline / identification"
            ),
        })

    def record_episode(self, summary: dict) -> None:
        """Persist one POC attempt for restart-safe performance plots."""
        episode = int(summary["episode"])
        crossing = next((
            row for row in summary.get("crossing_offsets", [])
            if int(row.get("gate", -1)) == self.poc_gate
        ), None)
        provenance_fields = (
            "config_sha256", "controller_config_sha256",
            "schedule_sha256", "actor_sha256", "secondary_actor_sha256",
            "reference_sha256", "seed_checkpoint_sha256", "map_sha256",
            "primary_detector_sha256", "refiner_detector_sha256",
            "gate_primary_detector_sha256", "crop_detector_sha256",
            "proposal_model_sha256", "calibration_sha256",
            "line_model_sha256",
        )
        result = {
            "run_number": len(self._results),
            "episode": episode,
            "run_id": summary.get("run_id"),
            "schedule_arm": summary.get("schedule_arm"),
            **{name: summary.get(name) for name in provenance_fields},
            "success": crossing is not None,
            "time_s": (
                float(summary["official_elapsed_s"])
                if summary.get("official_elapsed_s") is not None else
                (int(crossing["step"]) + 1) / self.control_hz
                if crossing is not None else None
            ),
            "failure": summary.get("failure"),
            "deterministic": bool(summary.get("deterministic", False)),
        }
        with self._lock:
            self._results.append(result)
            self._add_results_state(self._state)
        if self.history_path is not None:
            self.history_path.parent.mkdir(parents=True, exist_ok=True)
            with self.history_path.open("a") as stream:
                stream.write(json.dumps(result, separators=(",", ":")) + "\n")

    def _state_bytes(self) -> bytes:
        with self._lock:
            return json.dumps(self._state, separators=(",", ":")).encode()

    def start(self) -> None:
        dashboard = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass

            def do_GET(self):
                if self.path == "/" or self.path.startswith("/?"):
                    body, content_type = _HTML, "text/html; charset=utf-8"
                elif self.path.startswith("/state.json"):
                    body = dashboard._state_bytes()
                    content_type = "application/json"
                else:
                    self.send_response(404)
                    self.end_headers()
                    return
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                try:
                    self.wfile.write(body)
                except (
                    BrokenPipeError,
                    ConnectionAbortedError,
                    ConnectionResetError,
                ):
                    return

        self._server = ThreadingHTTPServer(("127.0.0.1", self.port), Handler)
        self._thread = threading.Thread(
            target=self._server.serve_forever, daemon=True
        )
        self._thread.start()

    def close(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
