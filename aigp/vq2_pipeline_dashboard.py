"""Persistent read-only dashboard for the VQ2 live-data flywheel."""

from __future__ import annotations

import json
import math
import re
import subprocess
import threading
import time
from collections import Counter
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


_VERSION = re.compile(r"v(\d+)")
_STAMP = re.compile(r"(20\d{6}_\d{6})")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        return rows
    for line in lines:
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def _version(path: Path) -> int:
    match = _VERSION.search(path.parent.name)
    return int(match.group(1)) if match else -1


def _iso(mtime: float | None) -> str | None:
    if mtime is None:
        return None
    return datetime.fromtimestamp(mtime).astimezone().isoformat(timespec="seconds")


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


class PipelineScanner:
    """Build a restart-safe dashboard snapshot from immutable artifacts."""

    def __init__(
        self,
        repo: Path,
        data_root: Path,
        training_root: Path,
        worldmodel_root: Path,
        manifest_root: Path,
    ) -> None:
        self.repo = repo
        self.data_root = data_root
        self.training_root = training_root
        self.worldmodel_root = worldmodel_root
        self.manifest_root = manifest_root
        self._episode_cache: dict[Path, tuple[int, int, list[dict[str, Any]]]] = {}

    def _episodes(self) -> list[dict[str, Any]]:
        paths = list(self.training_root.rglob("episodes.jsonl"))
        active = set(paths)
        for path in list(self._episode_cache):
            if path not in active:
                del self._episode_cache[path]
        rows: list[dict[str, Any]] = []
        for path in paths:
            try:
                stat = path.stat()
            except OSError:
                continue
            cached = self._episode_cache.get(path)
            identity = (stat.st_mtime_ns, stat.st_size)
            if cached is None or cached[:2] != identity:
                parsed = _read_jsonl(path)
                session = path.parent
                stamp_match = _STAMP.search(str(session))
                stamp = stamp_match.group(1) if stamp_match else session.name
                for row in parsed:
                    row["_session"] = str(session)
                    row["_stamp"] = stamp
                    row["_mtime"] = stat.st_mtime
                self._episode_cache[path] = (*identity, parsed)
            rows.extend(self._episode_cache[path][2])
        return sorted(rows, key=lambda row: (
            str(row.get("_stamp", "")), int(row.get("episode", -1))
        ))

    def _latest(self, pattern: str, *, by_version: bool = False) -> Path | None:
        paths = list(self.worldmodel_root.glob(pattern))
        if not paths:
            return None
        if by_version:
            return max(paths, key=lambda path: (_version(path), path.stat().st_mtime))
        return max(paths, key=lambda path: path.stat().st_mtime)

    def _git_commit_time(self) -> float | None:
        try:
            value = subprocess.run(
                ["git", "log", "-1", "--format=%ct"],
                cwd=self.repo,
                check=True,
                capture_output=True,
                text=True,
                timeout=2,
            ).stdout.strip()
            return float(value)
        except (OSError, ValueError, subprocess.SubprocessError):
            return None

    def _state_machine(
        self,
        latest_episode_mtime: float | None,
        live_activity_mtime: float | None,
        registry: Path | None,
        model: Path | None,
        audit: Path | None,
        ledger: Path | None,
    ) -> tuple[str, list[dict[str, Any]]]:
        now = time.time()
        explicit = _read_json(self.data_root / "pipeline_state.json")
        explicit_time = _finite(explicit.get("updated_unix_s"))
        active = None
        detail = None
        if explicit_time is not None and now - explicit_time < 180:
            active = str(explicit.get("active_stage", "")).lower() or None
            detail = explicit.get("message")
        elif live_activity_mtime is not None and now - live_activity_mtime < 15:
            active = "collect"
            detail = "A live flight is writing synchronized transitions."

        commit_time = self._git_commit_time()
        artifacts = {
            "collect": latest_episode_mtime,
            "ingest": registry.stat().st_mtime if registry else None,
            "train": model.stat().st_mtime if model else None,
            "audit": audit.stat().st_mtime if audit else None,
            "decide": ledger.stat().st_mtime if ledger else None,
            "publish": commit_time,
        }
        ordered_keys = [
            "collect", "ingest", "train", "audit", "decide", "publish"
        ]
        completed: dict[str, bool] = {}
        previous_time: float | None = None
        for key in ordered_keys:
            timestamp = artifacts[key]
            completed[key] = bool(
                timestamp is not None
                and (previous_time is None or timestamp >= previous_time)
            )
            if not completed[key]:
                previous_time = math.inf
            else:
                previous_time = timestamp
        definitions = [
            ("collect", "Collect flights", "Capture complete healthy and failed episodes with camera, MAVLink, actions, and timing."),
            ("ingest", "Ingest + quarantine", "Add healthy sessions to immutable train splits while excluding simulator timing faults."),
            ("train", "Train world model", "Fit the dynamics ensemble on Gipsydanger without competing with the live simulator."),
            ("audit", "Frozen audit", "Compare the new model against protected families on never-trained rollout starts."),
            ("decide", "Decide promotion", "Keep, reject, or scope the candidate using live reliability and audit evidence."),
            ("publish", "Freeze + publish", "Version the exact result, hashes, launcher settings, and protected champion state."),
        ]
        stages: list[dict[str, Any]] = []
        for key, label, explanation in definitions:
            timestamp = artifacts[key]
            status = (
                "active" if key == active else
                "complete" if completed[key] else "pending"
            )
            stages.append({
                "key": key,
                "label": label,
                "status": status,
                "explanation": explanation,
                "timestamp": _iso(timestamp),
                "detail": detail if key == active else None,
            })
        if active:
            overall = "Running: " + next(
                stage["label"] for stage in stages if stage["status"] == "active"
            )
        elif all(stage["status"] == "complete" for stage in stages):
            overall = "Cycle complete — ready for the next collection block"
        else:
            waiting = next(
                stage["label"] for stage in stages if stage["status"] == "pending"
            )
            overall = f"Waiting: {waiting}"
        return overall, stages

    def snapshot(self) -> dict[str, Any]:
        episodes = self._episodes()
        healthy = [row for row in episodes if row.get("timing_healthy") is not False]
        recent = healthy[-20:]
        finishes = [
            row for row in healthy
            if bool(row.get("finished"))
            and (official := _finite(row.get("official_elapsed_s"))) is not None
            and 20.0 <= official <= 120.0
        ]
        fastest = min(finishes, key=lambda row: float(row["official_elapsed_s"])) if finishes else None
        latest_finish = finishes[-1] if finishes else None

        registry_paths = list(self.manifest_root.glob("vq2_splits_v*.json"))
        registry = max(registry_paths, key=lambda path: path.stat().st_mtime) if registry_paths else None
        corpus_paths = list(self.manifest_root.glob("vq2_canonical_*.json"))
        corpus = max(corpus_paths, key=lambda path: path.stat().st_mtime) if corpus_paths else None
        dataset_manifest = self._latest(
            "g0g16_master_currentera_v*_registry/manifest.json", by_version=True
        )
        model = self._latest(
            "v*_allgate_registry_flywheel/residual_ensemble_v*.pt", by_version=True
        )
        audit = self._latest(
            "v*_allgate_registry_flywheel/fresh_registry_audit_h32.json", by_version=True
        )
        ledger_paths = list((self.repo / "data").glob("vq2_flywheel_cycle_*.json"))
        ledger = max(ledger_paths, key=lambda path: path.stat().st_mtime) if ledger_paths else None

        dataset = _read_json(dataset_manifest) if dataset_manifest else {}
        train_transitions = dataset.get("splits", {}).get("train", {}).get("transitions")
        corpus_count = _read_json(corpus).get("entry_count") if corpus else None
        audit_payload = _read_json(audit) if audit else {}
        model_name = model.parent.name.split("_", 1)[0] if model else None
        model_metrics = audit_payload.get("overall", {}).get(model_name, {})
        position = model_metrics.get("position_m", {})

        recent_finish_count = sum(bool(row.get("finished")) for row in recent)
        last_raw = episodes[-20:]
        healthy_count = sum(row.get("timing_healthy") is not False for row in last_raw)
        late_reach = sum(int(row.get("gate_reached", -1)) >= 16 for row in recent)
        metrics = [
            {
                "label": "Fastest official lap",
                "value": round(float(fastest["official_elapsed_s"]), 3) if fastest else None,
                "unit": "s",
                "explanation": "The lowest timing-healthy official full-course result preserved in the flight archive.",
            },
            {
                "label": "Last-20 completion rate",
                "value": round(100.0 * recent_finish_count / len(recent), 1) if recent else None,
                "unit": "%",
                "explanation": "The share of the latest 20 timing-healthy flights that passed all 17 gates.",
            },
            {
                "label": "Last-20 timing health",
                "value": round(100.0 * healthy_count / len(last_raw), 1) if last_raw else None,
                "unit": "%",
                "explanation": "The share of recent flights whose simulator and control-loop timing stayed inside quarantine limits.",
            },
            {
                "label": "Late-course reach",
                "value": f"{late_reach}/{len(recent)}" if recent else None,
                "unit": "",
                "explanation": "How many recent healthy flights reached gate 16 or finished, exposing the sections that matter most now.",
            },
            {
                "label": "Healthy training transitions",
                "value": int(train_transitions) if train_transitions is not None else None,
                "unit": "rows",
                "explanation": "Real synchronized state-action transitions currently eligible for dynamics training after quarantine.",
            },
            {
                "label": f"{model_name or 'Latest model'} position p90",
                "value": round(float(position["p90"]), 3) if position.get("p90") is not None else None,
                "unit": "m @ 1.07 s",
                "explanation": "Ninety percent of frozen 32-step model rollouts end within this position error.",
            },
        ]

        timeline: list[dict[str, Any]] = []
        best = math.inf
        for index, row in enumerate(finishes):
            official = float(row["official_elapsed_s"])
            best = min(best, official)
            timeline.append({
                "run": index + 1,
                "time_s": round(official, 6),
                "best_s": round(best, 6),
                "arm": row.get("schedule_arm") or row.get("probe_arm") or "unknown",
                "stamp": row.get("_stamp"),
                "episode": row.get("episode"),
            })

        failures = Counter()
        for row in healthy[-50:]:
            if not row.get("finished"):
                gate = int(row.get("gate_reached", -1))
                failures[f"Gate {gate}" if gate >= 0 else "Unknown"] += 1
        gate_failures = [
            {"gate": gate, "count": count}
            for gate, count in sorted(
                failures.items(), key=lambda item: int(item[0].split()[-1]) if item[0].startswith("Gate ") else 99
            )
        ]

        recent_rows = []
        for row in reversed(episodes[-12:]):
            recent_rows.append({
                "stamp": row.get("_stamp"),
                "episode": row.get("episode"),
                "arm": row.get("schedule_arm") or row.get("probe_arm") or "—",
                "gate": row.get("gate_reached"),
                "result": "FINISH" if row.get("finished") else row.get("failure") or "stopped",
                "official_s": _finite(row.get("official_elapsed_s")),
                "timing_healthy": row.get("timing_healthy") is not False,
            })

        latest_episode_mtime = max(
            (_finite(row.get("_mtime")) for row in episodes), default=None
        )
        activity_paths = list(self.training_root.rglob("steps.jsonl"))
        activity_mtimes = [
            path.stat().st_mtime
            for path in activity_paths
            if path.is_file()
        ]
        live_activity_mtime = max(
            [value for value in [latest_episode_mtime, *activity_mtimes]
             if value is not None],
            default=None,
        )
        overall_state, stages = self._state_machine(
            latest_episode_mtime, live_activity_mtime,
            registry, model, audit, ledger
        )
        return {
            "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "pipeline_state": overall_state,
            "stages": stages,
            "metrics": metrics,
            "timeline": timeline[-80:],
            "gate_failures": gate_failures,
            "recent": recent_rows,
            "context": {
                "latest_finish_s": _finite(latest_finish.get("official_elapsed_s")) if latest_finish else None,
                "fastest_session": fastest.get("_session") if fastest else None,
                "fastest_episode": fastest.get("episode") if fastest else None,
                "model": model_name,
                "corpus_artifacts": corpus_count,
                "registry": registry.name if registry else None,
                "dataset": dataset_manifest.parent.name if dataset_manifest else None,
            },
        }


_HTML = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>VQ2 Pipeline Control</title><style>
:root{color-scheme:dark;--bg:#070a0f;--panel:#101722;--line:#263246;--text:#edf3fb;
--muted:#94a3b8;--orange:#ff8a32;--green:#50e3a4;--red:#ff6277;--blue:#5cb8ff;--amber:#ffcf66}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 90% -10%,#182337 0,transparent 32%),var(--bg);color:var(--text);font:14px Inter,ui-sans-serif,system-ui,sans-serif}
.shell{max-width:1500px;margin:auto;padding:18px}.top{display:flex;justify-content:space-between;align-items:end;gap:16px;margin-bottom:16px}.eyebrow{color:var(--orange);font-size:12px;letter-spacing:.15em;text-transform:uppercase}.top h1{font-size:27px;margin:4px 0 0;font-weight:500}.state{text-align:right}.state strong{display:block;color:var(--green);font-weight:500}.muted{color:var(--muted)}
.metrics{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px}.card{background:linear-gradient(145deg,rgba(18,26,39,.96),rgba(12,18,28,.96));border:1px solid var(--line);border-radius:10px;padding:14px}.metric .label{color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.06em}.metric .value{font-size:28px;margin:7px 0 4px;font-variant-numeric:tabular-nums}.metric .unit{font-size:13px;color:var(--muted);margin-left:5px}.metric p,.stage p{color:var(--muted);margin:0;line-height:1.4;font-size:12px}
.section{margin-top:18px}.section h2{font-size:15px;font-weight:500;margin:0 0 10px}.machine{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:8px}.stage{position:relative;min-height:126px}.stage .num{width:24px;height:24px;border-radius:50%;display:grid;place-items:center;background:#1b2637;color:var(--muted);margin-bottom:9px}.stage.complete .num{background:rgba(80,227,164,.16);color:var(--green)}.stage.active{border-color:var(--orange)}.stage.active .num{background:var(--orange);color:#111}.stage h3{font-size:13px;margin:0 0 6px;font-weight:500}.status{font-size:11px;text-transform:uppercase;letter-spacing:.08em;margin-top:9px;color:var(--muted)}.complete .status{color:var(--green)}.active .status{color:var(--orange)}
.charts{display:grid;grid-template-columns:2fr 1fr;gap:10px}.chart{min-height:320px}.chart svg{width:100%;height:275px;overflow:visible}.axis{stroke:var(--line);stroke-width:1}.gridline{stroke:var(--line);stroke-width:1;opacity:.7}.bestline{fill:none;stroke:var(--green);stroke-width:2.5}.point{fill:var(--blue);opacity:.72}.bar{fill:var(--orange);opacity:.8}.chart-label{fill:var(--muted);font-size:11px}.best-label{fill:var(--green);font-size:11px}
.table-wrap{overflow:auto;border:1px solid var(--line);border-radius:10px}table{width:100%;border-collapse:collapse;background:rgba(14,21,32,.8)}th,td{padding:9px 11px;text-align:left;border-bottom:1px solid var(--line);white-space:nowrap}th{color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.06em;font-weight:500}td{font-variant-numeric:tabular-nums}.good{color:var(--green)}.bad{color:var(--red)}
@media(max-width:900px){.metrics{grid-template-columns:repeat(2,1fr)}.machine{grid-template-columns:repeat(2,1fr)}.charts{grid-template-columns:1fr}.top{align-items:start;flex-direction:column}.state{text-align:left}}
@media(max-width:520px){.metrics,.machine{grid-template-columns:1fr}.shell{padding:12px}}
</style></head><body><main class="shell"><header class="top"><div><div class="eyebrow">AI Grand Prix · VQ2</div><h1>Training Flywheel</h1></div><div class="state"><strong id="state">Loading pipeline…</strong><span class="muted" id="updated"></span></div></header>
<section class="metrics" id="metrics"></section><section class="section"><h2>Pipeline state machine</h2><div class="machine" id="machine"></div></section>
<section class="section charts"><div class="card chart"><h2>Fastest official time vs completed run</h2><svg id="times" role="img" aria-label="Official completion time and running record"></svg></div><div class="card chart"><h2>Failure gates · latest 50 healthy flights</h2><svg id="failures" role="img" aria-label="Count of failures by gate"></svg></div></section>
<section class="section"><h2>Recent flights</h2><div class="table-wrap"><table><thead><tr><th>Session</th><th>Episode</th><th>Arm</th><th>Reached</th><th>Result</th><th>Official</th><th>Timing</th></tr></thead><tbody id="recent"></tbody></table></div></section></main>
<script>
const ns='http://www.w3.org/2000/svg';function fmt(v,d=1){return v==null?'—':Number(v).toLocaleString(undefined,{maximumFractionDigits:d,minimumFractionDigits:d})}function el(n,a={}){let x=document.createElementNS(ns,n);Object.entries(a).forEach(([k,v])=>x.setAttribute(k,v));return x}
function metrics(rows){document.getElementById('metrics').innerHTML=rows.map(m=>`<article class="card metric"><div class="label">${m.label}</div><div class="value">${typeof m.value==='number'?m.value.toLocaleString():m.value??'—'}<span class="unit">${m.unit}</span></div><p>${m.explanation}</p></article>`).join('')}
function machine(rows){document.getElementById('machine').innerHTML=rows.map((s,i)=>`<article class="card stage ${s.status}"><div class="num">${i+1}</div><h3>${s.label}</h3><p>${s.explanation}</p><div class="status">${s.status}${s.timestamp?' · '+s.timestamp.slice(11,19):''}</div></article>`).join('')}
function lineChart(rows){let svg=document.getElementById('times'),W=760,H=260,p={l:46,r:22,t:18,b:34};svg.setAttribute('viewBox',`0 0 ${W} ${H}`);svg.innerHTML='';if(!rows.length)return;let lo=Math.min(...rows.map(r=>r.time_s))-.4,hi=Math.max(...rows.map(r=>r.time_s))+.4;let X=i=>p.l+i/Math.max(rows.length-1,1)*(W-p.l-p.r),Y=v=>p.t+(hi-v)/(hi-lo)*(H-p.t-p.b);[lo,(lo+hi)/2,hi].forEach(v=>{let y=Y(v);svg.append(el('line',{x1:p.l,x2:W-p.r,y1:y,y2:y,class:'gridline'}));let t=el('text',{x:4,y:y+4,class:'chart-label'});t.textContent=v.toFixed(1)+'s';svg.append(t)});let d=rows.map((r,i)=>(i?'L':'M')+X(i)+' '+Y(r.best_s)).join(' ');svg.append(el('path',{d,class:'bestline'}));rows.forEach((r,i)=>{let c=el('circle',{cx:X(i),cy:Y(r.time_s),r:3.2,class:'point'});let tt=el('title');tt.textContent=`${r.time_s.toFixed(3)}s · ${r.arm} · ${r.stamp} ep ${r.episode}`;c.append(tt);svg.append(c)});let last=rows[rows.length-1],t=el('text',{x:W-p.r-4,y:Y(last.best_s)-8,'text-anchor':'end',class:'best-label'});t.textContent='record '+last.best_s.toFixed(3)+'s';svg.append(t)}
function bars(rows){let svg=document.getElementById('failures'),W=420,H=260,p={l:34,r:10,t:18,b:38};svg.setAttribute('viewBox',`0 0 ${W} ${H}`);svg.innerHTML='';if(!rows.length)return;let max=Math.max(...rows.map(r=>r.count),1),bw=(W-p.l-p.r)/rows.length;rows.forEach((r,i)=>{let h=r.count/max*(H-p.t-p.b),x=p.l+i*bw+bw*.15,y=H-p.b-h;svg.append(el('rect',{x,y,width:bw*.7,height:h,rx:2,class:'bar'}));let t=el('text',{x:x+bw*.35,y:H-p.b+16,'text-anchor':'middle',class:'chart-label'});t.textContent=r.gate.replace('Gate ','G');svg.append(t);let v=el('text',{x:x+bw*.35,y:y-5,'text-anchor':'middle',class:'chart-label'});v.textContent=r.count;svg.append(v)})}
function recent(rows){document.getElementById('recent').innerHTML=rows.map(r=>`<tr><td>${r.stamp??'—'}</td><td>${r.episode??'—'}</td><td>${r.arm}</td><td>Gate ${r.gate??'—'}</td><td class="${r.result==='FINISH'?'good':'bad'}">${r.result}</td><td>${r.official_s==null?'—':fmt(r.official_s,3)+' s'}</td><td class="${r.timing_healthy?'good':'bad'}">${r.timing_healthy?'healthy':'quarantined'}</td></tr>`).join('')}
async function tick(){try{let s=await(await fetch('/api/state?'+Date.now())).json();document.getElementById('state').textContent=s.pipeline_state;document.getElementById('updated').textContent='Updated '+s.generated_at;metrics(s.metrics);machine(s.stages);lineChart(s.timeline);bars(s.gate_failures);recent(s.recent)}catch(e){document.getElementById('state').textContent='Dashboard data unavailable'}setTimeout(tick,2000)}tick();
</script></body></html>""".encode("utf-8")


class PipelineDashboardServer:
    def __init__(
        self,
        scanner: PipelineScanner,
        host: str = "127.0.0.1",
        port: int = 8900,
        refresh_s: float = 2.0,
    ) -> None:
        self.scanner = scanner
        self.host = host
        self.port = int(port)
        self.refresh_s = float(refresh_s)
        self._state: dict[str, Any] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._scan_thread: threading.Thread | None = None

    def _scan_loop(self) -> None:
        while not self._stop.is_set():
            try:
                state = self.scanner.snapshot()
                with self._lock:
                    self._state = state
            except Exception as exc:  # keep diagnostics available on bad files
                with self._lock:
                    self._state = {"error": f"{type(exc).__name__}: {exc}"}
            self._stop.wait(self.refresh_s)

    def serve_forever(self) -> None:
        dashboard = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args: Any) -> None:
                return

            def do_GET(self) -> None:
                if self.path == "/" or self.path.startswith("/?"):
                    body, kind = _HTML, "text/html; charset=utf-8"
                elif self.path.startswith("/api/state"):
                    with dashboard._lock:
                        body = json.dumps(dashboard._state, separators=(",", ":")).encode()
                    kind = "application/json"
                else:
                    self.send_response(404)
                    self.end_headers()
                    return
                self.send_response(200)
                self.send_header("Content-Type", kind)
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                try:
                    self.wfile.write(body)
                except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
                    return

        self._scan_thread = threading.Thread(target=self._scan_loop, daemon=True)
        self._scan_thread.start()
        server = ThreadingHTTPServer((self.host, self.port), Handler)
        try:
            server.serve_forever()
        finally:
            self._stop.set()
            server.server_close()
            if self._scan_thread is not None:
                self._scan_thread.join(timeout=2)
