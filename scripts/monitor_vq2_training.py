"""Fail-closed watchdog for unattended VQ2 live training.

The watchdog never starts, stops, or restarts the simulator executable.  It
only watches one explicitly named trainer run.  If timing faults recur or the
trainer disappears, it returns the drone to spawn and exits so a bad run
cannot crash-loop overnight.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from aigp.mavlink_io import MavIO  # noqa: E402


def append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, separators=(",", ":")) + "\n")


def trainer_processes(marker: str) -> list[psutil.Process]:
    rows = []
    for process in psutil.process_iter(["pid", "cmdline"]):
        try:
            command_line = process.info["cmdline"] or []
            command = " ".join(command_line)
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            continue
        is_trainer = any(
            Path(argument.strip('"')).name == "train_vq2_sac_live.py"
            for argument in command_line
        )
        if is_trainer and marker in command:
            rows.append(process)
    return rows


def read_episodes(run_root: Path) -> tuple[Path | None, list[dict]]:
    runs = sorted(
        (path for path in run_root.glob("*") if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not runs:
        return None, []
    path = runs[0] / "episodes.jsonl"
    if not path.exists():
        return runs[0], []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return runs[0], rows


def return_drone_to_spawn(port: int) -> str:
    try:
        mavlink = MavIO(port=port)
        for _ in range(6):
            mavlink.arm(False)
            mavlink.reset_sim()
            time.sleep(0.03)
        time.sleep(0.4)
        mavlink.close()
        return "reset_complete"
    except BaseException as error:
        return f"reset_failed:{error!r}"


def stop_trainer_tree(processes: list[psutil.Process]) -> None:
    targets: dict[int, psutil.Process] = {}
    for process in processes:
        try:
            for child in process.children(recursive=True):
                targets[child.pid] = child
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            pass
        targets[process.pid] = process
    ordered = sorted(
        targets.values(),
        key=lambda process: len(process.parents()),
        reverse=True,
    )
    for process in ordered:
        try:
            process.terminate()
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            pass
    _gone, alive = psutil.wait_procs(ordered, timeout=4.0)
    for process in alive:
        try:
            process.kill()
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--trainer-marker", required=True)
    parser.add_argument("--status-log", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=30.0)
    parser.add_argument("--unhealthy-limit", type=int, default=2)
    parser.add_argument("--mav-port", type=int, default=14550)
    args = parser.parse_args()

    seen_running = False
    while True:
        processes = trainer_processes(args.trainer_marker)
        run_dir, episodes = read_episodes(args.run_root)
        unhealthy_streak = 0
        for episode in reversed(episodes):
            if episode.get("timing_healthy", False):
                break
            unhealthy_streak += 1
        payload = {
            "time_utc": datetime.now(timezone.utc).isoformat(),
            "trainer_pids": sorted({process.pid for process in processes}),
            "run_dir": str(run_dir) if run_dir else None,
            "episode_count": len(episodes),
            "last_episode": episodes[-1] if episodes else None,
            "recent_gates": [
                int(row.get("gate_reached", -1)) for row in episodes[-10:]
            ],
            "unhealthy_streak": unhealthy_streak,
        }
        append_jsonl(args.status_log, payload)

        if processes:
            seen_running = True
        elif seen_running:
            payload["action"] = return_drone_to_spawn(args.mav_port)
            payload["alert"] = "trainer_exited"
            append_jsonl(args.status_log, payload)
            return

        if (
            processes
            and unhealthy_streak >= max(1, args.unhealthy_limit)
        ):
            stop_trainer_tree(processes)
            time.sleep(0.5)
            payload["action"] = return_drone_to_spawn(args.mav_port)
            payload["alert"] = "consecutive_timing_faults"
            append_jsonl(args.status_log, payload)
            return
        time.sleep(max(5.0, args.interval))


if __name__ == "__main__":
    main()
