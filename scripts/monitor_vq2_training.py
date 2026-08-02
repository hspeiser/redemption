"""Persistent watchdog for unattended VQ2 live training.

The watchdog never starts, stops, or restarts the simulator executable. It
watches the trainer process, publishes an atomic status snapshot, records
state changes, and raises a visible Windows alert whenever training exits or
timing becomes unhealthy. In persistent mode it remains alive across trainer
restarts, so supervision cannot silently disappear with the process it was
watching.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from aigp.mavlink_io import MavIO  # noqa: E402


def acquire_single_instance_guard():
    """Ensure one persistent watchdog owns notifications and state files."""
    if os.name != "nt":
        return None
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateMutexW.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_wchar_p,
    ]
    kernel32.CreateMutexW.restype = ctypes.c_void_p
    handle = kernel32.CreateMutexW(
        None, False, "Local\\AIGP_VQ2_TRAINING_WATCHDOG"
    )
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    if ctypes.get_last_error() == 183:
        kernel32.CloseHandle(handle)
        raise RuntimeError("the VQ2 training watchdog is already running")
    return handle


def append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, separators=(",", ":")) + "\n")


def write_json(path: Path, payload: dict) -> None:
    """Atomically publish the latest state for dashboards and future agents."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def notify_windows(message: str) -> str:
    """Show a desktop-visible alert without requiring an optional package."""
    if os.name != "nt":
        return "notification_unsupported"
    try:
        result = subprocess.run(
            ["msg.exe", "*", "/TIME:20", message[:900]],
            capture_output=True,
            text=True,
            timeout=5.0,
            check=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if result.returncode == 0:
            return "notification_sent"
        detail = (result.stderr or result.stdout).strip()
        return f"notification_failed:{result.returncode}:{detail}"
    except BaseException as error:
        return f"notification_failed:{error!r}"


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
            and (
                argument_index == 0
                or command_line[argument_index - 1] != "--trainer-marker"
            )
            for argument_index, argument in enumerate(command_line)
        )
        if is_trainer and marker in command:
            rows.append(process)
    return rows


def trainer_output_roots(
    processes: list[psutil.Process],
) -> list[Path]:
    roots: dict[str, Path] = {}
    for process in processes:
        try:
            command_line = process.cmdline()
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            continue
        for index, argument in enumerate(command_line[:-1]):
            if argument == "--output-root":
                root = Path(command_line[index + 1].strip('"'))
                roots[str(root).lower()] = root
    return list(roots.values())


def read_episodes(run_root: Path) -> tuple[Path | None, list[dict]]:
    """Find the newest run under either one campaign or the global run root."""
    episode_logs = list(run_root.glob("*/episodes.jsonl"))
    if not episode_logs:
        episode_logs = list(run_root.glob("*/*/episodes.jsonl"))
    if not episode_logs:
        runs = [path for path in run_root.glob("*") if path.is_dir()]
        if runs:
            return max(runs, key=lambda path: path.stat().st_mtime), []
        return None, []
    path = max(episode_logs, key=lambda candidate: candidate.stat().st_mtime)
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return path.parent, rows


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
    parser.add_argument("--status-file", type=Path, default=None)
    parser.add_argument("--alert-file", type=Path, default=None)
    parser.add_argument("--interval", type=float, default=5.0)
    parser.add_argument(
        "--unhealthy-limit",
        type=int,
        default=1,
        help="Alert after this many consecutive unhealthy episodes; zero disables.",
    )
    parser.add_argument("--mav-port", type=int, default=14550)
    parser.add_argument(
        "--persistent",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Remain alive and watch later trainer restarts.",
    )
    parser.add_argument(
        "--notify-windows",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--reset-on-exit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Return only the drone state to spawn when the trainer disappears.",
    )
    parser.add_argument(
        "--kill-on-unhealthy",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Let the trainer's own timing guard stop it unless explicitly enabled.",
    )
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    single_instance_guard = acquire_single_instance_guard()

    status_file = args.status_file or args.status_log.with_name(
        "watchdog_status.json"
    )
    alert_file = args.alert_file or args.status_log.with_name(
        "watchdog_alert.json"
    )
    previous_pids: set[int] = set()
    alerted_unhealthy_key: tuple[str | None, int] | None = None
    while True:
        processes = trainer_processes(args.trainer_marker)
        output_roots = trainer_output_roots(processes)
        watched_root = (
            next(
                (path for path in output_roots if path.exists()),
                output_roots[0],
            )
            if output_roots else args.run_root
        )
        run_dir, episodes = read_episodes(watched_root)
        unhealthy_streak = 0
        for episode in reversed(episodes):
            if episode.get("timing_healthy", False):
                break
            unhealthy_streak += 1
        current_pids = {process.pid for process in processes}
        payload = {
            "time_utc": datetime.now(timezone.utc).isoformat(),
            "watchdog_pid": os.getpid(),
            "state": "running" if current_pids else "stopped",
            "trainer_pids": sorted(current_pids),
            "watched_root": str(watched_root),
            "run_dir": str(run_dir) if run_dir else None,
            "episode_count": len(episodes),
            "last_episode": episodes[-1] if episodes else None,
            "recent_gates": [
                int(row.get("gate_reached", -1)) for row in episodes[-10:]
            ],
            "unhealthy_streak": unhealthy_streak,
        }
        write_json(status_file, payload)
        append_jsonl(args.status_log, payload)

        if current_pids and not previous_pids:
            started = {
                **payload,
                "event": "trainer_started",
            }
            append_jsonl(args.status_log, started)
            alerted_unhealthy_key = None
        elif previous_pids and not current_pids:
            alert = {
                **payload,
                "event": "trainer_exited",
                "alert": "trainer_exited",
            }
            if unhealthy_streak:
                alert["reason"] = "timing_unhealthy"
            elif episodes:
                alert["reason"] = (
                    episodes[-1].get("failure") or "process_exit"
                )
            if args.reset_on_exit:
                alert["action"] = return_drone_to_spawn(args.mav_port)
            if args.notify_windows:
                alert["notification"] = notify_windows(
                    "AI-GP TRAINER STOPPED\n"
                    f"reason: {alert.get('reason', 'unknown')}\n"
                    f"run: {run_dir}"
                )
            write_json(alert_file, alert)
            write_json(status_file, alert)
            append_jsonl(args.status_log, alert)
            if not args.persistent:
                return

        if (
            processes
            and args.unhealthy_limit > 0
            and unhealthy_streak >= max(1, args.unhealthy_limit)
        ):
            unhealthy_key = (
                str(run_dir) if run_dir else None,
                len(episodes),
            )
            if alerted_unhealthy_key != unhealthy_key:
                alert = {
                    **payload,
                    "event": "timing_unhealthy",
                    "alert": "consecutive_timing_faults",
                }
                if args.notify_windows:
                    alert["notification"] = notify_windows(
                        "AI-GP TIMING WARNING\n"
                        f"unhealthy episodes: {unhealthy_streak}\n"
                        f"run: {run_dir}"
                    )
                write_json(alert_file, alert)
                append_jsonl(args.status_log, alert)
                alerted_unhealthy_key = unhealthy_key
            if args.kill_on_unhealthy:
                stop_trainer_tree(processes)
        if args.once:
            return
        previous_pids = current_pids
        time.sleep(max(5.0, args.interval))


if __name__ == "__main__":
    main()
