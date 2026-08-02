"""Export real VQ2 gate-segment sequences for recurrent BC + offline RL.

Successful crossings provide actor supervision.  Healthy failures remain
critic-only evidence.  Infrastructure failures are indexed in the source
corpus but excluded from physical policy targets.  Every split is session
level and comes from the immutable split registry.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


STAMP = re.compile(r"(20\d{6}_\d{6})")
INFRA_FAILURES = {
    "stale_sensor_stream", "sim_clock_stalled", "simulator_respawn",
    "localizer_exception", "invalid_observation", "invalid_control_stream",
}
SPLITS = ("train", "validation", "policy_selection", "final_test")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def timestamp_key(path: Path, host: str = "local") -> str | None:
    match = STAMP.search(str(path))
    return f"{host.lower()}:{match.group(1)}" if match else None


def registry_map(path: Path) -> dict[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        str(row["group_key"]): str(row["split"])
        for row in payload.get("groups", [])
    }


def summaries(path: Path) -> dict[int, dict]:
    result = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return result
    for line in lines:
        try:
            row = json.loads(line)
            result[int(row["episode"])] = row
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            continue
    return result


def segment_bounds(gate: np.ndarray, focus: int, context: int) -> tuple[int, int] | None:
    active = np.flatnonzero(gate == focus)
    if not len(active):
        return None
    start = max(0, int(active[0]) - context)
    later = np.flatnonzero((np.arange(len(gate)) > active[-1]) & (gate > focus))
    end = min(len(gate), int(later[0]) + context) if len(later) else len(gate)
    return start, max(end, int(active[-1]) + 1)


def n_step_targets(
    reward: np.ndarray,
    done: np.ndarray,
    next_observation: np.ndarray,
    gamma: float,
    n_step: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    count = len(reward)
    total = np.zeros(count, np.float32)
    terminal = np.zeros(count, np.float32)
    discount = np.ones(count, np.float32)
    following = np.empty_like(next_observation)
    for start in range(count):
        carry = 0.0
        factor = 1.0
        last = start
        steps = 0
        for index in range(start, min(count, start + n_step)):
            carry += factor * float(reward[index])
            last = index
            steps += 1
            if done[index] > 0.5:
                terminal[start] = 1.0
                break
            factor *= gamma
        total[start] = carry
        discount[start] = gamma ** steps
        following[start] = next_observation[last]
    return total, terminal, discount, following


def append_chunk(chunks: dict[str, list[np.ndarray]], chunk: dict[str, np.ndarray]) -> None:
    for name, value in chunk.items():
        chunks[name].append(np.asarray(value))


def local_chunks(args, split_lookup: dict[str, str]):
    for journal in sorted(args.training_root.glob("**/episodes.jsonl")):
        key = timestamp_key(journal.parent)
        split = split_lookup.get(key or "", "train")
        if split not in SPLITS:
            continue
        for episode_index, summary in sorted(summaries(journal).items()):
            path = journal.parent / f"episode_{episode_index:04d}.npz"
            if not path.exists() or not bool(summary.get("timing_healthy", False)):
                continue
            failure = str(summary.get("failure") or "")
            if failure in INFRA_FAILURES:
                continue
            try:
                data = np.load(path, allow_pickle=False)
                required = {
                    "observation", "action", "reward", "next_observation",
                    # actor_mean marks the modern residual-action schema.
                    # teacher_action was added later and is not required to
                    # interpret the already-canonical residual in action.
                    "done", "gate_index", "actor_mean",
                }
                if not required.issubset(data.files):
                    continue
                observation = np.asarray(data["observation"], np.float32)
                action = np.asarray(data["action"], np.float32)
                reward = np.asarray(data["reward"], np.float32)
                next_observation = np.asarray(data["next_observation"], np.float32)
                done = np.asarray(data["done"], np.float32)
                gate = np.asarray(data["gate_index"], np.int16)
            except (OSError, KeyError, ValueError):
                continue
            if observation.ndim != 2 or observation.shape[1] != 53 or action.shape != (len(observation), 4):
                continue
            bounds = segment_bounds(gate, args.focus_gate, args.context_steps)
            if bounds is None:
                continue
            start, end = bounds
            sl = slice(start, end)
            crossed = (
                int(summary.get("gate_reached", -1)) > args.focus_gate
                or any(
                    int(row.get("gate", -1)) == args.focus_gate
                    for row in summary.get("crossing_offsets", [])
                )
            )
            focused_gate = gate[sl]
            actor_weight = np.zeros(len(focused_gate), np.float32)
            if crossed:
                boost = 4.0 if bool(summary.get("finished")) else 2.0
                if args.fast_token and args.fast_token in str(path):
                    boost = max(boost, args.fast_boost)
                actor_weight[focused_gate == args.focus_gate] = boost
            target_reward, target_done, discount, target_next = n_step_targets(
                reward[sl], done[sl], next_observation[sl], args.gamma, args.n_step,
            )
            yield split, {
                "observation": observation[sl],
                "action": action[sl],
                "reward": target_reward,
                "next_observation": target_next,
                "done": target_done,
                "discount": discount,
                "gate_index": focused_gate,
                "step": np.arange(start, end, dtype=np.int32),
                "actor_weight": actor_weight,
                "critic_weight": np.ones(len(focused_gate), np.float32),
                "outcome": np.full(
                    len(focused_gate), 0 if crossed else 1, np.int8
                ),
                "source": np.full(len(focused_gate), str(path), dtype="U512"),
            }, {
                "path": str(path), "split": split, "rows": end - start,
                "crossed_focus_gate": crossed,
                "finished": bool(summary.get("finished")),
                "failure": failure or None,
            }


def expert_chunks(args):
    if args.expert is None:
        return
    data = np.load(args.expert, allow_pickle=False)
    source = np.asarray(data["source_episode"]).astype(str)
    for name in dict.fromkeys(source.tolist()):
        row = np.flatnonzero(source == name)
        gate = np.asarray(data["gate_index"][row], np.int16)
        bounds = segment_bounds(gate, args.focus_gate, args.context_steps)
        if bounds is None:
            continue
        start, end = bounds
        selected = row[start:end]
        split_value = int(hashlib.sha256(name.encode()).hexdigest()[:8], 16) / 16**8
        split = "validation" if split_value < 0.15 else "train"
        reward = np.asarray(data["reward"][selected], np.float32)
        done = np.asarray(data["done"][selected], np.float32)
        next_observation = np.asarray(data["next_observation"][selected], np.float32)
        if "discount" in data.files:
            discount = np.asarray(data["discount"][selected], np.float32)
            target_reward, target_done, target_next = reward, done, next_observation
        else:
            target_reward, target_done, discount, target_next = n_step_targets(
                reward, done, next_observation, args.gamma, args.n_step,
            )
        focused_gate = gate[start:end]
        yield split, {
            "observation": np.asarray(data["observation"][selected], np.float32),
            "action": np.zeros((len(selected), 4), np.float32),
            "reward": target_reward,
            "next_observation": target_next,
            "done": target_done,
            "discount": discount,
            "gate_index": focused_gate,
            "step": np.arange(start, end, dtype=np.int32),
            "actor_weight": (focused_gate == args.focus_gate).astype(np.float32),
            "critic_weight": np.ones(len(selected), np.float32),
            "outcome": np.zeros(len(selected), np.int8),
            "source": np.full(
                len(selected), f"{args.expert}::{name}", dtype="U512"
            ),
        }, {
            "path": f"{args.expert}::{name}", "split": split,
            "rows": len(selected), "crossed_focus_gate": True,
            "finished": True, "failure": None,
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--expert", type=Path)
    parser.add_argument("--split-registry", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--focus-gate", type=int, default=5)
    parser.add_argument("--context-steps", type=int, default=45)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--n-step", type=int, default=12)
    parser.add_argument("--fast-token", default="20260802_110754")
    parser.add_argument("--fast-boost", type=float, default=8.0)
    args = parser.parse_args()

    split_lookup = registry_map(args.split_registry)
    chunks = {split: defaultdict(list) for split in SPLITS}
    sequences = []
    sequence_ids = Counter()
    sources = list(local_chunks(args, split_lookup))
    if args.expert is not None:
        sources.extend(expert_chunks(args))
    for split, chunk, metadata in sources:
        sequence_id = sequence_ids[split]
        sequence_ids[split] += 1
        count = len(chunk["action"])
        chunk["sequence_id"] = np.full(count, sequence_id, np.int32)
        append_chunk(chunks[split], chunk)
        metadata["sequence_id"] = sequence_id
        sequences.append(metadata)

    args.out.mkdir(parents=True, exist_ok=True)
    stats = {}
    for split in SPLITS:
        if not chunks[split]:
            continue
        output = {
            name: np.concatenate(values)
            for name, values in chunks[split].items()
        }
        np.savez_compressed(args.out / f"{split}.npz", **output)
        stats[split] = {
            "sequences": int(sequence_ids[split]),
            "rows": int(len(output["action"])),
            "actor_rows": int(np.count_nonzero(output["actor_weight"] > 0)),
            "success_sequences": sum(
                row["split"] == split and row["crossed_focus_gate"]
                for row in sequences
            ),
            "failure_sequences": sum(
                row["split"] == split and not row["crossed_focus_gate"]
                for row in sequences
            ),
        }
    manifest = {
        "schema": 1,
        "focus_gate": args.focus_gate,
        "context_steps": args.context_steps,
        "gamma": args.gamma,
        "n_step": args.n_step,
        "target": "bounded residual action",
        "actor_policy": "successful focus-gate rows only; failures critic-only",
        "training_root": str(args.training_root.resolve()),
        "expert": str(args.expert.resolve()) if args.expert else None,
        "split_registry": str(args.split_registry.resolve()),
        "split_registry_sha256": sha256(args.split_registry),
        "stats": stats,
        "sequences": sequences,
    }
    manifest_path = args.out / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(args.out.resolve()), "stats": stats, "manifest_sha256": sha256(manifest_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
