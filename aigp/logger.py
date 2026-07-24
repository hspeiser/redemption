"""Episode logger: records EVERYTHING from every flight.

Per episode directory:
  frames/<sim_time_ns>_<frame_id>.jpg   every camera frame, verbatim JPEG
  frames_index.csv                      frame_id, sim_time_ns, wall_ns, nbytes
  telemetry.npz                         full-rate odom/attitude/imu/local_pos/
                                        actuator/collisions/timesync/race_status/
                                        heartbeats/sent commands
  gates.json                            gate map (if received)
  meta.json                             free-form run metadata
"""

import csv
import json
import time
from pathlib import Path

import numpy as np


class EpisodeLogger:
    def __init__(self, root, name=None):
        name = name or time.strftime("%Y%m%d_%H%M%S")
        self.dir = Path(root) / name
        self.frames_dir = self.dir / "frames"
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self._index = []
        self._index_fh = open(self.dir / "frames_index.csv", "w", newline="")
        self._index_csv = csv.writer(self._index_fh)
        self._index_csv.writerow(["frame_id", "sim_time_ns", "wall_ns", "nbytes"])
        print(f"Logging episode to {self.dir}", flush=True)

    def on_frame(self, tup):
        frame_id, sim_time_ns, jpeg, wall_ns = tup
        (self.frames_dir / f"{sim_time_ns}_{frame_id}.jpg").write_bytes(jpeg)
        self._index_csv.writerow([frame_id, sim_time_ns, wall_ns, len(jpeg)])

    def finalize(self, mav, meta=None):
        self._index_fh.flush()
        self._index_fh.close()
        arrays = {
            "odom": np.array(list(mav.odom), dtype=np.float64),
            "attitude": np.array(list(mav.attitude), dtype=np.float64),
            "imu": np.array(list(mav.imu), dtype=np.float64),
            "local_pos": np.array(list(mav.local_pos), dtype=np.float64),
            "actuator": np.array(list(mav.actuator), dtype=np.float64),
            "collisions": np.array(list(mav.collisions), dtype=np.float64),
            "timesync": np.array(list(mav.timesync_msgs), dtype=np.float64),
            "race_status": np.array(list(mav.race_status_log), dtype=np.float64),
            "heartbeats": np.array(list(mav.heartbeats), dtype=np.float64),
        }
        sent = [(w, {"arm": 0, "reset": 1, "vel": 2, "rates": 3}.get(k, 9), *p)
                for (w, k, *p) in mav.sent_log]
        arrays["sent"] = np.array(sent, dtype=np.float64)
        np.savez_compressed(self.dir / "telemetry.npz", **arrays)
        if mav.gate_map:
            (self.dir / "gates.json").write_text(json.dumps(mav.gate_map, indent=1))
        (self.dir / "meta.json").write_text(json.dumps(meta or {}, indent=1))
        n = sum(1 for _ in self.frames_dir.iterdir())
        print(f"Episode finalized: {n} frames, "
              f"{len(arrays['odom'])} odom, {len(arrays['imu'])} imu samples", flush=True)
        return self.dir
