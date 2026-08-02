"""Non-blocking, crash-tolerant archive for live simulator sessions.

The control and receiver threads only enqueue immutable records. Dedicated
writer threads persist every completed JPEG frame and every received MAVLink
packet continuously, so a forced training stop still leaves usable data.
"""

from __future__ import annotations

import csv
import json
import os
import queue
import struct
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np


_RAW_HEADER = struct.Struct("<QHI")


class FullSessionRecorder:
    def __init__(self, root: Path, name: str, metadata: dict | None = None):
        self.dir = Path(root) / name
        self.frames_dir = self.dir / "frames"
        self.frames_dir.mkdir(parents=True, exist_ok=False)
        self.debug_frames_dir = self.dir / "localizer_debug" / "frames"
        self.debug_frames_dir.mkdir(parents=True, exist_ok=False)
        self.frame_queue: queue.Queue = queue.Queue(maxsize=4096)
        self.event_queue: queue.Queue = queue.Queue(maxsize=200_000)
        self.debug_queue: queue.Queue = queue.Queue(maxsize=512)
        self.running = True
        self.frame_count = 0
        self.mavlink_count = 0
        self.command_count = 0
        self.debug_frame_count = 0
        self.frame_queue_drops = 0
        self.event_queue_drops = 0
        self.debug_queue_drops = 0
        self.started_wall_ns = time.time_ns()
        self._metadata = dict(metadata or {})
        self._write_manifest(final=False)
        self._frame_thread = threading.Thread(
            target=self._frame_writer,
            name="full-session-frame-writer",
            daemon=True,
        )
        self._event_thread = threading.Thread(
            target=self._event_writer,
            name="full-session-event-writer",
            daemon=True,
        )
        self._debug_thread = threading.Thread(
            target=self._debug_writer,
            name="full-session-debug-writer",
            daemon=True,
        )
        self._frame_thread.start()
        self._event_thread.start()
        self._debug_thread.start()
        print(f"Full session archive: {self.dir}", flush=True)

    def _write_manifest(self, *, final: bool) -> None:
        payload = {
            **self._metadata,
            "archive_format": 1,
            "started_wall_ns": self.started_wall_ns,
            "updated_wall_ns": time.time_ns(),
            "finalized": bool(final),
            "frames": self.frame_count,
            "mavlink_rx_packets": self.mavlink_count,
            "commands": self.command_count,
            "localizer_debug_frames": self.debug_frame_count,
            "frame_queue_drops": self.frame_queue_drops,
            "event_queue_drops": self.event_queue_drops,
            "debug_queue_drops": self.debug_queue_drops,
        }
        path = self.dir / "manifest.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, indent=2, default=str))
        os.replace(temporary, path)

    def on_frame(self, frame_tuple) -> None:
        try:
            self.frame_queue.put_nowait(frame_tuple)
        except queue.Full:
            self.frame_queue_drops += 1

    def on_mavlink_event(
        self,
        stream: str,
        wall_ns: int,
        kind: str,
        payload: Any,
    ) -> None:
        try:
            self.event_queue.put_nowait(
                (stream, int(wall_ns), str(kind), payload)
            )
        except queue.Full:
            self.event_queue_drops += 1

    def on_localizer_debug(
        self,
        frame_id: int,
        wall_ns: int,
        image: np.ndarray,
        payload: dict,
    ) -> None:
        try:
            self.debug_queue.put_nowait((
                int(frame_id),
                int(wall_ns),
                np.asarray(image).copy(),
                payload,
            ))
        except queue.Full:
            self.debug_queue_drops += 1

    def _frame_writer(self) -> None:
        index_path = self.dir / "frames_index.csv"
        with index_path.open("w", newline="", buffering=1) as stream:
            writer = csv.writer(stream)
            writer.writerow([
                "frame_id",
                "sim_time_ns",
                "wall_ns",
                "nbytes",
                "path",
            ])
            last_manifest = time.monotonic()
            while self.running or not self.frame_queue.empty():
                try:
                    frame_id, sim_time_ns, jpeg, wall_ns = (
                        self.frame_queue.get(timeout=0.2)
                    )
                except queue.Empty:
                    continue
                name = (
                    f"{int(wall_ns)}_{int(sim_time_ns)}_"
                    f"{int(frame_id)}.jpg"
                )
                relative = Path("frames") / name
                (self.dir / relative).write_bytes(jpeg)
                writer.writerow([
                    int(frame_id),
                    int(sim_time_ns),
                    int(wall_ns),
                    len(jpeg),
                    relative.as_posix(),
                ])
                self.frame_count += 1
                self.frame_queue.task_done()
                if time.monotonic() - last_manifest >= 5.0:
                    self._write_manifest(final=False)
                    last_manifest = time.monotonic()

    def _event_writer(self) -> None:
        raw_path = self.dir / "mavlink_rx.bin"
        index_path = self.dir / "mavlink_index.csv"
        command_path = self.dir / "commands.jsonl"
        with (
            raw_path.open("wb", buffering=0) as raw_stream,
            index_path.open("w", newline="", buffering=1) as index_stream,
            command_path.open("w", buffering=1) as command_stream,
        ):
            index = csv.writer(index_stream)
            index.writerow([
                "wall_ns", "message_type", "offset", "record_bytes"
            ])
            while self.running or not self.event_queue.empty():
                try:
                    stream, wall_ns, kind, payload = self.event_queue.get(
                        timeout=0.2
                    )
                except queue.Empty:
                    continue
                if stream == "mavlink_rx":
                    raw = bytes(payload)
                    encoded_kind = kind.encode("utf-8")
                    offset = raw_stream.tell()
                    header = _RAW_HEADER.pack(
                        wall_ns, len(encoded_kind), len(raw)
                    )
                    raw_stream.write(header)
                    raw_stream.write(encoded_kind)
                    raw_stream.write(raw)
                    index.writerow([
                        wall_ns,
                        kind,
                        offset,
                        len(header) + len(encoded_kind) + len(raw),
                    ])
                    self.mavlink_count += 1
                else:
                    command_stream.write(json.dumps({
                        "wall_ns": wall_ns,
                        "stream": stream,
                        "kind": kind,
                        "payload": payload,
                    }, separators=(",", ":"), default=str) + "\n")
                    self.command_count += 1
                self.event_queue.task_done()

    @staticmethod
    def _json_default(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        return str(value)

    def _debug_writer(self) -> None:
        index_path = self.dir / "localizer_debug" / "debug.jsonl"
        with index_path.open("w", buffering=1) as stream:
            while self.running or not self.debug_queue.empty():
                try:
                    frame_id, wall_ns, image, payload = (
                        self.debug_queue.get(timeout=0.2)
                    )
                except queue.Empty:
                    continue
                relative = (
                    Path("localizer_debug")
                    / "frames"
                    / f"{wall_ns}_{frame_id}.jpg"
                )
                ok, encoded = cv2.imencode(
                    ".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95]
                )
                if ok:
                    (self.dir / relative).write_bytes(encoded.tobytes())
                stream.write(json.dumps({
                    "frame_id": frame_id,
                    "wall_ns": wall_ns,
                    "image": relative.as_posix() if ok else None,
                    "debug": payload,
                }, separators=(",", ":"), default=self._json_default) + "\n")
                self.debug_frame_count += 1
                self.debug_queue.task_done()

    def close(self) -> Path:
        self.running = False
        self._frame_thread.join(timeout=30.0)
        self._event_thread.join(timeout=30.0)
        self._debug_thread.join(timeout=30.0)
        self._write_manifest(final=True)
        print(
            "Full session archive finalized: "
            f"{self.frame_count} frames, "
            f"{self.mavlink_count} MAVLink packets, "
            f"{self.command_count} commands, "
            f"{self.debug_frame_count} localizer debug frames, "
            f"drops={self.frame_queue_drops + self.event_queue_drops + self.debug_queue_drops}",
            flush=True,
        )
        return self.dir
