"""Minimal UDP-5600 JPEG receiver for the NEW sim: keeps the latest decoded BGR frame.

Used only for RECORDING videos (state training flies on odometry, not vision), so the control
path is unaffected. Port 5600 is free during training.
"""
from __future__ import annotations
import socket, struct, threading
import cv2
import numpy as np

_HDR = "<IHHIIQ"
_HSZ = struct.calcsize(_HDR)


class VisionRX:
    def __init__(self, port=5600):
        self.latest = None            # (sim_time_ns, bgr)
        self.lock = threading.Lock()
        self.run = True
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("0.0.0.0", port))
        self.sock.settimeout(0.5)
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        frames = {}
        while self.run:
            try:
                packet, _ = self.sock.recvfrom(65536)
            except socket.timeout:
                continue
            except OSError:
                break
            if len(packet) < _HSZ:
                continue
            fid, cid, total, jsize, psize, tns = struct.unpack(_HDR, packet[:_HSZ])
            f = frames.setdefault(fid, {"chunks": {}, "total": total})
            f["chunks"][cid] = packet[_HSZ:]
            if len(f["chunks"]) == f["total"]:
                if all(i in f["chunks"] for i in range(f["total"])):
                    blob = b"".join(f["chunks"][i] for i in range(f["total"]))
                    img = cv2.imdecode(np.frombuffer(blob, np.uint8), cv2.IMREAD_COLOR)
                    if img is not None:
                        with self.lock:
                            self.latest = (tns, img)
                del frames[fid]
                if len(frames) > 40:
                    frames.clear()

    def get(self):
        with self.lock:
            return self.latest

    def stop(self):
        self.run = False
        try:
            self.sock.close()
        except OSError:
            pass
