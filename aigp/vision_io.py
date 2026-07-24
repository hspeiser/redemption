"""Camera frame receiver: reassembles chunked JPEG frames from the sim's UDP
video stream. Every completed frame is passed to an optional callback with its
sim timestamp and wall-clock receive time."""

import socket
import struct
import threading
import time

HEADER_FMT = "<IHHIIQ"
HEADER_SZ = struct.calcsize(HEADER_FMT)


class VisionRX:
    def __init__(self, port=5600, ip="0.0.0.0", on_frame=None):
        self.on_frame = on_frame
        self.latest = None            # (frame_id, sim_time_ns, jpeg_bytes, wall_ns)
        self.frame_count = 0
        self.dropped = 0
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8 * 1024 * 1024)
        self.sock.bind((ip, port))
        self.sock.settimeout(0.5)
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        frames = {}
        while self.running:
            try:
                packet, _ = self.sock.recvfrom(65536)
            except socket.timeout:
                continue
            except OSError:
                return
            frame_id, chunk_id, total_chunks, jpeg_size, payload_size, sim_time_ns = \
                struct.unpack(HEADER_FMT, packet[:HEADER_SZ])
            f = frames.setdefault(frame_id, {"chunks": {}, "total": total_chunks,
                                             "time": sim_time_ns})
            f["chunks"][chunk_id] = packet[HEADER_SZ:]
            if len(f["chunks"]) == f["total"]:
                try:
                    jpeg = b"".join(f["chunks"][i] for i in range(f["total"]))
                except KeyError:
                    self.dropped += 1
                    del frames[frame_id]
                    continue
                tup = (frame_id, f["time"], jpeg, time.time_ns())
                self.latest = tup
                self.frame_count += 1
                if self.on_frame is not None:
                    try:
                        self.on_frame(tup)
                    except Exception as e:
                        print(f"on_frame error: {e}", flush=True)
                del frames[frame_id]
            # GC stale partial frames
            if len(frames) > 90:
                for fid in sorted(frames)[:30]:
                    del frames[fid]
                    self.dropped += 1

    def close(self):
        self.running = False
        try:
            self.sock.close()
        except OSError:
            pass
