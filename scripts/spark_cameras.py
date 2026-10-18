#!/usr/bin/env python3
"""Read-only, loopback camera surface for the Spark desktop and visitor."""
from __future__ import annotations

import argparse
import base64
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import socket
import threading
import time
from urllib.parse import urlsplit

CAMERAS = {"kitchen": "proof", "worktop": "cam0", "side": "side"}
#: A viewer read asks the bridge at most this often per camera.
CACHE_S = .5
#: A refused bridge read keeps serving the last good image this long after it
#: last advanced: the extension's own threshold for "Not updating". On Newton
#: the bridge refuses a camera for ~0.3 s (p90 0.65-0.8 s) whenever its 0.5 s
#: fallback capture lands off the render rhythm. Raising there closed the
#: MJPEG stream, and the extension's 2/4/8 s reconnect backoff turned each gap
#: into a 4-7 s "Not updating" episode; holding keeps the stream open through
#: the gap instead (tests/test_spark_cameras_hold.py). Only this viewer holds:
#: the bridge, and every perception reader of it, still sees a missing frame
#: as missing.
HOLD_S = 5.
#: During a hold the bridge is asked again after this long, not CACHE_S.
RETRY_S = .1


def age_s(row, now=None):
    """Seconds since this surface first received the image in `row`: the
    X-Frame-Age header and `/state`'s `age_s`. It is NOT the capture age: the
    time the image spent in the simulator and the bridge before this viewer
    read it is not counted, so a fresh-looking age can be a slightly older
    image."""
    return (time.monotonic() if now is None else now) - row["advance"]


class Frames:
    """Share original JPEGs; poll only while a viewer requests a camera."""
    def __init__(self, bridge_port=8611, *, hold_s=HOLD_S, clock=time.monotonic):
        self.bridge_port = bridge_port
        self.hold_s = hold_s
        self.clock = clock
        self.lock = threading.Lock()
        self.rows = {}

    def frame(self, name):
        """The newest good frame of `name`. A refused bridge read (not-ready
        camera, socket error, bad reply) returns the last good row while it is
        younger than `hold_s` (its `age_s` says how old) and retries the bridge
        after RETRY_S; with no such row it raises, as before."""
        if name not in CAMERAS:
            raise ValueError("unknown camera")
        with self.lock:
            try:
                row = self._read(name)
            except (OSError, ValueError, KeyError):
                now = self.clock()  # after the refused read, which may have waited
                row = self.rows.get(name)
                if row is None or now - row["advance"] >= self.hold_s:
                    raise
                row["checked"], row["held"] = now - CACHE_S + RETRY_S, True
                return row
            if row.get("held") and self.clock() - row["advance"] >= self.hold_s:
                raise ValueError("camera not ready")  # a held row from the retry cache
            return row

    def _read(self, name):
        now = self.clock()
        old = self.rows.get(name, {})
        if now - old.get("checked", 0) < CACHE_S:
            return old
        # Fixed read-only operation. HTTP callers cannot supply bridge ops.
        with socket.create_connection(("127.0.0.1", self.bridge_port), timeout=5) as sock:
            with sock.makefile("rwb") as stream:
                stream.write(json.dumps({"op": "frame", "camera": CAMERAS[name]}).encode() + b"\n")
                stream.flush()
                raw = stream.readline(16 * 1024 * 1024 + 1)
        if len(raw) > 16 * 1024 * 1024:
            raise ValueError("camera response too large")
        data = json.loads(raw)
        if data.get("ok") is not True:
            raise ValueError("camera not ready")
        jpg = base64.b64decode(data["rgb_jpeg_b64"], validate=True)
        if not jpg.startswith(b"\xff\xd8") or not jpg.endswith(b"\xff\xd9"):
            raise ValueError("invalid JPEG")
        stamp = data.get("t")
        if type(stamp) not in (int, float) or not math.isfinite(stamp):
            raise ValueError("camera capture timestamp missing")
        changed = old.get("stamp") != stamp
        # A new image is first seen when the read returns, not when it was asked
        # for: a slow or hung read must not age it (X-Frame-Age 3 s after a 3 s hang).
        seen = self.clock()
        row = {"jpeg": jpg, "stamp": stamp, "checked": now,
               "advance": seen if changed else old.get("advance", seen),
               "frame_id": old.get("frame_id", 0) + int(changed)}
        self.rows[name] = row
        return row

    def state(self):
        rows = {}
        for name in CAMERAS:
            try:
                row = self.frame(name)
                age = age_s(row, self.clock())
                live = age < 8
                rows[name] = {"frame_id": row["frame_id"], "online": live, "age_s": round(age, 2),
                              "error": None if live else "Camera stopped advancing", "fps": 2}
            except (OSError, ValueError, KeyError):
                rows[name] = {"frame_id": None, "online": False, "error": "Camera is reconnecting", "fps": 0}
        return {"cameras": rows}


class CameraHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def respond(self, code, body, kind="application/json", headers=()):
        self.send_response(code)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        for key, value in headers:
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        frames = self.server.frames
        path = urlsplit(self.path).path
        if path in ("/state", "/api/status"):
            state = frames.state()
            state["camera_discovery"] = {"version": 1, "base_url": "/", "state_url": "/state",
                "default_camera": "worktop", "cameras": [
                    {"name": name, "stream_url": "/stream/" + name} for name in CAMERAS]}
            return self.respond(200, json.dumps(state).encode())
        snapshots = {f"/snapshot/{name}.jpg": name for name in CAMERAS}
        if path in snapshots:
            try:
                row = frames.frame(snapshots[path])
            except (OSError, ValueError, KeyError):
                return self.respond(503, b'{"error":"Camera is reconnecting"}')
            # X-Frame-Age: seconds since this viewer first saw the image (age_s), not its capture age
            return self.respond(200, row["jpeg"], "image/jpeg",
                                [("X-Frame-Age", f"{age_s(row, frames.clock()):.2f}")])
        streams = {f"/stream/{name}": name for name in CAMERAS}
        if path in streams:
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.connection.settimeout(10)
            try:
                while True:
                    row = frames.frame(streams[path])
                    frame = row["jpeg"]
                    # X-Frame-Age as on /snapshot: since this viewer first saw the image, not
                    # since capture. The extension shows it once it passes 1 s (core.mjs).
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                     + str(len(frame)).encode()
                                     + f"\r\nX-Frame-Age: {age_s(row, frames.clock()):.2f}".encode()
                                     + b"\r\n\r\n" + frame + b"\r\n")
                    self.wfile.flush()
                    time.sleep(.5)
            except (OSError, ValueError, KeyError):
                return
        self.respond(404, b'{"error":"Not found"}')

    def do_POST(self):
        self.respond(405, b'{"error":"Read only"}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8091)
    parser.add_argument("--bridge-port", type=int, default=8611)
    args = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), CameraHandler)
    server.daemon_threads = True
    server.frames = Frames(args.bridge_port)
    server.serve_forever()


if __name__ == "__main__":
    main()
