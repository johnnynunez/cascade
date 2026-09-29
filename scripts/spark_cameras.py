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


class Frames:
    """Share original JPEGs; poll only while a viewer requests a camera."""
    def __init__(self, bridge_port=8611):
        self.bridge_port = bridge_port
        self.lock = threading.Lock()
        self.rows = {}

    def frame(self, name):
        if name not in CAMERAS:
            raise ValueError("unknown camera")
        with self.lock:
            now = time.monotonic()
            old = self.rows.get(name, {})
            if now - old.get("checked", 0) < .5:
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
            row = {"jpeg": jpg, "stamp": stamp, "checked": now,
                   "advance": now if changed else old.get("advance", now),
                   "frame_id": old.get("frame_id", 0) + int(changed)}
            self.rows[name] = row
            return row

    def state(self):
        rows = {}
        for name in CAMERAS:
            try:
                row = self.frame(name)
                live = time.monotonic() - row["advance"] < 8
                rows[name] = {"frame_id": row["frame_id"], "online": live,
                              "error": None if live else "Camera stopped advancing", "fps": 2}
            except (OSError, ValueError, KeyError):
                rows[name] = {"frame_id": None, "online": False, "error": "Camera is reconnecting", "fps": 0}
        return {"cameras": rows}


class CameraHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def respond(self, code, body, kind="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlsplit(self.path).path
        if path in ("/state", "/api/status"):
            state = self.server.frames.state()
            state["camera_discovery"] = {"version": 1, "base_url": "/", "state_url": "/state",
                "default_camera": "worktop", "cameras": [
                    {"name": name, "stream_url": "/stream/" + name} for name in CAMERAS]}
            return self.respond(200, json.dumps(state).encode())
        snapshots = {f"/snapshot/{name}.jpg": name for name in CAMERAS}
        if path in snapshots:
            try:
                return self.respond(200, self.server.frames.frame(snapshots[path])["jpeg"], "image/jpeg")
            except (OSError, ValueError, KeyError):
                return self.respond(503, b'{"error":"Camera is reconnecting"}')
        streams = {f"/stream/{name}": name for name in CAMERAS}
        if path in streams:
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.connection.settimeout(10)
            try:
                while True:
                    frame = self.server.frames.frame(streams[path])["jpeg"]
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                     + str(len(frame)).encode() + b"\r\n\r\n" + frame + b"\r\n")
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
