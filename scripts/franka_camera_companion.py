#!/usr/bin/env python3
"""Read-only Chrome camera companion for an existing CASCADE stream server.

This process never imports camera drivers or robot code. It forwards a small
allowlist of GETs to one loopback stream server, so it cannot claim a second
RealSense pipeline or expose the dashboard's motion/STOP/reset routes.
"""
from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import re
from urllib.error import URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, build_opener


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def loopback_url(value: str) -> str:
    url = urlsplit(value)
    if (url.scheme != "http" or url.hostname not in ("127.0.0.1", "localhost", "::1")
            or url.username or url.password or url.query or url.fragment
            or url.path not in ("", "/") or not url.port):
        raise ValueError("stream URL must be a loopback HTTP origin with an explicit port")
    return value.rstrip("/")


class CameraSource:
    def __init__(self, origin: str):
        self.origin = loopback_url(origin)
        self.opener = build_opener(ProxyHandler({}), NoRedirect())

    def open(self, path: str):
        return self.opener.open(self.origin + path, timeout=10)

    def state(self) -> dict:
        with self.open("/state") as response:
            data = response.read(1024 * 1024 + 1)
        if len(data) > 1024 * 1024:
            raise ValueError("camera state is too large")
        state = json.loads(data)
        cameras = state.get("cameras")
        if not isinstance(cameras, dict):
            raise ValueError("camera state is missing")
        # Never expose task text, beliefs, user metadata or upstream URLs.
        rows = {}
        for name, row in cameras.items():
            if not re.fullmatch(r"[A-Za-z0-9_.-]+", name) or name == ".." or not isinstance(row, dict):
                continue
            rows[name] = {k: row[k] for k in ("fps", "frames", "frame_id", "error", "depth", "has_depth", "last_frame_age_s") if k in row}
            rows[name]["online"] = not row.get("error") and float(row.get("fps") or 0) > 0
        return {"cameras": rows, "camera_discovery": {
            "version": 1, "base_url": "/", "state_url": "/state",
            "default_camera": next(iter(rows), None),
            "cameras": [{"name": name, "stream_url": "/stream/" + name} for name in rows],
        }}


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
        request = urlsplit(self.path)
        if request.query or request.fragment:
            return self.respond(404, b'{"error":"Not found"}')
        path = request.path
        try:
            if path in ("/state", "/api/status"):
                return self.respond(200, json.dumps(self.server.source.state()).encode())
            match = re.fullmatch(r"/(stream|snapshot)/([A-Za-z0-9_.-]+)", path)
            if not match:
                return self.respond(404, b'{"error":"Not found"}')
            route, name = match.groups()
            name = name.removesuffix(".jpg") if route == "snapshot" else name
            if name not in self.server.source.state()["cameras"]:
                return self.respond(404, b'{"error":"Unknown camera"}')
            target = "/snapshot/" + name + ".jpg" if route == "snapshot" else "/stream/" + name
            with self.server.source.open(target) as response:
                kind = response.headers.get("Content-Type", "")
                if route == "snapshot":
                    body = response.read(16 * 1024 * 1024 + 1)
                    if len(body) > 16 * 1024 * 1024 or not kind.startswith("image/jpeg"):
                        raise ValueError("invalid snapshot")
                    return self.respond(200, body, "image/jpeg")
                if not kind.startswith("multipart/x-mixed-replace"):
                    raise ValueError("invalid camera stream")
                self.send_response(200)
                self.send_header("Content-Type", kind)
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.connection.settimeout(10)
                try:
                    while block := response.read1(64 * 1024):
                        self.wfile.write(block)
                        self.wfile.flush()
                except (OSError, URLError):
                    pass
                return
        except (OSError, ValueError, KeyError, TypeError):
            return self.respond(503, b'{"error":"Camera is reconnecting"}')

    def do_POST(self):
        self.respond(405, b'{"error":"Read only"}')

    do_PUT = do_DELETE = do_PATCH = do_POST


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8091)
    parser.add_argument("--source", default="http://127.0.0.1:8090")
    args = parser.parse_args()
    source = CameraSource(args.source)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), CameraHandler)
    server.daemon_threads = True
    server.source = source
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
