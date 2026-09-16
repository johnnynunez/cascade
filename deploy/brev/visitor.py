"""Serve a small public camera view without forwarding administrative routes."""

from __future__ import annotations

import argparse
import base64
import hmac
from itertools import chain
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading
from urllib.error import URLError
from urllib.parse import urlsplit
from urllib.request import urlopen

CAMERAS = ("kitchen", "worktop", "side")
ROOT = Path(__file__).resolve().parent


class VisitorServer(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 16

    def __init__(self, address, handler, max_connections=32):
        self.slots = threading.BoundedSemaphore(max_connections)
        super().__init__(address, handler)

    def get_request(self):
        connection, address = super().get_request()
        connection.settimeout(10)
        return connection, address

    def process_request(self, request, address):
        if not self.slots.acquire(blocking=False):
            try:
                request.sendall(b"HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\nConnection: close\r\nRetry-After: 2\r\n\r\n")
            except OSError:
                pass
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, address)
        except BaseException:
            self.slots.release()
            raise

    def process_request_thread(self, request, address):
        try:
            super().process_request_thread(request, address)
        finally:
            self.slots.release()


class VisitorHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    def log_message(self, format, *args):
        pass

    def respond(self, status, body, content_type="application/json"):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy", "default-src 'self'; img-src 'self' blob:; media-src 'self' blob:; style-src 'self'; script-src 'self'; frame-ancestors 'none'")
        if status == 401:
            self.send_header("WWW-Authenticate", 'Basic realm="Physical Agentic AI", charset="UTF-8"')
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.server.authorization and not hmac.compare_digest(
            self.headers.get("Authorization", ""), self.server.authorization
        ):
            return self.respond(401, b'{"error":"Authentication required"}')
        path = urlsplit(self.path).path
        if path in ("/", "/visitor.css", "/visitor.js", "/visitor-player.js",
                    "/staff/", "/staff/style.css", "/staff/media/openclaw-cameras.png"):
            name, content_type = {
                "/": ("visitor.html", "text/html; charset=utf-8"),
                "/visitor.css": ("visitor.css", "text/css; charset=utf-8"),
                "/visitor.js": ("visitor.js", "text/javascript; charset=utf-8"),
                "/visitor-player.js": ("visitor-player.js", "text/javascript; charset=utf-8"),
                "/staff/": ("staff.html", "text/html; charset=utf-8"),
                "/staff/style.css": ("staff.css", "text/css; charset=utf-8"),
                "/staff/media/openclaw-cameras.png": ("staff-openclaw-cameras.png", "image/png"),
            }[path]
            return self.respond(200, (ROOT / name).read_bytes(), content_type)
        try:
            if path in {f"/video/{name}.mp4" for name in CAMERAS}:
                return self.video(path.split("/")[-1][:-4])
            if path == "/api/status":
                with urlopen(self.server.camera_origin + "/state", timeout=5) as response:
                    state = json.load(response)
                rows = state.get("cameras", [])
                rows = {row["name"]: row for row in rows} if isinstance(rows, list) else rows
                cameras = []
                for name in CAMERAS:
                    row = rows.get(name, {})
                    cameras.append({
                        "name": name, "label": name.capitalize(),
                        "online": row.get("online", False),
                        "frame_id": row.get("frame_id"),
                        "fps": row.get("fps"),
                        "stream_url": f"/snapshot/{name}.jpg",
                    })
                return self.respond(200, json.dumps({"cameras": cameras,
                    "video": {"enabled": getattr(self.server, "video", None) is not None,
                              "codec": "h264", "nominal_fps": 3},
                    "camera_discovery": {"version": 1, "base_url": "/", "state_url": "/api/status",
                        "default_camera": "worktop", "cameras": cameras}}).encode())
            if path in {f"/snapshot/{name}.jpg" for name in CAMERAS}:
                with urlopen(self.server.camera_origin + path, timeout=5) as response:
                    frame = response.read(4 * 1024 * 1024)
                if not frame.startswith(b"\xff\xd8") or not frame.endswith(b"\xff\xd9"):
                    raise ValueError("Camera did not return JPEG")
                return self.respond(200, frame, "image/jpeg")
        except (URLError, OSError, ValueError):
            return self.respond(503, b'{"error":"Camera is reconnecting"}')
        return self.respond(404, b'{"error":"Not found"}')

    def video(self, camera):
        streams = getattr(self.server, "video", None)
        if streams is None:
            return self.respond(404, b'{"error":"Video is disabled"}')
        try:
            chunks = streams.get(camera).chunks()
            initial = next(chunks)
        except (OSError, ValueError, TimeoutError):
            return self.respond(503, b'{"error":"Video is reconnecting"}')
        self.send_response(200)
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Transfer-Encoding", "chunked")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        self.connection.settimeout(10)
        try:
            for chunk in chain((initial,), chunks):
                self.wfile.write(f"{len(chunk):x}\r\n".encode() + chunk + b"\r\n")
                self.wfile.flush()
        except (OSError, TimeoutError):
            pass
        finally:
            self.close_connection = True

    def do_POST(self):
        self.respond(405, b'{"error":"Method not allowed"}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8093)
    parser.add_argument("--camera-origin", default="http://127.0.0.1:8091")
    parser.add_argument("--auth-file", type=Path)
    parser.add_argument("--video", action="store_true", help="Share the camera feeds as NVENC H.264 video")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    args = parser.parse_args()
    server = VisitorServer(("127.0.0.1", args.port), VisitorHandler)
    server.camera_origin = args.camera_origin.rstrip("/")
    server.video = None
    if args.video:
        from visitor_video import VideoStreams
        server.video = VideoStreams(server.camera_origin, args.ffmpeg)
    server.authorization = ""
    if args.auth_file:
        credentials = json.loads(args.auth_file.read_text())
        value = f"{credentials['username']}:{credentials['password']}".encode()
        server.authorization = "Basic " + base64.b64encode(value).decode()
    try:
        server.serve_forever()
    finally:
        if server.video:
            server.video.stop()
        server.server_close()


if __name__ == "__main__":
    main()
