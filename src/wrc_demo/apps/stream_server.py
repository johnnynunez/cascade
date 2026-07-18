"""Multi-camera MJPEG livestream server (stdlib only, no new deps).

Serves every CameraRig stream over plain HTTP so the audience, the Hermes
chat user, and any browser on the LAN can watch what the robot sees while it
acts -- detections, agent status and world state overlaid live.

Routes:
    /                    dark dashboard: N camera tiles + world-state panel
    /stream/<name>       multipart/x-mixed-replace MJPEG (annotated)
    /snapshot/<name>.jpg one annotated frame
    /state               JSON: beliefs (with colors), cameras, agent status

The handler threads only ever *read* the latest frame slot of each stream,
so any number of viewers can attach without slowing perception or control.
"""

from __future__ import annotations

import json
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

from .live_view import draw_detections, draw_hud

_BOUNDARY = "wrcframe"

_INDEX_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>wrc-demo :: live</title><style>
 body {{ background:#0d1117; color:#d7dde3; font:14px/1.4 system-ui,sans-serif; margin:0; }}
 header {{ padding:10px 16px; background:#161c22; display:flex; gap:14px;
           align-items:baseline; border-bottom:1px solid #21262d; }}
 header h1 {{ font-size:16px; margin:0; color:#76b900; }}
 #task {{ color:#f0b429; font-weight:600; }}
 #status {{ color:#9fb0c0; margin-left:auto; font:12px ui-monospace,monospace; }}
 main {{ display:grid; grid-template-columns: 1fr 380px; gap:12px; padding:12px; }}
 #cams {{ display:flex; flex-wrap:wrap; gap:12px; align-content:start; }}
 .cam {{ background:#161c22; border-radius:8px; padding:8px; }}
 .cam h2 {{ font-size:13px; margin:0 0 6px 2px; color:#9fb0c0; font-weight:500; }}
 .cam img {{ max-width:min(44vw,760px); border-radius:4px; display:block; }}
 aside {{ display:flex; flex-direction:column; gap:12px; min-width:300px; }}
 .panel {{ background:#161c22; border-radius:8px; padding:10px 12px; }}
 .panel h3 {{ margin:0 0 8px; font-size:12px; color:#9fb0c0; text-transform:uppercase;
              letter-spacing:.06em; }}
 #feed {{ font:12px/1.55 ui-monospace,monospace; height:44vh; overflow-y:auto;
          white-space:pre-wrap; }}
 #feed .obs {{ color:#79c0ff; }} #feed .act {{ color:#7ee787; }}
 #feed .out {{ color:#ff7b72; }} #feed .note {{ color:#f0b429; }}
 table {{ width:100%; border-collapse:collapse; font:12px ui-monospace,monospace; }}
 td, th {{ padding:3px 6px; text-align:left; border-bottom:1px solid #21262d; }}
 .chip {{ display:inline-block; width:10px; height:10px; border-radius:2px;
          margin-right:6px; vertical-align:baseline; border:1px solid #444; }}
 .remembered {{ opacity:.55; }}
</style></head><body>
<header><h1>wrc-demo &middot; live</h1>
 <span id="task">no task</span><span id="status">connecting...</span></header>
<main>
 <div id="cams">{tiles}</div>
 <aside>
  <div class="panel"><h3>talk to the robot</h3>
   <form id="chat" style="display:flex;gap:6px">
    <input id="cmd" placeholder="pick and place pink object"
      style="flex:1;background:#0d1117;border:1px solid #30363d;border-radius:6px;
             color:#d7dde3;padding:7px 10px;font:13px system-ui">
    <button style="background:#76b900;border:0;border-radius:6px;color:#0d1117;
                   font-weight:700;padding:0 14px;cursor:pointer">send</button>
    <button type="button" id="stopbtn"
      style="background:#d9534f;border:0;border-radius:6px;color:#fff;
             font-weight:700;padding:0 14px;cursor:pointer">stop</button>
   </form><div id="chatmsg" style="font:11px ui-monospace,monospace;color:#9fb0c0;
                                   margin-top:6px"></div></div>
  <div class="panel"><h3>robot narration</h3><div id="feed">waiting...</div></div>
  <div class="panel"><h3>objects in the world model</h3>
   <table><thead><tr><th>object</th><th>color</th><th>position (m)</th><th>state</th></tr></thead>
   <tbody id="objs"></tbody></table></div>
 </aside>
</main>
<script>
 const CSS = {{observation:'obs', action:'act', outcome:'out', note:'note'}};
 const SWATCH = {{red:'#e5484d', orange:'#f76b15', yellow:'#ffe629', green:'#46a758',
   cyan:'#00a2c7', blue:'#0090ff', purple:'#8e4ec6', pink:'#f76190',
   brown:'#ad7f58', white:'#eee', gray:'#888', black:'#111'}};
 document.getElementById('stopbtn').addEventListener('click', async () => {{
   const msg = document.getElementById('chatmsg');
   try {{
     const r = await (await fetch('/cancel', {{method: 'POST'}})).json();
     msg.textContent = r.cancelled ? 'cancelled - the arm is stopping' : (r.error || 'cancel failed');
   }} catch (e) {{ msg.textContent = 'error: ' + e; }}
 }});
 document.getElementById('chat').addEventListener('submit', async (ev) => {{
   ev.preventDefault();
   const box = document.getElementById('cmd');
   const msg = document.getElementById('chatmsg');
   const task = box.value.trim();
   if (!task) return;
   msg.textContent = 'sending...';
   try {{
     const r = await (await fetch('/task', {{method: 'POST',
       headers: {{'Content-Type': 'application/json'}},
       body: JSON.stringify({{task}})}})).json();
     msg.textContent = r.accepted ? `running: ${{r.task}}` : (r.error || 'rejected');
     if (r.accepted) box.value = '';
   }} catch (e) {{ msg.textContent = 'error: ' + e; }}
 }});
 async function tick() {{
   try {{
     const s = await (await fetch('/state')).json();
     document.getElementById('task').textContent = s.task ? `task: ${{s.task}}` : 'idle - waiting for a command';
     document.getElementById('status').textContent =
       `${{Object.keys(s.cameras).length}} cams | holding: ${{s.holding || '-'}} | ` +
       `arm: ${{s.arm_connected ? 'up' : 'standby'}} | ${{s.agent_status}}`;
     const feed = document.getElementById('feed');
     const stick = feed.scrollTop + feed.clientHeight >= feed.scrollHeight - 8;
     feed.innerHTML = (s.events || []).map(l => {{
       const kind = (l.match(/\\] \\((\\w+)\\)/) || [])[1];
       return `<div class="${{CSS[kind] || ''}}">${{l.replace(/</g,'&lt;')}}</div>`;
     }}).join('');
     if (stick) feed.scrollTop = feed.scrollHeight;
     document.getElementById('objs').innerHTML = (s.objects || []).map(o =>
       `<tr class="${{o.state}}"><td>${{o.label}}</td>` +
       `<td><span class="chip" style="background:${{SWATCH[o.color] || '#333'}}"></span>${{o.color || '?'}}</td>` +
       `<td>[${{o.position.join(', ')}}]</td><td>${{o.state}} (${{o.age_s}}s)</td></tr>`
     ).join('');
   }} catch (e) {{}}
   setTimeout(tick, 500);
 }}
 tick();
</script></body></html>"""


def lan_ip() -> str:
    """Best-effort LAN address for printing a clickable URL."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except OSError:
        return "127.0.0.1"


class StreamServer:
    """HTTP front-end over a CameraRig + a state callback."""

    def __init__(
        self,
        rig,
        state_fn=None,
        host: str = "0.0.0.0",
        port: int = 8090,
        fps: float = 15.0,
        quality: int = 80,
    ):
        self._rig = rig
        self._state_fn = state_fn or (lambda: {})
        self._host = host
        self.port = port
        self._fps = fps
        self._quality = quality
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._stopping = False
        # One JPEG encode per camera per frame regardless of viewer count:
        # cache keyed by (frame id, overlay status) -- N viewers of the same
        # stream share the encode instead of costing O(viewers x fps).
        self._jpeg_cache: dict[str, tuple[tuple, bytes]] = {}
        self._jpeg_lock = threading.Lock()
        # Optional chat: POST /task hands a natural-language command to this
        # callback (the app wires it to its orchestrator). One at a time.
        self._task_fn = None
        self._cancel_fn = None
        self._task_busy = threading.Lock()

    def set_task_fn(self, fn) -> None:
        """fn(task_text) runs a command; called from a worker thread."""
        self._task_fn = fn

    def set_cancel_fn(self, fn) -> None:
        """fn() aborts the running command (POST /cancel, the STOP button)."""
        self._cancel_fn = fn

    # ── lifecycle ────────────────────────────────────────────────────────

    def start(self) -> None:
        handler = _make_handler(self)
        self._httpd = ThreadingHTTPServer((self._host, self.port), handler)
        self._httpd.daemon_threads = True
        self.port = self._httpd.server_address[1]  # resolves port=0
        self._thread = threading.Thread(
            target=self._httpd.serve_forever, daemon=True, name="stream-server"
        )
        self._thread.start()
        print(f"[stream] live view on {self.url}", file=sys.stderr)

    def stop(self) -> None:
        self._stopping = True  # MJPEG handler loops watch this and exit
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    @property
    def url(self) -> str:
        return f"http://{lan_ip()}:{self.port}/"

    # ── rendering ────────────────────────────────────────────────────────

    def annotated_jpeg(self, name: str) -> bytes | None:
        stream = self._rig.get(name)
        frame = stream.latest()
        if frame is None:
            return None
        dets, status = stream.overlay()
        key = (frame.frame_id, len(dets), status)
        with self._jpeg_lock:
            cached = self._jpeg_cache.get(name)
            if cached is not None and cached[0] == key:
                return cached[1]
        img = frame.rgb.copy()
        try:
            draw_detections(img, dets)
        except Exception:
            pass
        h, w = img.shape[:2]
        draw_hud(img, [
            f"{stream.name}  {w}x{h}  {stream.fps:4.1f} fps  depth: {frame.depth_source}",
            f"agent: {status}",
        ])
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, self._quality])
        if not ok:
            return None
        jpeg = buf.tobytes()
        with self._jpeg_lock:
            self._jpeg_cache[name] = (key, jpeg)
        return jpeg

    def state(self) -> dict:
        cams = self._rig.stats()
        base = {"cameras": cams, "t": time.time()}
        try:
            base.update(self._state_fn() or {})
        except Exception as e:
            base["state_error"] = str(e)
        base.setdefault("agent_status", "idle")
        return base


def _make_handler(server: StreamServer):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):  # stderr chatter only on errors
            pass

        # ── routes ───────────────────────────────────────────────────────

        def do_GET(self):  # noqa: N802 (BaseHTTPRequestHandler API)
            path = self.path.split("?", 1)[0].rstrip("/") or "/"
            try:
                if path == "/":
                    return self._index()
                if path == "/state":
                    return self._json(server.state())
                if path.startswith("/snapshot/"):
                    name = path.split("/", 2)[2].removesuffix(".jpg")
                    return self._snapshot(name)
                if path.startswith("/stream/"):
                    return self._mjpeg(path.split("/", 2)[2])
                self.send_error(404, "unknown path")
            except (BrokenPipeError, ConnectionResetError):
                pass  # viewer closed the tab mid-frame
            except KeyError as e:
                self.send_error(404, str(e))

        def do_POST(self):  # noqa: N802 (BaseHTTPRequestHandler API)
            if self.path.rstrip("/") == "/cancel":
                if server._cancel_fn is None:
                    return self.send_error(501, "no cancel handler wired")
                try:
                    server._cancel_fn()
                    return self._json({"cancelled": True})
                except Exception as e:
                    return self._json({"cancelled": False, "error": str(e)[:120]})
            if self.path.rstrip("/") != "/task":
                return self.send_error(404, "unknown path")
            if server._task_fn is None:
                return self.send_error(501, "no task handler wired")
            try:
                n = int(self.headers.get("Content-Length", 0))
                task = str(json.loads(self.rfile.read(n)).get("task", "")).strip()
            except (ValueError, json.JSONDecodeError):
                return self.send_error(400, "body must be JSON {\"task\": ...}")
            if not task:
                return self.send_error(400, "empty task")
            if not server._task_busy.acquire(blocking=False):
                return self._json({"accepted": False, "error": "busy with another command"})

            def _run():
                try:
                    server._task_fn(task)
                except Exception as e:
                    print(f"[stream] task failed: {e}", file=sys.stderr)
                finally:
                    server._task_busy.release()

            threading.Thread(target=_run, daemon=True, name="dashboard-task").start()
            self._json({"accepted": True, "task": task})

        def _index(self):
            tiles = "".join(
                f'<div class="cam"><h2>{n}</h2>'
                f'<img src="/stream/{n}" alt="{n}"></div>'
                for n in server._rig.names
            )
            body = _INDEX_HTML.format(tiles=tiles).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, payload: dict):
            body = json.dumps(payload, default=_np_safe).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _snapshot(self, name: str):
            jpeg = server.annotated_jpeg(name)
            if jpeg is None:
                return self.send_error(503, "no frame yet")
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(jpeg)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(jpeg)

        def _mjpeg(self, name: str):
            server._rig.get(name)  # 404 (KeyError) before headers go out
            self.send_response(200)
            self.send_header(
                "Content-Type", f"multipart/x-mixed-replace; boundary={_BOUNDARY}"
            )
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            period = 1.0 / max(server._fps, 1.0)
            while not server._stopping:
                t0 = time.monotonic()
                jpeg = server.annotated_jpeg(name)
                if jpeg is not None:
                    self.wfile.write(
                        b"--" + _BOUNDARY.encode() + b"\r\n"
                        b"Content-Type: image/jpeg\r\n"
                        + f"Content-Length: {len(jpeg)}\r\n\r\n".encode()
                    )
                    self.wfile.write(jpeg)
                    self.wfile.write(b"\r\n")
                    self.wfile.flush()
                dt = time.monotonic() - t0
                if dt < period:
                    time.sleep(period - dt)

    return Handler


def _np_safe(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)
