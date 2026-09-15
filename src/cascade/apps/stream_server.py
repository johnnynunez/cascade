"""Multi-camera MJPEG livestream server (stdlib only, no new deps).

Serves every CameraRig stream over plain HTTP so the audience, the Hermes
chat user, and any browser on the LAN can watch what the robot sees while it
acts -- detections, agent status and world state overlaid live.

Routes:
    /                    light dashboard: N camera tiles + world-state panel
    /stream/<name>       multipart/x-mixed-replace MJPEG (annotated)
    /snapshot/<name>.jpg one annotated frame
    /state               JSON: beliefs (with colors), cameras, agent status
    /keyframes           before/after keyframe filmstrip from the run dir
    /keyframe/<file>     one keyframe JPEG (basename-only, no traversal)

The handler threads only ever *read* the latest frame slot of each stream,
so any number of viewers can attach without slowing perception or control.
"""
#
# 2026-07-31: the dashboard is now a *diagnostic surface you attach*, not the
# interface. Chat (Hermes / OpenClaw / any MCP host) is the UI; this server is
# opened on demand by LiveViewController and auto-closes when idle. Three view
# modes per camera let a human see what the perception stack actually has:
#
#     /stream/<name>       detections + HUD          (what the detector sees)
#     /depth/<name>        depth colormap + source   (what the geometry sees)
#     /annotated/<name>    VIA marks + grid + optional configured display band
#     /analyze             one-shot JSON: detections, depth stats, description
#
# The depth view matters because `depth_source` silently degrades through
# sensor -> mono -> plane -> none, and a wrong grasp z is usually a depth
# problem the RGB view cannot show you.

from __future__ import annotations

import json
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np

from .live_view import draw_detections, draw_hud

_BOUNDARY = "wrcframe"

_INDEX_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="color-scheme" content="light"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Physical Agentic AI · OpenClaw Demo</title><style>
 :root {{ color-scheme: light; background: #f7f6f2; color: #242722; }}
 * {{ box-sizing: border-box; }}
 body {{ background: #f7f6f2; color: #242722; color-scheme: light; font: 15px/1.6 system-ui,sans-serif; margin: 0; -webkit-font-smoothing: antialiased; }}
 header {{ max-width: 1680px; margin: auto; padding: 23px 28px; display: flex; flex-wrap: wrap; gap: 12px 22px; align-items: center; border-bottom: 1px solid #d4d8c9; }}
 header h1 {{ font-size: 20px; line-height: 1.4; letter-spacing: -.4px; margin: 0; font-weight: 650; color: #242722; }}
 header a {{ text-underline-offset: 4px; }}
 #task {{ color: #4c5940; font-size: 13px; font-weight: 550; }}
 #status {{ color: #596153; margin-left: auto; font: 12px/1.6 system-ui,sans-serif; max-width: 380px; }}
 main {{ max-width: 1680px; margin: auto; display: grid; grid-template-columns: minmax(0, 2.2fr) minmax(310px, 1fr); gap: 22px; padding: 28px; align-items: start; }}
 #cams {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px; align-content: start; min-width: 0; }}
 .cam {{ position: relative; min-width: 0; border: 1px solid #d4d8c9; background: #fdfcf8; border-radius: 12px; padding: 12px; overflow: hidden; }}
 .cam:has(#img-worktop) {{ grid-column: 1 / -1; grid-row: 1; }}
 .cam h2 {{ font-size: 15px; line-height: 1.4; margin: 0 0 12px; color: #38422f; font-weight: 620; display: flex; align-items: center; gap: 8px; text-transform: capitalize; flex-wrap: wrap; }}
 .cam img {{ width: 100%; max-width: 100%; height: auto; aspect-ratio: 16 / 9; object-fit: contain; background: #e8e8df; border-radius: 7px; display: block; }}
 #cams[data-transport="offline"] .cam img, .cam[data-stream-state="offline"] img {{ visibility: hidden; }}
 #cams[data-transport="offline"] .cam::after, .cam[data-stream-state="offline"]::after {{ content: "Camera disconnected · reconnecting"; position: absolute; inset: 58px 12px 12px; display: grid; place-items: center; padding: 16px; text-align: center; color: #795a29; background: #efeee7; border-radius: 7px; font-size: 14px; }}
 .views {{ margin-left: auto; display: flex; gap: 5px; }}
 .views button {{ background: #f7f6f2; border: 1px solid #c4ccb8; color: #4c5940; border-radius: 6px; font: 550 11px/1.4 system-ui,sans-serif; min-height: 30px; padding: 6px 10px; cursor: pointer; }}
 .views button.on {{ background: #b4e35a; border-color: #91b749; color: #242722; font-weight: 650; }}
 button:focus-visible, a:focus-visible, input:focus-visible {{ outline: 3px solid #648e22; outline-offset: 3px; }}
 button:hover {{ filter: brightness(.97); }}
 aside {{ display: flex; flex-direction: column; gap: 16px; min-width: 0; }}
 .panel {{ border: 1px solid #d4d8c9; background: #efeee7; border-radius: 12px; padding: 18px; min-width: 0; overflow-wrap: anywhere; }}
 .panel h3 {{ margin: 0 0 12px; font-size: 12px; color: #4c5940; text-transform: uppercase; letter-spacing: .08em; font-weight: 650; }}
 .panel p {{ margin: 8px 0 12px; font-size: 14px; line-height: 1.7; }}
 #read-only-panel > a {{ display: inline-flex; align-items: center; background: #b4e35a; color: #242722 !important; border: 1px solid #a0c850; border-radius: 8px; font-size: 14px; font-weight: 650; padding: 10px 16px; min-height: 44px; text-decoration: none; }}
 #feed {{ font: 12px/1.65 ui-monospace,monospace; min-height: 120px; max-height: 28vh; overflow-y: auto; white-space: pre-wrap; }}
 #feed .obs {{ color: #375e7c; }} #feed .act {{ color: #436123; }}
 #feed .out {{ color: #aa3523; }} #feed .note {{ color: #795a29; }}
 table {{ width: 100%; border-collapse: collapse; font: 12px/1.6 system-ui,sans-serif; }}
 td, th {{ padding: 6px 4px; text-align: left; border-bottom: 1px solid #d4d8c9; }}
 th {{ color: #4c5940; font-weight: 600; }}
 .chip {{ display: inline-block; width: 10px; height: 10px; border-radius: 3px; margin-right: 6px; vertical-align: baseline; border: 1px solid #596153; }}
 .remembered {{ opacity: .6; }}
 [hidden] {{ display: none !important; }}
 @media (max-width: 1100px) {{
   header {{ padding: 22px; }}
   #status {{ width: 100%; max-width: none; margin-left: 0; }}
   main {{ padding: 22px; gap: 18px; grid-template-columns: minmax(0, 1.7fr) minmax(280px, 1fr); }}
   .cam {{ padding: 10px; }}
   .cam h2 {{ font-size: 14px; }}
   .views {{ gap: 4px; }}
   .views button {{ padding: 5px 7px; }}
 }}
 @media (max-width: 800px) {{
   header {{ padding: 20px; gap: 9px 16px; }}
   header h1 {{ font-size: 19px; }}
   #status {{ overflow-wrap: anywhere; }}
   main {{ grid-template-columns: minmax(0, 1fr); padding: 20px; }}
   #cams {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
   #feed {{ min-height: 80px; }}
   aside {{ min-width: 0; }}
 }}
 @media (max-width: 480px) {{
   main {{ padding: 16px; }}
   #cams {{ gap: 12px; grid-template-columns: minmax(0, 1fr); }}
   .cam:has(#img-worktop) {{ grid-column: auto; }}
   .cam {{ padding: 12px; }}
   .cam h2 {{ font-size: 15px; }}
   .views button {{ padding: 7px 10px; min-height: 34px; }}
 }}
</style></head><body>
<header><h1>Physical Agentic AI &middot; OpenClaw Demo</h1>
 <span id="task">{initial_task}</span>
 <a href="/keyframes" {command_hidden} style="color:#65695e;font-size:12px">keyframes</a>
 <span id="status">connecting...</span></header>
<main>
 <div id="cams">{tiles}</div>
 <aside>
  <div class="panel" id="read-only-panel" {read_only_hidden}><h3>Watch the robot</h3>
   <p>Keep these cameras beside your OpenClaw conversation. Watch Worktop for the object and Side for the lift. Use the guide's camera preview to check the current scene.</p>
   <a href="/openclaw/?connect=1" target="_blank" rel="noopener noreferrer"
      style="color:#375e7c">Open OpenClaw ↗</a>
   <p style="font-size:12px;color:#65695e">Send one order in the chat, wait for the movement, then ask for a physics check.</p></div>
  <div class="panel" id="command-panel" {command_hidden}><h3>Talk to the robot</h3>
   <form id="chat" style="display:flex;gap:6px">
    <input id="cmd" {command_disabled} placeholder="Pick up the green cube and place it in the target area"
      style="flex:1;background:#f7f6f2;border:1px solid #b3b9a8;border-radius:6px;
             color:#242722;padding:7px 10px;font:13px system-ui">
    <button {command_disabled} style="background:#b4e35a;border:0;border-radius:6px;color:#242722;
                   font-weight:700;padding:0 14px;cursor:pointer">Send</button>
    <button type="button" id="stopbtn" {command_disabled}
      style="background:#fff0e9;border:0;border-radius:6px;color:#833b24;
             font-weight:700;padding:0 14px;cursor:pointer">Stop</button>
   </form><div id="chatmsg" style="font:11px ui-monospace,monospace;color:#65695e;
                                   margin-top:6px;max-height:14vh;overflow-y:auto"></div></div>
  <div class="panel runtime-panel" {command_hidden}><h3>Robot activity</h3><div id="feed">Waiting for the next action…</div></div>
  <div class="panel runtime-panel" {command_hidden}><h3>What the robot sees
   <button id="anbtn" style="float:right;background:#f7f6f2;border:1px solid #b3b9a8;
     color:#65695e;border-radius:5px;font:11px ui-monospace,monospace;
     padding:2px 8px;cursor:pointer">Analyze</button></h3>
   <div id="analysis" style="font:12px/1.5 ui-monospace,monospace;
        white-space:pre-wrap;max-height:34vh;overflow-y:auto">Click Analyze to inspect the current scene and identify its objects.</div></div>
  <div class="panel runtime-panel" {command_hidden}><h3>Objects in view</h3>
   <table><thead><tr><th>Object</th><th>Color</th><th>Position (m)</th><th>State</th></tr></thead>
   <tbody id="objs"></tbody></table></div>
  <div class="panel runtime-panel" {command_hidden}><h3>Learning from previous grasps</h3>
   <div id="gmem" style="font:12px/1.55 ui-monospace,monospace;
        white-space:pre-wrap">empty</div></div>
 </aside>
</main>
<script>
 let readOnly = {read_only_json};
 let currentState = null, transportFailed = false;
 const reconnectTimers = new Map();
 function updateCameraStatus() {{
   if (transportFailed) {{
     document.getElementById('status').textContent = 'Camera connection lost · reconnecting automatically';
   }} else if (currentState) {{
     document.getElementById('status').textContent = statusSummary(currentState);
   }}
 }}
 function reconnectStream(img) {{
   clearTimeout(reconnectTimers.get(img)); reconnectTimers.delete(img);
   const url = new URL(img.src, location.href);
   url.searchParams.set('reconnect', Date.now());
   img.src = url.href;
 }}
 function cameraTransportFailed() {{
   transportFailed = true;
   document.getElementById('cams').dataset.transport = 'offline';
   document.querySelectorAll('.cam').forEach(tile => tile.dataset.streamState = 'offline');
   updateCameraStatus();
 }}
 document.querySelectorAll('.cam img').forEach(img => {{
   img.addEventListener('load', () => {{
     img.closest('.cam').dataset.streamState = 'connected';
     clearTimeout(reconnectTimers.get(img)); reconnectTimers.delete(img);
     updateCameraStatus();
   }});
   img.addEventListener('error', () => {{
     img.closest('.cam').dataset.streamState = 'offline';
     updateCameraStatus();
     clearTimeout(reconnectTimers.get(img));
     reconnectTimers.set(img, setTimeout(() => {{
       if (!transportFailed && navigator.onLine) reconnectStream(img);
     }}, 1500));
   }});
 }});
 window.addEventListener('offline', cameraTransportFailed);
 window.addEventListener('pageshow', event => {{ if (event.persisted) cameraTransportFailed(); }});
 document.addEventListener('visibilitychange', () => {{
   if (document.visibilityState === 'visible') cameraTransportFailed();
 }});
 const CSS = {{observation:'obs', action:'act', outcome:'out', note:'note'}};
 const SWATCH = {{red:'#e5484d', orange:'#f76b15', yellow:'#ffe629', green:'#46a758',
   cyan:'#00a2c7', blue:'#0090ff', purple:'#8e4ec6', pink:'#f76190',
   brown:'#ad7f58', white:'#eee', gray:'#888', black:'#111'}};
 // Per-camera view switch: rgb (detections) | depth (colormap + stats) |
 // agent (VIA marks + metric grid + configured display band). Swapping the <img>
 // src tears down the old MJPEG socket, so only one stream per tile is ever
 // encoding -- that is what keeps 3 cameras x 3 views affordable.
 function setView(cam, mode, btn) {{
   const img = document.getElementById('img-' + cam);
   const route = mode === 'depth' ? '/depth/' : (mode === 'agent' ? '/annotated/' : '/stream/');
   img.src = route + encodeURIComponent(cam) + '?t=' + Date.now();
   btn.parentNode.querySelectorAll('button').forEach(b => b.classList.remove('on'));
   btn.classList.add('on');
 }}
 document.getElementById('anbtn').addEventListener('click', async () => {{
   const out = document.getElementById('analysis');
   out.textContent = 'analyzing...';
   try {{
     const a = await (await fetch('/analyze')).json();
     const lines = [];
     for (const [name, c] of Object.entries(a.cameras || {{}})) {{
       if (c.error) {{ lines.push(`${{name}}: ${{c.error}}`); continue; }}
       lines.push(`${{name}}  ${{c.resolution.join('x')}} @ ${{c.fps}}fps  depth=${{c.depth_source}}`);
       if (c.depth) lines.push(`  depth  min ${{c.depth.min_m}}m  med ${{c.depth.median_m}}m  max ${{c.depth.max_m}}m  (${{Math.round(c.depth.valid_fraction*100)}}% valid)`);
       if (c.depth_warning) lines.push(`  WARNING ${{c.depth_warning}}`);
       lines.push('  detections: ' + ((c.detections || []).map(d => `${{d.label}} ${{d.conf}}`).join(', ') || 'none'));
     }}
     if (a.world && a.world.description) lines.push('', 'scene: ' + a.world.description);
     if (a.annotated_key) lines.push('', a.annotated_key);
     out.textContent = lines.join('\\n');
   }} catch (e) {{ out.textContent = 'analyze failed: ' + e; }}
 }});
 function log(text, color) {{
   const box = document.getElementById('chatmsg');
   const line = document.createElement('div');
   line.textContent = text;
   if (color) line.style.color = color;
   box.appendChild(line);
   while (box.childNodes.length > 8) box.removeChild(box.firstChild);
   box.scrollTop = box.scrollHeight;
 }}
 document.getElementById('stopbtn').addEventListener('click', async () => {{
   if (readOnly) return;
   try {{
     const r = await (await fetch('/cancel', {{method: 'POST'}})).json();
     log(r.cancelled ? 'cancelled - the arm is stopping' : (r.error || 'cancel failed'),
         r.cancelled ? '#795a29' : '#aa3523');
   }} catch (e) {{ log('error: ' + e, '#aa3523'); }}
 }});
 document.getElementById('chat').addEventListener('submit', async (ev) => {{
   ev.preventDefault();
   if (readOnly) return;
   const box = document.getElementById('cmd');
   const task = box.value.trim();
   if (!task) return;
   log('> ' + task, '#242722');
   box.value = '';                       // clear immediately: the chat stays
   try {{                                 // usable while the arm works
     const r = await (await fetch('/task', {{method: 'POST',
       headers: {{'Content-Type': 'application/json'}},
       body: JSON.stringify({{task}})}})).json();
     log(r.accepted ? `running: ${{r.task}}` : (r.error || 'rejected'),
         r.accepted ? '#436123' : '#aa3523');
     if (!r.accepted) box.value = task;   // give a rejected command back
   }} catch (e) {{ log('error: ' + e, '#aa3523'); box.value = task; }}
 }});
 function statusSummary(s) {{
   if (s.read_only === true) {{
     const cameras = Object.entries(s.cameras || {{}});
     const online = cameras.filter(([name, camera]) => {{
       const img = document.getElementById('img-' + name);
       return camera.online === true && img?.naturalWidth > 0
         && img.closest('.cam').dataset.streamState !== 'offline';
     }}).length;
     return cameras.length && online === cameras.length
       ? `${{online}} camera feeds connected · check the guide for current snapshots`
       : `${{online}} of ${{cameras.length}} views connected · cameras reconnect automatically`;
   }}
   const parts = [`${{Object.keys(s.cameras || {{}}).length}} cams`];
   const hasHolding = Object.prototype.hasOwnProperty.call(s, 'holding');
   const hasArm = typeof s.arm_connected === 'boolean';
   if (hasHolding) parts.push(`holding: ${{s.holding || '-'}}`);
   if (hasArm) parts.push(`arm: ${{s.arm_connected ? 'up' : 'standby'}}`);
   if (s.last_path) parts.push(`via: ${{s.last_path}}`);
   if (!hasHolding && !hasArm) parts.push('Robot state unavailable');
   if (s.agent_status) parts.push(s.agent_status);
   return parts.join(' | ');
 }}
 async function tick() {{
   const controller = new AbortController();
   const timeout = setTimeout(() => controller.abort(), 2000);
   try {{
     const response = await fetch('/state', {{signal: controller.signal, cache: 'no-store'}});
     if (!response.ok) throw Error('Camera status unavailable');
     const s = await response.json();
     if (!s || typeof s.cameras !== 'object' || !s.cameras) throw Error('Camera status missing');
     const recovering = transportFailed;
     transportFailed = false; currentState = s;
     document.getElementById('cams').dataset.transport = 'connected';
     if (recovering) document.querySelectorAll('.cam img').forEach(reconnectStream);
     readOnly = s.read_only === true;
     document.getElementById('read-only-panel').hidden = !readOnly;
     document.getElementById('command-panel').hidden = readOnly;
     document.querySelectorAll('.runtime-panel').forEach(el => el.hidden = readOnly);
     document.querySelectorAll('#chat input, #chat button').forEach(el => el.disabled = readOnly);
     document.getElementById('task').textContent = readOnly ? 'Live cameras · send orders in OpenClaw' :
       (s.task ? `task: ${{s.task}}` : 'idle - waiting for a command');
     updateCameraStatus();
     document.getElementById('gmem').textContent =
       (s.grasp_memory || []).join('\\n') || 'empty';
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
   }} catch (e) {{ cameraTransportFailed(); }}
   finally {{ clearTimeout(timeout); }}
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
        keyframes_dir=None,
        runtime_fn=None,
        depth_max_m: float = 2.0,
        on_poll=None,
    ):
        self._rig = rig
        self._state_fn = state_fn or (lambda: {})
        # Lazily resolved SkillRuntime, used by the depth/annotated/analyze
        # views. A callable (not the object) because the MCP server builds its
        # runtime AFTER the server may already exist.
        self._runtime_fn = runtime_fn
        self._depth_max_m = float(depth_max_m)
        # Called on every browser request so LiveViewController can auto-close
        # an idle dashboard instead of re-encoding JPEGs into the void.
        self._on_poll = on_poll
        # per-skill before/after evidence JPEGs (the run dir's keyframes/);
        # None disables the /keyframes routes
        self._keyframes_dir = Path(keyframes_dir) if keyframes_dir else None
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

    # ── depth + annotated views (the diagnostic surfaces) ────────────────

    def depth_jpeg(self, name: str) -> bytes | None:
        """Depth as a colormap, labelled with its provenance.

        `depth_source` degrades silently (sensor -> mono -> plane -> none) and
        a bad grasp z is usually a depth problem you cannot see in RGB. The
        label is drawn ON the image so a screenshot is self-describing.
        """
        from .live_view import depth_colormap

        stream = self._rig.get(name)
        frame = stream.latest()
        if frame is None:
            return None
        if not frame.has_depth or frame.depth_m is None:
            img = np.zeros((360, 640, 3), dtype=np.uint8)
            draw_hud(img, [f"{name}: NO DEPTH",
                           "grounding cannot localize from this camera"])
        else:
            depth = np.asarray(frame.depth_m, dtype=np.float32)
            img = depth_colormap(depth, max_m=float(self._depth_max_m))
            valid = depth[np.isfinite(depth) & (depth > 0)]
            if valid.size:
                draw_hud(img, [
                    f"{name}  depth: {frame.depth_source}  range {self._depth_max_m:.1f} m",
                    f"min {valid.min():.3f}  median {np.median(valid):.3f}  "
                    f"max {valid.max():.3f} m  ({100.0 * valid.size / depth.size:.0f}% valid)",
                ])
            else:
                draw_hud(img, [f"{name}  depth: {frame.depth_source}",
                               "no valid depth samples in this frame"])
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, self._quality])
        return buf.tobytes() if ok else None

    def annotated_view_jpeg(self, name: str) -> bytes | None:
        """VIA-style agent view: marks, grid, optional configured display band.

        Requires the runtime (for beliefs/extrinsics/config); without it we
        fall back to the plain detection view rather than erroring, so the
        route always renders something.
        """
        runtime = self._runtime_fn() if self._runtime_fn else None
        if runtime is None:
            return self.annotated_jpeg(name)
        try:
            from ..perception.visual_interface import VisualInterface, configured_grasp_band

            stream = self._rig.get(name)
            frame = stream.latest()
            if frame is None:
                return None
            cfg = runtime.cfg
            ws_raw = cfg.safety.get("workspace", {}) or {}
            grasp_cfg = cfg.grasp if hasattr(cfg, "grasp") else {}
            vi = VisualInterface(
                extrinsics=runtime.extrinsics,
                workspace=dict(getattr(ws_raw, "_data", ws_raw)),
                table_z=float(cfg.safety.get("table_z", 0.0)),
                reach_x=configured_grasp_band(grasp_cfg),
            )
            try:
                tcp = runtime._tcp()
            except Exception:
                tcp = None
            img, _ = vi.render(frame, beliefs=list(runtime.beliefs.all()), tcp=tcp)
            ok, buf = cv2.imencode(".jpg", img,
                                   [cv2.IMWRITE_JPEG_QUALITY, self._quality])
            return buf.tobytes() if ok else None
        except Exception:
            # a perception hiccup must never break the live view
            return self.annotated_jpeg(name)

    def analyze(self, camera: str | None = None) -> dict:
        """One-shot perception report: detections + depth + description.

        This is what the "analyze" button and the `analyze_scene` MCP tool
        both call. It reads only the latest cached frames, so it costs no
        extra capture and cannot disturb a motion in flight.
        """
        runtime = self._runtime_fn() if self._runtime_fn else None
        out: dict = {"ok": True, "t": time.time(), "cameras": {}}
        names = [camera] if camera else list(self._rig.names)
        for name in names:
            try:
                stream = self._rig.get(name)
            except KeyError:
                out["cameras"][name] = {"error": "unknown camera"}
                continue
            frame = stream.latest()
            if frame is None:
                out["cameras"][name] = {"error": "no frame yet"}
                continue
            dets, status = stream.overlay()
            entry: dict = {
                "fps": round(float(stream.fps), 1),
                "resolution": [int(frame.rgb.shape[1]), int(frame.rgb.shape[0])],
                "agent_status": status,
                "depth_source": frame.depth_source,
                "detections": [
                    {"label": getattr(d, "label", "?"),
                     "conf": round(float(getattr(d, "conf", 0.0)), 3)}
                    for d in dets
                ],
            }
            if frame.has_depth and frame.depth_m is not None:
                depth = np.asarray(frame.depth_m, dtype=np.float32)
                valid = depth[np.isfinite(depth) & (depth > 0)]
                entry["depth"] = {
                    "valid_fraction": round(float(valid.size) / float(depth.size), 3),
                    "min_m": round(float(valid.min()), 3) if valid.size else None,
                    "median_m": round(float(np.median(valid)), 3) if valid.size else None,
                    "max_m": round(float(valid.max()), 3) if valid.size else None,
                }
            else:
                entry["depth"] = None
                entry["depth_warning"] = (
                    "no depth on this camera: 3D grounding will refuse to localize"
                )
            out["cameras"][name] = entry

        if runtime is not None:
            try:
                out["world"] = runtime.execute("describe_scene", {})
            except Exception as e:
                out["world"] = {"ok": False, "error": str(e)[:200]}
            try:
                from ..perception.visual_interface import VisualInterface, annotate_frame

                _, marks = annotate_frame(runtime)
                out["annotated_key"] = VisualInterface.describe(marks)
                out["objects"] = [m.as_dict() for m in marks]
            except Exception:
                pass
        return out


def _make_handler(server: StreamServer):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):  # stderr chatter only on errors
            pass

        # ── routes ───────────────────────────────────────────────────────

        def do_GET(self):  # noqa: N802 (BaseHTTPRequestHandler API)
            path = self.path.split("?", 1)[0].rstrip("/") or "/"
            if server._on_poll is not None:
                try:
                    server._on_poll()  # keeps a lazily-opened view alive
                except Exception:
                    pass
            try:
                if path == "/":
                    return self._index()
                if path == "/state":
                    return self._json(server.state())
                if path == "/analyze":
                    cam = None
                    if "?" in self.path:
                        from urllib.parse import parse_qs, urlparse

                        cam = (parse_qs(urlparse(self.path).query).get("camera") or [None])[0]
                    return self._json(server.analyze(cam))
                if path.startswith("/snapshot/"):
                    name = path.split("/", 2)[2].removesuffix(".jpg")
                    return self._snapshot(name)
                if path.startswith("/stream/"):
                    return self._mjpeg(path.split("/", 2)[2], server.annotated_jpeg)
                if path.startswith("/depth/"):
                    return self._mjpeg(path.split("/", 2)[2], server.depth_jpeg)
                if path.startswith("/annotated/"):
                    return self._mjpeg(path.split("/", 2)[2], server.annotated_view_jpeg)
                if path == "/keyframes":
                    return self._keyframes_index()
                if path.startswith("/keyframe/"):
                    return self._keyframe(path.split("/", 2)[2])
                self.send_error(404, "unknown path")
            except (BrokenPipeError, ConnectionResetError):
                pass  # viewer closed the tab mid-frame
            except KeyError as e:
                self.send_error(404, str(e))

        def do_POST(self):  # noqa: N802 (BaseHTTPRequestHandler API)
            # API endpoints always answer JSON (even on error) so the
            # dashboard's fetch().json() never chokes on an HTML error page.
            if self.path.rstrip("/") == "/cancel":
                if server._cancel_fn is None:
                    return self._json({"cancelled": False,
                                       "error": "no cancel handler wired"}, code=501)
                try:
                    server._cancel_fn()
                    return self._json({"cancelled": True})
                except Exception as e:
                    return self._json({"cancelled": False, "error": str(e)[:120]})
            if self.path.rstrip("/") != "/task":
                return self._json({"error": "unknown path"}, code=404)
            if server._task_fn is None:
                return self._json({"accepted": False,
                                   "error": "no task handler wired (REPL mode drives tasks)"},
                                  code=501)
            try:
                n = int(self.headers.get("Content-Length", 0))
                task = str(json.loads(self.rfile.read(n)).get("task", "")).strip()
            except (ValueError, json.JSONDecodeError):
                return self._json({"accepted": False,
                                   "error": 'body must be JSON {"task": ...}'}, code=400)
            if not task:
                return self._json({"accepted": False, "error": "empty task"}, code=400)
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
            read_only = server.state().get("read_only") is True
            tiles = "".join(
                f'<div class="cam"><h2>{n}'
                f'<span class="views">'
                f"<button class=\"on\" onclick=\"setView('{n}','rgb',this)\">rgb</button>"
                f"<button {'hidden' if read_only else ''} onclick=\"setView('{n}','depth',this)\">depth</button>"
                f"<button {'hidden' if read_only else ''} onclick=\"setView('{n}','agent',this)\">agent</button>"
                f'</span></h2>'
                f'<img id="img-{n}" src="/stream/{n}" alt="{n}"></div>'
                for n in server._rig.names
            )
            body = _INDEX_HTML.format(
                tiles=tiles, read_only_json=json.dumps(read_only),
                command_hidden="hidden" if read_only else "",
                read_only_hidden="" if read_only else "hidden",
                command_disabled="disabled" if read_only else "",
                initial_task="Live cameras · send orders in OpenClaw" if read_only else "no task",
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, payload: dict, code: int = 200):
            body = json.dumps(payload, default=_np_safe).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _keyframes_index(self):
            kd = server._keyframes_dir
            if kd is None or not kd.is_dir():
                return self.send_error(404, "keyframes not available")
            try:
                files = sorted(kd.glob("*.jpg"), key=lambda p: p.name,
                               reverse=True)
            except OSError:  # dir vanished mid-request (run dir rotated)
                return self.send_error(404, "keyframes not available")
            cells = "".join(
                f'<figure style="margin:0"><img src="/keyframe/{p.name}" '
                f'style="max-width:340px;border-radius:4px;display:block">'
                f'<figcaption style="font:11px ui-monospace,monospace;'
                f'color:#65695e">{p.name}</figcaption></figure>'
                for p in files[:24]
            ) or "<p>no keyframes yet</p>"
            body = (
                '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="color-scheme" content="light">'
                '<meta http-equiv="refresh" content="3">'
                "<title>OpenClaw Demo · Keyframes</title></head>"
                '<body style="background:#f7f6f2;color:#242722;'
                'font:14px system-ui;padding:14px">'
                '<h1 style="font-size:16px;color:#436123">'
                'per-skill before/after keyframes (newest first) &middot; '
                '<a href="/" style="color:#65695e">dashboard</a></h1>'
                f'<div style="display:flex;flex-wrap:wrap;gap:12px">{cells}'
                "</div></body></html>"
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _keyframe(self, name: str):
            kd = server._keyframes_dir
            # basename-only, .jpg-only: no path traversal out of the run dir
            if (kd is None or not name.endswith(".jpg")
                    or name != Path(name).name):
                return self.send_error(404, "no such keyframe")
            p = kd / name
            try:
                data = p.read_bytes()
            except OSError:  # missing, or vanished between checks
                return self.send_error(404, "no such keyframe")
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

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

        def _mjpeg(self, name: str, render=None):
            server._rig.get(name)  # 404 (KeyError) before headers go out
            render = render or server.annotated_jpeg
            self.send_response(200)
            self.send_header(
                "Content-Type", f"multipart/x-mixed-replace; boundary={_BOUNDARY}"
            )
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            period = 1.0 / max(server._fps, 1.0)
            while not server._stopping:
                t0 = time.monotonic()
                jpeg = render(name)
                if jpeg is not None:
                    self.wfile.write(
                        b"--" + _BOUNDARY.encode() + b"\r\n"
                        b"Content-Type: image/jpeg\r\n"
                        + f"Content-Length: {len(jpeg)}\r\n\r\n".encode()
                    )
                    self.wfile.write(jpeg)
                    self.wfile.write(b"\r\n")
                    self.wfile.flush()
                # An open MJPEG socket is itself proof someone is watching:
                # refresh the idle timer or the reaper closes the view under a
                # viewer who never issues another GET.
                if server._on_poll is not None:
                    try:
                        server._on_poll()
                    except Exception:
                        pass
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
