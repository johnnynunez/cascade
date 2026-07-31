"""Lazy live-view lifecycle: headless by default, browser UI on demand.

The primary UI is the chat client (Hermes / OpenClaw / any MCP host). Cameras
and the browser dashboard are *diagnostic surfaces*, not the interface, so
running them all the time is pure cost: every open MJPEG stream re-encodes
JPEGs at the stream rate whether or not a human is looking, and a bound port
on a booth LAN is an attack surface nobody asked for.

This controller makes the dashboard behave like a debugger you attach:

    mode "lazy"  (default) -- nothing is bound at startup. The first
                 open() binds the port, starts serving, and returns a URL.
                 An idle timer closes it again once no browser has polled
                 for `idle_timeout_s`, so a forgotten tab does not keep the
                 encoder running forever.
    mode "eager"          -- classic behaviour: bind at startup (booth mode,
                 where the big screen must be live before doors open).
    mode "off"            -- never bind. `WRC_STREAM=0` maps here, and
                 open() reports honestly instead of silently ignoring it.

Perception itself is unaffected: the CameraRig keeps pumping frames and the
WorldWatcher keeps the belief store warm regardless of whether anyone is
watching. Only *serving* is deferred. That matters because the agent must be
able to answer "what do you see?" instantly, without a dashboard.

Thread-safety: open/close are called from MCP tool threads while the HTTP
server runs its own threads; every transition is taken under one lock, and
the idle reaper is a daemon thread that only ever calls close().
"""

from __future__ import annotations

import threading
import time

#: How long after the last browser poll to auto-close a lazily-opened view.
DEFAULT_IDLE_TIMEOUT_S = 900.0  # 15 min: one booth session

MODE_LAZY = "lazy"
MODE_EAGER = "eager"
MODE_OFF = "off"
_MODES = (MODE_LAZY, MODE_EAGER, MODE_OFF)


class LiveViewController:
    """Owns the StreamServer's lifecycle without owning its implementation.

    ``factory()`` builds a fresh, unstarted ``StreamServer`` -- it is called
    on every open so a re-open after a close gets a clean server (a
    ``ThreadingHTTPServer`` cannot be restarted once ``server_close()`` ran).
    """

    def __init__(
        self,
        factory,
        mode: str = MODE_LAZY,
        idle_timeout_s: float = DEFAULT_IDLE_TIMEOUT_S,
    ):
        if mode not in _MODES:
            mode = MODE_LAZY
        self._factory = factory
        self.mode = mode
        self.idle_timeout_s = float(idle_timeout_s)
        self._server = None
        self._lock = threading.RLock()
        self._last_poll = 0.0
        self._opened_at = 0.0
        self._reaper: threading.Thread | None = None
        self._stop_reaper = threading.Event()
        #: callbacks handed to each freshly built server (chat wiring survives
        #: close/open cycles -- otherwise the second open has a dead chat box)
        self._task_fn = None
        self._cancel_fn = None

    # ── wiring that must survive re-opens ────────────────────────────────

    def set_task_fn(self, fn) -> None:
        self._task_fn = fn
        with self._lock:
            if self._server is not None:
                self._server.set_task_fn(fn)

    def set_cancel_fn(self, fn) -> None:
        self._cancel_fn = fn
        with self._lock:
            if self._server is not None:
                self._server.set_cancel_fn(fn)

    # ── state ────────────────────────────────────────────────────────────

    @property
    def server(self):
        """The live StreamServer, or None when closed."""
        with self._lock:
            return self._server

    @property
    def is_open(self) -> bool:
        return self.server is not None

    @property
    def url(self) -> str | None:
        srv = self.server
        return srv.url if srv is not None else None

    def note_poll(self) -> None:
        """Called by the HTTP layer on every browser request (keeps it alive)."""
        self._last_poll = time.monotonic()

    def status(self) -> dict:
        with self._lock:
            server = self._server
            if server is None:
                return {
                    "open": False,
                    "mode": self.mode,
                    "url": None,
                    "idle_timeout_s": self.idle_timeout_s if self.mode == MODE_LAZY else None,
                    "open_for_s": None,
                    "idle_for_s": None,
                }
            return {
                "open": True,
                "mode": self.mode,
                "url": server.url,
                "idle_timeout_s": self.idle_timeout_s if self.mode == MODE_LAZY else None,
                "open_for_s": round(time.monotonic() - self._opened_at, 1),
                "idle_for_s": (
                    round(time.monotonic() - self._last_poll, 1) if self._last_poll else None
                ),
            }

    # ── lifecycle ────────────────────────────────────────────────────────

    def open(self, reason: str = "") -> dict:
        """Start serving (idempotent). Returns a JSON-able status dict."""
        with self._lock:
            if self.mode == MODE_OFF:
                return {
                    "ok": False,
                    "open": False,
                    "error": (
                        "live view is disabled for this session (stream.mode: off "
                        "or WRC_STREAM=0). Restart with WRC_STREAM=1 to allow it."
                    ),
                }
            if self._server is not None:
                self.note_poll()
                return {"ok": True, "open": True, "url": self._server.url,
                        "note": "already open"}
            try:
                server = self._factory()
                if self._task_fn is not None:
                    server.set_task_fn(self._task_fn)
                if self._cancel_fn is not None:
                    server.set_cancel_fn(self._cancel_fn)
                server.start()
            except OSError as e:
                return {
                    "ok": False,
                    "open": False,
                    "error": f"could not bind the live-view port: {e} "
                             "(set stream.port or WRC_STREAM_PORT)",
                }
            self._server = server
            self._opened_at = time.monotonic()
            self.note_poll()
            self._start_reaper()
            return {
                "ok": True,
                "open": True,
                "url": server.url,
                "reason": reason or None,
                "auto_close_after_s": (
                    self.idle_timeout_s if self.mode == MODE_LAZY else None
                ),
            }

    def close(self, reason: str = "") -> dict:
        """Stop serving and release the port (idempotent)."""
        with self._lock:
            if self._server is None:
                return {"ok": True, "open": False, "note": "already closed"}
            server = self._server
            self._server = None
        self._stop_reaper.set()
        try:
            server.stop()
        except Exception as e:
            return {"ok": False, "open": False, "error": f"stop failed: {e}"}
        return {"ok": True, "open": False, "reason": reason or "closed"}

    # ── idle reaping ─────────────────────────────────────────────────────

    def _start_reaper(self) -> None:
        """Auto-close a lazily-opened view once nobody is watching.

        Only in lazy mode: an eager/booth dashboard must stay up even when
        the audience wanders off between sessions.
        """
        if self.mode != MODE_LAZY or self.idle_timeout_s <= 0:
            return
        self._stop_reaper.clear()
        self._reaper = threading.Thread(
            target=self._reap_loop, daemon=True, name="live-view-reaper"
        )
        self._reaper.start()

    def _reap_loop(self) -> None:
        while not self._stop_reaper.wait(timeout=15.0):
            if not self.is_open:
                return
            idle = time.monotonic() - self._last_poll
            if idle >= self.idle_timeout_s:
                self.close(reason=f"auto-closed after {idle:.0f}s idle")
                return


def resolve_mode(cfg_stream, env_get) -> tuple[str, float]:
    """Decide (mode, idle_timeout) from config + environment.

    Precedence, most explicit first:
      WRC_STREAM=0            -> off   (hard kill switch; tests rely on it)
      WRC_STREAM=eager|lazy|off -> that mode
      stream.mode in config   -> that mode
      legacy stream.enabled=false -> off
      default                 -> lazy  (headless-first: chat is the UI)
    """
    raw_env = (env_get("WRC_STREAM") or "").strip().lower()
    idle = float((cfg_stream.get("idle_timeout_s", DEFAULT_IDLE_TIMEOUT_S)
                  if cfg_stream else DEFAULT_IDLE_TIMEOUT_S) or 0.0)
    if raw_env in ("0", "off", "false", "no"):
        return MODE_OFF, idle
    if raw_env in _MODES:
        return raw_env, idle
    if raw_env in ("1", "on", "true", "yes"):
        return MODE_EAGER, idle
    mode = str((cfg_stream.get("mode") if cfg_stream else "") or "").strip().lower()
    if mode in _MODES:
        return mode, idle
    if cfg_stream is not None and not bool(cfg_stream.get("enabled", True)):
        return MODE_OFF, idle
    return MODE_LAZY, idle
