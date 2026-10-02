"""One owned OVRTX child, isolated from Kit's native USD/renderer libraries.

The inherited socket is private to this parent/child pair. Only one request
can be outstanding; no unbounded snapshot queue or network listener exists.
"""
from __future__ import annotations

import os
from pathlib import Path
import socket
import subprocess
import threading
from multiprocessing.connection import Connection

from .ovrtx_renderer import OvrtxError


class OvrtxProcess:
    def __init__(self, python, renderer_config, *, startup_timeout_s=180., frame_timeout_s=15.):
        self._lock = threading.Lock()
        self._closed = False
        self._frame_timeout = frame_timeout_s
        self._startup_timeout = startup_timeout_s
        self._first_frame = True
        parent, child = socket.socketpair()
        self._connection = Connection(parent.detach())
        env = dict(os.environ)
        # Kit injects incompatible Python and C++ extension search paths.
        for key in ("PYTHONHOME", "LD_LIBRARY_PATH", "LD_PRELOAD"):
            env.pop(key, None)
        env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])
        try:
            self.process = subprocess.Popen(
                [str(python), "-m", "cascade.sim.ovrtx_worker", "--fd", str(child.fileno())],
                env=env, pass_fds=(child.fileno(),), stdin=subprocess.DEVNULL)
        except BaseException:
            self._connection.close()
            raise
        finally:
            child.close()
        try:
            self._request(("open", renderer_config), startup_timeout_s)
        except BaseException:
            self.close()
            raise

    def _request(self, value, timeout):
        if self._closed:
            raise OvrtxError("OVRTX process is closed")
        try:
            self._connection.send(value)
            if not self._connection.poll(timeout):
                raise OvrtxError("OVRTX process exceeded its response deadline")
            ok, result = self._connection.recv()
            if not ok:
                raise OvrtxError(result)
            return result
        except (EOFError, OSError) as exc:
            raise OvrtxError("OVRTX process ended before a complete response") from exc

    def render(self, snapshot):
        with self._lock:
            try:
                result = self._request(("render", snapshot),
                                       self._startup_timeout if self._first_frame else self._frame_timeout)
                self._first_frame = False
                return result
            except BaseException:
                self.close()
                raise

    def close(self):
        if self._closed:
            return
        self._closed = True
        if self.process.poll() is None:
            try:
                self._connection.send(("close", None))
            except (OSError, EOFError):
                pass
        self._connection.close()  # EOF asks the idle worker to destroy its owner.
        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
