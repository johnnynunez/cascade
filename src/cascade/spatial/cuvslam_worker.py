"""Owned cuVSLAM process: the native binding may hold Python's GIL.

One bounded request is in flight. A deadline quarantines and terminates this
worker, never starts a replacement behind the same map identity.
"""
from __future__ import annotations

import multiprocessing
import pickle
import queue
import threading
import time
from types import SimpleNamespace


def _run(connection, settings):
    """Only this child imports the optional SDK. No actuator or sensor handle."""
    try:
        from .cuvslam import _tracker
        camera, binding, gap, capacity = settings
        tracker = _tracker(SimpleNamespace(**camera), binding, gap, capacity)
        connection.send({"ok": True})

        def pose(value):
            return None if value is None else {
                "translation": tuple(value.translation), "rotation": tuple(value.rotation)}

        while True:
            request = pickle.loads(connection.recv_bytes(8 * 1024 * 1024))
            odom, slam = tracker.track(request["timestamp"], images=[request["rgb"]],
                                       depths=[request["depth"]])
            connection.send({"ok": True, "timestamp_ns": odom.timestamp_ns,
                "odometry": pose(odom.world_from_rig.pose) if odom.world_from_rig is not None else None,
                "slam": pose(slam)})
    except (EOFError, BrokenPipeError):
        pass
    except BaseException as exc:
        try:
            connection.send({"ok": False, "error": f"{type(exc).__name__}: {exc}"[:500]})
        except (OSError, EOFError):
            pass
    finally:
        connection.close()


class CuVslamProcess:
    def __init__(self, settings, *, timeout_s, target=_run):
        self.timeout_s = timeout_s
        context = multiprocessing.get_context("spawn")
        self._connection, child = context.Pipe()
        self._process = context.Process(target=target, args=(child, settings), daemon=True,
                                        name="cuvslam-localization")
        self._closed = threading.Event()
        self._closing = threading.Lock()
        self._requests = queue.Queue(maxsize=1)
        self._thread = threading.Thread(target=self._exchange, daemon=True, name="cuvslam-ipc")
        try:
            self._process.start()
            self._thread.start()
        except BaseException as exc:
            # Preserve ownership even if bounded cleanup is incomplete.
            exc.cuvslam_worker = self
            try:
                self.close()
            except BaseException:
                pass
            raise
        finally:
            child.close()

    def _exchange(self):
        try:
            while not self._closed.is_set():
                item = self._requests.get()
                if item is None:
                    break
                request, done, result = item
                try:
                    if request is not None:
                        self._connection.send(request)
                    # Only owned local primitive pose responses, bounded before
                    # unpickling. The caller's deadline also covers pipe writes.
                    value = pickle.loads(self._connection.recv_bytes(4096))
                    if not isinstance(value, dict) or value.get("ok") is not True:
                        raise RuntimeError(str(value.get("error", "invalid SDK response"))[:500])
                    result["value"] = value
                except BaseException as exc:
                    result["error"] = f"{type(exc).__name__}: {exc}"[:500]
                finally:
                    done.set()
                del request, done, result, item
        finally:
            self._connection.close()

    def _call(self, request):
        if self._closed.is_set():
            raise RuntimeError("cuVSLAM worker closed")
        done, result = threading.Event(), {}
        deadline = time.monotonic() + self.timeout_s
        self._requests.put_nowait((request, done, result))
        while not done.is_set() and not self._closed.is_set():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self.request_stop()
                raise TimeoutError("cuVSLAM worker deadline expired; map session invalidated")
            done.wait(min(.05, remaining))
        if time.monotonic() >= deadline:
            self.request_stop()
            raise TimeoutError("cuVSLAM response arrived after deadline; map session invalidated")
        if self._closed.is_set() or "error" in result:
            self.request_stop()
            raise RuntimeError(result.get("error", "cuVSLAM worker closed during call"))
        return result["value"]

    def warmup(self):
        self._call(None)

    def track(self, timestamp, *, images, depths):
        result = self._call({"timestamp": timestamp, "rgb": images[0], "depth": depths[0]})
        def pose(value):
            return None if value is None else SimpleNamespace(**value)
        odometry = pose(result["odometry"])
        return (SimpleNamespace(timestamp_ns=result["timestamp_ns"],
                    world_from_rig=None if odometry is None else SimpleNamespace(pose=odometry)),
                pose(result["slam"]))

    def request_stop(self):
        self._closed.set()
        # Process liveness/reaping and signalling are serialized; no unrelated
        # PID, process group, service or inherited CUDA context is touched.
        with self._closing:
            if self._process.is_alive():
                self._process.terminate()
        try:
            self._requests.put_nowait(None)
        except queue.Full:
            pass

    def close(self, timeout_s=.5):
        deadline = time.monotonic() + timeout_s
        self.request_stop()
        with self._closing:
            if self._process.pid is not None:
                self._process.join(max(0., min(.1, deadline - time.monotonic())))
                if self._process.is_alive():
                    self._process.kill()
                    self._process.join(max(0., deadline - time.monotonic()))
            exited = not self._process.is_alive()
        if self._thread.ident is not None:
            self._thread.join(max(0., deadline - time.monotonic()))
        else:
            self._connection.close()
        return exited and not self._thread.is_alive()
