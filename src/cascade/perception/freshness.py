"""Bounded frame barriers using producer clocks without comparing host clocks."""
from __future__ import annotations

import math
import time


def capture_marker(frame) -> dict:
    """Identity and clock of one capture, or a local backend's frame creation."""
    capture = getattr(frame, "capture", None)
    if isinstance(capture, dict) and capture.get("backend") == "isaac":
        state = capture.get("proprioception") or {}
        source = capture.get("source")
        timestamp = capture.get("t")
        if (not isinstance(state, dict) or not isinstance(source, (list, tuple))
                or len(source) != 2 or not isinstance(source[0], str)
                or type(source[1]) is not int or not isinstance(capture.get("camera"), str)
                or not capture["camera"] or not isinstance(state.get("robot_id"), str)
                or not state["robot_id"] or state.get("backend") != "isaac"
                or state.get("time_source") != "physics_loop_monotonic"
                or type(timestamp) not in (int, float) or not math.isfinite(timestamp)
                or state.get("t") != timestamp):
            raise ValueError("Isaac frame lacks a valid producer capture identity/clock")
        return {"channel": "producer_capture", "backend": "isaac", "source": list(source),
                "camera": capture["camera"], "robot_id": state["robot_id"],
                "clock": state["time_source"], "t": timestamp}
    timestamp = getattr(frame, "t", None)
    if type(timestamp) not in (int, float) or not math.isfinite(timestamp):
        raise ValueError("Camera frame lacks a finite local creation timestamp")
    return {"channel": "local_frame", "clock": "client_monotonic", "t": timestamp}


def newer_capture(frame, previous) -> bool:
    current, prior = capture_marker(frame), capture_marker(previous)
    current_t, prior_t = current.pop("t"), prior.pop("t")
    if current != prior:
        raise ValueError("Camera capture identity changed across the freshness barrier")
    return current_t > prior_t


def _deadline(timeout_s):
    if not math.isfinite(timeout_s) or not 0 < timeout_s <= 30:
        raise ValueError("Expected a positive frame timeout of at most 30 seconds")
    return time.monotonic() + timeout_s


def wait_stream_frame(stream, *, after=None, timeout_s=5):
    """Wait for a new delivery; with ``after``, require a newer real capture.

    Existing viewing APIs may return their last frame on timeout. Reset uses
    this explicit barrier, which never returns that stale fallback.
    """
    deadline = _deadline(timeout_s)
    with stream._cond:
        want = stream._latest_seq + 1
        while True:
            frame = stream._latest
            if stream._latest_seq >= want and frame is not None:
                capture_marker(frame)
                if after is None or newer_capture(frame, after):
                    return frame
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Camera did not produce a fresh capture before the reset deadline")
            stream._cond.wait(timeout=remaining)


def read_frame_after(camera, *, after=None, timeout_s=5):
    """Use a stream's bounded barrier or a direct camera's normal grab API."""
    deadline = _deadline(timeout_s)
    getter = getattr(camera, "get_fresh_frame", None)
    if getter is not None:
        return getter(after=after, timeout_s=timeout_s)
    # Direct local cameras create their frame during get_frame; they retain
    # their own device I/O timeouts. Production rig cameras use the stream
    # barrier above, so a blocked producer cannot exceed the reset deadline.
    while True:
        frame = camera.get_frame()
        capture_marker(frame)
        if after is None or newer_capture(frame, after):
            return frame
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Camera repeated its cached capture until the reset deadline")
        time.sleep(min(.01, remaining))


def frames_after_reset(cameras, *, timeout_s=5, on_floor=None):
    """Drain one post-reset delivery, then require a newer capture per camera.

    The first delivery can be an already in-flight pre-reset render. Its
    producer timestamp is the floor, even when its client receipt is new.
    Establish every camera's floor before waiting for their subsequent frames.
    """
    deadline = _deadline(timeout_s)
    floors = []
    for camera in cameras:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Reset camera freshness deadline expired")
        floor = read_frame_after(camera, timeout_s=remaining)
        floors.append((camera, floor))
        if on_floor is not None:
            on_floor(camera, floor)
    observed = []
    for camera, floor in floors:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Reset camera freshness deadline expired")
        frame = read_frame_after(camera, after=floor, timeout_s=remaining)
        observed.append((camera, floor, frame))
    return observed
