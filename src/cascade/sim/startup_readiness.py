"""Read-only Isaac verification readiness before a native manipulation proof.

Warm the existing bridge-side RigidPrim views, never a LazyArm. One failed or
late probe aborts startup; it is never resubmitted. Captured poses are discarded
as action authority. Every later verifier still performs its normal fresh read.
"""
from __future__ import annotations

import copy
import json
import math
import time

import numpy as np

from ..control.simulation_motion import PhysicsClock
from ..types import SafetyViolation
from .bridge_client import BridgeClient
from .truth import _PROBE, _PROP_ROOTS, _normalize

# The existing BridgeClient connection/request default, only for this startup
# phase. Action RPC deadlines and the 2 s camera freshness gate do not change.
STARTUP_TIMEOUT_S = 10.0
CAMERA_MAX_AGE_S = 2.0
_DONE = "CASCADE_STARTUP_TRUTH_DONE "
_CLOCK = "CASCADE_STARTUP_CLOCK "
_CLOCK_CODE = ('import json as _csr_json, time as _csr_time\n'
               'print(' + repr(_CLOCK) + ' + _csr_json.dumps({"clock": '
               '_motion_clock_snapshot(), "server_monotonic": _csr_time.monotonic()}))\n')


def _require(condition, message):
    if not condition:
        raise RuntimeError("Isaac verification readiness: " + message)


def _number(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def _vector(value, size):
    a = np.asarray(value)
    return a.shape == (size,) and a.dtype.kind in "fiu" and np.isfinite(a).all()


def _record(reply, marker):
    _require(isinstance(reply, dict) and reply.get("ok") is True, "read-only exec failed")
    lines = [line[len(marker):] for line in str(reply.get("stdout", "")).splitlines()
             if line.startswith(marker)]
    _require(len(lines) == 1, "missing or duplicate " + marker.strip())
    value = json.loads(lines[0])
    _require(isinstance(value, dict), "invalid " + marker.strip())
    return value


def _bound_clock(value, endpoint, robot):
    _require(isinstance(value, dict), "physical clock missing")
    value = copy.deepcopy(value)
    # Endpoint is bound locally to this dedicated transport, never trusted
    # from a server's assertion about its own network address.
    _require("source" not in value or value["source"] == endpoint, "foreign clock source")
    value["source"] = endpoint
    PhysicsClock(endpoint, robot).observe(value)
    return value


def _physical_sample(sample, done, cfg, endpoint):
    n = int(cfg.get("n_joints", 6))
    robot = cfg.get("bridge_robot_id")
    clock = _bound_clock(sample.get("physics_clock"), endpoint, robot)
    end_clock = _bound_clock(done.get("clock"), endpoint, robot)
    _require(clock == end_clock, "probe did not finish in its physical step")
    _require(type(sample.get("version")) is int and sample["version"] == 1
             and sample.get("joint_convention") == "asset"
             and sample.get("joint_names") == [f"joint{i}" for i in range(1, n + 1)]
             and _vector(sample.get("q_asset"), n), "invalid atomic control joints")
    indices = sample.get("joint_indices")
    _require(isinstance(indices, list) and len(indices) == n
             and all(type(i) is int and i >= 0 for i in indices)
             and len(set(indices)) == n, "invalid atomic joint map")
    _require(sample.get("pose_frame") == "world" and type(sample.get("meters_per_unit")) in (int, float)
             and sample["meters_per_unit"] == 1.
             and _vector(sample.get("base_position_world"), 3)
             and _vector(sample.get("base_orientation_wxyz"), 4)
             and abs(np.linalg.norm(sample["base_orientation_wxyz"]) - 1.) <= 1e-6,
             "invalid physical base pose or units")
    inventory = sample.get("physical_inventory")
    poses, paths = sample.get("physical_poses"), sample.get("physical_paths")
    _require(isinstance(inventory, list) and inventory and isinstance(poses, dict)
             and isinstance(paths, dict), "physical prop inventory unavailable")
    names, body_paths = [], []
    for item in inventory:
        _require(isinstance(item, dict), "malformed prop inventory")
        name, path = item.get("name"), item.get("path")
        _require(isinstance(name, str) and name and isinstance(path, str)
                 and path.startswith("/") and path.rsplit("/", 1)[-1] == name
                 and paths.get(name) == path and _vector(poses.get(name), 3),
                 "prop has no uniquely bound live physics pose")
        names.append(name)
        body_paths.append(path)
    _require(len(set(map(_normalize, names))) == len(names)
             and len(set(body_paths)) == len(names)
             and set(poses) == set(paths) == set(names), "ambiguous or incomplete physical inventory")
    stamp, finished = sample.get("server_monotonic"), done.get("server_monotonic")
    _require(_number(stamp) and _number(finished) and finished >= stamp,
             "invalid probe completion timestamp")
    return clock, finished, body_paths


def _packet(frame, endpoint, robot, epoch, n):
    capture = getattr(frame, "capture", None)
    _require(isinstance(capture, dict) and capture.get("backend") == "isaac"
             and capture.get("source") == endpoint, "camera source mismatch")
    state, ref = capture.get("proprioception"), capture.get("render_reference")
    stamp = capture.get("t")
    _require(_number(stamp) and isinstance(state, dict) and isinstance(ref, dict),
             "camera capture binding missing")
    _require(type(state.get("version")) is int and state["version"] == 1
             and state.get("backend") == "isaac" and state.get("robot_id") == robot
             and state.get("joint_convention") == "asset"
             and state.get("time_source") == "physics_loop_monotonic"
             and state.get("t") == stamp and state.get("producer_epoch") == epoch
             and _vector(state.get("q"), n), "camera physical state mismatch")
    jaws = state.get("gripper_joints", {})
    _require(isinstance(jaws, dict) and type(jaws.get("version")) is int and jaws["version"] == 1 and jaws.get("names") == ["joint_left", "joint_right"]
             and all(_vector(jaws.get(k), 2) for k in ("position_m", "lower_m", "upper_m")),
             "camera finger measurements missing")
    step = ref.get("history_physics_step")
    sim = ref.get("history_simulation_time")
    from .render_binding import valid_render_binding
    native_token = (ref.get("source") == "rpFabricTime"
                    and type(ref.get("numerator")) is int and ref["numerator"] >= 0
                    and type(ref.get("denominator")) is int and ref["denominator"] > 0
                    and _number(ref.get("render_simulation_time")))
    _require(type(step) is int and step >= 0 and _number(sim)
             and type(ref.get("version")) is int and ref["version"] == 1
             and valid_render_binding(capture) and (native_token or ref.get("source") == "ovrtx_snapshot")
             and ref.get("producer_epoch") == epoch
             and ref.get("snapshot_started_monotonic") == stamp
             and _number(ref.get("snapshot_finished_monotonic"))
             and ref["snapshot_finished_monotonic"] >= stamp
             and isinstance(ref.get("product"), str) and ref["product"], "invalid render/state association")
    rgb, depth = getattr(frame, "rgb", None), getattr(frame, "depth_m", None)
    _require(isinstance(rgb, np.ndarray) and rgb.dtype == np.uint8
             and rgb.ndim == 3 and rgb.shape[2] == 3 and rgb.size
             and isinstance(depth, np.ndarray) and depth.shape == rgb.shape[:2]
             and np.isfinite(depth).all() and np.any(depth > 0)
             and _vector(np.asarray(frame.K).reshape(-1), 9), "RGBD packet unavailable")
    return capture, state, ref


def _latest(stream, timeout_s):
    # CameraStream.latest() normally waits on this condition without a bound.
    # Keep even lock contention inside the single startup deadline.
    _require(stream._cond.acquire(timeout=timeout_s), "camera stream lock deadline expired")
    try:
        return stream._latest
    finally:
        stream._cond.release()


def prepare_isaac_verification(runtime):
    """Warm physics readers and admit new packets on every configured camera.

    Called only by the launcher's existing lazy runtime check. Non-Isaac
    profiles are untouched. No actuator object is inspected or materialized.
    A dedicated socket has one total 10 s deadline, including connect, the
    single native probe and subsequent camera/clock checks. A timeout fails;
    an uncertain probe is never sent again or treated as completed.
    """
    cfg = runtime.cfg.arm
    if cfg.get("type") != "isaac":
        return None
    endpoint = (str(cfg.get("bridge_host", "127.0.0.1")), int(cfg.get("bridge_port", 8611)))
    robot = cfg.get("bridge_robot_id")
    streams = list(runtime.rig)
    _require(streams, "no configured cameras")
    for stream in streams:
        camera = stream._camera
        _require(type(camera).__name__ == "IsaacCamera"
                 and getattr(camera.__dict__.get("_client"), "_addr", None) == endpoint,
                 "camera does not belong to the configured Isaac bridge")
    started = time.monotonic()
    deadline = started + STARTUP_TIMEOUT_S

    def remaining():
        left = deadline - time.monotonic()
        _require(left > 0, "startup deadline expired")
        return left

    client = BridgeClient(*endpoint, timeout_s=remaining())
    try:
        client.connect()
        code = (_PROBE % {"roots": repr(_PROP_ROOTS)}) + _CLOCK_CODE.replace(_CLOCK, _DONE)
        reply = client.request({"op": "exec", "code": code}, timeout_s=remaining())
        sample, done = _record(reply, "CASCADE_TRUTH_OBSERVATION "), _record(reply, _DONE)
        floor, finished, bodies = _physical_sample(sample, done, cfg, endpoint)
        # Discard all poses; readiness cannot be used as future pick evidence.
        validator = PhysicsClock(endpoint, robot)
        validator.observe(floor)
        last_server = finished
        while True:
            remaining()
            frames = [_latest(stream, remaining()) for stream in streams]
            clock_requested = time.monotonic()
            now = _record(client.request({"op": "exec", "code": _CLOCK_CODE},
                                         timeout_s=remaining()), _CLOCK)
            current = _bound_clock(now.get("clock"), endpoint, robot)
            validator.observe(current)
            server = now.get("server_monotonic")
            _require(_number(server) and server >= last_server, "server clock regressed")
            last_server = server
            packets = []
            for stream, frame in zip(streams, frames):
                if frame is None:
                    break
                capture, state, ref = _packet(frame, endpoint, robot, floor["epoch"], int(cfg.get("n_joints", 6)))
                stamp = capture["t"]
                _require(capture.get("camera") == stream._camera._camera, "camera name mismatch")
                frame_clock = dict(floor, physics_step=ref["history_physics_step"],
                                   sim_time=ref["history_simulation_time"])
                frame_validator = PhysicsClock(endpoint, robot)
                frame_validator.observe(frame_clock)
                frame_validator.observe(current)
                _require(stamp <= server, "camera is ahead of physical clock")
                if stamp <= finished or frame_clock["physics_step"] <= floor["physics_step"] or server - stamp > CAMERA_MAX_AGE_S:
                    break  # Whole old packets may repeat; they never count as new.
                packets.append({"camera": capture["camera"], "capture_t": stamp,
                                "age_server_s": server - stamp, "render_reference": copy.deepcopy(ref)})
            remaining()
            if len(packets) == len(streams):
                _require(len({p["camera"] for p in packets}) == len(streams)
                         and len({p["render_reference"]["product"] for p in packets}) == len(streams),
                         "duplicate configured camera or render product")
                # Add local elapsed time, not local minus remote timestamps.
                # This conservatively covers RPC delivery and validation work.
                elapsed_after_request = time.monotonic() - clock_requested
                if any(p["age_server_s"] + elapsed_after_request > CAMERA_MAX_AGE_S for p in packets):
                    time.sleep(min(.05, remaining()))
                    continue
                for packet in packets:
                    packet["age_upper_bound_s"] = packet["age_server_s"] + elapsed_after_request
                return {"pass": True, "source": endpoint, "producer_epoch": floor["epoch"],
                        "physical_paths": bodies, "probe_finished_server_monotonic": finished,
                        "camera_packets": packets, "elapsed_s": time.monotonic() - started,
                        "scope": "startup availability only; later truth reads remain fresh"}
            time.sleep(min(.05, remaining()))
    except (SafetyViolation, ValueError, KeyError, TypeError) as exc:
        raise RuntimeError("Isaac verification readiness failed: " + str(exc)) from exc
    finally:
        client.close()
