#!/usr/bin/env python3
"""Live camera fault injection for the real Isaac/nvblox reset path.

One producer getter returns the SAME captured Frame (including producer clock).
Empty case: first public reset may move/reset props before its camera barrier
fails; a second reset while pending must issue no actuator calls. --held first
requires a real grasp, then seeds pending via the private geometry barrier under
watcher.paused(); it does NOT claim a first public reset was actuator-free.
Restoring the original getter must allow the public reset to rebuild all three
camera contributions, reach home, reset props, and obtain fresh captures.

This is a fault diagnostic, not native/UI/Spark acceptance. Final cleanup only
closes consumers/cameras/Isaac sockets; it sends no park, release, or reset.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import threading
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
OBJECTS = ("orange", "green_cube", "pink_cube", "lemon", "tomato_can")


def encode(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, default=encode, indent=2, allow_nan=False) + "\n")


def array_hash(value):
    if value is None:
        return None
    return {"dtype": str(value.dtype), "shape": list(value.shape),
            "sha256": hashlib.sha256(value.tobytes()).hexdigest()}


class FrozenProducer:
    """Patch only the producer getter; the normal pump/barrier remain intact."""
    def __init__(self, stream):
        from cascade.perception.freshness import capture_marker
        self.stream, self.producer = stream, stream._camera
        self.frame = stream.get_frame()
        self.marker = capture_marker(self.frame)
        self.frame_t, self.frame_id = self.frame.t, self.frame.frame_id
        self.rgb, self.depth = array_hash(self.frame.rgb), array_hash(self.frame.depth_m)
        self.original = self.producer.get_frame
        self.deliveries = 0
        self.same_marker = True
        self._lock = threading.Lock()

    def _repeat(self):
        from cascade.perception.freshness import capture_marker
        with self._lock:
            self.deliveries += 1
            self.same_marker &= capture_marker(self.frame) == self.marker
        return self.frame

    def __enter__(self):
        self.producer.get_frame = self._repeat
        try:
            deadline = time.monotonic() + 8.
            while time.monotonic() < deadline:
                with self._lock:
                    delivered = self.deliveries >= 2
                if delivered and self.stream.latest() is self.frame:
                    return self
                time.sleep(.01)
            raise TimeoutError("frozen producer did not reach the unmodified camera pump")
        except BaseException:
            self.producer.get_frame = self.original
            raise

    def __exit__(self, *_):
        self.producer.get_frame = self.original

    def report(self):
        from cascade.perception.freshness import capture_marker
        with self._lock:
            return {"marker": self.marker, "rgb": self.rgb, "depth": self.depth,
                    "frame_t": self.frame_t, "frame_id": self.frame_id,
                    "deliveries": self.deliveries, "all_deliveries_same_marker": self.same_marker,
                    "unchanged_frame": (capture_marker(self.frame) == self.marker
                        and self.frame.t == self.frame_t and self.frame.frame_id == self.frame_id
                        and array_hash(self.frame.rgb) == self.rgb
                        and array_hash(self.frame.depth_m) == self.depth)}


class ActuatorTrace:
    """Record attempted API calls AND actual backend commands, without extra I/O."""
    def __init__(self, runtime, path):
        self.runtime, self.path = runtime, Path(path)
        self.events, self.phase = [], "setup"
        self._lock = threading.Lock()
        raw = runtime.arm.raw
        # This is an explicit live Isaac diagnostic. Resolving reset_props may
        # connect LazyArm, but invokes neither reset_props nor a motion.
        getattr(raw, "reset_props")
        backend = getattr(raw, "_arm", None) or raw
        for name in ("move_joints", "set_gripper"):
            self._wrap(runtime.arm, name, "request")
        for name in ("send_joint_target", "set_gripper", "reset_props"):
            self._wrap(backend, name, "actuator")

    def _record(self, event):
        event.update(phase=self.phase, monotonic=time.monotonic(),
                     held=self.runtime.held_object)
        with self._lock, self.path.open("a") as output:
            self.events.append(event)
            output.write(json.dumps(event, default=encode, allow_nan=False) + "\n")

    def _wrap(self, owner, name, layer):
        original = getattr(owner, name)
        def call(*args, **kwargs):
            call_id = time.monotonic_ns()
            self._record({"event": "begin", "layer": layer, "call": name, "id": call_id,
                          "args": args, "kwargs": {k: v for k, v in kwargs.items() if not k.startswith("_")}})
            try:
                result = original(*args, **kwargs)
            except Exception:
                self._record({"event": "exception", "layer": layer, "call": name, "id": call_id,
                              "traceback": traceback.format_exc()})
                raise
            self._record({"event": "end", "layer": layer, "call": name, "id": call_id})
            return result
        setattr(owner, name, call)

    def calls(self, phase):
        return [{k: event[k] for k in ("layer", "call", "id", "monotonic")}
                for event in self.events if event["phase"] == phase and event["event"] == "begin"]


def fresh_state(observer, runtime, *, timeout_s=15.):
    """Use a passive acquisition that STARTED after this requested boundary."""
    began, deadline = time.monotonic(), time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if observer.errors:
            raise RuntimeError("passive observer failed: " + repr(observer.errors))
        with observer._lock:
            row = copy.deepcopy(observer.records[-1]) if observer.records else None
        if row and row["client_started_monotonic"] >= began:
            occupancy = runtime.arm.harness.occupancy
            with occupancy._refresh_lock:
                mapping = {"pending": occupancy.scene_reset_pending,
                           "stale": occupancy.is_stale(), "last_error": occupancy.last_error,
                           "body_error": occupancy._body_error, "has_grid": occupancy._grid is not None,
                           "last_refresh_monotonic": occupancy._last_refresh,
                           "camera_history": {name: [f["t"] for f in frames]
                                              for name, frames in occupancy._depth_history.items()}}
            physics = row["physics"]
            keys = ("engine", "robot_id", "sim_time", "physics_step", "q", "dq", "gripper", "props",
                    "spawn_positions_m", "contacts", "gpu_attestation")
            return {"sequence": row["sequence"], "requested_after_monotonic": began,
                    "client_started_monotonic": row["client_started_monotonic"],
                    "client_finished_monotonic": row["client_finished_monotonic"],
                    "physics": {key: physics[key] for key in keys}, "map": mapping,
                    "held": runtime.held_object,
                    "held_provisional": copy.deepcopy(getattr(runtime, "_held_provisional", None))}
        time.sleep(.01)
    raise TimeoutError("no independent physics acquisition after the requested phase boundary")


def phase(observer, runtime, recorder, name, operation):
    before = fresh_state(observer, runtime)
    recorder.phase = name
    observer.mark(name + "_begin")
    try:
        result = operation()
    except Exception as exc:
        result = {"ok": False, "exception": type(exc).__name__, "error": str(exc),
                  "traceback": traceback.format_exc()}
    observer.mark(name + "_end")
    after = fresh_state(observer, runtime)
    return {"result": result, "before": before, "after": after, "calls": recorder.calls(name)}


def blocked_measurements(before, after):
    import numpy as np
    b, a = before["physics"], after["physics"]
    return {"q_max_drift_rad": float(np.max(np.abs(np.asarray(b["q"]) - a["q"]))),
            "jaw_max_drift_m": float(np.max(np.abs(np.asarray(b["gripper"]["q"]) - a["gripper"]["q"]))),
            "prop_drift_m": {name: float(np.linalg.norm(np.asarray(state["position_m"])
                               - a["props"][name]["position_m"])) for name, state in b["props"].items()}}


def fresh_camera_checks(result, expected_cameras):
    rows = result.get("observation_freshness") or []
    names, identities_valid, advanced = [], True, True
    for row in rows:
        floor, observed = dict(row.get("floor") or {}), dict(row.get("observed") or {})
        t0, t1 = floor.pop("t", None), observed.pop("t", None)
        names.append(floor.get("camera"))
        identities_valid &= bool(floor and floor == observed and floor.get("channel") == "producer_capture")
        advanced &= bool(type(t0) in (int, float) and type(t1) in (int, float) and t1 > t0)
    return {"all_camera_barriers_reported": len(names) == len(expected_cameras) and set(names) == set(expected_cameras),
            "same_camera_robot_clock_binding": identities_valid, "all_producer_clocks_advanced": advanced}


def retained_anchor_checks(result, expected_cameras):
    """Report the unchanged-scene rebuild before home/real prop reset discard it."""
    report = result.get("retained_anchor_refresh") or {}
    before, after = report.get("anchors_before") or {}, report.get("anchors_after") or {}
    integrated = report.get("integrated") or []
    epoch, floor = report.get("producer_epoch"), report.get("shared_floor")
    names = [row.get("camera") for row in integrated]
    return {
        "retained_measured_anchors": bool(before and before == after and set(before) == expected_cameras),
        "retained_epoch_and_all_fresh_commits": bool(isinstance(epoch, str) and epoch
            and type(floor) in (int, float) and math.isfinite(floor)
            and len(names) == len(expected_cameras) and set(names) == expected_cameras
            and all(row.get("producer_epoch") == epoch and row["t"] > floor
                    and {k: row.get(k) for k in ("source", "robot_id", "clock")}
                        == {k: before.get(row["camera"], {}).get(k) for k in ("source", "robot_id", "clock")}
                    for row in integrated)),
        "retained_background_replayed": report.get("replayed_frames", 0) >= len(expected_cameras),
    }


def reset_window(records, *, after, cfg, expected_props, engine):
    """Reset-only physical checks at the existing GPU auditor's thresholds."""
    import numpy as np
    samples = [row["physics"] for row in records if row["client_started_monotonic"] >= after]
    if not samples:
        return {"pass": False, "error": "missing post-recovery samples"}
    clocks = np.asarray([s["sim_time"] for s in samples])
    start = max(0, int(np.searchsorted(clocks, clocks[-1] - .5, side="right")) - 1)
    tail = samples[start:]
    checks = {"reset_settle_window": (np.isfinite(clocks).all() and (np.diff(clocks) >= 0).all()
        and len(tail) >= 4 and len({s["physics_step"] for s in tail}) >= 4
        and tail[-1]["sim_time"] - tail[0]["sim_time"] >= .5),
        "same_engine_robot": all(s["engine"] == engine and s["robot_id"] == cfg.arm.bridge_robot_id for s in tail),
        "expected_props": all(set(s["props"]) == set(expected_props) and set(s["spawn_positions_m"]) == set(expected_props) for s in tail)}
    home = np.asarray(cfg.arm.home_q)
    signs = np.asarray(cfg.arm.joint_signs)
    q = np.asarray([s["q"][:len(home)] for s in tail]) * signs
    home_error = float(np.abs(q - home).max())
    checks["actual_home_joints"] = np.isfinite(q).all() and home_error <= float(cfg.arm.get("settle_tol", .02))
    jaws = np.asarray([s["gripper"]["open_fractions"] for s in tail])
    checks["actual_jaws_open"] = np.isfinite(jaws).all() and jaws.min() >= .9
    props = {}
    for name in expected_props:
        positions = np.asarray([s["props"][name]["position_m"] for s in tail])
        spawn = np.asarray([s["spawn_positions_m"][name] for s in tail])
        speed = np.asarray([s["props"][name]["linear_velocity_m_s"] for s in tail])
        spin = np.asarray([s["props"][name]["angular_velocity_rad_s"] for s in tail])
        metrics = {"max_spawn_error_m": float(np.linalg.norm(positions - spawn, axis=1).max()),
                   "max_speed_m_s": float(np.linalg.norm(speed, axis=1).max()),
                   "max_spin_rad_s": float(np.linalg.norm(spin, axis=1).max()),
                   "max_spread_m": float(np.linalg.norm(positions - positions[-1], axis=1).max())}
        valid = np.isfinite(list(metrics.values())).all()
        checks[name + "_reset_settled"] = (valid and metrics["max_spawn_error_m"] <= .02
            and metrics["max_speed_m_s"] <= .02 and metrics["max_spin_rad_s"] <= .2 and metrics["max_spread_m"] <= .005)
        props[name] = metrics
    return {"pass": bool(all(checks.values())), "checks": {k: bool(v) for k, v in checks.items()},
            "home_max_error_rad": home_error, "props": props,
            "scope": "post-recovery reset only; no lift/transport/native/UI acceptance"}


def held_contact_steps(records, object_name, robot_id, channel):
    import numpy as np
    steps = set()
    jaw_root = robot_id + "/link1/link2/link3/link4/link5/link6/gripper_end"
    for row in records:
        sample = row["physics"]
        contact = sample["contacts"]
        forces, counts = np.asarray(contact["jaw_forces_n"]), np.asarray(contact["jaw_contact_counts"])
        if (sample["robot_id"] == robot_id and contact.get("channel") == channel
                and contact.get("physics_step") == sample["physics_step"]
                and contact.get("sensor_paths") == ["/World_Props/" + object_name]
                and contact.get("filter_paths") == [[jaw_root + "/gripper_left", jaw_root + "/gripper_right"]]
                and forces.shape == (2, 3) and counts.shape == (2,)
                and np.isfinite(forces).all() and (np.linalg.norm(forces, axis=1) > .01).all()
                and (counts > 0).all() and sample["props"][object_name]["position_m"][2]
                    - sample["spawn_positions_m"][object_name][2] >= .015):
            steps.add(sample["physics_step"])
    return sorted(steps)


def close_without_motion(runtime):
    """Isaac diagnostic teardown; unlike shutdown_runtime, never parks."""
    errors = []
    steps = [("watcher", lambda: runtime.watcher.stop()),
             ("stream_server", lambda: runtime.stream_server.stop() if getattr(runtime, "stream_server", None) else None),
             ("viewer", lambda: runtime.viewer.stop() if getattr(runtime, "viewer", None) else None),
             ("cameras", lambda: runtime.rig.close()),
             ("arm_socket", lambda: runtime.arm.disconnect())]
    for name, operation in steps:
        try:
            operation()
        except Exception:
            errors.append({"component": name, "traceback": traceback.format_exc()})
    return errors


def run(args, out):
    from kitchen_acceptance import (check_frozen_sources, configure_query_region,
                                    frozen_sources, load_proof, retarget_ports)
    from nvblox_contact_recovery import rebuilt_camera_map
    from nvblox_postclose_recovery import require_simulation_clock
    from cascade.apps.demo import build_runtime
    from cascade.config import load_demo_config
    cfg = load_demo_config(cameras=["isaac", "isaac_side", "isaac_proof"], arm="isaac_kitchen_gpu", llm="mock")
    if not retarget_ports(cfg._data, args.port):
        raise RuntimeError("resolved configuration contains no bridge endpoint")
    cfg._data["occupancy"].update(port=args.occupancy_port, track_payload=True,
        allowed_contact_paths=["/World_Props/" + name for name in OBJECTS])
    for camera in cfg._data["cameras"]:
        camera["map_depth"] = True
    cfg._data["grasp"]["pregrasp_offset_m"] = args.pregrasp_offset
    receipt = {"pass": False, "scope": __doc__, "held_requested": args.held, "errors": [], "phases": {},
        "barrier_trigger": "private _reset_camera_frames(require_geometry=True) under paused watcher" if args.held else "public reset_scene API",
        "port": args.port, "occupancy_port": args.occupancy_port, "engine": args.engine}
    receipt["experimental_map_query_region"] = configure_query_region(cfg, args)
    receipt["effective_configuration"] = {"home_q": cfg.arm.home_q, "joint_signs": cfg.arm.joint_signs,
        "robot_id": cfg.arm.bridge_robot_id, "grasp": cfg.grasp.as_dict(), "safety": cfg.safety.as_dict()}
    receipt["source_sha256"] = frozen_sources(args.scene_config)
    runtime = observer = recorder = None
    try:
        runtime, _ = build_runtime(cfg, out / "runtime", lazy_arm=True)
        recorder = ActuatorTrace(runtime, out / "commands.jsonl")
        receipt["simulation_clock"] = require_simulation_clock(runtime)
        occupancy = runtime.arm.harness.occupancy
        status = occupancy.probe(timeout_ms=2000) if occupancy is not None else None
        receipt["occupancy_probe"] = status
        if not status or status.get("backend") != "nvblox" or not str(status.get("device", "")).startswith("cuda:"):
            raise RuntimeError("diagnostic requires the real CUDA nvblox backend")
        proof = load_proof()
        expected = proof.shared.load_expected_scene_geometry(args.scene_config)
        observer = proof.shared.GpuProofObserver(out / "witness", port=args.port, interval_s=.15, budget_s=600,
            object_name=args.held or "orange", destination_name="open box", expected_scene_geometry=expected)
        with observer:
            with observer._lock:
                identity = [(r["physics"]["engine"], r["physics"]["robot_id"]) for r in observer.records]
            if not identity or set(identity) != {(args.engine, cfg.arm.bridge_robot_id)}:
                raise RuntimeError("passive observer does not match the requested engine/robot")
            check_frozen_sources(receipt["source_sha256"])
            if args.held:
                setup = phase(observer, runtime, recorder, "setup_grasp",
                              lambda: runtime.execute("grasp_object", {"label": args.held.replace("_", " ")}))
            else:
                setup = phase(observer, runtime, recorder, "setup_reset", lambda: runtime.execute("reset_scene", {}))
            receipt["phases"]["setup"] = setup
            if setup["result"].get("ok") is not True or bool(runtime.held_object) != bool(args.held):
                raise RuntimeError("could not establish the requested initial held/empty state")
            observer.settle(simulation_seconds=.65, wall_timeout=40)
            receipt["setup_state"] = fresh_state(observer, runtime)
            if args.held:
                with observer._lock:
                    rows = [r for r in observer.records if r["client_started_monotonic"] >= observer.marks["setup_grasp_end"]["monotonic"]]
                receipt["held_contact_steps"] = held_contact_steps(rows, args.held, cfg.arm.bridge_robot_id,
                                                                   proof.shared.CONTACT_CHANNELS[args.engine])
                current_contact = held_contact_steps([{"physics": receipt["setup_state"]["physics"]}],
                    args.held, cfg.arm.bridge_robot_id, proof.shared.CONTACT_CHANNELS[args.engine])
                if len(receipt["held_contact_steps"]) < 2 or not current_contact:
                    raise RuntimeError("held setup lacks two observed bilateral-contact lifted steps")
            cameras = runtime.watcher._cams
            streams = [c.stream for c in cameras if c.stream.name == args.camera
                       or getattr(c.stream._camera, "_camera", None) == args.camera]
            if len(streams) != 1:
                raise RuntimeError("camera name must identify exactly one consumed stream")
            with runtime.watcher.paused():
                freeze = FrozenProducer(streams[0])
                with freeze:
                    first_name = "private_pending_barrier" if args.held else "first_frozen_reset"
                    def seed_pending():
                        runtime._reset_camera_frames(require_geometry=True, scene_changed=False)
                        return {"ok": True, "unexpected_barrier_completion": True}
                    operation = seed_pending if args.held else (lambda: runtime.execute("reset_scene", {}))
                    check_frozen_sources(receipt["source_sha256"])
                    first = phase(observer, runtime, recorder, first_name, operation)
                    receipt["phases"]["first_barrier"] = first
                    if not occupancy.scene_reset_pending or not getattr(runtime, "_reset_observation_pending", False):
                        raise RuntimeError("first frozen barrier did not establish pending recovery; no second reset attempted")
                    check_frozen_sources(receipt["source_sha256"])
                    receipt["phases"]["blocked_reset"] = phase(observer, runtime, recorder, "blocked_reset",
                                                                lambda: runtime.execute("reset_scene", {}))
                receipt["freeze"] = freeze.report()
                check_frozen_sources(receipt["source_sha256"])
                recovery = phase(observer, runtime, recorder, "recovered_reset", lambda: runtime.execute("reset_scene", {}))
                receipt["phases"]["recovered_reset"] = recovery
                observer.settle(simulation_seconds=.65, wall_timeout=40)
                receipt["final_state"] = fresh_state(observer, runtime)
                receipt["camera_map_commits"] = rebuilt_camera_map(occupancy, recovery["result"])
            with observer._lock:
                records = copy.deepcopy(observer.records)
            receipt["reset_physics"] = reset_window(records, after=observer.marks["recovered_reset_end"]["monotonic"],
                cfg=cfg, expected_props=expected["prop_dimensions_m"], engine=args.engine)
            blocked = receipt["phases"]["blocked_reset"]
            measurements = blocked_measurements(blocked["before"], blocked["after"])
            receipt["blocked_measurements"] = measurements
            final_map = receipt["final_state"]["map"]
            expected_cameras = {c.stream._camera._camera for c in cameras if c.maps_depth}
            fresh_checks = fresh_camera_checks(recovery["result"], expected_cameras)
            receipt["checks"] = {
                "first_barrier_failed": first["result"].get("ok") is not True,
                "first_barrier_left_pending_map": first["after"]["map"]["pending"],
                "blocked_public_reset_refused": blocked["result"].get("ok") is False and blocked["result"].get("stage") == "reset_recovery",
                "blocked_issued_no_actuator_or_motion_calls": not blocked["calls"],
                "blocked_held_state_preserved": (blocked["before"]["held"] == blocked["after"]["held"]
                    and blocked["before"]["held_provisional"] == blocked["after"]["held_provisional"]),
                "blocked_physics_within_diagnostic_1mrad_2mm": measurements["q_max_drift_rad"] <= .001
                    and measurements["jaw_max_drift_m"] <= .002 and max(measurements["prop_drift_m"].values()) <= .002,
                "frozen_same_frame_and_producer_marker": freeze.deliveries >= 2 and receipt["freeze"]["all_deliveries_same_marker"] and receipt["freeze"]["unchanged_frame"],
                "recovered_public_reset_completed": recovery["result"].get("ok") is True and recovery["result"].get("geometry_recovered") is True,
                "all_props_reset_reported": set(recovery["result"].get("props_reset") or []) == set(expected["prop_dimensions_m"]),
                "all_camera_depths_rebuilt": set(final_map["camera_history"]) == expected_cameras,
                "all_reset_camera_commits_newer_than_floors": receipt["camera_map_commits"]["pass"],
                "fresh_esdf_without_pending_fault": final_map["has_grid"] and not final_map["pending"] and not final_map["stale"]
                    and final_map["last_error"] is None and final_map["body_error"] is None,
                "post_recovery_physics": receipt["reset_physics"]["pass"], **fresh_checks}
            if args.held:
                receipt["checks"].update(retained_anchor_checks(recovery["result"], expected_cameras))
    except Exception:
        receipt["errors"].append(traceback.format_exc())
    finally:
        if runtime is not None:
            if recorder is not None:
                recorder.phase = "cleanup_without_motion"
            receipt["cleanup_errors"] = close_without_motion(runtime)
            receipt["cleanup_calls"] = recorder.calls("cleanup_without_motion") if recorder else []
        if observer is not None:
            receipt["observer_complete"] = observer.complete
            receipt["observer_errors"] = list(observer.errors)
        receipt.setdefault("checks", {}).update(
            source_unchanged=all((ROOT / p).is_file() and hashlib.sha256((ROOT / p).read_bytes()).hexdigest() == h
                                 for p, h in receipt["source_sha256"].items()),
            observer_complete=receipt.get("observer_complete", False) and not receipt.get("observer_errors"),
            no_diagnostic_errors=not receipt["errors"] and not receipt.get("cleanup_errors"),
            cleanup_sent_no_commands=not receipt.get("cleanup_calls"))
        receipt["pass"] = bool(receipt["checks"] and all(receipt["checks"].values()))
        receipt["artifacts_sha256"] = {str(p.relative_to(out)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(out.rglob("*")) if p.is_file() and p.name != "receipt.json"}
        write_json(out / "receipt.json", receipt)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8691)
    parser.add_argument("--occupancy-port", type=int, default=25559)
    parser.add_argument("--engine", choices=("physx", "newton"), default="physx")
    parser.add_argument("--camera", default="side")
    parser.add_argument("--held", choices=OBJECTS)
    parser.add_argument("--pregrasp-offset", type=float, default=.10,
                        help="Explicit diagnostic approach/lift offset in metres")
    parser.add_argument("--query-region-min", type=float, nargs=3, metavar=("X", "Y", "Z"))
    parser.add_argument("--query-region-max", type=float, nargs=3, metavar=("X", "Y", "Z"))
    parser.set_defaults(occupancy="nvblox")
    parser.add_argument("--scene-config", type=Path, default=ROOT / "demo/scene/kitchen_config.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if not math.isfinite(args.pregrasp_offset) or args.pregrasp_offset <= 0:
        parser.error("pregrasp-offset must be finite and positive")
    if (args.query_region_min is None) != (args.query_region_max is None):
        parser.error("query-region-min and query-region-max must be supplied together")
    if args.query_region_min is not None and (
            any(not math.isfinite(v) for v in args.query_region_min + args.query_region_max)
            or any(lo >= hi for lo, hi in zip(args.query_region_min, args.query_region_max))):
        parser.error("query region requires finite increasing bounds")
    out, args.scene_config = args.output.resolve(), args.scene_config.resolve()
    if not out.is_relative_to(ROOT) or out.exists():
        parser.error("output must be a NEW directory inside this checkout")
    if not args.scene_config.is_relative_to(ROOT) or not args.scene_config.is_file():
        parser.error("scene-config must be an existing file inside this checkout")
    if not all(1 <= p <= 65535 for p in (args.port, args.occupancy_port)):
        parser.error("ports must be 1..65535")
    with open(f"/tmp/cascade-acceptance-{args.port}.lock", "a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("another diagnostic owns this bridge")
        out.mkdir(parents=True)
        sys.path.insert(0, str(ROOT / "src"))
        os.environ.update(CASCADE_BRIDGE_PORT=str(args.port), CASCADE_OCCUPANCY="1",
            CASCADE_INSTALL_PROFILE="spark", CASCADE_BELIEFS="0", CASCADE_ISAAC_PIXEL_MASK="1",
            CASCADE_PROOF_CAMERA="1", CASCADE_REQUIRE_CUDA="1", YOLO_OFFLINE="True", ULTRALYTICS_OFFLINE="True",
            CASCADE_GRASP_MEMORY_PATH=str(out / "grasp-memory.json"))
        os.chdir(ROOT / "models")
        receipt = run(args, out)
    print(out, "PASS" if receipt["pass"] else "FAIL", flush=True)
    return 0 if receipt["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
