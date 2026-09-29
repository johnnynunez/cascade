"""Passive event acceptance for the two shipped Spark kitchen orders.

The existing GPU auditor supplies actual contacts, lift, release, destination
geometry and reset checks. This adapter also checks all three event cameras.
It never commands motion or changes the stage.
"""
from __future__ import annotations

import base64
import hashlib
import importlib.util
import math
from pathlib import Path


_spec = importlib.util.spec_from_file_location(
    "cascade_spark_shared_audit", Path(__file__).with_name("gpu_proof_audit.py"))
shared = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(shared)

CAMERAS = ("cam0", "side", "proof")
CASES = (("green_cube", "green square"), ("orange", "open box"))


def audit_cameras(records, marks):
    """Require current, robot-bound images advancing through pick and reset."""
    result = {"pass": False, "cameras": {}}
    try:
        for camera in CAMERAS:
            captures = []
            hashes = set()
            for row in records:
                sample = row["physics"]
                frame = sample["cameras"][camera]
                capture = frame["capture_monotonic"]
                age = sample["server_monotonic"] - capture
                if (frame.get("available") is not True
                        or frame.get("robot_id") != sample["robot_id"]
                        or frame.get("producer_time_source") != "physics_loop_monotonic"
                        or type(capture) not in (int, float) or not math.isfinite(capture)
                        or not -.01 <= age <= 2):
                    raise ValueError(f"{camera}: missing, stale or unbound camera frame")
                captures.append(capture)
                hashes.add(frame["jpeg_sha256"])
            advancing = bool(captures and all(b >= a for a, b in zip(captures, captures[1:])))
            phases = {}
            for phase in ("pick", "reset"):
                begin, end = (marks[phase + suffix]["monotonic"] for suffix in ("_begin", "_end"))
                values = [capture for row, capture in zip(records, captures)
                          if row["client_started_monotonic"] >= begin
                          and row["client_finished_monotonic"] <= end]
                phases[phase] = bool(len(values) >= 2 and values[-1] > values[0])
            result["cameras"][camera] = {
                "pass": advancing and all(phases.values()) and len(hashes) > 1,
                "phase_advancement": phases, "unique_jpeg_hashes": len(hashes)}
        result["pass"] = all(value["pass"] for value in result["cameras"].values())
    except (KeyError, TypeError, ValueError) as exc:
        result["error"] = str(exc)
    return result


class SparkKitchenWitness(shared.GpuProofObserver):
    def __init__(self, out, *, scene_config, object_name, destination_name, port=8611):
        if (object_name, destination_name) not in CASES:
            raise ValueError("Expected green cube to green square or orange to open box")
        expected = shared.load_expected_scene_geometry(scene_config)
        super().__init__(out, port=port, interval_s=.15, budget_s=900,
                         object_name=object_name, destination_name=destination_name,
                         expected_scene_geometry=expected)

    def _snapshot_code(self):
        code = super()._snapshot_code()
        boundary = "_gpu_observed = _obs_collect()\n"
        if code.count(boundary) != 1:
            raise RuntimeError("Passive observer snapshot boundary changed")
        extra = '''_gpu_observed["cameras"] = {}
for _spark_camera in ("cam0", "side", "proof"):
    _spark_frame = _frames.get(_spark_camera)
    if _spark_frame is None:
        _gpu_observed["cameras"][_spark_camera] = {"available": False}
        continue
    _spark_capture = _spark_frame.get("proprioception") or {}
    _gpu_observed["cameras"][_spark_camera] = {
        "available": True, "capture_monotonic": _spark_frame.get("t"),
        "robot_id": _spark_capture.get("robot_id"),
        "producer_time_source": _spark_capture.get("time_source"),
        "jpeg_sha256": _obs_hash.sha256(_obs_base64.b64decode(_spark_frame["rgb_jpeg_b64"])).hexdigest()}
'''
        code = code.replace(boundary, boundary + extra)
        # Convex geometry already uses the shipped bounded lossless codec.
        # The smaller cube receipt needs the same compressed outer envelope.
        if 'import zlib as _obs_zlib' not in code:
            boundary = 'if len((_obs_payload + '
            if code.count(boundary) != 1:
                raise RuntimeError("Passive observer transport boundary changed")
            code = code.split(boundary)[0] + '''import zlib as _obs_zlib
if len(_obs_payload.encode("utf-8")) > 65536:
    raise RuntimeError("Observer decoded snapshot exceeds 65536 bytes")
_spark_encoded = _obs_base64.b64encode(_obs_zlib.compress(_obs_payload.encode("utf-8"), 6)).decode("ascii")
_spark_line = "KITCHEN_OBSERVER_ZLIB " + _spark_encoded
if len(_spark_line.encode("utf-8")) >= 7400:
    raise RuntimeError("Observer compressed snapshot exceeds safe bridge stdout size")
print(_spark_line)
'''
        return code

    def capture_frames(self, phase):
        """Save existing bridge frames and their identities; no video recording."""
        from cascade.sim.bridge_client import BridgeClient

        if phase not in ("placed", "reset"):
            raise ValueError("Expected a placed or reset screenshot phase")
        client = BridgeClient(host=self.host, port=self.port, timeout_s=8)
        images = {}
        try:
            client.connect()
            for camera in CAMERAS:
                frame = client.request({"op": "frame", "camera": camera})
                data = base64.b64decode(frame["rgb_jpeg_b64"], validate=True)
                if not data.startswith(b"\xff\xd8") or not data.endswith(b"\xff\xd9"):
                    raise ValueError(f"{camera}: invalid JPEG frame")
                path = self.out / f"{phase}-{camera}.jpg"
                path.write_bytes(data)
                images[camera] = {"path": path.name, "sha256": hashlib.sha256(data).hexdigest(),
                                  "capture_monotonic": frame.get("t"),
                                  "proprioception": frame.get("proprioception")}
        finally:
            client.close()
        shared._write(self.out / f"{phase}-frames.json", images)
        return images

    def audit(self):
        result = super().audit()
        cameras = audit_cameras(self.records, self.marks)
        result["event_cameras"] = cameras
        result["checks"]["all_event_cameras_advance"] = cameras["pass"]
        result["pass"] = result["pass"] and cameras["pass"]
        shared._write(self.out / "gpu-physical-audit.json", result)
        return result
