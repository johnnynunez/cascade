"""Validate explicit renderer-to-physics bindings without conflating clocks."""
import hashlib
import json
import math
import re


def valid_render_binding(capture):
    ref = capture.get("render_reference") if isinstance(capture, dict) else None
    if not isinstance(ref, dict):
        return False
    if ref.get("source") == "rpFabricTime":
        return True  # Existing callers retain their native token validation.
    if ref.get("source") != "ovrtx_snapshot":
        return False
    state = capture.get("proprioception")
    renderer = ref.get("renderer")
    if not isinstance(state, dict) or not isinstance(renderer, dict):
        return False
    try:
        digest = hashlib.sha256(json.dumps(state, sort_keys=True, separators=(",", ":"),
                                           allow_nan=False).encode()).hexdigest()
    except (ValueError, TypeError):
        return False

    def number(v):
        return type(v) in (int, float) and math.isfinite(v) and v >= 0

    if not all(isinstance(ref.get(k), str) and re.fullmatch("[0-9a-f]{64}", ref[k])
               for k in ("snapshot_sha256", "scene_sha256", "proprioception_sha256")):
        return False
    times = [renderer.get(k) for k in ("step_start_s", "sensor_start_s", "sensor_end_s", "step_end_s")]
    return (ref["proprioception_sha256"] == digest
            and ref.get("producer_epoch") == state.get("producer_epoch")
            and ref.get("snapshot_started_monotonic") == capture.get("t") == state.get("t")
            and number(ref.get("snapshot_finished_monotonic")) and number(capture.get("t"))
            and ref["snapshot_finished_monotonic"] >= capture["t"]
            and type(ref.get("history_physics_step")) is int and ref["history_physics_step"] >= 0
            and number(ref.get("history_simulation_time"))
            and isinstance(ref.get("renderer_epoch"), str) and bool(ref["renderer_epoch"])
            and type(renderer.get("ordinal")) is int and renderer["ordinal"] >= 2
            and all(number(v) for v in times)
            and times[0] <= times[1] == times[2] <= times[3] and times[0] < times[3])
