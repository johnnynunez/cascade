"""Fresh read-only kitchen placement evidence, separate from actuator feedback.

The installed checkout supplies the same scene binding and geometry auditors
as the external acceptance proof. Missing optional resources fail unverified.
No prior READY receipt, motion result or authored spawn pose is reused.
"""
from __future__ import annotations

from functools import lru_cache
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import time
import uuid

from .bridge_client import BridgeClient
from .truth import _match_label

ROOT = Path(__file__).resolve().parents[3]
WALL_BUDGET_S = 20.0
REQUEST_TIMEOUT_S = 4.0
INTERVAL_S = 0.1


@lru_cache(maxsize=1)
def _auditor():
    spec = importlib.util.spec_from_file_location(
        "cascade_kitchen_placement_verdict", ROOT / "demo/kitchen/physics/placement_verdict.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_placement(endpoint, robot_id, label, destination, *, evidence_dir=None):
    """Sample only after the caller's motion finishes, using a private socket.

    A failed/timed-out read cannot poison a camera or motion client's stream.
    Both wall time and sample count are bounded. Frozen or restarted simulation
    clocks never become a successful settling window.
    """
    report = {"status": "unverified", "evidence": "placement physics unavailable",
              "measured": {"destination": destination, "scope": "postplacement only"}}
    records, client, expected, code = [], None, None, ""
    started = time.monotonic()
    deadline = started + WALL_BUDGET_S
    try:
        auditor = _auditor()
        expected = auditor.gpu.load_expected_scene_geometry(ROOT / "demo/scene/kitchen_config.json")
        names = tuple(expected["prop_dimensions_m"])
        object_name = _match_label(label, {name: name for name in names})
        if object_name is None or destination not in {"green square", "open box"}:
            raise ValueError("requested object or destination has no unambiguous kitchen identity")
        code = auditor.gpu.snapshot_code(object_name=object_name, include_scene_geometry=True,
            include_open_box="open_box" in expected, expected_props=names)
        client = BridgeClient(host=endpoint[0], port=endpoint[1], timeout_s=REQUEST_TIMEOUT_S)
        client.connect()
        for _ in range(200):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("placement observation exceeded its wall-time budget")
            began = time.monotonic()
            reply = client.request({"op": "exec", "code": code},
                                   timeout_s=min(REQUEST_TIMEOUT_S, remaining))
            sample = auditor.gpu.original.parse_snapshot_reply(reply)
            finished = time.monotonic()
            if finished > deadline:
                raise TimeoutError("placement observation exceeded its wall-time budget")
            records.append({"sequence": len(records), "client_started_monotonic": began,
                            "client_finished_monotonic": finished, "physics": sample})
            clock = sample.get("sim_time")
            step = sample.get("physics_step")
            if (type(clock) not in (int, float) or not math.isfinite(clock)
                    or type(step) is not int or step < 0):
                raise ValueError("invalid simulation clock in placement observation")
            if len(records) > 1:
                previous = records[-2]["physics"]
                if clock < previous["sim_time"] or step < previous["physics_step"]:
                    raise ValueError("simulation restarted during placement observation")
            if len(records) >= 6 and clock - records[0]["physics"]["sim_time"] >= .6:
                break
            time.sleep(min(INTERVAL_S, max(0., deadline - time.monotonic())))
        else:
            raise TimeoutError("placement observation exhausted its sample budget")
        report = auditor.audit_placement(records, object_name=object_name,
            destination_name=destination, expected_scene_geometry=expected, robot_id=robot_id)
        report["measured"]["destination"] = destination
    except Exception as exc:
        report = {"status": "unverified",
                  "evidence": f"placement observation unavailable: {type(exc).__name__}: {exc}",
                  "measured": {"destination": destination, "scope": "postplacement only"}}
    finally:
        if client is not None:
            client.close()
    report["measured"].update(observed_samples=len(records),
        observation_wall_s=round(time.monotonic() - started, 3))
    if evidence_dir is not None:
        # Preserve the readings used by this call alongside its normal trace.
        # A receipt write failure cannot erase a physical refutation.
        try:
            out = Path(evidence_dir)
            out.mkdir(parents=True, exist_ok=True)
            path = out / f"{uuid.uuid4().hex}.json"
            payload = {"version": 1, "requested_object": label, "destination": destination,
                       "robot_id": robot_id, "expected_scene_geometry": expected,
                       "snapshot_code_sha256": hashlib.sha256(code.encode()).hexdigest(),
                       "records": records, "verdict": report}
            raw = json.dumps(payload, allow_nan=False, separators=(",", ":")) + "\n"
            path.write_text(raw)
            report["measured"].update(evidence_path=str(path),
                evidence_sha256=hashlib.sha256(raw.encode()).hexdigest())
        except (OSError, ValueError, TypeError) as exc:
            report["measured"]["evidence_write_error"] = type(exc).__name__
    return report
