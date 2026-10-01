#!/usr/bin/env python3
"""Keep pregrasp camera anchors across a labelled post-close mapper failure.

One new isolated Isaac scene and one runtime execute a real grasp. After the
normal two-stage close returns, only depth integration requests are refused
until the five-second payload barrier fails. Restoring the real mapper must
let ONE public reset withdraw that exact contact episode, reach home and reset.
The injection does not alter producer captures, contact forces, depth, physics,
grasp selection, RPC deadlines or motion gates. It is not a real network outage.

The experimental map query extends X to 0.58 m to cover observed held surfaces
at home; the TCP workspace still ends at 0.50 m and clearance stays 30 mm.
Cleanup sends no motion, jaw or prop-reset commands. No automatic retry occurs.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import os
from pathlib import Path
import sys
import threading
import time
import traceback

from nvblox_camera_recovery import (ROOT, OBJECTS, ActuatorTrace, close_without_motion,
    fresh_camera_checks, fresh_state, phase, reset_window, write_json)
from nvblox_contact_recovery import bilateral, rebuilt_camera_map, retained_retreat_phase, sha


class PostCloseMapFault:
    """Fail integrations after the actual close; leave every other RPC intact."""
    def __init__(self, runtime, out):
        self.runtime, self.out = runtime, out
        self.occupancy = runtime.arm.harness.occupancy
        self.original_request = self.occupancy._client.request
        self.original_close = runtime._close_two_stage
        self.active = False
        self.started_monotonic = None
        self.failures = []
        self.anchors = {}
        self._lock = threading.Lock()

    def _save_anchors(self):
        import numpy as np
        if not self.occupancy._refresh_lock.acquire(timeout=3.):
            raise RuntimeError("could not snapshot pre-close depth anchors")
        try:
            history = copy.deepcopy(self.occupancy._depth_history)
        finally:
            self.occupancy._refresh_lock.release()
        if set(history) != {"cam0", "side", "proof"} or not all(history.values()):
            raise RuntimeError("all three measured pre-close camera anchors are required")
        for camera, frames in history.items():
            anchor = frames[0]
            props = {f"prop_{i}": name for i, name in enumerate(sorted(anchor["props"]))}
            path = self.out / ("preclose-anchor-" + camera + ".npz")
            np.savez_compressed(path, depth=anchor["depth"], K=anchor["K"], T=anchor["T"],
                robot=anchor["robot"], **{k: anchor["props"][p] for k, p in props.items()})
            self.anchors[camera] = {"t": anchor["t"], "props": props, "sha256": sha(path)}

    def __enter__(self):
        from cascade.perception.occupancy import OccupancyError
        def request(packet, *args, **kwargs):
            with self._lock:
                fail = self.active and packet.get("action") in {"integrate_depth", "integrate_masked_depth"}
                if fail:
                    self.failures.append({"monotonic": time.monotonic(), "action": packet["action"]})
            if fail:
                raise OccupancyError("labelled diagnostic integration fault after actual jaw close")
            return self.original_request(packet, *args, **kwargs)
        def close(profile):
            self._save_anchors()
            result = self.original_close(profile)
            with self._lock:
                self.started_monotonic = time.monotonic()
                self.active = True
            return result
        self.occupancy._client.request = request
        self.runtime._close_two_stage = close
        return self

    def __exit__(self, *_):
        with self._lock:
            self.active = False
        self.occupancy._client.request = self.original_request
        self.runtime._close_two_stage = self.original_close

    def report(self):
        with self.occupancy._refresh_lock:
            current = {name: frames[0]["t"] for name, frames in self.occupancy._depth_history.items() if frames}
        return {"label": "synthetic RPC refusal after real close; not a measured network outage",
            "started_monotonic": self.started_monotonic, "refused_integrations": list(self.failures),
            "preclose_anchors": self.anchors, "current_anchor_stamps": current,
            "all_original_anchors_retained": bool(self.anchors) and all(
                current.get(name) == evidence["t"] for name, evidence in self.anchors.items())}


def run(args, out):
    from kitchen_acceptance import load_proof, retarget_ports, install_command_trace
    from cascade.apps.demo import build_runtime
    from cascade.config import load_demo_config
    from cascade.skills import contact_episode
    cfg = load_demo_config(cameras=["isaac", "isaac_side", "isaac_proof"], arm="isaac_kitchen_gpu", llm="mock")
    retarget_ports(cfg._data, args.port)
    original_workspace = cfg.safety.workspace.as_dict()
    original_clearance = float(cfg.safety.get("min_clearance_m", .03))
    cfg._data["occupancy"].update(port=args.occupancy_port, track_payload=True,
        allowed_contact_paths=["/World_Props/" + name for name in OBJECTS],
        region_min=[.10, -.30, -.01], region_max=[.58, .30, .55])
    cfg._data["grasp"].update(pregrasp_offset_m=.10, place_support_clearance_m=.04)
    cfg._data["grasp"]["graspgenx"]["port"] = args.grasp_port
    for camera in cfg._data["cameras"]:
        camera["map_depth"] = True
    receipt = {"pass": False, "scope": __doc__, "errors": [], "phases": {},
        "experimental_map_query_region": {"min": [.10, -.30, -.01], "max": [.58, .30, .55],
            "reason": "Read-only retained-scene replay showed the original home endpoint outside query support; expanded query used existing observed ESDF, with zero unknown samples at that endpoint."},
        "effective_configuration": {"safety": cfg.safety.as_dict(), "arm": cfg.arm.as_dict(),
            "occupancy": cfg.occupancy.as_dict(), "grasp": cfg.grasp.as_dict()},
        "workspace_and_clearance_unchanged": cfg.safety.workspace.as_dict() == original_workspace
            and float(cfg.safety.get("min_clearance_m", .03)) == original_clearance}
    sources = sorted(set([Path(__file__), ROOT / "benchmark/diagnostics/nvblox_camera_recovery.py",
        ROOT / "benchmark/diagnostics/nvblox_contact_recovery.py", ROOT / "benchmark/diagnostics/kitchen_acceptance.py",
        args.scene_config] + list((ROOT / "src/cascade").rglob("*.py"))
        + list((ROOT / "demo/kitchen/physics").glob("*.py")) + list((ROOT / "configs").rglob("*.yaml"))
        + list((ROOT / "scripts").glob("isaac*.py"))))
    receipt["source_sha256"] = {str(p.relative_to(ROOT)): sha(p) for p in sources}
    def check_sources():
        if not all(sha(ROOT / p) == h for p, h in receipt["source_sha256"].items()):
            raise RuntimeError("source changed before requested physical phase")
    runtime = observer = recorder = None
    try:
        runtime, _ = build_runtime(cfg, out / "runtime", lazy_arm=True)
        install_command_trace(runtime, out / "skills.jsonl")
        recorder = ActuatorTrace(runtime, out / "actuators.jsonl")
        occ = runtime.arm.harness.occupancy
        receipt["occupancy_probe"] = occ.probe(timeout_ms=500)
        if (receipt["occupancy_probe"].get("backend") != "nvblox"
                or not receipt["occupancy_probe"].get("masked_depth")
                or not str(receipt["occupancy_probe"].get("device", "")).startswith("cuda:")):
            raise RuntimeError("real native masked CUDA nvblox required")
        proof = load_proof()
        expected = proof.shared.load_expected_scene_geometry(args.scene_config)
        observer = proof.shared.GpuProofObserver(out / "witness", port=args.port, interval_s=.15,
            budget_s=900, object_name=args.object, destination_name="green square", expected_scene_geometry=expected)
        with observer:
            check_sources()
            setup = phase(observer, runtime, recorder, "setup_reset", lambda: runtime.execute("reset_scene", {}))
            receipt["phases"]["setup"] = setup
            if not setup["result"].get("ok"):
                raise RuntimeError("initial reset failed; no faulted grasp attempted")
            receipt["setup_camera_map_commits"] = rebuilt_camera_map(occ, setup["result"])
            if not receipt["setup_camera_map_commits"]["pass"]:
                raise RuntimeError("initial reset did not rebuild all three real map sources")
            check_sources()
            with PostCloseMapFault(runtime, out) as fault:
                grasp = phase(observer, runtime, recorder, "postclose_failure",
                    lambda: runtime.execute("grasp_object", {"label": args.object.replace("_", " ")}))
                receipt["phases"]["failed_grasp"] = grasp
                receipt["fault"] = fault.report()
                episode = getattr(runtime, "_contact_episode", None)
                begun = fault.started_monotonic
                after_close_calls = [e for e in recorder.events if begun is not None
                    and e["monotonic"] >= begun and e["event"] == "begin" and e["layer"] == "actuator"]
                receipt["postclose_actuator_calls"] = after_close_calls
                bilateral(grasp["after"]["physics"], "/World_Props/" + args.object)
                receipt["failure_checks"] = {
                    "integration_fault_exercised": bool(fault.failures),
                    "payload_deadline_reported": not grasp["result"].get("ok", True)
                        and "post-close payload geometry" in grasp["result"].get("error", "")
                        and "deadline" in grasp["result"].get("error", ""),
                    "no_actuation_after_close": not after_close_calls,
                    "episode_and_provisional_retained": episode is not None
                        and getattr(runtime, "_held_provisional", None) is not None,
                    "global_exemption_removed": runtime.arm.harness._grasp_exempt is None,
                    "preclose_anchors_retained": receipt["fault"]["all_original_anchors_retained"],
                }
                if not all(receipt["failure_checks"].values()):
                    raise RuntimeError("post-close failure did not satisfy the retained-contact contract")
            xy, radius, z_min = episode["cylinder"]
            historical = {"object": args.object, "q_pre": episode["q_pre"].copy(),
                "cylinder": [xy.tolist(), radius, z_min], "stationary_physics": grasp["after"]["physics"]}
            receipt["retained_episode"] = {"q_pre": historical["q_pre"], "cylinder": historical["cylinder"],
                "q_failure": episode["q_failure"], "expected_paths": episode["expected_paths"],
                "same_runtime": True, "rehydration": False}
            check_sources()
            original = contact_episode.recover
            contact_episode.recover = lambda rt: retained_retreat_phase(observer, rt, recorder, historical, out, original)
            try:
                reset = phase(observer, runtime, recorder, "recovered_reset", lambda: runtime.execute("reset_scene", {}))
            finally:
                contact_episode.recover = original
            receipt["phases"]["reset"] = reset
            receipt["phases"]["contact_retreat"] = getattr(runtime, "_diagnostic_contact_retreat", None)
            receipt["postclose_geometry"] = episode.get("barrier")
            observer.settle(simulation_seconds=.65, wall_timeout=40)
            receipt["final_state"] = fresh_state(observer, runtime)
            with observer._lock:
                records = copy.deepcopy(observer.records)
            receipt["reset_physics"] = reset_window(records, after=observer.marks["recovered_reset_end"]["monotonic"],
                cfg=cfg, expected_props=expected["prop_dimensions_m"], engine="physx")
            receipt["camera_map_commits"] = rebuilt_camera_map(occ, reset["result"])
            receipt["checks"] = {**receipt["failure_checks"],
                "public_reset_completed": reset["result"].get("ok") is True,
                "independent_retreat_evidence": bool(receipt["phases"]["contact_retreat"]
                    and receipt["phases"]["contact_retreat"]["pass"]),
                "reset_physics": receipt["reset_physics"]["pass"],
                "every_reset_camera_committed": receipt["camera_map_commits"]["pass"],
                "fresh_map": occ._grid is not None and not occ.scene_reset_pending and not occ.is_stale()
                    and occ._body_error is None and occ.last_error is None,
                "retained_episode_cleared": getattr(runtime, "_contact_episode", None) is None,
                **fresh_camera_checks(reset["result"], {"cam0", "side", "proof"}),
            }
    except Exception:
        receipt["errors"].append(traceback.format_exc())
    finally:
        if runtime is not None:
            if recorder:
                recorder.phase = "cleanup_without_motion"
            receipt["cleanup_errors"] = close_without_motion(runtime)
            receipt["cleanup_calls"] = recorder.calls("cleanup_without_motion") if recorder else []
        receipt.setdefault("checks", {}).update(
            source_unchanged=all(sha(ROOT / p) == h for p, h in receipt["source_sha256"].items()),
            workspace_and_clearance_unchanged=receipt["workspace_and_clearance_unchanged"],
            no_diagnostic_errors=not receipt["errors"] and not receipt.get("cleanup_errors"),
            observer_complete=bool(observer and observer.complete and not observer.errors),
            cleanup_sent_no_commands=not receipt.get("cleanup_calls"))
        receipt["pass"] = all(receipt["checks"].values())
        receipt["artifacts_sha256"] = {str(p.relative_to(out)): sha(p)
            for p in out.rglob("*") if p.is_file() and p.name != "receipt.json"}
        write_json(out / "receipt.json", receipt)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8692)
    parser.add_argument("--occupancy-port", type=int, default=25560)
    parser.add_argument("--grasp-port", type=int, default=25556)
    parser.add_argument("--object", choices=OBJECTS, default="tomato_can")
    parser.add_argument("--scene-config", type=Path, default=ROOT / "demo/scene/kitchen_config.json")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    args.output, args.scene_config = args.output.resolve(), args.scene_config.resolve()
    if (args.output.exists() or not args.output.is_relative_to(ROOT)
            or not args.scene_config.is_relative_to(ROOT) or not args.scene_config.is_file()):
        parser.error("scene-config must exist inside the checkout and output must be a new directory inside it")
    if args.port == 8691 or args.occupancy_port == 25559:
        parser.error("the retained trial16 scene/map is reserved; choose the new isolated pair")
    with open(f"/tmp/cascade-acceptance-{args.port}.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        args.output.mkdir(parents=True)
        sys.path.insert(0, str(ROOT / "src"))
        os.environ.update(CASCADE_BRIDGE_PORT=str(args.port), CASCADE_OCCUPANCY="1", CASCADE_INSTALL_PROFILE="spark",
            CASCADE_BELIEFS="0", CASCADE_ISAAC_PIXEL_MASK="1", CASCADE_PROOF_CAMERA="1", CASCADE_REQUIRE_CUDA="1",
            YOLO_OFFLINE="True", ULTRALYTICS_OFFLINE="True", CASCADE_GRASP_MEMORY_PATH=str(args.output / "grasp-memory.json"))
        os.chdir(ROOT / "models")
        receipt = run(args, args.output)
    print(args.output, "PASS" if receipt["pass"] else "FAIL", flush=True)
    return 0 if receipt["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
