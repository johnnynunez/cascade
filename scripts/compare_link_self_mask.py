#!/usr/bin/env python3
"""B39: the link-geometry self-mask against Isaac's render self-mask.

`workspace_filter.link_self_mask` builds the robot's pixels from the arm's
URDF collision geometry for frames that carry no render self-mask (the real
rig). Isaac frames carry both ingredients: the render mask
(CASCADE_ISAAC_PIXEL_MASK=1) and the joint snapshot taken at capture time.
This script measures one against the other, in two steps.

record (live, Isaac): build the real runtime with the link mask on, optionally
move the arm through the given poses (SafeArm: the harness vets every move and
a refusal is logged, never forced), and for every fusing camera save N frames
as compressed .npz: image, render self-mask (minus a held payload, exactly
what fusion's gate uses), K, cam->base T, the capture-time joint snapshot
(local convention), the mask the RUNTIME'S OWN link mask built for that frame
(nearest joint sample: the real-rig path) and, unless --no-detect, every
detection mask of the runtime's detector.

score (offline, CPU): re-build the link mask from the config (any dilate_px /
cell_m you pass) for each recorded frame at its capture-time joints and report
IoU, coverage (of the render mask) and bloat, per frame and per camera, plus
how often fusion's gate (robot fraction > self_mask_max_frac) decides the same
for each detection under both masks. A frame without a capture snapshot is
reported as such, never scored with another pose.

    # private bridge first (CASCADE_ISAAC_PIXEL_MASK=1, a port of your block)
    CASCADE_BRIDGE_PORT=<p> python scripts/compare_link_self_mask.py record \\
        --out <dir> --frames 5 --pose home --pose 0.3,0.6,1.6,0.2,-0.4,0.5
    python scripts/compare_link_self_mask.py score <dir> --json <dir>/score.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

SHARED_PORTS = {8611, 8080, 5556, 5557, 18789, 18790}


# ── recording ──────────────────────────────────────────────────────────────


def _np_mask(mask) -> np.ndarray:
    """A detection mask as a host bool array (the strict-CUDA detector's are
    torch tensors on the GPU)."""
    if hasattr(mask, "detach"):
        mask = mask.detach().to("cpu").numpy()
    return np.asarray(mask, dtype=bool)


def save_record(path, *, frame, camera: str, T, render_self, capture_state=None,
                runtime_mask=None, runtime_info=None, dets=()) -> Path:
    """One recorded frame as a compressed .npz (see `load_record`)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "rgb": np.asarray(frame.rgb, dtype=np.uint8),
        "K": np.asarray(frame.K, dtype=np.float64),
        "T": np.asarray(T, dtype=np.float64).reshape(4, 4),
        "t": np.float64(frame.t),
        "camera": np.str_(camera),
        "render_self": np.asarray(render_self, dtype=bool),
    }
    if capture_state is not None:
        data["capture_q"] = np.asarray(capture_state.q, dtype=np.float64)
        data["capture_gripper_pos"] = np.float64(capture_state.gripper_pos)
        data["capture_gripper_valid"] = np.bool_(capture_state.gripper_valid)
    if runtime_mask is not None:
        data["runtime_mask"] = np.asarray(runtime_mask, dtype=bool)
    data["runtime_info"] = np.str_(json.dumps(runtime_info or {}, default=str))
    dets = list(dets)
    if dets:
        data["det_masks"] = np.stack([_np_mask(d.mask) for d in dets])
        data["det_labels"] = np.asarray([str(d.label) for d in dets])
    np.savez_compressed(path, **data)
    return path


def load_record(path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        rec = {k: z[k] for k in z.files}
    rec["path"] = str(path)
    rec["camera"] = str(rec["camera"])
    rec["t"] = float(rec["t"])
    rec["runtime_info"] = json.loads(str(rec.get("runtime_info", "{}")))
    return rec


def _ports(node, path=""):
    if hasattr(node, "items"):
        for k, v in node.items():
            p = f"{path}.{k}" if path else str(k)
            if str(k).endswith("port") and isinstance(v, (int, str)) and str(v).isdigit():
                yield p, int(v)
            else:
                yield from _ports(v, p)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from _ports(v, f"{path}[{i}]")


def _parse_pose(text: str, home) -> np.ndarray:
    if text == "home":
        if home is None:
            raise SystemExit("--pose home: the arm profile has no home_q")
        return np.asarray(home, dtype=float)
    return np.asarray([float(v) for v in text.split(",")], dtype=float)


def record(args) -> int:
    from cascade.apps.demo import build_runtime, shutdown_runtime
    from cascade.config import load_demo_config
    from cascade.control.isaac_arm import IsaacArm

    cfg = load_demo_config(cameras=args.cameras.split(","), arm=args.arm, llm="mock")
    cfg._data.setdefault("stream", {})["mode"] = "off"
    block = cfg._data.setdefault("workspace_filter", {}).setdefault("link_self_mask", {})
    block["enabled"] = True
    ports = sorted(set(_ports(cfg._data)))
    bad = [(p, v) for p, v in ports if v in SHARED_PORTS]
    if bad:
        print(f"[!] refusing: shared ports resolved {bad}", file=sys.stderr)
        return 3
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rt, arm = build_runtime(cfg, out / "run", view=False, serve=False)
    log = {"ports": ports, "poses": [], "frames": 0, "skipped": []}
    try:
        link = rt._link_self_mask
        if link is None:
            print("[!] the runtime built no link self-mask (see the log above)", file=sys.stderr)
            return 4
        if len(link.arms) != 1:
            print("[!] one arm only: the capture snapshot names one robot", file=sys.stderr)
            return 4
        acfg = cfg.arm
        observer = IsaacArm(acfg)  # parses the capture snapshot; never connects
        cams = [c for c in rt.watcher._cams if c.fuse] if rt.watcher is not None else []
        if not cams:
            print("[!] no fusing camera", file=sys.stderr)
            return 4
        poses = [("current", None)] if not args.pose else [(p, _parse_pose(p, acfg.get("home_q")))
                                                             for p in args.pose]
        for pi, (name, q) in enumerate(poses):
            if q is not None:
                try:
                    rt.arm.move_joints(q, duration_s=args.move_s)
                except Exception as exc:  # noqa: BLE001 - the harness refusing is a result
                    log["poses"].append({"pose": name, "moved": False, "error": f"{type(exc).__name__}: {exc}"})
                    print(f"[pose {name}] not reached: {exc}", file=sys.stderr)
                    continue
                time.sleep(args.settle_s)
            log["poses"].append({"pose": name, "moved": q is not None})
            for cam in cams:
                for k in range(args.frames):
                    frame = cam.depth.ensure_depth(cam.stream.get_frame())
                    render = rt._workspace.self_pixels(frame)
                    if render is None:
                        log["skipped"].append({"pose": name, "camera": cam.stream.name,
                                               "why": "no render self-mask (CASCADE_ISAAC_PIXEL_MASK=1?)"})
                        continue
                    T = frame.T_base_cam if frame.T_base_cam is not None else cam.extrinsics.cam_to_base()
                    try:
                        capture = observer.state_from_frame(frame)
                    except Exception as exc:  # noqa: BLE001 - recorded as missing, never replaced
                        capture = None
                        log["skipped"].append({"pose": name, "camera": cam.stream.name,
                                               "why": f"capture snapshot: {exc}"})
                    bare = replace(frame, robot_mask=None, payload_mask=None)
                    link.sample()
                    runtime_mask = link.mask_for(bare, T)
                    dets = [] if args.no_detect else [
                        d for d in rt.detector.detect(frame, classes=rt._default_classes) if d.mask is not None]
                    save_record(out / f"p{pi:02d}_{cam.stream.name}_{k:02d}.npz", frame=frame,
                                camera=cam.stream.name, T=T, render_self=render, capture_state=capture,
                                runtime_mask=runtime_mask, runtime_info=link.last, dets=dets)
                    log["frames"] += 1
                    time.sleep(args.interval_s)
    finally:
        shutdown_runtime(rt, arm)
        (out / "record.json").write_text(json.dumps(log, indent=1, default=str))
    print(json.dumps({k: log[k] for k in ("frames", "poses", "skipped")}, default=str))
    return 0


# ── scoring ────────────────────────────────────────────────────────────────


def mask_metrics(mask, truth) -> dict:
    mask, truth = np.asarray(mask, dtype=bool), np.asarray(truth, dtype=bool)
    n, inter, union = int(truth.sum()), int((mask & truth).sum()), int((mask | truth).sum())
    return {"truth_px": n, "mask_px": int(mask.sum()),
            "iou": inter / union if union else 1.0,
            "coverage": inter / n if n else 1.0,
            "bloat": int((mask & ~truth).sum()) / n if n else None}


class _Recorded:
    """The arm a recorded frame was captured with (state_fn for LinkSelfMask)."""

    connected = True

    def __init__(self):
        self.state = None

    def get_state(self):
        return self.state


def _link_for(cfg):
    from cascade.control.kinematics import Kinematics
    from cascade.perception.link_mask import build_link_self_mask

    acfg = cfg.arm
    kin = Kinematics(model_path=acfg.model, ee_frame=acfg.get("ee_frame", "gripper_end"),
                     n_controlled=int(acfg.get("n_joints", 6)), joint_signs=acfg.get("joint_signs"))
    proto = build_link_self_mask(cfg, [(str(acfg.get("name", "arm0")), acfg, kin, _Recorded())])
    if proto is None:
        raise SystemExit("link self-mask unavailable (pinocchio missing or the gate is off)")
    return proto


def score_records(paths, cfg) -> dict:
    """Score recorded frames; `cfg` supplies the link-mask settings and arm."""
    from cascade.perception.link_mask import LinkSelfMask
    from cascade.perception.workspace import WorkspaceFilter
    from cascade.types import Frame, RobotState

    proto = _link_for(cfg)
    arm = proto.arms[0]
    gate = WorkspaceFilter.from_config(cfg.get("workspace_filter"))
    rows, per_cam = [], {}
    for path in sorted(paths):
        rec = load_record(path)
        row = {"file": Path(path).name, "camera": rec["camera"]}
        render = rec["render_self"]
        if "runtime_mask" in rec:
            row["runtime"] = mask_metrics(rec["runtime_mask"], render)
            row["runtime_reason"] = rec["runtime_info"].get("reason")
        if "capture_q" not in rec:
            row["capture"] = None
            row["why"] = "no capture-time joint snapshot: not scored"
            rows.append(row)
            continue
        state = RobotState(q=rec["capture_q"], gripper_pos=float(rec["capture_gripper_pos"]),
                           gripper_valid=bool(rec["capture_gripper_valid"]), t=rec["t"])
        reader = _Recorded()
        reader.state = state
        link = LinkSelfMask(dilate_px=proto.dilate_px, max_skew_s=proto.max_skew_s)
        link.add_arm(arm.name, arm.geometry, arm.poses, reader.get_state, base_T=arm.base_T)
        frame = Frame(rgb=rec["rgb"], depth_m=None, K=rec["K"], t=rec["t"])
        mask = link.mask_for(frame, rec["T"])
        if mask is None:
            row["capture"] = None
            row["why"] = f"no link mask: {link.last}"
            rows.append(row)
            continue
        row["capture"] = mask_metrics(mask, render)
        row["ms"] = round(link.last["ms"], 2)
        if "det_masks" in rec:
            decisions = []
            for label, dm in zip(rec["det_labels"], rec["det_masks"], strict=True):
                rest_r, f_render = gate.exclude_self(dm, render)
                rest_l, f_link = gate.exclude_self(dm, mask)
                drop_r, drop_l = rest_r is None, rest_l is None   # fusion's own decision
                decisions.append({"label": str(label), "frac_render": round(float(f_render), 4),
                                  "frac_link": round(float(f_link), 4), "drop_render": bool(drop_r),
                                  "drop_link": bool(drop_l), "agree": bool(drop_r == drop_l)})
            row["detections"] = decisions
        rows.append(row)
        per_cam.setdefault(rec["camera"], []).append(row)
    summary = {}
    for cam, cam_rows in per_cam.items():
        ious = [r["capture"]["iou"] for r in cam_rows]
        covs = [r["capture"]["coverage"] for r in cam_rows]
        dets = [d for r in cam_rows for d in r.get("detections", [])]
        summary[cam] = {"frames": len(cam_rows), "iou_min": min(ious), "iou_median": float(np.median(ious)),
                        "coverage_min": min(covs), "coverage_median": float(np.median(covs)),
                        "bloat_max": max(r["capture"]["bloat"] or 0.0 for r in cam_rows),
                        "detections": len(dets), "gate_agree": sum(d["agree"] for d in dets),
                        "dropped_render": sum(d["drop_render"] for d in dets),
                        "dropped_link": sum(d["drop_link"] for d in dets)}
    return {"settings": {"dilate_px": proto.dilate_px, "max_skew_s": proto.max_skew_s,
                         "self_mask_max_frac": gate.self_mask_max_frac,
                         "sources": dict(arm.geometry.sources)},
            "summary": summary, "unscored": sum(1 for r in rows if r.get("capture") is None),
            "rows": rows}


def score(args) -> int:
    from cascade.config import load_demo_config

    cfg = load_demo_config(cameras=args.cameras.split(","), arm=args.arm, llm="mock")
    block = cfg._data.setdefault("workspace_filter", {}).setdefault("link_self_mask", {})
    block["enabled"] = True
    for key, value in (("dilate_px", args.dilate_px), ("cell_m", args.cell_m)):
        if value is not None:
            block[key] = value
    paths = sorted(Path(args.dir).glob("*.npz"))
    if not paths:
        print(f"[!] no recorded frames in {args.dir}", file=sys.stderr)
        return 2
    report = score_records(paths, cfg)
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=1, default=str))
    print(json.dumps({"settings": {k: v for k, v in report["settings"].items() if k != "sources"},
                      "summary": report["summary"], "unscored": report["unscored"]}, indent=1))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    rec = sub.add_parser("record", help="live: record Isaac frames (render mask + capture joints)")
    rec.add_argument("--out", required=True)
    rec.add_argument("--frames", type=int, default=5, help="frames per camera per pose")
    rec.add_argument("--pose", action="append", default=[],
                     help="'home' or comma-separated joint values (local convention); repeatable")
    rec.add_argument("--move-s", type=float, default=4.0)
    rec.add_argument("--settle-s", type=float, default=1.0)
    rec.add_argument("--interval-s", type=float, default=0.3)
    rec.add_argument("--no-detect", action="store_true")
    sc = sub.add_parser("score", help="offline: score recorded frames")
    sc.add_argument("dir")
    sc.add_argument("--json")
    sc.add_argument("--dilate-px", type=int)
    sc.add_argument("--cell-m", type=float)
    for p in (rec, sc):
        p.add_argument("--cameras", default="isaac,isaac_side")
        p.add_argument("--arm", default="isaac")
    args = ap.parse_args(argv)
    return record(args) if args.cmd == "record" else score(args)


if __name__ == "__main__":
    sys.exit(main())
