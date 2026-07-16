"""The curated skill API the agent programs against (ASPIRE Appendix E.2
adapted to this rig). Every call is safety-gated, trace-logged with
before/after keyframes, and mirrored into episodic memory.

Skills return plain dicts with an `ok` flag; SkillError and SafetyViolation
become {"ok": false, "error": ...} so the agent can reason about failures
instead of crashing the loop.
"""

from __future__ import annotations

import time

import cv2
import numpy as np

from ..agent.trace import TraceLogger
from ..grasping import plan_grasps_from_fix, select_grasp, select_profile
from ..memory import BeliefStore, EpisodicMemory
from ..perception.grounding import (
    Extrinsics,
    localize_object,
    mask_to_points_cam,
    oriented_bbox,
)
from ..types import Frame, SafetyViolation, SkillError, make_transform, transform_points


def _jpeg(rgb: np.ndarray, quality: int = 85) -> bytes:
    ok, buf = cv2.imencode(".jpg", rgb, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes() if ok else b""


class SkillRuntime:
    def __init__(
        self,
        camera,
        depth_provider,
        detector,
        extrinsics: Extrinsics,
        kin,
        safe_arm,
        memory: EpisodicMemory,
        beliefs: BeliefStore,
        trace: TraceLogger,
        cfg,
    ):
        self.camera = camera
        self.depth = depth_provider
        self.detector = detector
        self.extrinsics = extrinsics
        self.kin = kin
        self.arm = safe_arm
        self.memory = memory
        self.beliefs = beliefs
        self.trace = trace
        self.cfg = cfg
        self.last_frame: Frame | None = None
        self.held_object: str | None = None
        g = cfg.arm.gripper
        self._grip_open = float(g.get("open_pos", 0.0))
        self._grip_closed = float(g.get("closed_pos", 1.0))
        self._max_width = float(g.get("max_width_m", 0.09))
        self._default_classes = list(cfg.get("detect_classes", ["cup", "bottle", "box", "fruit", "toy"]))

    # ── plumbing ─────────────────────────────────────────────────────────

    def execute(self, name: str, args: dict) -> dict:
        """Dispatch one skill call with tracing. Never raises."""
        fn = getattr(self, f"skill_{name}", None)
        if fn is None:
            return {"ok": False, "error": f"unknown skill {name!r}"}
        before = self.trace.save_keyframe(
            self.last_frame.rgb if self.last_frame is not None else None, f"{name}_before"
        )
        t0 = time.monotonic()
        try:
            result = fn(**args)
            if "ok" not in result:
                result["ok"] = True
        except (SkillError, SafetyViolation) as e:
            result = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        except TypeError as e:
            result = {"ok": False, "error": f"bad arguments for {name}: {e}"}
        except Exception as e:  # camera dropouts, CAN loss, ... : the agent
            # loop must survive and report, not crash mid-task.
            import traceback

            result = {
                "ok": False,
                "error": f"unexpected {type(e).__name__}: {e}",
                "hint": "hardware/runtime fault; consider task_done(success=false)",
            }
            traceback.print_exc()
        dur = (time.monotonic() - t0) * 1000
        after = self.trace.save_keyframe(
            self.last_frame.rgb if self.last_frame is not None else None, f"{name}_after"
        )
        self.trace.record(name, args, result, dur, before, after)
        self.memory.add(
            "action" if result["ok"] else "outcome",
            f"{name}({_short(args)}) -> " + ("ok" if result["ok"] else result["error"][:120]),
        )
        return result

    def observe(self) -> Frame:
        frame = self.camera.get_frame()
        frame = self.depth.ensure_depth(frame)
        self.last_frame = frame
        self.arm.harness.heartbeat()
        return frame

    def frame_jpeg(self) -> bytes | None:
        if self.last_frame is None:
            return None
        return _jpeg(self.last_frame.rgb)

    def _update_beliefs_from_frame(self, frame: Frame, dets) -> list[dict]:
        summaries = []
        if not frame.has_depth:
            return [
                {"label": d.label, "conf": round(d.conf, 2), "position": None}
                for d in dets
            ]
        T = self.extrinsics.cam_to_base()
        for d in dets:
            mask = d.mask
            if mask is None:
                h, w = frame.rgb.shape[:2]
                mask = np.zeros((h, w), dtype=bool)
                x0, y0, x1, y1 = d.bbox.astype(int)
                mask[max(y0, 0):min(y1, h), max(x0, 0):min(x1, w)] = True
            pts_cam = mask_to_points_cam(frame, mask)
            if pts_cam.shape[0] < 10:
                continue
            pts_base = transform_points(T, pts_cam)
            center, extents, _ = oriented_bbox(pts_base)
            self.beliefs.update(
                d.label, center, d.conf, extent=extents,
                top_z=float(pts_base[:, 2].max()), t=frame.t,
            )
            summaries.append(
                {
                    "label": d.label,
                    "conf": round(d.conf, 2),
                    "position": [round(float(x), 3) for x in center],
                }
            )
        return summaries

    def _tcp(self) -> np.ndarray:
        return self.kin.fk(self.arm.get_state().q)[:3, 3]

    def _gripper_width_frac(self) -> float | None:
        """0 = fully closed, 1 = fully open (from motor angle, linear map).
        None when the gripper position is unknown (feedback failure)."""
        state = self.arm.get_state()
        if not getattr(state, "gripper_valid", True):
            return None
        span = self._grip_closed - self._grip_open
        if abs(span) < 1e-9:
            return 0.0
        closed_frac = (state.gripper_pos - self._grip_open) / span  # 1 at closed
        return float(np.clip(1.0 - closed_frac, 0.0, 1.0))

    # ── skills ───────────────────────────────────────────────────────────

    def skill_get_observation(self) -> dict:
        frame = self.observe()
        dets = self.detector.detect(frame, classes=self._default_classes)
        objects = self._update_beliefs_from_frame(frame, dets)
        state = self.arm.get_state()
        tcp = self.kin.fk(state.q)[:3, 3]
        self.memory.add(
            "observation",
            f"saw {[o['label'] for o in objects]} (depth: {frame.depth_source})",
            rgb=frame.rgb,
        )
        wf = self._gripper_width_frac()
        return {
            "objects_visible": objects,
            "objects_remembered": self.beliefs.summary(),
            "robot": {
                "q_deg": [round(float(np.degrees(x)), 1) for x in state.q],
                "tcp_xyz": [round(float(x), 3) for x in tcp],
                "gripper_open_frac": round(wf, 2) if wf is not None else "unknown",
                "holding": self.held_object,
            },
            "depth_source": frame.depth_source,
        }

    def skill_list_objects(self) -> dict:
        return {
            "objects": self.beliefs.summary(),
            "holding": self.held_object,
            "note": "state=remembered means not currently visible; position is last known",
        }

    def skill_localize_object(self, label: str, spatial_hint: str | None = None) -> dict:
        frame = self.observe()
        fix = localize_object(
            frame, label, self.detector, self.extrinsics, spatial_hint=spatial_hint
        )
        self.beliefs.update(
            fix.label, fix.position, fix.detection.conf, extent=fix.extent,
            top_z=float(fix.points[:, 2].max()), t=frame.t,
        )
        return {
            "label": label,
            "position": [round(float(x), 3) for x in fix.position],
            "extent_m": [round(float(x), 3) for x in fix.extent],
            "n_points": int(fix.points.shape[0]),
        }

    def skill_grasp_object(
        self,
        label: str,
        material: str | None = None,
        spatial_hint: str | None = None,
    ) -> dict:
        if self.held_object:
            raise SkillError(f"already holding {self.held_object!r}; place it first")
        gcfg = self.cfg.grasp
        frame = self.observe()
        fix = localize_object(
            frame, label, self.detector, self.extrinsics, spatial_hint=spatial_hint
        )
        profile = select_profile(label, material)
        grasps = plan_grasps_from_fix(
            fix,
            table_z=float(self.cfg.safety.get("table_z", 0.0)),
            max_width_m=self._max_width,
            depth_fraction=float(gcfg.get("depth_fraction", 0.5)),
        )
        state = self.arm.get_state()
        grasp, q_pre, q_grasp = select_grasp(
            grasps,
            self.kin,
            state.q,
            max_width_m=self._max_width,
            pregrasp_offset_m=float(gcfg.get("pregrasp_offset_m", 0.12)),
        )

        # 1. open, go to pregrasp (normal speed)
        self.arm.set_gripper(self._grip_open, effort=0.8)
        if not self.arm.move_joints(q_pre, duration_s=float(gcfg.get("move_duration_s", 2.5))):
            raise SkillError("did not settle at pregrasp pose")

        # 2. descend inside the exemption cylinder (slow)
        self.arm.harness.allow_grasp_descent(
            grasp.position[:2],
            radius_m=float(gcfg.get("exempt_radius_m", 0.07)),
            z_min=float(grasp.position[2] - 0.02),
        )
        try:
            if not self.arm.move_joints(q_grasp, duration_s=float(gcfg.get("descend_duration_s", 2.0))):
                raise SkillError("did not settle at grasp pose")

            # 3. close with the material profile (two-stage, stall-aware)
            self._close_two_stage(profile)

            # 4. lift back to pregrasp (speed scaled by profile)
            lift_dur = float(gcfg.get("descend_duration_s", 2.0)) / max(profile.lift_speed_scale, 0.2)
            self.arm.move_joints(q_pre, duration_s=lift_dur)
        finally:
            self.arm.harness.clear_grasp_exemption()

        # 5. verify: ASPIRE heuristic on jaw travel after close. An object in
        # the jaws stalls them ABOVE the commanded close; if they reached the
        # commanded stage-2 fraction even though the object should be much
        # wider, nothing resisted: air grasp. Unknown feedback -> report the
        # grasp as unverified rather than dropping a possibly-held object.
        width_after_lift = self._gripper_width_frac()
        verified = width_after_lift is not None
        if verified:
            commanded_open = max(1.0 - profile.close_frac_stage2, 0.0)
            expected_open = min(grasp.width_m / self._max_width, 1.0)
            air_grasp = width_after_lift < float(gcfg.get("air_grasp_frac", 0.04)) or (
                width_after_lift <= commanded_open + 0.03
                and expected_open >= commanded_open + 0.07
            )
            if air_grasp:
                self.memory.add("outcome", f"grasp {label!r} FAILED: jaws closed on air")
                return {
                    "ok": False,
                    "error": "air grasp: gripper closed fully, object not held",
                    "suggestion": "re-localize the object or try the alternate yaw",
                }
        self.held_object = label
        self.beliefs.mark_removed(label, near=fix.position)
        self.memory.add(
            "action",
            f"grasped {label!r} (profile {profile.name}, "
            + (f"jaw at {width_after_lift:.2f})" if verified else "grip UNVERIFIED)"),
        )
        return {
            "held": label,
            "grip_profile": profile.name,
            "grip_verified": verified,
            "gripper_open_frac": round(width_after_lift, 2) if verified else None,
            "grasp_width_m": round(grasp.width_m, 3),
        }

    def _close_two_stage(self, profile) -> None:
        raw = self.arm.raw
        if hasattr(raw, "close_gripper_two_stage"):
            raw.close_gripper_two_stage(
                width_frac_stage1=profile.close_frac_stage1,
                width_frac_stage2=profile.close_frac_stage2,
                effort=profile.effort,
            )
            return
        span = self._grip_closed - self._grip_open
        for frac, eff in (
            (profile.close_frac_stage1, profile.effort * 0.7),
            (profile.close_frac_stage2, profile.effort),
        ):
            self.arm.set_gripper(self._grip_open + span * frac, effort=eff)
            time.sleep(float(self.cfg.grasp.get("close_settle_s", 0.0)))

    def skill_place_at(self, x: float, y: float, z: float | None = None) -> dict:
        if not self.held_object:
            raise SkillError("not holding anything")
        gcfg = self.cfg.grasp
        table_z = float(self.cfg.safety.get("table_z", 0.0))
        release_z = float(z) if z is not None else table_z + float(gcfg.get("release_height_m", 0.05))
        target = np.array([x, y, release_z])
        from ..grasping.obb_grasp import _yaw_rotation

        q_now = self.arm.get_state().q
        hover = target + np.array([0.0, 0.0, float(gcfg.get("pregrasp_offset_m", 0.12))])
        # Placement yaw is arbitrary: walk candidate yaws (radial first --
        # kindest to the wrist) until both hover and release poses solve.
        radial = float(np.arctan2(y, x))
        pre = low = None
        for yaw in (radial, 0.0, np.pi / 4, -np.pi / 4, np.pi / 2, -np.pi / 2):
            R = _yaw_rotation(yaw)
            cand_pre = self.kin.ik(make_transform(R, hover), q_now)
            if not cand_pre.success:
                continue
            cand_low = self.kin.ik(make_transform(R, target), cand_pre.q)
            if cand_low.success:
                pre, low = cand_pre, cand_low
                break
        if pre is None or low is None:
            raise SkillError(
                f"place pose unreachable at {target.round(3).tolist()} "
                f"(hover {hover.round(3).tolist()}, all yaws tried)"
            )

        if not self.arm.move_joints(pre.q, duration_s=float(gcfg.get("move_duration_s", 2.5))):
            raise SkillError("did not settle above the place target")
        self.arm.harness.allow_grasp_descent(target[:2], z_min=release_z - 0.02)
        try:
            if not self.arm.move_joints(low.q, duration_s=float(gcfg.get("descend_duration_s", 2.0))):
                raise SkillError("did not settle at place pose")
            self.arm.set_gripper(self._grip_open, effort=0.6)
            time.sleep(float(self.cfg.grasp.get("close_settle_s", 0.0)))
            self.arm.move_joints(pre.q, duration_s=float(gcfg.get("descend_duration_s", 2.0)))
        finally:
            self.arm.harness.clear_grasp_exemption()

        placed = self.held_object
        self.held_object = None
        self.beliefs.update(placed, target, 0.8)
        self.memory.add("action", f"placed {placed!r} at {target.round(3).tolist()}")
        return {"placed": placed, "at": [round(float(v), 3) for v in target]}

    def skill_place_on_object(self, label: str) -> dict:
        if not self.held_object:
            raise SkillError("not holding anything")
        belief = self.beliefs.find(label)
        if belief is None:
            frame = self.observe()
            fix = localize_object(frame, label, self.detector, self.extrinsics)
            self.beliefs.update(
                label, fix.position, fix.detection.conf, extent=fix.extent,
                top_z=float(fix.points[:, 2].max()),
            )
            belief = self.beliefs.find(label)
        # Use the actually observed highest point of the target, never OBB
        # extents (those are eigenvalue-ordered, not axis-aligned).
        top = belief.top_z if belief.top_z is not None else float(belief.position[2])
        drop = top + float(self.cfg.grasp.get("release_clearance_m", 0.06))
        return self.skill_place_at(
            float(belief.position[0]), float(belief.position[1]), drop
        )

    def skill_push_object(self, label: str, direction: str, distance_m: float = 0.08) -> dict:
        """ASPIRE push primitive: pre-contact approach then straight-line push."""
        dirs = {
            "forward": np.array([1.0, 0.0]), "back": np.array([-1.0, 0.0]),
            "left": np.array([0.0, 1.0]), "right": np.array([0.0, -1.0]),
        }
        if direction not in dirs:
            raise SkillError(f"direction must be one of {sorted(dirs)}")
        d2 = dirs[direction]
        frame = self.observe()
        fix = localize_object(frame, label, self.detector, self.extrinsics)
        gcfg = self.cfg.grasp
        table_z = float(self.cfg.safety.get("table_z", 0.0))
        # Contact LOW on the object (a third of its height, min 1.5 cm above
        # the table) so the push doesn't topple or skim over it.
        top_z = float(fix.points[:, 2].max())
        obj_h = max(top_z - table_z, 0.02)
        push_z = table_z + max(0.015, min(0.35 * obj_h, 0.06))
        # Pre-contact standoff must clear the object's own footprint.
        half_len = float(fix.extent[0]) / 2 if fix.extent is not None else 0.05
        standoff = 0.06 + half_len
        d3 = np.array([d2[0], d2[1], 0.0])
        start = fix.position - d3 * standoff
        end = fix.position + d3 * distance_m
        start[2] = end[2] = push_z
        from ..grasping.obb_grasp import _yaw_rotation

        R = _yaw_rotation(float(np.arctan2(d2[1], d2[0])))
        q_now = self.arm.get_state().q
        hover = start + np.array([0.0, 0.0, 0.10])
        ik_hover = self.kin.ik(make_transform(R, hover), q_now)
        ik_start = self.kin.ik(make_transform(R, start), ik_hover.q if ik_hover.success else q_now)
        ik_end = self.kin.ik(make_transform(R, end), ik_start.q if ik_start.success else q_now)
        if not (ik_hover.success and ik_start.success and ik_end.success):
            raise SkillError("push poses unreachable")
        self.arm.set_gripper(self._grip_closed, effort=0.8)  # push with closed jaws
        self.arm.move_joints(ik_hover.q, duration_s=2.0)
        center = (start[:2] + end[:2]) / 2
        radius = float(np.linalg.norm(end[:2] - start[:2]) / 2 + 0.10)
        self.arm.harness.allow_grasp_descent(center, radius_m=radius, z_min=push_z - 0.02)
        try:
            self.arm.move_joints(ik_start.q, duration_s=1.5)
            self.arm.move_joints(ik_end.q, duration_s=2.0)
            self.arm.move_joints(ik_hover.q, duration_s=1.5)
        finally:
            self.arm.harness.clear_grasp_exemption()
        self.beliefs.update(label, end, 0.6)
        self.memory.add("action", f"pushed {label!r} {direction} by {distance_m:.2f} m")
        return {"pushed": label, "direction": direction, "new_position_estimate": end.round(3).tolist()}

    def skill_open_gripper(self) -> dict:
        self.arm.set_gripper(self._grip_open, effort=0.8)
        if self.held_object:
            self.memory.add("action", f"released {self.held_object!r}")
            self.held_object = None
        return {"gripper": "open"}

    def skill_close_gripper(self) -> dict:
        self._close_two_stage(select_profile("", "rigid"))
        wf = self._gripper_width_frac()
        return {"gripper": "closed", "open_frac": round(wf, 2) if wf is not None else "unknown"}

    def skill_move_home(self) -> dict:
        home = np.asarray(self.cfg.arm.get("home_q", [0.0, -0.5, -0.9, 0.0, 0.6, 0.0]), dtype=float)
        if not self.arm.move_joints(home, duration_s=3.0):
            raise SkillError("did not settle at home")
        return {"at": "home"}

    def skill_recall_memory(self, query: str = "") -> dict:
        out: dict = {"recent_events": self.memory.digest(max_lines=15)}
        if query:
            b = self.beliefs.find(query)
            if b is not None:
                now = time.monotonic()
                out["object_memory"] = {
                    "label": b.label,
                    "last_position": [round(float(x), 3) for x in b.position],
                    "seen_s_ago": round(now - b.last_seen_t, 1),
                    "state": b.state(now),
                }
        return out

    def skill_task_done(self, success: bool, summary: str) -> dict:
        if isinstance(success, str):  # schema-lax backends send "false"
            success = success.strip().lower() in ("true", "yes", "1")
        return {"ok": True, "task_complete": True, "success": bool(success), "summary": summary}


def _short(args: dict) -> str:
    return ", ".join(f"{k}={v}" for k, v in args.items())


TOOL_SPECS: list[dict] = [
    {
        "name": "get_observation",
        "description": "Capture a fresh camera frame; returns visible objects with 3D positions (base frame, meters), remembered objects, and robot state.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "list_objects",
        "description": "List every object the robot knows about, including remembered (currently not visible) ones with last-known positions and age.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "localize_object",
        "description": "Precisely localize one object by name; returns base-frame position and size. Use a spatial_hint word (left/right/front/back) to disambiguate duplicates.",
        "parameters": {
            "type": "object",
            "properties": {
                "label": {"type": "string"},
                "spatial_hint": {"type": "string", "enum": ["left", "right", "front", "back", "near", "far"]},
            },
            "required": ["label"],
        },
    },
    {
        "name": "grasp_object",
        "description": "Full grasp pipeline on a named object: localize, plan a top-down grasp, approach, material-aware close, lift, verify. Returns failure details if the object was missed.",
        "parameters": {
            "type": "object",
            "properties": {
                "label": {"type": "string"},
                "material": {
                    "type": "string",
                    "enum": ["rigid", "fragile", "soft", "deformable", "slippery", "heavy"],
                    "description": "How gently to grip; infer from the object's looks.",
                },
                "spatial_hint": {"type": "string", "enum": ["left", "right", "front", "back", "near", "far"]},
            },
            "required": ["label"],
        },
    },
    {
        "name": "place_at",
        "description": "Place the held object at base-frame coordinates (meters). Omit z to release just above the table.",
        "parameters": {
            "type": "object",
            "properties": {
                "x": {"type": "number"},
                "y": {"type": "number"},
                "z": {"type": "number"},
            },
            "required": ["x", "y"],
        },
    },
    {
        "name": "place_on_object",
        "description": "Place the held object on/in another named object (e.g. a bowl or plate).",
        "parameters": {
            "type": "object",
            "properties": {"label": {"type": "string"}},
            "required": ["label"],
        },
    },
    {
        "name": "push_object",
        "description": "Push a named object along the table (for objects too wide to grasp, or to reposition). distance_m defaults to 0.08.",
        "parameters": {
            "type": "object",
            "properties": {
                "label": {"type": "string"},
                "direction": {"type": "string", "enum": ["forward", "back", "left", "right"]},
                "distance_m": {"type": "number"},
            },
            "required": ["label", "direction"],
        },
    },
    {
        "name": "open_gripper",
        "description": "Open the gripper (drops the held object where it is).",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "close_gripper",
        "description": "Close the gripper with the default profile.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "move_home",
        "description": "Return the arm to its home configuration (also clears the camera view).",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "recall_memory",
        "description": "Recall recent events (last ~15 s) and, optionally, where a named object was last seen.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "object name to look up"}},
            "required": [],
        },
    },
    {
        "name": "task_done",
        "description": "Declare the task finished (success or honest failure) with a one-paragraph summary.",
        "parameters": {
            "type": "object",
            "properties": {
                "success": {"type": "boolean"},
                "summary": {"type": "string"},
            },
            "required": ["success", "summary"],
        },
    },
]
