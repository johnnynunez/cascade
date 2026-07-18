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
from ..perception.colors import detection_color, parse_color_query
from ..perception.grounding import (
    Extrinsics,
    localize_object,
    mask_to_points_cam,
    oriented_bbox,
)
from ..types import Detection, Frame, SafetyViolation, SkillError, make_transform, transform_points

#: skills that move the arm: the WorldWatcher is held while they run so the
#: held/handled object is not re-fused at a bogus mid-air position.
_MOTION_SKILLS = {
    "grasp_object", "place_at", "place_on_object", "push_object",
    "open_gripper", "close_gripper", "move_home", "pick_and_place",
    "point_at", "wave", "handover", "sort_by_color", "move_relative",
}


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
        self._held_det_label: str | None = None
        self._held_color: str | None = None
        #: optional WorldWatcher (set by the app wiring); paused during motion
        self.watcher = None
        #: natural-language task currently executing (dashboard narration)
        self.current_task: str | None = None
        #: monotonic time the current top-level MOTION skill started; while
        #: the arm moves the WorldWatcher is paused, so belief ages measured
        #: from "now" are artificially inflated -- staleness checks measure
        #: from this epoch instead.
        self._motion_t0: float | None = None
        #: last time _reobserve completed a fresh scan DURING the current
        #: motion skill: belief-fallback staleness must advance with it, or
        #: a belief that was fresh at task start stays "fresh" through 8
        #: retries even after every re-scan failed to see the object.
        self._last_reobserve_t: float | None = None
        self._graspgenx = None  # lazy GraspGenXPlanner (grasp.backend)
        self._grounder = None  # lazy VLMGrounder (cfg "grounder", 2nd filter)
        g = cfg.arm.gripper
        self._grip_open = float(g.get("open_pos", 0.0))
        self._grip_closed = float(g.get("closed_pos", 1.0))
        self._max_width = float(g.get("max_width_m", 0.09))
        self._default_classes = list(cfg.get("detect_classes", ["cup", "bottle", "box", "fruit", "toy"]))

    # ── plumbing ─────────────────────────────────────────────────────────

    def _show_status(self, text: str) -> None:
        """Push agent state to EVERY live view (all rig cameras show the
        same agent, not just the primary). self.camera goes first: it may
        be a FrameHub WRAPPING the primary stream."""
        streams = [self.camera]
        rig = getattr(self, "rig", None)
        if rig is not None:
            streams += [s for s in getattr(rig, "streams", []) if s is not self.camera]
        for s in streams:
            if hasattr(s, "set_overlay"):
                s.set_overlay(status=text)

    def _show_detections(self, dets) -> None:
        if hasattr(self.camera, "set_overlay"):
            self.camera.set_overlay(detections=dets)

    def execute(self, name: str, args: dict) -> dict:
        """Dispatch one skill call with tracing. Never raises."""
        fn = getattr(self, f"skill_{name}", None)
        if fn is None:
            return {"ok": False, "error": f"unknown skill {name!r}"}
        self._show_status(f"{name}({_short(args)})")
        before = self.trace.save_keyframe(
            self.last_frame.rgb if self.last_frame is not None else None, f"{name}_before"
        )
        t0 = time.monotonic()
        try:
            import contextlib

            hold = (
                self.watcher.paused()
                if (self.watcher is not None and name in _MOTION_SKILLS)
                else contextlib.nullcontext()
            )
            owns_epoch = name in _MOTION_SKILLS and self._motion_t0 is None
            if owns_epoch:
                self._motion_t0 = time.monotonic()
            with hold:
                try:
                    result = fn(**args)
                finally:
                    if owns_epoch:
                        self._motion_t0 = None
                        self._last_reobserve_t = None
                    # Sync the held-object ignore BEFORE fusion resumes, or
                    # one tick could register the object dangling mid-air.
                    if self.watcher is not None:
                        self.watcher.ignore_label(
                            self._held_det_label if self.held_object else None
                        )
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
        err = str(result.get("error", "failed"))
        self._show_status(f"{name} -> " + ("ok" if result["ok"] else err[:60]))
        self.memory.add(
            "action" if result["ok"] else "outcome",
            f"{name}({_short(args)}) -> " + ("ok" if result["ok"] else err[:120]),
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

    def _update_beliefs_from_frame(self, frame: Frame, dets, T=None) -> list[dict]:
        summaries = []
        if not frame.has_depth:
            return [
                {"label": d.label, "conf": round(d.conf, 2), "position": None}
                for d in dets
            ]
        if T is None:
            T = (frame.T_base_cam if frame.T_base_cam is not None
                 else self.extrinsics.cam_to_base())
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
                points=pts_base if d.mask is not None else None,
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

    def _reconcile_held(self) -> None:
        """Drop a stale held-state: if we believe we hold something but the
        jaws report (near) fully closed, the object slipped out (or a crash
        left the flag latched). Without this, one dropped object bricks
        every subsequent pick with 'already holding'."""
        if not self.held_object:
            return
        wf = self._gripper_width_frac()
        if wf is None:  # unknown feedback: keep the cautious assumption
            return
        if wf < float(self.cfg.grasp.get("air_grasp_frac", 0.04)):
            self.memory.add(
                "outcome",
                f"I no longer feel {self.held_object!r} in the gripper "
                "(it must have slipped); clearing the held state",
            )
            self.held_object = None
            self._held_det_label = None
            self._held_color = None

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

    # ── language -> world resolution ─────────────────────────────────────

    def _resolve_query(self, query: str) -> dict:
        """Turn a user phrase ("pink object", "red mug", "bottle") into
        detector inputs, using the live belief store when it already knows
        the answer (the WorldWatcher keeps it warm)."""
        color, noun = parse_color_query(query)
        belief = self.beliefs.find(query)
        near = belief.position.copy() if belief is not None else None
        prompts = vocab = prefer = None
        if noun:
            q = query.strip().lower()
            prompts = [noun] if noun == q else [q, noun]
        elif belief is not None and (color is None or belief.color == color):
            # Warm path: the world model already knows the answer. When the
            # class is in the default vocabulary, detect with the WHOLE
            # vocabulary (no set_classes churn -- YOLOE re-embeds text on
            # every class change) and prefer that label among candidates.
            if belief.label in self._default_classes:
                vocab, prefer = self._default_classes, belief.label
            else:
                prompts = [belief.label]
        else:
            # No belief, or an UNTAGGED belief matched a color query only by
            # fallback -- don't trust it blindly: scan the vocabulary and let
            # the color filter decide (near still helps ranking).
            vocab = self._default_classes
        return {
            "prompts": prompts, "vocab": vocab, "color": color,
            "near_xyz": near, "belief": belief, "prefer_label": prefer,
        }

    def _plan_grasps(self, fix) -> list:
        """Grasp candidates: learned 6-DoF (GraspGen-X server) when
        configured, ALWAYS backstopped by the analytic OBB planner --
        a dead grasp server must degrade, never fail the grasp."""
        gcfg = self.cfg.grasp
        obb = plan_grasps_from_fix(
            fix,
            table_z=float(self.cfg.safety.get("table_z", 0.0)),
            max_width_m=self._max_width,
            depth_fraction=float(gcfg.get("depth_fraction", 0.5)),
        )
        if str(gcfg.get("backend", "obb")) != "graspgenx":
            return obb
        try:
            if self._graspgenx is None:
                from ..grasping.graspgenx_backend import GraspGenXPlanner

                self._graspgenx = GraspGenXPlanner(gcfg)
            learned = self._graspgenx.plan(fix, max_width_m=self._max_width)
            self.memory.add(
                "note",
                f"graspgenx: {len(learned)} grasps in {self._graspgenx.last_latency_s}s "
                f"(top {learned[0].quality:.2f})",
            )
            return learned + obb  # learned first; OBB stays as IK fallback
        except Exception as e:
            self.memory.add("note", f"graspgenx unavailable ({str(e)[:90]}); OBB fallback")
            return obb

    def _localize(self, query: str, spatial_hint: str | None = None):
        """Fresh frame + color/proximity-aware 3D fix for a user phrase.

        A single detector pass at low confidence flickers frame to frame, so
        a miss is retried on a couple of FRESH frames before giving up on
        detection. After that it falls back to a fresh belief: the
        WorldWatcher fuses at ~3 Hz with temporal stability, and an object
        the world model has seen recently is still perfectly actionable
        (that is the whole point of keeping the model warm). Staleness is
        measured against the moment the current motion skill STARTED, not
        "now" -- the watcher is paused while the arm moves, so beliefs age
        artificially during exactly the retries that need them."""
        r = self._resolve_query(query)
        frame = None
        last_err: SkillError | None = None
        tries = int(self.cfg.get("perception_loop", {}).get("localize_frames", 3))
        for _ in range(max(tries, 1)):
            frame = self.observe()
            try:
                fix = localize_object(
                    frame, query, self.detector, self.extrinsics,
                    prompts=r["prompts"], spatial_hint=spatial_hint,
                    color=r["color"], near_xyz=r["near_xyz"], vocab=r["vocab"],
                    prefer_label=r["prefer_label"],
                )
                return frame, fix
            except SkillError as e:
                last_err = e
        # Second filter: THE OTHER CAMERAS. An object 40 px small (or
        # occluded) in the primary view may be plainly visible from the side
        # or wrist camera; each pass costs one detector call.
        from ..perception.grounding import Extrinsics as _Ext

        for cam in self._other_cams():
            try:
                cframe = cam.depth.ensure_depth(cam.stream.get_frame())
                if not cframe.has_depth:
                    continue
                ext = (_Ext(T=cframe.T_base_cam)
                       if cframe.T_base_cam is not None else cam.extrinsics)
                fix = localize_object(
                    cframe, query, self.detector, ext,
                    prompts=r["prompts"], spatial_hint=spatial_hint,
                    color=r["color"], near_xyz=r["near_xyz"], vocab=r["vocab"],
                    prefer_label=r["prefer_label"],
                )
                self.memory.add(
                    "note",
                    f"localize {query!r}: primary camera missed it; found "
                    f"through {getattr(cam.stream, 'name', 'another camera')}",
                )
                return cframe, fix
            except Exception:
                continue  # a miss or camera hiccup: try the next view

        belief = r["belief"] or self.beliefs.find(query)
        max_age = float(self.cfg.get("perception_loop", {}).get(
            "belief_fallback_age_s", 3.0))
        # Staleness reference: the last fresh re-scan during this motion (a
        # re-scan that FAILED to see the object must age the belief), else
        # the motion start (the watcher is paused while the arm moves), else
        # now.
        if self._last_reobserve_t is not None:
            ref_t = self._last_reobserve_t
        elif self._motion_t0 is not None:
            ref_t = self._motion_t0
        else:
            ref_t = time.monotonic()
        mem_ok = (
            belief is not None
            and (ref_t - belief.last_seen_t) <= max_age
            # Grasping from memory demands an EXACT color match: the belief
            # store's neighbor tolerance (pink~red) is for conversation, not
            # for choosing what the jaws close on.
            and (r["color"] is None or belief.color == r["color"])
        )
        if mem_ok:
            self.memory.add(
                "note",
                f"localize {query!r}: instant detection missed; using the "
                f"world-model fix ({belief.label}, "
                f"{time.monotonic() - belief.last_seen_t:.1f}s old)",
            )
            return frame, self._fix_from_belief(query, belief)
        # Last filter: VLM grounding. YOLOE's text embeddings miss what a
        # full VLM reads easily; one slow call only ever runs on this
        # failure path.
        fix = self._vlm_ground_fix(frame, query)
        if fix is not None:
            return frame, fix
        raise last_err

    def _vlm_ground_fix(self, frame, query: str):
        """Ask the VLM (2nd perception filter) for the object's bbox and
        lift it to a 3D fix -- trying EVERY camera's view (the primary may
        see the object at 40 px while the side camera fills the frame with
        it). Fail-soft: any error returns None so the caller reports the
        original detector failure."""
        gcfg = self.cfg.get("grounder", None)
        if not gcfg:
            return None
        try:
            if self._grounder is None:
                from ..perception.vlm_ground import VLMGrounder

                self._grounder = VLMGrounder(
                    base_url=str(gcfg["base_url"]),
                    model=str(gcfg.get("model", "")),
                    timeout_s=float(gcfg.get("timeout_s", 45.0)),
                )
            # Views to try: (frame, cam->base T). Primary first, then the
            # other fusing cameras with their own extrinsics.
            views = []
            if frame is not None and frame.has_depth:
                views.append((frame, frame.T_base_cam if frame.T_base_cam
                              is not None else self.extrinsics.cam_to_base()))
            for cam in self._other_cams():
                try:
                    cframe = cam.depth.ensure_depth(cam.stream.get_frame())
                except Exception:
                    continue
                if not cframe.has_depth:
                    continue
                views.append((cframe, cframe.T_base_cam if cframe.T_base_cam
                              is not None else cam.extrinsics.cam_to_base()))
            det = gframe = T = None
            for cframe, cT in views:
                det = self._grounder.ground(cframe, query)
                if det is not None:
                    gframe, T = cframe, cT
                    break
            if det is None:
                return None
            h, w = gframe.rgb.shape[:2]
            mask = np.zeros((h, w), dtype=bool)
            x0, y0, x1, y1 = det.bbox.astype(int)
            mask[max(y0, 0):min(y1, h), max(x0, 0):min(x1, w)] = True
            pts_cam = mask_to_points_cam(gframe, mask)
            if pts_cam.shape[0] < 30:
                return None
            pts = transform_points(T, pts_cam)
            # A bbox rectangle sweeps in table pixels; shave the table plane
            # or the OBB extent balloons (the GGX bbox-slab lesson).
            table_z = float(self.cfg.safety.get("table_z", 0.0))
            above = pts[pts[:, 2] > table_z + 0.005]
            if above.shape[0] >= 30:
                pts = above
            center, extents, axes = oriented_bbox(pts)
            from ..types import ObjectFix

            self.memory.add(
                "note",
                f"YOLOE missed {query!r}; the VLM (2nd filter) found it at "
                f"{[round(float(v), 2) for v in center]}",
            )
            return ObjectFix(
                label=query, position=center, points=pts,
                detection=det, extent=extents, axes=axes,
            )
        except Exception as e:
            self.memory.add("note", f"VLM grounding unavailable: {str(e)[:80]}")
            return None

    def _other_cams(self) -> list:
        """Non-primary fusing cameras from the watcher wiring: each entry
        has .stream, .depth and .extrinsics (an object may be visible from
        the side camera while the primary sees it at 5 px)."""
        cams = getattr(self.watcher, "_cams", None) if self.watcher else None
        if not cams:
            return []
        return [c for c in cams[1:] if getattr(c, "fuse", True)]

    def _reobserve(self, frames: int = 2) -> None:
        """Refresh beliefs with fresh detector passes over EVERY fusing
        camera while the WorldWatcher is paused (motion skills hold it):
        fusion by hand, exactly what the 3 Hz loop would do. The held
        object (if any) is excluded -- fusing it dangling mid-air would
        corrupt its belief."""
        held = self._held_det_label if self.held_object else None
        for _ in range(max(int(frames), 1)):
            try:
                frame = self.observe()
                dets = self.detector.detect(frame, classes=self._default_classes)
                if held is not None:
                    dets = [d for d in dets if d.label != held]
                self._show_detections(dets)
                self._update_beliefs_from_frame(frame, dets)
                self._last_reobserve_t = time.monotonic()
            except Exception:
                return  # best effort: a camera hiccup must not kill the retry
        for cam in self._other_cams():
            try:
                frame = cam.depth.ensure_depth(cam.stream.get_frame())
                if not frame.has_depth:
                    continue
                dets = self.detector.detect(frame, classes=self._default_classes)
                if held is not None:
                    dets = [d for d in dets if d.label != held]
                T = (frame.T_base_cam if frame.T_base_cam is not None
                     else cam.extrinsics.cam_to_base())
                self._update_beliefs_from_frame(frame, dets, T=T)
            except Exception:
                continue

    @staticmethod
    def _fix_from_belief(query: str, belief) -> "ObjectFix":
        """Synthesize an ObjectFix from a belief.

        Preferred: the belief's remembered REAL point cloud (true shape ->
        true jaw width; a banana reads 35mm, not the 78mm of its bbox slab),
        re-centered on the fused position. Fallback: an axis-aligned box at
        the remembered extent (dense enough for both the OBB planner and the
        GraspGen-X backend)."""
        from ..types import ObjectFix

        center = np.asarray(belief.position, dtype=float).reshape(3)
        pts_mem = getattr(belief, "points", None)
        if pts_mem is not None and len(pts_mem) >= 50:
            pts = np.asarray(pts_mem, dtype=float)
            obb_center, extents, axes = oriented_bbox(pts)
            pts = pts + (center - obb_center)
            det = Detection(
                label=belief.label, conf=float(belief.conf),
                bbox=np.zeros(4, dtype=np.float32),
            )
            return ObjectFix(
                label=query, position=center.copy(), points=pts,
                detection=det, extent=extents, axes=axes,
            )
        ext = (np.sort(np.abs(np.asarray(belief.extent, dtype=float)))[::-1]
               if belief.extent is not None else np.array([0.05, 0.05, 0.05]))
        half = np.clip(ext[:3] / 2.0, 0.01, 0.2)
        top_z = float(belief.top_z) if belief.top_z is not None else float(center[2] + half[2])
        bottom_z = top_z - 2 * half[2]
        rng = np.random.default_rng(0)
        pts = []
        for ax in range(3):  # sample the 6 box faces
            for sign in (-1.0, 1.0):
                p = (rng.random((60, 3)) - 0.5) * 2 * half
                p[:, ax] = sign * half[ax]
                pts.append(p)
        pts = np.concatenate(pts)
        pts[:, 2] = np.clip(pts[:, 2] + (top_z + bottom_z) / 2, bottom_z, top_z)
        pts[:, :2] += center[:2]
        det = Detection(
            label=belief.label, conf=float(belief.conf),
            bbox=np.zeros(4, dtype=np.float32),
        )
        return ObjectFix(
            label=query,
            position=np.array([center[0], center[1], (top_z + bottom_z) / 2]),
            points=pts,
            detection=det,
            extent=np.array([2 * half[0], 2 * half[1], 2 * half[2]]),
            axes=np.eye(3),
        )

    # ── skills ───────────────────────────────────────────────────────────

    def skill_get_observation(self) -> dict:
        frame = self.observe()
        dets = self.detector.detect(frame, classes=self._default_classes)
        self._show_detections(dets)
        objects = self._update_beliefs_from_frame(frame, dets)
        self.memory.add(
            "observation",
            f"saw {[o['label'] for o in objects]} (depth: {frame.depth_source})",
            rgb=frame.rgb,
        )
        # A pure LOOK must not power the motors: a LazyArm that has not
        # materialized yet reports standby instead of being poked awake.
        if getattr(self.arm.raw, "connected", True):
            state = self.arm.get_state()
            tcp = self.kin.fk(state.q)[:3, 3]
            wf = self._gripper_width_frac()
            robot = {
                "q_deg": [round(float(np.degrees(x)), 1) for x in state.q],
                "tcp_xyz": [round(float(x), 3) for x in tcp],
                "gripper_open_frac": round(wf, 2) if wf is not None else "unknown",
                "holding": self.held_object,
            }
        else:
            robot = {"status": "standby (motors unpowered until the first motion)",
                     "holding": self.held_object}
        return {
            "objects_visible": objects,
            "objects_remembered": self.beliefs.summary(),
            "robot": robot,
            "depth_source": frame.depth_source,
        }

    def skill_list_objects(self) -> dict:
        return {
            "objects": self.beliefs.summary(),
            "holding": self.held_object,
            "note": "state=remembered means not currently visible; position is last known",
        }

    def skill_localize_object(self, label: str, spatial_hint: str | None = None) -> dict:
        frame, fix = self._localize(label, spatial_hint=spatial_hint)
        color = detection_color(frame.rgb, fix.detection)
        # Beliefs live under DETECTOR labels (what the watcher re-fuses);
        # recording the user's query words would create ghost duplicates.
        self.beliefs.update(
            fix.detection.label or fix.label, fix.position, fix.detection.conf,
            extent=fix.extent, top_z=float(fix.points[:, 2].max()), t=frame.t,
            color=color,
        )
        return {
            "label": label,
            "detected_as": fix.detection.label,
            "color": color,
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
        self._reconcile_held()
        if self.held_object:
            raise SkillError(f"already holding {self.held_object!r}; place it first")
        gcfg = self.cfg.grasp
        frame, fix = self._localize(label, spatial_hint=spatial_hint)
        profile = select_profile(fix.detection.label or label, material)
        grasps = self._plan_grasps(fix)

        # Learned grasp z can overshoot below the table by a few mm (the
        # tip-offset conversion is empirical): a millimeter under the
        # workspace floor must shallow the grasp slightly, not abort it.
        lim = self.arm.harness.limits
        z_floor = max(
            float(lim.workspace_min[2]) + 0.002,
            lim.table_z + float(gcfg.get("min_grasp_z_m", 0.008)),
        )
        for g in grasps:
            dz = z_floor - float(g.position[2])
            if 0.0 < dz <= 0.03:  # bigger misses are garbage; let vetting drop them
                g.position = np.asarray(g.position, dtype=float).copy()
                g.position[2] = z_floor

        # Pre-vet every candidate against the harness geometry (with the
        # exemption cylinder the descent will open) so a doomed candidate
        # loses the ranking up front instead of aborting mid-motion.
        harness = self.arm.harness
        exempt_r = float(gcfg.get("exempt_radius_m", 0.07))

        def _vet(g, q_pre, q_grasp):
            # The approach leg executes BEFORE allow_grasp_descent opens the
            # cylinder: vet the pregrasp with NO exemption, or a candidate
            # that needed one is guaranteed to abort at the end of the
            # approach. The descent leg holds the exemption, and min-jerk is
            # a straight segment in joint space, so sampling the q_pre ->
            # q_grasp segment vets the actual executed path, not just its
            # endpoints (near-horizontal approaches can dip below the floor
            # OUTSIDE the cylinder mid-descent).
            reason = harness.vet_pose(q_pre)
            if reason:
                return f"pregrasp unsafe: {reason}"
            z_min = float(g.position[2] - 0.02)
            for s in (0.25, 0.5, 0.75, 1.0):
                q = q_pre + s * (np.asarray(q_grasp) - np.asarray(q_pre))
                reason = harness.vet_pose(
                    q, exempt_xy=g.position[:2],
                    exempt_radius_m=exempt_r, exempt_z_min=z_min,
                )
                if reason:
                    return f"descent unsafe: {reason}"
            return None

        state = self.arm.get_state()
        grasp, q_pre, q_grasp = select_grasp(
            grasps,
            self.kin,
            state.q,
            max_width_m=self._max_width,
            pregrasp_offset_m=float(gcfg.get("pregrasp_offset_m", 0.12)),
            validate=_vet,
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
        self._held_det_label = fix.detection.label
        self._held_color = detection_color(frame.rgb, fix.detection)
        self.beliefs.mark_removed(self._held_det_label or label, near=fix.position)
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

    def _adopt_unknown_held(self) -> None:
        """'save it in the box' must work even when the held object was
        never registered (a crashed task, or a human placed something in
        the jaws): a mid-open gripper stall means SOMETHING is in there --
        adopt it as 'object' so place skills can act on it. (Wrist-camera
        visual confirmation is the ROADMAP upgrade.)"""
        if self.held_object:
            return
        wf = self._gripper_width_frac()
        air = float(self.cfg.grasp.get("air_grasp_frac", 0.04))
        if wf is not None and air < wf < 0.9:
            self.held_object = "object"
            self._held_det_label = None
            self.memory.add(
                "note",
                "the jaws are holding something unregistered; "
                "treating it as 'object'",
            )

    def skill_place_at(self, x: float, y: float, z: float | None = None) -> dict:
        self._adopt_unknown_held()
        if not self.held_object:
            raise SkillError("not holding anything")
        gcfg = self.cfg.grasp
        table_z = float(self.cfg.safety.get("table_z", 0.0))
        release_z = float(z) if z is not None else table_z + float(gcfg.get("release_height_m", 0.05))
        # Strict top-down poses only solve below ~0.15 m on this wrist: a
        # tall destination (the bin walls) must become "release from the
        # ceiling and let it drop", not an unreachable-pose failure.
        z_cap = float(gcfg.get("topdown_z_max", 0.15)) - 0.005
        if release_z > z_cap:
            self.memory.add(
                "note",
                f"place height {release_z:.2f} m is above the wrist's "
                f"top-down ceiling; releasing from {z_cap:.2f} m instead",
            )
            release_z = z_cap
        target = np.array([x, y, release_z])
        from ..grasping.obb_grasp import _yaw_rotation

        q_now = self.arm.get_state().q
        hover = target + np.array([0.0, 0.0, float(gcfg.get("pregrasp_offset_m", 0.12))])
        hover[2] = min(hover[2], z_cap)  # same wrist ceiling as the release
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
            # From here the object IS placed: reconcile the held state BEFORE
            # the ascent, or an ascent abort leaves held_object latched and a
            # retry descends onto the object we just released.
            placed = self.held_object
            self.beliefs.update(
                # Re-register under the DETECTOR label so the watcher's next
                # fusion merges here instead of creating a query-string ghost.
                self._held_det_label or placed, target, 0.8, color=self._held_color
            )
            self.held_object = None
            self._held_det_label = None
            self._held_color = None
            self.memory.add("action", f"placed {placed!r} at {target.round(3).tolist()}")
            try:
                self.arm.move_joints(pre.q, duration_s=float(gcfg.get("descend_duration_s", 2.0)))
            except (SkillError, SafetyViolation) as e:
                # The place already happened; report success and let the
                # caller's move_home park the arm.
                self.memory.add("note", f"placed, but the ascent aborted: {e}")
        finally:
            self.arm.harness.clear_grasp_exemption()
        return {"placed": placed, "at": [round(float(v), 3) for v in target]}

    def skill_place_on_object(self, label: str) -> dict:
        self._adopt_unknown_held()
        if not self.held_object:
            raise SkillError("not holding anything")
        belief = self.beliefs.find(label)
        if belief is None:
            frame, fix = self._localize(label)
            self.beliefs.update(
                fix.detection.label or label, fix.position, fix.detection.conf,
                extent=fix.extent, top_z=float(fix.points[:, 2].max()),
                color=detection_color(frame.rgb, fix.detection),
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
        frame, fix = self._localize(label)
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

    def skill_pick_and_place(
        self,
        object: str,
        destination: str | None = None,
        material: str | None = None,
    ) -> dict:
        """One-call pick-and-place that PERSISTS: resolve (color queries
        work) -> grasp -> place on the destination object or the configured
        drop zone -> home. Both stages keep retrying with fresh perception
        (re-home, re-scan, re-plan -- GraspGen-X sampling is stochastic, so
        every retry is a genuinely new grasp) until they succeed or the
        persistence budget (grasp.persist_seconds / grasp.max_pick_attempts)
        runs out. No LLM in the loop; this is the reflex the web chat calls
        for "pick and place pink object"."""
        self._reconcile_held()
        already_held = None
        if self.held_object:
            hq, oq = self.held_object.lower(), str(object).lower()
            if hq in oq or oq in hq:
                # The jaws already hold what was asked for (a previous task
                # grasped it and died before placing): skip straight to the
                # place stage instead of refusing.
                already_held = self.held_object
                self.memory.add(
                    "note",
                    f"already holding {already_held!r} -- going straight to the place stage",
                )
            else:
                raise SkillError(
                    f"already holding {self.held_object!r}; place it first"
                )
        t0 = time.monotonic()
        timings: dict[str, float] = {}
        gcfg = self.cfg.grasp
        max_attempts = max(int(gcfg.get("max_pick_attempts", 8)), 1)
        deadline = t0 + float(gcfg.get("persist_seconds", 120.0))

        grasp = (
            {"held": already_held, "grip_verified": None, "grip_profile": None}
            if already_held
            else None
        )
        placed = None
        last_err = "unknown"
        place_err = "unknown"
        attempt = 0
        p_attempt = 0
        tp = t0
        while placed is None:
            # ── grasp stage (skipped when re-entering after a mid-carry slip
            # left something verified in the jaws -- cannot happen today, but
            # the guard keeps the loop honest) ─────────────────────────────
            while grasp is None and attempt < max_attempts:
                attempt += 1
                if attempt == 1:
                    self.memory.add(
                        "note",
                        f"picking up {object!r} (attempt 1/{max_attempts}; "
                        "I will keep trying until it works or the budget runs out)",
                    )
                if attempt > 1:
                    if time.monotonic() > deadline:
                        attempt -= 1  # this attempt never ran
                        break
                    self.memory.add(
                        "note",
                        f"not giving up: attempt {attempt}/{max_attempts} on "
                        f"{object!r} -- re-homing, re-scanning, planning a fresh grasp",
                    )
                    try:  # clear the camera view so the re-scan actually sees
                        self.skill_move_home()
                    except (SkillError, SafetyViolation):
                        pass
                    self._reobserve()
                tg = time.monotonic()
                try:
                    res = self.skill_grasp_object(object, material=material)
                except (SkillError, SafetyViolation) as e:
                    res = {"ok": False, "error": f"{type(e).__name__}: {e}"}
                timings[f"grasp_attempt{attempt}_s"] = round(time.monotonic() - tg, 2)
                if res.get("ok", True) and res.get("held"):
                    grasp = res
                    break
                last_err = str(res.get("error", "grasp failed"))
                self.memory.add("outcome", f"pick attempt {attempt} failed: {last_err[:100]}")
                if self.arm.harness.estopped:
                    last_err += " (e-stop latched; not retrying)"
                    break
                # Fail fast on errors persistence cannot cure: an object the
                # world model has NEVER seen after full re-scans is a typo or
                # simply not on the table -- burning 2 minutes on it reads as
                # a hang at a live booth.
                if (
                    attempt >= 2
                    and "no detections" in last_err
                    and self.beliefs.find(object) is None
                ):
                    last_err += " (never seen after re-scans; giving up early)"
                    break
            if grasp is None:
                # Never leave the arm hanging mid-pose over the table after a
                # failed attempt -- park it (best effort, nothing is held).
                try:
                    self.skill_move_home()
                except (SkillError, SafetyViolation):
                    pass
                return {
                    "ok": False, "stage": "grasp",
                    "error": f"grasp failed after {attempt} attempts "
                             f"({round(time.monotonic() - t0, 1)}s): {last_err}",
                    "suggestion": "check world_state / camera_snapshot; the object may be unreachable or mis-detected",
                }

            # ── place stage ────────────────────────────────────────────────
            tp = time.monotonic()
            while p_attempt < max_attempts:
                p_attempt += 1
                if p_attempt > 1:
                    if time.monotonic() > deadline:
                        p_attempt -= 1
                        break
                    self.memory.add(
                        "note",
                        f"still holding {object!r}: place attempt "
                        f"{p_attempt}/{max_attempts} -- re-locating the target",
                    )
                    try:  # carry it home first: clears the view of the arm
                        self.skill_move_home()  # + held object for the re-scan
                    except (SkillError, SafetyViolation):
                        pass
                    self._reobserve()
                try:
                    if destination:
                        res = self.skill_place_on_object(destination)
                    else:
                        dz = gcfg.get("drop_zone", [0.30, -0.20])
                        res = self.skill_place_at(float(dz[0]), float(dz[1]))
                    if not res.get("ok", True):
                        raise SkillError(str(res.get("error", "place failed")))
                    placed = res
                    break
                except (SkillError, SafetyViolation) as e:
                    place_err = f"{type(e).__name__}: {e}"
                    self.memory.add(
                        "outcome", f"place attempt {p_attempt} failed: {place_err[:100]}"
                    )
                    if self.arm.harness.estopped:
                        place_err += " (e-stop latched; not retrying)"
                        break
                    if not self.held_object:
                        break  # slipped mid-carry: placing again is pointless
            if placed is None:
                can_regrasp = (
                    not self.held_object
                    and not self.arm.harness.estopped
                    and time.monotonic() < deadline
                    and attempt < max_attempts
                )
                if not can_regrasp:
                    return {
                        "ok": False, "stage": "place",
                        "error": f"place failed after {p_attempt} attempts: {place_err}",
                        "note": (
                            f"still holding {object!r}; try place_at with explicit coordinates"
                            if self.held_object
                            else f"{object!r} slipped while carrying; grasp budget exhausted"
                        ),
                        "grip_verified": grasp.get("grip_verified"),
                    }
                self.memory.add(
                    "note",
                    f"{object!r} slipped while I was carrying it -- starting over",
                )
                grasp = None  # back to the grasp stage within the same budget
        timings["place_s"] = round(time.monotonic() - tp, 2)

        if bool(self.cfg.grasp.get("home_after_place", True)):
            try:  # clear the camera view for the next command; best effort
                self.skill_move_home()
            except (SkillError, SafetyViolation):
                pass
        total = round(time.monotonic() - t0, 2)
        self.memory.add(
            "action",
            f"pick_and_place {object!r} -> {destination or 'drop zone'} in {total}s",
        )
        return {
            "picked": object,
            "placed_at": placed.get("at"),
            "destination": destination or "drop zone",
            "grip_profile": grasp.get("grip_profile"),
            "grip_verified": grasp.get("grip_verified"),
            "grasp_attempts": attempt,
            "place_attempts": p_attempt,
            "duration_s": total,
            "timings": timings,
        }

    # ── social / audience skills (deterministic, harness-gated) ─────────

    def skill_describe_scene(self) -> dict:
        """INSTANT text description from the live world model (no detector
        pass, no motion): what is where, colors, what the gripper holds."""
        objs = self.beliefs.summary()
        parts = []
        for o in objs:
            color = f"{o['color']} " if o.get("color") else ""
            parts.append(
                f"a {color}{o['label']} at [{o['position'][0]}, {o['position'][1]}]"
                + (" (remembered)" if o["state"] == "remembered" else "")
            )
        text = (
            "I see " + "; ".join(parts) + "." if parts
            else "I don't know of any objects yet -- let me look around."
        )
        if self.held_object:
            text += f" I am holding the {self.held_object}."
        return {"description": text, "objects": objs, "holding": self.held_object}

    def skill_count_objects(self, query: str | None = None) -> dict:
        beliefs = self.beliefs.all()
        if query:
            color, noun = parse_color_query(query)
            if color:
                beliefs = [b for b in beliefs if b.color == color]
            if noun:
                beliefs = [
                    b for b in beliefs
                    if noun in b.label.lower() or b.label.lower() in noun
                ]
        return {
            "count": len(beliefs),
            "query": query or "all",
            "labels": [f"{b.color or '?'} {b.label}" for b in beliefs],
        }

    def skill_point_at(self, label: str) -> dict:
        """Deictic gesture: hover the closed gripper above the object for a
        moment ("this one!"), then return. Answers 'which one is X?'."""
        _, fix = self._localize(label)
        gcfg = self.cfg.grasp
        hover = fix.position.copy()
        # Strict top-down poses only solve below ~0.15 m on the B601-RS
        # (wrist limits); clamp the hover or tall objects become unpointable.
        z_max = float(gcfg.get("topdown_z_max", 0.15)) - 0.01
        hover[2] = min(
            float(fix.points[:, 2].max()) + float(gcfg.get("pregrasp_offset_m", 0.08)),
            z_max,
        )
        from ..grasping.obb_grasp import _yaw_rotation

        q_now = self.arm.get_state().q
        ik = None
        for z in (hover[2], z_max - 0.02):
            for yaw in (float(np.arctan2(hover[1], hover[0])), 0.0, np.pi / 4, -np.pi / 4):
                cand = self.kin.ik(
                    make_transform(_yaw_rotation(yaw), [hover[0], hover[1], z]), q_now
                )
                if cand.success:
                    ik = cand
                    break
            if ik is not None:
                break
        if ik is None:
            raise SkillError(f"cannot reach a pointing pose above {label!r}")
        if not self.held_object:
            self.arm.set_gripper(self._grip_closed, effort=0.6)
        if not self.arm.move_joints(ik.q, duration_s=2.0):
            raise SkillError("did not settle at the pointing pose")
        time.sleep(float(self.cfg.get("gesture", {}).get("point_hold_s", 1.2)))
        self.memory.add("action", f"pointed at {label!r}")
        return {"pointed_at": label, "position": [round(float(x), 3) for x in fix.position]}

    def skill_wave(self, cycles: int = 2) -> dict:
        """Greeting gesture: wag the base + wrist around home. Every
        waypoint still goes through the safety harness."""
        home = np.asarray(
            self.cfg.arm.get("home_q", [0.0, 1.2, 1.2, 0.0, 0.75, 0.0]), dtype=float
        )
        if not self.arm.move_joints(home, duration_s=2.0):
            raise SkillError("could not reach home to wave")
        cycles = int(np.clip(cycles, 1, 4))
        for side in [+1, -1] * cycles:
            q = home.copy()
            q[0] += 0.25 * side
            q[4] += 0.30 * side
            self.arm.move_joints(q, duration_s=0.7)
        self.arm.move_joints(home, duration_s=0.8)
        self.memory.add("action", "waved at the audience")
        return {"waved": True, "cycles": cycles}

    def skill_handover(self, label: str | None = None) -> dict:
        """Hand the object to the human: grasp it if needed, present it at
        the handover pose, and KEEP HOLDING -- the human says 'open gripper'
        (or the agent calls open_gripper) once they have grabbed it."""
        if not self.held_object:
            if not label:
                raise SkillError("not holding anything; say which object to hand over")
            res = self.skill_grasp_object(label)
            if not res.get("ok", True) or not res.get("held"):
                return {
                    "ok": False,
                    "error": f"could not grasp {label!r} for handover: {res.get('error')}",
                }
        hand_q = np.asarray(
            self.cfg.arm.get("handover_q", [0.5, 1.2, 1.2, 0.0, 0.75, 0.0]), dtype=float
        )
        if not self.arm.move_joints(hand_q, duration_s=2.0):
            raise SkillError("did not settle at the handover pose")
        self.memory.add("action", f"offering {self.held_object!r} to the human")
        return {
            "offering": self.held_object,
            "note": "holding steady; call open_gripper once the human has it",
        }

    def skill_sort_by_color(self, max_objects: int = 6) -> dict:
        """Crowd-pleaser: group everything on the table into per-color zones
        along the front edge. Pure composition of pick_and_place."""
        # +-0.24 is outside the top-down IK envelope at x=0.30 on this arm
        # (verified against the RS URDF); stay within +-0.20.
        zones_y = (-0.20, -0.10, 0.0, 0.10, 0.20)
        zone_x = float(self.cfg.grasp.get("drop_zone", [0.30, -0.20])[0])
        colors: dict[str, tuple[float, float]] = {}
        moved, failed = [], []
        for b in list(self.beliefs.all())[: int(max_objects)]:
            color = b.color or "unknown"
            if color not in colors:
                if len(colors) >= len(zones_y):
                    failed.append({"label": b.label, "error": "no free color zone"})
                    continue
                colors[color] = (zone_x, zones_y[len(colors)])
            zx, zy = colors[color]
            if np.linalg.norm(b.position[:2] - np.array([zx, zy])) < 0.07:
                continue  # already sorted
            query = f"{b.color} {b.label}" if b.color else b.label
            try:
                res = self.skill_grasp_object(query)
            except (SkillError, SafetyViolation) as e:
                failed.append({"label": query, "error": str(e)})
                continue
            if not res.get("ok", True) or not res.get("held"):
                failed.append({"label": query, "error": str(res.get("error", "?"))})
                continue
            try:
                self.skill_place_at(zx, zy)
            except (SkillError, SafetyViolation) as e:
                # still holding: stop sorting rather than cascade failures
                failed.append({"label": query, "error": f"place: {e}"})
                break
            moved.append({"label": query, "color": color, "zone": [zx, zy]})
        ok = not failed or bool(moved)
        out = {
            "ok": ok,
            "moved": moved,
            "failed": failed,
            "zones": {c: list(z) for c, z in colors.items()},
        }
        if not ok:
            out["error"] = "; ".join(
                f"{f['label']}: {f['error'][:60]}" for f in failed[:3]
            ) or "nothing to sort"
        return out

    def skill_move_relative(self, direction: str, distance_m: float = 0.05) -> dict:
        """Nudge the TCP: fine chat-driven control ("a bit to the left")."""
        dirs = {
            "forward": [1, 0, 0], "back": [-1, 0, 0],
            "left": [0, 1, 0], "right": [0, -1, 0],
            "up": [0, 0, 1], "down": [0, 0, -1],
        }
        if direction not in dirs:
            raise SkillError(f"direction must be one of {sorted(dirs)}")
        distance_m = float(np.clip(distance_m, 0.01, 0.15))
        q_now = self.arm.get_state().q
        T = self.kin.fk(q_now)
        target = T.copy()
        target[:3, 3] += np.asarray(dirs[direction], dtype=float) * distance_m
        ik = self.kin.ik(target, q_now)
        if not ik.success:
            raise SkillError(f"cannot move {distance_m:.2f} m {direction} from here")
        if not self.arm.move_joints(ik.q, duration_s=1.0):
            raise SkillError("did not settle after the nudge")
        tcp = self.kin.fk(self.arm.get_state().q)[:3, 3]
        return {"moved": direction, "distance_m": distance_m,
                "tcp_xyz": [round(float(x), 3) for x in tcp]}

    def skill_open_gripper(self) -> dict:
        self.arm.set_gripper(self._grip_open, effort=0.8)
        if self.held_object:
            self.memory.add("action", f"released {self.held_object!r}")
            self.held_object = None
            self._held_det_label = None
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
        "name": "pick_and_place",
        "description": (
            "FAST PATH: complete pick-and-place in ONE call. Resolves the "
            "object against the live world model (color queries like 'pink "
            "object' work), grasps, places on the named destination object "
            "(or the default drop zone if omitted), returns home, and "
            "reports stage timings. Both stages PERSIST: they keep retrying "
            "with fresh perception and fresh grasp plans until they succeed "
            "or the persistence budget runs out. Prefer this over manual "
            "localize/grasp/place for any 'pick X [put it in Y]' request."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "object": {"type": "string", "description": "what to pick, e.g. 'pink object', 'red mug'"},
                "destination": {"type": "string", "description": "named object to place on/in; omit for the drop zone"},
                "material": {
                    "type": "string",
                    "enum": ["rigid", "fragile", "soft", "deformable", "slippery", "heavy"],
                },
            },
            "required": ["object"],
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
        "name": "describe_scene",
        "description": "INSTANT text description of everything the robot knows (objects, colors, positions, what it holds) from the live world model -- no motion, no camera wait. Prefer this for 'what do you see?' questions.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "count_objects",
        "description": "Count known objects, optionally filtered by a query like 'red' or 'cube' or 'pink object'. Instant, from the live world model.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": [],
        },
    },
    {
        "name": "point_at",
        "description": "Point at a named object (hover the gripper above it for a moment). Great for answering 'which one is the pink object?'.",
        "parameters": {
            "type": "object",
            "properties": {"label": {"type": "string"}},
            "required": ["label"],
        },
    },
    {
        "name": "wave",
        "description": "Wave at the audience (a friendly greeting gesture around the home pose).",
        "parameters": {
            "type": "object",
            "properties": {"cycles": {"type": "integer", "minimum": 1, "maximum": 4}},
            "required": [],
        },
    },
    {
        "name": "handover",
        "description": "Hand an object to the human: grasp it (if not already held), present it at the handover pose, and keep holding until open_gripper is called. Use for 'hand me / give me the X'.",
        "parameters": {
            "type": "object",
            "properties": {"label": {"type": "string"}},
            "required": [],
        },
    },
    {
        "name": "sort_by_color",
        "description": "Sort every known object into per-color zones along the table edge (repeated pick-and-place; takes a while). A crowd favorite.",
        "parameters": {
            "type": "object",
            "properties": {"max_objects": {"type": "integer", "minimum": 1, "maximum": 8}},
            "required": [],
        },
    },
    {
        "name": "move_relative",
        "description": "Nudge the gripper a few centimeters (fine adjustment): forward/back/left/right/up/down, default 0.05 m, max 0.15 m.",
        "parameters": {
            "type": "object",
            "properties": {
                "direction": {"type": "string", "enum": ["forward", "back", "left", "right", "up", "down"]},
                "distance_m": {"type": "number"},
            },
            "required": ["direction"],
        },
    },
    {
        "name": "open_gripper",
        "description": "Open the gripper (drops the held object where it is; also how a handover finishes).",
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
