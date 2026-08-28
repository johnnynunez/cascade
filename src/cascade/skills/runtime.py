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
from ..perception.workspace import WorkspaceFilter
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
    "throw", "grasp_at_pixel",
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
        # (3,) base-frame vector from TCP to the held object, or None. Set at
        # grasp, refreshed by re-observation before a place.
        self._held_offset = None
        #: optional external pose lookup, set by attach_verifier
        self._object_pose = None
        self._held_det_label: str | None = None
        self._held_color: str | None = None
        #: optional WorldWatcher (set by the app wiring); paused during motion
        self.watcher = None
        #: RGB frame captured immediately before the current motion skill, for
        #: CaP-X visual differencing. Set by execute(); None between motions.
        self._pre_motion_frame = None
        #: LiveViewController (set by the app wiring): the on-demand browser
        #: dashboard. None in bare/unit-test runtimes.
        self.live_view = None
        #: the live StreamServer while the view is open, else None
        self.stream_server = None
        #: natural-language task currently executing (dashboard narration)
        self.current_task: str | None = None
        #: dispatch tier that served the last command ("reflex" |
        #: "experience" | "llm" | "mcp-host"), for the dashboard "via:" chip
        self.last_path: str | None = None
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
        # Fake-RL grasp memory (RPent/Harness-VLA pattern): learns which grasp
        # geometry works per object profile from past attempts (wins AND
        # failures), persisted so accuracy improves across sessions.
        from ..memory.grasp_memory import GraspOutcomeMemory
        from pathlib import Path as _Path
        _gm_path = cfg.grasp.get("memory_path", "~/.cascade/grasp_memory.json")
        self.grasp_memory = GraspOutcomeMemory(
            path=_Path(str(_gm_path)).expanduser())
        # Harness-VLA (arXiv:2607.08448) generalised to EVERY primitive: the
        # learned operating envelope + failure model, fed from the same
        # outcomes and injected into the agent's context by the orchestrator.
        from ..memory.envelope import OperatingEnvelope
        _env_path = cfg.get("memory", {}).get(
            "envelope_path", "~/.cascade/envelope.json") if hasattr(cfg, "get") else None
        self.envelope = OperatingEnvelope(
            path=_Path(str(_env_path or "~/.cascade/envelope.json")).expanduser())
        # Pigey (arXiv:2607.21725): verify each primitive's physical effect
        # against a channel the actuator does not own. Wired lazily by the
        # app (needs the sim bridge / belief store) via attach_verifier().
        self.effects = None
        g = cfg.arm.gripper
        self._grip_open = float(g.get("open_pos", 0.0))
        self._grip_closed = float(g.get("closed_pos", 1.0))
        self._max_width = float(g.get("max_width_m", 0.09))
        # Optional vocabulary restriction. Empty (the default) means
        # open-world: the detector reports whatever it sees, so an object
        # nobody listed still reaches the agent. A non-empty list is a CLOSED
        # SET and hides everything else -- benchmarks only, never the booth.
        self._default_classes = list(cfg.get("detect_classes") or []) or None
        self._workspace = WorkspaceFilter.from_config(cfg.get("workspace_filter"))
        # Learned point segmenter for pixel addressing. Off by default: it
        # costs a 74 MB checkpoint and only the pixel path uses it, so a booth
        # that never clicks pays nothing. When absent, `fix_at_pixel` falls
        # back to depth connectivity, which cannot separate touching objects.
        scfg = cfg.get("segmenter") or {}
        if scfg.get("enabled"):
            from ..perception.segmenter import PointSegmenter

            self._segmenter = PointSegmenter(
                model_path=str(scfg.get("model", "sam2.1_t.pt")),
                device=str(scfg.get("device", "auto")),
            )
        else:
            self._segmenter = None

    def attach_verifier(self, object_pose=None) -> None:
        """Enable postcondition checking (Pigey closed loop).

        ``object_pose`` is an optional ground-truth pose lookup -- in sim the
        Isaac bridge can read a RigidPrim directly, which beats perception.
        Without it the checker falls back to the belief store, and to CaP-X
        visual differencing when the belief would only be confirming itself.
        """
        from ..agent.effects import PostconditionChecker

        # Kept on the runtime too: `_held_object_offset` uses it to see where
        # the held object actually sits when the camera cannot (benchmark
        # oracle mode). Same channel, same provenance caveat.
        self._object_pose = object_pose
        self.effects = PostconditionChecker(
            object_pose=object_pose,
            belief_pose=self._belief_pose,
            gripper_frac=self._gripper_width_frac,
            reobserve=lambda: self._reobserve(frames=1),
            visual_diff=self._visual_diff,
            table_z=float(self.cfg.safety.get("table_z", 0.0)),
            air_grasp_frac=float(self.cfg.grasp.get("air_grasp_frac", 0.04)),
        )

    @property
    def _tool_axis_order(self) -> str:
        """Tool-frame convention of the arm currently wired in.

        MEASURED BUG this centralises: the grasp planner was taught the Panda's
        convention while `place_on_object`, `place_at` and the hover search
        kept calling `_yaw_rotation` with the reBot default. Grasping then
        worked (7/10 lifting ~20 cm and carrying up to 47 cm) while every place
        died with "place pose unreachable", because the two paths were asking
        for frames 92.6 degrees apart on the same robot.
        """
        return str(self.cfg.arm.get("tool_axis_order", "down_open"))

    def _profile_q(self, key: str, what: str) -> np.ndarray:
        """A joint-space keyframe (home_q, handover_q, ...) from the arm profile.

        These poses belong to ONE arm on ONE table. A default baked in here is
        worse than no pose at all: substituting the reBot's 6-element home pose
        on a 5-DoF SO-101 both mis-sizes the vector and aims a different chain
        at a table nobody measured, and the harness cannot catch a pose that is
        geometrically legal yet wrong for this robot. So a missing keyframe is
        an honest failure -- every shipped profile declares its own, pinned by
        tests/test_arm_profiles.py.
        """
        q = self.cfg.arm.get(key)
        if q is None:
            raise SkillError(
                f"arm profile declares no {key!r}, which is needed to {what}; "
                f"add it to configs/arms/<profile>.yaml"
            )
        q = np.asarray(q, dtype=float).reshape(-1)
        n = int(self.cfg.arm.get("n_joints", len(q)))
        if len(q) != n:
            raise SkillError(
                f"arm profile {key!r} has {len(q)} joints but n_joints={n}"
            )
        return q

    @property
    def _gesture_joint(self) -> int:
        """Wrist joint that `wave` wags, as an index into q.

        Declared by the profile because chains differ in length: index 4 is the
        wrist on both a 6-DoF reBot and a 5-DoF SO-101, but it is the elbow on a
        3-DoF arm. Defaults to the wrist-most joint that is not the last one on
        long chains, which reproduces the reBot's tuned choice exactly.
        """
        n = int(self.cfg.arm.get("n_joints", 6))
        idx = self.cfg.arm.get("gesture_joint")
        return int(idx) if idx is not None else min(4, max(0, n - 1))

    def _visual_diff(self, source_xyz=None, target_xyz=None):
        """CaP-X: compare the pre-motion frame with a fresh one.

        `_pre_motion_frame` is captured by execute() before any motion skill,
        so this is a genuine before/after pair. Returns None (abstain) when
        there is no pair, rather than pretending to know.
        """
        before = getattr(self, "_pre_motion_frame", None)
        if before is None:
            return None
        frame = self.last_frame or self.observe()
        if frame is None:
            return None
        # A camera that re-renders the same synthetic image every grab cannot
        # witness motion, so "the pixels did not change" is not evidence about
        # the arm -- it is a property of the sensor. Reporting UNCHANGED from it
        # fails every place whose source and target both land in frame (the
        # reBot mock only escaped this because its drop zone projects out of
        # view). Abstaining is the same answer this method already gives when
        # there is no before/after pair at all.
        if bool(self.cfg.camera.get("static_scene", False)):
            return None

        from ..perception.visual_diff import VisualDiffChannel

        def _project(xyz):
            try:
                import numpy as _np

                T = frame.T_base_cam
                if T is None:
                    T = self.extrinsics.cam_to_base()
                p_cam = (_np.linalg.inv(_np.asarray(T, float))
                         @ _np.append(_np.asarray(xyz, float)[:3], 1.0))[:3]
                if p_cam[2] <= 1e-6:
                    return None
                uv = _np.asarray(frame.K, float) @ (p_cam / p_cam[2])
                return (float(uv[0]), float(uv[1]))
            except Exception:
                return None

        return VisualDiffChannel(_project).compare(
            before, frame.rgb, source_xyz, target_xyz
        )

    def _belief_pose(self, label: str):
        """Best-known 3D position of ``label`` from the belief store."""
        if not label:
            return None
        try:
            best, score = None, 0
            want = set(str(label).lower().split())
            for b in self.beliefs.all():
                have = set(str(getattr(b, "label", "")).lower().split())
                overlap = len(want & have)
                if overlap > score:
                    best, score = b, overlap
            return list(best.position[:3]) if best is not None else None
        except Exception:
            return None

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
        # Pigey: snapshot the target's pose BEFORE the motion so the
        # postcondition can measure a displacement rather than guess one.
        # NOTE the arg name varies across the skill API and getting this wrong
        # silently degrades every check to "unverified": pick_and_place takes
        # `object`, grasp/push take `label`, and place_at takes no object at
        # all (the held one is the subject). Verified against TOOL_SPECS --
        # keep this in sync when adding a motion skill.
        # place_on_object is the exception: its `label` is the DESTINATION,
        # so snapshotting it would make the checker compare the target with
        # itself ("box sits on box"). The subject there is the held object.
        pre_state = {}
        if self.effects is not None and name in _MOTION_SKILLS:
            # CaP-X: keep the pre-motion pixels so the postcondition can ask a
            # channel the actuator does not own. Copied because CameraStream
            # reuses its buffer -- holding the reference would silently give
            # us the AFTER frame twice and confirm everything.
            f = self.last_frame
            self._pre_motion_frame = f.rgb.copy() if f is not None else None
            if name == "place_on_object":
                target_label = self.held_object
            else:
                target_label = (
                    args.get("label")
                    or args.get("object")
                    or args.get("query")
                    or self.held_object
                    or None
                )
            pre_state = self.effects.snapshot(target_label)
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
        # Pigey closed loop: was the claimed effect real? A refuted
        # postcondition DOWNGRADES a self-reported success (annotate_result).
        if self.effects is not None and result.get("ok") is not None:
            try:
                from ..agent.effects import annotate_result

                pc = self.effects.verify(name, args, result, before=pre_state)
                result = annotate_result(result, pc)
            except Exception:
                pass
        # Harness-VLA: fold the outcome into the learned operating envelope.
        try:
            self.envelope.record(
                name, args, ok=bool(result.get("ok")),
                error=str(result.get("error", "")), duration_ms=dur,
            )
        except Exception:
            pass
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
            # Same geometric gate the watcher applies. Without it this path
            # registers the arm and the backdrop as objects, because an
            # open vocabulary has names for them.
            if self._workspace.reject(
                center, extents,
                mask_frac=float(mask.sum()) / float(mask.size) if mask.size else None,
            ) is not None:
                continue
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
            # class is in a restricted vocabulary, detect with the WHOLE
            # vocabulary (no set_classes churn -- YOLOE re-embeds text on
            # every class change) and prefer that label among candidates.
            # Open-world (no restriction) needs neither: just ask by name.
            if self._default_classes and belief.label in self._default_classes:
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

    def _plan_grasps(self, fix, label: str | None = None) -> list:
        """Grasp candidates: learned 6-DoF (GraspGen-X server) when
        configured, ALWAYS backstopped by the analytic OBB planner --
        a dead grasp server must degrade, never fail the grasp.

        Fake-RL layer: past-attempt memory re-ranks the candidates so grasp
        geometry that historically WORKED for this object profile goes first
        (RPent strategy-prior pattern), and applies a learned grasp-z nudge.
        """
        gcfg = self.cfg.grasp
        obb = plan_grasps_from_fix(
            fix,
            table_z=float(self.cfg.safety.get("table_z", 0.0)),
            max_width_m=self._max_width,
            depth_fraction=float(gcfg.get("depth_fraction", 0.5)),
            axis_order=self._tool_axis_order,
        )
        if str(gcfg.get("backend", "obb")) != "graspgenx":
            grasps = obb
        else:
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
                grasps = learned + obb  # learned first; OBB stays as IK fallback
            except Exception as e:
                self.memory.add("note", f"graspgenx unavailable ({str(e)[:90]}); OBB fallback")
                grasps = obb

        # ---- fake-RL memory prior: re-rank + z-nudge -----------------------
        lbl = label or getattr(fix, "label", None) or "object"
        try:
            prior = self.grasp_memory.prior(lbl, fix)
            if prior:
                grasps = self.grasp_memory.rerank(grasps, lbl, fix)
                dz = float(prior["nudges"].get("grasp_z_delta", 0.0))
                if abs(dz) > 1e-4:
                    for g in grasps:
                        p = np.asarray(g.position, dtype=float).copy()
                        p[2] = p[2] + dz
                        g.position = p
                self._last_grasp_z_nudge = dz
                self.memory.add(
                    "note",
                    f"grasp-memory prior for {prior['profile']}: "
                    f"{prior['wins']}W/{prior['losses']}L sr={prior['success_rate']:.0%}"
                    + (f", z_nudge={dz:+.3f}" if abs(dz) > 1e-4 else "")
                    + (f", avoid={prior['top_fail']}" if prior.get("top_fail") else ""),
                )
            else:
                self._last_grasp_z_nudge = 0.0
        except Exception as e:
            self._last_grasp_z_nudge = 0.0
            self.memory.add("note", f"grasp-memory prior skipped ({str(e)[:60]})")
        return grasps

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
            # Re-center on the fused position in XY ONLY.
            #
            # The fused position tracks a detection centroid and, for a body
            # whose origin is not its geometric centre, differs from the
            # cloud's bbox centre in z for reasons that have nothing to do
            # with where the object is. MEASURED on LIBERO's akita bowl: the
            # body origin sits 52 mm below the mesh top and 26 mm below the
            # cloud's bbox centre, so shifting z dragged the cloud down until
            # its top read 0.9242 against a true rim at 0.9507. The rim grasp
            # was then planned 29.5 mm BELOW the rim, on the outer wall.
            #
            # Heights come from the cloud, which is measured surface; xy comes
            # from the fused estimate, which is what tracking is good at.
            shift = center - obb_center
            shift[2] = 0.0
            pts = pts + shift
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

    def skill_halt_motion(self, reason: str = "superseded") -> dict:
        """Stop the motion currently in flight, without latching an e-stop.

        VoLo's `monitor - halt - redirect`: a physical agent has to be able to
        abandon an action that has stopped being the right one, because the
        world keeps moving while it thinks. Takes effect within one 50 Hz
        waypoint since `SafetyHarness.approve()` is what enforces it.

        The halt clears itself when the next motion starts, so the normal
        recovery is simply to issue the corrected command. Use `stop` (e-stop)
        instead when the rig is actually unsafe.
        """
        self.arm.harness.halt(reason)
        self.memory.add("note", f"halted motion: {reason}")
        return {"ok": True, "halted": reason,
                "note": "clears automatically when the next motion begins"}

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

    def skill_preview_grasp(
        self,
        label: str,
        spatial_hint: str | None = None,
    ) -> dict:
        """VIA-style waypoint preview (arXiv 2607.11119): plan the grasp and
        REPORT it without moving. Lets the agent observe the proposed gripper
        waypoint (position, approach, confidence, learned-memory prior) and
        decide before committing -- the observe-then-act loop, not blind
        execution. Follow with grasp_object to actually execute it."""
        gcfg = self.cfg.grasp
        frame, fix = self._localize(label, spatial_hint=spatial_hint)
        grasps = self._plan_grasps(fix, label=label)
        if not grasps:
            return {"ok": False, "error": f"no grasp candidates for {label!r}"}
        g = grasps[0]  # already reranked by the fake-RL memory prior
        appr = np.asarray(g.approach, dtype=float)
        vert = float(-appr[2] / (np.linalg.norm(appr) + 1e-9))
        prior = None
        try:
            prior = self.grasp_memory.prior(label, fix)
        except Exception:
            pass
        out = {
            "ok": True,
            "object": label,
            "object_xyz": [round(float(x), 3) for x in fix.position],
            "planned_grasp": {
                "tcp_xyz": [round(float(x), 3) for x in g.position],
                "approach": ("top-down" if vert > 0.7
                             else "angled" if vert > 0.3 else "side"),
                "approach_vert": round(vert, 2),
                "jaw_width_m": round(float(g.width_m), 3),
                "confidence": round(float(getattr(g, "quality", 0.0)), 2),
                "candidates": len(grasps),
            },
            "hint": "call grasp_object to execute, or reposition/re-localize "
                    "if this waypoint looks wrong.",
        }
        if prior:
            out["memory_prior"] = {
                "seen": prior["wins"] + prior["losses"],
                "success_rate": prior["success_rate"],
                "avoid": prior.get("top_fail"),
            }
        self.memory.add("note", f"grasp preview {label!r}: {out['planned_grasp']['approach']} "
                        f"conf {out['planned_grasp']['confidence']}")
        return out

    def skill_grasp_object(
        self,
        label: str,
        material: str | None = None,
        spatial_hint: str | None = None,
        _fix=None,
        _frame=None,
    ) -> dict:
        """Grasp a named object.

        `_fix`/`_frame` are an internal entry point for callers that already
        localized the target by other means (see `skill_grasp_at_pixel`). They
        skip the detector lookup and reuse this method's grasp planning,
        harness checks and postcondition verification verbatim, rather than
        duplicating that logic in a second code path where the two would
        inevitably drift apart.
        """
        self._reconcile_held()
        if self.held_object:
            raise SkillError(f"already holding {self.held_object!r}; place it first")
        gcfg = self.cfg.grasp
        if _fix is not None and _frame is not None:
            frame, fix = _frame, _fix
        else:
            frame, fix = self._localize(label, spatial_hint=spatial_hint)
        profile = select_profile(fix.detection.label or label, material)
        grasps = self._plan_grasps(fix, label=label)

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
            # Exemption floor a bit BELOW the table: the reBot RS gripper's
            # finger/wrist links extend ~5 cm below the TCP, so a top-down
            # grasp of a low object legitimately dips a link to z ~ -0.018
            # (below the table plane) while the fingers straddle the object.
            # Flooring the exemption at table_z left that link 18 mm outside
            # the cylinder -> false "link would hit the table" abort. The
            # narrow XY radius still confines this to directly over the
            # target, so allowing a small sub-table dip there is safe.
            z_min = float(harness.limits.table_z) - 0.06
            # Sample the q_pre -> q_grasp segment densely: min-jerk streaming
            # generates many intermediate waypoints, and a coarse 4-sample vet
            # can miss a mid-segment configuration where the elbow dips below
            # the table (IK multi-solution: the interpolated path can bow down
            # even when both endpoints are safe).
            for s in (0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.0):
                q = q_pre + s * (np.asarray(q_grasp) - np.asarray(q_pre))
                reason = harness.vet_pose(
                    q, exempt_xy=g.position[:2],
                    exempt_radius_m=exempt_r, exempt_z_min=z_min,
                )
                if reason:
                    return f"descent unsafe: {reason}"
            return None

        state = self.arm.get_state()
        # Seed grasp IK from HOME (elbow-up), not the live pose: seeding from
        # an arbitrary current configuration can converge to an elbow-down IK
        # branch whose approach path dips a link under the table.
        _home = self.cfg.arm.get("home_q")
        _seed = np.asarray(_home, dtype=float) if _home is not None else state.q
        try:
            grasp, q_pre, q_grasp = select_grasp(
                grasps,
                self.kin,
                _seed,
                max_width_m=self._max_width,
                pregrasp_offset_m=float(gcfg.get("pregrasp_offset_m", 0.12)),
                validate=_vet,
            )
        except (SkillError, SafetyViolation) as e:
            # No candidate survived IK + harness vetting. Log it against the
            # object profile so the fake-RL memory learns to bias future
            # candidates (raise z, prefer top-down) for this object.
            try:
                self.grasp_memory.record(
                    label, fix, grasps[0] if grasps else None,
                    success=False, reason=str(e),
                    z_nudge_applied=getattr(self, "_last_grasp_z_nudge", 0.0))
            except Exception:
                pass
            raise

        # 1. open, go to pregrasp (normal speed). Re-home first so the
        # pregrasp IK seeds from a known elbow-up posture: seeding from an
        # arbitrary current pose can converge to an elbow/wrist-down IK
        # solution whose approach path dips a link below the table near the
        # base (observed: link7 at xy~(0.05,0.04) z=-0.023, far from the
        # target so no grasp-exemption cylinder can cover it).
        self.arm.set_gripper(self._grip_open, effort=0.8)
        _home = self.cfg.arm.get("home_q")
        if _home is not None:
            try:
                self.arm.move_joints(np.asarray(_home, dtype=float),
                                     duration_s=1.5)
            except Exception:
                pass  # best-effort re-home; pregrasp move is the real gate
        if not self.arm.move_joints(q_pre, duration_s=float(gcfg.get("move_duration_s", 2.5))):
            raise SkillError("did not settle at pregrasp pose")

        # 2. descend inside the exemption cylinder (slow)
        self.arm.harness.allow_grasp_descent(
            grasp.position[:2],
            radius_m=float(gcfg.get("exempt_radius_m", 0.07)),
            z_min=float(self.arm.harness.limits.table_z) - 0.06,
        )
        try:
            if not self.arm.move_joints(q_grasp,
                                        duration_s=float(gcfg.get("descend_duration_s", 2.0)),
                                        bias_compensate=True):
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
                try:
                    self.grasp_memory.record(
                        label, fix, grasp, success=False, reason="air grasp",
                        z_nudge_applied=getattr(self, "_last_grasp_z_nudge", 0.0))
                except Exception:
                    pass
                return {
                    "ok": False,
                    "error": "air grasp: gripper closed fully, object not held",
                    "suggestion": "re-localize the object or try the alternate yaw",
                }
        self.held_object = label
        self._held_det_label = fix.detection.label
        self._held_color = detection_color(frame.rgb, fix.detection)
        # Where the object sat relative to the TCP at the moment of grasp, in
        # the base frame. `place_at` aims the TCP, so without this the object
        # lands wherever the jaws happen to be holding it. MEASURED on LIBERO:
        # 1.1 cm horizontal offset at grasp, 5.1 cm at release (it slides
        # 4.3 cm in transit), which is most of the ~6 cm placement error
        # against a 3 cm success predicate.
        try:
            tcp_at_grasp = self.kin.fk(self.arm.get_state().q)[:3, 3]
            self._held_offset = np.asarray(fix.position, float) - tcp_at_grasp
        except Exception:
            self._held_offset = None
        self.beliefs.mark_removed(self._held_det_label or label, near=fix.position)
        try:
            self.grasp_memory.record(
                label, fix, grasp, success=True,
                z_nudge_applied=getattr(self, "_last_grasp_z_nudge", 0.0))
        except Exception:
            pass
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

    def _held_object_offset(self) -> np.ndarray | None:
        """Where is the held object, relative to the TCP, RIGHT NOW?

        A human looks at what is in their hand before setting it down. This is
        that check, and it exists because the object does not stay where the
        grasp put it: MEASURED on LIBERO, the horizontal offset from the TCP
        was 1.1 cm at grasp and 5.1 cm at release, i.e. it slid 4.3 cm in
        transit. `place_at` aims the TCP, so that slip lands directly in the
        placement error, against a 3 cm success predicate.

        Re-observes rather than trusting the grasp-time offset, since the
        whole point is that the grasp-time value goes stale. Falls back to the
        grasp-time offset, then to None, so a camera that cannot see the
        gripper degrades to today's behaviour instead of failing the place.

        MEASURED LIMIT: the re-observation only helps when a real detector is
        running. On the LIBERO benchmark in `--perception oracle` the detector
        is `MockDetector`, which reports nothing, so every call falls through
        to the grasp-time value and compensates 0.5 cm instead of the true
        4.8 cm. Median aim error improved only 6.1 -> 4.9 cm there, and that
        residual is the slip this cannot see. Do not read that number as the
        ceiling for this approach; read it as the cost of benchmarking with
        perception switched off.
        """
        if not self.held_object:
            return None
        tcp = None
        try:
            tcp = self.kin.fk(self.arm.get_state().q)[:3, 3]
        except Exception:
            pass
        # An external pose channel, when one is attached, sees the held object
        # even while the camera cannot. On the LIBERO benchmark this is the
        # same physics feed the verifier uses, which keeps the comparison
        # honest: it is a PERCEPTION substitute, exactly like the seeded
        # beliefs, and it is labelled as such in the results.
        if tcp is not None and self._object_pose is not None:
            try:
                p = self._object_pose(self._held_det_label or self.held_object)
                if p is not None:
                    offset = np.asarray(p, float)[:3] - tcp
                    if float(np.linalg.norm(offset[:2])) <= 0.12:
                        return offset
            except Exception:
                pass
        try:
            frame = self.observe()
            if tcp is None:
                tcp = self.kin.fk(self.arm.get_state().q)[:3, 3]
            label = self._held_det_label or self.held_object
            fix = localize_object(
                frame, label, self.detector, self.extrinsics,
                prompts=[label], near_xyz=tcp,
            )
            offset = np.asarray(fix.position, float) - tcp
            # Sanity gate: the object is IN the gripper, so it cannot be far
            # from the TCP. A larger "match" is a different object on the
            # table, and trusting it would throw the place further off than
            # doing nothing.
            if float(np.linalg.norm(offset[:2])) <= 0.12:
                return offset
        except Exception:
            pass
        return getattr(self, "_held_offset", None)

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
        # Aim the OBJECT at the target, not the TCP. The IK below drives the
        # TCP, so the requested point has to be shifted by wherever the object
        # currently sits in the jaws, or the object lands offset by exactly
        # that much. Only the horizontal part is compensated: z is governed by
        # the release height and the wrist ceiling above.
        held_offset = self._held_object_offset()
        if held_offset is not None:
            target[0] -= float(held_offset[0])
            target[1] -= float(held_offset[1])
            self.memory.add(
                "note",
                f"object sits {np.linalg.norm(held_offset[:2])*100:.1f} cm off "
                f"the gripper centre; aiming the TCP at "
                f"[{target[0]:.3f}, {target[1]:.3f}] so the OBJECT lands on "
                f"[{x:.3f}, {y:.3f}]",
            )
        from ..grasping.obb_grasp import _yaw_rotation

        q_now = self.arm.get_state().q
        hover = target + np.array([0.0, 0.0, float(gcfg.get("pregrasp_offset_m", 0.12))])
        hover[2] = min(hover[2], z_cap)  # same wrist ceiling as the release
        # Placement yaw is arbitrary: walk candidate yaws (radial first --
        # kindest to the wrist) until both hover and release poses solve.
        radial = float(np.arctan2(y, x))
        pre = low = None
        for yaw in (radial, 0.0, np.pi / 4, -np.pi / 4, np.pi / 2, -np.pi / 2):
            R = _yaw_rotation(yaw, axis_order=self._tool_axis_order)
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
                # Use the OBJECT's aim point, not the TCP's: `target` may have
                # been shifted to compensate for how the object sits in the
                # jaws, and recording that would seed the belief offset by
                # exactly the amount the compensation just removed.
                self._held_det_label or placed,
                np.array([x, y, release_z], dtype=float),
                0.8, color=self._held_color
            )
            self.held_object = None
            self._held_det_label = None
            self._held_offset = None
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
        # Report where the OBJECT was aimed, not where the TCP was sent. The
        # postcondition channel scores the object's final pose against this,
        # so returning the offset-compensated TCP point would grade the place
        # against the wrong thing and quietly forgive the compensation error.
        return {"placed": placed,
                "at": [round(float(x), 3), round(float(y), 3),
                       round(float(release_z), 3)],
                "tcp_at": [round(float(v), 3) for v in target]}

    def _destination_fix(self, label: str):
        """Where is the destination RIGHT NOW, not when the task started?

        MEASURED FAILURE this guards against (LIBERO-Pro `libero_spatial_swap`,
        task 3): the destination moved after its belief was seeded, the skill
        aimed 3.4 cm off, the object landed 0.8 cm from that stale aim, and the
        postcondition CONFIRMED the place because it asks "did the object reach
        where I aimed", not "did it reach the destination". LIBERO scored it
        false. Verified false claims went 5 -> 13 on that suite while the
        instruction-perturbation suite stayed at 5, so this is specifically a
        targeting failure, not an actuation one.

        A stale belief is worse than no belief here, so a fresh observation
        wins over the stored one whenever it is available. Returns
        `(position, top_z)` or None when nothing can be observed, in which case
        the caller keeps today's behaviour.
        """
        # The external pose channel, when attached, sees the destination even
        # when the camera cannot (benchmark oracle mode). Same provenance
        # caveat as everywhere else it is used.
        if self._object_pose is not None:
            try:
                p = self._object_pose(label)
                if p is not None:
                    p = np.asarray(p, float)[:3]
                    return p, float(p[2])
            except Exception:
                pass
        try:
            frame, fix = self._localize(label)
            self.beliefs.update(
                fix.detection.label or label, fix.position, fix.detection.conf,
                extent=fix.extent, top_z=float(fix.points[:, 2].max()),
                color=detection_color(frame.rgb, fix.detection),
            )
            return np.asarray(fix.position, float), float(fix.points[:, 2].max())
        except Exception:
            return None

    def skill_place_on_object(self, label: str) -> dict:
        self._adopt_unknown_held()
        if not self.held_object:
            raise SkillError("not holding anything")
        # Re-check the destination immediately before committing to a drop
        # point. The belief may be stale: it was seeded when the task started
        # and the world does not hold still, which is the whole premise of the
        # perturbed benchmarks.
        fresh = self._destination_fix(label)
        if fresh is not None:
            pos, top = fresh
        else:
            belief = self.beliefs.find(label)
            if belief is None:
                frame, fix = self._localize(label)
                self.beliefs.update(
                    fix.detection.label or label, fix.position,
                    fix.detection.conf, extent=fix.extent,
                    top_z=float(fix.points[:, 2].max()),
                    color=detection_color(frame.rgb, fix.detection),
                )
                belief = self.beliefs.find(label)
            # Use the actually observed highest point of the target, never OBB
            # extents (those are eigenvalue-ordered, not axis-aligned).
            pos = belief.position
            top = (belief.top_z if belief.top_z is not None
                   else float(belief.position[2]))
        drop = top + float(self.cfg.grasp.get("release_clearance_m", 0.06))
        return self.skill_place_at(float(pos[0]), float(pos[1]), drop)

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

        R = _yaw_rotation(float(np.arctan2(d2[1], d2[0])),
                          axis_order=self._tool_axis_order)
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
                    make_transform(_yaw_rotation(yaw, axis_order=self._tool_axis_order),
                                   [hover[0], hover[1], z]), q_now
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
        home = self._profile_q("home_q", "wave")
        if not self.arm.move_joints(home, duration_s=2.0):
            raise SkillError("could not reach home to wave")
        cycles = int(np.clip(cycles, 1, 4))
        wrist = self._gesture_joint
        for side in [+1, -1] * cycles:
            q = home.copy()
            q[0] += 0.25 * side
            q[wrist] += 0.30 * side
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
        hand_q = self._profile_q("handover_q", "present the object to a human")
        if not self.arm.move_joints(hand_q, duration_s=2.0):
            raise SkillError("did not settle at the handover pose")
        self.memory.add("action", f"offering {self.held_object!r} to the human")
        return {
            "offering": self.held_object,
            "note": "holding steady; call open_gripper once the human has it",
        }

    def skill_throw(self, label: str | None = None, direction: str = "forward") -> dict:
        """Booth crowd-pleaser: grasp the object (if not already held), wind
        the arm up, swing it toward `direction`, and RELEASE at the top of
        the swing so the object is launched. The safety harness still vets
        every waypoint (velocity cap, workspace AABB, table clearance), so
        this is a *gestural* throw -- arm swing speed + release timing give
        the toss, never an unsafe joint velocity. Great for "grab the banana
        and throw it".
        """
        dirs = {
            "forward": 0.0, "left": np.pi / 2, "right": -np.pi / 2,
            "back": np.pi,
        }
        if direction not in dirs:
            raise SkillError(f"direction must be one of {sorted(dirs)}")
        # 1) make sure we're holding something.
        if not self.held_object:
            if not label:
                raise SkillError("not holding anything; say which object to throw")
            res = self.skill_grasp_object(label)
            if not res.get("ok", True) or not res.get("held"):
                return {
                    "ok": False,
                    "error": f"could not grasp {label!r} to throw: {res.get('error')}",
                }
        thrown = self.held_object
        yaw = dirs[direction]

        # 2) wind-up pose: arm drawn back and low, jaws still closed.
        #    Joint-space keyframes keep this reachable on the B601-RS wrist
        #    envelope (top-down IK is limited above z~0.15). base yaw (j1)
        #    aims the throw; j2/j3 load the swing.
        base = self._profile_q("home_q", "throw")
        windup = base.copy()
        windup[0] = yaw                     # aim
        windup[1] = base[1] - 0.5           # shoulder back/down (loaded)
        windup[2] = base[2] + 0.4           # elbow tucked
        release = base.copy()
        release[0] = yaw                    # same aim
        release[1] = base[1] + 0.6          # shoulder swings up/forward
        release[2] = base[2] - 0.3          # elbow extends

        # 3) execute: settle at wind-up, then swing FAST (short duration --
        #    the harness auto-stretches it only if it would breach the cap,
        #    so we get the quickest *safe* swing) and pop the jaws open at
        #    the peak. Releasing while the wrist is still moving imparts the
        #    launch impulse.
        if not self.arm.move_joints(windup, duration_s=1.5):
            raise SkillError("did not settle at the wind-up pose")
        self.memory.add("action", f"winding up to throw {thrown!r} {direction}")
        # kick off the swing in a background stream and release mid-arc.
        self.arm.move_joints(release, duration_s=0.6)
        self.arm.set_gripper(self._grip_open, effort=1.0)  # let it fly
        released_obj = self.held_object
        self.held_object = None
        self._held_det_label = None
        self._held_color = None
        # follow-through, then home so the camera view clears.
        try:
            self.skill_move_home()
        except (SkillError, SafetyViolation):
            pass
        self.memory.add("action", f"threw {released_obj!r} {direction}")
        return {
            "ok": True,
            "thrown": released_obj,
            "direction": direction,
            "note": "gestural throw: harness-vetted swing + timed release",
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
        home = self._profile_q("home_q", "move home")
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

    def skill_annotated_view(self) -> dict:
        """VIA-style annotated interface (arXiv:2607.11119).

        Returns the numbered-object key plus a saved annotated frame. The
        image itself goes to the live view / trace rather than into the tool
        result: the orchestrator already attaches the newest frame to context,
        and returning a second base64 image per call would double token cost
        for no gain.
        """
        from ..perception.visual_interface import VisualInterface, annotate_frame

        self.observe()
        img, marks = annotate_frame(self)
        if img is None:
            raise SkillError("no frame available to annotate")
        path = self.trace.save_keyframe(img, "annotated_view")
        if hasattr(self.camera, "set_overlay"):
            self.camera.set_overlay(status="annotated view")
        return {
            "key": VisualInterface.describe(marks),
            "objects": [m.as_dict() for m in marks],
            "image": path,
            "note": (
                "Objects carry numbered badges; the green band is the region "
                "where strict top-down IK actually solves on this arm."
            ),
        }

    def skill_open_live_view(self, reason: str = "") -> dict:
        """Open the browser dashboard on demand (it is closed by default).

        The chat client is the primary UI; this binds the HTTP port only when
        a human actually wants to look, and it auto-closes when idle.
        """
        if self.live_view is None:
            raise SkillError(
                "no live view configured in this runtime (headless/unit-test build)"
            )
        out = self.live_view.open(reason=reason)
        self.stream_server = self.live_view.server
        if not out.get("ok"):
            raise SkillError(str(out.get("error", "could not open the live view")))
        out["views"] = {
            "rgb": "detections + HUD (what the detector sees)",
            "depth": "depth colormap + range stats (what the geometry sees)",
            "agent": "numbered marks + metric grid + reachable IK band",
        }
        out["hint"] = (
            "Share the URL with the human. Each camera tile has rgb/depth/agent "
            "buttons, an analyze panel, and a chat box that drives this same robot."
        )
        return out

    def skill_close_live_view(self) -> dict:
        """Close the dashboard and release the port."""
        if self.live_view is None:
            return {"ok": True, "open": False, "note": "no live view configured"}
        out = self.live_view.close(reason="closed on request")
        self.stream_server = self.live_view.server
        return out

    def skill_live_view_status(self) -> dict:
        """Is the dashboard open, on what URL, and how idle is it?"""
        if self.live_view is None:
            return {"ok": True, "open": False, "mode": "unavailable"}
        out = dict(self.live_view.status())
        out["ok"] = True
        return out

    def skill_analyze_scene(self, camera: str | None = None) -> dict:
        """Full perception report: detections, depth quality, description.

        Answers "what do you see?" WITHOUT needing the dashboard open -- this
        is the headless counterpart of the dashboard's analyze button, and it
        is why cameras can stay closed by default.
        """
        rig = getattr(self, "rig", None)
        if rig is None:
            raise SkillError("no camera rig in this runtime")
        out: dict = {"ok": True, "cameras": {}}
        names = [camera] if camera else list(rig.names)
        for name in names:
            try:
                stream = rig.get(name)
            except KeyError:
                out["cameras"][name] = {"error": f"unknown camera {name!r}"}
                continue
            frame = stream.latest()
            if frame is None:
                out["cameras"][name] = {"error": "no frame yet"}
                continue
            dets, status = stream.overlay()
            entry: dict = {
                "fps": round(float(stream.fps), 1),
                "resolution": [int(frame.rgb.shape[1]), int(frame.rgb.shape[0])],
                "depth_source": frame.depth_source,
                "agent_status": status,
                "detections": [
                    {"label": getattr(d, "label", "?"),
                     "conf": round(float(getattr(d, "conf", 0.0)), 3)}
                    for d in dets
                ],
            }
            if frame.has_depth and frame.depth_m is not None:
                depth = np.asarray(frame.depth_m, dtype=np.float32)
                valid = depth[np.isfinite(depth) & (depth > 0)]
                entry["depth"] = {
                    "valid_fraction": round(float(valid.size) / float(depth.size), 3),
                    "min_m": round(float(valid.min()), 3) if valid.size else None,
                    "median_m": round(float(np.median(valid)), 3) if valid.size else None,
                    "max_m": round(float(valid.max()), 3) if valid.size else None,
                }
            else:
                entry["depth"] = None
                entry["depth_warning"] = (
                    "no depth: 3D grounding will refuse to localize from this camera"
                )
            out["cameras"][name] = entry
        try:
            out["scene"] = self.skill_describe_scene()
        except Exception as e:
            out["scene"] = {"error": str(e)[:200]}
        try:
            from ..perception.visual_interface import VisualInterface, annotate_frame

            _, marks = annotate_frame(self)
            out["objects"] = [m.as_dict() for m in marks]
            out["key"] = VisualInterface.describe(marks)
        except Exception:
            pass
        if self.live_view is not None:
            st = self.live_view.status()
            out["live_view"] = {"open": st["open"], "url": st["url"]}
            if not st["open"]:
                out["live_view"]["hint"] = (
                    "call open_live_view to watch this in a browser"
                )
        return out

    def skill_grasp_at_pixel(
        self, u: float, v: float, camera: str | None = None,
        normalized: bool = False, material: str | None = None,
    ) -> dict:
        """Grasp whatever is at this pixel, without needing its class name.

        VIA (arXiv:2607.11119) withholds perception APIs entirely and has the
        agent click in an RGB-D point cloud, on the reasoning that a frontier
        model can already SEE the object; what it lacks is a metric way to
        address it. Every other grasp skill here routes through
        `_localize(label)`, so the agent can only act on things the detector
        names, which is a measured ceiling: on LIBERO frames the detector
        emits 38-44 detections and never says `bowl`, while the bowl is
        plainly visible.

        The pixel is segmented (learned segmenter when configured, depth
        connectivity otherwise) and the OBB centre of that region becomes the
        grasp target. Using the probed surface point directly would grasp
        high: a ray hits the first surface, so the middle of a cube probes its
        top face (+22 mm on a 4.5 cm cube).

        Measured on a LIBERO frame against MuJoCo body poses, the segmenter is
        what makes this usable on a tabletop: depth connectivity alone fills
        the table (34 cm error), SAM2.1 with the same point prompt lands the
        bowl at 3.0 cm and the plate at 0.9 cm.

        Safety is unchanged: the same harness, workspace filter and
        postcondition checks apply, because this produces the same ObjectFix
        the detector path produces.
        """
        from ..perception.segmenter import fix_at_pixel

        self._reconcile_held()
        if self.held_object:
            raise SkillError(f"already holding {self.held_object!r}; place it first")

        frame = self.observe()
        h, w = frame.rgb.shape[:2]
        uu, vv = float(u), float(v)
        if normalized or (0.0 <= uu <= 1.0 and 0.0 <= vv <= 1.0 and max(w, h) > 4):
            uu, vv = uu * (w - 1), vv * (h - 1)

        # Eye-in-hand cameras carry per-frame extrinsics; fall back to the
        # camera profile's static ones. Same rule as the detector path.
        T = (frame.T_base_cam if frame.T_base_cam is not None
             else self.extrinsics.cam_to_base())
        fix = fix_at_pixel(frame, T, int(round(uu)), int(round(vv)),
                           segmenter=self._segmenter)

        mask = fix.detection.mask
        why = self._workspace.reject(
            fix.position, fix.extent,
            mask_frac=(float(mask.sum()) / float(w * h)) if mask is not None else 0.0,
        )
        if why:
            raise SkillError(
                f"the object at pixel ({int(uu)}, {int(vv)}) is not graspable: {why}"
            )
        return self.skill_grasp_object(
            fix.label, material=material, _fix=fix, _frame=frame,
        )

    def skill_probe_point(
        self, u: float, v: float, camera: str | None = None,
        normalized: bool = False,
    ) -> dict:
        """Queryable cursor: what is at this pixel, in metres.

        Anthropic's Claude-Plays-Robotics ablation found that static depth and
        segmentation OVERLAYS were roughly neutral, while a cursor the model
        can move and query lifted manipulation success from 6% to 32% -- the
        signal has to be a number you asked for, not a texture you must read.
        """
        from ..perception.probe import PointProbe

        try:
            return PointProbe(self).probe(u, v, camera=camera, normalized=normalized)
        except KeyError:
            raise SkillError(f"unknown camera {camera!r}")
        except ValueError as e:
            raise SkillError(str(e))

    def skill_locate_pixel(self, label: str, camera: str | None = None) -> dict:
        """Where is a known object in the image? (inverse of probe_point)

        Lets you move the cursor to something you already track, and reason in
        the same pixel space you are looking at.
        """
        from ..perception.probe import PointProbe

        try:
            return PointProbe(self).locate_pixel(label, camera=camera)
        except KeyError:
            raise SkillError(f"unknown camera {camera!r}")
        except ValueError as e:
            raise SkillError(str(e))

    def skill_task_done(self, success: bool, summary: str) -> dict:
        if isinstance(success, str):  # schema-lax backends send "false"
            success = success.strip().lower() in ("true", "yes", "1")
        return {"ok": True, "task_complete": True, "success": bool(success), "summary": summary}


def _short(args: dict) -> str:
    return ", ".join(f"{k}={v}" for k, v in args.items())


TOOL_SPECS: list[dict] = [
    {
        "name": "grasp_at_pixel",
        "description": (
            "Grasp whatever is at this pixel, WITHOUT needing to name it. Use "
            "this when you can see the object in the image but grasp_object "
            "fails with 'no detections': the detector's vocabulary does not "
            "have to contain the object for this to work. The pixel is "
            "segmented by depth and the object's centre is computed from that "
            "region, so point anywhere on the object body (avoid edges and "
            "shadows). Same safety checks as grasp_object."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "u": {"type": "number", "description": "Column in pixels, or 0..1 if normalized."},
                "v": {"type": "number", "description": "Row in pixels, or 0..1 if normalized."},
                "normalized": {
                    "type": "boolean",
                    "description": "True if u,v are fractions of image size.",
                },
                "material": {
                    "type": "string",
                    "description": "Optional material hint for grasp force.",
                },
            },
            "required": ["u", "v"],
        },
    },
    {
        "name": "halt_motion",
        "description": (
            "Stop the motion currently in flight because it is no longer the "
            "right action (wrong object, subgoal already satisfied, the scene "
            "changed). Takes effect within ~20 ms. This is NOT an emergency "
            "stop: it does not latch, it clears when the next motion starts, "
            "and the arm stays powered, so recovery is just issuing the "
            "corrected command. Use 'stop' if the rig is actually unsafe."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "description": "Why this motion is being abandoned.",
                },
            },
            "required": [],
        },
    },
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
        "name": "annotated_view",
        "description": (
            "Render the annotated camera view: every tracked object gets a NUMBERED badge, "
            "the table is overlaid with a 5 cm base-frame grid, and the region where "
            "top-down grasps are actually kinematically reachable is shaded green. "
            "Returns a text key mapping each number to its label and 3D position. Use it "
            "when a scene is cluttered, when labels are ambiguous, or before choosing "
            "where to place something -- it shows what is reachable instead of guessing."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "probe_point",
        "description": (
            "CURSOR: point at a pixel in the camera image and get hard numbers back -- "
            "distance in metres, the 3D position in the robot's base frame, which tracked "
            "object is at that point, whether the arm can actually reach it (workspace + "
            "top-down IK band), and the offset from the current gripper position. "
            "Use it whenever you need spatial precision: before a grasp on a cluttered "
            "or ambiguous scene, to check a placement spot is reachable, or to measure "
            "how far off the gripper is. Coordinates may be pixels or 0..1 normalized."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "u": {"type": "number", "description": "x pixel (or 0..1 if normalized)"},
                "v": {"type": "number", "description": "y pixel (or 0..1 if normalized)"},
                "camera": {"type": "string", "description": "camera name; omit for the manipulation camera"},
                "normalized": {"type": "boolean", "description": "treat u,v as 0..1 fractions"},
            },
            "required": ["u", "v"],
        },
    },
    {
        "name": "locate_pixel",
        "description": (
            "Inverse of probe_point: given an object the robot already tracks, return "
            "WHERE IT IS IN THE IMAGE (pixel + normalized coords) plus its distance and "
            "whether it is currently in view. Use it to aim the cursor at a known object, "
            "or to check an object is visible in a given camera before acting on it."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "label": {"type": "string"},
                "camera": {"type": "string"},
            },
            "required": ["label"],
        },
    },
    {
        "name": "analyze_scene",
        "description": (
            "Full perception report WITHOUT opening any window: per-camera detections "
            "with confidence, depth quality (source + min/median/max range + valid "
            "fraction), a natural-language scene description, and the numbered object "
            "key. This is how you answer 'what do you see?' while the cameras/UI stay "
            "closed. Prefer it over camera_snapshot when you need facts rather than "
            "pixels -- it costs no image tokens."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "camera": {"type": "string",
                           "description": "one camera name; omit for all"},
            },
            "required": [],
        },
    },
    {
        "name": "open_live_view",
        "description": (
            "Open the browser dashboard for the human (it is CLOSED by default -- chat "
            "is the interface). Returns a URL to share. The page shows all cameras "
            "together with per-tile rgb / depth / agent view switches, an analyze panel "
            "(detections + depth stats + description), the world model, robot narration, "
            "and a chat box that drives this same robot. Auto-closes when nobody is "
            "watching. Call this when the human asks to see/watch the cameras."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "reason": {"type": "string",
                           "description": "why it is being opened (shown in logs)"},
            },
            "required": [],
        },
    },
    {
        "name": "close_live_view",
        "description": (
            "Close the browser dashboard and release its port. Perception keeps "
            "running; only the HTTP view stops. Call it when the human is done looking."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "live_view_status",
        "description": (
            "Whether the browser dashboard is currently open, its URL, and how long it "
            "has been idle. Check before offering a link so you never hand out a dead URL."
        ),
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
        "name": "preview_grasp",
        "description": "Plan a grasp on a named object and REPORT the proposed gripper waypoint (position, approach direction, confidence, learned-memory prior) WITHOUT moving. Use it to check a grasp looks right before committing, then call grasp_object. Cheap observe-before-act step.",
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
        "name": "throw",
        "description": "Grab an object (if not already held) and throw it by winding up and releasing at the top of a harness-vetted swing. direction is forward/left/right/back (default forward). Use for 'grab the banana and throw it' / 'launch the X'.",
        "parameters": {
            "type": "object",
            "properties": {
                "label": {"type": "string"},
                "direction": {
                    "type": "string",
                    "enum": ["forward", "left", "right", "back"],
                },
            },
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
