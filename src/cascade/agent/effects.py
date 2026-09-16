"""Postconditions: verifying what a primitive actually DID, independently.

Pigey (arXiv:2607.21725) isolates the "orchestration gap" -- the difference
between what frozen motor skills achieve alone and what they achieve inside a
closed loop that *tracks and verifies the outcome from low-level observations
and recovers from failures*.  On LIBERO-PRO that loop is worth 12.8% -> 53.3%
with identical weights.  Nothing is retrained; only the inference-time
process changes.

This module supplies the "verify the outcome" half for cascade.

The problem it fixes is concrete and already documented in this repo's own
notes: **skills currently self-report**.  ``grasp_object`` returns ok when the
jaw stopped short of fully closed (``air_grasp_frac``), which is a *proxy* for
holding something -- it cannot distinguish a grasped cube from a jammed
finger, and it says nothing about whether the object actually left the table.
``place_at`` reports ok when the gripper opened.  An actuator asserting its
own success is the weakest possible evidence.

A postcondition here is checked against an INDEPENDENT channel:

* the belief store / a fresh detector pass (did the object move to where we
  claim?),
* the physics-truth pose via the sim bridge when available (exact),
* the gripper width (necessary but never sufficient),

and it returns one of ``CONFIRMED`` / ``REFUTED`` / ``UNVERIFIED``.  That
third state is the honest one and is why this module exists: the demo's
failure mode was never "the robot lied", it was "the robot did not know".

Cheap by design: verification runs only after motion skills, reuses the
perception pass the runtime already performs, and degrades to UNVERIFIED
rather than blocking when a channel is unavailable.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

CONFIRMED = "confirmed"
REFUTED = "refuted"
UNVERIFIED = "unverified"

#: Skills whose physical effect is worth an independent check, mapped to the
#: postcondition kind.  Skills absent from this map are not verified (their
#: effect is either cosmetic -- wave -- or already terminal -- task_done).
POSTCONDITIONS: dict[str, str] = {
    "grasp_object": "holding",
    "pick_and_place": "relocated",
    "place_at": "released_at",
    "place_on_object": "released_on",
    "push_object": "moved",
    "throw": "gone",
    "handover": "released",
    "open_gripper": "empty",
    "close_gripper": "closed",
    "move_home": "at_home",
}

#: An object that rose by at least this much (m) genuinely left the table.
LIFT_EPS_M = 0.015
#: Two positions within this distance (m) are "the same place". Used to decide
#: whether the object moved AT ALL, where being generous is correct.
SAME_PLACE_M = 0.05
#: How close the object must end up to a NAMED destination to count as placed
#: on it. Deliberately separate from SAME_PLACE_M, and deliberately tight.
#:
#: MEASURED: with the old shared threshold (SAME_PLACE_M * 2 = 10 cm) the
#: checker confirmed a place that ended 4.25 cm from the plate, while LIBERO's
#: On() predicate requires 3 cm. A verifier looser than the task's own success
#: criterion cannot catch a near miss, which is precisely the failure mode
#: worth catching: the arm did something plausible and slightly wrong.
#:
#: 0.03 matches LIBERO's predicate. It is a placement tolerance, not a
#: perception tolerance; if the pose channel is noisier than this the right
#: fix is a better channel, not a looser check.
DEST_TOLERANCE_M = 0.03
#: A push must displace the object by at least this fraction of the request.
PUSH_MIN_FRAC = 0.3


@dataclass
class Postcondition:
    """Verdict on one primitive's physical effect."""

    skill: str
    kind: str
    status: str = UNVERIFIED
    evidence: str = ""
    channel: str = ""        # "physics" | "belief" | "gripper" | ""
    measured: dict = field(default_factory=dict)
    t: float = field(default_factory=time.time)

    @property
    def confirmed(self) -> bool:
        return self.status == CONFIRMED

    @property
    def refuted(self) -> bool:
        return self.status == REFUTED

    def as_dict(self) -> dict:
        return {
            "skill": self.skill,
            "kind": self.kind,
            "status": self.status,
            "evidence": self.evidence,
            "channel": self.channel,
            "measured": self.measured,
        }

    def agent_line(self) -> str:
        mark = {CONFIRMED: "VERIFIED", REFUTED: "CONTRADICTED", UNVERIFIED: "UNVERIFIED"}
        return f"[{mark[self.status]}] {self.skill}: {self.evidence}"


class PostconditionChecker:
    """Verifies primitive effects against channels the actuator does not own.

    All hooks are optional callables so this works in mock/offline runs and in
    unit tests without a rig:

    ``object_pose(label) -> (x, y, z) | None``
        Ground-truth pose, e.g. the Isaac bridge reading a RigidPrim.  This is
        the strongest channel and is preferred whenever present.
    ``belief_pose(label) -> (x, y, z) | None``
        Perceived pose from the belief store.
    ``gripper_frac() -> float | None``
        Jaw opening fraction (0 closed .. 1 open).
    ``reobserve() -> None``
        Force a fresh detector pass before reading beliefs.
    """

    def __init__(
        self,
        object_pose: Callable[[str], Any] | None = None,
        belief_pose: Callable[[str], Any] | None = None,
        gripper_frac: Callable[[], float | None] | None = None,
        reobserve: Callable[[], None] | None = None,
        visual_diff: Callable[..., Any] | None = None,
        table_z: float = 0.0,
        air_grasp_frac: float = 0.04,
    ):
        self._object_pose = object_pose
        self._belief_pose = belief_pose
        self._gripper_frac = gripper_frac
        self._reobserve = reobserve
        #: CaP-X visual differencing (source_xyz, target_xyz) -> DiffVerdict.
        #: The only actuator-independent channel available on real hardware.
        self._visual_diff = visual_diff
        self.table_z = float(table_z)
        self.air_grasp_frac = float(air_grasp_frac)
        self.history: list[Postcondition] = []

    # ── snapshots ────────────────────────────────────────────────────────

    def snapshot(self, label: str | None) -> dict:
        """Pose of ``label`` before a motion, from the best channel available."""
        if not label:
            return {}
        pose, channel = self._best_pose(label)
        if pose is None:
            return {}
        return {"label": label, "pose": list(pose), "channel": channel}

    @staticmethod
    def _comparable_start(before: dict, channel_after: str):
        """The pre-motion pose, but ONLY if it can be compared with the
        post-motion reading.

        A displacement is a difference of two readings of the SAME channel.
        Mixing them is not a looser measurement, it is a wrong one -- and it
        happens by construction under the MCP server: the arm is lazy, so at
        snapshot time the physics channel is not bound yet and the snapshot
        falls back to a belief RESTORED FROM DISK (a previous episode's drop
        point); the pick materializes the arm, the physics channel comes up
        for the AFTER reading, and "physics_after - belief_before" reads as
        "still within 1.8 cm of where it started" for a pick that visibly
        succeeded (reproduced 2026-09-09, chat path, refuted a good pick).
        Returning None makes the check report the after-pose without a
        displacement claim instead of a false REFUTED.
        """
        start = before.get("pose")
        if start is None:
            return None
        ch_before = before.get("channel") or ""
        if ch_before and channel_after and ch_before != channel_after:
            return None
        return start

    def _best_pose(self, label: str) -> tuple[Any, str]:
        for fn, channel in ((self._object_pose, "physics"), (self._belief_pose, "belief")):
            if fn is None:
                continue
            try:
                pose = fn(label)
            except Exception:
                continue
            if pose is not None and len(pose) >= 3:
                return [float(v) for v in pose[:3]], channel
        return None, ""

    def _visual_evidence(self, source_xyz=None, target_xyz=None):
        """CaP-X visual differencing: pixels as an actuator-independent channel.

        On real hardware `belief` is the only pose channel and it is written
        by the same perception pass the skill just ran, so agreeing with it is
        a tautology. The camera is not the actuator, so a before/after pixel
        comparison is genuinely independent evidence -- coarse, but real.
        Returns None when no diff was captured or the scene was unreadable.
        """
        if self._visual_diff is None:
            return None
        try:
            v = self._visual_diff(source_xyz, target_xyz)
        except Exception:
            return None
        if v is None or getattr(v, "status", "unknown") == "unknown":
            return None
        return v

    # A placement needs an independent pose channel; the skill also writes belief.
    _SELF_REPORTED_TARGET_NEEDS_INDEPENDENT_CHANNEL = True

    # ── verification ─────────────────────────────────────────────────────

    def verify(
        self,
        skill: str,
        args: dict,
        result: dict,
        before: dict | None = None,
        fresh: bool = False,
    ) -> Postcondition | None:
        """Check the physical effect of a completed skill call.

        ``before`` is a prior ``snapshot()``.  Returns None when the skill has
        no postcondition worth checking.  Never raises.

        ``fresh`` forces a detector pass first and defaults to **False** by
        design: verification must be READ-ONLY with respect to the world model
        it is judging.  Forcing a re-observation here corrupts exactly the
        state under test -- ``place_at`` deliberately authors the moved
        object's belief at the drop point, and an immediate re-scan overwrites
        it with whatever the camera still sees (caught by
        ``test_grasp_and_place_happy_path``, where the belief snapped back to
        the mock camera's fixed detection instead of the place target).
        The ``perception_loop`` WorldWatcher already keeps beliefs warm at
        3 Hz, and in sim the truth channel needs no perception at all, so the
        fresh pass buys nothing and costs correctness.
        """
        kind = POSTCONDITIONS.get(skill)
        if kind is None:
            return None
        pc = Postcondition(skill=skill, kind=kind)
        try:
            if fresh and self._reobserve is not None and kind not in ("at_home", "closed", "empty"):
                self._reobserve()
            handler = getattr(self, f"_check_{kind}", None)
            if handler is not None:
                handler(pc, args, result, before or {})
        except Exception as e:
            pc.status, pc.evidence = UNVERIFIED, f"verification error: {type(e).__name__}: {e}"
        self.history.append(pc)
        return pc

    # ── individual postconditions ────────────────────────────────────────

    def _check_holding(self, pc, args, result, before) -> None:
        """A grasp succeeded iff the object LEFT THE TABLE with the gripper.

        The jaw fraction alone cannot show this -- it is necessary, not
        sufficient.  We require a measured rise in the object's own pose.
        """
        label = str(args.get("label") or result.get("object") or before.get("label") or "")
        frac = self._frac()
        if frac is not None and frac <= self.air_grasp_frac:
            pc.status, pc.channel = REFUTED, "gripper"
            pc.evidence = f"jaw closed to {frac:.3f} (<= air-grasp threshold): nothing between the fingers"
            pc.measured = {"gripper_frac": frac}
            return
        pose, channel = self._best_pose(label) if label else (None, "")
        if pose is None:
            pc.status, pc.channel = UNVERIFIED, "gripper"
            pc.evidence = (
                f"jaw held at {frac:.3f} but {label or 'the object'} could not be "
                "re-located, so the lift is unconfirmed"
                if frac is not None
                else "no independent channel could confirm the grasp"
            )
            pc.measured = {"gripper_frac": frac} if frac is not None else {}
            return
        start = self._comparable_start(before, channel)
        z0 = float((start or [0, 0, self.table_z])[2])
        rise = pose[2] - z0
        pc.measured = {"z_before": round(z0, 4), "z_after": round(pose[2], 4),
                       "rise_m": round(rise, 4), "gripper_frac": frac}
        pc.channel = channel
        if rise >= LIFT_EPS_M:
            pc.status = CONFIRMED
            pc.evidence = f"{label} rose {rise*100:.1f} cm off the table ({channel} pose)"
        else:
            pc.status = REFUTED
            pc.evidence = (
                f"{label} did not rise (dz={rise*100:+.1f} cm): the jaw closed but "
                "the object stayed on the table"
            )

    def _check_relocated(self, pc, args, result, before) -> None:
        # pick_and_place names its subject `object`; the resolved label comes
        # back in the result, which is the most reliable source (the request
        # may have been a colour query like "pink object").
        label = str(
            result.get("picked")
            or result.get("object")
            or args.get("label")
            or args.get("object")
            or before.get("label")
            or ""
        )
        # Where the skill BELIEVES it put the object. pick_and_place reports
        # `placed_at`; only reading `target` made this fall through to the weak
        # "it moved, good enough" branch and confirm a real miss on the live
        # rig (2026-07-31: cube ended at (0.132, 0.090), 7 cm from where it
        # started and nowhere near the bin, reported as confirmed).
        #
        # Provenance matters for what the agreement is worth: a `target` in the
        # ARGS was chosen by the caller, so matching it is real evidence. A
        # `placed_at` in the RESULT was chosen by the same code path that wrote
        # the belief, so matching it in the belief channel proves nothing.
        requested = args.get("target")
        target = (
            requested
            or result.get("target")
            or result.get("placed_at")
            or result.get("at")
        )
        target_is_independent = requested is not None
        # A NAMED destination ("place it on the plate") is independent
        # evidence in a way a coordinate from the result never is: the
        # destination's pose is read from the same channel that scores the
        # object, so a stale aim cannot launder itself into a confirmation.
        #
        # MEASURED (LIBERO-Pro libero_spatial_swap, task 3): the destination
        # moved after its belief was seeded, the skill aimed 3.4 cm off, the
        # object landed 0.8 cm from that stale aim, and this checker CONFIRMED
        # it while LIBERO scored it false. Verified false claims were 13/100 on
        # that suite against 5/100 where nothing moves. Scoring against the
        # destination's CURRENT pose is what closes that hole.
        dest_label = (
            args.get("destination")
            or (args.get("label") if str(pc.skill) == "place_on_object" else None)
        )
        dest_pose = None
        # A calibrated floor mark has no movable body. A colour-only pose
        # fallback can otherwise resolve "green square" to the green cube
        # itself and report a meaningless zero destination error.
        if (dest_label and str(dest_label) != str(label)
                and result.get("destination_kind") != "configured_point"):
            dest_pose, _ = self._best_pose(str(dest_label))
        pose, channel = self._best_pose(label) if label else (None, "")
        if pose is None:
            pc.status, pc.evidence = UNVERIFIED, f"{label or 'object'} not re-located after the move"
            return
        pc.channel = channel
        start = self._comparable_start(before, channel)
        if start is not None:
            moved = _dist(pose, start)
            pc.measured = {"moved_m": round(moved, 4), "final": [round(v, 4) for v in pose]}
            if moved < SAME_PLACE_M:
                pc.status = REFUTED
                pc.evidence = f"{label} is still within {moved*100:.1f} cm of where it started"
                return
        else:
            final: dict[str, Any] = {"final": [round(v, 4) for v in pose]}
            if before.get("pose") is not None:
                final["start_channel_mismatch"] = f"{before.get('channel')}->{channel}"
            pc.measured = final
        if dest_pose is not None:
            err = _dist(pose[:2], dest_pose[:2])
            pc.measured["dest_err_m"] = round(err, 4)
            if err > DEST_TOLERANCE_M:
                pc.status = REFUTED
                pc.evidence = (
                    f"{label} ended {err*100:.1f} cm from {dest_label} "
                    f"(at {[round(v, 3) for v in pose[:2]]}, {dest_label} is at "
                    f"{[round(float(v), 3) for v in dest_pose[:2]]})"
                )
                return
        if isinstance(target, (list, tuple)) and len(target) >= 2:
            err = _dist(pose[:2], [float(v) for v in target[:2]])
            pc.measured["target_err_m"] = round(err, 4)
            if err > SAME_PLACE_M * 2:
                pc.status = REFUTED
                pc.evidence = (
                    f"{label} ended {err*100:.1f} cm from where it was meant to go "
                    f"(at {[round(v, 3) for v in pose[:2]]}, wanted "
                    f"{[round(float(v), 3) for v in target[:2]]})"
                )
                return
            # The kitchen's bounded areas require containment, beyond a point-placement check.
            if (result.get("destination_kind") == "configured_point"
                    and result.get("destination") in {"green square", "open box"}):
                pc.status = UNVERIFIED
                point_name = result.get("destination") or "configured drop point"
                pc.measured["destination"] = point_name
                pc.evidence = (
                    f"{label} is {err*100:.1f} cm from the {point_name} center ({channel}); "
                    "full object containment and release are not established by a center position"
                )
                return
            # Agreement with a SELF-REPORTED drop point only counts when it
            # comes from a channel the skill does not own. `placed_at` and the
            # belief were both written by the same code path, so "0.0 cm from
            # the requested drop point (belief)" is a tautology, not evidence.
            if channel != "physics" and not target_is_independent:
                # CaP-X: before falling back to "unverified", ask the pixels.
                # The camera is not the actuator, so this IS independent.
                vis = self._visual_evidence(before.get("pose"), target)
                if vis is not None and vis.status == "changed":
                    pc.status = CONFIRMED
                    pc.channel = "visual_diff"
                    pc.evidence = (
                        f"{label} is {err*100:.1f} cm from the requested drop point, "
                        f"corroborated by pixels ({vis.detail})"
                    )
                    pc.measured.update(vis.measured or {})
                    return
                if vis is not None and vis.status == "unchanged":
                    pc.status = REFUTED
                    pc.channel = "visual_diff"
                    pc.evidence = (
                        f"the world model says {label} moved, but the camera sees "
                        f"no change ({vis.detail})"
                    )
                    pc.measured.update(vis.measured or {})
                    return
                pc.status = UNVERIFIED
                pc.evidence = (
                    f"{label} matches the drop point the skill itself reported, but only "
                    f"in the {channel} channel it also wrote -- no independent confirmation"
                )
                return
            pc.status = CONFIRMED
            point_name = result.get("destination") or "requested drop point"
            pc.measured["destination"] = point_name
            pc.evidence = f"{label} is {err*100:.1f} cm from the {point_name} center ({channel})"
            return
        # No drop point to compare against: "it moved" is NOT evidence that it
        # went where it was asked to go. Say so instead of confirming.
        pc.status = UNVERIFIED
        pc.evidence = (
            f"{label} moved, but no drop point was reported, so 'placed correctly' "
            "could not be verified"
        )

    def _check_released_at(self, pc, args, result, before) -> None:
        label = str(before.get("label") or result.get("object") or "")
        want = [args.get("x"), args.get("y")]
        pose, channel = self._best_pose(label) if label else (None, "")
        frac = self._frac()
        if pose is None:
            pc.status = UNVERIFIED
            pc.evidence = (
                f"gripper opened to {frac:.2f} but the released object was not re-located"
                if frac is not None else "release not independently observed"
            )
            return
        pc.channel = channel
        if want[0] is not None and want[1] is not None:
            err = _dist(pose[:2], [float(want[0]), float(want[1])])
            pc.measured = {"target_err_m": round(err, 4), "z": round(pose[2], 4)}
            pc.status = CONFIRMED if err <= SAME_PLACE_M * 2 else REFUTED
            pc.evidence = (
                f"{label} rests {err*100:.1f} cm from the requested point ({channel})"
                if pc.status == CONFIRMED
                else f"{label} ended {err*100:.1f} cm away from where it was placed"
            )
            return
        pc.status, pc.evidence = CONFIRMED, f"{label} located after release ({channel})"

    def _check_released_on(self, pc, args, result, before) -> None:
        # The HELD object is the subject; `label` is the DESTINATION. Reading
        # the subject from before["label"] compares the target with itself and
        # cheerfully reports "box sits on box (offset 0.0 cm)" -- a vacuous
        # confirmation that masked a real miss on the live rig (2026-07-31:
        # the cube landed at (0.272, 0.084), well outside the bin, while this
        # check said confirmed). `result["placed"]` is what the skill actually
        # released; fall back to the snapshot only when it is absent.
        target = str(args.get("label") or "")
        held = str(
            result.get("placed")
            or result.get("object")
            or (before.get("label") if before.get("label") != target else "")
            or ""
        )
        if not held or held == target:
            pc.status = UNVERIFIED
            pc.evidence = (
                "cannot tell which object was released, so 'on target' is unverifiable"
            )
            return
        hp, channel = self._best_pose(held)
        tp, _ = self._best_pose(target) if target else (None, "")
        if hp is None or tp is None:
            pc.status = UNVERIFIED
            pc.evidence = f"could not re-locate {'the held object' if hp is None else target}"
            return
        pc.channel = channel
        dxy = _dist(hp[:2], tp[:2])
        above = hp[2] - tp[2]
        pc.measured = {"dxy_m": round(dxy, 4), "dz_m": round(above, 4)}
        if dxy <= SAME_PLACE_M * 2 and above > -LIFT_EPS_M:
            pc.status = CONFIRMED
            pc.evidence = f"{held} sits on {target} (offset {dxy*100:.1f} cm, dz {above*100:+.1f} cm)"
        else:
            pc.status = REFUTED
            pc.evidence = f"{held} is {dxy*100:.1f} cm from {target} (dz {above*100:+.1f} cm): not on it"

    def _check_moved(self, pc, args, result, before) -> None:
        label = str(args.get("label") or before.get("label") or "")
        want = float(args.get("distance_m", 0.08) or 0.0)
        pose, channel = self._best_pose(label) if label else (None, "")
        start = before.get("pose")
        if pose is None or start is None:
            pc.status, pc.evidence = UNVERIFIED, f"{label or 'object'} displacement not observed"
            return
        moved = _dist(pose[:2], start[:2])
        pc.channel, pc.measured = channel, {"moved_m": round(moved, 4), "requested_m": want}
        if moved >= max(want * PUSH_MIN_FRAC, 0.01):
            pc.status = CONFIRMED
            pc.evidence = f"{label} moved {moved*100:.1f} cm (asked {want*100:.0f} cm, {channel})"
        else:
            pc.status = REFUTED
            pc.evidence = f"{label} barely moved ({moved*100:.1f} cm of {want*100:.0f} cm requested)"

    def _check_gone(self, pc, args, result, before) -> None:
        label = str(args.get("label") or before.get("label") or "")
        pose, channel = self._best_pose(label) if label else (None, "")
        start = before.get("pose")
        if pose is None:
            pc.status, pc.channel = CONFIRMED, "belief"
            pc.evidence = f"{label} is no longer on the table"
            return
        pc.channel = channel
        if start is not None:
            moved = _dist(pose, start)
            pc.measured = {"moved_m": round(moved, 4)}
            pc.status = CONFIRMED if moved >= SAME_PLACE_M * 2 else REFUTED
            pc.evidence = (
                f"{label} travelled {moved*100:.0f} cm" if pc.status == CONFIRMED
                else f"{label} only travelled {moved*100:.0f} cm: the throw did not release"
            )
            return
        pc.status, pc.evidence = UNVERIFIED, f"{label} still tracked; displacement unknown"

    def _check_released(self, pc, args, result, before) -> None:
        frac = self._frac()
        if frac is None:
            pc.status, pc.evidence = UNVERIFIED, "gripper state unavailable"
            return
        pc.channel, pc.measured = "gripper", {"gripper_frac": frac}
        pc.status = CONFIRMED if frac > self.air_grasp_frac * 2 else REFUTED
        pc.evidence = (
            f"gripper open at {frac:.2f}: the object was released"
            if pc.status == CONFIRMED else f"gripper still closed at {frac:.2f}"
        )

    def _check_empty(self, pc, args, result, before) -> None:
        self._check_released(pc, args, result, before)

    def _check_closed(self, pc, args, result, before) -> None:
        frac = self._frac()
        if frac is None:
            pc.status, pc.evidence = UNVERIFIED, "gripper state unavailable"
            return
        pc.channel, pc.measured = "gripper", {"gripper_frac": frac}
        pc.status = CONFIRMED
        pc.evidence = (
            f"jaw at {frac:.2f}: closed on something"
            if frac > self.air_grasp_frac
            else f"jaw at {frac:.2f}: closed on empty air"
        )

    def _check_at_home(self, pc, args, result, before) -> None:
        pc.status, pc.channel = CONFIRMED, "arm"
        pc.evidence = "arm commanded home (streamed waypoints approved by the harness)"

    # ── helpers ──────────────────────────────────────────────────────────

    def _frac(self) -> float | None:
        if self._gripper_frac is None:
            return None
        try:
            v = self._gripper_frac()
            return None if v is None else float(v)
        except Exception:
            return None

    # ── reporting ────────────────────────────────────────────────────────

    def digest(self, last: int = 4) -> str:
        """Recent verified effects, for the agent context."""
        if not self.history:
            return ""
        rows = self.history[-last:]
        lines = ["Independently verified effects of your recent actions:"]
        lines += [f"- {pc.agent_line()}" for pc in rows]
        if any(pc.status == UNVERIFIED for pc in rows):
            lines.append(
                "UNVERIFIED means the effect could not be confirmed from an "
                "independent observation -- do NOT report it as done."
            )
        return "\n".join(lines)

    def contradictions(self) -> list[Postcondition]:
        """Calls that reported ok but whose effect was refuted."""
        return [pc for pc in self.history if pc.status == REFUTED]


def _dist(a, b) -> float:
    return sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)) ** 0.5


def annotate_result(result: dict, pc: Postcondition | None) -> dict:
    """Fold a postcondition into a skill result dict.

    A refuted postcondition **downgrades a reported success**: this is the
    single most important line in the module.  A skill that claimed ok while
    the world says otherwise becomes a failure the agent can react to, which
    is exactly the closed loop Pigey measures.
    """
    if pc is None:
        return result
    result["postcondition"] = pc.as_dict()
    if pc.refuted and result.get("ok"):
        result["ok"] = False
        result["error"] = f"postcondition failed: {pc.evidence}"
        result["self_reported_ok"] = True
    elif pc.status == UNVERIFIED and result.get("ok"):
        result["verified"] = False
        result["verification_note"] = pc.evidence
    elif pc.confirmed:
        result["verified"] = True
    return result
