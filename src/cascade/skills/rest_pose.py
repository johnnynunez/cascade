"""Runtime side of `move_to_rest`: fold the arm onto its rest pose, verify it
there by measurement, then switch the motors off so the robot can be powered
down.

Two poses, two words -- the reBot owner's convention, and the routing the
tool descriptions and the reflex grammar both follow:

- the INITIAL / ready pose is the profile's `home_q`: raised, motors on,
  gripper forward, ready for work. That is `move_home` (reset_scene,
  calibration, the grasp pipeline and the reflexes keep using it).
- "home", "pose 0", "rest", "park", "fold" mean the profile's `park_q`: on
  the reBot every joint at 0, the arm folded back over its base with the
  links lying on their stops and the wrist camera on top. That is this
  module. From there the operator can switch the robot off.

Rules, each pinned by tests/test_move_to_rest.py:

1. Refusals come before any command. A latched e-stop means "do not move";
   a held object (or a possible one: the same retained-load markers the
   shutdown park checks) would be carried into the base and then left in
   jaws without torque -- the person is asked to put it down first. (An arm
   with no rest pose skips the load refusal: see the last paragraph.)

2. Every motion is a SafeArm call, so the harness vets every streamed step
   (joint limits, workspace, table, speed cap, occupancy, the wrist-camera-up
   rule). The route is the one the rig has proven
   (scripts/home_rebot_rs.py moved it rest -> home -> rest with every segment
   within 0.013 rad): up to the initial pose first unless the arm is already
   there (`move_planned`, the route `move_home` takes), then home -> rest in
   SEGMENTS joint-space steps, each MEASURED against its target. Only the
   last step relaxes the joint margin to 0, because joints 2/3 have their
   lower limit AT the rest pose; every earlier step keeps it.

3. The whole route is vetted with the harness's static API before the first
   command (and the fold again from the measured initial pose before it
   starts), so a fold the harness would stop half-way never starts. Unlike
   the teardown park (apps/demo.py `_park_move`), the occupancy map is never
   bypassed here: teardown has to choose between an unmapped park and
   dropping a raised arm, this tool can simply stay where it is with the
   motors on and say why.

4. Torque is released ONLY when every measured joint is within
   REST_TOL_RAD of `park_q`. Anything else -- a step that did not track, a
   halt, an e-stop, a joint that sagged short of its stop -- leaves the
   motors on (a raised arm without torque falls) and reports the measured
   pose.

5. Release mirrors the shutdown: close the gripper with the backend's
   torque-detected close (the reBot RS), then `LazyArm.disconnect()` -- the
   backend's own disconnect cuts torque and the LazyArm returns to standby.
   The SafeArm, its harness and any motion planner stay; the next motion
   re-opens the backend by itself. (On the RS, opening clears latched faults
   with a torque-off stop frame: harmless AT rest, which is the only place
   this module ever releases.) A backend that was opened directly (not a
   LazyArm, e.g. the demo CLI) cannot be re-opened by the runtime, and a
   simulator's drives are not motors to switch off: both keep their
   connection and say so.

An arm profile without `park_q` has no known folded pose and none is
guessed: the request falls back to the initial pose (`move_home`, a held
object carried along exactly as "go home" did before this tool existed) with
the motors on and `safe_to_power_off: false`.
"""

from __future__ import annotations

import time

import numpy as np

from ..types import SafetyViolation, SkillError

#: every measured joint must be this close to park_q before torque is cut
#: (rad). The rig's fold tracked within 0.013 rad; the RS reads +0.004 at the
#: joint-2 stop.
REST_TOL_RAD = 0.05
#: measured vs commanded after each step of the route (scripts/home_rebot_rs.py)
TRACK_TOL_RAD = 0.08
#: checked steps on the fold (scripts/home_rebot_rs.py)
SEGMENTS = 4
#: nominal seconds per fold step; SafeArm stretches it under the speed cap
SEGMENT_S = 2.5
#: the up leg's nominal duration (move_home's)
HOME_S = 3.0
#: wall budget for vetting the fold before it starts
_VET_BUDGET_S = 3.0

#: runtime attributes that mean the jaws hold (or may hold) something; the
#: same markers apps/demo.py `_park_arm` checks before parking
_RUNTIME_LOAD = (
    ("held_object", None),
    ("_held_provisional", "a possible grasp that is not confirmed or refuted yet"),
    ("_contact_episode", "an open grasp contact episode"),
    ("_carry_attachment", "a retained carry attachment"),
    ("_release_episode", "an unfinished release"),
)
_HARNESS_LOAD = (
    ("_pending_contact_episode", "an open grasp contact episode"),
    ("_pending_release_episode", "an unfinished release"),
)


# ── small helpers ───────────────────────────────────────────────────────────


def _bool(value, name: str, default: bool) -> bool:
    """A boolean tool argument. Chat hosts fill optional schema fields with
    null (= the default) and sometimes send "true"/"false" as strings."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lower() in ("true", "false"):
        return value.strip().lower() == "true"
    raise SkillError(f"{name} must be true or false, got {value!r}")


def _rounded(q) -> list[float] | None:
    return None if q is None else [round(float(v), 4) for v in q]


def _measured(arm) -> np.ndarray:
    return np.asarray(arm.get_state().q, dtype=float).reshape(-1)


def _err(q, target) -> float:
    return float(np.max(np.abs(np.asarray(q, dtype=float) - np.asarray(target, dtype=float))))


def retained_load(rt) -> str | None:
    """Why the jaws may hold something, or None."""
    for attr, what in _RUNTIME_LOAD:
        value = getattr(rt, attr, None)
        if value is not None:
            return f"it is holding {value!r}" if what is None else f"there is {what}"
    harness = rt.arm.harness
    for attr, what in _HARNESS_LOAD:
        if getattr(harness, attr, None) is not None:
            return f"there is {what}"
    return None


def rest_target(harness, n_joints: int) -> np.ndarray | None:
    """The arm profile's rest pose (`park_q`, set on the harness by
    apps/demo.py `_build_arm`), or None when the profile declares none."""
    park = getattr(harness, "park_q", None)
    if park is None:
        return None
    q = np.asarray(park, dtype=float).reshape(-1)
    if len(q) != int(n_joints) or not np.isfinite(q).all():
        raise SkillError(f"the arm profile's park_q {list(park)!r} does not fit this "
                         f"{int(n_joints)}-joint arm")
    return q


def fold_waypoints(start, park, segments: int = SEGMENTS) -> list[np.ndarray]:
    start, park = np.asarray(start, dtype=float), np.asarray(park, dtype=float)
    return [start + (park - start) * (k / segments) for k in range(1, segments + 1)]


def _margin(index: int, count: int) -> float | None:
    """Only the LAST fold step may drive onto the joint stop."""
    return 0.0 if index == count - 1 else None


def vet_fold(harness, start, park, rate_hz: float) -> tuple[list[np.ndarray], str | None]:
    """The fold's waypoints and the first harness objection, if any.

    `safety.trajectory.vet_segment` over each step's real executor edges,
    except that the last step is vetted with the relaxed joint margin it will
    be streamed with (vet_segment has no margin parameter)."""
    from ..control.motion_profile import nominal_profile
    from ..safety.trajectory import geometry_guard, motion_duration

    waypoints = fold_waypoints(start, park)
    deadline = time.monotonic() + _VET_BUDGET_S
    previous = np.asarray(start, dtype=float)
    with geometry_guard(harness, deadline=deadline):
        for index, goal in enumerate(waypoints):
            margin = _margin(index, len(waypoints))
            duration = motion_duration(harness, previous, goal, SEGMENT_S)
            for target in nominal_profile(previous, goal, duration, rate_hz, max_steps=10000):
                for q_prev, q_next, dt in target.checks:
                    if time.monotonic() >= deadline:
                        raise SkillError("the fold's safety preflight ran out of time; nothing moved")
                    reason = harness.vet_step(q_prev, q_next, dt, joint_margin=margin)
                    if reason:
                        return waypoints, f"fold step {index + 1}/{len(waypoints)}: {reason}"
            previous = goal
    return waypoints, None


def _rate(raw) -> float:
    from ..control.motion_profile import resolve_motion_rate

    return resolve_motion_rate(raw, None)


# ── results ─────────────────────────────────────────────────────────────────


def _held_by_earlier_session(raw) -> dict | None:
    """The hold marker of a standby arm an earlier session left powered and
    holding (control/arm_hold.py), or None. Never materializes the arm; a
    marker that cannot be read counts as held."""
    from ..control import arm_hold

    try:
        return arm_hold.held_marker(raw)
    except Exception as exc:  # noqa: BLE001 -- fail closed: unknown counts as held
        return {"reason": f"its hold could not be read ({exc})"}


def _refused(why: str, error: str, torque: str = "on") -> dict:
    return {"ok": False, "refused": why, "moved": False, "at": "unchanged",
            "torque": torque, "safe_to_power_off": False, "error": error}


def _not_at_rest(arm, start, park, route, why: str) -> dict:
    """The arm is not (verified) at rest: the motors stay on."""
    try:
        q = _measured(arm)
    except Exception:  # noqa: BLE001 -- report what is known, never cut torque
        q = None
    moved = bool(route) or q is None or _err(q, start) > 1e-3
    return {
        "ok": False, "at": "not_rest", "moved": moved, "route": list(route),
        "torque": "on", "safe_to_power_off": False,
        "measured_q": _rounded(q), "rest_q": _rounded(park),
        "max_error_rad": None if q is None else round(_err(q, park), 4),
        "rest_tolerance_rad": REST_TOL_RAD,
        "error": (f"{why}. The arm is NOT at its rest pose, so the motors stay ON (without "
                  "torque a raised arm falls). Do not tell the user it is safe to power off."),
    }


# ── the tool ────────────────────────────────────────────────────────────────


def move_to_rest(rt, *, keep_torque=False) -> dict:
    from ..control.lazy_arm import LazyArm

    keep_torque = _bool(keep_torque, "keep_torque", default=False)
    arm = rt.arm
    harness, raw = arm.harness, arm.raw
    lazy = isinstance(raw, LazyArm)
    held = _held_by_earlier_session(raw) if lazy and not raw.connected else None
    torque_now = "released" if lazy and not raw.connected and held is None else "on"

    if harness.estopped:
        return _refused("estopped", (
            "the e-stop is latched: nothing moved. The latch means 'do not move'; once the rig "
            "is safe, staff clear it (reset_stop) and ask again."), torque=torque_now)

    park = rest_target(harness, int(arm.n_joints))
    if held is not None:
        # Not standby: an earlier session's teardown left this arm POWERED
        # and HOLDING (control/arm_hold.py), and its connect would cut the
        # torque. Nothing is sent and nothing is promised (#426 review
        # Important 2: this answered "safe to power off").
        return {
            "ok": False, "at": "held", "moved": False, "torque": "on", "safe_to_power_off": False,
            "arm_state": "held", "refused": "arm_held", "rest_q": _rounded(park) if park is not None else None,
            "error": (f"an earlier session left this arm POWERED and HOLDING ({held.get('reason')}): "
                      "nothing was sent, the arm was not moved and its pose was not read. It is NOT safe "
                      "to switch the robot off: without torque a raised arm falls. Do not tell the user it "
                      "is safe to power off. Ask the operator to support the arm (they then switch its "
                      "power off, or confirm at the robot's own terminal so the next motion connects it); "
                      "an agent cannot confirm."),
        }

    if park is None:
        # No folded pose is known for this arm, so the motors will not be
        # released either: "home" keeps its pre-move_to_rest meaning here
        # (move_home, a held object carried along as before).
        result = dict(rt.skill_move_home())
        result.update(
            at="initial", moved=True, torque="on", safe_to_power_off=False, rest_pose=None,
            note=("this arm profile declares no rest pose (park_q), so none is guessed: the arm "
                  "went to its INITIAL pose and the motors stay on. Do not tell the user it is "
                  "safe to power it off."))
        return result

    load = retained_load(rt)
    if load is not None:
        return _refused("holding", (
            f"{load}: put it down first (place_at / place_on_object, or open_gripper over a "
            "safe spot), then ask again. Nothing moved; folding with a load would carry it "
            "into the base and leave it in jaws without torque."), torque=torque_now)

    if lazy and not raw.connected and not keep_torque:
        # Standby: cascade has not opened the motors since start (or since they
        # were released). Nothing holds the arm up, so nothing can drop -- and
        # reading its pose would switch the motors on for nothing.
        return {
            "at": "standby", "moved": False, "torque": "released", "safe_to_power_off": True,
            "arm_state": "standby", "rest_q": _rounded(park),
            "note": ("the motors are already off (standby): the arm was not moved and its pose "
                     "was not read. It is safe to switch the robot off. The next motion command "
                     "switches the motors on again by itself."),
        }

    home = rt._profile_q("home_q", "fold the arm to rest through the initial pose")
    start = _measured(arm)            # opens a standby LazyArm (keep_torque only)
    rate = _rate(raw)
    route: list[str] = []
    up = _err(start, park) > REST_TOL_RAD and _err(start, home) > TRACK_TOL_RAD
    fold = _err(start, park) > REST_TOL_RAD

    # Preflight the WHOLE route before the first command.
    if up:
        from ..safety.trajectory import plan_route

        try:
            plan_route(harness, start, home, HOME_S, rate_hz=rate)
        except SkillError as exc:
            return _refused("route", f"the way up to the initial pose was refused before "
                                     f"anything moved: {exc}. The motors stay on.")
    if fold:
        _, reason = vet_fold(harness, home if up else start, park, rate)
        if reason:
            return _refused("route", f"the fold to the rest pose was refused before anything "
                                     f"moved: {reason}. The motors stay on.")

    try:
        if up:
            move = getattr(arm, "move_planned", arm.move_joints)
            settled = move(home, duration_s=HOME_S)
            q = _measured(arm)
            if not settled or _err(q, home) > TRACK_TOL_RAD:
                return _not_at_rest(arm, start, park, route, (
                    f"the arm did not reach the initial pose (off by {_err(q, home):.3f} rad, "
                    f"settled={settled})"))
            route.append("initial_pose")
        if fold:
            qh = _measured(arm)
            waypoints, reason = vet_fold(harness, qh, park, rate)
            if reason:
                return _not_at_rest(arm, start, park, route, f"the fold was refused before it "
                                                      f"started: {reason}")
            for index, wp in enumerate(waypoints):
                last = index == len(waypoints) - 1
                settled = arm.move_joints(wp, duration_s=SEGMENT_S,
                                          joint_margin=_margin(index, len(waypoints)))
                if not last:
                    q = _measured(arm)
                    if not settled or _err(q, wp) > TRACK_TOL_RAD:
                        return _not_at_rest(arm, start, park, route, (
                            f"fold step {index + 1}/{len(waypoints)} did not track (off by "
                            f"{_err(q, wp):.3f} rad, settled={settled})"))
            route.append("fold")
    except (SafetyViolation, SkillError) as exc:
        return _not_at_rest(arm, start, park, route, f"the motion stopped: {type(exc).__name__}: {exc}")

    # THE guard: the motors go off only on a MEASURED rest pose.
    q = _measured(arm)
    err = _err(q, park)
    if err > REST_TOL_RAD:
        return _not_at_rest(arm, start, park, route, (
            f"measured {err:.3f} rad from the rest pose (tolerance {REST_TOL_RAD})"))

    out = {
        "at": "rest", "moved": bool(route), "route": route,
        "measured_q": _rounded(q), "rest_q": _rounded(park),
        "max_error_rad": round(err, 4), "rest_tolerance_rad": REST_TOL_RAD,
    }
    keep_reason = None
    if keep_torque:
        keep_reason = "kept on as asked (keep_torque)"
    elif not lazy:
        keep_reason = ("this arm was opened directly, not in standby mode, so the runtime "
                       "could not open it again; quit the program (its shutdown parks and "
                       "releases) before switching the robot off")
    elif getattr(raw, "disconnect_preserves_drive_state", False):
        keep_reason = "simulated arm: its drives are not motors to switch off"
    if keep_reason is None and harness.estopped:
        keep_reason = "the e-stop latched before the motors could be switched off"
    if keep_reason is not None:
        out.update(torque="on", safe_to_power_off=False, arm_state="energized",
                   note=f"at the rest pose; motors on: {keep_reason}.")
        return out

    # Release, as the shutdown does: jaws closed (torque-detected on the RS),
    # then the backend's disconnect cuts torque and the LazyArm goes standby.
    closer = getattr(raw, "close_gripper_torque", None)
    if callable(closer):
        try:
            closer()
            out["gripper"] = "closed"
        except Exception as exc:  # noqa: BLE001 -- at rest the jaws are not a safety matter
            out["gripper"] = f"close failed ({type(exc).__name__}: {exc}); left as it was"
    else:
        out["gripper"] = "left as it was (this arm has no torque-detected close)"
    try:
        raw.disconnect()
    except Exception as exc:  # noqa: BLE001
        out.update(ok=False, torque="unknown", safe_to_power_off=False,
                   error=f"switching the motors off failed: {type(exc).__name__}: {exc}")
        return out
    if raw.connected:
        out.update(ok=False, torque="unknown", safe_to_power_off=False,
                   error="the arm did not return to standby after the release")
        return out
    out.update(torque="released", safe_to_power_off=True, arm_state="standby",
               note=("the arm is folded on its rest pose and the motors are off: it is safe to "
                     "switch the robot off now. The next motion command switches the motors on "
                     "again by itself, from this pose."))
    rt.memory.add("note", "arm folded to its rest pose; motors released (safe to power off)")
    return out
