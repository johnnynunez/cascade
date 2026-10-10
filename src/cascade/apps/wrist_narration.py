"""Opt-in "what the gripper sees" narration for the dashboard's wrist tile (B47).

ROADMAP "Wrist cam follow-ups": serve the wrist stream a narration highlight on
the dashboard. With `stream.wrist_narration: true` AND a rig stream whose
profile is a wrist view (`perception.camera_base.is_wrist_view`: `role: wrist`
or eye-in-hand extrinsics), `build_runtime` attaches a `WristNarrator` to the
runtime. While a motion skill runs the dashboard highlights THAT stream's tile
and shows ONE line under it; once the motion ends the line keeps its outcome.

The line is built only from state the runtime can vouch for:

* the motion skill and its target as dispatched (`SkillRuntime.execute`);
* the runtime's held-state: `held_object`, or the `_held_provisional` marker of
  a close that has not completed (which is never reported as a hold);
* the three-state postcondition (agent/effects.py) once the motion has one.

It never reads pixels (no detector or VLM claim about what is in the wrist
frame), never reads the arm (a dashboard poll must not touch a LazyArm or the
bus), and never upgrades evidence: a hold reads "unverified" unless a grasp
verdict exists for that same continuous hold, a refuted grasp reads refuted,
and a motion in progress claims no outcome. No wrist stream means no narrator
at all, so the dashboard never labels a front view as the gripper's.
"""

from __future__ import annotations

import threading

from ..agent.effects import CONFIRMED, REFUTED, UNVERIFIED

#: Motion skills that cannot START a new grasp. A grasp verdict recorded for
#: the current hold may keep qualifying that hold while one of these runs.
#: Every other motion skill -- grasp_object, pick_and_place, throw, handover,
#: sort_by_color, grasp_at_pixel, the Pigey composites, turn_screw, push,
#: reset_scene, and any motion skill added later -- drops the verdict at its
#: start: it may release and grasp again inside one call, and a verdict about
#: the old hold must never vouch for the new one. Absent = dropped (fail-safe).
CARRIES_GRASP_VERDICT = frozenset({
    "place_at", "place_on_object", "move_home", "move_relative",
    "point_at", "wave", "open_gripper", "close_gripper",
})

#: Longest label shown verbatim (quotes excluded); longer ones are cut.
MAX_LABEL_CHARS = 40

_OUTCOME_WORDS = {"ok": "motion finished", "failed": "failed", "stuck": "stuck"}


def _quote(text: str) -> str:
    text = " ".join(text.split())
    if len(text) > MAX_LABEL_CHARS:
        text = text[: MAX_LABEL_CHARS - 3] + "..."
    return f"'{text}'"


def motion_target(name: str, args: dict) -> str | None:
    """What the dispatched call names as its subject, as shown in the line:
    the quoted `label`/`object`/`query` argument (the same keys the runtime's
    postcondition snapshot reads), or `place_at`'s commanded point. None when
    the call names nothing (`move_home`, `wave`)."""
    for key in ("label", "object", "query"):
        value = args.get(key)
        if isinstance(value, str) and value.strip():
            return _quote(value)
    if name == "place_at":
        try:
            return f"({float(args['x']):.2f}, {float(args['y']):.2f})"
        except (KeyError, TypeError, ValueError):
            return None
    return None


def _verdict_words(status: str | None, channel: str) -> str:
    if status == CONFIRMED:
        return "grasp confirmed" + (f" by {channel}" if channel else "")
    if status == REFUTED:
        return "grasp refuted" + (f" by {channel}" if channel else "")
    return "grasp unverified"


class WristNarrator:
    """Records the motion in flight and renders the wrist tile's line.

    `begin`/`end` run on the dispatching thread (`SkillRuntime.execute`);
    `snapshot` runs on dashboard / MCP threads. One lock guards the record;
    the runtime's held-state attributes are read as plain attributes."""

    def __init__(self, camera: str):
        #: the rig stream name whose tile carries the highlight
        self.camera = str(camera)
        self._lock = threading.Lock()
        self._active: tuple[str, str | None] | None = None
        self._last: dict | None = None
        #: (held label, status, channel) of grasp_object's verdict on the hold it
        #: left; it qualifies the hold only while `held_object` is that label
        self._grasp: tuple[str | None, str, str] | None = None

    def begin(self, runtime, name: str, args: dict) -> bool:
        """A motion skill starts. False (and nothing recorded) when another
        motion already owns the line: a nested dispatch never replaces it."""
        with self._lock:
            if self._active is not None:
                return False
            if name not in CARRIES_GRASP_VERDICT:
                self._grasp = None          # this motion may release and grasp again
            self._active = (name, motion_target(name, dict(args or {})))
            return True

    def end(self, runtime, result) -> None:
        """The owning motion ended. `result` is None when dispatch raised."""
        with self._lock:
            if self._active is None:
                return
            name, target = self._active
            self._active = None
            held = getattr(runtime, "held_object", None)
            pc = result.get("postcondition") if isinstance(result, dict) else None
            pc = ({"kind": str(pc.get("kind") or ""), "status": str(pc.get("status") or UNVERIFIED),
                   "channel": str(pc.get("channel") or "")}
                  if isinstance(pc, dict) else None)
            if pc is not None and pc["kind"] == "holding":
                # grasp_object's own verdict, about the hold it leaves behind
                self._grasp = (held, pc["status"], pc["channel"])
            elif self._grasp is not None and self._grasp[0] != held:
                self._grasp = None          # that hold ended: released, slipped or replaced
            outcome = result.get("outcome") if isinstance(result, dict) else None
            self._last = {"skill": name, "target": target, "outcome": outcome, "postcondition": pc}

    def _wrist_frame(self, runtime) -> bool:
        streams = getattr(getattr(runtime, "rig", None), "streams", None) or {}
        stream = streams.get(self.camera)
        try:
            frame = stream.latest() if stream is not None else None
        except Exception:  # noqa: BLE001 -- a faulting camera reads as "no frame"
            return False
        return frame is not None

    def snapshot(self, runtime) -> dict:
        """The `wrist_view` entry of the dashboard / world state."""
        with self._lock:
            active, last, grasp = self._active, self._last, self._grasp
        held = getattr(runtime, "held_object", None) or None
        provisional = getattr(runtime, "_held_provisional", None)
        closing = provisional[0] if isinstance(provisional, tuple) and provisional else None
        verdict = grasp if grasp is not None and held is not None and grasp[0] == held else None
        if held is not None:
            gripper = "holding"
            status = verdict[1] if verdict is not None else UNVERIFIED
            grip = f"holding {_quote(str(held))} ({_verdict_words(status, verdict[2] if verdict else '')})"
        elif closing is not None:
            gripper, status = "closing", None
            grip = f"closing on {_quote(str(closing))} (grasp not complete, unverified)"
        else:
            gripper, status, grip = "not_holding", None, "not holding"
        current = active if active is not None else (
            (last["skill"], last["target"]) if last is not None else None)
        line = None
        if current is not None:
            head = current[0] + (f" {current[1]}" if current[1] else "")
            if active is not None:
                line = f"now: {head} · {grip}"
            else:
                done = _OUTCOME_WORDS.get(last["outcome"], "ended without a result")
                line = f"last: {head} {done} · {grip}"
                pc = last["postcondition"]
                # a holding verdict on a held object is already in the grip clause
                if pc is not None and not (pc["kind"] == "holding" and held is not None):
                    line += f" · {pc['kind']}: {pc['status']}"
                    if pc["channel"] and pc["status"] in (CONFIRMED, REFUTED):
                        line += f" ({pc['channel']})"
        frame = self._wrist_frame(runtime)
        if line is not None and not frame:
            line += " · wrist camera: no frame"
        return {
            "camera": self.camera,
            "active": active is not None,
            "skill": current[0] if current is not None else None,
            "target": current[1] if current is not None else None,
            "gripper": gripper,
            "held": held,
            "grasp_verdict": status,
            "outcome": last["outcome"] if active is None and last is not None else None,
            "postcondition": last["postcondition"] if active is None and last is not None else None,
            "wrist_frame": frame,
            "line": line,
        }


def build_narrator(runtime, stream_cfg) -> WristNarrator | None:
    """`stream.wrist_narration: true` -> a narrator on the rig's FIRST wrist
    stream (rig order); no wrist stream -> None, with one line saying why."""
    if not bool(stream_cfg.get("wrist_narration", False)):
        return None
    wrists = runtime._wrist_streams()
    if not wrists:
        import sys

        print("[cascade] stream.wrist_narration is on but no wrist camera is in this rig "
              "(no `role: wrist` / eye-in-hand profile): no wrist panel", file=sys.stderr)
        return None
    return WristNarrator(wrists[0][0])
