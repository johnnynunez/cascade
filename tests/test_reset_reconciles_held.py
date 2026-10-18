"""reset_scene on a real gripper reconciles the held flag before any motion.

An operator STOP late in the grasp lift (jaws already stalled on the object)
ends the grasp before its `held_object = label` line: only the close's
provisional marker (`_held_provisional`) says the jaws may hold something.
reset_scene on a real arm checked `held_object` alone, so it homed with the
object in the jaws, sent no jaw command, forgot every belief and returned
ok; a later open_gripper dropped the object from home height, the case #357
refuses. Now the real-gripper path reconciles the flag first, as every grasp
skill's entry does (`_reconcile_held`):

- jaws stalled on the object: the hold is promoted and the reset refuses
  with #357's text ("holding X: place it first"), before any command;
- jaws that cannot be read: the reset refuses too, before any command, and
  says the jaw state could not be read (fail closed);
- jaws that read open, or closed on air: nothing is held and the reset runs
  as with empty jaws (closed ones get their one opening at home).

A held flag is not reconciled: it refuses as before (#357).

Simulated and mock arms are unchanged.

Real `build_runtime` wiring (LazyArm -> SafeArm -> SafetyHarness of each
profile) over a MockArm that records every streamed arm command and every jaw
command, and the real grasp on the mock camera; no driver, no bus.
"""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import pytest

from cascade.control.mock_arm import MockArm
from cascade.types import MotionHalted

REFUSAL = "holding red cube: place it first (place_at), then reset"
UNREAD = ("jaw state could not be read: red cube may still be held after the interrupted grasp; "
          "check the gripper, then reset. Error: ")
#: the halt lands on this streamed command of the grasp lift (~100 commands
#: on rebot_rs): late in the lift, the jaws long stalled on the object
LATE_IN_LIFT = 90


class _RecArm(MockArm):
    """MockArm that logs every streamed arm command and every jaw command,
    runs `r.on_send()` after each arm command, and reads the jaws as unknown
    (`jaws = "unknown"`) or fails the whole read (`jaws = "error"`)."""

    def __init__(self, cfg, kinematics, r):
        super().__init__(cfg, kinematics)
        self.r = r
        self.jaws = "ok"

    def send_joint_target(self, q):
        self.r.log.append(("arm",))
        super().send_joint_target(q)
        if self.r.on_send is not None:
            self.r.on_send()

    def set_gripper(self, pos, effort=1.0):
        self.r.log.append(("jaw", float(pos), float(effort)))
        super().set_gripper(pos, effort)

    def get_state(self):
        if self.jaws == "error":
            raise TimeoutError("gripper feedback timed out")
        state = super().get_state()
        return dataclasses.replace(state, gripper_valid=False) if self.jaws == "unknown" else state


@pytest.fixture
def rig(monkeypatch, tmp_path):
    from cascade.apps import demo
    from cascade.apps.demo import shutdown_runtime
    from cascade.config import load_demo_config

    built = []
    monkeypatch.delenv("CASCADE_OBSERVED_FINGER_GATE", raising=False)

    def build(profile):
        cfg = load_demo_config(camera="mock", arm=profile, llm="mock")
        # rebot_rs requires its live camera map and conftest pins
        # CASCADE_OCCUPANCY=0: that combination refuses startup by design.
        occ = cfg._data.get("occupancy")
        if isinstance(occ, dict) and occ.get("required"):
            occ["required"] = False
        r = SimpleNamespace(cfg=cfg, log=[], arms=[], on_send=None)

        def make_arm(acfg, kinematics=None):
            arm = _RecArm(acfg, kinematics, r)
            r.arms.append(arm)
            return arm

        monkeypatch.setattr(demo, "make_arm", make_arm)
        r.runtime, arm = demo.build_runtime(cfg, tmp_path / profile, lazy_arm=True)
        built.append((r.runtime, arm))
        return r

    yield build
    for runtime, arm in built:
        shutdown_runtime(runtime, arm)


def _stop_late_in_the_lift(r, *, stall):
    """The real grasp of the red cube, halted on the LATE_IN_LIFT-th command
    of its lift. `stall`: the jaws stall at that fraction of their travel
    (on the object), None: they close fully (on air). Clears the log."""
    runtime = r.runtime
    runtime.arm.get_state()                      # materialize the LazyArm
    r.arms[0].object_stop_frac = stall
    carry = runtime._move_carry
    lift, stop = [], []

    def lift_spy(*args, **kwargs):
        lift.append(len(r.log))
        return carry(*args, **kwargs)

    def on_send():
        if lift and not stop and len(r.log) - lift[0] >= LATE_IN_LIFT:
            runtime.arm.harness.halt("operator: stop")
            stop.append(len(r.log))

    runtime._move_carry = lift_spy
    r.on_send = on_send
    out = runtime.execute("grasp_object", {"label": "red cube"})
    r.on_send = None
    del runtime._move_carry
    assert stop and len(lift) == 1, "the halt never reached the lift"
    assert out["ok"] is False and MotionHalted.__name__ in out["error"], out
    assert r.log[stop[0]:] == [], "commands after the halt"
    assert runtime.held_object is None and runtime._held_provisional is not None
    r.log.clear()


def _reset(r, call):
    return r.runtime.execute("reset_scene", {}) if call == "execute" else r.runtime.skill_reset_scene()


def _refused(out, error):
    assert out["ok"] is False and out["stage"] == "release" and out["home_skipped"] is True, out
    assert out["beliefs_forgotten"] == 0 and not out["observation_refreshed"], out
    assert out["error"].startswith(error), out


@pytest.mark.parametrize("call", ["execute", "direct"])
def test_jaws_stalled_on_the_object_after_a_stop_in_the_lift_refuse_the_reset(rig, call):
    r = rig("rebot_rs")
    _stop_late_in_the_lift(r, stall=0.5)
    out = _reset(r, call)
    _refused(out, REFUSAL)
    assert out["error"] == REFUSAL
    assert r.log == [], r.log                      # no arm, no jaw command
    rt = r.runtime
    assert rt.held_object == "red cube" and rt._held_provisional is None    # promoted
    assert rt._held_width_m == pytest.approx(0.5 * rt._max_width)


@pytest.mark.parametrize("jaws, why", [("unknown", "no valid jaw feedback"),
                                       ("error", "gripper feedback timed out")])
@pytest.mark.parametrize("call", ["execute", "direct"])
def test_unreadable_jaws_after_a_stop_in_the_lift_refuse_the_reset(rig, call, jaws, why):
    """Fail closed: the jaws may hold the object; nothing moves."""
    r = rig("rebot_rs")
    _stop_late_in_the_lift(r, stall=0.5)
    r.arms[0].jaws = jaws
    out = _reset(r, call)
    _refused(out, UNREAD)
    assert out["error"] == UNREAD + why
    assert len(out["error"]) <= 240, out["error"]  # the agent's failure line keeps 240 characters
    assert r.log == [], r.log
    assert r.runtime.held_object is None and r.runtime._held_provisional is not None


@pytest.mark.parametrize("call", ["execute", "direct"])
def test_jaws_closed_on_nothing_after_a_stop_in_the_lift_get_their_opening_at_home(rig, call):
    """Jaws that read closed (short of the air tolerance) hold nothing: the
    marker is dropped, the reset runs, and the closed empty jaws get the
    reset's one opening at home."""
    r = rig("rebot_rs")
    _stop_late_in_the_lift(r, stall=None)
    r.arms[0].set_gripper(r.cfg.arm.gripper.closed_pos)        # nothing between the jaws
    r.log.clear()
    out = _reset(r, call)
    assert out["ok"] and out["jaws_open_commanded"] is True, out
    jaws = [e[1:] for e in r.log if e[0] == "jaw"]
    assert jaws == [(pytest.approx(r.runtime._grip_open), 0.8)], jaws
    assert r.log[-1][0] == "jaw" and len(r.log) > 1  # home first, then the opening
    assert r.runtime.held_object is None and r.runtime._held_provisional is None


def test_jaws_opened_after_a_stop_in_the_lift_let_the_reset_run(rig):
    """Jaws at the open position hold nothing: the reset runs, no jaw command."""
    r = rig("rebot_rs")
    _stop_late_in_the_lift(r, stall=0.5)
    r.arms[0].set_gripper(r.cfg.arm.gripper.open_pos)          # opened by hand, not open_gripper
    r.log.clear()
    out = r.runtime.execute("reset_scene", {})
    assert out["ok"] and "jaws_open_commanded" not in out, out
    assert r.log and all(e[0] == "arm" for e in r.log), r.log
    assert r.runtime.held_object is None and r.runtime._held_provisional is None


def test_a_held_flag_refuses_as_it_is_whatever_the_jaws_read(rig):
    """Only a provisional hold is reconciled: a held flag keeps #357's
    refusal even on jaws that read closed (no slip is inferred here; on a
    handless mount every reading is "closed")."""
    r = rig("rebot_rs")
    r.runtime.arm.get_state()
    r.runtime.arm.set_gripper(r.cfg.arm.gripper.closed_pos)
    r.runtime.held_object, r.runtime._held_det_label = "red cube", "cube"
    r.log.clear()
    out = r.runtime.execute("reset_scene", {})
    _refused(out, REFUSAL)
    assert r.log == [] and r.runtime.held_object == "red cube", r.log


def test_the_sim_reset_after_the_same_stop_is_unchanged(rig):
    """A mock arm keeps today's reset: it homes and sends no jaw command
    (the provisional hold is neither promoted nor opened)."""
    r = rig("mock")
    _stop_late_in_the_lift(r, stall=0.5)
    out = r.runtime.execute("reset_scene", {})
    assert out["ok"] and "jaws_open_commanded" not in out, out
    assert r.log and all(e[0] == "arm" for e in r.log), r.log
    assert r.runtime.held_object is None and r.runtime._held_provisional is not None
