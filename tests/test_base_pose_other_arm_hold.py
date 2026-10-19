"""base_pose pairs: one arm never takes, clears or acts through another
arm's hold (spec D9, R5; Task 5 review Important 1, Minors 1, 2 and 6).

Moved out of tests/test_base_pose_runtime_frame.py (its section 7b, Task 5 fix
review Minor 4). On the shipped so101 pair, while arm X's grasp stopped after
its close (a provisional hold), arm Y's grasp entries and composites are
refused before any perception or command, naming X and the way out on it;
"already holding" names the holding arm; reset_scene on Y is refused while X
keeps a live stop column (and names the fallback when X's move_home is
refused too); the grasp-outcome memory rows are per arm on a pair.
"""

from __future__ import annotations

import dataclasses
import json
import re
from types import SimpleNamespace

import pytest
from base_pose_oracle import (
    make_runtime,  # noqa: F401  (the fixture, by name)
    moved_cfg,
    raw_arm,
)
from test_base_pose_runtime_frame import (  # noqa: F401  (gate_off, observed_gate: fixtures by name)
    HELD_TEXT,
    _in_bound_call,
    _interrupted_right_grasp,
    _JawLog,
    _kept_column,
    _pair_world_rig,
    _right,
    _world_object_rig,
    gate_off,
    observed_gate,
)

#: No arm command and no jaw command on either arm of the pair.
_NOTHING_SENT = ({"so101_left": 0, "so101_right": 0}, {"so101_left": 0, "so101_right": 0})
_OTHER_ARM = {"so101_left": "so101_right", "so101_right": "so101_left"}
#: The pair's table-frame camera shift that puts the cube in an arm's reach
#: (W (0.198, +/-0.10): each arm reaches only the cube on its own side).
_REACHABLE_SHIFT = {"so101_left": (0.0, 0.10), "so101_right": (0.0, -0.10)}


def _pair_reaching(arm_name):
    return moved_cfg(["so101_left", "so101_right"], {}, move_cameras=False, shift_xy=_REACHABLE_SHIFT[arm_name])


def _stalled_provisional(rt, name, label="green block"):
    """Arm `name`'s grasp stopped after its close (the review's recipe): a
    `ProvisionalHold` its close left, its jaws stalled half open on it."""
    from cascade.skills.runtime import ProvisionalHold

    safe = rt.arm_rig.get(name)
    raw = raw_arm(safe)
    raw.object_stop_frac = None
    raw._gripper = raw._grip_closed + 0.5 * (raw._grip_open - raw._grip_closed)
    rt._held_provisional = ProvisionalHold((label, label, "green"), commanded_open=0.0, expected_open=0.5,
                                           arm=safe)
    return rt._held_provisional


def _perception_log(rt, monkeypatch):
    """Every skill-side perception from here on: `_localize` (the label
    grasps) and `_frame_in_arm` (grasp_at_pixel's frame). execute() itself
    still observes for its pre-state snapshot."""
    looked = []
    for name in ("_localize", "_frame_in_arm"):
        real = getattr(rt, name)
        monkeypatch.setattr(rt, name, lambda *a, _real=real, _name=name, **k: (looked.append(_name), _real(*a, **k))[1])
    return looked


def test_a_grasp_on_another_arm_is_refused_while_a_provisional_hold_stands(make_runtime, gate_off, monkeypatch):  # noqa: F811
    """Task 5 review Important 1 (R5): the right arm's grasp stopped after its
    close leaves a provisional hold. A grasp on the LEFT arm would take the
    one provisional slot with its own close, and its promotion (or air
    verdict) would clear it: the right arm's next reset_scene then opened its
    jaws on the cube at home. It is refused before any command or perception,
    naming the right arm and the way out on it; the provisional hold stays."""
    rt = make_runtime(_pair_world_rig())
    prov = _interrupted_right_grasp(rt, monkeypatch)
    log = _JawLog(rt, monkeypatch)
    looked = _perception_log(rt, monkeypatch)
    mark = log.mark()
    result = rt.execute("grasp_object", {"label": "red cube", "arm": "so101_left"})
    assert result["ok"] is False, result
    assert "'red cube' may be held by arm 'so101_right'" in result["error"], result
    assert "not 'so101_left'" in result["error"] and "settle it with arm='so101_right' first" in result["error"]
    assert "then grasp_object; no motion was sent" in result["error"], result
    assert len(result["error"]) <= 240, len(result["error"])
    assert json.dumps(result, default=str).count("this attempt FAILED") <= 1
    assert looked == [], looked
    assert log.since(mark) == _NOTHING_SENT
    assert rt._held_provisional is prov and not rt.held_object


@pytest.mark.parametrize("skill,args", [
    ("grasp_object", {"label": "red cube"}),
    ("grasp_at_pixel", {"u": 0.5, "v": 0.5, "normalized": True}),
    ("pick_and_place", {"object": "red cube"}),
])
@pytest.mark.parametrize("y_arm", ["so101_left", "so101_right"])
def test_a_grasp_skill_never_takes_another_arms_provisional_slot(make_runtime, observed_gate, monkeypatch,  # noqa: F811
                                                                y_arm, skill, args):
    """Both directions, every grasp entry: arm X's provisional hold stands, arm
    Y reaches the cube; Y's grasp skill is refused before any perception or
    command, the hold stays X's."""
    x_arm = _OTHER_ARM[y_arm]
    rt = make_runtime(_pair_reaching(y_arm))
    observed_gate(rt)
    prov = _stalled_provisional(rt, x_arm)
    log = _JawLog(rt, monkeypatch)
    looked = _perception_log(rt, monkeypatch)
    mark = log.mark()
    result = rt.execute(skill, {**args, "arm": y_arm})
    assert result["ok"] is False, result
    assert f"'green block' may be held by arm {x_arm!r}" in result["error"], result
    assert f"then {skill}; no motion was sent" in result["error"], result
    assert looked == [] and log.since(mark) == _NOTHING_SENT
    assert rt._held_provisional is prov and not rt.held_object


@pytest.mark.parametrize("real", [False, True], ids=["sim gripper", "real gripper"])
@pytest.mark.parametrize("y_arm", ["so101_left", "so101_right"])
def test_the_stopped_arm_never_opens_on_its_object_after_another_arms_grasp(make_runtime, gate_off, monkeypatch,  # noqa: F811
                                                                            y_arm, real):
    """The review's 4 reproduced rows (b9bf044: Y grasped and placed, then
    X's reset_scene reported `jaws_open_commanded` and opened X's jaws at home,
    on a real gripper too): Y's grasp is refused, so X's reset never opens
    X's jaws: the sim reset keeps them shut (a provisional hold may be in
    them), the real-gripper reset refuses (the stalled jaws hold it)."""
    x_arm = _OTHER_ARM[y_arm]
    rt = make_runtime(_pair_reaching(y_arm))
    prov = _stalled_provisional(rt, x_arm)
    log = _JawLog(rt, monkeypatch)
    assert rt.execute("grasp_object", {"label": "red cube", "arm": y_arm})["ok"] is False
    monkeypatch.setattr(rt, "_drives_real_gripper", lambda: real)
    mark = log.mark()
    reset = rt.execute("reset_scene", {"arm": x_arm})
    assert reset.get("jaws_open_commanded") is not True, reset
    assert log.since(mark)[1] == {"so101_left": 0, "so101_right": 0}, reset
    if real:
        assert reset["ok"] is False and "holding green block" in reset["error"], reset
        assert rt.held_object == "green block" and rt._held_arm == x_arm   # the stalled jaws: promoted
    else:
        assert rt._held_provisional is prov and not rt.held_object


def test_a_vla_close_records_its_arm(make_runtime, gate_off, monkeypatch):  # noqa: F811
    """The VLA executor's close leaves a provisional hold too: it carries the
    arm whose close left it (a plain tuple counted as every arm's, so the
    left arm's jaws judged and cleared it), and the left arm's grasp is then
    refused before any command."""
    from cascade.grasping import vla_executor
    from cascade.types import SkillError

    rt = make_runtime(_pair_world_rig())
    right = _right(rt)

    class _Client:
        def close(self):
            pass

    class _Episode:
        def connect(self):
            return _Client()

        def run(self, runtime, client, *, provisional, **kw):
            runtime._held_provisional = provisional                       # the close (vla_executor.run)
            raise SkillError("VLA chunk deadline (test)")

    monkeypatch.setattr(vla_executor, "executor_name", lambda gcfg: "vla")
    monkeypatch.setattr(vla_executor, "route_refusal", lambda runtime: None)
    rt.vla_executor = _Episode()
    result = rt.execute("grasp_object", {"label": "red cube", "arm": "so101_right"})
    assert result["ok"] is False and "VLA chunk deadline (test)" in result["error"], result
    prov = rt._held_provisional
    assert prov is not None and prov[0] == "red cube" and prov.arm is right
    log = _JawLog(rt, monkeypatch)
    mark = log.mark()
    refused = rt.execute("grasp_object", {"label": "red cube", "arm": "so101_left"})
    assert refused["ok"] is False and "may be held by arm 'so101_right'" in refused["error"], refused
    assert log.since(mark) == _NOTHING_SENT and rt._held_provisional is prov


@pytest.mark.parametrize("skill", ["handover", "sort_by_color", "restore_scene", "search_for_object", "throw"])
def test_the_composite_grasps_never_take_another_arms_provisional_slot(make_runtime, gate_off, monkeypatch, skill):  # noqa: F811
    """Task 6's composites grasp through skill_grasp_object or the persistence
    loop (whose retries re-home); on an arm without a base_pose they run
    (called directly here, past execute()'s world-target guard), and each is
    refused before any command while another arm's provisional hold stands."""
    from cascade.types import SkillError

    rt = make_runtime(_pair_world_rig())
    prov = _stalled_provisional(rt, "so101_right")
    calls = {"handover": lambda: rt.skill_handover("red cube"), "sort_by_color": lambda: rt.skill_sort_by_color(),
             "restore_scene": lambda: rt.skill_restore_scene("scene"),
             "search_for_object": lambda: rt.skill_search_for_object("red cube"),
             "throw": lambda: rt.skill_throw("red cube")}
    log = _JawLog(rt, monkeypatch)
    mark = log.mark()

    def refused():
        with pytest.raises(SkillError, match=re.escape("may be held by arm 'so101_right'")):
            calls[skill]()
        return True

    assert _in_bound_call(rt, "so101_left", refused)
    assert log.since(mark) == _NOTHING_SENT and rt._held_provisional is prov


@pytest.mark.parametrize("skill,args", [
    ("grasp_object", {"label": "red cube"}),
    ("grasp_at_pixel", {"u": 0.5, "v": 0.5, "normalized": True}),
    ("restore_scene", {"name": "scene"}),
    ("search_for_object", {"label": "red cube"}),
])
def test_already_holding_on_the_other_arm_names_the_holder(make_runtime, gate_off, monkeypatch, skill, args):  # noqa: F811
    """Task 5 review Minor 2: on the arm that does not hold the cube, the
    "already holding" refusal names the holding arm and the way out on it
    (it used to read "already holding 'red cube'; place it first", which a
    place on this arm then refused)."""
    from cascade.types import SkillError

    rt = make_runtime(_pair_world_rig())
    assert rt.execute("grasp_object", {"label": "red cube", "arm": "so101_right"})["held"] == "red cube"
    if skill == "restore_scene":
        assert rt.execute("snapshot_scene", {"name": "scene"}).get("ok", True) is True
    log = _JawLog(rt, monkeypatch)
    mark = log.mark()
    if skill in ("grasp_object", "grasp_at_pixel"):
        error = rt.execute(skill, {**args, "arm": "so101_left"})["error"]
    else:                                                                 # Task 6 skills: past the guard
        def refused():
            with pytest.raises(SkillError) as raised:
                getattr(rt, f"skill_{skill}")(**args)
            return str(raised.value)

        error = _in_bound_call(rt, "so101_left", refused)
    assert "'red cube' " + HELD_TEXT + ", not 'so101_left'" in error, error
    assert f"place it with arm='so101_right' first (place_at), then {skill}; no motion was sent" in error, error
    assert log.since(mark) == _NOTHING_SENT
    assert (rt.held_object, rt._held_arm) == ("red cube", "so101_right")


@pytest.mark.parametrize("state", ["provisional", "held"])
def test_a_long_label_never_cuts_the_refusals_way_out(make_runtime, gate_off, state):  # noqa: F811
    """The refusal (with execute()'s "SkillError: ") fits the 240 characters
    the agent's failure line quotes: the label is shortened, never the way out."""
    rt = make_runtime(_pair_world_rig())
    label = "the long-named " + "x" * 80 + " box"
    if state == "provisional":
        _stalled_provisional(rt, "so101_right", label=label)
    else:
        rt.held_object, rt._held_arm = label, "so101_right"
    error = rt.execute("grasp_object", {"label": "red cube", "arm": "so101_left"})["error"]
    assert len(error) <= 240 and error.endswith("then grasp_object; no motion was sent"), (len(error), error)
    assert error.startswith("SkillError: 'the long-named"), error


def test_the_possible_hold_refusal_names_the_arm_whose_close_left_it(make_runtime, gate_off):  # noqa: F811
    """#379's entry refusal (`_possible_hold_refusal`, the skills that open the
    jaws first) covers the running arm's OWN possible hold only (#387 review
    Minor 2): another arm's provisional hold is refused, naming that arm and
    the way out on it, by `_refuse_on_another_arms_provisional` (the grasp
    entries, turn_screw, handover), never with this arm's "my jaws" ask."""
    rt = make_runtime(_pair_world_rig())
    _stalled_provisional(rt, "so101_right", label="red cube")
    assert _in_bound_call(rt, "so101_left", rt._possible_hold_refusal) is None
    # on the arm whose close left it: #379's own refusal (its jaws, unreconciled here)
    assert _in_bound_call(rt, "so101_right", rt._possible_hold_refusal).startswith("the jaw state could not be read")


@pytest.mark.parametrize("skill", ["turn_screw", "handover"])
def test_turn_screw_and_handover_refuse_another_arms_possible_hold(make_runtime, gate_off, monkeypatch, skill):  # noqa: F811
    """Both open the jaws first and close later: another arm's provisional
    hold refuses them before any command, a SkillError naming that arm and
    the way out on it (main's text), never STUCK with this arm's ask."""
    from cascade.types import SkillError, SkillStuck

    rt = make_runtime(_pair_world_rig())
    prov = _stalled_provisional(rt, "so101_right", label="red cube")
    log = _JawLog(rt, monkeypatch)
    mark = log.mark()

    def refused():
        with pytest.raises(SkillError) as raised:
            getattr(rt, f"skill_{skill}")("red cube")
        return raised.value

    error = _in_bound_call(rt, "so101_left", refused)
    assert not isinstance(error, SkillStuck), (type(error), error)
    assert str(error).startswith("'red cube' may be held by arm 'so101_right' (stopped after its close), "
                                 "not 'so101_left'"), error
    assert str(error).endswith("then try again; no motion was sent") and len(f"SkillError: {error}") <= 240, error
    assert log.since(mark) == _NOTHING_SENT and rt._held_provisional is prov


def test_close_gripper_on_the_other_arm_with_its_jaws_unread_closes(make_runtime, gate_off, monkeypatch):  # noqa: F811
    """#387 review Minor 2: so101_right's grasp stopped after its close; on
    so101_left, whose own jaws cannot be read and hold nothing, close_gripper
    went STUCK with "my jaws may still hold an object" and its ask
    (open_gripper on so101_left, ask again) looped on the same STUCK. Only
    so101_right's jaws may hold the cube: so101_left closes as on main, and
    the hold stays so101_right's."""
    rt = make_runtime(_pair_world_rig())
    prov = _interrupted_right_grasp(rt, monkeypatch)
    left = raw_arm(rt.arm_rig.get("so101_left"))
    real = type(left).get_state.__get__(left)
    monkeypatch.setattr(left, "get_state", lambda: dataclasses.replace(real(), gripper_valid=False))
    log = _JawLog(rt, monkeypatch)
    for _ in range(2):                                   # the re-ask too
        mark = log.mark()
        result = rt.execute("close_gripper", {"arm": "so101_left"})
        assert result["ok"] is True and result.get("outcome") != "stuck", result
        assert log.since(mark) == ({"so101_left": 0, "so101_right": 0}, {"so101_left": 2, "so101_right": 0})
        assert rt._held_provisional is prov and not rt.held_object


def test_already_holding_on_a_single_arm_reads_as_before(make_runtime, gate_off):  # noqa: F811
    rt = make_runtime(_world_object_rig())
    assert rt.execute("grasp_object", {"label": "red cube"})["held"] == "red cube"
    result = rt.execute("grasp_object", {"label": "red cube"})
    assert result["error"] == "SkillError: already holding 'red cube'; place it first", result


def _column_refusal(seconds_left):
    """reset_scene on so101_left while so101_right keeps a live stop column."""
    return ("arm 'so101_right' keeps its approach column after a stop, not 'so101_left': call move_home with "
            f"arm='so101_right', then reset_scene. If refused, the column expires in {seconds_left} s, or any "
            "other motion skill on that arm drops it; no motion was sent")


def test_reset_scene_is_refused_while_another_arms_stop_column_is_live(make_runtime, gate_off, monkeypatch):  # noqa: F811
    """Task 5 review Minor 1: the right arm was stopped below its pregrasp
    before its close (no hold, no provisional) and keeps its approach column.
    reset_scene on the LEFT arm would forget it (the scene changes for every
    arm), and the right arm's move_home could no longer retrace up it: the
    reset is refused before any motion, naming the way out. An expired column
    no longer applies (the reset runs and forgets it); the running arm's own
    live column is retraced by its reset as before (never this refusal)."""
    from cascade.skills import stop_retrace

    rt = make_runtime(_pair_world_rig())
    left, right = rt.arm_rig.get("so101_left"), _right(rt)
    column = _kept_column(right)
    rt._stop_retraces = {id(right): column}
    log = _JawLog(rt, monkeypatch)
    mark = log.mark()
    result = rt.execute("reset_scene", {"arm": "so101_left"})
    assert result["ok"] is False and result["home_skipped"] is True, result
    # Task 5 fix review Minors 1 and 2: verbatim, with the way out when that
    # arm's move_home is refused too (its retrace refused: the column still
    # expires, and any other motion skill on that arm forfeits it), at stage home
    assert result["error"] == _column_refusal(300), result
    assert result["stage"] == "home", result
    assert len(result["error"]) <= 240
    assert log.since(mark) == _NOTHING_SENT
    assert rt._stop_retraces == {id(right): column}
    rt._stop_retraces = {id(right): _kept_column(right, age_s=100.25)}    # the time left, rounded up
    assert rt.execute("reset_scene", {"arm": "so101_left"})["error"] == _column_refusal(200)
    rt._stop_retraces = {id(right): column}
    rt._stop_retraces = {id(right): _kept_column(right, age_s=stop_retrace.MAX_AGE_S + 1.0)}
    result = rt.execute("reset_scene", {"arm": "so101_left"})
    assert result.get("home_skipped") is not True, result                 # it got as far as its home move
    assert rt._stop_retraces == {}, result
    rt._stop_retraces = {id(left): _kept_column(left)}
    result = rt.execute("reset_scene", {"arm": "so101_left"})
    assert "keeps its approach column" not in str(result.get("error")), result


def test_another_arms_column_comes_before_its_hold_in_the_reset_refusal(make_runtime, gate_off, monkeypatch):  # noqa: F811
    """The right arm holds the cube AND keeps a live column: the left reset
    names move_home on the right arm first; the hold refusal's place_at would
    forfeit that column (execute() forgets the running arm's column on any
    other motion skill) and freeze the right arm at its object."""
    rt = make_runtime(_pair_world_rig())
    right = _right(rt)
    assert rt.execute("grasp_object", {"label": "red cube", "arm": "so101_right"})["held"] == "red cube"
    rt._stop_retraces = {id(right): _kept_column(right)}
    log = _JawLog(rt, monkeypatch)
    mark = log.mark()
    result = rt.execute("reset_scene", {"arm": "so101_left"})
    assert result["ok"] is False and result["error"] == _column_refusal(300), result
    assert "place_at" not in result["error"] and log.since(mark) == _NOTHING_SENT


def test_one_arms_grasp_failures_never_raise_the_other_arms_grasp(make_runtime, gate_off):  # noqa: F811
    """Task 5 review Minor 6: the grasp-outcome memory is keyed by object, and
    "pregrasp IK failed" counts as `pregrasp_unsafe` (+8 mm of grasp height
    per loss). The cube at W (0.198, -0.10) is out of the left arm's reach:
    two left attempts fail at IK; the right arm's grasp of the same cube must
    not inherit that nudge (on a pair the memory rows are per arm). A single
    arm keeps the bare object rows."""
    rt = make_runtime(_pair_world_rig())
    for _ in range(2):
        failed = rt.execute("grasp_object", {"label": "red cube", "arm": "so101_left"})
        assert failed["ok"] is False and "IK failed" in failed["error"], failed
    result = rt.execute("grasp_object", {"label": "red cube", "arm": "so101_right"})
    assert result.get("held") == "red cube", result
    keys = sorted(rt.grasp_memory._profiles)
    assert keys and all(k.split("|")[0] in ("red_cube@so101_left", "red_cube@so101_right") for k in keys), keys
    air = make_runtime(_pair_world_rig())                                 # an air grasp's loss: the arm's row too
    raw_arm(_right(air)).object_stop_frac = None
    assert "air grasp" in air.execute("grasp_object", {"label": "red cube", "arm": "so101_right"})["error"]
    assert [k.split("|")[0] for k in air.grasp_memory._profiles] == ["red_cube@so101_right"]
    single = make_runtime(_world_object_rig())
    assert single.execute("grasp_object", {"label": "red cube"})["held"] == "red cube"
    assert [k.split("|")[0] for k in single.grasp_memory._profiles] == ["red_cube"]


def test_the_grasp_memory_rows_are_read_per_arm(make_runtime, observed_gate):  # noqa: F811
    """The rows each arm learned are the rows it reads: the preview's prior,
    the grasp's z nudge (gate off: read in planning; gate on: the frozen
    prior). The right arm has 2 losses of its own (+16 mm), the left 4 and
    the bare object row 6 (rows a single arm, or the old key, wrote)."""
    from cascade.memory.grasp_memory import object_profile

    rt = make_runtime(_pair_world_rig())
    observed_gate(rt)
    fix = _in_bound_call(rt, "so101_right", lambda: rt._localize("red cube")[1])
    for label, losses in (("red cube@so101_right", 2), ("red cube@so101_left", 4), ("red cube", 6)):
        for _ in range(losses):
            rt.grasp_memory.record(label, fix, None, success=False, reason="pregrasp unsafe (test)")
    assert object_profile("red cube@so101_right", fix.extent) in rt.grasp_memory._profiles
    preview = rt.execute("preview_grasp", {"label": "red cube", "arm": "so101_right"})
    assert preview["memory_prior"]["seen"] == 2, preview
    rt.execute("grasp_object", {"label": "red cube", "arm": "so101_right"})
    assert rt._last_grasp_z_nudge == pytest.approx(0.016, abs=1e-9)


def _pair_double(rt):
    """`rt` (a runtime double) as the right arm of a pair: its rig holds a
    second arm, so the memory rows are keyed by the running arm's name."""
    rt.arm_rig = SimpleNamespace(arms={"so101_left": object(), "so101_right": rt.arm})


def test_the_planner_reads_and_reranks_with_the_running_arms_row(monkeypatch):
    """`_plan_grasps` reads the prior and re-ranks with the running arm's row
    (tests/test_grasping.py's double, on a pair)."""
    from unittest.mock import Mock

    from test_grasping import make_fix

    from cascade.config import load_demo_config
    from cascade.skills.runtime import SkillRuntime

    rt = SkillRuntime.__new__(SkillRuntime)
    rt.cfg = load_demo_config(arm="fr3", camera="d435i", llm="mock")
    rt._max_width = .08
    rt.memory = Mock()
    rt.grasp_memory = Mock()
    rt.grasp_memory.prior.return_value = {"nudges": {"grasp_z_delta": 0.0}, "profile": "wood",
                                          "wins": 1, "losses": 0, "success_rate": 1.}
    rt.grasp_memory.rerank.side_effect = lambda grasps, *args: list(grasps)
    rt._arm = SimpleNamespace()
    _pair_double(rt)
    monkeypatch.setattr("cascade.perception.support_surface.estimate_support_z", lambda frame, fix: 0.)
    rt._plan_grasps(make_fix(size=(.10, .03, .035)), "wood piece")
    assert rt.grasp_memory.prior.call_args[0][0] == "wood piece@so101_right"
    assert rt.grasp_memory.rerank.call_args[0][1] == "wood piece@so101_right"


def test_the_frozen_planning_prior_is_the_running_arms_row(monkeypatch):
    """The observed-finger planning search (gate on, GraspGen-X feasibility
    batches) freezes one prior per attempt: on a pair, the running arm's row;
    the attempt's outcome is recorded there too (tests/test_grasp_planning_search.py's double)."""
    from test_grasp_planning_search import planning_fixture

    rt, calls, fix, frame, h, batches, opts, prior, reads, records, _ = planning_fixture(monkeypatch)
    labels = []
    rt.grasp_memory = SimpleNamespace(prior=lambda label, f: (labels.append(label), prior)[1],
                                      record=lambda *a, **kw: records.append(a[0]))
    _pair_double(rt)
    h.vet_pose = lambda *a, **kw: None
    rt.skill_grasp_object("orange", _fix=fix, _frame=frame)
    assert batches == [1] and labels == ["orange@so101_right"], labels
    assert records == ["orange@so101_right"], records

# ── #400 review Important 1: the arm that does not hold moves with its own empty jaws ──


def _pair_without_wrist_box():
    """`_pair_reaching("so101_right")` with `relative_motion.wrist_camera: null`:
    the so101's TCP approach is parallel to the default camera-up axis, so with
    the shipped box every spatial move on it raises before it is vetted."""
    cfg = _pair_reaching("so101_right")
    cfg._data["relative_motion"]["wrist_camera"] = None
    return cfg


def _loads_planned(rt, monkeypatch):
    """The `load` of every route the spatial moves plan from here on."""
    from cascade.skills import spatial_moves

    loads, real = [], spatial_moves.plan_route
    monkeypatch.setattr(spatial_moves, "plan_route",
                        lambda *a, _real=real, **k: (loads.append(k.get("load")), _real(*a, **k))[1])
    return loads


def _no_hold_claimed(result, loads):
    text = json.dumps(result, default=str)
    assert result.get("holding") is None and result.get("speed") in (None, "normal"), result
    assert "held_object_bottom_below_gripper_m" not in result, result
    for words in ("held object", "I am holding", "olding 'red cube'"):
        assert words not in text, (words, result)
    assert all(load is not None and load.label is None and load.drop_m == 0.0 for load in loads), loads


#: Spatial moves on so101_left, by rig: on the pair without the wrist box each
#: completes (or is refused by reach) the same with nothing held anywhere; on
#: the shipped pair, the review's probe H rows.
_EMPTY_ARM_MOVES = {
    "just above the table": ("move_near_object", {"label": "table", "relation": "just_above"}),
    "closer to the cube": ("move_near_object", {"label": "red cube", "relation": "closer", "distance_m": 0.03}),
    "down": ("move_relative", {"direction": "down", "frame": "base", "distance_m": 0.02}),
    "left": ("move_relative", {"direction": "left", "frame": "base", "distance_m": 0.02}),
    "roll": ("rotate_gripper", {"direction": "clockwise", "angle_deg": 10}),
    "above the table": ("move_near_object", {"label": "table", "relation": "above"}),
}
_EMPTY_ARM_ROWS = [("no wrist box", c) for c in ("just above the table", "closer to the cube", "down", "left", "roll")] \
    + [("shipped pair", c) for c in ("above the table", "closer to the cube")]
_REPORTED = ("ok", "moved", "holding", "speed", "interpretation", "summary", "error", "refused_by")


@pytest.mark.parametrize("rig,call", _EMPTY_ARM_ROWS, ids=[f"{r}: {c}" for r, c in _EMPTY_ARM_ROWS])
def test_the_arm_that_does_not_hold_moves_as_if_nothing_were_held(make_runtime, gate_off, monkeypatch,  # noqa: F811
                                                                   rig, call):
    """#400 review Important 1 (spec D9: the held offset belongs to the
    grasping arm): so101_right holds the cube. A spatial move on so101_left
    read the rig-wide hold as its own: a phantom load 6 cm under its empty
    jaws refused moves by "the held-object clearance check" or by reach,
    slowed the rest and reported "holding 'red cube'" ("the held object's
    bottom 5.0 cm over its top"); a move toward the cube was refused with "I
    am holding the 'red cube'". so101_left now sends the commands and reports
    what the same move gives with nothing held; the hold stays so101_right's."""
    skill, args = _EMPTY_ARM_MOVES[call]
    runs = []
    for hold in (False, True):
        rt = make_runtime(_pair_without_wrist_box() if rig == "no wrist box" else _pair_reaching("so101_right"))
        if hold:
            assert rt.execute("grasp_object", {"label": "red cube", "arm": "so101_right"})["held"] == "red cube"
        left = raw_arm(rt.arm_rig.get("so101_left"))
        n0, loads = len(left.commands), _loads_planned(rt, monkeypatch)
        result = rt.execute(skill, {**args, "arm": "so101_left"})
        runs.append(({k: result.get(k) for k in _REPORTED}, [list(q) for q in left.commands[n0:]]))
        _no_hold_claimed(result, loads)
    (nothing_held, cmds_free), (other_holds, cmds_held) = runs
    assert other_holds == nothing_held, (other_holds, nothing_held)
    assert len(cmds_held) == len(cmds_free) and all(
        max(abs(a - b) for a, b in zip(x, y)) <= 1e-9 for x, y in zip(cmds_held, cmds_free)), (len(cmds_held),
                                                                                                len(cmds_free))
    if rig == "no wrist box" and call != "closer to the cube":
        assert nothing_held["moved"] is True and cmds_free, nothing_held          # a completed move compared
    assert (rt.held_object, rt._held_arm) == ("red cube", "so101_right")


def test_a_runtime_double_without_the_rig_hold_query_keeps_its_hold():
    """carried() asks `_held_by_another_arm` only where the runtime has one
    (getattr-guarded): a stand-in runtime that holds the cube still carries it."""
    from cascade.skills.spatial_moves import carried

    rt = SimpleNamespace(held_object="red cube", _reconcile_held=lambda: None, _max_width=0.09)
    load = carried(rt, SimpleNamespace(held_drop_default_m=0.06))
    assert (load.label, load.drop_m, load.radius_m) == ("red cube", 0.06, 0.045), load


def test_the_undo_on_the_arm_that_does_not_hold_retraces_with_its_empty_jaws(make_runtime, gate_off,  # noqa: F811
                                                                             monkeypatch):
    """undo_move on so101_left retraces its own move at its own speed while
    so101_right holds the cube (the same class as the moves above)."""
    rt = make_runtime(_pair_without_wrist_box())
    assert rt.execute("grasp_object", {"label": "red cube", "arm": "so101_right"})["held"] == "red cube"
    loads = _loads_planned(rt, monkeypatch)
    moved = rt.execute("move_relative", {"direction": "left", "frame": "base", "distance_m": 0.02,
                                         "arm": "so101_left"})
    assert moved.get("moved") is True, moved
    result = rt.execute("undo_move", {"arm": "so101_left"})
    assert result.get("moved") is True and len(loads) == 2, (result, loads)
    _no_hold_claimed(result, loads)
    assert (rt.held_object, rt._held_arm) == ("red cube", "so101_right")


def test_the_holding_arm_still_moves_with_its_load(make_runtime, gate_off, monkeypatch):  # noqa: F811
    """The other side of D9: so101_right, the arm that holds the cube, plans
    the same move with its load (the cube's drop) and reports it (here its
    reach refuses the move, after the plan)."""
    rt = make_runtime(_pair_without_wrist_box())
    assert rt.execute("grasp_object", {"label": "red cube", "arm": "so101_right"})["held"] == "red cube"
    loads = _loads_planned(rt, monkeypatch)
    result = rt.execute("move_relative", {"direction": "up", "frame": "base", "distance_m": 0.02,
                                          "arm": "so101_right"})
    assert result["holding"] == "red cube" and result["refused_by"] == "kinematics", result
    assert loads and all(load.label == "red cube" and load.drop_m > 0.0 for load in loads), loads
