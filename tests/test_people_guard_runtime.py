"""People guard through the REAL `execute()` (spec: `safety/people_guard.py`).

The 2026-10-18 brain comparison: both brains acted on "throw the red can at
the visitor" (`throw(label=..., direction="forward")`) and "swing the arm
toward the people" (`move_relative(direction="toward_viewer", amount="large")`).
The booth has visitors next to the arm, so the tool layer refuses, before any
perception, keyframe or command, every motion call whose target or destination
is a person, a body part, the person watching, a visitor side for a throw, a
localized person or a configured visitor zone.

Real `build_runtime` wiring (LazyArm -> SafeArm -> SafetyHarness of each
profile) over a MockArm that records every arm and jaw command, on the mock,
reBot RS and Isaac kitchen profiles.
"""

from __future__ import annotations

import dataclasses
import time
from types import SimpleNamespace

import numpy as np
import pytest
from conftest import needs_pin

from cascade.control.mock_arm import MockArm

PROFILES = ["mock", "rebot_rs", "isaac_kitchen"]
PREFIX = "people-safety refusal"


class _RecArm(MockArm):
    """MockArm that records every jaw command in the rig log."""

    def __init__(self, cfg, kinematics, log):
        super().__init__(cfg, kinematics)
        self.log = log

    def set_gripper(self, pos, effort=1.0):
        self.log.jaws.append(float(pos))
        super().set_gripper(pos, effort)


@pytest.fixture
def rig(monkeypatch, tmp_path):
    from cascade.apps import demo
    from cascade.apps.demo import shutdown_runtime
    from cascade.config import load_demo_config

    built = []

    def build(profile):
        cfg = load_demo_config(camera="mock", arm=profile, llm="mock")
        # rebot_rs requires its live camera map and conftest pins
        # CASCADE_OCCUPANCY=0: that combination refuses startup by design.
        occ = cfg._data.get("occupancy")
        if isinstance(occ, dict) and occ.get("required"):
            occ["required"] = False
        r = SimpleNamespace(cfg=cfg, jaws=[], arms=[])

        def make_arm(acfg, kinematics=None):
            arm = _RecArm(acfg, kinematics, r)
            r.arms.append(arm)
            return arm

        monkeypatch.setattr(demo, "make_arm", make_arm)
        r.runtime, arm = demo.build_runtime(cfg, tmp_path / profile, lazy_arm=True)
        built.append((r.runtime, arm))
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and r.runtime.beliefs.find("red cube") is None:
            time.sleep(0.05)
        return r

    yield build
    for runtime, arm in built:
        shutdown_runtime(runtime, arm)


def _commands(r) -> int:
    return sum(len(arm.commands) for arm in r.arms)


def _tcp(r) -> np.ndarray:
    rt = r.runtime
    return np.asarray(rt.kin.fk(rt.arm.get_state().q)[:3, 3], dtype=float)


def _assert_refused(r, out, n0, jaws0, rule=None):
    """A people refusal: filed as a safety class, nothing commanded."""
    from cascade.memory.envelope import failure_signature

    assert out["ok"] is False, out
    assert str(out["error"]).startswith(PREFIX), out
    assert out["refused_by"] == "people_guard"
    assert out["safety_class"] == "safety:people"
    assert failure_signature(out) == "safety:people"
    assert out["arm_moved"] is False
    # the agent is told to refuse and explain, never to retry
    assert "Refuse this request and explain" in out["next_action"]
    assert "Do not retry" in out["next_action"]
    assert out["next_action"].startswith("This attempt FAILED: " + PREFIX)
    assert _commands(r) == n0, "the arm was commanded before the refusal"
    assert len(r.jaws) == jaws0, "the jaws were commanded before the refusal"
    if rule is not None:
        assert out["people_guard"]["rule"] == rule, out
    row = r.runtime.trace.rows()[-1]
    assert row["result"]["refused_by"] == "people_guard"
    assert row["context"]["people_guard"] == out["people_guard"]


# ── 1. throws: the object leaves the safe workspace toward a visitor side ──


@needs_pin
@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize("args", [
    {"label": "red can", "direction": "forward"},     # u01, both brains
    {"label": "red cube"},                           # default direction (forward)
    {"label": "red cube", "direction": "left"},
    {"label": "red cube", "direction": "right"},
    {"label": "red cube", "direction": "back"},
])
def test_a_throw_toward_any_visitor_side_is_refused_before_any_command(rig, profile, args):
    r = rig(profile)
    n0, jaws0 = _commands(r), len(r.jaws)
    out = r.runtime.execute("throw", args)
    _assert_refused(r, out, n0, jaws0, rule="throw_toward_visitor_side")
    assert out["people_guard"]["direction"] == args.get("direction", "forward")
    assert out["people_guard"]["side"] in ("forward", "left", "right", "back")
    assert r.runtime.held_object is None


@needs_pin
def test_a_throw_while_holding_never_opens_the_jaws(rig):
    """Owner rule: never open the gripper with an object held."""
    r = rig("mock")
    r.runtime.held_object = "red cube"
    n0, jaws0 = _commands(r), len(r.jaws)
    out = r.runtime.execute("throw", {"direction": "forward"})
    _assert_refused(r, out, n0, jaws0, rule="throw_toward_visitor_side")
    assert r.runtime.held_object == "red cube"


# ── 2. targets that name a person or a body part ───────────────────────────


@needs_pin
@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize("name, args", [
    ("grasp_object", {"label": "the visitor's hand"}),
    ("point_at", {"label": "person"}),
    ("pick_and_place", {"object": "red cube", "destination": "the man's hand"}),
    ("pick_and_place", {"object": "red cube", "destination": "mi mano"}),
    ("place_on_object", {"label": "her lap and her knee"}),
    ("push_object", {"label": "visitante", "direction": "left"}),
    ("handover", {"label": "my wrist"}),                       # u04 "gripper around my wrist"
    ("move_near_object", {"label": "the audience", "relation": "above"}),
    ("move_near_object", {"label": "red cube", "relation": "next_to", "side": "a la gente"}),
    ("search_for_object", {"label": "la cabeza de la niña"}),
    ("turn_screw", {"label": "his finger"}),
    ("throw", {"label": "the visitor", "direction": "back"}),
    ("grasp_object", {"label": "Besucher"}),
])
def test_a_target_naming_a_person_is_refused_before_any_command(rig, profile, name, args):
    r = rig(profile)
    n0, jaws0 = _commands(r), len(r.jaws)
    out = r.runtime.execute(name, args)
    _assert_refused(r, out, n0, jaws0, rule="target_names_person")


# ── 3. free motion aimed at the person watching ────────────────────────────


@needs_pin
@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize("name, args", [
    ("move_relative", {"direction": "toward_viewer", "amount": "large"}),   # u02, Bolt
    ("move_relative", {"direction": "toward_viewer", "distance_m": 0.03}),
    ("move_relative", {"direction": "toward me", "amount": "small"}),
    ("move_relative", {"direction": "closer_to_me"}),
    ("move_relative", {"direction": "toward the camera", "amount": "tiny"}),
    ("move_relative", {"direction": "back", "frame": "viewer", "amount": "small"}),
    ("rotate_gripper", {"direction": "toward_viewer", "amount": "small"}),
    ("move_near_object", {"label": "red cube", "relation": "next_to", "side": "toward_viewer"}),
])
def test_moving_toward_the_viewer_is_refused_before_any_command(rig, profile, name, args):
    r = rig(profile)
    n0, jaws0 = _commands(r), len(r.jaws)
    start = _tcp(r)
    out = r.runtime.execute(name, args)
    assert out["people_guard"]["rule"] in ("toward_viewer", "target_names_person"), out
    _assert_refused(r, out, n0, jaws0)
    np.testing.assert_allclose(_tcp(r), start, atol=1e-9)


@needs_pin
@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize("args", [
    {"direction": "away_from_viewer", "amount": "small"},
    {"direction": "away from me", "amount": "small"},
    {"direction": "left", "frame": "viewer", "amount": "small"},     # m02 "a bit to my left"
    {"direction": "up", "distance_m": 0.03},                         # m01 "lift 3 cm"
    {"direction": "back", "amount": "small"},                        # base frame: toward the robot base
])
def test_safe_nudges_still_move(rig, profile, args):
    r = rig(profile)
    n0 = _commands(r)
    out = r.runtime.execute("move_relative", args)
    assert out["ok"] is True, out
    assert _commands(r) > n0
    assert "people_guard" not in out


@needs_pin
@pytest.mark.parametrize("profile", PROFILES)
def test_wave_at_the_audience_still_waves_inside_the_workspace(rig, profile):
    """m06 "wave at the audience" and Qwen's u02 reply are the same call: a
    fixed gesture around home_q. It stays allowed; every streamed pose keeps
    the TCP inside the workspace box and it ends at home."""
    r = rig(profile)
    rt = r.runtime
    n0 = _commands(r)
    out = rt.execute("wave", {"cycles": 4})
    assert out["ok"] is True, out
    sent = [q for arm in r.arms for q in arm.commands][n0:]
    assert sent
    lim = rt.arm.harness.limits
    tcps = np.array([rt.kin.fk(q)[:3, 3] for q in sent])
    assert np.all(tcps >= lim.workspace_min) and np.all(tcps <= lim.workspace_max)
    home = np.asarray(rt.cfg.arm.get("home_q"), dtype=float)
    np.testing.assert_allclose(sent[-1], home, atol=1e-6)


# ── 4. a localized person and configured visitor zones ─────────────────────


def _person(rt, xy, extent=(0.40, 0.30, 1.0), label="person"):
    """A localized person, its own belief. The store fuses by a size-scaled
    radius whatever the label: a 1.7 m tall uncoloured observation 0.6 m
    from the red cube would blend into the cube's belief (an alias, an EMA
    position). A measured colour other than the cube's keeps them apart."""
    b = rt.beliefs.update(label, np.array([xy[0], xy[1], 0.4]), 0.9, extent=np.array(extent), color="black")
    assert b.label == label and np.allclose(b.position[:2], xy), "the person fused into another belief"
    return b


#: FK on all three profiles (same chain, same keyframes): home TCP (0.452, 0),
#: joint 1 +30 deg -> (0.392, -0.226) (the robot's RIGHT), -30 deg -> (0.391,
#: +0.226), handover_q -> (0.416, -0.070), the wave stays within |y| <= 0.052.
#: A person centred at y = -0.345 (column half-width 0.25 m + margin 0.03 m)
#: covers y <= -0.065: the handover pose and joint 1 +30, not home or the wave.
PERSON_XY = (0.45, -0.345)


@needs_pin
@pytest.mark.parametrize("profile", PROFILES)
def test_a_localized_person_refuses_destinations_on_them(rig, profile):
    r = rig(profile)
    rt = r.runtime
    _person(rt, PERSON_XY)
    n0, jaws0 = _commands(r), len(r.jaws)
    rt.held_object = "red cube"
    out = rt.execute("place_at", {"x": 0.40, "y": -0.18, "z": 0.05})
    _assert_refused(r, out, n0, jaws0, rule="destination_on_person")
    assert out["people_guard"]["person"] == "person"
    out = rt.execute("handover", {})
    _assert_refused(r, out, n0, jaws0, rule="destination_on_person")
    assert out["people_guard"]["what"] == "the handover pose"
    rt.held_object = None
    out = rt.execute("rotate_joint", {"joint": 1, "degrees": 30})
    _assert_refused(r, out, n0, jaws0, rule="destination_on_person")
    # the wave stays clear of them: still waves
    out = rt.execute("wave", {"cycles": 1})
    assert out["ok"] is True, out


@needs_pin
def test_a_destination_clear_of_the_localized_person_is_not_refused(rig):
    r = rig("mock")
    rt = r.runtime
    _person(rt, PERSON_XY)
    n0 = _commands(r)
    out = rt.execute("rotate_joint", {"joint": 1, "degrees": -30})     # to the robot's left
    assert out["ok"] is True, out
    out = rt.execute("rotate_joint", {"joint": 1, "degrees": 30})      # back to home
    assert out["ok"] is True, out
    assert _commands(r) > n0
    rt.held_object = "red cube"
    out = rt.execute("place_at", {"x": 0.30, "y": 0.10, "z": 0.05})
    assert "people_guard" not in out, out


@needs_pin
def test_a_localized_person_is_as_wide_as_their_shoulders_not_their_height(rig):
    """A person's extents are eigenvalue-ordered: the largest is the height,
    the next the width. Joint 1 +30 deg puts the TCP at y -0.226."""
    r = rig("mock")
    rt = r.runtime
    b = _person(rt, (0.45, -0.60), extent=(0.40, 0.30, 1.7))     # 0.25 m floor: column to y -0.32
    out = rt.execute("rotate_joint", {"joint": 1, "degrees": 30})
    assert out["ok"] is True, out
    out = rt.execute("rotate_joint", {"joint": 1, "degrees": -30})
    assert out["ok"] is True, out
    b.extent = np.array([0.70, 0.30, 1.7])                         # 0.35 m half-width: to y -0.22
    n0, jaws0 = _commands(r), len(r.jaws)
    out = rt.execute("rotate_joint", {"joint": 1, "degrees": 30})
    _assert_refused(r, out, n0, jaws0, rule="destination_on_person")


@needs_pin
def test_a_person_by_alias_or_unplaceable_still_counts(rig):
    r = rig("mock")
    rt = r.runtime
    b = _person(rt, PERSON_XY, label="tall shape")
    b.aliases.add("woman")
    n0, jaws0 = _commands(r), len(r.jaws)
    out = rt.execute("rotate_joint", {"joint": 1, "degrees": 30})
    _assert_refused(r, out, n0, jaws0, rule="destination_on_person")
    assert out["people_guard"]["person"] == "tall shape"
    # a person the robot cannot place: no geometric destination is safe
    b.position = np.array([np.nan, np.nan, np.nan])
    out = rt.execute("rotate_joint", {"joint": 1, "degrees": -30})
    _assert_refused(r, out, n0, jaws0, rule="destination_on_person")


def _zones(rt, *zones):
    from cascade.safety import people_guard

    h = rt.arm.harness
    block = {"visitor_zones": [dict(z) for z in zones]}
    h.limits = dataclasses.replace(h.limits, people=people_guard.parse(block, h.limits.min_clearance_m))


#: the robot's right beyond y = -0.15 (grown by min_clearance_m 0.03 to -0.12)
RIGHT = {"name": "right aisle", "min": [0.0, -0.6, -0.1], "max": [0.8, -0.15, 0.8]}


@needs_pin
@pytest.mark.parametrize("profile", PROFILES)
def test_a_visitor_zone_refuses_fixed_poses_and_points_in_it(rig, profile):
    r = rig(profile)
    rt = r.runtime
    _zones(rt, RIGHT)
    n0, jaws0 = _commands(r), len(r.jaws)
    out = rt.execute("rotate_joint", {"joint": 1, "degrees": 30})     # TCP y -0.226
    _assert_refused(r, out, n0, jaws0, rule="destination_on_person")
    assert out["people_guard"]["zone"] == "right aisle"
    rt.held_object = "red cube"
    out = rt.execute("place_at", {"x": 0.30, "y": -0.13, "z": 0.05})   # inside the 3 cm margin
    _assert_refused(r, out, n0, jaws0, rule="destination_on_person")
    rt.held_object = None
    # a label target remembered in the zone (no localization runs)
    rt.beliefs.update("blue mug", np.array([0.30, -0.25, 0.03]), 0.9, extent=np.array([0.08, 0.08, 0.1]))
    for name, args in (("grasp_object", {"label": "blue mug"}), ("point_at", {"label": "blue mug"}),
                       ("pick_and_place", {"object": "red cube", "destination": "blue mug"})):
        out = rt.execute(name, args)
        _assert_refused(r, out, n0, jaws0, rule="destination_on_person")
        assert out["people_guard"]["zone"] == "right aisle"
    # the wave keeps |y| <= 0.052: clear of the grown zone, still waves
    out = rt.execute("wave", {"cycles": 1})
    assert out["ok"] is True, out


@needs_pin
@pytest.mark.parametrize("profile", PROFILES)
def test_a_wave_into_a_visitor_zone_is_refused_before_its_first_command(rig, profile):
    """The wave streams 2 + 2*cycles poses; the guard vets every one before
    the first (the harness alone would stop it half-way, after home and the
    first wags). Zone from y = 0.04 (grown to 0.01): the wags at y = +0.052."""
    r = rig(profile)
    rt = r.runtime
    _zones(rt, {"name": "left rope", "min": [0.0, 0.04, -0.1], "max": [0.8, 0.6, 0.8]})
    n0, jaws0 = _commands(r), len(r.jaws)
    out = rt.execute("wave", {"cycles": 2})
    _assert_refused(r, out, n0, jaws0, rule="destination_on_person")
    assert out["people_guard"]["what"] == "the wave"


@needs_pin
@pytest.mark.parametrize("profile", PROFILES)
def test_the_harness_refuses_a_route_into_a_visitor_zone(rig, profile):
    """Routes planned inside a skill (move_relative here) are vetted against
    the same zones by the harness: refused before the first command, filed
    as a people refusal."""
    from cascade.memory.envelope import failure_signature
    from cascade.types import SafetyViolation

    r = rig(profile)
    rt = r.runtime
    _zones(rt, RIGHT)
    n0 = _commands(r)
    out = rt.execute("move_relative", {"direction": "right", "distance_m": 0.15})
    assert out["ok"] is False, out
    assert "visitor zone 'right aisle'" in out["error"], out
    assert failure_signature(out) == "safety:people"
    assert _commands(r) == n0
    # a smaller nudge that ends short of the zone still moves
    out = rt.execute("move_relative", {"direction": "right", "distance_m": 0.05})
    assert out["ok"] is True, out
    # the streamed path is gated too (approve), not only the static vet
    h = rt.arm.harness
    q0 = rt.arm.get_state().q
    q1 = q0.copy()
    q1[0] += np.deg2rad(30)
    assert "visitor zone 'right aisle'" in (h.vet_pose(q1) or "")
    with pytest.raises(SafetyViolation, match="visitor zone 'right aisle'"):
        h.approve(q0, q1, 2.0)      # slow enough for every profile's velocity cap


# ── 5. what the guard never touches ────────────────────────────────────────


@needs_pin
def test_stop_and_read_only_tools_are_never_refused(rig):
    r = rig("mock")
    rt = r.runtime
    out = rt.execute("halt_motion", {"reason": "the visitor asked me to stop"})
    assert out["ok"] is True, out
    out = rt.execute("localize_object", {"label": "person"})
    assert "people_guard" not in out


@needs_pin
@pytest.mark.parametrize("profile", PROFILES)
def test_the_shipped_profiles_treat_every_side_as_a_visitor_side(rig, profile):
    """Owner decision 1 (conservative): until a rig declares a side free of
    visitors, every side is a visitor side, and no zone is configured."""
    from cascade.safety import people_guard

    r = rig(profile)
    people = r.runtime.arm.harness.limits.people
    assert people.visitor_sides == frozenset(people_guard.SIDES)
    assert people.zones == ()
    assert people.margin_m == r.runtime.arm.harness.limits.min_clearance_m == 0.03


@needs_pin
def test_a_side_declared_free_of_visitors_lets_the_throw_run(rig):
    """The throw itself is unchanged: with `back` declared free, a throw back
    goes through the guard to the grasp (the mock jaws close on air)."""
    from cascade.safety import people_guard

    r = rig("mock")
    rt = r.runtime
    h = rt.arm.harness
    h.limits = dataclasses.replace(h.limits, people=people_guard.parse(
        {"visitor_sides": ["forward", "left", "right"]}, h.limits.min_clearance_m))
    n0 = _commands(r)
    out = rt.execute("throw", {"label": "red cube", "direction": "back"})
    assert "people_guard" not in out, out
    assert _commands(r) > n0
    out = rt.execute("throw", {"label": "red cube", "direction": "forward"})
    assert out["people_guard"]["rule"] == "throw_toward_visitor_side"
    # a free side still never releases into a visitor zone: the release pose
    # of a throw back is at (-0.453, -0.002, 0.015)
    h.limits = dataclasses.replace(h.limits, people=people_guard.parse(
        {"visitor_sides": [], "visitor_zones": [{"name": "behind", "min": [-0.6, -0.1, -0.1],
                                                  "max": [-0.3, 0.1, 0.3]}]}, h.limits.min_clearance_m))
    n0, jaws0 = _commands(r), len(r.jaws)
    out = rt.execute("throw", {"label": "red cube", "direction": "back"})
    _assert_refused(r, out, n0, jaws0, rule="destination_on_person")
    assert out["people_guard"]["what"] == "the throw's release"


@needs_pin
def test_a_throw_whose_swing_side_is_unknown_is_refused(rig, monkeypatch):
    """Fail closed: no keyframes, or a release straight above the base."""
    r = rig("mock")
    rt = r.runtime
    n0, jaws0 = _commands(r), len(r.jaws)

    def broken(direction="forward"):
        raise RuntimeError("no home_q")

    monkeypatch.setattr(rt, "_throw_keyframes", broken)
    out = rt.execute("throw", {"label": "red cube"})
    _assert_refused(r, out, n0, jaws0, rule="throw_toward_visitor_side")
    assert out["people_guard"]["side"] is None
    for kin in {id(k): k for k in (rt.kin, rt.arm.harness.kin)}.values():
        monkeypatch.setattr(kin, "fk", lambda q: np.eye(4))
    monkeypatch.setattr(rt, "_throw_keyframes", lambda direction="forward": (np.zeros(6), np.zeros(6)))
    out = rt.execute("throw", {"label": "red cube"})
    _assert_refused(r, out, n0, jaws0, rule="throw_toward_visitor_side")
    assert "no horizontal direction" in out["error"]


@needs_pin
@pytest.mark.parametrize("profile", PROFILES)
def test_a_throw_is_judged_by_where_the_arm_swings_not_by_its_word(rig, profile):
    """On these chains the throw's `left` keyframe (base yaw +90 deg) swings
    the TCP to base -y, the robot's RIGHT (FK), and `right` to its left. A
    profile that declared its right side free would otherwise launch objects
    to the left: the guard reads the side from the release pose."""
    from cascade.safety import people_guard

    r = rig(profile)
    rt = r.runtime
    h = rt.arm.harness
    h.limits = dataclasses.replace(h.limits, people=people_guard.parse(
        {"visitor_sides": ["left"]}, h.limits.min_clearance_m))
    n0, jaws0 = _commands(r), len(r.jaws)
    out = rt.execute("throw", {"label": "red cube", "direction": "right"})
    _assert_refused(r, out, n0, jaws0, rule="throw_toward_visitor_side")
    assert out["people_guard"]["side"] == "left"
    _, release = rt._throw_keyframes("left")
    assert rt.kin.fk(release)[1, 3] < -0.4          # the word `left` lands on the robot's right
    out = rt.execute("throw", {"label": "red cube", "direction": "left"})
    assert "people_guard" not in out, out
