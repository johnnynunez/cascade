"""Every arm's safety config is validated BEFORE any arm is made or connected.

`SafetyHarness(...)` is where a bad arm safety config refuses: a malformed
`safety.tool_volume` block, or a tool volume on a tilted `base_pose`.
`_build_arm` used to make (and, eager, connect) the backend first and build
the harness after it, so on an eager real arm the refusal came after the
connect had enabled torque, and the failed build left the arm connected. On a
two-arm rig, arm 0 was fully built and connected before arm 1's config was
even read. Every backend below is a recording MockArm (no hardware). Pinned:

1. an eager `_build_arm` whose harness refuses makes and connects nothing;
2. an eager two-arm `build_runtime` whose SECOND arm refuses makes and
   connects neither arm;
3. lazy builds (the MCP server path) still make and connect nothing;
4. a valid build makes the same objects, harnesses first, then each arm in
   order;
5. the tilted-mount refusal names the arm and its roll / pitch in degrees;
6. every other pure-config error `_build_arm` / `build_runtime` used to raise
   after an eager connect (a scalar `park_q`, a bad `body_mask` block, a mesh
   mask on an isaac arm or one whose geometry cannot load, a bad
   `body_mask_radius_m`, `track_payload` on a non-isaac or a second arm, a
   mesh mask beside payload tracking, duplicate arm names) refuses before any
   arm is made or connected, alone and as the second arm of a pair; valid
   builds keep their order (the mesh body mask is built before its arm).
"""

from __future__ import annotations

import re

import numpy as np
import pytest
from conftest import needs_pin, needs_pin_so101
from test_occupancy_mesh_body_mask import needs_rs

from cascade.safety.harness import SafetyHarness, SafetyLimits
from cascade.safety.tool_volume import ToolVolume
from cascade.types import pose_to_transform

#: roll 0.1 rad (5.73 deg): a tilted mount
TILTED = [0.0, -0.22, 0.0, 0.1, 0.0, 0.0]
BAD = ("tilted mount", "bad tool_volume block")
#: what each case's refusal says
REFUSAL = {"tilted mount": "level base_pose", "bad tool_volume block": "table_band_m must be in"}


@pytest.fixture
def events(monkeypatch):
    """Every make / connect / harness construction, in order. A harness is
    recorded once it is built, by its base_pose y (None: no base_pose)."""
    from cascade.apps import demo
    from cascade.control.mock_arm import MockArm

    log = []

    def make_arm(acfg, kinematics=None):
        name = str(acfg.get("name", acfg.get("type", "arm")))
        log.append(("make", name))

        class Recorded(MockArm):
            def connect(self):
                log.append(("connect", name))
                super().connect()

        return Recorded(acfg, kinematics=kinematics)

    real = demo.SafetyHarness

    def harness(*args, **kwargs):
        h = real(*args, **kwargs)
        log.append(("harness", None if h.base_pose is None else round(float(h.base_pose[1, 3]), 3)))
        return h

    monkeypatch.setattr(demo, "make_arm", make_arm)
    monkeypatch.setattr(demo, "SafetyHarness", harness)
    # the mesh body mask's construction and every map registration
    from cascade.perception import robot_mask
    from cascade.perception.occupancy import OccupancyMap

    real_mesh = robot_mask.build_mesh_body_mask

    def mesh(acfg, kin, state_fn, *, name):
        out = real_mesh(acfg, kin, state_fn, name=name)
        if out is not None:
            log.append(("mesh", name))
        return out

    monkeypatch.setattr(robot_mask, "build_mesh_body_mask", mesh)
    for meth in ("add_robot_body", "add_robot_body_mask", "track_payload"):
        real_meth = getattr(OccupancyMap, meth)

        def registered(self, *a, _real=real_meth, _meth=meth, **k):
            out = _real(self, *a, **k)
            log.append((_meth,))
            return out

        monkeypatch.setattr(OccupancyMap, meth, registered)
    return log


def _arm_calls(log):
    return [e for e in log if e[0] in ("make", "connect")]


def _rebot(bad=None):
    from cascade.config import load_demo_config

    cfg = load_demo_config(arm="rebot_rs", camera="mock", llm="mock")
    acfg = cfg.arm
    assert acfg.get("resolved").get("safety").get("tool_volume") is not None   # premise: it has one
    if bad == "tilted mount":
        acfg._data["base_pose"] = [0.0, 0.0, 0.0, 0.1, 0.0, 0.0]
    elif bad == "bad tool_volume block":
        acfg._data["resolved"]["safety"]["tool_volume"]["table_band_m"] = 0.2
    return cfg, acfg


def _pair(bad=None):
    """The so101 pair; `bad` breaks the SECOND arm (so101_right) only."""
    from cascade.config import load_demo_config

    cfg = load_demo_config(arms=["so101_left", "so101_right"], camera="mock", llm="mock")
    right = cfg._data["arms"][1]
    assert right["name"] == "so101_right"
    if bad is not None:
        right["resolved"]["safety"]["tool_volume"] = {"wrist_camera": False, "jaws": True}
    if bad == "tilted mount":
        right["base_pose"] = TILTED
    elif bad == "bad tool_volume block":
        right["resolved"]["safety"]["tool_volume"]["table_band_m"] = 0.2
    return cfg


# ── 1. one arm: the harness refuses before the arm exists ────────────────


@needs_pin
@pytest.mark.parametrize("lazy", [False, True], ids=["eager", "lazy"])
@pytest.mark.parametrize("bad", BAD)
def test_a_refusing_harness_config_makes_and_connects_no_arm(events, bad, lazy):
    """RED on b0df584 for eager: make + connect (torque on a real reBot),
    then the ValueError. Lazy never made the arm: pinned unchanged."""
    from cascade.apps import demo

    cfg, acfg = _rebot(bad)
    with pytest.raises(ValueError, match=REFUSAL[bad]):
        demo._build_arm(acfg, lazy, None, cfg)
    assert _arm_calls(events) == []


# ── 2. two arms: arm 1 refusing leaves arm 0 untouched ──────────────────


@needs_pin_so101
@pytest.mark.parametrize("lazy", [False, True], ids=["eager", "lazy"])
@pytest.mark.parametrize("bad", BAD)
def test_a_two_arm_build_whose_second_arm_refuses_touches_neither_arm(events, tmp_path, bad, lazy):
    """RED on b0df584 for eager: so101_left was made and connected before
    so101_right's harness refused (and `_abandon_build` leaves arms as they
    are). Every arm's harness is built in a pre-pass now."""
    from cascade.apps import demo

    with pytest.raises(ValueError, match=REFUSAL[bad]):
        demo.build_runtime(_pair(bad), tmp_path / "run", lazy_arm=lazy)
    assert _arm_calls(events) == []


# ── 3. valid configs: same objects, harnesses first ──────────────────────


@needs_pin
@pytest.mark.parametrize("lazy", [False, True], ids=["eager", "lazy"])
def test_a_valid_rebot_build_builds_its_harness_before_its_arm(events, lazy):
    from cascade.apps import demo
    from cascade.control.lazy_arm import LazyArm

    cfg, acfg = _rebot()
    raw, safe, kin = demo._build_arm(acfg, lazy, None, cfg)
    if lazy:
        assert events == [("harness", None)]
        assert isinstance(raw, LazyArm) and not raw.connected
    else:
        assert events == [("harness", None), ("make", "rebot_rs"), ("connect", "rebot_rs")]
        assert raw._connected
    h = safe.harness
    assert h.kin is kin and h.base_pose is None and h.tool_volume is not None
    assert h.park_q == list(acfg.get("park_q"))
    assert h.limits.min_clearance_m == pytest.approx(0.03)        # margins untouched


@needs_pin_so101
def test_a_valid_eager_pair_builds_both_harnesses_then_each_arm_in_order(events, tmp_path):
    from cascade.apps import demo

    runtime, owner = demo.build_runtime(_pair(), tmp_path / "run", lazy_arm=False)
    try:
        assert events[:6] == [("harness", 0.22), ("harness", -0.22),
                              ("make", "so101_left"), ("connect", "so101_left"),
                              ("make", "so101_right"), ("connect", "so101_right")]
        assert _arm_calls(events[6:]) == []
        rig = runtime.arm_rig
        for name, y in (("so101_left", 0.22), ("so101_right", -0.22)):
            h = rig.get(name).harness
            assert h.base_pose[1, 3] == pytest.approx(y) and h.kin is not None
        # each arm got its OWN harness, and the primary is still the first arm
        assert rig.get("so101_left").harness is not rig.get("so101_right").harness
        assert runtime.arm.harness is rig.get("so101_left").harness
    finally:
        demo.shutdown_runtime(runtime, owner)


# ── 4. the tilted-mount refusal says which arm and how it is tilted ──────


@needs_pin_so101
def test_the_tilted_mount_refusal_names_the_arm_and_its_roll_and_pitch(events):
    from cascade.apps import demo

    cfg = _pair("tilted mount")
    cfg._data["arms"][1]["base_pose"] = [0.0, -0.22, 0.0, 0.1, -0.05, 0.4]
    acfg = demo._arm_cfgs(cfg)[1]
    with pytest.raises(ValueError) as err:
        demo._build_arm(acfg, False, None, cfg)
    msg = str(err.value)
    assert re.search(r"arm 'so101_right'.*roll 5\.73 deg, pitch -2\.86 deg", msg), msg
    assert "level base_pose" in msg and "safety.tool_volume" in msg, msg


@pytest.mark.parametrize(("rpy", "said"), [
    ((0.1, 0.0, 0.0), "roll 5.73 deg, pitch 0.00 deg"),
    ((0.0, -0.1, 0.0), "roll 0.00 deg, pitch -5.73 deg"),
    ((0.05, 0.05, 0.4), "roll 2.86 deg, pitch 2.86 deg"),          # yaw is not a tilt
    ((np.pi, 0.0, 0.0), "roll 180.00 deg, pitch 0.00 deg"),         # upside down
], ids=["roll", "pitch", "roll+pitch+yaw", "upside down"])
def test_the_harness_refusal_states_the_mounts_roll_and_pitch(rpy, said):
    """The harness alone (no arm name given) still states the tilt."""
    tool = ToolVolume(local=np.zeros((1, 3)), part=np.array([0]), names=("jaws",))
    limits = SafetyLimits(np.array([-1.0, -1.0, -0.2]), np.array([1.0, 1.0, 1.0]))
    with pytest.raises(ValueError, match="level base_pose") as err:
        SafetyHarness(limits, base_pose=pose_to_transform([0.0, -0.22, 0.0, *rpy]), tool_volume=tool)
    assert said in str(err.value), str(err.value)
    assert "arm '" not in str(err.value)


# ── 6. every other pure-config error: refused before any arm exists ──────

#: what each config error says (a regex)
CONFIG_REFUSAL = {
    "scalar park_q": r"arm 'so101_(left|right)': park_q must be a list of joint positions",
    "body_mask unknown key": r"body_mask: unknown keys \['inflate_mm'\]",
    "body_mask bad mode": r"body_mask: mode must be `capsule` or `mesh`",
    "mesh mask on an isaac arm": r"body_mask: mesh is for real arms, not type isaac",
    "mesh mask that cannot load": r"body_mask: mesh for arm 'so101_(left|right)'",
    "bad body_mask_radius_m": r"could not convert string to float: 'wide'",
    "track_payload on a non-isaac arm": r"payload segmentation currently requires the Isaac capture contract",
}
#: only a rig of several arms can get these wrong
RIG_REFUSAL = {
    "track_payload on a second arm": r"payload tracking currently requires a single arm",
    "duplicate arm names": r"duplicate arm names in rig: \['so101_left', 'so101_left'\]",
}


def _map():
    """An offline occupancy map (no bridge process, no port)."""
    from occupancy_grid_bridge import TABLE, GridBridge, make_map

    return make_map(GridBridge([TABLE], voxel=0.01), required=False, unobserved="free")


def _track_payload(cfg):
    cfg._data.setdefault("occupancy", {})["track_payload"] = True


def _break(cfg, arm: dict, bad: str) -> None:
    """Make `arm` (an arm profile's data) carry the config error `bad`."""
    if bad == "scalar park_q":
        arm["park_q"] = 0.5
    elif bad == "body_mask unknown key":
        arm["body_mask"] = {"mode": "mesh", "inflate_mm": 0.02}
    elif bad == "body_mask bad mode":
        arm["body_mask"] = {"mode": "sphere"}
    elif bad == "mesh mask on an isaac arm":
        arm.update(type="isaac", body_mask="mesh")
    elif bad == "mesh mask that cannot load":
        # the so101 URDF ships without its meshes, and the attachment's link
        # does not exist: either way build_mesh_body_mask refuses
        arm["body_mask"] = {"mode": "mesh", "attachments": [
            {"name": "cam", "link": "no_such_link", "box": [0.01, 0.01, 0.01]}]}
    elif bad == "bad body_mask_radius_m":
        arm["body_mask_radius_m"] = "wide"
    elif bad == "track_payload on a non-isaac arm":
        _track_payload(cfg)
    else:
        raise AssertionError(bad)


def _refuses(match, build, events, since=0):
    """`build()` raises a config error that says `match`; nothing was made
    or connected first (`events[since:]`; asserted before the message:
    that is the defect)."""
    from cascade.perception.occupancy import OccupancyError

    with pytest.raises((ValueError, TypeError, OccupancyError)) as err:
        build()
    assert _arm_calls(events[since:]) == [], events
    assert re.search(match, str(err.value)), str(err.value)


@needs_pin_so101
@pytest.mark.parametrize("bad", CONFIG_REFUSAL)
def test_a_config_error_refuses_one_eager_arm_before_it_is_made(events, bad):
    """`_build_arm` alone (an eager caller like the calibration CLI), on a
    map. RED on 57231d1: make + connect (torque on a real arm), then the
    error."""
    from cascade.apps import demo
    from cascade.config import load_demo_config

    cfg = load_demo_config(arm="so101_left", camera="mock", llm="mock")
    _break(cfg, cfg.arm._data, bad)
    _refuses(CONFIG_REFUSAL[bad], lambda: demo._build_arm(cfg.arm, False, _map(), cfg), events)


@needs_pin_so101
@pytest.mark.parametrize("bad", [*CONFIG_REFUSAL, *RIG_REFUSAL])
def test_a_config_error_in_the_second_arm_refuses_before_either_arm_is_made(
        events, tmp_path, monkeypatch, bad):
    """The eager `cascade demo` path (build_runtime) on a map, the so101 pair
    with the error in so101_right only. RED on 57231d1: so101_left (and for
    most errors so101_right too) made and connected first, and the failed
    build leaves arms as they are."""
    from cascade.apps import demo
    from cascade.perception.occupancy import OccupancyMap

    monkeypatch.setattr(OccupancyMap, "from_config", classmethod(lambda cls, *a, **k: _map()))
    cfg = _pair()
    left, right = cfg._data["arms"]
    if bad == "track_payload on a second arm":
        left["type"] = right["type"] = "isaac"
        _track_payload(cfg)
    elif bad == "duplicate arm names":
        right["name"] = "so101_left"
    else:
        if bad == "track_payload on a non-isaac arm":
            left["type"] = "isaac"       # arm 0 may track the payload; arm 1 may not
        _break(cfg, right, bad)
    match = {**CONFIG_REFUSAL, **RIG_REFUSAL}[bad]
    _refuses(match, lambda: demo.build_runtime(cfg, tmp_path / "run", lazy_arm=False), events)


@needs_pin_so101
def test_a_second_build_arm_on_a_map_that_tracks_a_payload_refuses_before_its_arm(events):
    """Two `_build_arm` calls on one map: the map already tracks the first
    arm's payload, so the second arm refuses before it is made."""
    from cascade.apps import demo

    cfg = _pair()
    _track_payload(cfg)
    for arm in cfg._data["arms"]:
        arm["type"] = "isaac"
    left, right = demo._arm_cfgs(cfg)
    occ = _map()
    demo._build_arm(left, False, occ, cfg)
    assert _arm_calls(events) == [("make", "so101_left"), ("connect", "so101_left")]
    assert occ.tracks_payload
    _refuses(RIG_REFUSAL["track_payload on a second arm"],
             lambda: demo._build_arm(right, False, occ, cfg), events, since=len(events))


def _rebot_beside_an_isaac_arm_that_tracks_the_payload():
    """so101_left as an isaac arm tracking the payload, then the reBot with
    its shipped mesh body mask as the second arm."""
    import copy

    from cascade.config import load_demo_config

    cfg = _pair()
    _track_payload(cfg)
    cfg._data["arms"][0]["type"] = "isaac"
    rebot = copy.deepcopy(load_demo_config(arm="rebot_rs", camera="mock", llm="mock").arm._data)
    assert rebot["body_mask"]["mode"] == "mesh"      # premise: the shipped reBot masks by mesh
    cfg._data["arms"][1] = rebot
    return cfg


@needs_rs
def test_a_mesh_masked_second_arm_beside_payload_tracking_refuses_before_either_arm(
        events, tmp_path, monkeypatch):
    """RED on 57231d1: both arms made and connected, then the map refused the
    reBot's mesh mask (payload replay re-sends history with render masks only)."""
    from cascade.apps import demo
    from cascade.perception.occupancy import OccupancyMap

    monkeypatch.setattr(OccupancyMap, "from_config", classmethod(lambda cls, *a, **k: _map()))
    cfg = _rebot_beside_an_isaac_arm_that_tracks_the_payload()
    _refuses(r"payload tracking cannot be combined with a mesh body mask",
             lambda: demo.build_runtime(cfg, tmp_path / "run", lazy_arm=False), events)


@needs_rs
def test_a_mesh_mask_on_an_isaac_rebot_refuses_before_it_is_made(events):
    """The reBot's meshes load, so only the isaac check stops this one
    (the so101 case above also fails to load its meshes)."""
    from cascade.apps import demo

    cfg, acfg = _rebot()
    acfg._data["type"] = "isaac"
    _refuses(CONFIG_REFUSAL["mesh mask on an isaac arm"],
             lambda: demo._build_arm(acfg, False, _map(), cfg), events)


@needs_rs
def test_a_mesh_masked_arm_on_a_map_that_tracks_a_payload_refuses_before_it_is_made(events):
    """`_build_arm` alone, the reBot (its own config: no `track_payload`) on
    a map an isaac arm already tracks a payload on: only the map's state
    says no, and it says so before the reBot is made."""
    from cascade.apps import demo
    from cascade.config import load_demo_config

    occ = _map()
    isaac = load_demo_config(arm="so101_left", camera="mock", llm="mock")
    isaac.arm._data["type"] = "isaac"
    _track_payload(isaac)
    demo._build_arm(isaac.arm, False, occ, isaac)
    assert occ.tracks_payload
    cfg, acfg = _rebot()
    assert not cfg.get("occupancy").get("track_payload", False)    # premise: the reBot asks for none
    _refuses(r"payload tracking cannot be combined with a mesh body mask",
             lambda: demo._build_arm(acfg, False, occ, cfg), events, since=len(events))


# ── 7. valid builds on a map: same registrations, mesh built before its arm ──


@needs_rs
@pytest.mark.parametrize("lazy", [False, True], ids=["eager", "lazy"])
def test_a_valid_rebot_build_on_a_map_builds_its_mesh_mask_before_its_arm(events, lazy):
    """The mesh body mask (pure config: the URDF meshes, ~1.5 s) is built
    before the arm is made, and reads the arm built after it: a connected
    arm's state, a standby LazyArm's none (never woken)."""
    from cascade.apps import demo

    cfg, acfg = _rebot()
    occ = _map()
    raw, safe, _ = demo._build_arm(acfg, lazy, occ, cfg)
    arm_calls = [] if lazy else [("make", "rebot_rs"), ("connect", "rebot_rs")]
    assert events == [("harness", None), ("mesh", "rebot_rs"), *arm_calls, ("add_robot_body_mask",)]
    mesh = safe.body_mask
    assert mesh is not None and occ._pixel_body_fns == [mesh]
    state = mesh._state_fn()
    if lazy:
        assert state is None and not raw.connected
    else:
        np.testing.assert_allclose(state.q, raw.get_state().q)


@needs_pin_so101
@pytest.mark.parametrize("lazy", [False, True], ids=["eager", "lazy"])
def test_a_valid_capsule_and_payload_build_on_a_map_keeps_its_order(events, lazy):
    """so101_left (capsule mask) and an isaac arm tracking the payload:
    nothing moves except the checks; the map registrations still follow the
    arm."""
    from cascade.apps import demo
    from cascade.config import load_demo_config

    cfg = load_demo_config(arm="so101_left", camera="mock", llm="mock")
    _, safe, _ = demo._build_arm(cfg.arm, lazy, _map(), cfg)
    arm_calls = [] if lazy else [("make", "so101_left"), ("connect", "so101_left")]
    assert events == [("harness", 0.22), *arm_calls, ("add_robot_body",)]
    assert safe.body_mask is None

    events.clear()
    _track_payload(cfg)
    cfg.arm._data["type"] = "isaac"
    occ = _map()
    demo._build_arm(cfg.arm, lazy, occ, cfg)
    assert events == [("harness", 0.22), *arm_calls, ("add_robot_body",), ("track_payload",)]
    assert occ.tracks_payload
