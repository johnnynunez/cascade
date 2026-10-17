"""An arm with a `base_pose` reads the shared occupancy map in the MAP's frame.

`build_runtime` hands ONE map to every arm, and that map lives in the shared
TABLE frame: body masks, the attached-object gate and the tool-volume gate all
lift arm points by `base_pose` before reading it. The TCP / joint-origin gate
used to read RAW base-frame FK points, so on the so101 pair (base_pose y
+-0.22 m) a TCP or elbow physically ON an obstacle was admitted and the same
part 22 cm away was refused, in vet_pose, vet_step and approve. Pinned here:

1. the arm gate reads `base_pose @ FK(q)` in all three phases, and so does the
   spatial tools' wrist-camera box check;
2. the default map region covers every arm's workspace in the table frame;
3. the grasp exemption (given in the arm's BASE frame, where its IK targets
   live) is compared in the frame of the lifted points and of the voxels: it
   still covers the arm's own target and nothing 22 cm away;
4. an arm without `base_pose` (the live reBot, the kitchen sim) is unchanged;
5. the tool volume's table band (`table_z` + `table_band_m`, a base-frame
   height) is compared with map-frame voxels only after lifting it by
   `base_pose`, and a tilted mount with a tool volume refuses to build.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import needs_pin_so101
from occupancy_grid_bridge import GridBridge, centred_box, depth_frame, make_map

from cascade.safety.harness import SafetyHarness, SafetyLimits
from cascade.safety.tool_volume import ToolVolume
from cascade.types import SafetyViolation, pose_to_transform

T_CAM = np.array([[0.0, 0.0, -1.0, 1.2], [1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.3], [0, 0, 0, 1.0]])

#: the so101 pair's geometry, and a translated + yawed mount (rotation matters)
POSES = {
    "y-0.22": [0.0, -0.22, 0.0, 0.0, 0.0, 0.0],
    "y-0.22 yaw0.4": [0.05, -0.22, 0.02, 0.0, 0.0, 0.4],
}
#: base-frame offset of each sampled arm part from q (synthetic kinematics)
PARTS = {"tcp": np.zeros(3), "link 2": np.array([0.0, 0.0, -0.08])}
PHASES = ("vet_pose", "vet_step", "approve")
#: a table-frame obstacle, well above the table and inside the map region
W = np.array([0.25, -0.10, 0.20])


class Kin:
    """TCP = q (identity rotation); joint origins: base, 8 cm below the TCP, TCP."""

    joint_limits = (np.full(3, -2.0), np.full(3, 2.0))

    def fk(self, q):
        T = np.eye(4)
        T[:3, 3] = q
        return T

    def link_positions(self, q):
        q = np.asarray(q, dtype=float)
        return np.array([[0.0, 0.0, 0.05], q - [0.0, 0.0, 0.08], q])


def lift(T, p):
    return (np.asarray(T, dtype=float) @ np.r_[np.asarray(p, dtype=float), 1.0])[:3]


def voxel_map(*points, size=0.01):
    """Occupied cubes of `size` centred on `points` (table frame), optional map."""
    bridge = GridBridge([centred_box(p, [size] * 3) for p in points], voxel=0.01)
    occ = make_map(bridge, required=False, unobserved="free")
    occ.refresh(depth_frame(), T_CAM)
    assert occ.last_error is None and occ.usable()
    return occ


def synthetic(base_pose, occ, *, tool=None):
    limits = SafetyLimits(np.array([-1.0, -1.0, -0.01]), np.array([1.0, 1.0, 1.0]))
    return SafetyHarness(limits, Kin(), occupancy=occ, base_pose=base_pose, tool_volume=tool)


def decide(h, phase, q, *, q_prev=None, exempt=None):
    """None = admitted, else the refusal text. vet_pose takes `exempt` as its
    kwargs; the stepping phases come from 5 cm above unless `q_prev` is given
    and use whatever exemption the harness holds."""
    q = np.asarray(q, dtype=float)
    q_prev = q + [0.0, 0.0, 0.05] if q_prev is None else np.asarray(q_prev, dtype=float)
    h.heartbeat()
    try:
        if phase == "vet_pose":
            return h.vet_pose(q, **(exempt or {}))
        if phase == "vet_step":
            return h.vet_step(q_prev, q, 1.0)
        h.approve(q_prev, q, 1.0)
        return None
    except SafetyViolation as exc:
        return str(exc)


# ── 1. the arm gate reads the lifted points ──────────────────────────────


@pytest.mark.parametrize("phase", PHASES)
@pytest.mark.parametrize("part", sorted(PARTS))
@pytest.mark.parametrize("pose", sorted(POSES))
def test_a_part_physically_on_a_voxel_is_refused(pose, part, phase):
    Tb = pose_to_transform(POSES[pose])
    q = lift(np.linalg.inv(Tb), W) - PARTS[part]           # base_pose @ part(q) == W
    assert np.allclose(lift(Tb, q + PARTS[part]), W)
    reason = decide(synthetic(Tb, voxel_map(W)), phase, q)
    assert reason is not None and "occupancy map" in reason, reason


@pytest.mark.parametrize("phase", PHASES)
@pytest.mark.parametrize("part", sorted(PARTS))
@pytest.mark.parametrize("pose", sorted(POSES))
def test_a_part_whose_base_frame_numbers_equal_the_voxel_is_admitted(pose, part, phase):
    """The phantom: the part's RAW base-frame coordinates equal W, but the part
    itself is 13-22 cm from it."""
    Tb = pose_to_transform(POSES[pose])
    q = W - PARTS[part]
    assert np.linalg.norm(lift(Tb, q + PARTS[part]) - W) > 0.10
    assert decide(synthetic(Tb, voxel_map(W)), phase, q) is None


@pytest.mark.parametrize("phase", PHASES)
@pytest.mark.parametrize("base_pose", [None, np.eye(4)], ids=["none", "identity"])
def test_without_a_base_pose_the_base_frame_is_the_map_frame(base_pose, phase):
    h = synthetic(base_pose, voxel_map(W))
    assert "occupancy map" in (decide(h, phase, W) or "")
    assert decide(synthetic(base_pose, voxel_map(W)), phase, W + [0.0, 0.22, 0.0]) is None


# ── 2. the real so101 pair, both arms on one shared map ──────────────────


@pytest.fixture
def so101_pair():
    """Both shipped so101 arms built by `_build_arm` on ONE map, as
    build_runtime does (no build_runtime: no bridge probe, no watcher)."""
    from cascade.apps import demo
    from cascade.config import load_demo_config

    cfg = load_demo_config(arms=["so101_left", "so101_right"], camera="mock", llm="mock")
    occ = voxel_map([0.0, 0.0, 0.5])                                   # a far placeholder voxel
    arms = {}
    for acfg in demo._arm_cfgs(cfg):
        raw, safe, kin = demo._build_arm(acfg, False, occ, cfg)
        arms[str(acfg.get("name"))] = (raw, safe, kin)
    try:
        yield occ, arms
    finally:
        for raw, _, _ in arms.values():
            raw.disconnect()


def _put_voxel(occ, point, size=0.01):
    occ._client.boxes = [centred_box(point, [size] * 3)]
    occ.refresh(depth_frame(), T_CAM)
    assert occ.last_error is None and occ.usable()


@needs_pin_so101
@pytest.mark.parametrize("phase", PHASES)
@pytest.mark.parametrize("part", ["tcp", "link 3"])
@pytest.mark.parametrize("name", ["so101_right", "so101_left"])
def test_so101_pair_gates_each_arm_where_it_physically_is(so101_pair, name, part, phase):
    occ, arms = so101_pair
    raw, safe, kin = arms[name]
    h = safe.harness
    assert h.base_pose is not None and abs(abs(h.base_pose[1, 3]) - 0.22) < 1e-9
    q = np.asarray(raw.get_state().q, dtype=float)
    base_point = kin.fk(q)[:3, 3] if part == "tcp" else kin.link_positions(q)[2]
    table_point = lift(h.base_pose, base_point)

    _put_voxel(occ, table_point)                         # ON the part: refused
    reason = decide(h, phase, q, q_prev=q)
    assert reason is not None and "occupancy map" in reason, reason

    _put_voxel(occ, base_point)                          # 22 cm from the part: admitted
    assert decide(h, phase, q, q_prev=q) is None


@needs_pin_so101
@pytest.mark.parametrize("phase", PHASES)
def test_so101_grasp_exemption_covers_the_arms_own_target(so101_pair, phase):
    """The descent cylinder is centred on the TCP target in the arm's BASE
    frame (`allow_grasp_descent(grasp.position[:2])`): the target object sits
    in the map at base_pose @ that point, and must stay exempt."""
    occ, arms = so101_pair
    raw, safe, kin = arms["so101_right"]
    h = safe.harness
    q = np.asarray(raw.get_state().q, dtype=float)
    target = kin.fk(q)[:3, 3]
    _put_voxel(occ, lift(h.base_pose, target), size=0.02)
    assert "occupancy map" in (decide(h, phase, q, q_prev=q) or "")    # premise: the object blocks
    exempt = {"exempt_xy": target[:2], "exempt_radius_m": 0.07, "exempt_z_min": -0.06}
    h.allow_grasp_descent(target[:2], radius_m=0.07, z_min=-0.06)
    try:
        assert decide(h, phase, q, q_prev=q, exempt=exempt) is None
    finally:
        h.clear_grasp_exemption()


# ── 3. the default map region spans every arm's table-frame workspace ────


@needs_pin_so101
def test_default_map_region_covers_each_arms_outer_strip(monkeypatch, tmp_path):
    from cascade.apps.demo import build_runtime, shutdown_runtime
    from cascade.config import load_demo_config
    from cascade.perception.occupancy import OccupancyMap

    captured = {}

    def from_config(block, workspace_min=None, workspace_max=None):
        lo, hi = np.asarray(workspace_min, dtype=float), np.asarray(workspace_max, dtype=float)
        occ = OccupancyMap(client=GridBridge([], voxel=0.01), region_min=lo, region_max=hi, required=False)
        occ.probe()
        captured.update(region=(lo, hi), map=occ)
        return occ

    monkeypatch.setattr(OccupancyMap, "from_config", staticmethod(from_config))
    cfg = load_demo_config(arms=["so101_left", "so101_right"], camera="mock", llm="mock")
    runtime, arm = build_runtime(cfg, tmp_path / "run", view=False, serve=False)
    try:
        lo, hi = captured["region"]
        occ = captured["map"]
        far = np.array([0.2, -0.52, 0.1])                 # so101_right's base y -0.30, on the table
        assert np.all(far >= lo) and np.all(far <= hi), (lo, hi)

        right = runtime.arm_rig.get("so101_right")
        h = right.harness
        q = np.asarray(right.get_state().q, dtype=float)
        q[0] += 1.5                                        # pan outward, away from the left arm
        tcp = lift(h.base_pose, h.kin.fk(q)[:3, 3])
        assert tcp[1] < -0.36                              # outside the old base-frame union
        _put_voxel(occ, tcp + [0.0, 0.0, 0.5])            # control: a clear map admits the pose
        assert h.vet_pose(q) is None
        occ._client.boxes = [centred_box(far, [0.04] * 3), centred_box(tcp, [0.01] * 3)]
        occ.refresh(depth_frame(), T_CAM)
        assert occ.last_error is None
        assert occ.clearance(far[None, :])[0] < 0.01        # mapped, not "outside the map = inf"
        reason = h.vet_pose(q)
        assert reason is not None and "occupancy map" in reason, reason
    finally:
        shutdown_runtime(runtime, arm)


def test_default_map_region_is_unchanged_for_arms_without_base_pose(monkeypatch, tmp_path):
    from cascade.apps.demo import build_runtime
    from cascade.config import load_demo_config
    from cascade.perception.occupancy import OccupancyMap

    class Captured(Exception):
        pass

    def from_config(block, workspace_min=None, workspace_max=None):
        raise Captured(workspace_min, workspace_max)

    monkeypatch.setattr(OccupancyMap, "from_config", staticmethod(from_config))
    cfg = load_demo_config(camera="mock", llm="mock")
    assert cfg.arm.get("base_pose") is None
    with pytest.raises(Captured) as got:
        build_runtime(cfg, tmp_path / "run", view=False, serve=False)
    ws = cfg.safety.workspace
    assert [float(v) for v in got.value.args[0]] == [float(v) for v in ws.min]
    assert [float(v) for v in got.value.args[1]] == [float(v) for v in ws.max]


# ── 4. the exemption is compared in the frame of the lifted points ───────
#
# g: the grasp target in the arm's BASE frame (the frame of its IK targets and
# of the exemption cylinder); G = base_pose @ g is where that target is in the
# map. g_raw: g's numbers read as a TABLE point, i.e. where a cylinder taken in
# the wrong frame would sit, 13-22 cm from the real target.

g = np.array([0.25, 0.12, 0.14])
CYL = {"exempt_xy": g[:2], "exempt_radius_m": 0.07, "exempt_z_min": 0.0}
JAW = np.array([0.12, 0.0, 0.0])          # one tool-volume sample 12 cm ahead of the TCP
HELD = np.array([0.0, 0.0, -0.15])        # one attached-object sample 15 cm below the TCP


def _jaws():
    return ToolVolume(local=JAW[None, :], part=np.array([0]), names=("jaws",))


def _with_exemption(h, phase):
    if phase != "vet_pose":
        h.allow_grasp_descent(g[:2], radius_m=0.07, z_min=0.0)
    return h


@pytest.mark.parametrize("phase", PHASES)
@pytest.mark.parametrize("pose", sorted(POSES))
def test_arm_gate_exemption_covers_the_target_and_nothing_where_the_wrong_frame_puts_it(pose, phase):
    Tb = pose_to_transform(POSES[pose])
    G, g_raw = lift(Tb, g), g
    # the TCP descends onto its own target: exempt
    h = _with_exemption(synthetic(Tb, voxel_map(G)), phase)
    assert decide(h, phase, g, exempt=CYL) is None
    assert "occupancy map" in (decide(synthetic(Tb, voxel_map(G)), phase, g) or "")     # premise
    # a TCP physically at g_raw, on a voxel there: outside the real cylinder
    q = lift(np.linalg.inv(Tb), g_raw)
    assert np.linalg.norm(q[:2] - g[:2]) > 0.10
    h = _with_exemption(synthetic(Tb, voxel_map(g_raw)), phase)
    reason = decide(h, phase, q, exempt=CYL)
    assert reason is not None and "occupancy map" in reason, reason


@pytest.mark.parametrize("phase", PHASES)
@pytest.mark.parametrize("pose", sorted(POSES))
def test_tool_volume_exemption_is_taken_in_the_arms_base_frame(pose, phase):
    Tb = pose_to_transform(POSES[pose])
    G, g_raw = lift(Tb, g), g
    # the jaws on the target: the target's voxels are excluded
    q = g - JAW
    h = _with_exemption(synthetic(Tb, voxel_map(G, size=0.02), tool=_jaws()), phase)
    assert decide(h, phase, q, exempt=CYL) is None
    assert "jaws" in (decide(synthetic(Tb, voxel_map(G, size=0.02), tool=_jaws()), phase, q) or "")   # premise
    # the jaws physically on a voxel at g_raw: never excluded
    q = lift(np.linalg.inv(Tb), g_raw) - JAW
    h = _with_exemption(synthetic(Tb, voxel_map(g_raw, size=0.02), tool=_jaws()), phase)
    reason = decide(h, phase, q, exempt=CYL)
    assert reason is not None and "jaws" in reason, reason


class _MapWithoutExcluding:
    """A map stand-in without clearance_excluding: the tool volume falls back
    to its sample-based exemption."""

    required = False
    unknown_blocks = False

    def __init__(self, occ):
        self._occ = occ

    def clearance(self, points):
        return self._occ.clearance(points)


@pytest.mark.parametrize("phase", PHASES)
@pytest.mark.parametrize("pose", sorted(POSES))
def test_tool_volume_sample_exemption_is_taken_in_the_arms_base_frame(pose, phase):
    Tb = pose_to_transform(POSES[pose])
    G, g_raw = lift(Tb, g), g
    q = g - JAW
    h = _with_exemption(synthetic(Tb, _MapWithoutExcluding(voxel_map(G)), tool=_jaws()), phase)
    assert decide(h, phase, q, exempt=CYL) is None
    q = lift(np.linalg.inv(Tb), g_raw) - JAW
    h = _with_exemption(synthetic(Tb, _MapWithoutExcluding(voxel_map(g_raw)), tool=_jaws()), phase)
    reason = decide(h, phase, q, exempt=CYL)
    assert reason is not None and "jaws" in reason, reason


@pytest.mark.parametrize("phase", PHASES)
@pytest.mark.parametrize("pose", sorted(POSES))
def test_attached_object_exemption_is_taken_in_the_arms_base_frame(pose, phase):
    Tb = pose_to_transform(POSES[pose])
    G, g_raw = lift(Tb, g), g

    def held_map(point):
        occ = voxel_map(point)
        occ._payload_samples["scene"] = HELD[None, :]
        return occ

    q = g - HELD                                       # the held sample on the target: exempt
    h = _with_exemption(synthetic(Tb, held_map(G)), phase)
    assert decide(h, phase, q, exempt=CYL) is None
    assert (decide(synthetic(Tb, held_map(G)), phase, q) or "").startswith("attached object")   # premise
    q = lift(np.linalg.inv(Tb), g_raw) - HELD          # the held sample on a voxel at g_raw
    h = _with_exemption(synthetic(Tb, held_map(g_raw)), phase)
    reason = decide(h, phase, q, exempt=CYL)
    assert reason is not None and reason.startswith("attached object"), reason


# ── 5. the spatial tools' wrist-camera box reads the lifted map too ──────


@pytest.mark.parametrize("pose", sorted(POSES))
def test_spatial_wrist_camera_box_is_checked_where_it_physically_is(pose):
    from types import SimpleNamespace

    from cascade.skills import relative_motion as rm
    from cascade.skills import spatial_moves

    Tb = pose_to_transform(POSES[pose])
    T = Kin().fk(np.array([0.25, 0.10, 0.20]))
    s = SimpleNamespace(wrist_camera=(0.08, 0.14, 0.075), tool_up_axis=(0.0, 0.0, 1.0), camera_occupancy=True)
    box = rm.camera_box(T, np.eye(3)[rm._TOOL_COLUMNS["down_open"][0]], s.tool_up_axis,
                        protrusion_m=0.08, behind_m=0.14, half_width_m=0.075)

    def camera_volume(voxel):
        h = synthetic(Tb, voxel_map(voxel))
        rt = SimpleNamespace(_tool_axis_order="down_open", arm=SimpleNamespace(harness=h))
        return spatial_moves._camera_volume(rt, [T], s)

    found = camera_volume(lift(Tb, box.center))           # the box physically on a voxel
    assert found is not None and found[0] == "wrist_camera" and "occupancy map" in found[1], found
    assert camera_volume(box.center) is None              # the box's base-frame numbers: 13-22 cm off


# ── 6. the tool volume's table band is a height in the MAP frame ─────────
#
# `clearance_excluding` drops occupied voxels at or below `below_z` (the table
# plane plus `table_band_m`), and those voxels are map-frame points, while
# `table_z` is a base-frame height. Every scene below has its physical table
# top at map z = 0, so an arm mounted `bz` above the table frame states
# `table_z = -bz`. Read unlifted, the band sits `bz` too high for a base below
# the table frame (a real object above the band is dropped: fail-open) and
# `bz` too low for one above it (the table itself blocks: fail-closed).

#: one tool-volume sample 12 cm ahead of and 10 cm below the TCP: the TCP and
#: the elbow (8 cm under it) stay > 3 cm from the table and from the object
LOW_JAW = np.array([0.12, 0.0, -0.10])
#: a 2 cm object resting 2.5 cm above the table top: voxel centres at map z
#: 0.025..0.045, above the 2 cm band, so it is an obstacle for every mount
OBJECT = np.array([0.30, -0.15, 0.035])
BESIDE_OBJECT = OBJECT + [0.0, 0.025, 0.0]          # 1.5 cm from its face
OVER_TABLE = np.array([0.30, -0.125, 0.02])         # 2 cm over the bare table: inside the band's reach
LEVEL = {
    "base 5 cm below": [0.0, -0.22, -0.05, 0.0, 0.0, 0.0],
    "base 5 cm below yaw0.4": [0.05, -0.22, -0.05, 0.0, 0.0, 0.4],
    "base 5 cm above": [0.0, -0.22, 0.05, 0.0, 0.0, 0.0],
    "base 5 cm above yaw0.4": [0.05, -0.22, 0.05, 0.0, 0.0, 0.4],
}
MOUNTS = {"none": None, "identity": np.eye(4), **{k: pose_to_transform(v) for k, v in LEVEL.items()}}


def _low_jaws():
    return ToolVolume(local=LOW_JAW[None, :], part=np.array([0]), names=("jaws",))


def table_scene(*objects):
    """The table layer at map z = 0 plus optional 2 cm objects (optional map)."""
    from occupancy_grid_bridge import TABLE

    bridge = GridBridge([TABLE, *[centred_box(p, [0.02] * 3) for p in objects]], voxel=0.01)
    occ = make_map(bridge, required=False, unobserved="free")
    occ.refresh(depth_frame(), T_CAM)
    assert occ.last_error is None and occ.usable()
    return occ


def on_the_table(base_pose, occ):
    """A tool-volume harness whose `table_z` puts the table top at map z = 0."""
    bz = 0.0 if base_pose is None else float(base_pose[2, 3])
    limits = SafetyLimits(np.array([-1.0, -1.0, -0.2]), np.array([1.0, 1.0, 1.0]), table_z=-bz)
    return SafetyHarness(limits, Kin(), occupancy=occ, base_pose=base_pose, tool_volume=_low_jaws())


def jaw_at(base_pose, point):
    """The q whose tool sample sits at map point `point`."""
    Tb = np.eye(4) if base_pose is None else base_pose
    q = lift(np.linalg.inv(Tb), point) - LOW_JAW
    assert np.allclose(_low_jaws().points(Tb @ Kin().fk(q)), [point])
    return q


@pytest.mark.parametrize("phase", PHASES)
@pytest.mark.parametrize("mount", sorted(MOUNTS))
def test_an_object_above_the_table_band_blocks_the_tool_volume(mount, phase):
    """Fail-open before the fix for a base below the table frame: the band
    read 5 cm too high and dropped the object's voxels."""
    Tb = MOUNTS[mount]
    q = jaw_at(Tb, BESIDE_OBJECT)
    reason = decide(on_the_table(Tb, table_scene(OBJECT)), phase, q)
    assert reason is not None and "jaws" in reason, reason
    assert decide(on_the_table(Tb, table_scene()), phase, q) is None        # premise: the object decides


@pytest.mark.parametrize("phase", PHASES)
@pytest.mark.parametrize("mount", sorted(MOUNTS))
def test_the_table_inside_its_band_does_not_block_the_tool_volume(mount, phase):
    """Fail-closed before the fix for a base above the table frame: the band
    read 5 cm too low and the table top itself blocked the jaws."""
    Tb = MOUNTS[mount]
    q = jaw_at(Tb, OVER_TABLE)
    assert decide(on_the_table(Tb, table_scene()), phase, q) is None
    occ = table_scene()
    assert occ.clearance(OVER_TABLE[None, :])[0] < 0.03                     # premise: the grid alone would block


@pytest.mark.parametrize("rpy", [(0.1, 0.0, 0.0), (0.0, -0.1, 0.0), (0.05, 0.05, 0.4)],
                         ids=["roll", "pitch", "roll+pitch+yaw"])
def test_a_tilted_mount_with_a_tool_volume_refuses_to_build(rpy):
    """No horizontal table band exists in the map for a tilted base: fail
    closed at construction instead of guessing one."""
    Tb = pose_to_transform([0.0, -0.22, 0.0, *rpy])
    with pytest.raises(ValueError, match="level base_pose"):
        on_the_table(Tb, table_scene())
    # without a tool volume there is no band to place: the arm still builds
    assert synthetic(Tb, table_scene()).base_pose is not None
    # and a level mount (any yaw, any height) builds with one
    for pose in LEVEL.values():
        assert on_the_table(pose_to_transform(pose), table_scene()).tool_volume is not None
