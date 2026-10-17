"""PR-A's own map reads lift an arm with a `base_pose` into the map's frame.

#348 made every arm gate read the shared (table-frame) occupancy map at
`base_pose @ FK(q)`. PR-A added map reads of its own on the same path, and
each must lift the same way (`SafetyHarness._arm_occupancy_violation`):

1. `route_end_violation` (a planned route never ENDS inside the margin);
2. the escape rule's samples: the same TCP + joint origins at q_prev, and the
   interior configurations of an escaping step (a step whose ends retreat
   but whose interior jumps a voxel);
3. the refusal reports (`drain_clearance_reports`): `sample`, `sample_tcp`
   and the snapshot centre in the map frame, plus `base_pose`;
4. `scripts/replay_clearance.py` re-vets a report with the recorded
   `base_pose` exactly as the live gate read it (`_static_reason` is
   vet_pose's map verdict on a base_pose arm too; ReplayMap takes the
   exemption's `cylinder_frame`).

Physical oracle: a voxel at `base_pose @ FK(q)` refuses, the same voxel at
the raw base-frame numbers `FK(q)` (13-22 cm from the arm) admits. Arms
without a base_pose are byte-identical (decision diff in the pre-final
report).
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from conftest import needs_pin
from test_base_pose_map_frame import PARTS, POSES, W, decide, lift, synthetic, voxel_map
from test_replay_clearance import WHOLE, at_x, lattice, rc, rebot, recorder, voxel_box  # noqa: F401 -- fixtures

from cascade.safety.trajectory import route_end_reason, vet_route
from cascade.types import pose_to_transform

STEP_PHASES = ("vet_step", "approve")


def at(Tb, p):
    """The synthetic arm's q whose TCP sits at map point `p`."""
    return lift(np.linalg.inv(Tb), p)


# ── 1. a route never ends inside the margin, where the arm really is ─────


@pytest.mark.parametrize("part", sorted(PARTS))
@pytest.mark.parametrize("pose", sorted(POSES))
def test_route_end_reads_the_arm_where_it_physically_is(pose, part):
    Tb = pose_to_transform(POSES[pose])
    q = at(Tb, W) - PARTS[part]                              # base_pose @ part(q) == W
    h = synthetic(Tb, voxel_map(W))
    reason = h.route_end_violation(q)
    assert reason is not None and "occupancy map" in reason, reason
    assert route_end_reason(h, q) == "route ends inside the margin: " + reason
    # the phantom: the part's raw base-frame numbers equal W, the part is far
    q_raw = W - PARTS[part]
    assert np.linalg.norm(lift(Tb, q_raw + PARTS[part]) - W) > 0.10
    assert synthetic(Tb, voxel_map(W)).route_end_violation(q_raw) is None


@pytest.mark.parametrize("pose", sorted(POSES))
def test_a_route_of_retreating_steps_that_ends_inside_the_margin_is_refused(pose):
    """Each step strictly retreats from W (escape admitted per step), but the
    route ends 2 cm from it: vet_route refuses at the end, where the arm is."""
    Tb = pose_to_transform(POSES[pose])
    start, mid, end = (at(Tb, W + [0.0, 0.0, dz]) for dz in (0.015, 0.018, 0.022))
    h = synthetic(Tb, voxel_map(W))
    h.heartbeat()
    assert h.vet_step(start, mid, 1.0) is None and h.vet_step(mid, end, 1.0) is None
    reason = vet_route(h, start, [mid, end], 2.0)
    assert reason is not None and reason.startswith("route ends inside the margin: point 0 clearance"), reason
    # out of the margin at the end: the same route is fine
    far = at(Tb, W + [0.0, 0.0, 0.04])                     # TCP and the elbow 8 cm under it: 3.5 cm
    assert vet_route(h, start, [mid, end, far], 3.0) is None


# ── 2. the escape rule compares the step in ONE frame ────────────────────


@pytest.mark.parametrize("phase", STEP_PHASES)
@pytest.mark.parametrize("pose", sorted(POSES))
def test_an_arm_inside_the_margin_may_retreat_where_it_physically_is(pose, phase):
    """The TCP 1.5 cm over W rises 0.5 cm: a strict retreat, admitted."""
    Tb = pose_to_transform(POSES[pose])
    q_prev, q = at(Tb, W + [0.0, 0.0, 0.02]), at(Tb, W + [0.0, 0.0, 0.025])
    h = synthetic(Tb, voxel_map(W))
    assert "occupancy map" in (h.vet_pose(q) or "")                     # premise: q is inside the margin
    assert decide(h, phase, q, q_prev=q_prev) is None
    assert "occupancy map" in (decide(h, phase, q_prev, q_prev=q) or "")  # closing back in: refused


@pytest.mark.parametrize("phase", STEP_PHASES)
@pytest.mark.parametrize("pose", sorted(POSES))
def test_a_voxel_at_q_prevs_raw_numbers_never_makes_a_closing_step_an_escape(pose, phase):
    """Fail-open if q_prev were read raw: a voxel P sits ON the raw
    base-frame numbers of q_prev (22 cm from the arm), so a raw read puts the
    previous TCP at distance 0 and any next distance looks like a retreat.
    The arm really closes in on W, 2.3 -> 2.1 cm: refused."""
    Tb = pose_to_transform(POSES[pose])
    q_prev, q = at(Tb, W + [0.0, 0.0, 0.028]), at(Tb, W + [0.0, 0.0, 0.026])
    P = q_prev + [0.0, 0.0, 0.01]                  # 1 cm over q_prev's raw numbers: the raw step moves away
    assert np.linalg.norm(lift(Tb, q_prev) - P) > 0.10
    h = synthetic(Tb, voxel_map(W, P))
    reason = decide(h, phase, q, q_prev=q_prev)
    assert reason is not None and reason.startswith("point 0 clearance") and "mid-step" not in reason, reason


#: W2 sits on the straight path between the two ends of the escaping step
#: below, at its midpoint; both ends are 1.77 cm from it and the start is
#: 1.5 cm over W: each end retreats, the interior jumps W2
STEP_FROM, STEP_TO = np.array([0.0, 0.0, 0.02]), np.array([0.045, 0.0, 0.035])
W2 = W + (STEP_FROM + STEP_TO) / 2.0


@pytest.mark.parametrize("phase", STEP_PHASES)
@pytest.mark.parametrize("pose", sorted(POSES))
def test_an_escaping_step_that_jumps_a_voxel_is_sampled_where_it_moves(pose, phase):
    Tb = pose_to_transform(POSES[pose])
    q_prev, q = at(Tb, W + STEP_FROM), at(Tb, W + STEP_TO)
    assert decide(synthetic(Tb, voxel_map(W)), phase, q, q_prev=q_prev) is None   # premise: an escape
    reason = decide(synthetic(Tb, voxel_map(W, W2)), phase, q, q_prev=q_prev)
    assert reason is not None and "mid-step" in reason, reason


@pytest.mark.parametrize("phase", STEP_PHASES)
@pytest.mark.parametrize("pose", sorted(POSES))
def test_a_point_that_started_inside_the_exemption_gets_no_escape_note(pose, phase):
    """The note "inside the margin: only a strictly retreating step" names a
    point that was inside the margin OUTSIDE the exemption at q_prev. Here
    the only such point (the TCP, 2 cm over W) sat inside the base-frame
    cylinder, then leaves it closing in: refused, without the note."""
    Tb = pose_to_transform(POSES[pose])
    q_prev = at(Tb, W + [0.0, 0.0, 0.025])
    q = q_prev + [0.012, 0.0, -0.004]                     # 1.2 cm out of the 1 cm cylinder, closer to W
    h = synthetic(Tb, voxel_map(W))
    h.allow_grasp_descent(q_prev[:2], radius_m=0.01, z_min=float(q_prev[2]) - 0.2)
    try:
        reason = decide(h, phase, q, q_prev=q_prev)
    finally:
        h.clear_grasp_exemption()
    assert reason is not None and reason.startswith("point 0 clearance"), reason
    assert "inside the margin" not in reason, reason


# ── 2b. the air-grasp exit's ceiling is a base-frame height ──────────────
#
# The exit holds the approach column capped at z_max = hover TCP z + 5 mm,
# both in the arm's BASE frame. For a mount 5 cm below the table frame, a
# voxel 2 cm ABOVE the ceiling (base frame) has map z 3 cm BELOW it: compared
# in the wrong frame it would be exempt (fail-open).

LOW_MOUNT = pose_to_transform([0.0, -0.22, -0.05, 0.0, 0.0, 0.0])
COLUMN = (np.array([0.25, 0.12]), 0.07, 0.0)
CEILING = 0.20
ABOVE = np.array([0.25, 0.12, CEILING + 0.02])        # base frame, inside the column's footprint


@pytest.mark.parametrize("phase", STEP_PHASES)
def test_the_exit_ceiling_checks_a_tool_sample_above_it_in_the_base_frame(phase):
    from test_base_pose_map_frame import JAW, _jaws

    V = lift(LOW_MOUNT, ABOVE)
    assert V[2] < CEILING                               # its map z reads under the ceiling
    q = ABOVE + [0.015, 0.0, 0.0] - JAW                 # the jaw sample 1.5 cm beside it
    q_prev = q + [0.01, 0.0, 0.0]
    h = synthetic(LOW_MOUNT, voxel_map(V), tool=_jaws())
    with h.target_column_exemption(COLUMN, z_max=CEILING):
        reason = decide(h, phase, q, q_prev=q_prev)
    assert reason is not None and reason.startswith("jaws clearance"), reason
    # without the ceiling the column exempts it (the approach's unbounded column)
    with h.target_column_exemption(COLUMN):
        assert decide(h, phase, q, q_prev=q_prev) is None


@pytest.mark.parametrize("phase", STEP_PHASES)
def test_the_exit_ceiling_checks_an_arm_point_above_it_in_the_base_frame(phase):
    q = ABOVE.copy()                                    # the TCP above the ceiling, inside the footprint
    assert lift(LOW_MOUNT, q)[2] < CEILING
    V = lift(LOW_MOUNT, q + [0.02, 0.0, 0.0])           # a voxel 1.5 cm from it
    q_prev = q - [0.01, 0.0, 0.0]
    h = synthetic(LOW_MOUNT, voxel_map(V))
    with h.target_column_exemption(COLUMN, z_max=CEILING):
        reason = decide(h, phase, q, q_prev=q_prev)
    assert reason is not None and reason.startswith("point 0 clearance"), reason
    with h.target_column_exemption(COLUMN):
        assert decide(h, phase, q, q_prev=q_prev) is None


# ── 3. reports carry map-frame samples (+ base_pose) ─────────────────────


def _one_report(h):
    (r,) = h.drain_clearance_reports()
    return json.loads(json.dumps(r))


def _assert_map_frame(r, Tb, T_base, sample):
    """`sample` is the map-frame point, `sample_tcp` the same point in the
    TCP frame (base_pose @ FK), the snapshot is centred on it and its
    nearest voxel is measured from it."""
    np.testing.assert_allclose(r["sample"], sample, atol=1e-9)
    T_map = Tb @ T_base
    np.testing.assert_allclose(r["sample_tcp"], (np.linalg.inv(T_map) @ [*sample, 1.0])[:3], atol=1e-9)
    assert r["base_pose"] == np.round(Tb, 6).tolist()
    assert r["snapshot"]["centre"] == r["sample"]
    vox = np.asarray(r["voxels"], dtype=float)
    assert len(vox) and np.allclose(vox[:, 3], np.round(np.linalg.norm(vox[:, :3] - sample, axis=1), 3),
                                    atol=1.5e-3)
    assert vox[0, 3] < 0.03


@pytest.mark.parametrize("pose", sorted(POSES))
def test_route_end_and_mid_step_reports_carry_map_frame_samples(pose):
    Tb = pose_to_transform(POSES[pose])
    # route end: the TCP on W
    q = at(Tb, W + [0.0, 0.0, 0.015])
    h = synthetic(Tb, voxel_map(W))
    assert h.route_end_violation(q)
    r = _one_report(h)
    assert r["phase"] == "route_end" and r["gate"] == "tcp"
    _assert_map_frame(r, Tb, h.kin.fk(q), W + [0.0, 0.0, 0.015])
    # an escaping step that jumps W2: the interior sample that refused
    q_prev, q = at(Tb, W + STEP_FROM), at(Tb, W + STEP_TO)
    h = synthetic(Tb, voxel_map(W, W2))
    h.heartbeat()
    reason = h.vet_step(q_prev, q, 1.0)
    assert "mid-step" in (reason or ""), reason
    r = _one_report(h)
    assert r["phase"] == "vet_step" and r["gate"] == "tcp"
    sample = np.asarray(r["sample"], dtype=float)
    # on the lifted straight path between the two ends, never on the raw one
    a, b = W + STEP_FROM, W + STEP_TO
    t = float(np.dot(sample - a, b - a) / np.dot(b - a, b - a))
    assert 0.0 < t < 1.0 and np.linalg.norm(a + t * (b - a) - sample) < 1e-9
    _assert_map_frame(r, Tb, h.kin.fk(q), sample)               # sample_tcp: in the gate's TCP frame (q)


# ── 4. the offline replay lifts with the recorded base_pose ──────────────

#: a level mount (tool volumes refuse a tilted one) with a yaw: a point near
#: the reBot's TCP lies ~30 cm from its raw base-frame numbers
REBOT_MOUNT = pose_to_transform([0.05, 0.12, 0.02, 0.0, 0.0, 0.4])
#: the same with a small yaw: a map point's numbers read in the base frame
#: land 15+ cm from where the base frame puts that point near the reBot's TCP
TOOL_MOUNT = pose_to_transform([0.0, -0.22, 0.02, 0.0, 0.0, 0.1])


@needs_pin
def test_a_base_pose_tcp_refusal_replays_where_it_was_recorded(rc, rebot):  # noqa: F811
    q = at_x(rebot, 0.4498)
    T_map = REBOT_MOUNT @ rebot.kin.fk(q)
    centre = lattice(T_map[:3, 3] + T_map[:3, 0] * 0.02)          # 2 cm ahead of the fingertips
    h, _ = recorder(rebot, voxel_box(centre), base_pose=REBOT_MOUNT, snapshot=WHOLE)
    reason = h.vet_pose(q)
    assert reason is not None and reason.startswith("point 0 clearance"), reason
    rep = _one_report(h)
    np.testing.assert_allclose(rep["sample"], T_map[:3, 3], atol=1e-9)
    out = rc.replay(rep, arm="rebot_rs")
    assert out["old"].startswith("refused: point 0 clearance"), out["old"]
    assert out["replay_clearance_m"] == pytest.approx(rep["clearance_m"], abs=1e-6)
    assert out["model_mismatch"] is False
    # the same voxel at the TCP's raw base-frame numbers: 22 cm from the arm
    h, _ = recorder(rebot, voxel_box(lattice(rebot.kin.fk(q)[:3, 3] + [0.02, 0.0, 0.0])),
                    base_pose=REBOT_MOUNT, snapshot=WHOLE)
    assert h.vet_pose(q) is None


@needs_pin
def test_a_tool_refusal_under_an_exemption_replays_with_the_cylinder_in_the_base_frame(rc, rebot):  # noqa: F811
    """The exemption cylinder is in the arm's base frame: the live gate takes
    voxels into that frame (`cylinder_frame`), and so must the replay, both
    for the verdict and for the re-measured clearance at the sample. The
    cylinder here sits on the camera voxel's MAP numbers read as base-frame
    ones (the wrong frame would exempt it)."""
    q = at_x(rebot, 0.4498)
    tcp = rebot.kin.fk(q)[:3, 3]
    centre = lattice(lift(TOOL_MOUNT, tcp + [-0.015, 0.0, 0.07]))   # over the camera box, as placed live
    cyl = (centre[:2].copy(), 0.03, float(centre[2]) - 0.05)
    assert np.linalg.norm(lift(np.linalg.inv(TOOL_MOUNT), centre)[:2] - cyl[0]) > 0.10
    h, _ = recorder(rebot, voxel_box(centre), base_pose=TOOL_MOUNT, snapshot=WHOLE)
    reason = h.vet_pose(q, *cyl)
    assert reason is not None and reason.startswith("wrist camera clearance"), reason
    rep = _one_report(h)
    assert rep["exemption"] == {"cylinder": [float(cyl[0][0]), float(cyl[0][1]), cyl[1], cyl[2]]}
    out = rc.replay(rep, arm="rebot_rs")
    assert out["old"].startswith("refused: wrist camera clearance"), out["old"]
    assert out["replay_clearance_m"] == pytest.approx(rep["clearance_m"], abs=1e-6)


@needs_pin
def test_the_static_replay_check_is_vet_poses_map_verdict_on_a_base_pose_arm(rc, rebot):  # noqa: F811
    """`_static_reason` (what `old` and L4 run) against vet_pose on a
    base_pose reBot over random poses, obstacles placed where the arm
    physically is, with and without an exemption (base frame) and a held
    payload."""
    from collections import Counter

    from test_replay_clearance import OBJ, live_map

    from cascade.safety.harness import SafetyHarness, SafetyLimits

    rng = np.random.default_rng(20261017)
    limits = SafetyLimits.from_config(rebot.cfg.safety)
    seen = Counter()
    for scene in range(24):
        q0 = (rebot.home, rebot.q_pre, rebot.q_grasp)[scene % 3] + rng.normal(0, 0.1, 6)
        T0 = REBOT_MOUNT @ rebot.kin.fk(q0)
        boxes = []
        for _ in range(int(rng.integers(1, 3))):
            c = T0[:3, 3] + rng.normal(0, 0.05, 3)
            boxes.append((tuple(c - 0.01), tuple(c + 0.01)))
        occ = live_map(*boxes)
        if scene % 2:
            occ._payload_samples["scene"] = np.array([[0.0, 0.0, -0.1], [0.02, 0.0, -0.08]])
        h = SafetyHarness(limits, kinematics=rebot.kin, occupancy=occ, base_pose=REBOT_MOUNT,
                          tool_volume=rebot.volume)
        exempt = None if scene % 4 < 2 else (OBJ[:2].copy(), 0.07, -0.06 if scene % 8 < 4 else 0.0)
        for _ in range(6):
            q = q0 + rng.normal(0, 0.01, 6)
            reason = h.vet_pose(q, *(() if exempt is None else exempt))
            filed = h.drain_clearance_reports()
            static = rc._static_reason(h, q, exempt)
            assert h.drain_clearance_reports() == []
            if filed:
                assert static == reason
                seen["map refusal " + filed[0]["gate"].split(" ")[0]] += 1
            elif reason is None:
                assert static is None
                seen["clear"] += 1
            else:
                seen["refused before the map"] += 1
    assert sum(n for k, n in seen.items() if k.startswith("map refusal")) >= 20 and seen["clear"] >= 5, seen
    assert "map refusal tcp" in seen, seen


@needs_pin
@pytest.mark.parametrize("bad", [[[1.0, 0.0], [0.0, 1.0]], [[float("nan")] * 4] * 4, "east",
                                 [[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 1.0, 0], [0, 0, 0, 2.0]]])
def test_a_malformed_base_pose_is_not_replayable(rc, rebot, bad):  # noqa: F811
    q = at_x(rebot, 0.4498)
    T_map = REBOT_MOUNT @ rebot.kin.fk(q)
    h, _ = recorder(rebot, voxel_box(lattice(T_map[:3, 3] + T_map[:3, 0] * 0.02)), base_pose=REBOT_MOUNT,
                    snapshot=WHOLE)
    assert h.vet_pose(q)
    rep = _one_report(h)
    rep["base_pose"] = bad
    out = rc.replay(rep, arm="rebot_rs")
    assert out["replayable"] is False and out["not_replayable"].startswith("bad base_pose"), out
