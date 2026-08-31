"""Generic MuJoCo arm backend, both physics runtimes.

Split on purpose:

  - the profile/name-mapping contract, which needs no MuJoCo and no assets and
    therefore runs everywhere. This is where the destructive mistakes live: a
    joint list that disagrees with `n_joints`, or index drift when a scene adds
    a prop, both mis-address the q vector silently.

  - real stepping, which needs `mujoco` (engine=mjc) or additionally
    `mujoco-warp` (engine=warp) AND the fetched MJCF + meshes (not vendored --
    ~17 MB). Skipped with a message naming the fetch command rather than passing
    vacuously.

The physics half is PARAMETRIZED over every engine available on the box, so the
MuJoCo Warp path gets exactly the same behavioural coverage as the C engine --
the repo's "engine agreement is the gold metric" rule applied to the sim
backend itself.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import REPO

from cascade.config import load_profile
from cascade.control.arm_base import make_arm
from cascade.control.mujoco_arm import MujocoArm, _select_warp_device

MJCF = REPO / "assets" / "mjcf" / "so101" / "scene.xml"


def has_mujoco() -> bool:
    try:
        import mujoco  # noqa: F401

        return True
    except ImportError:
        return False


def has_mjwarp() -> bool:
    try:
        import mujoco_warp  # noqa: F401
        import warp  # noqa: F401

        return True
    except ImportError:
        return False


needs_mujoco = pytest.mark.skipif(
    not has_mujoco() or not MJCF.exists(),
    reason="needs `mujoco` and `python scripts/fetch_robot_assets.py so101`",
)

# Every engine we can actually step on this box. `mjc` rides on `mujoco`;
# `warp` additionally needs `mujoco-warp` + `warp-lang`. Parametrizing here
# means each physics test below runs once per available engine and the IDs read
# `...[mjc]` / `...[warp]` in the report.
_ENGINES = []
if has_mujoco() and MJCF.exists():
    _ENGINES.append("mjc")
    if has_mjwarp():
        _ENGINES.append("warp")


@pytest.fixture
def profile():
    return load_profile("arms", "so101_mujoco")


@pytest.fixture
def warp_profile():
    return load_profile("arms", "so101_mjwarp")


def _profile_with_engine(engine: str):
    """so101_mujoco with `engine` forced -- one MJCF, one contract, N runtimes.

    Using the mjc profile as the base for BOTH keeps the physics assertions
    identical across engines; only the runtime under them changes.
    """
    data = load_profile("arms", "so101_mujoco").as_dict()
    data["engine"] = engine
    from cascade.config import Cfg

    return Cfg(data)


@pytest.fixture(params=_ENGINES)
def engine_arm(request):
    """A connected arm on each available engine; disconnected on teardown."""
    arm = make_arm(_profile_with_engine(request.param))
    arm.connect()
    yield request.param, arm
    arm.disconnect()


# ── contract, no MuJoCo needed ───────────────────────────────────────────


def test_factory_routes_the_mujoco_type(profile):
    """Construction must not touch MuJoCo: the demo builds the arm object at
    startup and only materializes physics on connect()."""
    arm = make_arm(profile)
    assert isinstance(arm, MujocoArm)
    assert arm.n_joints == 5
    # Default engine is the C runtime -- the one that runs fast anywhere.
    assert arm._engine_kind == "mjc"


def test_mjwarp_profile_is_the_same_arm_on_the_warp_engine(warp_profile):
    """so101_mjwarp must inherit the ENTIRE so101 contract and change only the
    engine -- that is the whole point of `extends: so101_mujoco`. If any
    kinematic constant diverged, the two sims would describe different arms."""
    arm = make_arm(warp_profile)
    assert isinstance(arm, MujocoArm)
    assert arm._engine_kind == "warp"
    assert arm.n_joints == 5
    assert arm._joint_names == [
        "shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll",
    ]
    mjc = load_profile("arms", "so101_mujoco")
    # Everything that defines the robot is shared; only `engine` differs.
    for key in ("mj_joints", "mj_actuators", "mj_gripper_joint", "home_q",
                "n_joints", "substeps", "settle_tol", "settle_timeout_s"):
        assert warp_profile.get(key) == mjc.get(key), key
    assert warp_profile.get("engine") == "warp"
    assert mjc.get("engine") is None  # the default profile does not name one


def test_unknown_engine_is_rejected(profile):
    data = profile.as_dict()
    data["engine"] = "physx"  # not a MuJoCo runtime
    arm = MujocoArm(profile.__class__(data))
    with pytest.raises((ValueError, RuntimeError), match="engine"):
        arm.connect()


def test_warp_device_selection_degrades_without_cuda():
    """The README "Compute" contract: an explicit accelerator the host lacks
    WARNS and falls back to CPU, never kills the run. Simulated with a stub so
    the logic is tested even on a CUDA box (and on one without)."""
    class _NoCuda:
        def is_cuda_available(self):
            return False

    class _Cuda:
        def is_cuda_available(self):
            return True

    assert _select_warp_device(_NoCuda(), "auto") == "cpu"
    assert _select_warp_device(_NoCuda(), "cuda:0") == "cpu"   # degrade, not die
    assert _select_warp_device(_Cuda(), "auto") == "cuda:0"
    assert _select_warp_device(_Cuda(), "cuda:1") == "cuda:1"  # honor the index
    assert _select_warp_device(_Cuda(), "cpu") == "cpu"        # explicit cpu wins


def test_joint_list_must_agree_with_n_joints(profile):
    """They address the same q vector. A mismatch would otherwise truncate or
    broadcast every commanded pose."""
    data = profile.as_dict()
    data["mj_joints"] = data["mj_joints"][:3]
    with pytest.raises(ValueError, match="mj_joints"):
        MujocoArm(load_profile("arms", "so101_mujoco").__class__(data))


def test_actuator_list_must_agree_with_n_joints(profile):
    data = profile.as_dict()
    data["mj_actuators"] = data["mj_actuators"] + ["extra"]
    with pytest.raises(ValueError, match="mj_actuators"):
        MujocoArm(profile.__class__(data))


def test_actuators_default_to_the_joint_names(profile):
    data = profile.as_dict()
    data.pop("mj_actuators")
    arm = MujocoArm(profile.__class__(data))
    assert arm._act_names == arm._joint_names


def test_profile_names_joints_that_exist_in_the_upstream_mjcf(profile):
    """Pinned against the Menagerie model's own <joint> names. If a re-fetch
    renames one, connect() raises -- but only when someone runs the sim, which
    might be at a demo."""
    assert profile.get("mj_joints") == [
        "shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll",
    ]
    assert profile.get("mj_gripper_joint") == "gripper"


def test_missing_mjcf_says_how_to_get_it(profile):
    """The assets are deliberately not vendored, so the error has to name the
    fetch command -- otherwise it reads as a broken install."""
    if not has_mujoco():
        pytest.skip("needs `mujoco` to reach the path check")
    data = profile.as_dict()
    data["mjcf"] = str(REPO / "assets" / "mjcf" / "nope" / "nope.xml")
    arm = MujocoArm(profile.__class__(data))
    with pytest.raises(RuntimeError, match="fetch_robot_assets"):
        arm.connect()


def test_substeps_match_the_waypoint_rate(profile):
    """The stream runs at 50 Hz (20 ms); substeps x MJCF timestep should be
    about one tick, or sim time drifts from the trajectory the safety layer
    vetted."""
    assert profile.get("substeps") * 0.005 == pytest.approx(0.020, abs=0.011)


def test_settle_tolerances_are_loosened_for_finite_gain_actuators(profile):
    """A position actuator does not arrive exactly, unlike the kinematic mock.
    Inheriting the mock's timeout makes every pregrasp report "did not settle"."""
    assert profile.get("settle_timeout_s") >= 2.0


# ── real physics, once per available engine (mjc, and warp if installed) ──


@pytest.mark.skipif(not _ENGINES, reason="needs `mujoco` + fetched so101 assets")
def test_connect_starts_at_the_profile_home_pose(engine_arm):
    engine, arm = engine_arm
    q = arm.get_state().q
    assert q.size == 5
    home = np.asarray(load_profile("arms", "so101_mujoco").get("home_q"), float)
    # mj_forward only, no stepping yet, so this should be exact. Warp round-trips
    # through float32 on upload, so allow a float32 epsilon there.
    atol = 1e-9 if engine == "mjc" else 1e-5
    assert np.allclose(q, home, atol=atol), f"[{engine}] home {q} != {home}"


@pytest.mark.skipif(not _ENGINES, reason="needs `mujoco` + fetched so101 assets")
def test_a_commanded_move_converges_under_gravity(engine_arm):
    engine, arm = engine_arm
    profile = load_profile("arms", "so101_mujoco")
    target = np.asarray(profile.get("home_q"), float).copy()
    target[0] += 0.3
    for _ in range(200):  # the framework streams; one call is one tick
        arm.send_joint_target(target)
    reached = arm.get_state().q
    assert abs(reached[0] - target[0]) < float(profile.get("settle_tol")), \
        f"[{engine}] joint 0 stalled at {reached[0]:.4f}, wanted {target[0]:.4f}"


@pytest.mark.skipif(not _ENGINES, reason="needs `mujoco` + fetched so101 assets")
def test_stop_holds_position_instead_of_driving_to_zero(engine_arm):
    """A position actuator left at ctrl=0 drives the arm TO zero, so a naive
    stop is a fast move to the zero pose -- the opposite of stopping."""
    engine, arm = engine_arm
    profile = load_profile("arms", "so101_mujoco")
    target = np.asarray(profile.get("home_q"), float).copy()
    target[1] += 0.2
    for _ in range(100):
        arm.send_joint_target(target)
    before = arm.get_state().q.copy()
    arm.stop()
    for _ in range(50):
        arm.send_joint_target(np.zeros(5))  # ignored while stopped
    after = arm.get_state().q
    assert np.allclose(before, after, atol=0.05), \
        f"[{engine}] drifted {np.abs(after - before).max():.3f} rad after stop"
    assert np.abs(after).max() > 0.1, f"[{engine}] should not have collapsed toward zero"


@pytest.mark.skipif(not _ENGINES, reason="needs `mujoco` + fetched so101 assets")
def test_the_gripper_moves_and_reports_back(engine_arm):
    engine, arm = engine_arm
    profile = load_profile("arms", "so101_mujoco")
    g = profile.get("gripper")
    arm.set_gripper(float(g.get("open_pos")))
    opened = arm.get_state().gripper_pos
    arm.set_gripper(float(g.get("closed_pos")))
    closed = arm.get_state().gripper_pos
    # Travel is sign-aware: this profile's open_pos > closed_pos.
    assert closed < opened, f"[{engine}] jaw did not close ({opened:.3f} -> {closed:.3f})"


@pytest.mark.skipif(not _ENGINES, reason="needs `mujoco` + fetched so101 assets")
def test_engines_agree_on_a_commanded_pose(engine_arm):
    """Engine agreement is the repo's gold metric (see docs/NEWTON_ENGINE.md).
    Whatever engine is under the arm, the same command must land in the same
    place to within finite-gain settle tolerance -- otherwise a policy tuned in
    one sim would not transfer to the other."""
    engine, arm = engine_arm
    profile = load_profile("arms", "so101_mujoco")
    target = np.asarray(profile.get("home_q"), float).copy()
    target[0] += 0.4
    target[2] -= 0.3
    for _ in range(300):
        arm.send_joint_target(target)
    reached = arm.get_state().q
    tol = float(profile.get("settle_tol"))
    assert abs(reached[0] - target[0]) < tol and abs(reached[2] - target[2]) < tol, \
        f"[{engine}] reached {np.round(reached, 3)} for target {np.round(target, 3)}"


@needs_mujoco
def test_fk_agrees_between_the_urdf_and_the_mjcf(profile):
    """The kinematics layer reads the URDF while physics runs the MJCF. If the
    two models disagree, every IK solution lands somewhere else in sim -- the
    same class of drift tests/test_usd_model.py pins for the reBot assets.

    Model geometry is engine-independent (both runtimes load the same MjModel),
    so this runs on the C engine only -- it is a property of the FILES, not the
    stepper."""
    pytest.importorskip("pinocchio")
    import mujoco

    from cascade.control.kinematics import Kinematics

    kin = Kinematics(profile.get("model"), profile.get("ee_frame"), n_controlled=5)
    arm = make_arm(profile)
    arm.connect()
    try:
        model, data = arm._model, arm._data
        site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
        if site < 0:
            pytest.skip("MJCF has no gripperframe site to compare against")

        # The MJCF's `gripperframe` site and the URDF's `gripper_frame_link` are
        # both at the fingertip but are NOT the same frame -- MEASURED, the site
        # sits 19.9 mm along the TCP's -col0 and is rotated 90 deg from it. So
        # the invariant is that their relative transform is CONSTANT: that is
        # what proves the two files describe one kinematic chain. (Differencing
        # world positions instead would fail on a rigid offset, because a fixed
        # TOOL-frame offset rotates with the wrist.)
        offsets = []
        for q in ([0.0] * 5, [0.3, -0.4, 0.5, 0.6, 0.2],
                  [-0.5, 0.3, -0.2, 0.4, -1.0], [0.9, 0.5, -1.0, 0.8, 1.5]):
            for adr, v in zip(arm._qadr, q):
                data.qpos[adr] = v
            mujoco.mj_forward(model, data)
            T = kin.fk(np.asarray(q))
            R, t = T[:3, :3], T[:3, 3]
            offsets.append(R.T @ (np.array(data.site_xpos[site]) - t))
        spread = np.abs(np.asarray(offsets) - offsets[0]).max()
        assert spread < 1e-5, (
            f"URDF and MJCF disagree: the tool-frame offset between "
            f"gripper_frame_link and the gripperframe site moves by "
            f"{spread * 1000:.3f} mm across poses, so the two models are not "
            f"the same chain"
        )
        assert np.allclose(offsets[0], [-0.019899, -0.000002, 0.0], atol=2e-5), (
            f"offset changed to {np.round(np.asarray(offsets[0]) * 1000, 3)} mm; "
            f"the MJCF was probably re-fetched at a different commit"
        )
    finally:
        arm.disconnect()
