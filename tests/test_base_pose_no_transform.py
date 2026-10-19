"""No transform without a base_pose (base_pose spec §4, D2, D5; plan Task 8 Step 1).

On rigs whose arms declare no base_pose (the reBot model and the kitchen arm on
a MockArm, `mock`, `so101_mock`) and on the identity rig (R12: left at the exact
identity, right at the shipped 0.44 m spacing, calls on the PRIMARY), seeded
random skill calls never reach `arm_frame._rigid_apply`, the one place a
base_pose multiplies anything, and every call binds a frame whose
`T_world_base` is None. The byte-identical command / result diff against main
is the ledger's diff probe (it compares two trees; this pins the mechanism).
"""

from __future__ import annotations

import numpy as np
import pytest
from base_pose_oracle import make_runtime  # noqa: F401  (the fixture, by name)

#: rig -> (arm profiles, camera, base_pose by arm, occupancy.required off)
RIGS = {
    "rebot_rs": (["rebot_rs"], "mock", {}, True),
    "isaac_kitchen": (["isaac_kitchen"], "mock", {}, True),
    "mock": (["mock"], "mock", {}, False),
    "so101_mock": (["so101_mock"], "mock_small", {}, False),
    "pair_identity": (["so101_left", "so101_right"], "mock_small",
                      {"so101_left": [0.0] * 6, "so101_right": [0.0, -0.44, 0.0, 0.0, 0.0, 0.0]}, False),
}
LABELS = ["red cube", "red cube", "red cube", "blue cube", "banana"]


def random_call(rng):
    """The diff probe's call pool (joint, gripper, relative and world-target
    skills; world-target skills drawn twice as often)."""
    joint = [
        ("move_home", lambda: {}),
        ("rotate_joint", lambda: {"joint": int(rng.integers(1, 6)),
                                  "degrees": float(rng.choice([-1, 1]) * rng.uniform(2, 12))}),
        ("wave", lambda: {"cycles": int(rng.integers(1, 3))}),
        ("open_gripper", lambda: {}),
        ("close_gripper", lambda: {}),
        ("move_relative", lambda: {"direction": str(rng.choice(["up", "down", "left", "right"])),
                                   "distance_m": float(rng.uniform(0.01, 0.04))}),
    ]
    world = [
        ("grasp_object", lambda: {"label": str(rng.choice(LABELS))}),
        ("place_at", lambda: {"x": float(rng.uniform(0.15, 0.35)), "y": float(rng.uniform(-0.15, 0.15)),
                              "z": float(rng.uniform(0.03, 0.10))}),
        ("point_at", lambda: {"label": str(rng.choice(LABELS))}),
        ("push_object", lambda: {"label": str(rng.choice(LABELS)),
                                 "direction": str(rng.choice(["left", "right", "forward"])),
                                 "distance_m": float(rng.uniform(0.03, 0.08))}),
        ("preview_grasp", lambda: {"label": str(rng.choice(LABELS))}),
        ("move_near_object", lambda: {"label": str(rng.choice(LABELS)), "relation": "above"}),
        ("pick_and_place", lambda: {"object": str(rng.choice(LABELS))}),
        ("grasp_at_pixel", lambda: {"u": float(rng.uniform(0.3, 0.7)), "v": float(rng.uniform(0.3, 0.7)),
                                    "normalized": True}),
        ("localize_object", lambda: {"label": str(rng.choice(LABELS))}),
        ("get_observation", lambda: {}),
        ("list_objects", lambda: {}),
    ]
    pool = joint + world + world
    name, make = pool[int(rng.integers(len(pool)))]
    return name, make()


@pytest.fixture
def mock_arms_only(monkeypatch):
    """Every arm a MockArm (no CAN, SDK or simulator), and no perception
    server reachable: the kitchen profile's brain URL points at a held dead
    port, as the diff probe does."""
    import cascade.apps.demo as demo
    from cascade.control.mock_arm import MockArm
    from owned_server import held_dead_port

    monkeypatch.setattr(demo, "make_arm", lambda acfg, kinematics=None, *a, **k: MockArm(acfg, kinematics=kinematics))
    with held_dead_port() as port:
        monkeypatch.setenv("CASCADE_PERCEPTION_URL", f"http://127.0.0.1:{port}/v1")
        yield


def _cfg(rig):
    from cascade.config import load_demo_config

    arms, camera, poses, required_off = RIGS[rig]
    cfg = (load_demo_config(arm=arms[0], camera=camera, llm="mock") if len(arms) == 1
           else load_demo_config(arms=arms, camera=camera, llm="mock"))
    if required_off and "occupancy" in cfg._data:
        cfg._data["occupancy"]["required"] = False
    for entry in [cfg._data["arm"], *(cfg._data.get("arms") or [])]:
        if entry.get("name") in poses:
            entry["base_pose"] = list(poses[entry["name"]])
    return cfg


def _no_multiply(monkeypatch):
    """`_rigid_apply` refuses (and records, in case a broad handler swallows
    the AssertionError)."""
    from cascade.skills import arm_frame

    ran = []

    def refuse(*_a, **_k):
        ran.append(1)
        raise AssertionError("arm_frame._rigid_apply ran on a rig without a base_pose")

    monkeypatch.setattr(arm_frame, "_rigid_apply", refuse)
    return ran


def _bound_frames(rt, monkeypatch):
    frames, real = [], rt._resolve_arm_frame
    monkeypatch.setattr(rt, "_resolve_arm_frame", lambda *a, **k: (frames.append(real(*a, **k)), frames[-1])[1])
    return frames


@pytest.mark.parametrize("rig", sorted(RIGS))
def test_no_transform_runs_without_a_base_pose(make_runtime, mock_arms_only, monkeypatch, rig):  # noqa: F811
    ran = _no_multiply(monkeypatch)
    calls = 0
    for seed in range(3):
        rt = make_runtime(_cfg(rig))
        frames = _bound_frames(rt, monkeypatch)
        rng = np.random.default_rng(seed)
        for _ in range(8):
            name, args = random_call(rng)
            result = rt.execute(name, dict(args))
            calls += 1
            assert not ran and "_rigid_apply ran" not in str(result), (rig, seed, name, result)
        assert len(frames) == 8, (rig, seed, len(frames))
        assert all(f.T_world_base is None for f in frames), (rig, seed)
    assert calls == 24


def test_the_identity_rig_really_has_a_moved_neighbour(make_runtime, mock_arms_only):  # noqa: F811
    """The identity rig is not degenerate (R12): its right arm's base_pose
    moves it 0.44 m, so the pin above runs beside a real transform."""
    rt = make_runtime(_cfg("pair_identity"))
    right = rt.arm_rig.get("so101_right").__dict__["harness"].base_pose
    assert right is not None and np.isclose(right[1, 3], -0.44)
    assert rt.arm_rig.get("so101_left").__dict__["harness"].__dict__.get("base_pose") is None or np.array_equal(
        rt.arm_rig.get("so101_left").__dict__["harness"].base_pose, np.eye(4))
