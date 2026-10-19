"""Acceptance on the MuJoCo so101 pair (base_pose spec §0.8, §7, R1, R2; plan Task 8 Step 4).

Physical equivariance in one MuJoCo world per rig, both on the rig arm's own
profile (`so101_right_mujoco`):

- rig B, the pair: `--arms so101_left_mujoco,so101_right_mujoco --camera
  mock_small`, the camera fixed in the table frame and shifted 10 cm in -y, so
  the planted prop stands at W (0.198, -0.10); calls on `arm=so101_right_mujoco`;
- rig A, the control (R2): `so101_right_mujoco` alone with its `base_pose` and
  `mj_prefix` popped and the camera moved by T^-1 (+0.22 m in y), so the prop
  stands where rig B's is in the right arm's base frame, B (0.198, 0.12).

For each sequence (pick_and_place; grasp_object + place_at at W (0.22, -0.30),
B (0.22, -0.08)): rig B's prop physics position (the MuJoCo truth channel, W)
lies within 1 cm of T times rig A's, both postconditions are CONFIRMED by
physics, and so101_left_mujoco's joints move less than 1e-3 rad. place_at also
keeps the absolute bar: within `DEST_TOLERANCE_M` (0.03 m) of its target.
Rig A's pick_and_place leaves the prop ~5 cm from its drop zone, tipped (R1:
pre-existing, out of scope), so that sequence is judged by equivariance only.

Needs `mujoco`, `pinocchio` and the fetched MJCF (`scripts/fetch_robot_assets.py
so101`); skips otherwise, like the other MuJoCo tests. CI runs this file in its
own step after the suite (#407 review follow-up 6): it fetches the MJCF into
`assets/mjcf/so101` (pinned, where `MJCF` points) and sets
`CASCADE_REQUIRE_MUJOCO_PAIR=1`, which turns that skip into a collection
failure, so the step can never pass by skipping. CPU only: no rendered camera.
"""

from __future__ import annotations

import contextlib
import os

import numpy as np
import pytest
from base_pose_oracle import SHIPPED_RIGHT, lift, moved_cfg, transform
from test_base_pose_runtime_frame import gate_off  # noqa: F401  (a fixture, by name)
from test_multi_arm_physics import needs_rig

#: Set by CI's MuJoCo pair step: a missing requirement fails instead of skipping.
REQUIRE_ENV = "CASCADE_REQUIRE_MUJOCO_PAIR"
if os.environ.get(REQUIRE_ENV) == "1" and needs_rig.args[0]:
    pytest.fail(f"{REQUIRE_ENV}=1 but this acceptance cannot run: {needs_rig.kwargs['reason']}", pytrace=False)

pytestmark = needs_rig

#: The prop's table-frame target per sequence, as the agent states it.
PLACE_W = (0.22, -0.30)
EQUIVARIANCE_M = 0.01          # R1
DEST_TOLERANCE_M = 0.03        # effects.DEST_TOLERANCE_M (never widened)


def _sequence(name, T):
    """Rig A's and rig B's calls for one sequence (rig B's place in W, rig A's
    the same point in the arm's base frame)."""
    if name == "pick_and_place":
        call = [("pick_and_place", {"object": "red cube"})]
        return call, call
    place_b = {"x": PLACE_W[0], "y": PLACE_W[1]}
    a = np.linalg.solve(np.asarray(T), [PLACE_W[0], PLACE_W[1], 0.0, 1.0])
    place_a = {"x": float(a[0]), "y": float(a[1])}
    return ([("grasp_object", {"label": "red cube"}), ("place_at", place_a)],
            [("grasp_object", {"label": "red cube"}), ("place_at", place_b)])


def _rig_a_cfg():
    cfg = moved_cfg("so101_right_mujoco", None, move_cameras=False, shift_xy=(0.0, 0.12))
    for entry in [cfg._data["arm"], *(cfg._data.get("arms") or [])]:
        entry.pop("mj_prefix", None)            # a lone rig arm: no rig, no joint-name prefix (R2)
    cfg._data["stream"] = {"enabled": False}
    return cfg


def _rig_b_cfg():
    cfg = moved_cfg(["so101_left_mujoco", "so101_right_mujoco"], {}, move_cameras=False, shift_xy=(0.0, -0.10))
    cfg._data["stream"] = {"enabled": False}
    return cfg


@contextlib.contextmanager
def _runtime(cfg, tmp_path, monkeypatch, tag):
    """One rig at a time: its own stores, shut down (its MuJoCo world
    released) before the next rig is built."""
    from cascade.apps.demo import build_runtime, shutdown_runtime

    stores = tmp_path / f"{tag}_stores"
    stores.mkdir()
    for env, name in (("CASCADE_BELIEFS_PATH", "beliefs.json"), ("CASCADE_GRASP_MEMORY_PATH", "grasp.json"),
                      ("CASCADE_ENVELOPE_PATH", "envelope.json"), ("CASCADE_EPISODIC_PATH", "episodic.json")):
        monkeypatch.setenv(env, str(stores / name))
    rt, arm = build_runtime(cfg, tmp_path / tag, view=False, serve=False)
    try:
        yield rt
    finally:
        shutdown_runtime(rt, arm)


def _run(rt, calls, arm=None):
    results = []
    for skill, args in calls:
        results.append(rt.execute(skill, {**args, **({"arm": arm} if arm else {})}))
    return results, np.asarray(rt._object_pose("red cube"), dtype=float)


def _confirmed_by_physics(result):
    post = result.get("postcondition") or {}
    return str(post.get("status", "")).lower() == "confirmed" and post.get("channel") == "physics"


@pytest.mark.parametrize("sequence", ["pick_and_place", "grasp_object + place_at"])
def test_the_mujoco_pair_places_like_the_arm_without_a_base_pose(tmp_path, gate_off, monkeypatch,  # noqa: F811
                                                                 sequence):
    monkeypatch.setenv("MUJOCO_GL", "disable")
    monkeypatch.setenv("CASCADE_GRASP_BACKEND", "obb")
    T = transform(SHIPPED_RIGHT)
    calls_a, calls_b = _sequence(sequence, T)

    with _runtime(_rig_a_cfg(), tmp_path, monkeypatch, "rig_a") as rt_a:
        results_a, prop_a = _run(rt_a, calls_a)
    with _runtime(_rig_b_cfg(), tmp_path, monkeypatch, "rig_b") as rt_b:
        left = rt_b.arm_rig.get("so101_left_mujoco")
        q_left = np.asarray(left.get_state().q, dtype=float)
        results_b, prop_b = _run(rt_b, calls_b, arm="so101_right_mujoco")
        q_left_after = np.asarray(left.get_state().q, dtype=float)
    print(f"\nRESULT {sequence}: rig A prop {np.round(prop_a, 4).tolist()} -> T a {np.round(lift(T, prop_a), 4).tolist()}"
          f"; rig B prop {np.round(prop_b, 4).tolist()}; |B - T a| {np.linalg.norm(prop_b - lift(T, prop_a)):.4f} m; "
          f"status A {[(r.get('postcondition') or {}).get('status') for r in results_a]} "
          f"B {[(r.get('postcondition') or {}).get('status') for r in results_b]}; "
          f"left max dq {np.max(np.abs(q_left_after - q_left)):.2e} rad")

    for ra, rb in zip(results_a, results_b):
        assert ra.get("ok", True) is True and rb.get("ok", True) is True, (ra, rb)
        assert _confirmed_by_physics(ra) and _confirmed_by_physics(rb), (ra.get("postcondition"),
                                                                        rb.get("postcondition"))
    want = lift(T, prop_a)
    assert np.linalg.norm(prop_b - want) <= EQUIVARIANCE_M, (prop_b, want, prop_a)
    if sequence != "pick_and_place":
        assert np.hypot(prop_b[0] - PLACE_W[0], prop_b[1] - PLACE_W[1]) <= DEST_TOLERANCE_M, prop_b
    assert np.max(np.abs(q_left_after - q_left)) < 1e-3, "the left arm stayed put"
