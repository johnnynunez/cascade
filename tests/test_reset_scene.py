"""`reset_scene`: the between-visitors skill. Arm home, sim props back on
their spawn pose, beliefs and the task's visual memory cleared, one fresh
observation. The launcher calls it after its own proof turn so the first
visitor does not start "put both cubes in the drop zone" with one cube
already there."""
from __future__ import annotations

import numpy as np
import pytest
from conftest import REPO, needs_pin

from cascade.agent.reflex import parse_command
from cascade.config import load_demo_config

MJCF = REPO / "assets" / "mjcf" / "so101" / "scene.xml"


def _has_gl() -> bool:
    try:
        import mujoco

        if not MJCF.exists():
            return False
        r = mujoco.Renderer(mujoco.MjModel.from_xml_path(str(MJCF)), height=16, width=16)
        r.close()
        return True
    except Exception:  # noqa: BLE001
        return False


needs_gl = pytest.mark.skipif(not _has_gl(), reason="needs mujoco + SO-101 assets + offscreen GL")


def test_reset_phrases_are_reflexes_and_nothing_else_matches():
    """Works with the LLM down: every cheat-card phrasing is tier-1."""
    for t in ("reset the scene", "reinicia la escena", "start over", "empieza de nuevo",
              "reset", "new demo", "nueva demo"):
        p = parse_command(t)
        assert p is not None and p.calls == [("reset_scene", {})], t
    assert parse_command("go home").calls == [("move_home", {})]
    assert parse_command("reset the gripper force") is None


def test_reset_scene_is_a_tool_and_a_motion_skill():
    from cascade.skills.runtime import _MOTION_SKILLS, TOOL_SPECS

    assert "reset_scene" in _MOTION_SKILLS  # it moves the arm home
    spec = next(t for t in TOOL_SPECS if t["name"] == "reset_scene")
    assert "spawn" in spec["description"] and "real robot" in spec["description"]


@pytest.fixture
def mock_runtime(demo_cfg, tmp_path):
    from cascade.apps.demo import build_runtime

    runtime, arm = build_runtime(demo_cfg, tmp_path / "run")
    yield runtime, arm
    runtime.camera.close()
    arm.disconnect()


@needs_pin
def test_reset_on_the_mock_stack_clears_memory_and_beliefs_and_reobserves(mock_runtime):
    runtime, arm = mock_runtime
    arm.object_stop_frac = 0.5
    r = runtime.execute("grasp_object", {"label": "red cube", "material": "rigid"})
    assert r["ok"], r
    assert runtime.held_object == "red cube"
    assert len(runtime.memory.memory_frames(8)) == 2
    out = runtime.execute("reset_scene", {})
    assert out["ok"], out
    assert out["props_reset"] == []  # mock arm: nothing to teleport
    assert out["world"] is None
    assert runtime.held_object is None
    # the grasped prop's belief was already removed by the grasp (it is in
    # the gripper, not on the table), so on this one-prop scene the clear
    # finds nothing; the two-prop physics test below asserts the count.
    assert out["beliefs_forgotten"] == 0
    assert "scene reset" in runtime.memory.digest()
    # the fresh observation re-populated the world model
    assert any(o["label"] == "red cube" for o in out["objects_visible"]), out
    # the new episode starts with ONE frame: the fresh observation, not the
    # reset's own after-frame and not the old episode's initial state
    fr = runtime.memory.memory_frames(8)
    assert len(fr) == 1 and fr[0]["kind"] == "observation", [(f["kind"], f["text"]) for f in fr]


@pytest.fixture
def two_prop_runtime(tmp_path, monkeypatch):
    monkeypatch.setenv("CASCADE_OCCUPANCY", "0")
    from cascade.apps.demo import build_runtime

    cfg = load_demo_config(camera="mujoco_scene_two", arm="so101_mujoco", llm="mock")
    cfg._data["stream"] = {"enabled": False}
    runtime, arm = build_runtime(cfg, tmp_path / "run")
    yield cfg, runtime, arm
    runtime.camera.close()
    arm.disconnect()


@needs_pin
@needs_gl
def test_reset_puts_a_physically_moved_prop_back_on_spawn(two_prop_runtime):
    """Physics truth is the judge: after a confirmed pick the red cube sits at
    the drop zone; after reset_scene it is back on its MJCF spawn pose, at
    rest, and perception sees it there again."""
    cfg, runtime, arm = two_prop_runtime
    world = arm.world
    spawn = {n: np.asarray(world.body_pos(n)) for n in world.free_body_names()}
    r = runtime.execute("pick_and_place", {"object": "red cube"})
    assert r["ok"] and (r.get("postcondition") or {}).get("status") == "confirmed", r
    moved = np.asarray(world.body_pos("red_cube"))
    assert np.linalg.norm(moved[:2] - spawn["red_cube"][:2]) > 0.15, "the pick must have moved it"

    out = runtime.execute("reset_scene", {})
    assert out["ok"], out
    assert sorted(out["props_reset"]) == ["blue_cube", "red_cube"]
    assert out["beliefs_forgotten"] >= 2  # red (at the drop zone) + blue
    assert out["world"] == "mujoco"
    for n, p0 in spawn.items():
        p1 = np.asarray(world.body_pos(n))
        assert np.linalg.norm(p1 - p0) < 0.002, (n, p0.round(3).tolist(), p1.round(3).tolist())
    # settle a little physics: it must STAY there (resting, not exploding)
    q = arm.get_state().q
    for _ in range(50):
        arm.send_joint_target(q)
    for n, p0 in spawn.items():
        assert np.linalg.norm(np.asarray(world.body_pos(n)) - p0) < 0.005, n
    labels = sorted(o["label"] for o in out["objects_visible"])
    if labels != ["blue cube", "red cube"]:
        import cv2
        f = runtime.observe()
        cv2.imwrite("/tmp/reset_fail.jpg", f.rgb)
        dets = runtime.detector.detect(f, classes=None)
        raise AssertionError(
            f"labels={labels} beliefs={[(b.label, b.color, np.round(b.position, 3).tolist()) for b in runtime.beliefs.all()]} "
            f"dets_now={[(d.label, d.bbox.astype(int).tolist()) for d in dets]} "
            f"truth={ {n: np.round(world.body_pos(n), 3).tolist() for n in world.free_body_names()} } "
            f"q={np.round(arm.get_state().q, 2).tolist()} held={runtime.held_object!r} out={ {k: v for k, v in out.items() if k != 'objects_visible'} }"
        )
    b = runtime.beliefs.find("red cube")
    assert b is not None and np.linalg.norm(np.asarray(b.position)[:2] - spawn["red_cube"][:2]) < 0.01
    assert len(runtime.memory.memory_frames(8)) == 1  # fresh episode
