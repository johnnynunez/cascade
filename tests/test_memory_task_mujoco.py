"""E2E memory task on the MuJoCo world with the rendered camera (real run):
two props, two chat-style pick_and_place calls, physics-verified, and the
memory harness must let a planner answer "how many did I move?" from its
frames -- the Vesta demo shape (arXiv:2606.20905 §4.4, Count Fruits /
Memorize Candy) on this rig.

Physics + GL required (skips with the fetch command otherwise); the
mujoco_scene_two profile generates the two-prop scene.
"""
from __future__ import annotations

import hashlib
import json
import time

import numpy as np
import pytest
from conftest import REPO, needs_pin

from cascade.config import load_demo_config

MJCF = REPO / "assets" / "mjcf" / "so101" / "scene.xml"


def _has_gl() -> bool:
    try:
        import mujoco

        if not MJCF.exists():
            return False
        m = mujoco.MjModel.from_xml_path(str(MJCF))
        r = mujoco.Renderer(m, height=16, width=16)
        r.close()
        return True
    except Exception:  # noqa: BLE001
        return False


needs_gl = pytest.mark.skipif(
    not _has_gl(), reason="needs mujoco + fetched SO-101 assets + offscreen GL "
    "(python scripts/fetch_robot_assets.py so101)"
)


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
def test_two_prop_scene_is_seen_and_physically_present(two_prop_runtime):
    cfg, runtime, arm = two_prop_runtime
    obs = runtime.execute("get_observation", {})
    assert obs["ok"], obs
    labels = sorted(o["label"] for o in obs["objects_visible"])
    assert labels == ["blue cube", "red cube"], labels
    world = runtime.arm.raw.world if hasattr(runtime.arm.raw, "world") else arm.world
    assert sorted(world.free_body_names()) == ["blue_cube", "red_cube"]
    # perception lands on physics for BOTH props
    for b in runtime.beliefs.all():
        truth = np.asarray(world.body_pos(b.label.replace(" ", "_")))
        lat = float(np.linalg.norm(np.asarray(b.position)[:2] - truth[:2]))
        assert lat < 0.01, (b.label, lat)
    count = runtime.execute("count_objects", {})
    assert count["ok"] and count["count"] == 2, count


@needs_pin
@needs_gl
def test_memory_frames_record_two_physics_confirmed_picks(two_prop_runtime):
    """The task a memory-less planner cannot answer: after the first pick the
    current view shows ONE cube on the pick side and cannot say whether one
    moved or none. The harness must show initial state (two), the frame after
    each pick, and a physics verdict per pick; count_objects must say 2 at the
    end and the two props must sit near the drop zone in physics truth."""
    cfg, runtime, arm = two_prop_runtime
    world = arm.world
    runtime.memory.reset_frames()
    t0 = time.monotonic()
    r1 = runtime.execute("pick_and_place", {"object": "red cube"})
    assert r1["ok"], r1
    pc1 = r1.get("postcondition") or {}
    assert pc1.get("status") == "confirmed" and pc1.get("channel") == "physics", pc1

    # mid-task: what a planner would see
    frames = runtime.memory.memory_frames(4)
    assert [f["text"][:13] for f in frames] == ["initial state", "pick_and_plac"], [f["text"] for f in frames]
    assert frames[1]["verdict"] == "confirmed"
    assert hashlib.md5(frames[0]["jpeg"]).hexdigest() != hashlib.md5(frames[1]["jpeg"]).hexdigest(), \
        "initial state and after-first-pick frames must differ (the red cube moved)"

    r2 = runtime.execute("pick_and_place", {"object": "blue cube"})
    assert r2["ok"], r2
    pc2 = r2.get("postcondition") or {}
    assert pc2.get("status") == "confirmed" and pc2.get("channel") == "physics", pc2

    frames = runtime.memory.memory_frames(4)
    assert len(frames) == 3, [f["text"] for f in frames]
    assert [f["verdict"] for f in frames] == ["", "confirmed", "confirmed"]
    assert frames[0]["age_s"] >= (time.monotonic() - t0) - 1.0, "initial state must still be there after ~1 min"
    caps = [runtime.memory.frame_caption(f) for f in frames]
    assert "pick_and_place(object=red cube) -> ok [CONFIRMED]" in caps[1], caps
    assert "pick_and_place(object=blue cube) -> ok [CONFIRMED]" in caps[2], caps

    # physics: both props ended within reach of the drop zone
    dz = np.asarray(cfg.grasp.get("drop_zone"), float)
    for body in ("red_cube", "blue_cube"):
        p = np.asarray(world.body_pos(body))
        assert float(np.linalg.norm(p[:2] - dz)) < 0.06, (body, p.round(3).tolist(), dz.tolist())
        assert p[2] > 0.0, (body, "fell off the table?")
    # and the robot's own count agrees
    runtime.execute("get_observation", {})
    count = runtime.execute("count_objects", {})
    assert count["ok"] and count["count"] == 2, count
    # trace rows carry the same verdicts (what judge_run.py reads)
    rows = [json.loads(l) for l in (runtime.trace.run_dir / "trace.jsonl").read_text().splitlines() if l.strip()]
    picks = [r for r in rows if r["skill"] == "pick_and_place"]
    assert len(picks) == 2 and all((p["result"].get("postcondition") or {}).get("status") == "confirmed" for p in picks)
