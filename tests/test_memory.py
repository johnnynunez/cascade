import numpy as np

from cascade.memory.beliefs import BeliefStore
from cascade.memory.episodic import EpisodicMemory


def test_horizon_pruning():
    fake_now = [8.0]
    mem = EpisodicMemory(horizon_s=10.0, clock=lambda: fake_now[0])
    mem.add("note", "old event", t=0.0)
    mem.add("note", "recent event", t=8.0)
    fake_now[0] = 12.0
    texts = [e.text for e in mem.events()]
    assert "old event" not in texts
    assert "recent event" in texts


def test_digest_format():
    mem = EpisodicMemory(horizon_s=15.0)
    mem.add("action", "grasped cube", t=100.0)
    digest = mem.digest(now=105.0)
    assert "5.0s ago" in digest and "grasped cube" in digest
    assert EpisodicMemory(horizon_s=15.0).digest() == "(memory empty)"


def test_thumbnail_stored():
    mem = EpisodicMemory(horizon_s=15.0)
    rgb = np.full((120, 160, 3), 128, dtype=np.uint8)
    ev = mem.add("observation", "frame", rgb=rgb)
    assert ev.thumb_jpeg is not None and len(ev.thumb_jpeg) > 100
    assert mem.last_frame_jpeg() == ev.thumb_jpeg


def test_embedding_recall():
    mem = EpisodicMemory(horizon_s=15.0, embed_dim=32)
    e1 = np.ones(32)
    e2 = -np.ones(32)
    ev1 = mem.add("observation", "red cube seen", embedding=e1)
    mem.add("observation", "blue ball seen", embedding=e2)
    hits = mem.recall_similar(e1, k=1)
    assert hits and hits[0] is ev1


def test_belief_matching_and_ema():
    store = BeliefStore(match_radius_m=0.08, pos_alpha=0.5)
    b1 = store.update("cup", [0.30, 0.00, 0.02], conf=0.9, t=1.0)
    b2 = store.update("cup", [0.32, 0.00, 0.02], conf=0.9, t=2.0)
    assert b1 is b2  # matched, not duplicated
    assert np.isclose(b2.position[0], 0.31)
    b3 = store.update("cup", [0.60, 0.20, 0.02], conf=0.9, t=3.0)
    assert b3 is not b2  # too far: second instance
    assert len(store.all(now=3.0)) == 2


def test_belief_states_and_find():
    store = BeliefStore()
    store.update("mug", [0.3, 0.1, 0.0], conf=0.8, t=100.0)
    assert store.find("mug").state(now=100.5) == "visible"
    assert store.find("mug").state(now=110.0) == "remembered"
    assert store.find("red mug") is not None  # loose contains-match
    assert store.find("banana") is None


def test_belief_removal():
    store = BeliefStore()
    store.update("cube", [0.3, 0.0, 0.0], conf=0.9)
    assert store.mark_removed("cube", near=[0.3, 0.0, 0.0])
    assert store.find("cube") is None
    assert not store.mark_removed("cube")


def test_belief_remembers_real_cloud_and_fix_uses_it():
    """Grasp-from-memory must use the object's TRUE shape when a real mask
    cloud was stored: a banana's box-slab approximation (78mm) reads as
    ungraspable, while its real cloud reads ~35mm (GGX-empty root cause)."""
    import numpy as np

    from cascade.skills.runtime import SkillRuntime

    store = BeliefStore()
    rng = np.random.default_rng(1)
    # thin curved-ish object: 16cm long, 3.5cm wide, 3cm tall
    pts = (rng.random((1500, 3)) - 0.5) * np.array([0.16, 0.035, 0.03])
    pts += np.array([0.24, 0.14, 0.02])
    b = store.update("banana", pts.mean(axis=0), 0.8, points=pts)
    assert b.points is not None and b.points.shape == (384, 3)  # subsampled

    fix = SkillRuntime._fix_from_belief("banana", b)
    assert fix.points.shape[0] == 384  # real cloud, not the 360-pt box synth
    ext = np.sort(fix.extent)[::-1]
    assert ext[1] < 0.05  # true jaw span, not a bbox slab
    # fused position moved (EMA) -> cloud must follow the fused center
    b.position = b.position + np.array([0.03, 0.0, 0.0])
    fix2 = SkillRuntime._fix_from_belief("banana", b)
    assert abs(fix2.points[:, 0].mean() - fix.points[:, 0].mean() - 0.03) < 5e-3


def test_belief_bbox_fallback_never_stores_points():
    """bbox-rectangle masks sweep in table pixels; update() is only called
    with points= for real masks -- and a points=None update must not clear
    a previously remembered cloud."""
    import numpy as np

    store = BeliefStore()
    pts = np.random.default_rng(2).random((200, 3)).astype(np.float32)
    b = store.update("cube", np.array([0.2, 0.0, 0.02]), 0.8, points=pts)
    assert b.points is not None
    store.update("cube", np.array([0.2, 0.0, 0.02]), 0.9)  # no points
    assert b.points is not None and b.points.shape[0] == 200


def test_mark_removed_when_not_first_element():
    """list.remove() falls back to dataclass __eq__ over numpy fields and
    raises 'truth value of an array is ambiguous' unless the victim happens
    to be the FIRST belief -- removal must be by identity (crashed a live
    grasp: held state stuck at 'already holding')."""
    import numpy as np

    store = BeliefStore()
    store.update("cup", np.array([0.3, 0.1, 0.02]), 0.9)
    store.update("banana", np.array([0.2, 0.0, 0.02]), 0.9)
    assert store.mark_removed("banana", near=np.array([0.2, 0.0, 0.02]))
    assert store.find("banana") is None
    assert store.find("cup") is not None


def test_two_colours_inside_the_match_radius_stay_two_objects():
    """Proximity fusion is for label ALIASES of one object; two props whose
    measured mask colours differ are two objects even when they are closer
    than `match_radius_m`. Measured on the two-cube MuJoCo scene: 3.5 cm
    cubes 5.8 cm apart fused into one belief, count_objects said 1 and the
    fused position sat a cube-width off physics truth."""
    store = BeliefStore(match_radius_m=0.08, pos_alpha=0.5)
    store.update("red cube", np.array([0.20, 0.10, 0.025]), 0.9, color="red")
    store.update("blue cube", np.array([0.20, 0.158, 0.025]), 0.9, color="blue")
    assert len(store.all()) == 2
    assert {b.color for b in store.all()} == {"red", "blue"}
    # an alias of the SAME colour still fuses
    store.update("hassock", np.array([0.20, 0.10, 0.025]), 0.5, color="red")
    assert len(store.all()) == 2
    # an observation with no colour measurement keeps the old proximity rule
    store.update("thing", np.array([0.20, 0.155, 0.025]), 0.5)
    assert len(store.all()) == 2
