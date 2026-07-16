import numpy as np

from wrc_demo.memory.beliefs import BeliefStore
from wrc_demo.memory.episodic import EpisodicMemory


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
