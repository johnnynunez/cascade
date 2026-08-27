"""Regression tests for the physics verification channel.

Written after a live ablation run recorded a 123-metre "displacement":
TruthPoseReader handed the postcondition checker a pose of
[-11.8, -10.6, -122.1] read from a body caught mid-solver. The checker has no
way to tell that from a real measurement, so a numerical fault became a
confident verdict. A verification channel that reports nonsense with
confidence is worse than one that stays silent.
"""

from __future__ import annotations

import json

import pytest

from cascade.sim.truth import TruthPoseReader, _is_sane


class FakeClient:
    """Bridge stub: returns whatever poses the test dictates."""

    def __init__(self, poses: dict):
        self.poses = poses
        self.calls = 0

    def request(self, req):
        self.calls += 1
        return {"ok": True,
                "stdout": "CASCADE_TRUTH_POSES " + json.dumps(self.poses)}


# ── the sanity predicate ────────────────────────────────────────────────

@pytest.mark.parametrize("xyz", [
    [0.17, 0.15, 0.04],
    [0.0, 0.0, 0.0],
    [-0.3, 0.42, 1.1],
])
def test_plausible_poses_pass(xyz):
    assert _is_sane(xyz)


@pytest.mark.parametrize("xyz", [
    [-11.8081, -10.6423, -122.1323],   # the one observed live
    [float("nan"), 0.0, 0.0],
    [float("inf"), 0.0, 0.0],
    [0.0, 0.0, -2000.0],               # the PhysX "flung prop" failure
    [1.0, 2.0],                        # wrong arity
    None,
])
def test_impossible_poses_are_rejected(xyz):
    assert not _is_sane(xyz)


# ── the reader ──────────────────────────────────────────────────────────

def test_reader_returns_a_plausible_pose():
    r = TruthPoseReader(FakeClient({"pink_cube": [0.17, 0.15, 0.04]}),
                        ttl_s=0.0)
    assert r.pose("pink cube") == [0.17, 0.15, 0.04]


def test_exploded_pose_is_not_reported_as_truth():
    """The core regression: an insane pose must NOT reach the checker.

    Returning None makes PostconditionChecker fall through to the next
    channel, which is the honest outcome -- we did not measure anything.
    """
    client = FakeClient({"pink_cube": [-11.8081, -10.6423, -122.1323]})
    r = TruthPoseReader(client, ttl_s=0.0)
    assert r.pose("pink cube") is None
    assert r.rejected == 1


def test_one_exploded_prop_does_not_poison_the_others():
    r = TruthPoseReader(FakeClient({
        "pink_cube": [-11.8, -10.6, -122.1],     # exploded
        "green_cube": [0.30, 0.16, 0.04],        # fine
    }), ttl_s=0.0)
    assert r.pose("green cube") == [0.30, 0.16, 0.04]
    assert r.pose("pink cube") is None


def test_spanish_labels_still_match():
    """The ES->EN map is load-bearing: without it a Spanish command silently
    loses the only independent channel (live rig, 2026-07-31)."""
    r = TruthPoseReader(FakeClient({"pink_cube": [0.17, 0.15, 0.04]}),
                        ttl_s=0.0)
    assert r.pose("cubo rosa") == [0.17, 0.15, 0.04]


def test_probe_failure_degrades_to_none_not_to_a_lie():
    class Broken:
        def request(self, req):
            raise RuntimeError("bridge closed the connection")

    r = TruthPoseReader(Broken(), ttl_s=0.0)
    assert r.pose("pink cube") is None


def test_a_single_shared_token_is_not_identification():
    """"pink cube" must NOT resolve to green_cube just because both are cubes.

    This is the failure the exploded-pose guard exposed: once pink_cube was
    dropped from the reading, token overlap on {cube} alone was enough to
    return the GREEN cube's pose, and the checker would then confirm a
    placement against the wrong object.
    """
    r = TruthPoseReader(FakeClient({"green_cube": [0.30, 0.16, 0.04]}),
                        ttl_s=0.0)
    assert r.pose("pink cube") is None
    assert r.pose("green cube") == [0.30, 0.16, 0.04]


def test_unknown_object_returns_none():
    r = TruthPoseReader(FakeClient({"pink_cube": [0.17, 0.15, 0.04]}),
                        ttl_s=0.0)
    assert r.pose("banana") is None
