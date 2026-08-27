"""Open-vocabulary perception must stay object-agnostic.

These tests pin the booth requirement: a visitor puts an arbitrary object on
the table and the system must see it without anyone having named it. The
regression they guard against is real -- three separate layers used to inject
a hard-coded class list, and the always-on watcher shared its detector with
the skill runtime, so one closed list silently narrowed the whole system.
"""

from __future__ import annotations

import numpy as np
import pytest

from cascade.memory import BeliefStore
from cascade.perception.workspace import WorkspaceFilter


# ── the geometric gate is class-agnostic ─────────────────────────────────

@pytest.mark.parametrize(
    "name,center,extent",
    [
        ("screwdriver", [0.30, 0.05, 0.015], [0.15, 0.02, 0.02]),
        ("phone", [0.25, -0.05, 0.020], [0.15, 0.07, 0.01]),
        ("mug", [0.22, 0.10, 0.045], [0.09, 0.09, 0.09]),
        ("m8 screw", [0.28, 0.02, 0.014], [0.02, 0.008, 0.008]),
        ("shoebox", [0.30, 0.00, 0.10], [0.30, 0.18, 0.12]),
    ],
)
def test_arbitrary_objects_are_accepted(name, center, extent):
    """Anything of plausible size sitting on the table passes, whatever it is.

    The filter must never encode WHICH objects are allowed; a booth cannot
    enumerate what visitors bring.
    """
    w = WorkspaceFilter()
    assert w.reject(np.array(center), np.array(extent), mask_frac=0.05) is None, name


@pytest.mark.parametrize(
    "name,center,extent,mask_frac,reason",
    [
        # the arm itself, named "transformer"/"amplifier" by a 4.5k vocabulary
        ("robot base", [0.029, -0.008, 0.116], [0.10, 0.10, 0.20], 0.05, "robot"),
        # the whole scene as one detection
        ("studio shot", [0.25, 0.0, 0.08], [0.90, 0.60, 0.30], 0.80, "scene"),
        # a shadow: flat on the table, no volume
        ("shadow", [0.099, -0.387, 0.0], [0.12, 0.08, 0.001], 0.04, "flat"),
        # furniture
        ("table", [0.30, 0.0, -0.015], [1.20, 0.80, 0.03], 0.30, "oversize"),
    ],
)
def test_scenery_and_robot_are_rejected(name, center, extent, mask_frac, reason):
    w = WorkspaceFilter()
    got = w.reject(np.array(center), np.array(extent), mask_frac=mask_frac)
    assert got == reason, f"{name}: expected {reason}, got {got}"


def test_filter_thresholds_come_from_config():
    class _Cfg:
        def __init__(self, d):
            self._d = d

        def get(self, k, default=None):
            return self._d.get(k, default)

    w = WorkspaceFilter.from_config(_Cfg({"reach_m": 0.42, "min_height_m": 0.99}))
    assert w.reach_m == pytest.approx(0.42)
    assert w.min_height_m == pytest.approx(0.99)
    # unset keys keep their defaults
    assert w.base_radius_m == pytest.approx(WorkspaceFilter().base_radius_m)


# ── beliefs fuse by geometry, not by name ────────────────────────────────

def test_same_object_under_two_names_is_one_belief():
    """An open vocabulary renames things between frames.

    The bin came back as "storage box" on one frame and "building block" on
    the next. Matching on label alone registered both, so a table with 3
    objects reported 10.
    """
    store = BeliefStore()
    pos = np.array([0.18, -0.17, 0.03])
    store.update("storage box", pos, conf=0.68, extent=np.array([0.14, 0.14, 0.06]))
    store.update("building block", pos + 0.01, conf=0.55,
                 extent=np.array([0.14, 0.14, 0.06]))

    assert len(store.all()) == 1, "one physical object must be one belief"


def test_alias_query_still_resolves():
    """The name that lost the confidence vote must still find the object."""
    store = BeliefStore()
    pos = np.array([0.18, -0.17, 0.03])
    store.update("storage box", pos, conf=0.80)
    store.update("building block", pos, conf=0.40)

    b = store.find("building block")
    assert b is not None
    assert b.label == "storage box", "the more confident name wins"
    assert "building block" in b.aliases


def test_distinct_objects_are_not_merged():
    """Label-agnostic matching must not collapse genuinely separate objects."""
    store = BeliefStore()
    store.update("cube", np.array([0.17, 0.15, 0.04]), conf=0.7,
                 extent=np.array([0.05, 0.05, 0.05]))
    store.update("cube", np.array([0.30, 0.16, 0.04]), conf=0.7,
                 extent=np.array([0.05, 0.05, 0.05]))
    assert len(store.all()) == 2


def test_match_radius_scales_with_object_size():
    """A big object's centre wanders more between views than a small one's.

    Two 5 cm cubes 8 cm apart are two objects; two views of a 30 cm bin
    disagreeing by 8 cm are one.
    """
    small = BeliefStore()
    small.update("cube", np.array([0.0, 0.0, 0.04]), conf=0.7,
                 extent=np.array([0.05, 0.05, 0.05]))
    small.update("cube", np.array([0.09, 0.0, 0.04]), conf=0.7,
                 extent=np.array([0.05, 0.05, 0.05]))
    assert len(small.all()) == 2

    big = BeliefStore()
    big.update("bin", np.array([0.0, 0.0, 0.03]), conf=0.7,
               extent=np.array([0.30, 0.20, 0.06]))
    big.update("bin", np.array([0.09, 0.0, 0.03]), conf=0.7,
               extent=np.array([0.30, 0.20, 0.06]))
    assert len(big.all()) == 1


def test_label_agnostic_can_be_disabled():
    store = BeliefStore(label_agnostic=False)
    pos = np.array([0.18, -0.17, 0.03])
    store.update("storage box", pos, conf=0.7)
    store.update("building block", pos, conf=0.7)
    assert len(store.all()) == 2


# ── the config ships open ────────────────────────────────────────────────

def test_shipped_config_has_no_closed_vocabulary():
    """`detect_classes` must ship EMPTY.

    A non-empty list turns the always-on world model into a closed set: an
    object not named there is invisible to the whole system however clearly
    the camera sees it. That is a booth-breaking regression, so it is pinned.
    """
    from cascade.config import load_demo_config

    cfg = load_demo_config(cameras=["mock"], arm="mock", llm="mock")
    assert not (cfg.get("detect_classes") or []), (
        "detect_classes must ship empty; a closed vocabulary hides visitor objects"
    )
