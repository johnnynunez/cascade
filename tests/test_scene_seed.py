"""Seeded kitchen prop layouts for the Isaac reset: pure geometry, no Isaac.

`cascade.sim.scene_seed` decides where `reset_props` puts every prop when a
reset seed is set. These tests use an independent oracle (their own circle
and rectangle maths) over the shipped kitchen configuration.
"""
from __future__ import annotations

import itertools
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from cascade.sim.scene_seed import (
    REBOT_BASE_KEEPOUT_M,
    kitchen_layout,
    seeded_spawns,
    validate_reset_seed,
)

ROOT = Path(__file__).resolve().parents[1]
KITCHEN = json.loads((ROOT / "demo/scene/kitchen_config.json").read_text())
S, FOOTPRINTS, KEEPOUTS = kitchen_layout(KITCHEN)
K = {"footprints": FOOTPRINTS, "keepouts": KEEPOUTS}


def _rect_distance(p, rect):
    xmin, ymin, xmax, ymax = rect
    dx = max(xmin - p[0], 0.0, p[0] - xmax)
    dy = max(ymin - p[1], 0.0, p[1] - ymax)
    return math.hypot(dx, dy)


def _in_keepout(p, radius, keepouts):
    return any(_rect_distance(p, rect) < radius - 1e-12 for rect in keepouts)


def _gap(out, a, b, footprints=FOOTPRINTS):
    return math.dist(out[a][:2], out[b][:2]) - footprints[a] - footprints[b]


def _assert_valid_layout(out, spawns, footprints, keepouts, *, fallbacks=(), jitter_m=0.04,
                         min_gap_m=0.01, reach_m=0.45):
    assert set(out) == set(spawns)
    for a, b in itertools.combinations(out, 2):
        assert _gap(out, a, b, footprints) >= min_gap_m - 1e-9, (a, b)
    for name, p in out.items():
        if name in fallbacks:
            assert p == tuple(spawns[name])
            continue
        assert math.hypot(p[0], p[1]) <= reach_m
        assert not _in_keepout(p, footprints[name], keepouts), name
        assert math.dist(p[:2], spawns[name][:2]) <= jitter_m + 1e-9
        assert p[2] == spawns[name][2]


def test_kitchen_layout_reads_the_shipped_spawns_and_footprints():
    cubes = {n: tuple(p) for n, p in KITCHEN["cube_positions"].items()}
    props = {p["name"]: tuple(p["position"]) for p in KITCHEN["props"]}
    assert S == {**cubes, **props}
    assert set(S) == {"green_cube", "pink_cube", "tomato_can", "lemon", "orange"}
    # Boxes (the cubes, spawned axis-aligned) take their XY half-diagonal;
    # round proxies (cylinder, ellipsoid) their largest XY semi-axis.
    assert FOOTPRINTS["green_cube"] == pytest.approx(math.hypot(0.0375, 0.0375) / 2)
    assert FOOTPRINTS["pink_cube"] == pytest.approx(math.hypot(0.0375, 0.0375) / 2)
    assert FOOTPRINTS["lemon"] == pytest.approx(0.0684 / 2)
    assert FOOTPRINTS["tomato_can"] == pytest.approx(0.05445 / 2)
    assert FOOTPRINTS["orange"] == pytest.approx(0.05232211947441101 / 2)


def test_kitchen_keepouts_cover_pad_box_base_and_the_counter_edge():
    pad = (0.06, -0.35, 0.22, -0.19)  # green square: centre (0.14, -0.27), 0.16 m side
    box = (0.23, -0.21, 0.37, -0.07)  # open box: centre (0.30, -0.14), 0.14 m outer
    for want in (pad, box, REBOT_BASE_KEEPOUT_M):
        assert any(all(math.isclose(a, b, abs_tol=1e-12) for a, b in zip(rect, want))
                   for rect in KEEPOUTS), want
    lo, hi = KITCHEN["counter"]["min"], KITCHEN["counter"]["max"]
    for off_counter in [(hi[0] + 1e-3, 0.0), (lo[0] - 1e-3, 0.0), (0.3, hi[1] + 1e-3), (0.3, lo[1] - 1e-3)]:
        assert _in_keepout(off_counter, 1e-6, KEEPOUTS), off_counter
    # A footprint that hangs over the counter edge is refused too.
    assert _in_keepout((0.3, hi[1] - 0.01), 0.02, KEEPOUTS)
    assert not _in_keepout((0.3, 0.2), 0.03, KEEPOUTS)


def test_base_keepout_covers_every_rebot_base_collision_sphere():
    xrdf = yaml.safe_load((ROOT / "assets/cumotion/rebot/rebot.xrdf").read_text())
    # The xrdf models base_link only in its self-collision sphere group.
    spheres = xrdf["geometry"]["rebot_self_only_collision_spheres"]["spheres"]["base_link"]
    assert spheres
    xmin, ymin, xmax, ymax = REBOT_BASE_KEEPOUT_M
    for sphere in spheres:
        (x, y, _), r = sphere["center"], sphere["radius"]
        assert xmin <= x - r and x + r <= xmax and ymin <= y - r and y + r <= ymax, sphere


def test_the_default_kitchen_layout_itself_is_valid():
    """Fallbacks keep the default spawn, so the defaults must be a valid layout."""
    _assert_valid_layout(S, S, FOOTPRINTS, KEEPOUTS, jitter_m=0.0)


@pytest.mark.parametrize("seed", [0, None])
def test_seed_zero_and_none_keep_the_default_spawns(seed):
    out, fallbacks = seeded_spawns(S, seed, **K)
    assert out == S and fallbacks == []


def test_same_seed_same_layout_other_seed_other_layout():
    assert seeded_spawns(S, 7, **K) == seeded_spawns(S, 7, **K) != seeded_spawns(S, 8, **K)


def test_layout_does_not_depend_on_dict_order_or_hash_seed():
    want, fallbacks = seeded_spawns(S, 7, **K)
    assert seeded_spawns(dict(reversed(list(S.items()))), 7, **K) == (want, fallbacks)
    code = ("import json, sys; from cascade.sim.scene_seed import kitchen_layout, seeded_spawns;"
            "s, f, k = kitchen_layout(json.load(open(sys.argv[1])));"
            "print(json.dumps(seeded_spawns(s, 7, footprints=f, keepouts=k)))")
    for hash_seed in ("0", "4242"):
        env = {**os.environ, "PYTHONHASHSEED": hash_seed, "PYTHONPATH": str(ROOT / "src"),
               "PYTHONDONTWRITEBYTECODE": "1"}
        done = subprocess.run([sys.executable, "-c", code, str(ROOT / "demo/scene/kitchen_config.json")],
                              env=env, capture_output=True, text=True, timeout=30, check=True)
        out, again = json.loads(done.stdout)
        assert {n: tuple(p) for n, p in out.items()} == want and again == fallbacks


@pytest.mark.parametrize("seed", range(1, 201))
def test_layouts_respect_gaps_keepouts_reach_and_jitter(seed):
    out, fallbacks = seeded_spawns(S, seed, **K)
    _assert_valid_layout(out, S, FOOTPRINTS, KEEPOUTS, fallbacks=fallbacks)
    assert fallbacks == sorted(fallbacks) and set(fallbacks) <= set(S)
    # A placed prop really moved; a fallback stayed exactly on its spawn.
    assert all(out[n] != S[n] for n in S if n not in fallbacks)


def test_seeded_layouts_rarely_fall_back():
    placed = sum(len(S) - len(seeded_spawns(S, seed, **K)[1]) for seed in range(1, 201))
    assert placed >= 0.95 * 200 * len(S)


def test_unplaceable_prop_keeps_its_spawn_and_is_listed():
    # A keepout covering the lemon's whole jitter disc: no candidate can fit.
    x, y, _ = S["lemon"]
    blocked = {**K, "keepouts": [*KEEPOUTS, (x - 0.05, y - 0.05, x + 0.05, y + 0.05)]}
    for seed in range(1, 41):
        out, fallbacks = seeded_spawns(S, seed, **blocked)
        assert out["lemon"] == S["lemon"] and fallbacks == ["lemon"]
        # Every other prop kept clear of where the lemon stayed.
        _assert_valid_layout(out, S, FOOTPRINTS, blocked["keepouts"], fallbacks=fallbacks)


def test_reach_limit_rejects_far_candidates():
    spawns = {"far": (0.44, 0.0, 0.02)}
    for seed in range(1, 41):
        out, fallbacks = seeded_spawns(spawns, seed, footprints={"far": 0.02}, keepouts=[])
        assert fallbacks == [] and math.hypot(*out["far"][:2]) <= 0.45


def test_rounding_never_pushes_a_prop_past_its_jitter():
    # Every micrometre grid point is >= 0.64 um from this spawn, so with a
    # 0.3 um jitter the rounding alone would leave the disc: the prop stays.
    spawns = {"tiny": (0.30000045, 0.10000045, 0.02)}
    for seed in range(1, 21):
        out, fallbacks = seeded_spawns(spawns, seed, footprints={"tiny": 0.02}, keepouts=[],
                                       jitter_m=3e-7)
        assert math.dist(out["tiny"][:2], spawns["tiny"][:2]) <= 3e-7
        assert fallbacks == ["tiny"]


def test_jitter_and_gap_arguments_are_honoured():
    for seed in range(1, 41):
        out, fallbacks = seeded_spawns(S, seed, **K, jitter_m=0.01, min_gap_m=0.02)
        _assert_valid_layout(out, S, FOOTPRINTS, KEEPOUTS, fallbacks=fallbacks,
                             jitter_m=0.01, min_gap_m=0.02)


@pytest.mark.parametrize("seed", ["5", 5.0, True, False, -1, 2**31, [5], float("nan")])
def test_bad_seeds_fail_closed(seed):
    with pytest.raises(ValueError, match="seed"):
        validate_reset_seed(seed)
    with pytest.raises(ValueError, match="seed"):
        seeded_spawns(S, seed, **K)


@pytest.mark.parametrize("seed", [None, 0, 1, 205, 2**31 - 1])
def test_good_seeds_pass_validation_unchanged(seed):
    assert validate_reset_seed(seed) == seed


@pytest.mark.parametrize("change", [
    "missing-footprint", "negative-footprint", "inverted-keepout-x", "inverted-keepout-y", "nan-keepout",
    "nan-spawn", "negative-jitter", "zero-tries",
])
def test_malformed_layout_inputs_fail_closed(change):
    spawns, footprints, keepouts = dict(S), dict(FOOTPRINTS), list(KEEPOUTS)
    kwargs = {}
    if change == "missing-footprint":
        del footprints["lemon"]
    elif change == "negative-footprint":
        footprints["lemon"] = -0.01
    elif change == "inverted-keepout-x":
        keepouts.append((0.3, 0.3, 0.2, 0.4))
    elif change == "inverted-keepout-y":
        keepouts.append((0.1, 0.4, 0.2, 0.3))
    elif change == "nan-keepout":
        keepouts.append((0.1, float("nan"), 0.2, 0.3))
    elif change == "nan-spawn":
        spawns["lemon"] = (float("nan"), 0.0, 0.02)
    elif change == "negative-jitter":
        kwargs["jitter_m"] = -0.01
    else:
        kwargs["max_tries"] = 0
    with pytest.raises(ValueError):
        seeded_spawns(spawns, 3, footprints=footprints, keepouts=keepouts, **kwargs)


@pytest.mark.parametrize("missing", ["target_pad", "open_box", "counter", "cube_dimensions"])
def test_kitchen_layout_refuses_an_incomplete_kitchen(missing):
    config = json.loads(json.dumps(KITCHEN))
    del config[missing]
    with pytest.raises(ValueError):
        kitchen_layout(config)


@pytest.mark.parametrize("config", [None, [], "kitchen", 5], ids=["none", "list", "string", "int"])
def test_kitchen_layout_refuses_a_non_dict_config_with_value_error(config):
    # The docstring's contract: every unusable config is a ValueError.
    with pytest.raises(ValueError, match="scene config"):
        kitchen_layout(config)


@pytest.mark.parametrize("argument,value", [
    ("spawns", None), ("spawns", 5), ("spawns", [("lemon", (0.3, 0.0, 0.02))]),
    ("footprints", None), ("footprints", [0.02]),
    ("keepouts", None), ("keepouts", 5),
])
def test_seeded_spawns_refuses_none_or_non_iterable_inputs_with_value_error(argument, value):
    kwargs = {"spawns": dict(S), "footprints": dict(FOOTPRINTS), "keepouts": list(KEEPOUTS), argument: value}
    for seed in (0, 3):  # refused whether or not the seed moves anything
        with pytest.raises(ValueError, match=argument):
            seeded_spawns(kwargs["spawns"], seed, footprints=kwargs["footprints"], keepouts=kwargs["keepouts"])
