"""Seeded prop layouts for the Isaac kitchen reset; pure Python, no Isaac.

`scripts/isaac_bridge.py` imports this module: with a reset seed set (bridge
op `set_reset_seed`), `reset_props` places and verifies every prop on
`seeded_spawns(...)` instead of its fixed spawn. No seed (every live demo)
keeps today's spawns. The module decides prop poses inside the kitchen
scene, so it is pinned with the bridge in `demo/scene/own_assets.json`
(`demo/scene_identity.py` SOURCE_FILES).

Frame: robot base, metres; the kitchen's robot base sits at the origin.
"""
from __future__ import annotations

import math
import random

#: Largest accepted reset seed (seeds are small run indices: 1..8, 101..105, ...).
MAX_RESET_SEED = 2**31 - 1

#: reBot base_link footprint (xmin, ymin, xmax, ymax). Covers the XY extent of
#: every base_link sphere in assets/cumotion/rebot/rebot.xrdf
#: (`rebot_self_only_collision_spheres`: x within +-0.094, y within +-0.124),
#: rounded outward. A prop born inside the base collider is ejected on the
#: first physics tick (isaac_bridge.py PROPS).
REBOT_BASE_KEEPOUT_M = (-0.10, -0.13, 0.10, 0.13)

_ROUND_PROXIES = {"cylinder", "ellipsoid"}


def validate_reset_seed(seed):
    """Return `seed` (None or an int in [0, MAX_RESET_SEED]); raise ValueError otherwise."""
    if seed is None:
        return None
    if type(seed) is not int or not 0 <= seed <= MAX_RESET_SEED:
        raise ValueError(f"reset seed must be null or an integer in 0..{MAX_RESET_SEED}, got {seed!r}")
    return seed


def _finite(values, what):
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in values):
        raise ValueError(f"{what} must be finite metres")
    return tuple(float(v) for v in values)


def _rect_distance(x, y, rect):
    xmin, ymin, xmax, ymax = rect
    return math.hypot(max(xmin - x, 0.0, x - xmax), max(ymin - y, 0.0, y - ymax))


def seeded_spawns(spawns, seed, *, footprints, keepouts, jitter_m=0.04, min_gap_m=0.01,
                  reach_m=0.45, max_tries=50):
    """Jitter every prop's XY spawn by a deterministic, seeded amount.

    Seed None or 0 returns the spawns unchanged. Otherwise props are placed in
    sorted name order: each draws up to `max_tries` uniform points in the disc
    of radius `jitter_m` around its spawn (z unchanged) and takes the first
    whose footprint circle stays `min_gap_m` clear of every other prop's
    circle (placed props at their new spot, unplaced ones at their spawn),
    overlaps no keepout rectangle and whose centre is within `reach_m` of the
    base. A prop with no such point keeps its spawn and is listed in
    `fallbacks`; because earlier props kept clear of it, the layout stays valid.
    Coordinates are rounded to the micrometre before they are checked.

    `spawns` (name -> 3D position) and `footprints` (name -> radius) are
    dicts; `keepouts` is an iterable of (xmin, ymin, xmax, ymax) rectangles.
    Returns `(layout, fallbacks)`. Raises ValueError on a bad seed or input,
    including None or non-iterable spawns, footprints or keepouts.
    """
    seed = validate_reset_seed(seed)
    if (type(max_tries) is not int or max_tries < 1
            or not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
                       for v in (jitter_m, min_gap_m, reach_m))
            or jitter_m < 0 or min_gap_m < 0 or reach_m <= 0):
        raise ValueError("seeded layout needs jitter_m >= 0, min_gap_m >= 0, reach_m > 0, max_tries >= 1")
    if not isinstance(spawns, dict):
        raise ValueError("seeded layout spawns must be a dict of name -> position")
    if not isinstance(footprints, dict):
        raise ValueError("seeded layout footprints must be a dict of name -> radius")
    try:
        keepouts = list(keepouts)
    except TypeError:
        raise ValueError("seeded layout keepouts must be an iterable of rectangles") from None
    current = {}
    for name, position in spawns.items():
        if not isinstance(name, str) or not isinstance(position, (tuple, list)) or len(position) != 3:
            raise ValueError(f"seeded layout spawn {name!r} must be a 3D position")
        current[name] = _finite(position, f"spawn {name}")
    radii = {}
    for name in current:
        radius = footprints.get(name)
        if radius is None:
            raise ValueError(f"seeded layout has no footprint for {name}")
        radius = _finite([radius], f"footprint {name}")[0]
        if radius <= 0:
            raise ValueError(f"seeded layout footprint for {name} must be positive")
        radii[name] = radius
    rects = []
    for rect in keepouts:
        if not isinstance(rect, (tuple, list)) or len(rect) != 4:
            raise ValueError("keepouts must be (xmin, ymin, xmax, ymax) rectangles")
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or math.isnan(v) for v in rect):
            raise ValueError("keepout rectangle values must be numbers")
        xmin, ymin, xmax, ymax = (float(v) for v in rect)
        if xmin > xmax or ymin > ymax:
            raise ValueError("keepout rectangle must have min <= max")
        rects.append((xmin, ymin, xmax, ymax))
    if not seed:
        return dict(current), []

    def fits(name, x, y):
        x0, y0, _ = spawn_of[name]
        if math.hypot(x, y) > reach_m or math.hypot(x - x0, y - y0) > jitter_m:
            return False
        radius = radii[name]
        if any(_rect_distance(x, y, rect) < radius for rect in rects):
            return False
        return all(math.hypot(x - ox, y - oy) >= radius + radii[other] + min_gap_m
                   for other, (ox, oy, _) in current.items() if other != name)

    rng = random.Random(seed)
    spawn_of = dict(current)
    fallbacks = []
    for name in sorted(current):
        x0, y0, z0 = spawn_of[name]
        for _ in range(max_tries):
            distance = jitter_m * math.sqrt(rng.random())
            angle = 2.0 * math.pi * rng.random()
            x = round(x0 + distance * math.cos(angle), 6)
            y = round(y0 + distance * math.sin(angle), 6)
            if fits(name, x, y):
                current[name] = (x, y, z0)
                break
        else:
            fallbacks.append(name)
    return {name: current[name] for name in spawns}, fallbacks


def _rect(center, size_x, size_y):
    x, y = _finite(center, "keepout centre")
    return (x - size_x / 2, y - size_y / 2, x + size_x / 2, y + size_y / 2)


def kitchen_layout(scene_config):
    """Spawns, footprint radii and keepouts of the kitchen scene config.

    Returns `(spawns, footprints, keepouts)`:
    - spawns: `cube_positions` plus every `props[].position` (body centres);
    - footprints: XY bounding-circle radius from `cube_dimensions` /
      `props[].dimensions` (half the XY diagonal for boxes, the largest XY
      semi-axis for cylinder and ellipsoid proxies);
    - keepouts: the green-square `target_pad`, the `open_box` (outer size),
      the reBot base and everything off the `counter` top.
    Raises ValueError when the config is not a dict (None included) or any of
    them is missing or malformed.
    """
    if not isinstance(scene_config, dict):
        raise ValueError("kitchen layout needs the kitchen scene config (a dict)")
    try:
        cube_positions = scene_config["cube_positions"]
        cube_dimensions = scene_config["cube_dimensions"]
        pad, box, counter = scene_config["target_pad"], scene_config["open_box"], scene_config["counter"]
        props = scene_config["props"]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"kitchen layout needs {exc} in the scene config") from None
    spawns, footprints = {}, {}

    def add(name, position, dimensions, proxy):
        if not isinstance(name, str) or name in spawns:
            raise ValueError(f"kitchen layout prop name {name!r} is invalid or repeated")
        if not isinstance(position, list) or len(position) != 3:
            raise ValueError(f"kitchen layout {name} needs a 3D position")
        if not isinstance(dimensions, list) or len(dimensions) != 3:
            raise ValueError(f"kitchen layout {name} needs 3D dimensions")
        dx, dy, _ = _finite(dimensions, f"{name} dimensions")
        if dx <= 0 or dy <= 0:
            raise ValueError(f"kitchen layout {name} dimensions must be positive")
        spawns[name] = _finite(position, f"{name} position")
        footprints[name] = max(dx, dy) / 2 if proxy in _ROUND_PROXIES else math.hypot(dx, dy) / 2

    try:
        for name in sorted(cube_positions):
            add(name, cube_positions[name], cube_dimensions.get(name), "box")
        for prop in props:
            add(prop.get("name"), prop.get("position"), prop.get("dimensions"), prop.get("proxy", "box"))
        keepouts = [
            _rect(pad["center_xy_m"], pad["outer_size_m"], pad["outer_size_m"]),
            _rect(box["center_xy_m"], *box["outer_dimensions_m"][:2]),
            REBOT_BASE_KEEPOUT_M,
        ]
        (xmin, ymin, _), (xmax, ymax, _) = (_finite(counter["min"], "counter"),
                                            _finite(counter["max"], "counter"))
    except (AttributeError, KeyError, TypeError) as exc:
        raise ValueError(f"kitchen layout config is malformed: {exc!r}") from None
    if xmin >= xmax or ymin >= ymax:
        raise ValueError("kitchen counter must have min < max")
    inf = math.inf
    keepouts += [(-inf, -inf, xmin, inf), (xmax, -inf, inf, inf),
                 (-inf, -inf, inf, ymin), (-inf, ymax, inf, inf)]
    return spawns, footprints, keepouts
