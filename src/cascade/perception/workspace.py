"""Geometric gate on detections: is this a manipuland, or scenery?

Open-vocabulary perception is a hard requirement for a public booth: a visitor
puts an arbitrary object on the table and the robot has to see it without
anyone having listed it in advance.  The cost is that a ~4.5k-concept
vocabulary happily names things that are not manipulands.  Measured on the
Isaac scene with 3 real props, one frame:

    studio shot     0.756   the whole scene, as one detection
    storage box     0.713   the bin           (real)
    cube            0.704   a cube            (real)
    hassock         0.613   the other cube    (real)
    transformer     0.310   the robot itself
    amplifier       0.299   the robot itself
    computer tower  0.272   the robot itself
    underdrawers    0.258   the robot itself

Those are not hallucinations.  The detector is right that something is there;
it is simply naming the arm, the table and the scene.  A closed vocabulary hid
this by having no word for any of them, which is why the fix for one problem
(a hard-coded 8-word list) exposed the other.

The filter is deliberately **class-agnostic**: it decides from geometry alone
and never looks at the label.  Anything keyed off names would re-introduce the
closed set through the back door, and would break the moment a visitor brings
an object nobody enumerated.

Every threshold is a property of the RIG (where the arm is bolted, how far it
reaches, how big its jaws are), not of the objects, so they stay valid for
objects never seen before.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class WorkspaceFilter:
    """Accept only detections that could be an object on the table.

    Args:
        base_radius_m: cylinder around the arm's own base to exclude. The
            robot is always in frame on an eye-in-hand rig and is the single
            largest source of non-manipuland detections.
        base_height_m: how far up that cylinder extends. Kept generous: the
            links stand well above the table.
        max_extent_m: largest OBB extent for a manipuland. Tables, walls and
            whole-scene detections exceed it.
        min_extent_m: smallest OBB extent worth reporting; below this the
            "object" is a depth speck or a mask sliver.
        table_z_m: points below this are under the table surface.
        max_z_m: points above this are not on the table (ceiling, backdrop).
        reach_m: horizontal distance beyond which the arm cannot act anyway.
        max_frame_frac: a mask covering more of the image than this is the
            scene, not a thing in it.
        min_height_m: an object's centre must sit at least this far above the
            table plane. Shadows and surface markings lift to a flat patch at
            z ~= 0 and are otherwise indistinguishable from a thin object;
            measured on the rig the lowest real prop centres at 2.8 cm.
    """

    base_radius_m: float = 0.12
    base_height_m: float = 0.50
    max_extent_m: float = 0.35
    min_extent_m: float = 0.015
    table_z_m: float = -0.05
    max_z_m: float = 0.60
    reach_m: float = 1.00
    max_frame_frac: float = 0.45
    min_height_m: float = 0.010

    def reject(
        self,
        center: np.ndarray,
        extents: np.ndarray | None = None,
        mask_frac: float | None = None,
    ) -> str | None:
        """Return None to accept, else a short reason string.

        The reason is returned rather than a bare bool so callers can log or
        surface WHY something was dropped; a booth demo that silently discards
        a visitor's object is worse than one that says it ignored it.
        """
        c = np.asarray(center, dtype=float).reshape(3)

        if mask_frac is not None and mask_frac > self.max_frame_frac:
            return "scene"

        if extents is not None:
            e = np.asarray(extents, dtype=float).reshape(3)
            if float(np.max(e)) > self.max_extent_m:
                return "oversize"
            if float(np.max(e)) < self.min_extent_m:
                return "speck"

        # The arm's own base, as a cylinder: the links rise above the table
        # near the origin, so a sphere would either miss them or eat the
        # workspace.
        if float(np.hypot(c[0], c[1])) < self.base_radius_m and c[2] < self.base_height_m:
            return "robot"

        if c[2] < self.table_z_m:
            return "below_table"
        if c[2] < self.min_height_m:
            return "flat"  # shadow or surface marking, not an object
        if c[2] > self.max_z_m:
            return "too_high"
        if float(np.hypot(c[0], c[1])) > self.reach_m:
            return "out_of_reach"
        return None

    @classmethod
    def from_config(cls, cfg) -> "WorkspaceFilter":
        """Build from a `workspace_filter:` config block, falling back to the
        defaults above for anything unset."""
        base = cls()
        if cfg is None or not hasattr(cfg, "get"):
            return base

        def f(key: str, default: float) -> float:
            v = cfg.get(key, default)
            return default if v is None else float(v)

        return cls(
            base_radius_m=f("base_radius_m", base.base_radius_m),
            base_height_m=f("base_height_m", base.base_height_m),
            max_extent_m=f("max_extent_m", base.max_extent_m),
            min_extent_m=f("min_extent_m", base.min_extent_m),
            table_z_m=f("table_z_m", base.table_z_m),
            max_z_m=f("max_z_m", base.max_z_m),
            reach_m=f("reach_m", base.reach_m),
            max_frame_frac=f("max_frame_frac", base.max_frame_frac),
            min_height_m=f("min_height_m", base.min_height_m),
        )
