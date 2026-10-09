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
        self_mask: consult the frame's render self-mask (``Frame.robot_mask``
            minus a held ``payload_mask``) when it carries one. The base
            cylinder only covers the links near the base: on Isaac the side
            camera sees the upper arm 0.42 m up and 0.32 m out, and YOLOE
            named it "biplane", which became a belief (backlog B32).
        self_mask_max_frac: a detection with more than this fraction of its
            pixels on the robot IS the robot, wherever it is. Measured on
            Isaac (both demo cameras, 24 rounds): robot detections 0.885 to
            0.995, props at most 0.046. Below it, the robot's pixels are
            removed before the mask is lifted to 3D.
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
    self_mask: bool = True
    self_mask_max_frac: float = 0.5

    def __post_init__(self) -> None:
        if not 0.0 < float(self.self_mask_max_frac) <= 1.0:
            raise ValueError("self_mask_max_frac must be in (0, 1]")

    def self_pixels(self, frame) -> np.ndarray | None:
        """The robot's own pixels in ``frame``: its render self-mask minus a
        held payload (a prop in the jaws is still an object).

        None when the gate is off, the frame carries no self-mask (the real
        rig, unless fusion handed in a copy carrying B39's link-geometry mask,
        `perception/link_mask.py`), or its payload mask does not align with
        it; nothing is excluded then.
        """
        if not self.self_mask:
            return None
        robot = getattr(frame, "robot_mask", None)
        if robot is None:
            return None
        robot = np.asarray(robot, dtype=bool)
        payload = getattr(frame, "payload_mask", None)
        if payload is not None:
            payload = np.asarray(payload, dtype=bool)
            if payload.shape != robot.shape:
                return None
            robot = robot & ~payload
        return robot

    def exclude_self(self, mask, self_px):
        """Split one detection mask against the robot's pixels.

        Returns ``(rest, frac)``. ``frac`` is the fraction of the detection's
        pixels that are robot pixels. ``rest`` is the mask without them, or
        None when ``frac`` exceeds ``self_mask_max_frac`` (the detection is the
        robot). Without usable self pixels (none, or another shape) the mask
        comes back untouched with ``frac`` None. Accepts numpy masks and the
        strict-CUDA detector's torch masks, which stay on their device.
        """
        if self_px is None or tuple(mask.shape) != tuple(self_px.shape):
            return mask, None
        if hasattr(mask, "device") and hasattr(mask, "numel"):
            import torch

            m = mask.to(torch.bool)
            s = torch.as_tensor(self_px, device=m.device)
            n = int(m.sum().item())
            on = int((m & s).sum().item()) if n else 0
            rest = (m & ~s) if on else mask
        else:
            m = np.asarray(mask, dtype=bool)
            n = int(np.count_nonzero(m))
            on = int(np.count_nonzero(m & self_px)) if n else 0
            rest = (m & ~self_px) if on else mask
        frac = on / n if n else 0.0
        if frac > self.self_mask_max_frac:
            return None, frac
        return rest, frac

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

        def flag(key: str, default: bool) -> bool:
            v = cfg.get(key, default)
            if v is None:
                return default
            if not isinstance(v, bool):
                raise ValueError(f"workspace_filter.{key} must be true or false, got {v!r}")
            return v

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
            self_mask=flag("self_mask", base.self_mask),
            self_mask_max_frac=f("self_mask_max_frac", base.self_mask_max_frac),
        )
