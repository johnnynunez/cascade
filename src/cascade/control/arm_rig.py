"""N named arms behind one handle. The first arm is the manipulation arm.

The deliberate twin of `perception/stream.py`'s CameraRig: same shape (named
members, ordered, `primary` is the first, `get(None)` returns it), so a reader
who knows one knows the other. That symmetry is the point -- the rig already
supported N cameras and exactly one arm, and "second arm" was the only axis of
the system that needed new structure rather than a new profile.

WHAT THIS IS NOT: a dual-arm *controller*. There is no shared trajectory
optimizer, no inter-arm collision check, no synchronized bimanual primitive.
Each arm keeps its OWN SafetyHarness (own workspace box, own joint limits, own
velocity cap) and moves independently. Two arms that share a table can still
hit each other -- see `keep_out` below.

Why per-arm harnesses rather than one shared one: every limit in SafetyLimits
is a property of a particular robot on a particular table. A 5-DoF SO-101 with
a 0.48 m reach and a 6-DoF reBot with 0.50 m have different workspace boxes,
different velocity caps (printed PLA vs. RobStride) and different joint counts.
One harness would have to hold the intersection, which is both wrong (it would
reject poses arm A can reach) and unsafe (it would accept the looser of two
velocity caps for the weaker arm). `configs/arms/*.yaml` already declares these
per robot in its `overrides:` block; the rig just stops them from being merged
into one global blob.

INTER-ARM COLLISION IS NOT SOLVED HERE, and this docstring is the warning. The
harness vets one arm's links against the table, the workspace box and the
keep-out list -- it has no idea another arm exists. For a real shared-envelope
dual-arm setup, give each arm a `keep_out` box covering the other's side of the
table (see configs/arms/dual_so101.yaml); that is a static partition, not true
collision avoidance, and it is the honest limit of what this ships with.
"""

from __future__ import annotations

import sys


class ArmRig:
    """N named arms. The first is the manipulation arm.

    Members are whatever the composition root wraps them in -- in the shipped
    stack, SafeArm instances, so the skill layer never sees a raw backend
    (the invariant in safety/harness.py). The rig does not enforce that: it is
    a container, and tests wire mocks through it directly.
    """

    def __init__(self, arms: list, names: list[str]):
        if not arms:
            raise ValueError("ArmRig needs at least one arm")
        if len(arms) != len(names):
            raise ValueError(
                f"ArmRig got {len(arms)} arms but {len(names)} names; they are "
                "positional and must correspond"
            )
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate arm names in rig: {names}")
        self.arms: dict[str, object] = dict(zip(names, arms))
        self._order = list(names)

    @property
    def primary(self):
        """The manipulation arm: what `runtime.arm` binds to."""
        return self.arms[self._order[0]]

    @property
    def names(self) -> list[str]:
        return list(self._order)

    def __iter__(self):
        return (self.arms[n] for n in self._order)

    def __len__(self) -> int:
        return len(self._order)

    def get(self, name: str | None):
        """Look one up by name; None yields the primary.

        Skills that take an optional `arm` argument route through here, so
        omitting it keeps the single-arm behaviour byte for byte.
        """
        if name is None:
            return self.primary
        if name not in self.arms:
            raise KeyError(f"no arm {name!r}; available: {self._order}")
        return self.arms[name]

    def connect(self) -> None:
        """Connect every arm, unwinding cleanly if one fails.

        Same all-or-nothing contract as CameraRig.open(): a half-connected rig
        is worse than none, because the caller's teardown only knows about the
        arms it thinks it has. Arms already connected are disconnected before
        the exception propagates -- and for a real arm `disconnect()` disables
        torque, so partial state here means a powered, unsupervised motor.
        """
        connected = []
        try:
            for a in self:
                a.connect()
                connected.append(a)
        except Exception:
            for a in connected:
                try:
                    a.disconnect()
                except Exception:
                    pass
            raise

    def disconnect(self) -> None:
        """Disconnect every arm; never raises (teardown path)."""
        for name in self._order:
            try:
                self.arms[name].disconnect()
            except Exception as e:
                print(f"[armrig] disconnect {name}: {e}", file=sys.stderr)

    def stop(self) -> None:
        """E-stop every arm, even if one raises.

        The loop deliberately swallows per-arm errors and keeps going: a
        failure stopping arm 1 must never leave arm 2 running. Errors are
        reported after every arm has been told to stop.
        """
        errors = []
        for name in self._order:
            try:
                self.arms[name].stop()
            except Exception as e:  # noqa: BLE001 - every arm must get the call
                errors.append(f"{name}: {e}")
        if errors:
            print(f"[armrig] stop errors: {'; '.join(errors)}", file=sys.stderr)

    def stats(self) -> dict:
        """Per-arm snapshot for the dashboard.

        Deliberately does NOT touch a LazyArm that has not materialized:
        reading state would power the motors as a side effect of rendering a
        status panel (the LazyArm trap in CLAUDE.md). `connected` is on
        LazyArm's own surface, so probing it is safe.
        """
        out = {}
        for name in self._order:
            arm = self.arms[name]
            entry: dict = {"connected": bool(getattr(arm, "connected", True))}
            if entry["connected"]:
                try:
                    entry["q"] = [round(float(v), 4) for v in arm.get_state().q]
                except Exception as e:  # noqa: BLE001 - status must not crash
                    entry["error"] = str(e)
            harness = getattr(arm, "harness", None)
            if harness is not None:
                entry["estopped"] = bool(harness.estopped)
            out[name] = entry
        return out
