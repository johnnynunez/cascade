"""Helpers shared by the base_pose Option 2 tests (spec
docs/superpowers/specs/2026-10-17-base-pose-runtime-frame-design.md).

Task 1 only needs the hand-built-rig helper; Task 3 adds the equivariance
oracle here.
"""

from __future__ import annotations


def bind_like_primary(rt, *names) -> None:
    """Register the runtime's own Kinematics and Cfg for hand-built rig arms.

    `execute(arm=<non-primary>)` refuses an arm with no bound kinematics /
    profile (spec D4: fail closed). Tests that build an ArmRig by hand (data
    doubles for the arms, one shared profile) register the primary's binding
    for their other arm names, which is exactly what those arms ran with
    before the per-call frame existed.
    """
    from cascade.skills.arm_frame import ArmBinding

    rt.register_arm_bindings({str(name): ArmBinding(rt.kin, rt.cfg, None) for name in names})
