"""Both reBot RS backends take their settle window from the arm profile.

ArmBase documents `settle_timeout_s` as a per-backend override "from config
(`arm.settle_timeout_s`)", and the motorbridge sibling reads it, but the SDK
backend silently kept the class default. Seeed's WRC rig needed a longer
window for large joint moves (commits e97998c: 2.0 -> 4.0 s, ba4e110: 6.0 s,
"IK settle oscillation" on an 86.7 deg joint-4 swing); that has to be an
onsite profile edit, not a code change. The DEFAULT is unchanged (2.0 s)
and no shipped profile changes its value here.
"""

from __future__ import annotations

import pytest

from cascade.config import Cfg
from cascade.control.arm_base import ArmBase


def _rs(cfg):
    from cascade.control.rebot_rs_arm import RebotRSArm

    return RebotRSArm(Cfg(cfg))


def _mb(cfg):
    from cascade.control.rebot_rs_mb_arm import RebotRSMotorBridgeArm

    return RebotRSMotorBridgeArm(Cfg({"mit_kp": [1.0] * 6, "mit_kd": [0.1] * 6, **cfg}))


@pytest.mark.parametrize("build", [_rs, _mb], ids=["rebot_rs", "rebot_rs_mb"])
def test_settle_timeout_comes_from_the_profile(build):
    assert build({"settle_timeout_s": 4.5}).settle_timeout_s == pytest.approx(4.5)


@pytest.mark.parametrize("build", [_rs, _mb], ids=["rebot_rs", "rebot_rs_mb"])
def test_settle_timeout_default_is_unchanged(build):
    assert build({}).settle_timeout_s == pytest.approx(ArmBase.settle_timeout_s)
