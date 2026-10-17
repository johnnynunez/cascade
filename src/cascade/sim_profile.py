"""Simulation or real robot, decided from the configuration alone (ASPIRE spec §4.4).

Learned notes and the bounded retry behave differently on a simulated rig and on
hardware: a sim profile may serve notes on their task count and may grant one
retry after a refusal; a real profile gets no retry and only verifier-passed
notes. The class is read from the arm profiles the runtime builds
(`apps.demo._arm_cfgs`): "sim" only when EVERY arm `type` is a simulated backend
and the booth overlay is off. Everything else -- a hardware type, a missing or
unknown type, an empty arm list, a mobile or composed config, a config this
module cannot read -- is "real" (fail closed). No arm object is ever touched, so
a LazyArm is never materialized.
"""
from __future__ import annotations

import os
from typing import Literal

#: arm `type` values whose backend is a simulator or a mock (control/arm_base.make_arm)
SIM_ARM_TYPES = frozenset({"isaac", "mujoco", "mock"})

#: CASCADE_SIM_RETRY values that leave the sim retry on (unset counts as empty);
#: any other value is the kill switch, so a typo can only turn the retry off
_RETRY_ON_VALUES = frozenset({"", "1", "true", "yes", "on"})


def arm_types(cfg) -> list[str]:
    """The `type` of every arm profile in `cfg`, in rig order. A type that is not a
    string reads as "". A config without readable arm profiles gives []."""
    try:
        from .apps.demo import _arm_cfgs

        return [kind if isinstance(kind, str) else "" for kind in (arm.get("type") for arm in _arm_cfgs(cfg))]
    except Exception:  # noqa: BLE001 -- unreadable config: no arms, so "real"
        return []


def profile_class(cfg) -> Literal["sim", "real"]:
    """Return "sim" only when the booth is off and every arm type is in SIM_ARM_TYPES, else "real"."""
    try:
        from .config import booth_mode_enabled

        if booth_mode_enabled() or cfg.get("booth_mode"):
            return "real"
        types = arm_types(cfg)
        if types and all(kind in SIM_ARM_TYPES for kind in types):
            return "sim"
        return "real"
    except Exception:  # noqa: BLE001 -- fail closed
        return "real"


def _agent_flag(cfg, key: str):
    """`agent.<key>` as written in the config, or None when absent or unreadable."""
    try:
        agent = cfg.get("agent")
        return None if agent is None else agent.get(key)
    except Exception:  # noqa: BLE001
        return None


def sim_retry_enabled(cfg) -> bool:
    """One bounded retry after a refusal: only on a sim profile with
    `agent.sim_retry: true` and the kill switch CASCADE_SIM_RETRY not set to off."""
    if profile_class(cfg) != "sim" or _agent_flag(cfg, "sim_retry") is not True:
        return False
    return os.environ.get("CASCADE_SIM_RETRY", "").strip().lower() in _RETRY_ON_VALUES


def learning_surface_enabled(cfg) -> bool:
    """The chat line and panel switch `agent.learning_surface` (true only when the
    config says exactly true), on any profile class."""
    return _agent_flag(cfg, "learning_surface") is True
