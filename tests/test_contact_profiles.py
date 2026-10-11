"""Which arm profiles ship the motor-feedback contact monitor, and with what.

* the real reBot RS profiles enable it; `rebot_rs` (and `rebot_rs_reach`,
  which extends it) enforce, `rebot_rs_mb` (gains unvalidated, no gravity
  compensation) measures until calibrated;
* every other profile -- mock, sims, other arms -- leaves it off;
* the tolerance stays above each enabled profile's settle tolerance, so
  the monitor never stops a motion whose final error the end-of-move
  settle check would accept (light grasp-descent contact keeps working);
* the demo's harness is built with it (offline: MockArm stands in for the
  RS backend, nothing touches a bus).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cascade.config import load_demo_config
from cascade.safety.contact import ContactMonitorConfig
from cascade.safety.harness import SafetyLimits

ARMS = Path(__file__).resolve().parent.parent / "configs" / "arms"
PROFILES = sorted(p.stem for p in ARMS.glob("*.yaml"))
EXPECTED = {"rebot_rs": "enforce", "rebot_rs_reach": "enforce", "rebot_rs_mb": "measure"}


def contact(profile: str) -> ContactMonitorConfig | None:
    cfg = load_demo_config(camera="mock", arm=profile, llm="mock")
    return SafetyLimits.from_config(cfg.safety).contact, cfg


@pytest.mark.parametrize("profile", PROFILES)
def test_who_ships_the_monitor(profile):
    c, _ = contact(profile)
    if profile in EXPECTED:
        assert c is not None and c.enabled, f"{profile} must monitor contact"
        assert c.mode == EXPECTED[profile]
    else:
        assert c is None or not c.enabled, f"{profile}: off unless calibrated for it"


@pytest.mark.parametrize("profile", sorted(EXPECTED))
def test_rs_thresholds_are_conservative_and_above_settle(profile):
    c, cfg = contact(profile)
    settle_tol = float(cfg.arm.get("settle_tol", 0.05))
    assert len(c.track_tol_rad) in (1, int(cfg.arm.n_joints))
    # never trips on a final error the settle check would accept
    assert min(c.track_tol_rad) >= settle_tol + 0.02
    # 7x the worst segment error measured on the rig (0.013 rad, 2026-10-11)
    assert min(c.track_tol_rad) >= 7 * 0.013
    # the MIT loop has no velocity feedforward: kd/kp up to ~0.1 s of lag
    assert c.lag_s >= 0.2
    assert c.debounce_samples >= 2
    # a 50 Hz stream reading 1-2 of 6 joints per cycle samples each joint
    # every 60-120 ms; the gap must allow one missed read
    assert c.max_feedback_gap_s >= 0.3
    assert c.contact_zone_m > 0


def test_the_demo_block_documents_the_monitor_disabled():
    text = (ARMS.parent / "demo.yaml").read_text()
    assert "contact_monitor:" in text
    cfg = load_demo_config(camera="mock", arm="mock", llm="mock")
    c = SafetyLimits.from_config(cfg.safety).contact
    assert c is None or not c.enabled


def test_the_rebot_rs_harness_monitors_every_motion(tmp_path, monkeypatch):
    """build_runtime on the rebot_rs profile with a MockArm in place of the
    RS backend (no bus): its SafeArm hands every stream a monitor."""
    from cascade.apps import demo
    from cascade.control.mock_arm import MockArm
    from cascade.safety.contact import ContactMonitor

    monkeypatch.setattr(demo, "make_arm", lambda a, kinematics=None: MockArm(a, kinematics=kinematics))
    cfg = load_demo_config(camera="mock", arm="rebot_rs", llm="mock")
    raw, safe, _kin = demo._build_arm(cfg.arm, lazy_arm=False, occupancy=None, fallback_cfg=cfg)
    seen = []
    orig = raw.stream_to

    def stream_to(q, duration_s, **kw):
        seen.append(kw.get("contact_monitor"))
        return orig(q, duration_s, **kw)
    raw.stream_to = stream_to
    safe.harness.heartbeat()
    q = raw.get_state().q.copy()
    q[0] += 0.05
    assert safe.move_joints(q, duration_s=1.0)
    assert len(seen) == 1 and isinstance(seen[0], ContactMonitor)
    assert seen[0].cfg.mode == "enforce"
