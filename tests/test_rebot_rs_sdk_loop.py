"""cascade's reBot RS drivers stream MIT commands directly; they never run the
SDK's RebotArmEndPose 500 Hz control loop.

Rig finding (Seeed WRC fork, docs/HARDWARE_VERIFICATION_HANDOVER.md, Finding
#2): `RebotArmEndPose._loop_cb` re-sends its own `_q_target` to the arm group
500 times a second, so any direct `send_mit` from outside is overwritten
within 2 ms -- WRC's attempt to drive home with `send_mit` "did nothing".
That only matters to a driver that STARTS that loop. cascade's `RebotRSArm`
(and the motorbridge sibling) own the stream themselves: one MIT frame per
harness-approved 50 Hz waypoint, nothing else writing. These tests pin that
so a future "use the SDK controller" refactor cannot silently reintroduce a
second writer that fights the harness-vetted stream.

Offline: the SDK is faked (see test_rebot_clear_error); no bus is opened.
"""

from __future__ import annotations

import ast
import inspect

import numpy as np

from test_rebot_clear_error import _Log, _install_fake_sdk, _rs_arm


def _calls_and_imports(module):
    tree = ast.parse(inspect.getsource(module))
    attrs = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    imported = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom):
            imported.add(n.module or "")
            imported.update(a.name for a in n.names)
        elif isinstance(n, ast.Import):
            imported.update(a.name for a in n.names)
    return attrs | names, imported


def test_rs_drivers_never_reference_the_sdk_controller_loop():
    from cascade.control import rebot_rs_arm, rebot_rs_mb_arm

    for module in (rebot_rs_arm, rebot_rs_mb_arm):
        used, imported = _calls_and_imports(module)
        for forbidden in ("RebotArmEndPose", "start_control_loop", "_q_target"):
            assert forbidden not in used, f"{module.__name__} uses {forbidden}"
        assert not any("controllers" in m for m in imported), (
            f"{module.__name__} imports the SDK controllers package")


def test_each_waypoint_is_exactly_one_direct_mit_frame(monkeypatch):
    """The fake SDK's start_control_loop raises; connect + streaming still
    work, and every send_joint_target is one arm-group send_mit -- the only
    writer on the bus is the harness-approved stream."""
    from cascade.control import robstride

    monkeypatch.setattr(robstride.time, "sleep", lambda s: None)
    log = _Log()
    _install_fake_sdk(monkeypatch, log)
    arm = _rs_arm()
    arm.connect()
    before = sum(1 for ev in log if ev == ("group_send_mit", "arm"))
    for q in (np.zeros(6), np.full(6, 0.01), np.full(6, 0.02)):
        arm.send_joint_target(q)
    after = sum(1 for ev in log if ev == ("group_send_mit", "arm"))
    assert after - before == 3
