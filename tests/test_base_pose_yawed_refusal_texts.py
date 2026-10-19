"""Honest refusal texts on rotated base_pose mounts (#407 review follow-up 3).

The harness checks the workspace box and keep-out zones in the arm's BASE
frame, and a base_pose arm's refusal texts quote table-frame numbers (spec
Q1 a). Main lifted only the two corners of a box: on a yawed mount that is no
box at all (yaw 90 / 180: lo > hi on an axis) and on yaw 45 plus a raise the
refused TCP lay INSIDE the quoted range. Now:

- every text quotes the table-frame axis-aligned bounds of all 8 lifted
  corners (lo <= hi on every axis);
- when the mount turns the box by a quarter turn (or not at all) those bounds
  ARE the box: the text says no more, and a refused TCP lies outside them;
- otherwise (yaw 45, a tilt) the text adds why: the bounds are of a box this
  arm's base_pose rotates, quoted in the arm's base frame together with the
  TCP there (a workspace refusal: outside that box);
- an arm without base_pose reads exactly as before, and a mount that only
  shifts / raises quotes the same numbers as main (its corners are the bounds).

Sites: SafetyHarness._workspace_refusal / _keep_out_refusal and the
spatial-move wrist-camera keep-out refusal (spatial_moves._camera_volume).
"""

from __future__ import annotations

import itertools
import math
import re

import numpy as np
import pytest
from base_pose_oracle import lift, transform

WS_LO, WS_HI = np.array([0.10, -0.30, 0.00]), np.array([0.40, 0.30, 0.40])
KO_LO, KO_HI = np.array([0.20, -0.05, 0.00]), np.array([0.30, 0.05, 0.20])

YAW90 = [0.0, -0.22, 0.0, 0.0, 0.0, math.pi / 2]
YAW180 = [0.5, 0.0, 0.0, 0.0, 0.0, math.pi]
YAW45_RAISED = [0.1, 0.2, 0.03, 0.0, 0.0, math.pi / 4]
SHIFTED_RAISED = [0.0, -0.22, 0.03, 0.0, 0.0, 0.0]
QUARTER_TURNS = [YAW90, YAW180]

_VEC = re.compile(r"\[(-?\d+(?:\.\d+)?), (-?\d+(?:\.\d+)?), (-?\d+(?:\.\d+)?)\]")


def _q(p) -> list:
    return np.round(np.asarray(p, dtype=float), 3).tolist()


def corner_bounds(T, lo, hi):
    """Independent oracle: the table-frame AABB of the box's 8 lifted corners."""
    corners = np.array(list(itertools.product(*zip(np.asarray(lo, float), np.asarray(hi, float)))))
    lifted = lift(T, corners)
    return lifted.min(axis=0), lifted.max(axis=0)


def _turns_axes_onto_axes(T) -> bool:
    R = np.abs(np.asarray(T, dtype=float)[:3, :3])
    return bool(np.allclose(R, np.round(R), rtol=0.0, atol=1e-9))


def box_text(kind, T, lo, hi, tcp=None) -> str:
    """The refusal text a base_pose arm at T owes the agent for its base-frame
    box [lo, hi] (`kind` "workspace": the refused base-frame `tcp`)."""
    lo_w, hi_w = corner_bounds(T, lo, hi)
    if kind == "workspace":
        text = f"TCP {_q(lift(T, tcp))} outside workspace [{_q(lo_w)} .. {_q(hi_w)}]"
        if not _turns_axes_onto_axes(T):
            text += (" (table-frame bounds of this arm's workspace, which its base_pose rotates: in the arm's "
                     f"base frame it is [{_q(lo)} .. {_q(hi)}] and the TCP is at {_q(tcp)})")
        return text
    text = f"TCP inside keep-out zone {_q(lo_w)}..{_q(hi_w)}"
    if not _turns_axes_onto_axes(T):
        text += (" (table-frame bounds of this arm's keep-out zone, which its base_pose rotates: in the arm's "
                 f"base frame it is {_q(lo)}..{_q(hi)})")
    return text


def zone_text(T, lo, hi) -> str:
    """The keep-out zone as a wrist-camera refusal quotes it ("zone <lo>..<hi>"
    plus the base-frame note on a mount that does not turn axes onto axes)."""
    return box_text("keep_out", T, lo, hi).removeprefix("TCP inside keep-out ")


def assert_text_close(actual: str, expected: str, atol: float = 1.1e-3) -> None:
    """Equal words, and every quoted [x, y, z] within `atol` (texts rebuilt
    from another rig's rounded numbers differ in the last digit)."""
    assert _VEC.sub("[v]", actual) == _VEC.sub("[v]", expected), (actual, expected)
    got = np.array(_VEC.findall(actual), dtype=float)
    want = np.array(_VEC.findall(expected), dtype=float)
    assert got.shape == want.shape and np.allclose(got, want, rtol=0.0, atol=atol), (actual, expected)


def _harness(pose):
    from cascade.safety.harness import SafetyHarness, SafetyLimits

    limits = SafetyLimits(WS_LO.copy(), WS_HI.copy(), table_z=0.0, table_clearance=0.02,
                          keep_out=[(KO_LO.copy(), KO_HI.copy())])
    return SafetyHarness(limits, None, base_pose=None if pose is None else transform(pose))


def _quoted(text):
    return [np.array(v, dtype=float) for v in _VEC.findall(text)]


#: base-frame TCPs outside the workspace box, one beyond each face
_OUTSIDE = [np.array(p) for p in ([0.05, 0.0, 0.2], [0.45, 0.0, 0.2], [0.25, -0.35, 0.2], [0.25, 0.35, 0.2],
                                  [0.25, 0.0, -0.05], [0.25, 0.0, 0.45], [0.45, 0.25, 0.1])]


@pytest.mark.parametrize("pose", QUARTER_TURNS, ids=["yaw90", "yaw180"])
def test_a_quarter_turned_workspace_is_quoted_as_the_exact_table_box(pose):
    """Yaw 90 / 180: the quoted bounds are a box (lo <= hi), they ARE the
    workspace in the table frame, and every refused TCP lies outside them.
    Main quoted two lifted corners: lo > hi on x (and y)."""
    h, T = _harness(pose), transform(pose)
    for tcp in _OUTSIDE:
        text = h._workspace_refusal(tcp)
        assert text == box_text("workspace", T, WS_LO, WS_HI, tcp)
        tcp_w, lo, hi = _quoted(text)
        assert np.all(lo <= hi), text
        assert np.any(tcp_w < lo - 1e-3) or np.any(tcp_w > hi + 1e-3), text


@pytest.mark.parametrize("pose", QUARTER_TURNS, ids=["yaw90", "yaw180"])
def test_a_quarter_turned_keep_out_zone_is_quoted_as_the_exact_table_box(pose):
    h, T = _harness(pose), transform(pose)
    text = h._keep_out_refusal(KO_LO, KO_HI)
    assert text == box_text("keep_out", T, KO_LO, KO_HI)
    lo, hi = _quoted(text)
    assert np.all(lo <= hi), text
    inside = lift(T, (KO_LO + KO_HI) / 2)
    assert np.all(inside >= lo) and np.all(inside <= hi)


def test_a_yawed_raised_workspace_refusal_says_why_the_tcp_is_inside_its_bounds():
    """Yaw 45 + a raise: the table-frame bounds are larger than the rotated
    box, and this refused TCP lies inside them (main quoted it inside the
    corners' range with no word why). The text says the bounds are of a box
    the base_pose rotates, and quotes that box and the TCP in the arm's base
    frame, where the TCP is outside it."""
    h, T = _harness(YAW45_RAISED), transform(YAW45_RAISED)
    tcp = np.array([0.45, 0.25, 0.1])
    lo_w, hi_w = corner_bounds(T, WS_LO, WS_HI)
    tcp_w = lift(T, tcp)
    assert np.all(tcp_w >= lo_w) and np.all(tcp_w <= hi_w), "the case: the TCP is inside the table-frame bounds"
    text = h._workspace_refusal(tcp)
    assert text == box_text("workspace", T, WS_LO, WS_HI, tcp)
    q_tcp_w, q_lo_w, q_hi_w, q_lo, q_hi, q_tcp = _quoted(text)
    assert np.all(q_lo_w <= q_hi_w) and np.all(q_lo <= q_hi), text
    assert np.allclose(q_tcp_w, tcp_w, atol=5e-4) and np.allclose(q_tcp, tcp, atol=5e-4)
    assert np.any(q_tcp < q_lo) or np.any(q_tcp > q_hi), "the base-frame TCP is outside the base-frame box"
    assert "base frame" in text and "rotates" in text


def test_a_yawed_raised_keep_out_refusal_names_the_zone_in_the_base_frame():
    h, T = _harness(YAW45_RAISED), transform(YAW45_RAISED)
    text = h._keep_out_refusal(KO_LO, KO_HI)
    assert text == box_text("keep_out", T, KO_LO, KO_HI)
    lo_w, hi_w, lo, hi = _quoted(text)
    assert np.all(lo_w <= hi_w)
    np.testing.assert_allclose(lo, KO_LO)
    np.testing.assert_allclose(hi, KO_HI)


def test_a_shifted_raised_mount_quotes_what_main_quoted():
    """No rotation: the lifted corners are the bounds; the texts are main's."""
    h, T = _harness(SHIFTED_RAISED), transform(SHIFTED_RAISED)
    tcp = _OUTSIDE[1]
    assert h._workspace_refusal(tcp) == (
        f"TCP {_q(lift(T, tcp))} outside workspace [{_q(lift(T, WS_LO))} .. {_q(lift(T, WS_HI))}]")
    assert h._keep_out_refusal(KO_LO, KO_HI) == f"TCP inside keep-out zone {_q(lift(T, KO_LO))}..{_q(lift(T, KO_HI))}"


def test_without_a_base_pose_the_texts_are_unchanged():
    h = _harness(None)
    tcp = np.array([0.4512345, 0.0, 0.2])
    assert h._workspace_refusal(tcp) == (f"TCP {np.round(tcp, 3).tolist()} outside workspace "
                                         f"[{WS_LO.tolist()} .. {WS_HI.tolist()}]")
    assert h._keep_out_refusal(KO_LO, KO_HI) == f"TCP inside keep-out zone {KO_LO.tolist()}..{KO_HI.tolist()}"
