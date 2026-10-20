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
- otherwise (yaw 45, a tilt) the text says the base_pose rotates the box and
  quotes it in the arm's base frame FIRST, together with the TCP there (a
  workspace refusal: outside that box), then the table-frame bounds: the
  agent's FAILED line keeps only the first 240 characters of the error, and
  the base-frame numbers decide (#412 review Minor 4). A workspace refusal
  OPENS with them (\"TCP <base> outside workspace [<lo> .. <hi>] in the arm's
  base frame (...; table frame: TCP <table>, bounds [...])\"): they end within
  101 characters, so the longest prefix the code puts before the text keeps
  them too (#413 review Minor 4: the table-frame TCP and a long phrase came
  first, and a handover's 58-character prefix cut the box);
- a quarter turn written with rounded numbers (yaw 1.5708) is a quarter turn
  (#412 review Minor 3);
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
from base_pose_oracle import (  # noqa: F401  (make_runtime: the fixture, by name)
    lift,
    make_runtime,
    moved_cfg,
    raised_table,
    transform,
)

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


#: A rotation within this of turning axes onto axes (per matrix entry) is a
#: quarter turn: yaw 1.5708 is 3.7e-6 off pi/2, 4.7124 is 1.1e-5 off 3 pi/2.
QUARTER_TURN_TOL = 1e-4


def _turns_axes_onto_axes(T) -> bool:
    R = np.abs(np.asarray(T, dtype=float)[:3, :3])
    return bool(np.allclose(R, np.round(R), rtol=0.0, atol=QUARTER_TURN_TOL))


def box_text(kind, T, lo, hi, tcp=None) -> str:
    """The refusal text a base_pose arm at T owes the agent for its base-frame
    box [lo, hi] (`kind` "workspace": the refused base-frame `tcp`)."""
    lo_w, hi_w = corner_bounds(T, lo, hi)
    if kind == "workspace":
        if _turns_axes_onto_axes(T):
            return f"TCP {_q(lift(T, tcp))} outside workspace [{_q(lo_w)} .. {_q(hi_w)}]"
        return (f"TCP {_q(tcp)} outside workspace [{_q(lo)} .. {_q(hi)}] in the arm's base frame (its base_pose "
                f"rotates it; table frame: TCP {_q(lift(T, tcp))}, bounds [{_q(lo_w)} .. {_q(hi_w)}])")
    return f"TCP inside keep-out {zone_text(T, lo, hi)}"


def zone_text(T, lo, hi) -> str:
    """A keep-out zone as a refusal quotes it ("zone <lo>..<hi>"): its
    table-frame bounds, or, on a mount that does not turn axes onto axes, the
    zone in the arm's base frame first, then those bounds."""
    lo_w, hi_w = corner_bounds(T, lo, hi)
    if _turns_axes_onto_axes(T):
        return f"zone {_q(lo_w)}..{_q(hi_w)}"
    return (f"zone {_q(lo)}..{_q(hi)} in the arm's base frame (its base_pose rotates it; table-frame bounds "
            f"{_q(lo_w)}..{_q(hi_w)})")


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
    q_tcp, q_lo, q_hi, q_tcp_w, q_lo_w, q_hi_w = _quoted(text)
    assert np.all(q_lo_w <= q_hi_w) and np.all(q_lo <= q_hi), text
    assert np.allclose(q_tcp_w, tcp_w, atol=5e-4) and np.allclose(q_tcp, tcp, atol=5e-4)
    assert np.any(q_tcp < q_lo) or np.any(q_tcp > q_hi), "the base-frame TCP is outside the base-frame box"
    assert "base frame" in text and "rotates" in text


def test_a_yawed_raised_keep_out_refusal_names_the_zone_in_the_base_frame():
    h, T = _harness(YAW45_RAISED), transform(YAW45_RAISED)
    text = h._keep_out_refusal(KO_LO, KO_HI)
    assert text == box_text("keep_out", T, KO_LO, KO_HI)
    lo, hi, lo_w, hi_w = _quoted(text)
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


# ── #412 review Minor 3: a rounded quarter turn is a quarter turn ─────────


#: quarter turns as a config writes them (4 decimals), with the exact turn
_ROUNDED_QUARTER_TURNS = {
    "yaw1.5708": ([0.0, -0.22, 0.0, 0.0, 0.0, 1.5708], math.pi / 2),
    "yaw-1.5708": ([0.0, 0.22, 0.0, 0.0, 0.0, -1.5708], -math.pi / 2),
    "yaw3.1416": ([0.5, 0.0, 0.02, 0.0, 0.0, 3.1416], math.pi),
    "yaw4.7124": ([0.0, 0.22, 0.0, 0.0, 0.0, 4.7124], 3 * math.pi / 2),
}


@pytest.mark.parametrize("name", list(_ROUNDED_QUARTER_TURNS))
def test_a_quarter_turn_written_with_rounded_numbers_is_quoted_as_one(name):
    """main's exactness tolerance was 1e-9, so yaw 1.5708 (3.7e-6 off pi/2)
    counted as rotated and got the longer base-frame text. It reads as the
    exact quarter turn does now, and every refused TCP lies outside the
    quoted box."""
    from cascade.safety.harness import table_frame_bounds

    pose, exact_yaw = _ROUNDED_QUARTER_TURNS[name]
    h, T = _harness(pose), transform(pose)
    T_exact = transform([*pose[:5], exact_yaw])
    assert table_frame_bounds(T, WS_LO, WS_HI)[2] is True
    for tcp in _OUTSIDE:
        text = h._workspace_refusal(tcp)
        assert "rotates" not in text, text
        assert_text_close(text, box_text("workspace", T_exact, WS_LO, WS_HI, tcp))
        tcp_w, lo, hi = _quoted(text)
        assert np.any(tcp_w < lo - 1e-3) or np.any(tcp_w > hi + 1e-3), text
    text = h._keep_out_refusal(KO_LO, KO_HI)
    assert "rotates" not in text, text
    assert_text_close(text, box_text("keep_out", T_exact, KO_LO, KO_HI))


def test_a_turn_a_little_more_than_rounding_off_a_quarter_turn_is_rotated():
    """3e-4 rad off pi/2 is no rounding of it: the text says the base_pose
    rotates the box (the tolerance is pinned from both sides)."""
    pose = [0.0, -0.22, 0.0, 0.0, 0.0, math.pi / 2 + 3e-4]
    text = _harness(pose)._workspace_refusal(_OUTSIDE[1])
    assert "its base_pose rotates it" in text, text


# ── #412 review Minor 4: the agent's 240-character FAILED line ─────────────

#: what each site puts before the harness's text, and after it
_SITES = {
    "move_relative": ("refused by the safety harness: ", ". The arm did not move."),
    "execute": ("SafetyViolation: ", ""),
    "wrist_camera": ("refused by the wrist-camera rule: the wrist camera (it sticks out 4.5 cm above the gripper) "
                     "would enter keep-out zone ", ". The arm did not move."),
}

#: mounts that rotate the box (the r412 review's rows), and a long-number one
_ROTATED = {
    "yaw23": [0.03, -0.22, 0.0, 0.0, 0.0, 0.4],
    "yaw45": [0.1, 0.2, 0.0, 0.0, 0.0, math.pi / 4],
    "yaw45_raised": YAW45_RAISED,
    "yaw-45_raised": [0.1, 0.2, 0.05, 0.0, 0.0, -math.pi / 4],
    "yaw135_raised": [0.3, 0.1, 0.02, 0.0, 0.0, 3 * math.pi / 4],
    "roll10": [0.0, 0.0, 0.05, math.radians(10), 0.0, 0.0],
    "long_numbers": [-0.513, -0.377, 0.041, 0.0, 0.0, -2.2],
}


def _failed_quote(error: str) -> str:
    """What the agent's FAILED line quotes of `error` (cut at 240 characters:
    up to the cut marker)."""
    from cascade.skills.runtime import FAILED_INSTRUCTION, _with_failure_instruction

    line = _with_failure_instruction({"ok": False, "error": error})["next_action"]
    m = re.match(rf"{FAILED_INSTRUCTION}: (.*?) Tell the user it failed;", line)
    assert m, line
    quoted = m.group(1)
    if "(truncated; see error)" in quoted:
        quoted = quoted.split(" (truncated; see error)")[0]
        assert " ".join(error.split()).startswith(quoted), (quoted, error)
    return quoted


def _assert_keeps_the_base_frame(error: str, whole, words: str) -> None:
    """The FAILED line quotes `words` and every point in `whole` (the
    base-frame numbers that decide, computed from the box itself) whole, and
    every [x, y, z] it quotes whole is the error's own: a cut falls only in
    the table-frame bounds after them, or after every number."""
    quoted = _failed_quote(error)
    assert words in quoted, (quoted, error)
    for point in whole:
        assert str(_q(point)) in quoted, (_q(point), quoted, error)
    vectors = _VEC.findall(error)
    assert _VEC.findall(quoted) == vectors[:len(_VEC.findall(quoted))], (quoted, error)


@pytest.mark.parametrize("mount", list(_ROTATED))
def test_the_failed_line_keeps_a_rotated_mounts_base_frame_numbers(mount):
    """#412 review Minor 4: the rotated-mount texts ran to ~330 characters, and
    the agent's FAILED line (240 characters of the error) cut them inside the
    base-frame note: \"...in the arm's base frame it is [[0.1, -0.3,
    (truncated; see error)\". The base-frame box (and the TCP there) now come
    first: the FAILED line keeps them whole at every site, also with long
    numbers, and a cut falls only in the table-frame bounds after them."""
    from cascade.safety.harness import SafetyHarness, SafetyLimits, table_frame_zone_text

    rng = np.random.default_rng(412)
    T = transform(_ROTATED[mount])
    for _ in range(40):
        # 3-decimal negatives: the longest numbers a text quotes
        lo = np.round(rng.uniform(-0.6, 0.2, 3), 3)
        hi = np.round(lo + rng.uniform(0.05, 0.7, 3), 3)
        ws_lo, ws_hi = np.round(rng.uniform(-0.6, 0.0, 3), 3), np.round(rng.uniform(0.3, 0.9, 3), 3)
        h = SafetyHarness(SafetyLimits(ws_lo, ws_hi, table_z=0.0, table_clearance=0.02, keep_out=[(lo, hi)]),
                          None, base_pose=T)
        tcp = ws_lo
        while np.all(tcp >= ws_lo) and np.all(tcp <= ws_hi):
            tcp = np.round(rng.uniform(ws_lo - 0.2, ws_hi + 0.2), 3)
        ws = h._workspace_refusal(tcp)
        assert ws == box_text("workspace", T, ws_lo, ws_hi, tcp)
        for site in ("move_relative", "execute"):
            before, after = _SITES[site]
            _assert_keeps_the_base_frame(before + ws + after, [tcp, ws_lo, ws_hi], _ROTATED_WORDS)
            _assert_keeps_the_base_frame(before + h._keep_out_refusal(lo, hi) + after, [lo, hi], _ROTATED_WORDS)
        before, after = _SITES["wrist_camera"]
        _assert_keeps_the_base_frame(before + table_frame_zone_text(T, lo, hi) + after, [lo, hi], _ROTATED_WORDS)


#: what a rotated mount's texts say right after the base-frame numbers
_ROTATED_WORDS = "in the arm's base frame (its base_pose rotates it"
#: a label longer than the FAILED line keeps (`_fit_label` cuts it to its
#: least, _MIN_LABEL_ROOM characters, when the reason after it is long)
_LONG_LABEL = "the small red cube behind the blue bowl, next to the left arm"


def _handover(why: str) -> str:
    """skill_handover's error when its grasp failed for `why` (its own words)."""
    from cascade.skills.runtime import _fit_label

    return _fit_label("could not grasp {} for handover: ", repr(_LONG_LABEL), keep=len(why)) + why


#: the longest texts the code puts before a harness refusal (#413 review
#: Minor 4; the 3 `_SITES` aside): a handover whose grasp the harness refused
#: (the review's 58-character prefix, here with a long label cut to its least),
#: the same grasp run by a VLA chunk the harness refused before it moved
#: (`vla_executor`, the longest found: 137 characters), and a pick-and-place
#: carry lift (`place_geometry`)
_LONG_PREFIXES = {
    "handover": lambda text: _handover("SafetyViolation: " + text),
    "handover_vla": lambda text: _handover("SafetyViolation: harness refused VLA chunk 12 at waypoint 16 before any "
                                           "of it moved: " + text),
    "carry_lift": lambda text: "_PreCarryLiftError: pre-carry lift is unsafe: " + text,
}


@pytest.mark.parametrize("prefix", list(_LONG_PREFIXES))
@pytest.mark.parametrize("mount", list(_ROTATED))
def test_the_failed_line_keeps_the_deciding_numbers_after_the_longest_prefixes(mount, prefix):
    """#413 review Minor 4: with worst-case numbers the rotated workspace
    text ended its base-frame box at character 194, so a prefix over 46
    characters (the review's handover one: 58) cut it on the FAILED line. It
    opens with the base-frame TCP and box now: they end within 101
    characters, and the FAILED line keeps them whole after each of the
    longest prefixes in the code, with random boxes and with the longest
    numbers a text quotes (every one a 3-decimal negative), and the words
    naming their frame after every prefix but the longest."""
    from cascade.safety.harness import SafetyHarness, SafetyLimits

    rng = np.random.default_rng(413)
    T = transform(_ROTATED[mount])
    worst = (np.array([-0.789, -0.788, -0.787]), np.array([-0.123, -0.122, -0.121]), np.array([-0.999, -0.998, -0.997]))
    for i in range(41):
        if i == 0:
            ws_lo, ws_hi, tcp = worst
            lo, hi = ws_lo, ws_hi
        else:
            lo = np.round(rng.uniform(-0.6, 0.2, 3), 3)
            hi = np.round(lo + rng.uniform(0.05, 0.7, 3), 3)
            ws_lo, ws_hi = np.round(rng.uniform(-0.6, 0.0, 3), 3), np.round(rng.uniform(0.3, 0.9, 3), 3)
            tcp = ws_lo
            while np.all(tcp >= ws_lo) and np.all(tcp <= ws_hi):
                tcp = np.round(rng.uniform(ws_lo - 0.2, ws_hi + 0.2), 3)
        h = SafetyHarness(SafetyLimits(ws_lo, ws_hi, table_z=0.0, table_clearance=0.02, keep_out=[(lo, hi)]),
                          None, base_pose=T)
        ws = h._workspace_refusal(tcp)
        assert ws == box_text("workspace", T, ws_lo, ws_hi, tcp)
        deciding = ws.index(f"{_q(ws_hi)}]") + len(f"{_q(ws_hi)}]")
        assert deciding <= 101, (deciding, ws)
        # after the longest prefix (137) the words naming the frame can fall
        # past the 240th character with the longest numbers; `error` keeps them
        words = "" if prefix == "handover_vla" else _ROTATED_WORDS
        _assert_keeps_the_base_frame(_LONG_PREFIXES[prefix](ws), [tcp, ws_lo, ws_hi], words)
        _assert_keeps_the_base_frame(_LONG_PREFIXES[prefix](h._keep_out_refusal(lo, hi)), [lo, hi], words)


def test_the_longest_prefix_is_the_one_the_test_names():
    """Before a reason too long for the FAILED line the handover-through-VLA
    prefix is 137 characters (the label cut to its least): with the 101 the
    deciding numbers take, 238 of the FAILED line's 240."""
    long_reason = "TCP " + "x" * 236
    assert _LONG_PREFIXES["handover_vla"](long_reason).index(long_reason) == 137
    assert _LONG_PREFIXES["handover"](long_reason).index(long_reason) == 69


_E2E = {k: _ROTATED[k] for k in ("yaw23", "yaw45_raised", "yaw-45_raised", "yaw135_raised")}


@pytest.mark.parametrize("mount", list(_E2E))
def test_a_rotated_mounts_move_relative_refusal_keeps_its_numbers_on_the_failed_line(make_runtime, mount):  # noqa: F811
    """The r412 review's end-to-end rows: move_relative 10 cm forward on the
    mock at these mounts is refused by the workspace. main's FAILED line cut
    the base-frame note (\"...in the arm's base frame it is [[0.1, -0.3,
    (truncated; see error)\"); it keeps the TCP in both frames and the
    base-frame box whole now, and `error` keeps every number."""
    pose = _E2E[mount]
    cfg = moved_cfg("mock", pose)
    if pose[2] != 0:
        raised_table(cfg)
    cfg._data["relative_motion"]["wrist_camera"] = None
    rt = make_runtime(cfg)

    out = rt.execute("move_relative", {"direction": "forward", "frame": "base", "distance_m": 0.10})

    assert out["ok"] is False and out.get("refused_by") == "safety_harness", out
    error = " ".join(out["error"].split())
    assert error.startswith("refused by the safety harness: TCP ") and "outside workspace" in error, error
    assert len(_VEC.findall(error)) == 6, error
    quoted = _failed_quote(error)
    assert out["next_action"].startswith(f"This attempt FAILED: {quoted}"), out["next_action"]
    limits = rt.arm.harness.limits
    tcp_base = np.array(_VEC.findall(error)[0], dtype=float)
    _assert_keeps_the_base_frame(error, [tcp_base, limits.workspace_min, limits.workspace_max], _ROTATED_WORDS)
