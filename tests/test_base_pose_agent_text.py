"""base_pose agent contract text (spec §3.5, D7; plan Task 7; #352 Minors 1 and 3).

On a rig where any arm declares a base_pose, every coordinate the agent reads is
in the shared TABLE frame (spec Q1 a): the coordinate tools' descriptions, the
robot description, the plausibility prompt, annotated_view's key, the viewers'
TCP and workspace overlay, and the safety harness's refusal texts (#400
re-review (4)). Rigs without a base_pose keep every string byte-identical.
"""

from __future__ import annotations

import hashlib
import json
import re

import numpy as np
import pytest
from base_pose_oracle import (
    RAISED,
    SHIPPED_RIGHT,
    YAWED,
    assert_equivariant,
    lift,
    make_pair_rigs,  # noqa: F401  (the fixture, by name)
    make_runtime,  # noqa: F401  (the fixture, by name)
    moved_cfg,
    raised_table,
    run_pair,
    transform,
)
from test_base_pose_yawed_refusal_texts import YAW90, assert_text_close, box_text, zone_text

# ── 1. the safety harness's refusal texts (#400 re-review (4)) ─────────────


def _harness(make_runtime, pose, *, keep_out=None):  # noqa: F811
    """so101_right's harness at `pose` (None: no base_pose), on a base 3 cm
    up's table plane (RAISED's `table_z` on both rigs, so the same joint pose
    is refused for the same reason), with an optional keep-out box (its base
    frame)."""
    cfg = raised_table(moved_cfg("so101_right", pose))
    if keep_out is not None:
        for section in _safety_sections(cfg):
            section["keep_out"] = [{"min": list(keep_out[0]), "max": list(keep_out[1])}]
    rt = make_runtime(cfg)
    return rt.arm.harness, np.asarray(rt.arm.get_state().q, dtype=float)


def _safety_sections(cfg):
    out = [cfg._data["safety"]]
    for entry in [cfg._data.get("arm"), *(cfg._data.get("arms") or [])]:
        entry = getattr(entry, "_data", entry)
        resolved = None if entry is None else entry.get("resolved")
        resolved = getattr(resolved, "_data", resolved)
        if resolved is not None and resolved.get("safety") is not None:
            out.append(getattr(resolved["safety"], "_data", resolved["safety"]))
    return out


#: The first words of each refusal the conversion covers (approve() and
#: vet_pose() alike).
_KINDS = {"workspace": "outside workspace", "table": "below table clearance", "link": "would hit the table",
          "keep_out": "inside keep-out zone"}


def _pose_refused_by(harness, kind, rng, home):
    """A joint pose (seeded search) that `vet_pose` refuses for `kind`."""
    lo, hi = harness.kin.joint_limits
    for _ in range(20000):
        q = rng.uniform(lo, hi)
        reason = harness.vet_pose(q) or ""
        if _KINDS[kind] in reason and harness.vet_step(home, q, 30.0):
            return q
    raise AssertionError(f"no {kind} pose found")


def _texts_equivariant(a, b, T):
    a, b = _honest_boxes(a, b, T)
    assert_equivariant({"cmds_a": [], "cmds_b": [], "jaws_a": [], "jaws_b": [], "beliefs_a": [], "beliefs_b": [],
                        "result_a": {"error": a}, "result_b": {"error": b}}, T)


#: A box a refusal quotes: the workspace with the TCP it refused, or a
#: keep-out zone; on a mount that rotates it, in the arm's base frame first
#: (#412 review Minor 4), the table frame's numbers in the note after it
#: (#413 review Minor 4).
_BOX_QUOTE = re.compile(
    r"TCP (\[[^\]]*\]) outside workspace \[(\[[^\]]*\]) \.\. (\[[^\]]*\])\]"
    r"( in the arm's base frame \(its base_pose rotates it; [^)]*\))?"
    r"|keep-out (zone (\[[^\]]*\])\.\.(\[[^\]]*\])"
    r"( in the arm's base frame \(its base_pose rotates it; [^)]*\))?)")


def _honest_boxes(a: str, b: str, T):
    """Rig B's quoted boxes are the texts rig A's base-frame numbers owe on a
    base_pose arm (#407 review follow-up 3: the table-frame bounds of all 8
    corners, after the base frame's box when the mount rotates it), not two
    lifted corners; returns both texts with those quotes blanked, for the
    equivariance oracle to compare the rest. A shifted / raised mount's bounds
    are its lifted corners: the same numbers as T a."""
    if not isinstance(a, str) or not isinstance(b, str):
        return a, b
    quotes_a, quotes_b = list(_BOX_QUOTE.finditer(a)), list(_BOX_QUOTE.finditer(b))
    assert len(quotes_a) == len(quotes_b), (a, b)
    for qa, qb in zip(quotes_a, quotes_b, strict=True):
        if qa.group(2) is not None:
            tcp, lo, hi = (json.loads(qa.group(i)) for i in (1, 2, 3))
            assert qa.group(4) is None, a        # rig A has no base_pose
            assert_text_close(qb.group(0), box_text("workspace", T, lo, hi, tcp))
        else:
            lo, hi = json.loads(qa.group(6)), json.loads(qa.group(7))
            assert qa.group(8) is None, a
            assert_text_close(qb.group(5), zone_text(T, lo, hi))
    return _BOX_QUOTE.sub("<box>", a), _BOX_QUOTE.sub("<box>", b)


#: execute()'s FAILED line (`runtime._with_failure_instruction`): it quotes
#: the error up to 240 characters, so a longer table-frame quote is cut there.
_FAILED_LINE = re.compile(
    r"This attempt FAILED: .*? Tell the user it failed; never report a failed or refuted result as success\.")


def _honest_run(run, T):
    """`_honest_boxes` over every prose field of a run's two results. A FAILED
    line that cut rig B's longer error (" (truncated; see error)") must quote
    a prefix of that error; it is blanked on both rigs."""
    error_b = " ".join(str(run["result_b"].get("error")).split())
    for key in set(run["result_a"]) | set(run["result_b"]):
        a, b = run["result_a"].get(key), run["result_b"].get(key)
        if isinstance(a, str) and isinstance(b, str):
            fa, fb = _FAILED_LINE.search(a), _FAILED_LINE.search(b)
            if fa and fb and "(truncated; see error)" in fb.group(0):
                quoted = fb.group(0).removeprefix("This attempt FAILED: ").split(" (truncated; see error)")[0]
                assert error_b.startswith(quoted), (quoted, error_b)
                a, b = a.replace(fa.group(0), "<failed>"), b.replace(fb.group(0), "<failed>")
            run["result_a"][key], run["result_b"][key] = _honest_boxes(a, b, T)
    return run


@pytest.mark.parametrize("pose", [RAISED, YAWED, YAW90], ids=["raised", "yawed", "yaw90"])
@pytest.mark.parametrize("kind", sorted(_KINDS))
def test_harness_refusals_quote_table_frame_numbers_on_a_base_pose_arm(make_runtime, pose, kind):  # noqa: F811
    """The same joint pose on so101_right without and with a base_pose: both
    harnesses refuse it for the same reason (the geometry is the arm's own),
    and the base_pose arm's text quotes the TCP, the workspace / keep-out
    boxes and the heights in the table frame: rig A's numbers mapped by T
    (points by T, heights by T's z; a box by the bounds of its 8 corners, with
    its base-frame numbers when the mount rotates it, `_honest_boxes`). On
    main they quoted base-frame numbers."""
    box = ([0.05, -0.30, -0.05], [0.30, 0.30, 0.10]) if kind == "keep_out" else None
    h_a, home = _harness(make_runtime, None, keep_out=box)
    h_b, _ = _harness(make_runtime, pose, keep_out=box)
    T = transform(pose)
    q = _pose_refused_by(h_a, kind, np.random.default_rng(7), home)
    pose_a, pose_b = h_a.vet_pose(q), h_b.vet_pose(q)
    step_a, step_b = h_a.vet_step(home, q, 30.0), h_b.vet_step(home, q, 30.0)
    assert _KINDS[kind] in pose_a and _KINDS[kind] in pose_b, (pose_a, pose_b)
    _texts_equivariant(pose_a, pose_b, T)
    _texts_equivariant(step_a, step_b, T)


def test_harness_refusals_without_a_base_pose_read_as_before(make_runtime):  # noqa: F811
    """No base_pose: the four texts in today's exact words and numbers (an
    unrounded keep-out box, so any rounding of its corners shows)."""
    box = ([0.0512345, -0.3037, -0.0499], [0.3011, 0.3023, 0.1004])
    h, home = _harness(make_runtime, None, keep_out=box)
    rng = np.random.default_rng(7)
    lo_w, hi_w = h.limits.workspace_min, h.limits.workspace_max
    floor = h.limits.table_z + h.limits.table_clearance
    for kind in _KINDS:
        q = _pose_refused_by(h, kind, rng, home)
        tcp = h.kin.fk(q)[:3, 3]
        reason = h.vet_pose(q)
        if kind == "workspace":
            assert reason == f"TCP {np.round(tcp, 3).tolist()} outside workspace [{lo_w.tolist()} .. {hi_w.tolist()}]"
        elif kind == "table":
            assert reason == f"TCP z={tcp[2]:.3f} below table clearance {floor:.3f}"
        elif kind == "keep_out":
            kmin, kmax = h.limits.keep_out[0]
            assert reason == f"TCP inside keep-out zone {kmin.tolist()}..{kmax.tolist()}"
        else:
            i = int(reason.split()[1])
            z = h.kin.link_positions(q)[i - 1][2]
            assert reason == f"link/joint {i} at z={z:.3f} would hit the table"


#: End to end through the skills: a spatial move whose route the harness
#: refuses, quoted in the result's `error` (oracle rows, plan Task 8's raised
#: rows): `mock`'s forward step leaves its workspace; so101_right's 10 cm
#: descent on RAISED goes under the clearance plane.
_HARNESS_ROWS = [("mock", RAISED, ("move_relative", {"direction": "forward", "frame": "base", "distance_m": 0.10})),
                 ("mock", YAWED, ("move_relative", {"direction": "forward", "frame": "base", "distance_m": 0.10})),
                 ("so101_right", RAISED, ("move_relative", {"direction": "down", "frame": "base",
                                                            "distance_m": 0.10}))]


def _no_wrist_box(raise_table):
    def edit(cfg):
        if raise_table:
            raised_table(cfg)
        cfg._data["relative_motion"]["wrist_camera"] = None
    return edit


@pytest.mark.parametrize("arm,pose,call", _HARNESS_ROWS,
                         ids=["mock raised workspace", "mock yawed workspace", "so101 raised table"])
def test_a_harness_refusal_reaches_the_agent_in_the_table_frame(make_pair_rigs, arm, pose, call):  # noqa: F811
    rt_a, rt_b, T = make_pair_rigs(pose, arm=arm, edit=_no_wrist_box(pose is RAISED))
    skill, args = call
    run = run_pair(rt_a, rt_b, T, skill, args)
    assert "refused by the safety harness" in str(run["result_a"].get("error")), run["result_a"]
    assert not run["cmds_a"] and not run["cmds_b"]
    assert_equivariant(_honest_run(run, T), T)


# ── 2. tool specs, robot description, plausibility prompt (spec §3.5, R11) ──

#: sha256 of json.dumps(..., sort_keys=True), computed on origin/main d68c7165
#: (where #352's list_arms sentence is already gone, Task 6): the four rigs
#: without a base_pose keep them byte for byte.
GOLDEN = {
    "isaac_kitchen": {"describe_robot": "79e6e6b9f232baa5c492a5cdcbfe26287e352eb80bcd94ce97bfd0e8dcb4f5af",
                      "tool_specs": "398cf5bf71c1a6527920afe95867c70e67fdf6e6213d48eae73f7181421062e1"},
    "mock": {"describe_robot": "3cbda18ab0dc9ebed14434abb1918eab86e470abcbc01896af3ad9a57fd32f40",
             "tool_specs": "da9e717dca41a710e0e819e8ef0fc442e12f762ccf38e29d0a65a673ef8ae3c6"},
    "rebot_rs": {"describe_robot": "a0f69ae5ba1668c13ec672172c2f301d3e33c0da6fc8841f03cb63d81452b5d0",
                 "tool_specs": "37c1eebc74b5c2696e81b8b6336ccbebfa6cca01d1bade00f57e69257738348e"},
    "so101_mock": {"describe_robot": "9876d4817023c1a2f89422fd57fe5f28970bc927ee3e763e832d68d0d6bcbefa",
                   "tool_specs": "ad84f7f33a45368039e847aeb87634aa2d0fb93d44bf65dda569aecee4ebe3b8"},
}
GOLDEN_PLAUSIBILITY_USER = "d087582e3ac479db1edc8188374ba162456c2c0d0d20d983551a5109a1bf8cb8"
_RIGS = {"rebot_rs": dict(arm="rebot_rs", camera="mock"), "mock": dict(arm="mock", camera="mock"),
         "so101_mock": dict(arm="so101_mock", camera="mock_small"),
         "isaac_kitchen": dict(arm="isaac_kitchen", camera="mock")}
#: Base-frame position wording; move_relative keeps "the robot base frame"
#: for its DIRECTIONS (spec D8: directions stay the arm's own).
_BASE_WORDS = ("base-frame", "the robot base frame", "the robot's base frame")
_TABLE_SENTENCE = " Coordinates are in the shared table frame: the same numbers for every arm on this rig."


def _digest(x) -> str:
    return hashlib.sha256(json.dumps(x, sort_keys=True).encode()).hexdigest()


def _rig_cfg(rig):
    from cascade.config import load_demo_config

    if rig == "so101_right alone":
        return load_demo_config(arm="so101_right", camera="mock_small", llm="mock", _ignore_robot_environment=True)
    if rig == "pair":
        return load_demo_config(arms=["so101_left", "so101_right"], camera="mock_small", llm="mock",
                                _ignore_robot_environment=True)
    return load_demo_config(llm="mock", _ignore_robot_environment=True, **_RIGS[rig])


def _without_base_poses(cfg):
    for entry in [cfg._data.get("arm"), *(cfg._data.get("arms") or [])]:
        entry = getattr(entry, "_data", entry)
        if isinstance(entry, dict):
            entry.pop("base_pose", None)
            resolved = getattr(entry.get("resolved"), "_data", entry.get("resolved"))
            arm = None if resolved is None else getattr(resolved.get("arm"), "_data", resolved.get("arm"))
            if isinstance(arm, dict):
                arm.pop("base_pose", None)
    return cfg


@pytest.mark.parametrize("rig", sorted(_RIGS))
def test_tool_specs_and_robot_text_are_unchanged_on_rigs_without_a_base_pose(rig):
    from cascade.robot_context import declares_base_pose, describe_robot
    from cascade.skills.runtime import tool_specs_for_config

    cfg = _rig_cfg(rig)
    assert not declares_base_pose(cfg)
    assert _digest(tool_specs_for_config(cfg)) == GOLDEN[rig]["tool_specs"]
    assert _digest(describe_robot(cfg)) == GOLDEN[rig]["describe_robot"]


def test_the_plausibility_prompt_is_unchanged_without_a_base_pose():
    from cascade.agent.prompts import PLAUSIBILITY_USER

    assert _digest(PLAUSIBILITY_USER) == GOLDEN_PLAUSIBILITY_USER


@pytest.mark.parametrize("rig", ["so101_right alone", "pair"])
def test_coordinate_tools_say_table_frame_on_a_base_pose_rig(rig):
    """#352 Minor 1: so101_right ALONE (list_arms is withheld on one arm, so
    text placed there is never seen) and the pair: every coordinate tool says
    table frame, once, and no other tool changes."""
    from cascade.skills.runtime import TABLE_FRAME_TOOLS, tool_specs_for_config

    assert TABLE_FRAME_TOOLS == {"localize_object", "place_at", "annotated_view", "probe_point"}
    specs = {s["name"]: s for s in tool_specs_for_config(_rig_cfg(rig))}
    plain = {s["name"]: s for s in tool_specs_for_config(_without_base_poses(_rig_cfg(rig)))}
    for name in TABLE_FRAME_TOOLS:
        text = specs[name]["description"]
        assert "table frame" in text or "table-frame" in text, (name, text)
        assert not any(w in text for w in _BASE_WORDS), (name, text)
        assert text.endswith(_TABLE_SENTENCE) and text.count(_TABLE_SENTENCE) == 1, (name, text)
        assert specs[name]["parameters"] == plain[name]["parameters"]
    for name, spec in specs.items():
        if name not in TABLE_FRAME_TOOLS:
            assert spec == plain[name], name
        if name != "move_relative":
            assert not any(w in spec["description"] for w in _BASE_WORDS), (name, spec["description"])


def test_every_tool_that_states_a_position_frame_is_a_coordinate_tool():
    """A new tool whose description names the base frame for a position must
    join TABLE_FRAME_TOOLS (or be a direction, D8)."""
    from cascade.skills.runtime import TABLE_FRAME_TOOLS, TOOL_SPECS

    named = {s["name"] for s in TOOL_SPECS if any(w in json.dumps(s) for w in _BASE_WORDS)}
    assert named == TABLE_FRAME_TOOLS


@pytest.mark.parametrize("rig", ["so101_right alone", "pair"])
def test_the_robot_description_says_table_frame_on_a_base_pose_rig(rig):
    """R11: describe_robot reaches the system prompt, the advisor and the MCP
    domains; on a base_pose rig its coordinates are table-frame and each arm's
    base_pose is quoted."""
    from cascade.robot_context import declares_base_pose, describe_robot

    cfg = _rig_cfg(rig)
    text = describe_robot(cfg)
    assert declares_base_pose(cfg)
    assert "use the robot base frame" not in text
    poses = {"so101_left": [0.0, 0.22, 0.0, 0.0, 0.0, 0.0], "so101_right": [0.0, -0.22, 0.0, 0.0, 0.0, 0.0]}
    names = ["so101_right"] if rig == "so101_right alone" else ["so101_left", "so101_right"]
    assert text.count("use the shared table frame, in metres: the same numbers for every arm on this rig") == len(names)
    for name in names:
        assert f"stands at base_pose {poses[name]}" in text, text


def test_an_identity_base_pose_is_no_base_pose_for_the_agent_text():
    """Spec §4: [0]*6 counts as none, so so101_left alone at the identity keeps
    the plain rig's strings; a pose that moves the base declares one."""
    from cascade.robot_context import declares_base_pose, describe_robot
    from cascade.skills.runtime import tool_specs_for_config

    identity = moved_cfg("so101_left", [0.0] * 6)
    plain = moved_cfg("so101_left", None)
    assert not declares_base_pose(identity)
    assert tool_specs_for_config(identity) == tool_specs_for_config(plain)
    assert describe_robot(identity) == describe_robot(plain)
    assert declares_base_pose(moved_cfg("so101_left", [0.0, 0.0, 0.0, 0.0, 0.0, 1e-9]))


@pytest.mark.parametrize("rig", ["mock", "pair"])
def test_the_plausibility_prompt_names_the_table_frame_on_a_base_pose_rig(rig):
    """R11: the critic's world model lists table-frame beliefs; on a base_pose
    rig its heading says so and it quotes no single arm's base-frame box."""
    from cascade.agent.llm import MockLLM
    from cascade.apps.demo import _premotion_critic

    class _RT:
        beliefs = None
        held_object = None

    critic = _premotion_critic(_rig_cfg(rig), MockLLM([]), _RT(), is_mock=False)
    prompt = critic.prompt("move_home", {})
    if rig == "pair":
        assert "World model (shared table frame, meters;" in prompt and "robot base frame" not in prompt
        assert "reachable TCP box" not in prompt
    else:
        assert "World model (robot base frame, meters;" in prompt and "reachable TCP box" in prompt


# ── 3. viewers: the TCP and the workspace where they are (spec §3.4, R14) ──


def _pair_rig():
    return moved_cfg(["so101_left", "so101_right"], {}, move_cameras=False, shift_xy=(0.0, -0.10))


def _lifted_tcp(rt, name):
    """Arm `name`'s TCP in the table frame: its own Kinematics, its base_pose."""
    from base_pose_oracle import raw_arm

    safe = rt.arm_rig.get(name)
    harness = safe.__dict__["harness"]
    q = np.asarray(raw_arm(safe).get_state().q, dtype=float)
    return lift(harness.base_pose, harness.kin.fk(q)[:3, 3])


def _tcp_spy(monkeypatch):
    from cascade.perception.visual_interface import VisualInterface

    drawn, real = [], VisualInterface._draw_tcp
    monkeypatch.setattr(VisualInterface, "_draw_tcp",
                        lambda self, img, tcp, T, K: (drawn.append(np.asarray(tcp, dtype=float)),
                                                      real(self, img, tcp, T, K))[1])
    return drawn


@pytest.mark.parametrize("arm", [None, "so101_right"], ids=["primary", "so101_right"])
def test_annotated_view_draws_the_running_arms_tcp_where_it_is(make_runtime, monkeypatch, arm):  # noqa: F811
    """R14: annotated_view on a skill thread draws the BOUND arm's TCP lifted
    by that arm's own base_pose (main: the base-frame TCP drawn with the
    table-frame camera, 22 cm off on so101_right; the primary's T would put it
    44 cm off)."""
    rt = make_runtime(_pair_rig())
    drawn = _tcp_spy(monkeypatch)
    result = rt.execute("annotated_view", {} if arm is None else {"arm": arm})
    assert result.get("ok", True) is not False, result
    want = _lifted_tcp(rt, arm or "so101_left")
    assert len(drawn) == 1 and np.allclose(drawn[0], want, rtol=0.0, atol=1e-9), (drawn, want)


def test_viewers_read_the_primary_frame_during_a_bound_call(make_runtime, monkeypatch):  # noqa: F811
    """Review Focus 4: while execute(..., arm="so101_right") runs, another
    thread (the stream server, the MCP reader) sees the PRIMARY's TCP and
    base_pose: its overlay is the left arm's, lifted by the left's pose."""
    import threading

    from cascade.perception.visual_interface import arm_overlay

    rt = make_runtime(_pair_rig())
    entered, release, seen = threading.Event(), threading.Event(), {}

    def bound():
        entered.set()
        assert release.wait(10)
        return {"ok": True}

    monkeypatch.setattr(rt, "skill_list_objects", bound)
    worker = threading.Thread(target=lambda: rt.execute("list_objects", {"arm": "so101_right"}))
    worker.start()
    try:
        assert entered.wait(10)
        seen["tcp"] = rt._tcp_world_if_awake()
        _, seen["overlay_tcp"], seen["T"] = arm_overlay(rt, rt.extrinsics)
    finally:
        release.set()
        worker.join(10)
    left = rt.arm_rig.get("so101_left").__dict__["harness"].base_pose
    want = _lifted_tcp(rt, "so101_left")
    assert np.allclose(seen["tcp"], want, rtol=0.0, atol=1e-9), (seen["tcp"], want)
    assert np.allclose(seen["overlay_tcp"], want, rtol=0.0, atol=1e-9)
    assert np.array_equal(seen["T"], left)


def _top_camera_frame():
    from types import SimpleNamespace

    T_cam = np.eye(4)
    T_cam[:3, :3] = np.diag([1.0, -1.0, -1.0])           # looking straight down at the table
    T_cam[:3, 3] = [0.15, -0.22, 1.2]
    K = np.array([[420.0, 0.0, 320.0], [0.0, 420.0, 240.0], [0.0, 0.0, 1.0]])
    return SimpleNamespace(rgb=np.zeros((480, 640, 3), dtype=np.uint8), K=K, T_base_cam=T_cam)


_SO101_BOX = {"min": [-0.15, -0.36, -0.02], "max": [0.4, 0.36, 0.4]}


def test_the_workspace_overlay_stands_where_the_arm_does():
    """The grid and the band are this arm's configuration (base frame): with
    a base 25 cm to the right they are drawn exactly where the same box
    shifted by the base_pose would be (the legend's two lines aside, which
    name the frame)."""
    from cascade.perception.visual_interface import VisualInterface

    # Binary-exact numbers (1/16 m), so "lifted" and "shifted" round alike.
    frame, T = _top_camera_frame(), transform([0.0, -0.25, 0.0, 0.0, 0.0, 0.0])
    box = {"min": [-0.125, -0.375, -0.0625], "max": [0.375, 0.375, 0.375]}
    shifted = {k: [v[0], v[1] - 0.25, v[2]] for k, v in box.items()}
    kw = dict(reach_x=(0.125, 0.25), grid_step_m=0.0625)
    lifted, _ = VisualInterface(workspace=box, **kw).render(frame, T_world_base=T)
    moved, _ = VisualInterface(workspace=shifted, **kw).render(frame)
    unlifted, _ = VisualInterface(workspace=box, **kw).render(frame)
    body = slice(0, 480 - 40)
    assert np.array_equal(lifted[body], moved[body])
    assert not np.array_equal(lifted[body], unlifted[body])


def test_a_belief_is_flagged_against_the_arms_own_workspace():
    """T3 re-review: `in_workspace` compares a table-frame belief with the
    arm's base-frame box only after taking the belief into that frame."""
    from types import SimpleNamespace

    from cascade.perception.visual_interface import VisualInterface

    frame, T = _top_camera_frame(), transform(SHIPPED_RIGHT)
    inside_b = SimpleNamespace(label="near", position=np.array([0.2, -0.5, 0.03]), last_seen_t=0.0, observations=3)
    outside_b = SimpleNamespace(label="far", position=np.array([0.2, 0.3, 0.03]), last_seen_t=0.0, observations=3)
    vi = VisualInterface(workspace=_SO101_BOX)
    _, marks = vi.render(frame, beliefs=[inside_b, outside_b], T_world_base=T, now=0.0)
    assert [m.extra["in_workspace"] for m in marks] == [True, False]
    _, plain = vi.render(frame, beliefs=[inside_b, outside_b], now=0.0)
    assert [m.extra["in_workspace"] for m in plain] == [False, True]          # no base_pose: as before


def test_an_unreadable_base_pose_never_breaks_a_viewer(make_runtime, monkeypatch):  # noqa: F811
    """T1 review Minor 6: an UNBOUND T_world_base read raises ArmFrameError
    when the primary's base_pose is not rigid. The viewers draw no arm
    overlay (TCP, grid, workspace flags) and keep the badges."""
    from cascade.perception.visual_interface import annotate_frame
    from cascade.skills.arm_frame import ArmFrameError

    rt = make_runtime(_pair_rig())
    rt.observe_fresh()
    rt.arm.__dict__["harness"].__dict__["base_pose"] = np.diag([2.0, 1.0, 1.0, 1.0])
    with pytest.raises(ArmFrameError):
        rt.T_world_base
    drawn = _tcp_spy(monkeypatch)
    img, marks = annotate_frame(rt)
    assert img is not None and drawn == []
    assert all(m.extra["in_workspace"] is None for m in marks)


# ── 4. annotated_view / analyze_scene result text ──────────────────────────


def test_annotated_view_text_names_the_table_frame_on_a_base_pose_rig(make_runtime):  # noqa: F811
    rt = make_runtime(_pair_rig())
    result = rt.execute("annotated_view", {"arm": "so101_right"})
    assert "Coordinates are table-frame metres" in result["key"], result["key"]
    assert "base-frame" not in result["key"] and "base frame" not in result["note"], result
    assert "table frame" in result["note"]
    analysis = rt.execute("analyze_scene", {})
    assert "Coordinates are table-frame metres" in analysis.get("key", "Coordinates are table-frame metres")


def test_annotated_view_text_reads_as_before_without_a_base_pose(make_runtime):  # noqa: F811
    rt = make_runtime(moved_cfg("so101_right", None))
    rt._reobserve(1)
    result = rt.execute("annotated_view", {})
    assert result["objects"], result                      # a key with badges, so its frame sentence is there
    assert "Entries outside the image have no badge. Coordinates are base-frame metres (x forward, y left, z up). " \
        "Workspace bounds and any configured grasp band do not verify" in result["key"], result["key"]
    assert "table" not in result["key"]
    assert result["note"].startswith(
        "Badges describe tracked estimates, which may be stale or mislabeled. No badges does not mean the scene is "
        "empty. This refreshes the camera image, not object detection; use get_observation to refresh tracking. "
        "Grid coordinates use the robot base frame (+X forward, +Y left, +Z up). Any shaded grasp band is a "
        "configured display hint"), result["note"]


# ── 5. the wrist-camera keep-out text (spatial_moves; T3 re-review) ─────────


@pytest.mark.parametrize("pose", [SHIPPED_RIGHT, YAWED, YAW90], ids=["shifted", "yawed", "yaw90"])
def test_the_wrist_camera_keep_out_refusal_quotes_the_table_frame(make_pair_rigs, make_runtime, pose):  # noqa: F811
    """`mock`'s wrist camera (8 cm above the gripper) already sits in a
    keep-out box over its home TCP (the arm's own config, base frame): the
    refusal quotes the box in the table frame on a base_pose arm (main:
    base-frame corners): the bounds of its 8 corners, plus the base-frame box
    when the mount rotates it (#407 review follow-up 3: two lifted corners
    read lo > hi on yaw 90)."""
    rt0 = make_runtime(moved_cfg("mock", None))
    tcp = rt0.kin.fk(rt0.arm.get_state().q)[:3, 3]
    lo = [float(tcp[0]) - 0.05, -0.05, float(tcp[2]) + 0.04]
    hi = [float(tcp[0]) + 0.05, 0.05, float(tcp[2]) + 0.2]

    def edit(cfg):
        for section in _safety_sections(cfg):
            section["keep_out"] = [{"min": lo, "max": hi}]

    rt_a, rt_b, T = make_pair_rigs(pose, arm="mock", edit=edit)
    run = run_pair(rt_a, rt_b, T, "move_relative", {"direction": "up", "frame": "base", "distance_m": 0.03})
    assert "refused by the wrist-camera rule" in run["result_a"]["error"], run["result_a"]
    assert "keep-out zone" in run["result_a"]["error"]
    assert_equivariant(_honest_run(run, T), T)
