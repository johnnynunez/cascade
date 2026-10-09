"""B47 -- opt-in "what the gripper sees" narration highlight on the dashboard.

ROADMAP "Wrist cam follow-ups": serve the wrist stream a narration highlight on
the dashboard. With `stream.wrist_narration: true` and a rig stream whose
profile is a wrist view (`is_wrist_view`: `role: wrist` or eye-in-hand
extrinsics), the dashboard highlights THAT tile while a motion skill runs and
shows one line under it, built only from state the runtime can vouch for: the
dispatched skill + target, the runtime's held-state (`held_object`, the
provisional marker of a close that has not completed) and the three-state
postcondition (agent/effects.py) once one exists. An unverified grasp reads
unverified, a refuted one refuted; no wrist camera = no panel at all.

Everything runs on the mock stack through the real `build_runtime` /
`SkillRuntime.execute` / `StreamServer` path (no MuJoCo, no GPU). The wrist
camera is the mock camera declared `role: wrist`, as in tests/test_keyframes.py.
Premise / golden tests pass on main by design: the flag defaults off, and off
is byte-identical (dashboard HTML sha256, /state keys, no narrator object).
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import threading
import time
import urllib.request

import pytest
from conftest import loopback_host

from cascade.apps.demo import _runtime_state, build_runtime, shutdown_runtime
from cascade.config import load_demo_config

#: sha256 of GET / for the mock wrist rig (tiles `mock_small`, `mock`; not
#: read-only), measured on origin/main be57535 before this change. The flag
#: off (absent or false) must serve exactly these bytes.
GOLDEN_INDEX_SHA256 = "7b753a830b44a534b2d5b41ee3e025e38576841a2e782f7802d2e47ca4337385"

#: `_runtime_state` keys on main for a mock rig with a watcher (flag off).
GOLDEN_STATE_KEYS = {
    "agent_status", "task", "holding", "last_path", "objects", "arm_connected",
    "events", "grasp_memory", "backends", "capabilities", "perception",
}


def _cfg(*, wrist: bool = True, narration: bool | None = None):
    if wrist:
        cfg = load_demo_config(cameras=["mock_small", "mock"], arm="so101_mock", llm="mock")
        cfg.cameras[1]["role"] = "wrist"          # the generic wrist-view flag
        cfg.cameras[1]["fuse_beliefs"] = False
    else:
        cfg = load_demo_config(camera="mock_small", arm="so101_mock", llm="mock")
    cfg._data["stream"]["port"] = 0               # ephemeral, never a shared port
    if narration is None:
        cfg._data["stream"].pop("wrist_narration", None)   # key absent: the code default
    else:
        cfg._data["stream"]["wrist_narration"] = narration
    return cfg


@pytest.fixture
def make_runtime(tmp_path):
    built = []

    def make(**kw):
        runtime, arm = build_runtime(_cfg(**kw), tmp_path / f"run{len(built)}", serve=True)
        built.append((runtime, arm))
        return runtime, arm

    yield make
    for runtime, arm in built:
        shutdown_runtime(runtime, arm)


def _wait_for_cube(runtime):
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline and runtime.beliefs.find("red object") is None:
        time.sleep(0.05)
    assert runtime.beliefs.find("red object") is not None, "watcher never fused the mock cube"


def _view(runtime):
    return _runtime_state(runtime).get("wrist_view")


def _get(server, path):
    url = f"http://{loopback_host()}:{server.port}{path}"
    return urllib.request.urlopen(url, timeout=5).read()


def _probe_skill(monkeypatch, runtime, name, hook):
    """Wrap `skill_<name>` on THIS runtime: `hook()` runs inside the real
    `execute()` dispatch, after the motion began and before the skill body."""
    real = getattr(runtime, f"skill_{name}")
    seen = []

    def probe(**kw):
        seen.append(hook())
        return real(**kw)

    monkeypatch.setattr(runtime, f"skill_{name}", probe)
    return seen


class _ProbeStream:
    """The wrist stream, recording the narration whenever the runtime grabs a
    FRESH wrist frame -- which it does for the AFTER keyframe of a motion
    skill, i.e. after the skill body returned and the postcondition was
    folded, but while execute() is still dispatching that motion."""

    def __init__(self, inner, runtime):
        self._inner, self._runtime, self.seen = inner, runtime, []
        self._owner = threading.current_thread()    # the thread that dispatches

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def get_frame(self, *a, **k):
        if threading.current_thread() is self._owner:   # never a watcher/pump grab
            self.seen.append(_view(self._runtime))
        return self._inner.get_frame(*a, **k)


def _force_grasp_verdict(monkeypatch, runtime, status, channel="physics"):
    """Make the REAL effects checker return a chosen three-state verdict for
    grasp_object (what an Isaac/MuJoCo physics channel would say); every
    other skill keeps its real verification."""
    from cascade.agent.effects import Postcondition

    real = runtime.effects.verify

    def verify(name, args, result, before=None):
        if name == "grasp_object":
            return Postcondition(skill=name, kind="holding", status=status,
                                 evidence="forced by the test", channel=channel)
        return real(name, args, result, before=before)

    monkeypatch.setattr(runtime.effects, "verify", verify)


# ── premises / golden pins (pass on main by design) ─────────────────────────


def test_premise_default_config_leaves_wrist_narration_off():
    cfg = load_demo_config(camera="mock", arm="mock", llm="mock")
    assert cfg.stream.get("wrist_narration", False) is False


def test_premise_the_mock_rig_declares_exactly_one_wrist_stream(make_runtime):
    runtime, _ = make_runtime()
    assert [n for n, _ in runtime._wrist_streams()] == ["mock"]
    plain, _ = make_runtime(wrist=False)
    assert plain._wrist_streams() == []


@pytest.mark.parametrize("narration", [None, False])
def test_golden_flag_off_is_byte_identical(make_runtime, narration):
    """Key absent (the code default) or false (what configs/demo.yaml ships):
    no narrator, the exact /state keys of main, and the exact dashboard bytes
    of main (sha256 measured on be57535)."""
    runtime, _ = make_runtime(narration=narration)
    assert getattr(runtime, "wrist_narrator", None) is None
    assert set(_runtime_state(runtime)) == GOLDEN_STATE_KEYS
    runtime.execute("move_home", {})
    assert set(_runtime_state(runtime)) == GOLDEN_STATE_KEYS
    assert runtime.live_view.open(reason="test")["ok"]
    html = _get(runtime.live_view.server, "/")
    assert hashlib.sha256(html).hexdigest() == GOLDEN_INDEX_SHA256
    assert "wrist_view" not in json.loads(_get(runtime.live_view.server, "/state"))


# ── behaviour through the real runtime / stream path ────────────────────────


def test_flag_on_attaches_a_narrator_to_the_wrist_stream(make_runtime):
    runtime, _ = make_runtime(narration=True)
    assert runtime.wrist_narrator.camera == "mock"
    view = _view(runtime)
    assert view["camera"] == "mock" and view["active"] is False
    assert view["line"] is None                     # nothing has moved yet: no claim
    assert view["gripper"] == "not_holding" and view["held"] is None


def test_flag_on_without_a_wrist_camera_serves_no_panel(make_runtime, capsys):
    """Never a front view standing in for the gripper's: no narrator, no
    state key, no panel -- and the operator is told why."""
    runtime, _ = make_runtime(wrist=False, narration=True)
    err = capsys.readouterr().err
    assert "stream.wrist_narration" in err and "no wrist" in err
    assert getattr(runtime, "wrist_narrator", None) is None
    runtime.execute("move_home", {})
    assert "wrist_view" not in _runtime_state(runtime)
    assert runtime.live_view.open(reason="test")["ok"]
    html = _get(runtime.live_view.server, "/").decode()
    assert "wrist-line" not in html and "renderWrist" not in html


def test_motion_skill_highlights_the_wrist_view_while_it_runs(make_runtime, monkeypatch):
    runtime, _ = make_runtime(narration=True)
    assert runtime.live_view.open(reason="test")["ok"]
    server = runtime.live_view.server
    seen = _probe_skill(monkeypatch, runtime, "move_home",
                        lambda: (_view(runtime), json.loads(_get(server, "/state"))["wrist_view"]))
    res = runtime.execute("move_home", {})
    assert res["ok"] is True
    (direct, served), = seen
    for view in (direct, served):                   # in-process and over HTTP
        assert view["active"] is True and view["skill"] == "move_home"
        assert view["line"] == "now: move_home · not holding"
        assert view["outcome"] is None and view["postcondition"] is None
    after = _view(runtime)
    assert after["active"] is False and after["outcome"] == "ok"
    assert after["postcondition"] == {"kind": "at_home", "status": "confirmed", "channel": "arm"}
    assert after["line"] == "last: move_home motion finished · not holding · at_home: confirmed (arm)"


def test_dashboard_puts_the_line_on_the_wrist_tile_only(make_runtime):
    runtime, _ = make_runtime(narration=True)
    assert runtime.live_view.open(reason="test")["ok"]
    html = _get(runtime.live_view.server, "/").decode()
    assert html.count('id="wrist-line"') == 1
    front = html.index('id="img-mock_small"')
    wrist = html.index('id="img-mock"')
    line = html.index('id="wrist-line"')
    assert front < wrist < line                     # inside the wrist tile, after its image
    assert html.count('data-wrist="1"') == 1 and html.index('data-wrist="1"') > front
    assert 'const WRIST = "mock";' in html
    assert "updateCameraStatus(); renderWrist(s);" in html   # wired into the state poll
    assert ".cam.wrist-active" in html


def test_unverified_grasp_reads_unverified_during_and_after(make_runtime):
    runtime, arm = make_runtime(narration=True)
    _wait_for_cube(runtime)
    arm.object_stop_frac = 0.5                      # the jaws stall on the cube: a hold
    probe = runtime.rig.streams["mock"] = _ProbeStream(runtime.rig.streams["mock"], runtime)
    res = runtime.execute("grasp_object", {"label": "red object"})
    assert res["ok"] is True and res["postcondition"]["status"] == "unverified"
    mid = probe.seen[-1]                            # AFTER keyframe grab: still dispatching
    assert mid["active"] is True and mid["held"] == "red object"
    assert mid["line"] == "now: grasp_object 'red object' · holding 'red object' (grasp unverified)"
    after = _view(runtime)
    assert after["active"] is False and after["grasp_verdict"] == "unverified"
    assert after["line"] == ("last: grasp_object 'red object' motion finished · "
                             "holding 'red object' (grasp unverified)")
    assert "confirmed" not in after["line"]


@pytest.mark.parametrize("status,words", [
    ("confirmed", "grasp confirmed by physics"),
    ("refuted", "grasp refuted by physics"),
    ("unverified", "grasp unverified"),
])
def test_the_next_motion_carries_the_grasp_verdict_it_has(make_runtime, monkeypatch, status, words):
    """A hold carried into a motion that cannot re-grasp is qualified by the
    grasp's own three-state verdict -- never upgraded, never a channel named
    for a verdict that is not confirmed/refuted."""
    runtime, arm = make_runtime(narration=True)
    _wait_for_cube(runtime)
    arm.object_stop_frac = 0.5
    _force_grasp_verdict(monkeypatch, runtime, status)
    res = runtime.execute("grasp_object", {"label": "red object"})
    assert res["postcondition"]["status"] == status
    assert runtime.held_object == "red object"      # the runtime's held-state
    assert _view(runtime)["grasp_verdict"] == status
    seen = _probe_skill(monkeypatch, runtime, "move_home", lambda: _view(runtime))
    runtime.execute("move_home", {})
    assert seen[0]["line"] == f"now: move_home · holding 'red object' ({words})"
    assert seen[0]["outcome"] is None and seen[0]["postcondition"] is None   # no claim mid-motion
    assert _view(runtime)["line"] == (f"last: move_home motion finished · holding 'red object' "
                                      f"({words}) · at_home: confirmed (arm)")


def test_a_motion_that_can_regrasp_drops_the_verdict(make_runtime, monkeypatch):
    """A confirmed verdict belongs to ONE hold. A skill that may release and
    grasp again inside one call (here grasp_object itself) must not inherit
    it: until its own verdict exists the hold reads unverified."""
    runtime, arm = make_runtime(narration=True)
    _wait_for_cube(runtime)
    arm.object_stop_frac = 0.5
    _force_grasp_verdict(monkeypatch, runtime, "confirmed")
    runtime.execute("grasp_object", {"label": "red object"})
    assert _view(runtime)["grasp_verdict"] == "confirmed"
    seen = _probe_skill(monkeypatch, runtime, "grasp_object", lambda: _view(runtime))
    runtime.execute("grasp_object", {"label": "red object"})
    assert seen[0]["held"] == "red object"
    assert seen[0]["line"] == "now: grasp_object 'red object' · holding 'red object' (grasp unverified)"
    assert seen[0]["grasp_verdict"] == "unverified"


def test_a_released_hold_drops_the_verdict(make_runtime, monkeypatch):
    runtime, arm = make_runtime(narration=True)
    _wait_for_cube(runtime)
    arm.object_stop_frac = 0.5
    _force_grasp_verdict(monkeypatch, runtime, "confirmed")
    runtime.execute("grasp_object", {"label": "red object"})
    res = runtime.execute("open_gripper", {})
    assert res["ok"] is True and runtime.held_object is None
    view = _view(runtime)
    assert view["grasp_verdict"] is None
    assert view["line"] == "last: open_gripper motion finished · not holding · empty: confirmed (gripper)"
    # the same label held again by some later path is a NEW hold: unverified
    runtime.held_object = "red object"
    try:
        assert _view(runtime)["grasp_verdict"] == "unverified"
        assert "(grasp unverified)" in _view(runtime)["line"]
    finally:
        runtime.held_object = None


def test_a_different_hold_never_inherits_a_verdict(make_runtime, monkeypatch):
    runtime, arm = make_runtime(narration=True)
    _wait_for_cube(runtime)
    arm.object_stop_frac = 0.5
    _force_grasp_verdict(monkeypatch, runtime, "confirmed")
    runtime.execute("grasp_object", {"label": "red object"})

    def swap():
        runtime.held_object = "blue object"           # a different hold mid-carry
        return _view(runtime)

    seen = _probe_skill(monkeypatch, runtime, "move_home", swap)
    try:
        runtime.execute("move_home", {})
    finally:
        runtime.held_object = None
    assert seen[0]["line"] == "now: move_home · holding 'blue object' (grasp unverified)"


def test_a_close_that_has_not_completed_is_never_a_hold(make_runtime, monkeypatch):
    runtime, _ = make_runtime(narration=True)

    def provisional():
        runtime._held_provisional = ("red object", "red cube", "red")
        try:
            return _view(runtime)
        finally:
            runtime._held_provisional = None

    seen = _probe_skill(monkeypatch, runtime, "move_home", provisional)
    runtime.execute("move_home", {})
    assert seen[0]["gripper"] == "closing" and seen[0]["held"] is None
    assert seen[0]["line"] == "now: move_home · closing on 'red object' (grasp not complete, unverified)"


def test_a_refuted_air_grasp_reads_refuted_and_not_holding(make_runtime):
    runtime, arm = make_runtime(narration=True)
    _wait_for_cube(runtime)
    arm.object_stop_frac = None                     # the jaws close on air
    res = runtime.execute("grasp_object", {"label": "red object"})
    assert res["ok"] is False and res["postcondition"]["status"] == "refuted"
    view = _view(runtime)
    assert view["outcome"] == "failed" and view["held"] is None
    assert view["line"] == ("last: grasp_object 'red object' failed · not holding · "
                            "holding: refuted (belief)")


def test_an_unverified_verdict_never_names_a_channel(make_runtime, monkeypatch):
    """A channel is named only for a verdict it actually reached (confirmed
    or refuted); 'unverified (physics)' would read as a physics check."""
    runtime, arm = make_runtime(narration=True)
    _wait_for_cube(runtime)
    arm.object_stop_frac = None
    _force_grasp_verdict(monkeypatch, runtime, "unverified", channel="physics")
    runtime.execute("grasp_object", {"label": "red object"})
    assert _view(runtime)["line"] == ("last: grasp_object 'red object' failed · not holding · "
                                      "holding: unverified")


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed: the dashboard "
                    "script is only exercised where a JS runtime exists")
def test_the_served_script_highlights_and_captions_the_wrist_tile(make_runtime, monkeypatch):
    """Run the page's own renderWrist (extracted from the served HTML) under
    node against a minimal DOM, fed with REAL /state snapshots: the tile is
    highlighted only while a motion runs, the caption is the server's line,
    and no line means a hidden caption."""
    runtime, _ = make_runtime(narration=True)
    assert runtime.live_view.open(reason="test")["ok"]
    html = _get(runtime.live_view.server, "/").decode()
    script = html[html.index(" const WRIST = "):html.index(" tick();\n</script>")]
    first = _view(runtime)                                      # nothing moved yet
    seen = _probe_skill(monkeypatch, runtime, "move_home", lambda: _view(runtime))
    runtime.execute("move_home", {})
    states = [{"wrist_view": first}, {"wrist_view": seen[0]}, {"wrist_view": _view(runtime)}, {}]
    harness = (
        "const classes = new Set();\n"
        "const tile = {classList: {toggle: (c, on) => on ? classes.add(c) : classes.delete(c)}};\n"
        "const img = {closest: () => tile};\n"
        "const line = {textContent: 'stale', hidden: false};\n"
        "const document = {getElementById: id => id === 'img-mock' ? img"
        " : (id === 'wrist-line' ? line : null)};\n"
        + script +
        f"const out = [];\nfor (const s of {json.dumps(states)}) {{\n"
        "  renderWrist(s);\n"
        "  out.push([line.textContent, line.hidden, classes.has('wrist-active')]);\n}\n"
        "console.log(JSON.stringify(out));\n"
    )
    proc = subprocess.run(["node", "-e", harness], capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout) == [
        ["", True, False],
        ["now: move_home · not holding", False, True],
        ["last: move_home motion finished · not holding · at_home: confirmed (arm)", False, False],
        ["", True, False],
    ]


def test_the_wrist_name_cannot_close_the_script():
    """The stream name is spliced into a <script> as JSON; a `<` is escaped
    so a name can never end the script element early."""
    from cascade.apps.stream_server import StreamServer

    class _Rig:
        names = ["front", "w</script>"]

        def stats(self):
            return {}

    server = StreamServer(_Rig(), port=0, wrist_view="w</script>")
    server.start()
    try:
        html = _get(server, "/").decode()
    finally:
        server.stop()
    assert 'const WRIST = "w\\u003c/script>";' in html
    start = html.index("<script>")
    assert "function renderWrist" in html[start:html.index("</script>", start)]


def test_a_dead_wrist_stream_is_named_and_never_blocks(make_runtime):
    runtime, _ = make_runtime(narration=True)

    class _Dead:                                    # never delivered a frame
        name = "mock"

        def latest(self):
            return None

        def get_frame(self, *a, **k):
            raise RuntimeError("wrist camera unplugged")

    class _Raising(_Dead):                          # a stream whose slot read faults
        def latest(self):
            raise RuntimeError("wrist camera unplugged")

    real = runtime.rig.streams["mock"]
    runtime.rig.streams["mock"] = _Dead()
    try:
        res = runtime.execute("move_home", {})
        view = _view(runtime)
        runtime.rig.streams["mock"] = _Raising()
        raising = runtime.wrist_narrator.snapshot(runtime)
    finally:
        runtime.rig.streams["mock"] = real
    assert res["ok"] is True
    assert view["wrist_frame"] is False and raising["wrist_frame"] is False
    assert view["line"].endswith(" · wrist camera: no frame")
    assert raising["line"] == view["line"]
    assert _view(runtime)["wrist_frame"] is True and "no frame" not in _view(runtime)["line"]


def test_a_runtime_fault_never_leaves_the_highlight_on(make_runtime, monkeypatch):
    runtime, _ = make_runtime(narration=True)

    def boom(*a, **k):
        raise RuntimeError("trace disk full")

    monkeypatch.setattr(runtime.trace, "record", boom)
    with pytest.raises(RuntimeError, match="trace disk full"):
        runtime.execute("move_home", {})
    view = _view(runtime)
    assert view["active"] is False and view["outcome"] is None
    assert view["line"] == "last: move_home ended without a result · not holding"


def test_a_stuck_motion_reads_stuck_never_finished(make_runtime, monkeypatch):
    from cascade.types import SkillStuck

    runtime, _ = make_runtime(narration=True)

    def stuck(**kw):
        raise SkillStuck("the path home is blocked", ask="clear the arm's way home")

    monkeypatch.setattr(runtime, "skill_move_home", stuck)
    res = runtime.execute("move_home", {})
    assert res["outcome"] == "stuck"
    view = _view(runtime)
    assert view["outcome"] == "stuck"
    assert view["line"].startswith("last: move_home stuck · not holding")
    assert "finished" not in view["line"]


def test_query_skills_leave_the_narration_alone(make_runtime):
    runtime, _ = make_runtime(narration=True)
    runtime.execute("get_observation", {})
    view = _view(runtime)
    assert view["active"] is False and view["skill"] is None and view["line"] is None


def test_a_nested_dispatch_keeps_the_outer_motion(make_runtime, monkeypatch):
    runtime, _ = make_runtime(narration=True)

    def nested():
        runtime.execute("open_gripper", {})       # e.g. a program step inside a motion
        return _view(runtime)

    seen = _probe_skill(monkeypatch, runtime, "move_home", nested)
    runtime.execute("move_home", {})
    assert seen[0]["active"] is True and seen[0]["skill"] == "move_home"
    assert seen[0]["line"] == "now: move_home · not holding"
    assert _view(runtime)["skill"] == "move_home" and _view(runtime)["active"] is False


def test_place_at_names_its_commanded_point(make_runtime, monkeypatch):
    runtime, _ = make_runtime(narration=True)
    seen = _probe_skill(monkeypatch, runtime, "place_at", lambda: _view(runtime))
    runtime.execute("place_at", {"x": 0.2, "y": "-0.1"})
    assert seen[0]["target"] == "(0.20, -0.10)"
    assert seen[0]["line"] == "now: place_at (0.20, -0.10) · not holding"


def test_labels_are_quoted_collapsed_and_bounded():
    from cascade.apps.wrist_narration import motion_target

    assert motion_target("grasp_object", {"label": "  red\n  object "}) == "'red object'"
    assert motion_target("pick_and_place", {"object": "cup", "label": None}) == "'cup'"
    assert motion_target("search_for_object", {"query": "keys"}) == "'keys'"
    long = motion_target("grasp_object", {"label": "x" * 200})
    assert long.startswith("'xxx") and long.endswith("...'") and len(long) == 42
    assert motion_target("move_home", {}) is None
    assert motion_target("place_at", {"x": "left", "y": 0.1}) is None
    assert motion_target("wave", {"label": 7}) is None


def test_the_carry_set_holds_only_skills_that_cannot_grasp():
    """Fail-safe direction: a motion skill absent from the set (any new one)
    drops a grasp verdict; every member must be a real motion skill, and no
    skill that can start a grasp may be in it."""
    from cascade.apps.wrist_narration import CARRIES_GRASP_VERDICT
    from cascade.skills.runtime import _MOTION_SKILLS

    assert CARRIES_GRASP_VERDICT <= _MOTION_SKILLS
    for grasping in ("grasp_object", "pick_and_place", "grasp_at_pixel", "sort_by_color",
                     "throw", "handover", "restore_scene", "search_for_object",
                     "reset_scene", "turn_screw", "push_object"):
        assert grasping not in CARRIES_GRASP_VERDICT
