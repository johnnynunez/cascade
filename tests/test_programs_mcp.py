"""B42 (ROADMAP #8 follow-up): the programs tier for MCP chat hosts, and
opt-in embedding retrieval of programs.

Design note: docs/PROGRAMS_TIER.md ("MCP chat hosts", "Storage and
retrieval"). What these tests pin, most important first:

* `agent.programs` off (the default): the MCP catalog, every call result
  (`run_program` is an unknown tool, exactly as before), `world_state`'s
  capability matrix and the cancel path are byte-identical to the pre-change
  server. These golden pins pass on main by design;
* tier on: `list_programs` lists only PROMOTED programs; `run_program` runs a
  promoted program -- or a host-submitted spec, once -- through the SAME
  runner: every step a top-level `execute()` with its own trace row and
  ledger verdict, `$target` queries re-grounded before the first motion, the
  first unverified step stops the program with a next_action;
* a candidate is refused by reference with zero motion; a submitted spec is
  stored only from a fully CONFIRMED execution and promoted only across two
  distinct tasks (the orchestrator's admission rule, unchanged);
* a stop (`notifications/cancelled` on the in-flight run_program) latches the
  e-stop and no later step runs: between steps the runner never dispatches
  the next call, during a step the harness refuses it;
* the capability matrix withholds the tools with a reason (library
  unavailable; run_program without the verifier); a program naming a
  withheld or operator-hidden tool is refused before anything runs;
* with a memory embedder, programs are ranked by text embedding with the
  skill-library floor-or-guard rule; without one, keyword overlap as before;
* two processes sharing one store never lose each other's evidence.

New symbols are imported inside each test, so the file collects on main and
every behaviour test fails on its own (RED); the golden pins pass there.
Ports: this item's block (45400-45499); no sidecar is ever dialed.
"""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from conftest import needs_pin
from test_mcp_server import McpClient, _tool_payload

from cascade.agent.effects import CONFIRMED, POSTCONDITIONS, UNVERIFIED, Postcondition
from cascade.agent.llm import LLMResponse, MockLLM, ToolCall

REPO = Path(__file__).resolve().parents[1]
#: inside this item's port block; nothing listens there (dead sidecars)
PORTS = {"CASCADE_GRASPGENX_PORT": "45411", "CASCADE_OCCUPANCY_PORT": "45412",
         "CASCADE_BRIDGE_PORT": "45413", "CASCADE_HUG_PORT": "45414"}
PROGRAM_TOOLS = ("list_programs", "run_program")
#: the host extras as they were before B42, in catalog order
PRE_B42_EXTRAS = ("camera_snapshot", "world_state", "live_view_url", "emergency_stop",
                  "reset_stop", "robot_knowledge", "verify_last_action", "task_memory")
PRE_B42_CAPABILITY_KEYS = {"probed", "cameras", "arms", "bases", "depth_3d", "depth_heights",
                           "learned_grasps", "occupancy", "multi_arm", "mobile_base", "verifier",
                           "memory"}
TASK = "move the red cube to the front-left of the table"


def _mp():
    from cascade.memory import programs

    return programs


def _ap():
    from cascade.agent import programs

    return programs


def _gripper_spec(name="home-close-open"):
    """Four registered effects the mock stack CONFIRMS on its own channels
    (measured: at_home / closed / empty), so a subprocess server can verify
    a program end to end without a stand-in physics channel."""
    return {"name": name, "description": "close and open the gripper between two home moves",
            "params": {}, "steps": [
                {"tool": "move_home", "args": {}},
                {"tool": "close_gripper", "args": {}},
                {"tool": "open_gripper", "args": {}},
                {"tool": "move_home", "args": {}}]}


def _place_spec(name="move-front-left", offset=(-0.09, -0.15)):
    """grasp $object, then place it at localize_object($object) + offset."""
    return {"version": 1, "name": name,
            "description": "move an object to the front-left of where it is",
            "params": {"object": "the object to move"},
            "steps": [
                {"tool": "grasp_object", "args": {"label": {"$param": "object"}}},
                {"tool": "place_at", "args": {"$target": {
                    "query": "localize_object", "label": {"$param": "object"},
                    "offset_m": list(offset), "args": ["x", "y"]}}}]}


def _admit(lib, spec, task, run, origin="authored"):
    ap = _ap()
    prog = ap.Program.from_spec(spec)
    return lib.admit(prog.spec, prog.signature, task=task, run=run, origin=origin, verdict=CONFIRMED)


def _seed(path, spec, tasks):
    lib = _mp().ProgramLibrary(path)
    for i, task in enumerate(tasks):
        _admit(lib, spec, task, f"seed-{i}")
    return lib


def _pre_b42_catalog(*, single_arm: bool, withheld=()):
    """The pre-change `list_tools()` algorithm over the pre-change candidate
    list (TOOL_SPECS + the eight original host extras)."""
    from cascade.apps.mcp_server import _EXCLUDED_TOOLS, _EXTRA_TOOLS
    from cascade.skills.runtime import _MOTION_SKILLS, TOOL_SPECS

    extras = [t for t in _EXTRA_TOOLS if t["name"] not in PROGRAM_TOOLS]
    assert tuple(t["name"] for t in extras) == PRE_B42_EXTRAS
    out = []
    for t in TOOL_SPECS + extras:
        if t["name"] in _EXCLUDED_TOOLS or t["name"] in withheld:
            continue
        schema = t["parameters"]
        if single_arm and t["name"] in _MOTION_SKILLS and "arm" in schema.get("properties", {}):
            schema = copy.deepcopy(schema)
            schema["properties"].pop("arm")
        out.append({"name": t["name"], "description": t["description"], "inputSchema": schema})
    return out


def _env(monkeypatch, tmp_path, *, programs: bool | None = True, camera="mock", extra=None):
    """In-process server environment: mock rig, ports in this item's block,
    a private program store."""
    for key, value in PORTS.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("CASCADE_CAMERA", camera)
    monkeypatch.delenv("CASCADE_CAMERAS", raising=False)
    monkeypatch.setenv("CASCADE_ARM", "mock")
    monkeypatch.delenv("CASCADE_ARMS", raising=False)
    monkeypatch.delenv("CASCADE_HIDE_TOOLS", raising=False)
    monkeypatch.setenv("CASCADE_RUN_DIR", str(tmp_path / "run"))
    monkeypatch.setenv("CASCADE_STREAM", "0")
    monkeypatch.setenv("CASCADE_VIEW", "0")
    monkeypatch.setenv("CASCADE_PROGRAMS_PATH", str(tmp_path / "programs.jsonl"))
    if programs is None:
        monkeypatch.delenv("CASCADE_PROGRAMS", raising=False)
    else:
        monkeypatch.setenv("CASCADE_PROGRAMS", "1" if programs else "0")
    for key, value in (extra or {}).items():
        monkeypatch.setenv(key, value)


def _wrap_config(monkeypatch, mutate):
    """Route the server's config through `mutate(cfg)` (it imports
    load_demo_config from cascade.config at call time)."""
    import cascade.config as config_mod

    real = config_mod.load_demo_config

    def wrapped(**kwargs):
        cfg = real(**kwargs)
        mutate(cfg)
        return cfg

    monkeypatch.setattr(config_mod, "load_demo_config", wrapped)


@pytest.fixture
def make_server():
    servers = []

    def make():
        from cascade.apps.mcp_server import McpSkillServer

        server = McpSkillServer()
        servers.append(server)
        return server

    yield make
    for server in servers:
        server.shutdown()


def _payload(result: dict) -> dict:
    return json.loads(next(b["text"] for b in reversed(result["content"]) if b["type"] == "text"))


def _trace_rows(runtime) -> list[dict]:
    path = runtime.trace.run_dir / "trace.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def _confirm_effects(runtime) -> None:
    """Stand-in physics channel (same as tests/test_programs_tier.py and the
    recipe tests): the static mock camera leaves every pick `unverified`;
    these tests are about the program contract, so registered effects are
    pinned CONFIRMED. Motion, the harness and the ledger are untouched."""
    def verify(name, args, result, before=None):
        kind = POSTCONDITIONS.get(name)
        if kind is None:
            return None
        return Postcondition(name, kind, CONFIRMED, "stand-in physics channel (test)", channel="physics")

    runtime.effects.verify = verify


def _wait_for_cube(runtime) -> None:
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline and runtime.beliefs.find("red cube") is None:
        time.sleep(0.05)
    assert runtime.beliefs.find("red cube") is not None, "watcher never saw the mock cube"


def _client(tmp_path, *, programs: bool, **extra) -> McpClient:
    env = {**PORTS, "CASCADE_PROGRAMS_PATH": str(tmp_path / "programs.jsonl"),
           "CASCADE_PROGRAMS": "1" if programs else "0", **extra}
    return McpClient(str(tmp_path / "run"), extra_env=env)


# ── A. golden: tier off = the pre-change MCP server, byte for byte ──────────


@needs_pin
def test_tier_off_catalog_calls_and_matrix_are_byte_identical(tmp_path, monkeypatch):
    """PREMISE/GOLDEN (passes on main): with agent.programs at its shipped
    default the program tools are not candidates at all -- the catalog
    before and after the build, the result of calling them (an unknown tool,
    exactly as before), world_state's capability matrix and withheld list
    are the pre-change ones, and no program store is touched."""
    monkeypatch.delenv("CASCADE_PROGRAMS", raising=False)
    c = McpClient(str(tmp_path / "run"), extra_env={
        **PORTS, "CASCADE_PREWARM": "0", "CASCADE_PROGRAMS_PATH": str(tmp_path / "programs.jsonl")})
    try:
        c.request("initialize", {"protocolVersion": "2025-06-18"})
        before = c.request("tools/list")["result"]["tools"]
        assert json.dumps(before) == json.dumps(_pre_b42_catalog(single_arm=False))

        state, is_err = _tool_payload(c.request("tools/call", {"name": "world_state", "arguments": {}}))
        assert not is_err
        assert set(state["capabilities"]) == PRE_B42_CAPABILITY_KEYS
        assert set(state["tools_withheld"]) == {"list_arms"}

        after = c.request("tools/list")["result"]["tools"]
        assert json.dumps(after) == json.dumps(_pre_b42_catalog(single_arm=True, withheld={"list_arms"}))

        unknown, unknown_err = _tool_payload(
            c.request("tools/call", {"name": "warp_drive", "arguments": {}}))
        assert unknown_err
        for name in PROGRAM_TOOLS:
            payload, is_err = _tool_payload(c.request("tools/call", {
                "name": name, "arguments": {"program": "x", "task": "y", "query": "z"}}))
            assert is_err
            assert payload == {**unknown, "error": unknown["error"].replace("warp_drive", name)}
    finally:
        c.close()
    assert not (tmp_path / "programs.jsonl").exists(), "the tier off touches no program store"
    banner = [line for line in (tmp_path / "run" / "server.log").read_text().splitlines()
              if "capabilities:" in line]
    assert banner and not any("programs=" in line for line in banner)


@needs_pin
def test_tier_off_runtime_and_cancel_path_are_unchanged(monkeypatch, tmp_path, make_server):
    """PREMISE/GOLDEN (passes on main): no attribute is attached to the
    runtime, and a cancel of an in-flight call NAMED run_program is not a
    motion cancel while the tool does not exist."""
    _env(monkeypatch, tmp_path, programs=None)
    server = make_server()
    runtime = server._ensure_runtime()
    assert not hasattr(runtime, "program_tier")
    server._inflight = (5, "run_program")
    server.cancel_request(5)
    assert not runtime.arm.harness.estopped
    assert "programs" not in server.capabilities()


# ── B. tier on: the host's surface, end to end over JSON-RPC ────────────────


@needs_pin
def test_host_program_becomes_a_candidate_then_promoted_then_listed_and_reused(tmp_path):
    """The whole lifecycle through a real stdio server on the mock stack: a
    host-submitted program runs step by step with every effect CONFIRMED and
    is stored as a CANDIDATE (not listed, refused by reference); the same
    structure verified in a second distinct task is PROMOTED, listed with its
    parameters and evidence, and reused by name."""
    store = tmp_path / "programs.jsonl"
    c = _client(tmp_path, programs=True)
    try:
        c.request("initialize", {"protocolVersion": "2025-06-18"})
        names = [t["name"] for t in c.request("tools/list")["result"]["tools"]]
        assert set(PROGRAM_TOOLS) <= set(names)
        state, _ = _tool_payload(c.request("tools/call", {"name": "world_state", "arguments": {}}))
        assert state["capabilities"]["programs"]["available"] is True
        assert not set(PROGRAM_TOOLS) & set(state["tools_withheld"])
        after = {t["name"]: t for t in c.request("tools/list")["result"]["tools"]}
        assert set(PROGRAM_TOOLS) <= set(after)
        assert after["run_program"]["inputSchema"]["required"] == ["task"]

        def run(args):
            return _tool_payload(c.request("tools/call", {"name": "run_program", "arguments": args},
                                           timeout=120))

        first, is_err = run({"spec": _gripper_spec(), "bindings": {}, "task": "close and open the gripper"})
        assert not is_err and first["ok"] is True, first
        assert first["status"] == "completed" and first["verdict"] == CONFIRMED and first["verified"]
        assert [s["tool"] for s in first["steps"]] == ["move_home", "close_gripper", "open_gripper", "move_home"]
        assert all(s["verdict"] == CONFIRMED for s in first["steps"])
        assert first["source"] == "submitted"
        assert first["library"]["outcome"] == "admitted" and first["library"]["status"] == "candidate"

        listed, _ = _tool_payload(c.request("tools/call", {"name": "list_programs", "arguments": {}}))
        assert listed["programs"] == [], "a candidate is never offered"
        assert listed["library"]["candidates"] == 1

        refused, is_err = run({"program": "home-close-open", "task": "close and open the gripper again"})
        assert is_err and refused["status"] == "refused" and "not promoted" in refused["error"]

        second, is_err = run({"spec": _gripper_spec(name="other-name"), "task": "test the jaws"})
        assert not is_err and second["library"]["status"] == "promoted", second["library"]

        listed, _ = _tool_payload(c.request("tools/call", {
            "name": "list_programs", "arguments": {"query": "exercise the gripper"}}))
        (prog,) = listed["programs"]
        assert prog["name"] == "home-close-open" and prog["params"] == {}
        assert prog["evidence"]["distinct_tasks"] == 2 and prog["evidence"]["verified_runs"] == 2
        assert [s["tool"] for s in prog["steps"]][:2] == ["move_home", "close_gripper"]

        reused, is_err = run({"program": "home-close-open", "bindings": {}, "task": "exercise the gripper"})
        assert not is_err and reused["source"] == "stored" and reused["verdict"] == CONFIRMED
        assert reused["library"]["verified_runs"] == 3 and reused["library"]["distinct_tasks"] == 3
    finally:
        c.close()
    rows = [json.loads(line) for line in (tmp_path / "run" / "trace.jsonl").read_text().splitlines()]
    program_rows = [r["skill"] for r in rows if r.get("tier") == "program"]
    assert program_rows == ["move_home", "close_gripper", "open_gripper", "move_home"] * 3, program_rows
    assert len(list((tmp_path / "run" / "programs").glob("*.json"))) == 3, "one receipt per execution"
    (record,) = [json.loads(line) for line in store.read_text().splitlines() if line.strip()]
    assert set(record["origins"]) == {"authored", "reused"} and record["losses"] == 0


@needs_pin
def test_notifications_cancelled_on_an_in_flight_run_program_freezes_the_arm(tmp_path):
    """Esc in the host mid-program: the stdin reader sees the cancel of the
    in-flight run_program, latches the e-stop like for any motion tool, the
    step in flight is refused by the harness and no later step runs."""
    c = _client(tmp_path, programs=True)
    try:
        c.request("initialize", {"protocolVersion": "2025-06-18"})
        c.request("tools/call", {"name": "get_observation", "arguments": {}})  # runtime up
        spec = {"name": "pick-then-home", "params": {}, "steps": [
            {"tool": "pick_and_place", "args": {"object": "red object"}},
            {"tool": "move_home", "args": {}}]}
        call_id = c.send("tools/call", {"name": "run_program", "arguments": {
            "spec": spec, "bindings": {}, "task": "pick the red object"}})
        time.sleep(0.5)  # the mock pick persists for seconds; it is in flight now
        c.notify("notifications/cancelled", {"requestId": call_id, "reason": "user"})
        resp = c.recv(timeout=120)
        assert resp["id"] == call_id
        payload, is_err = _tool_payload(resp)
        assert is_err and payload["status"] == "stopped" and payload["stopped_at"] == 1, payload
        # interrupted mid-motion: the harness refused the rest of the pick, and
        # the ledger may also REFUTE its effect (the object never arrived)
        step = payload["steps"][0]
        assert step["tool"] == "pick_and_place" and step["verdict"] in ("refused", "refuted"), step
        assert "e-stop" in json.dumps(step) and payload["estop_latched"] is True, step
        payload, is_err = _tool_payload(c.request("tools/call", {"name": "move_home", "arguments": {}}))
        assert is_err and "e-stop" in payload["error"]
    finally:
        c.close()
    rows = [json.loads(line) for line in (tmp_path / "run" / "trace.jsonl").read_text().splitlines()]
    assert [r["skill"] for r in rows if r.get("tier") == "program"] == ["pick_and_place"]
    assert "cancelled 'run_program' mid-motion -> e-stop" in (tmp_path / "run" / "server.log").read_text()


# ── C. tier on, in process: the runner, re-grounding, refusals, stops ───────


@needs_pin
def test_a_promoted_program_is_listed_and_run_step_by_step_with_regrounding(monkeypatch, tmp_path, make_server):
    _env(monkeypatch, tmp_path)
    store = tmp_path / "programs.jsonl"
    _seed(store, _place_spec(), ["move the blue cube to the front-left", "shift the green cube front-left"])
    server = make_server()
    runtime = server._ensure_runtime()
    assert int(runtime.cfg.grasp.graspgenx.port) == 45411 and int(runtime.cfg.occupancy.port) == 45412
    server._arm._ensure().object_stop_frac = 0.5   # the mock jaws close on the cube
    _confirm_effects(runtime)
    _wait_for_cube(runtime)
    cube = runtime.beliefs.find("red cube").position.copy()

    listed = _payload(server.call_tool("list_programs", {"query": TASK}))
    assert listed["retrieval"].startswith("keyword")
    (prog,) = listed["programs"]
    assert prog["name"] == "move-front-left" and list(prog["params"]) == ["object"]
    assert prog["run_with"] == {"program": "move-front-left", "bindings": {"object": "<object label>"}}
    assert "localize_object" in prog["summary"]

    with runtime.watcher.paused():
        result = server.call_tool("run_program", {"program": "move-front-left",
                                                  "bindings": {"object": "red cube"}, "task": TASK})
    data = _payload(result)
    assert not result["isError"] and data["ok"] and data["verified"], data
    assert data["source"] == "stored" and data["grounded"][0]["label"] == "red cube"
    rows = [r for r in _trace_rows(runtime) if r.get("tier") == "program"]
    assert [r["skill"] for r in rows] == ["localize_object", "grasp_object", "place_at"]
    for row in rows[1:]:   # each step its own top-level row and verdict
        assert row["result"]["postcondition"]["status"] == CONFIRMED
        assert row["keyframe_before"] and row["keyframe_after"]
    place = rows[2]["args"]
    assert place["x"] == pytest.approx(cube[0] - 0.09, abs=0.01)
    assert place["y"] == pytest.approx(cube[1] - 0.15, abs=0.01)
    assert runtime.last_path == "program"
    record = _mp().ProgramLibrary(store).get("move-front-left")
    assert record.occurrences == 3 and record.n_tasks == 3 and "reused" in record.origins
    receipt = json.loads(Path(data["receipt"]).read_text())
    assert receipt["verdict"] == CONFIRMED and [s["verdict"] for s in receipt["steps"]] == [CONFIRMED] * 2


@needs_pin
def test_an_unverified_step_stops_the_program_with_a_next_action(monkeypatch, tmp_path, make_server):
    _env(monkeypatch, tmp_path)
    server = make_server()
    runtime = server._ensure_runtime()
    server._arm._ensure().object_stop_frac = 0.5   # held, but the static mock cannot confirm it
    _wait_for_cube(runtime)
    with runtime.watcher.paused():
        result = server.call_tool("run_program", {"spec": _place_spec(), "bindings": {"object": "red cube"},
                                                  "task": TASK})
    data = _payload(result)
    assert result["isError"] and data["ok"] is False
    assert data["status"] == "stopped" and data["stopped_at"] == 1 and data["verdict"] == UNVERIFIED
    assert data["steps"][0]["tool"] == "grasp_object" and data["steps"][0]["verdict"] == UNVERIFIED
    assert "UNVERIFIED" in data["next_action"] and "Do not re-run" in data["next_action"]
    skills = [r["skill"] for r in _trace_rows(runtime)]
    assert "grasp_object" in skills and "place_at" not in skills, skills
    assert data["library"]["outcome"] == "not admitted"
    assert len(_mp().ProgramLibrary(tmp_path / "programs.jsonl")) == 0, "never admitted"


@needs_pin
def test_a_candidate_is_not_listed_and_running_it_by_name_is_refused(monkeypatch, tmp_path, make_server):
    _env(monkeypatch, tmp_path)
    store = tmp_path / "programs.jsonl"
    _seed(store, _place_spec(), ["move the blue cube to the front-left"])   # one task: candidate
    server = make_server()
    runtime = server._ensure_runtime()
    listed = _payload(server.call_tool("list_programs", {"query": TASK}))
    assert listed["programs"] == [] and listed["library"]["candidates"] == 1
    result = server.call_tool("run_program", {"program": "move-front-left",
                                              "bindings": {"object": "red cube"}, "task": TASK})
    data = _payload(result)
    assert result["isError"] and data["status"] == "refused"
    assert "not promoted" in data["error"] and "No motion" in data["next_action"]
    assert not [r for r in _trace_rows(runtime) if r.get("tier") == "program"], "nothing ran"
    assert _mp().ProgramLibrary(store).get("move-front-left").occurrences == 1


@needs_pin
@pytest.mark.parametrize("when", ["between", "during"])
def test_a_cancel_mid_program_latches_the_estop_and_runs_nothing_after(monkeypatch, tmp_path, make_server, when):
    """Deterministic twin of the JSON-RPC cancel test: the cancel reaches the
    server's real cancel_request() while the worker runs the program, either
    after step 1 finished (the runner never dispatches step 2) or as step 2
    starts (the harness refuses it). Either way nothing after it runs."""
    _env(monkeypatch, tmp_path)
    server = make_server()
    runtime = server._ensure_runtime()
    executed: list[str] = []
    locked: list[bool] = []
    real = runtime.execute

    def spy(name, args):
        locked.append(server._exec_lock.locked())  # one arm, one command: the chat cannot interleave
        if when == "during" and len(executed) == 1:
            server.cancel_request(77)          # arrives as step 2 is dispatched
        executed.append(name)
        result = real(name, args)
        if when == "between" and len(executed) == 1:
            server.cancel_request(77)          # arrives after step 1 finished
        return result

    runtime.execute = spy
    server._inflight = (77, "run_program")     # what the worker publishes for the call
    spec = {"name": "four-steps", "params": {}, "steps": [
        {"tool": "close_gripper", "args": {}}, {"tool": "move_home", "args": {}},
        {"tool": "open_gripper", "args": {}}, {"tool": "move_home", "args": {}}]}
    data = _payload(server.call_tool("run_program", {"spec": spec, "task": "exercise the arm"}))
    assert runtime.arm.harness.estopped, "a cancel of the in-flight run_program latches the e-stop"
    assert data["ok"] is False and data["status"] == "stopped" and data["estop_latched"] is True
    assert data["stopped_at"] == 2
    if when == "between":
        assert executed == ["close_gripper"]
        assert len(data["steps"]) == 1 and "stopped before step 2" in data["reason"]
    else:
        assert executed == ["close_gripper", "move_home"]
        assert data["steps"][1]["verdict"] == "refused"
    assert "Do not re-run" in data["next_action"]
    assert locked and all(locked), "every step ran under the server's execution lock"
    assert [r["skill"] for r in _trace_rows(runtime) if r.get("tier") == "program"] == executed
    assert data["library"]["outcome"] == "not admitted"
    res = _payload(server.call_tool("move_home", {}))
    assert res["ok"] is False and "e-stop" in res["error"], "the latch holds until reset_stop"


@needs_pin
def test_a_latched_estop_means_the_program_never_starts(monkeypatch, tmp_path, make_server):
    _env(monkeypatch, tmp_path)
    server = make_server()
    runtime = server._ensure_runtime()
    server.stop_now()
    data = _payload(server.call_tool("run_program", {"spec": _gripper_spec(), "task": "exercise the arm"}))
    assert data["status"] == "aborted" and data["steps"] == [] and "e-stop" in data["reason"]
    assert "No motion" in data["next_action"]
    assert not [r for r in _trace_rows(runtime) if r.get("tier") == "program"]
    server.reset_now()
    data = _payload(server.call_tool("run_program", {"spec": _gripper_spec(), "task": "exercise the arm"}))
    assert data["ok"] and data["status"] == "completed"


@needs_pin
def test_bad_requests_and_unavailable_tools_are_refused_before_anything_runs(monkeypatch, tmp_path, make_server):
    _env(monkeypatch, tmp_path, extra={"CASCADE_HIDE_TOOLS": "close_gripper"})
    server = make_server()
    runtime = server._ensure_runtime()
    raw = {"name": "raw", "params": {}, "steps": [{"tool": "place_at", "args": {"x": 0.2, "y": -0.1}}]}
    cases = [
        ({"spec": raw, "task": TASK}, "coordinate"),
        ({"spec": _gripper_spec(), "program": "home-close-open", "task": TASK}, "exactly one"),
        ({"task": TASK}, "exactly one"),
        ({"spec": _gripper_spec()}, "task"),
        ({"spec": _gripper_spec(), "task": "   "}, "task"),
        ({"program": "no-such-program", "task": TASK}, "unknown"),
        ({"spec": _place_spec(), "bindings": ["red cube"], "task": TASK}, "bind"),
        ({"spec": _place_spec(), "bindings": {"object": 3}, "task": TASK}, "object"),
        ({"spec": _gripper_spec(), "task": TASK}, "close_gripper is disabled by the operator"),
    ]
    for args, why in cases:
        result = server.call_tool("run_program", args)
        data = _payload(result)
        assert result["isError"] and data["status"] == "refused", (args, data)
        assert why in data["error"], (why, data["error"])
    assert not [r for r in _trace_rows(runtime) if r.get("tier") == "program"], "nothing ran"
    # a spec may arrive JSON-encoded (chat models do that): same contract
    data = _payload(server.call_tool("run_program", {"spec": json.dumps(raw), "task": TASK}))
    assert data["status"] == "refused" and "coordinate" in data["error"]


@needs_pin
def test_an_rgb_only_rig_refuses_programs_that_need_3d_grounding(monkeypatch, tmp_path, make_server):
    """A program may not reach a tool the capability matrix withholds -- the
    runner calls execute() directly, so the MCP layer checks every step and
    the grounding query (localize_object) against the served catalog."""
    _env(monkeypatch, tmp_path, camera="mock_rgb")
    server = make_server()
    runtime = server._ensure_runtime()
    assert "localize_object" in server.withheld_tools()
    data = _payload(server.call_tool("run_program", {"spec": _place_spec(), "bindings": {"object": "red cube"},
                                                     "task": TASK}))
    assert data["status"] == "refused" and "grasp_object is not available on this rig" in data["error"]
    place_only = {"name": "place-only", "params": {"anchor": "where"}, "steps": [
        {"tool": "place_at", "args": {"$target": {"label": {"$param": "anchor"}, "offset_m": [0.0, 0.05]}}}]}
    data = _payload(server.call_tool("run_program", {"spec": place_only, "bindings": {"anchor": "bowl"},
                                                     "task": TASK}))
    assert data["status"] == "refused" and "localize_object is not available on this rig" in data["error"]
    assert not _trace_rows(runtime)


@needs_pin
def test_capability_matrix_withholds_program_tools_with_a_reason(monkeypatch, tmp_path, make_server):
    # (1) no verifier: nothing could confirm a step -> run_program withheld, list_programs offered
    _env(monkeypatch, tmp_path)
    _wrap_config(monkeypatch, lambda cfg: cfg._data.__setitem__("verify_effects", False))
    server = make_server()
    server._ensure_runtime()
    withheld = server.withheld_tools()
    assert "run_program" in withheld and "verifier" in withheld["run_program"]
    assert "list_programs" not in withheld
    names = {t["name"] for t in server.list_tools()}
    assert "list_programs" in names and "run_program" not in names
    data = _payload(server.call_tool("run_program", {"spec": _gripper_spec(), "task": TASK}))
    assert data["ok"] is False and "not available on this rig" in data["error"]


@needs_pin
def test_an_unopenable_library_withholds_both_tools(monkeypatch, tmp_path, make_server):
    _env(monkeypatch, tmp_path)
    (tmp_path / "a-directory").mkdir()
    monkeypatch.setenv("CASCADE_PROGRAMS_PATH", str(tmp_path / "a-directory"))
    server = make_server()
    server._ensure_runtime()
    matrix = server.capabilities()
    assert matrix["programs"]["available"] is False and "library" in matrix["programs"]["why"]
    withheld = server.withheld_tools()
    assert set(PROGRAM_TOOLS) <= set(withheld)
    names = {t["name"] for t in server.list_tools()}
    assert not set(PROGRAM_TOOLS) & names
    data = _payload(server.call_tool("list_programs", {}))
    assert data["ok"] is False and "not available on this rig" in data["error"]


@needs_pin
def test_list_programs_sees_a_program_promoted_by_another_session(monkeypatch, tmp_path, make_server):
    """OpenClaw runs one MCP server per session: the store is shared, so a
    program promoted elsewhere after this server loaded it is listed, and
    another one is runnable by name without listing first -- each path
    re-reads the store on its own."""
    _env(monkeypatch, tmp_path)
    server = make_server()
    server._ensure_runtime()
    assert _payload(server.call_tool("list_programs", {}))["programs"] == []
    _seed(tmp_path / "programs.jsonl", _gripper_spec(), ["close and open the gripper", "test the jaws"])
    listed = _payload(server.call_tool("list_programs", {}))
    assert [p["name"] for p in listed["programs"]] == ["home-close-open"]
    assert listed["retrieval"].startswith("all promoted")
    close_open = {"name": "close-open", "params": {}, "steps": [
        {"tool": "close_gripper", "args": {}}, {"tool": "open_gripper", "args": {}}]}
    _seed(tmp_path / "programs.jsonl", close_open, ["snap the jaws", "check the gripper"])
    data = _payload(server.call_tool("run_program", {"program": "close-open", "task": "exercise the jaws"}))
    assert data["ok"] and data["source"] == "stored", data
    listed = _payload(server.call_tool("list_programs", {}))
    assert {p["name"]: p["evidence"]["distinct_tasks"] for p in listed["programs"]} == {
        "close-open": 3, "home-close-open": 2}


@needs_pin
def test_a_completed_but_unverified_program_is_not_ok_and_not_stored(monkeypatch, tmp_path, make_server):
    """`wave` moves the arm but has no registered postcondition: the program
    completes, yet its own report is the only evidence for that step, so the
    run is UNVERIFIED -- not ok, not stored, and the host is told not to
    re-run it."""
    _env(monkeypatch, tmp_path)
    server = make_server()
    runtime = server._ensure_runtime()
    spec = {"name": "close-and-wave", "params": {}, "steps": [
        {"tool": "close_gripper", "args": {}}, {"tool": "wave", "args": {"cycles": 1}}]}
    result = server.call_tool("run_program", {"spec": spec, "task": "close the gripper and wave"})
    data = _payload(result)
    assert result["isError"] and data["ok"] is False and data["verified"] is False
    assert data["status"] == "completed" and data["verdict"] == UNVERIFIED
    assert [s["verdict"] for s in data["steps"]] == [CONFIRMED, "unchecked"]
    assert "UNVERIFIED" in data["next_action"] and "Do not re-run" in data["next_action"]
    assert data["library"]["outcome"] == "not admitted"
    assert [r["skill"] for r in _trace_rows(runtime) if r.get("tier") == "program"] == ["close_gripper", "wave"]
    assert len(_mp().ProgramLibrary(tmp_path / "programs.jsonl")) == 0


@needs_pin
def test_list_programs_ranks_by_embedding_when_a_memory_embedder_is_configured(monkeypatch, tmp_path, make_server):
    _env(monkeypatch, tmp_path, extra={"CASCADE_MEMORY_EMBEDDER": "hash"})
    _seed(tmp_path / "programs.jsonl", _place_spec(),
          ["move the blue cube to the front-left", "shift the green cube front-left"])
    server = make_server()
    server._ensure_runtime()
    listed = _payload(server.call_tool("list_programs", {"query": "shifting cubes frontwards"}))
    assert "embedding" in listed["retrieval"] and "hash" in listed["retrieval"]
    (prog,) = listed["programs"]
    assert prog["name"] == "move-front-left" and prog["similarity"] >= 0.3


def test_the_mcp_tier_switch_is_the_cli_switch(monkeypatch, tmp_path):
    """One flag for both front-ends: CASCADE_PROGRAMS beats agent.programs;
    the MCP host is a real brain, so the CLI's mock-brain exclusion does not
    apply; mobile/composed servers never get the tier."""
    from cascade.apps.mcp_server import McpSkillServer

    _env(monkeypatch, tmp_path, programs=None)
    assert McpSkillServer()._programs_on() is False                 # shipped default
    _wrap_config(monkeypatch, lambda cfg: cfg._data["agent"].__setitem__("programs", True))
    assert McpSkillServer()._programs_on() is True                  # agent.programs: true
    monkeypatch.setenv("CASCADE_PROGRAMS", "0")
    assert McpSkillServer()._programs_on() is False                 # the env kill switch wins
    monkeypatch.setenv("CASCADE_PROGRAMS", "1")
    monkeypatch.setenv("CASCADE_BASE", "mock_base")
    assert McpSkillServer()._programs_on() is False                 # arm runtimes only


# ── D. units: runner halt, matrix cell, library retrieval and sharing ───────


def test_the_runner_checks_halt_before_grounding_and_before_every_step():
    from test_programs_tier import _LedgerRuntime

    ap = _ap()
    prog = ap.Program.from_spec(_place_spec())
    rt = _LedgerRuntime()
    run = ap.run_program(prog, {"object": "red cube"}, rt, tool_log=[], halt=lambda: "the e-stop is latched")
    assert run.status == ap.ABORTED and run.steps == [] and rt.calls == []
    assert "e-stop" in run.reason and "No motion" in run.next_action

    rt = _LedgerRuntime()
    run = ap.run_program(prog, {"object": "red cube"}, rt, tool_log=[],
                         halt=lambda: "operator stop" if len(rt.calls) >= 2 else None)
    assert [c[0] for c in rt.calls] == ["localize_object", "grasp_object"], "step 2 never dispatched"
    assert run.status == ap.STOPPED and run.stopped_at == 2 and len(run.steps) == 1
    assert run.reason.startswith("stopped before step 2 place_at: operator stop")
    assert "Do not re-run" in run.next_action and "red cube" in run.next_action
    assert not run.verified

    rt = _LedgerRuntime()
    run = ap.run_program(prog, {"object": "red cube"}, rt, tool_log=[], halt=lambda: None)
    assert run.status == ap.COMPLETED and run.verified, "a quiet halt changes nothing"


def test_the_programs_capability_cell_exists_only_when_a_tier_is_attached():
    from cascade.apps.capabilities import (CAP_PROGRAMS, CAP_VERIFIER, TOOL_REQUIREMENTS,
                                           capability_matrix, format_matrix, withheld_tools)

    ap, mp = _ap(), _mp()
    assert TOOL_REQUIREMENTS["list_programs"] == (CAP_PROGRAMS,)
    assert set(TOOL_REQUIREMENTS["run_program"]) == {CAP_PROGRAMS, CAP_VERIFIER}

    def rig(**kw):
        return SimpleNamespace(backends=lambda: {}, arm_rig=None, memory=None, effects=object(), **kw)

    off = capability_matrix(rig())
    assert set(off) == PRE_B42_CAPABILITY_KEYS and "programs=" not in format_matrix(off)
    # tier off on a rig WITHOUT the verifier: the pre-tier withheld list exactly
    # (run_program needs the verifier, but a tool that is not served is not withheld)
    bare = capability_matrix(SimpleNamespace(backends=lambda: {}, arm_rig=None, memory=None))
    assert set(withheld_tools(bare)) == {"list_arms", "verify_last_action", "task_memory", "recall_memory"}
    on = capability_matrix(rig(program_tier=ap.ProgramTier(mp.ProgramLibrary(None))))
    assert on[CAP_PROGRAMS]["available"] is True and "programs=yes" in format_matrix(on)
    assert not set(PROGRAM_TOOLS) & set(withheld_tools(on))
    broken = capability_matrix(rig(program_tier=ap.ProgramTierUnavailable("IsADirectoryError: x")))
    assert broken[CAP_PROGRAMS]["available"] is False and "IsADirectoryError" in broken[CAP_PROGRAMS]["why"]
    assert set(PROGRAM_TOOLS) <= set(withheld_tools(broken))
    no_verifier = capability_matrix(SimpleNamespace(backends=lambda: {}, arm_rig=None, memory=None,
                                                    program_tier=ap.ProgramTier(mp.ProgramLibrary(None))))
    assert "run_program" in withheld_tools(no_verifier) and "list_programs" not in withheld_tools(no_verifier)


def test_embedding_retrieval_follows_the_skill_library_floor_or_guard_rule(tmp_path):
    from cascade.memory.embedder import HashEmbedder

    mp = _mp()
    lib = mp.ProgramLibrary(tmp_path / "programs.jsonl")
    _admit(lib, _place_spec(), "move the blue cube to the front-left", "r1")
    place = _admit(lib, _place_spec(), "shift the green cube front-left", "r2")     # promoted
    _admit(lib, _gripper_spec(), "close and open the gripper", "r3")
    _admit(lib, _gripper_spec(), "test the jaws", "r4")                            # promoted
    _admit(lib, _place_spec(name="nudge", offset=(0.05, 0.05)), "shifting cubes frontwards", "r5")  # candidate
    emb = HashEmbedder()
    paraphrase = "shifting cubes frontwards"   # stems shift/cub/front; no raw word in common

    assert lib.retrievable(paraphrase) == [], "keyword overlap (the default) misses an inflection"
    ranked = lib.ranked(paraphrase, embedder=emb)
    assert [r.name for r, _ in ranked] == ["move-front-left"], "the candidate never surfaces, however similar"
    assert ranked[0][1] >= emb.text_floor
    assert [r.name for r in lib.retrievable(paraphrase, embedder=emb)] == ["move-front-left"]
    # floor OR guard: an embedder never drops a keyword match ...
    assert [r.name for r in lib.retrievable("front", embedder=emb, min_sim=0.99)] == ["move-front-left"]
    # ... and below the floor without a shared word nothing qualifies
    assert lib.retrievable(paraphrase, embedder=emb, min_sim=0.99) == []
    assert lib.retrievable("wave at the audience", embedder=emb) == []
    # ranked by similarity, most similar first
    both = lib.ranked("test the jaws, then shift the green cube front-left", embedder=emb, max_entries=5)
    assert {r.name for r, _ in both} == {"move-front-left", "home-close-open"}
    assert [s for _, s in both] == sorted((s for _, s in both), reverse=True)
    # no embedder: the keyword path, unchanged, with no similarity
    assert [(r.name, s) for r, s in lib.ranked("move the cube front-left")] == [("move-front-left", None)]
    # demoted programs stay out of every ranking
    lib.record_failure(place.signature, run="f1")
    lib.record_failure(place.signature, run="f2")
    assert lib.retrievable(paraphrase, embedder=emb) == []


def test_the_authoring_turn_offers_programs_by_embedding_when_configured(tmp_path):
    """The orchestrator path: with runtime.memory.embedder the offered
    programs are ranked by embedding; without one the pre-B42 keyword call."""
    from cascade.agent.orchestrator import AgentOrchestrator
    from cascade.memory.embedder import HashEmbedder
    from test_programs_tier import _LedgerRuntime

    ap, mp = _ap(), _mp()
    lib = mp.ProgramLibrary(tmp_path / "programs.jsonl")
    _admit(lib, _place_spec(), "move the blue cube to the front-left", "r1")
    _admit(lib, _place_spec(), "shift the green cube front-left", "r2")
    prompts = {}
    for embedder in (None, HashEmbedder()):
        rt = _LedgerRuntime()
        rt.memory.embedder = embedder
        llm = MockLLM([LLMResponse(text="NONE"), LLMResponse(text="", tool_calls=[
            ToolCall("task_done", {"success": False, "summary": "nothing"})])])
        agent = AgentOrchestrator(llm, rt, advisor=None, decompose=False, attach_images=False,
                                  verify_milestones=False, max_steps=3, programs=ap.ProgramTier(lib))
        agent.run_task("shifting cubes frontwards")
        prompts[embedder is not None] = str(llm.requests[0]["messages"][0]["content"])
    assert "move-front-left" not in prompts[False], "keyword overlap stays the default"
    assert "move-front-left" in prompts[True] and "promoted" in prompts[True]


def test_two_processes_sharing_one_store_never_lose_evidence(tmp_path):
    """Two server processes (per-session MCP servers) on one programs.jsonl:
    each write re-reads the store under an advisory lock, so neither erases
    the other's records, occurrences or losses."""
    mp = _mp()
    path = tmp_path / "programs.jsonl"
    a = mp.ProgramLibrary(path)
    b = mp.ProgramLibrary(path)            # loaded before a wrote anything: stale
    rec = _admit(a, _place_spec(), "task one", "ra1")
    _admit(a, _place_spec(), "task two", "ra2")
    assert b.record_failure(rec.signature, run="rb1") is not None, "b sees a's record when it writes"
    _admit(b, _gripper_spec(), "task three", "rb2")
    _admit(a, _place_spec(), "task four", "ra3")   # a still holds its old view in memory
    fresh = mp.ProgramLibrary(path)
    place = fresh.get(rec.signature)
    assert (place.occurrences, place.losses, place.n_tasks) == (3, 1, 3)
    assert fresh.get("home-close-open") is not None, "a's write kept b's program"
    assert b.refresh() is True and b.get(rec.signature).occurrences == 3
    assert b.refresh() is False, "unchanged store: no re-read"


_WRITER = r"""
import sys, time
from pathlib import Path
from cascade.memory.programs import ProgramLibrary
store, go, tag, n = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], int(sys.argv[4])
spec = {"version": 1, "name": "home-close-open", "description": "", "params": {},
        "steps": [{"tool": "move_home", "args": {}}]}
lib = ProgramLibrary(store)   # loaded before the other writer starts: stale on purpose
(go.parent / f"ready-{tag}").touch()
while not go.exists():
    time.sleep(0.001)
for i in range(n):
    lib.admit(spec, "a" * 64, task=f"{tag} task {i % 3}", run=f"{tag}-{i}", origin="authored",
              verdict="confirmed")
"""


def test_two_concurrent_writer_processes_lose_no_update(tmp_path):
    """Two real processes admitting into one store at the same time (two
    chat sessions' servers): the advisory lock makes each read-modify-write
    atomic across processes, so every admitted run is counted. The result is
    deterministic; only the interleaving varies."""
    import os
    import subprocess
    import sys

    store, go, n = tmp_path / "programs.jsonl", tmp_path / "go", 40
    env = {**os.environ, "PYTHONPATH": str(REPO / "src"), "CUDA_VISIBLE_DEVICES": "-1"}
    procs = [subprocess.Popen([sys.executable, "-c", _WRITER, str(store), str(go), tag, str(n)], env=env)
             for tag in ("p", "q")]
    deadline = time.monotonic() + 120
    while not all((tmp_path / f"ready-{t}").exists() for t in ("p", "q")):  # both loaded the empty store
        assert time.monotonic() < deadline and all(p.poll() is None for p in procs), "a writer never got ready"
        time.sleep(0.01)
    go.touch()
    assert [p.wait(timeout=120) for p in procs] == [0, 0]
    rec = _mp().ProgramLibrary(store).get("a" * 64)
    assert rec.occurrences == 2 * n and len(rec.source_runs) == 2 * n, (rec.occurrences, 2 * n)
    assert rec.n_tasks == 6


# ── E. docs ──────────────────────────────────────────────────────────────


def test_docs_record_the_mcp_programs_tools_and_embedding_retrieval():
    from cascade.apps.mcp_server import _EXCLUDED_TOOLS, _EXTRA_TOOLS
    from cascade.skills.runtime import TOOL_SPECS

    n_tools = len({t["name"] for t in TOOL_SPECS} - _EXCLUDED_TOOLS) + len(_EXTRA_TOOLS)
    assert {t["name"] for t in _EXTRA_TOOLS} >= set(PROGRAM_TOOLS)
    roadmap = (REPO / "docs/ROADMAP.md").read_text()
    arch = (REPO / "docs/ARCHITECTURE.md").read_text()
    readme = (REPO / "README.md").read_text()
    design = (REPO / "docs/PROGRAMS_TIER.md").read_text()
    demo_yaml = (REPO / "configs/demo.yaml").read_text()
    assert "~~not exposed to MCP chat hosts~~" in roadmap and "~~retrieval is keyword overlap~~" in roadmap
    assert "landed 2026-10-09" in roadmap
    for doc in (arch, readme, design):
        assert "list_programs" in doc and "run_program" in doc
    assert f"{n_tools} tools total" in readme and f"{n_tools} tools" in arch
    assert "embedding" in design.lower() and "floor" in design.lower()
    assert "list_programs" in demo_yaml
