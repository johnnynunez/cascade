"""CASCADE_HUG_PORT and the sidecar host variables join B34's contract (backlog B41).

B34 (#259) made load_demo_config apply CASCADE_BRIDGE_PORT / _GRASPGENX_PORT /
_OCCUPANCY_PORT last. Three endpoint variables were still read only when a planner was
CONSTRUCTED -- CASCADE_GRASPGENX_HOST, CASCADE_HUG_HOST and CASCADE_HUG_PORT -- parsed
with `int()` or taken verbatim, the variable beating whatever the config said; and
scripts/setup_agents.py registered the MCP server with none of the endpoint variables.
So on 4e896c3:

  - a resolved config did not say where the HUG or GraspGen-X client would dial;
  - a malformed or empty value was accepted ('' became the host, ' 45318' a port);
  - an explicit override written into `cfg._data` after loading (the sanctioned runtime
    idiom; benchmark/diagnostics write `--grasp-port` that way) was silently undone at
    construction whenever the variable was exported -- the variable applied twice;
  - a host registered with setup_agents.py dialled the default ports.

Now load_demo_config applies all six variables last (empty = unset; a malformed port or
host raises naming the variable); a planner takes its endpoint from its section and
falls back to the same variables only for what the section lacks (so a planner built
from a hand-made config still honours them); setup_agents.py copies every
CASCADE_*_PORT / CASCADE_*_HOST present into each server entry; scripts/serve_hug.py
reads CASCADE_HUG_PORT by the client's rule.

The bridge and the occupancy sidecar get NO host variable, on purpose: launch.sh starts
both on this machine and waits on them there (and the bridge, whose `exec` op is
arbitrary code execution, binds loopback by default), so their hosts stay
profile/demo.yaml settings.

No test opens a socket: every dial is intercepted by a spy first. Ports are from this
item's block (45300-45399); hosts use the reserved `.invalid` top-level domain.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import sys
import types
from pathlib import Path

import pytest

from conftest import REPO
from cascade.config import Cfg, load_demo_config, load_profile, load_robot_config
from test_port_env_overrides import _registered_env

sys.path.insert(0, str(REPO / "scripts"))
import setup_agents  # noqa: E402

CONTRACT = ("CASCADE_BRIDGE_PORT", "CASCADE_GRASPGENX_PORT", "CASCADE_OCCUPANCY_PORT",
            "CASCADE_HUG_PORT", "CASCADE_GRASPGENX_HOST", "CASCADE_HUG_HOST")
HOST_VARS = ("CASCADE_GRASPGENX_HOST", "CASCADE_HUG_HOST")
PRIVATE = {"CASCADE_BRIDGE_PORT": "45311", "CASCADE_GRASPGENX_PORT": "45316",
           "CASCADE_OCCUPANCY_PORT": "45317", "CASCADE_HUG_PORT": "45318",
           "CASCADE_GRASPGENX_HOST": "ggx.b41.invalid", "CASCADE_HUG_HOST": "hug.b41.invalid"}
#: the config key each variable owns (see `_locations`)
KIND = {"CASCADE_BRIDGE_PORT": "bridge_port", "CASCADE_GRASPGENX_PORT": "graspgenx.port",
        "CASCADE_OCCUPANCY_PORT": "occupancy.port", "CASCADE_HUG_PORT": "hug.port",
        "CASCADE_GRASPGENX_HOST": "graspgenx.host", "CASCADE_HUG_HOST": "hug.host"}
# The Isaac kitchen presenter rig plus two more arms: three `resolved` views built from
# different `overrides:` (isaac_kitchen_hug selects the HUG backend).
KITCHEN = dict(cameras=["isaac", "isaac_side", "isaac_proof", "isaac_wrist"],
               arms=["isaac_kitchen_gpu", "isaac_cumotion", "isaac_kitchen_hug"], llm="mock")
_VIEWS = ("", "arms.0.resolved.", "arms.1.resolved.", "arms.2.resolved.")
#: locations the walker must find for each kind: the top level and every arm's view
REQUIRED = {
    "bridge_port": {"camera.bridge_port", "cameras.3.bridge_port", "arm.bridge_port",
                    "arms.2.bridge_port", "arms.1.resolved.cameras.2.bridge_port"},
    **{kind: {view + path for view in _VIEWS} for kind, path in (
        ("graspgenx.port", "grasp.graspgenx.port"), ("graspgenx.host", "grasp.graspgenx.host"),
        ("hug.port", "grasp.hug.port"), ("hug.host", "grasp.hug.host"),
        ("occupancy.port", "occupancy.port"))},
}
ENDPOINT_NAME = re.compile(r"CASCADE_[A-Z0-9_]+_(?:PORT|HOST)")
MALFORMED_PORTS = ["abc", "0", "65536", "70000", "-1", "+45318", " 45318", "45318 ", "45_318",
                   "45318.0", "0x1F", "1e3", "  ",
                   "\u0664\u0665\u0663\u0661\u0668",   # Arabic-Indic digits: int() accepts them
                   "\uff14\uff15\uff13\uff11\uff18"]   # fullwidth digits: int() accepts them
BAD_HOSTS = [" ggx", "ggx ", "g gx", "ggx\n", "\t", "ggx:45316", "tcp://ggx", "ggx/grasp",
             "user@ggx", "::1", "[::1]", "-ggx", ".ggx", "_ggx", "ggx,hug", 'gg"x', "gg\\x",
             "gg\u00e9x", "\u0661\u0662\u0667.0.0.1", "x" * 254]
GOOD_HOSTS = ["localhost", "127.0.0.2", "10.0.0.7", "gx10", "spark-01.local", "ggx_server",
              "a", "host.example.com.", "x" * 253]


@pytest.fixture(autouse=True)
def _clean_environment(monkeypatch):
    for key in list(os.environ):
        if ENDPOINT_NAME.fullmatch(key) or key.startswith("CASCADE_MICRODUCK_") or key in {
                "CASCADE_ROBOT", "CASCADE_BASE", "CASCADE_BOOTH", "CASCADE_GRASP_BACKEND",
                "CASCADE_OCCUPANCY"}:
            monkeypatch.delenv(key, raising=False)


def _private(monkeypatch, values=PRIVATE):
    for key, value in values.items():
        monkeypatch.setenv(key, value)


def _value(name):
    return int(PRIVATE[name]) if name.endswith("_PORT") else PRIVATE[name]


def _kitchen():
    return load_demo_config(**KITCHEN)


def _locations(tree):
    """[(kind, path)] of every endpoint key a client reads, path as a key tuple."""
    found = []

    def walk(node, path):
        if isinstance(node, dict):
            for key, value in node.items():
                if key in ("bridge_port", "bridge_host"):
                    found.append((key, (*path, key)))
                elif key in ("port", "host") and path[-1:] in (("graspgenx",), ("hug",),
                                                                ("occupancy",)):
                    found.append((f"{path[-1]}.{key}", (*path, key)))
                walk(value, (*path, str(key)))
        elif isinstance(node, list):
            for i, value in enumerate(node):
                walk(value, (*path, str(i)))

    walk(tree, ())
    return found


def _flat(tree, path=()):
    """{path: leaf} of a resolved config, lists included."""
    if isinstance(tree, dict):
        items = ((str(k), v) for k, v in tree.items())
    elif isinstance(tree, list):
        items = ((str(i), v) for i, v in enumerate(tree))
    else:
        return {path: tree}
    out = {}
    for key, value in items:
        out.update(_flat(value, (*path, key)))
    return out


def test_the_fixture_ports_are_inside_this_items_block():
    ports = [int(v) for k, v in PRIVATE.items() if k.endswith("_PORT")]
    assert ports and all(45300 <= p <= 45399 for p in ports)


# ── the resolved config ───────────────────────────────────────────────────


@pytest.mark.parametrize("name", CONTRACT)
def test_each_variable_moves_exactly_its_own_endpoint_in_every_view(monkeypatch, name):
    unset = _kitchen().as_dict()
    owned = {path for kind, path in _locations(unset) if kind == KIND[name]}
    assert REQUIRED[KIND[name]] <= {".".join(p) for p in owned}
    _private(monkeypatch, {name: PRIVATE[name]})
    moved = _kitchen().as_dict()
    before, after = _flat(unset), _flat(moved)
    assert set(after) == set(before), "an override moves a value; it never adds or drops a key"
    assert {path for path in before if after[path] != before[path]} == owned
    assert {after[path] for path in owned} == {_value(name)}


def test_all_six_at_once_each_location_holds_its_own_variable(monkeypatch):
    _private(monkeypatch)
    resolved = _kitchen().as_dict()
    leaves, found = _flat(resolved), _locations(resolved)
    by_kind = {kind: {leaves[path] for k, path in found if k == kind} for kind in KIND.values()}
    assert by_kind == {KIND[name]: {_value(name)} for name in CONTRACT}
    # No host variable for the bridge or the occupancy sidecar (see the module docstring).
    assert {leaves[path] for k, path in found
            if k in ("bridge_host", "occupancy.host")} == {"127.0.0.1"}


@pytest.fixture
def layered(tmp_path):
    """demo.yaml < booth.yaml < arm `overrides:` all name sidecar endpoints."""
    for kind in ("cameras", "arms", "llm"):
        (tmp_path / kind).mkdir()
    (tmp_path / "demo.yaml").write_text(
        "grasp:\n"
        "  graspgenx:\n    host: 127.0.0.1\n    port: 5556\n"
        "  hug:\n    host: 127.0.0.1\n    port: 5558\n"
        "occupancy:\n  host: 127.0.0.1\n  port: 5557\n")
    (tmp_path / "booth.yaml").write_text("grasp:\n  hug:\n    host: booth-hug.invalid\n")
    (tmp_path / "cameras" / "usb.yaml").write_text("type: uvc\n")
    (tmp_path / "arms" / "a.yaml").write_text(
        "type: mock\noverrides:\n  grasp:\n"
        "    graspgenx:\n      host: arm-ggx.invalid\n"
        "    hug:\n      port: 7002\n")
    (tmp_path / "arms" / "b.yaml").write_text("type: mock\n")
    (tmp_path / "llm" / "mock.yaml").write_text("type: mock\n")
    return tmp_path


def _sidecars(view):
    grasp = view["grasp"]
    return {"graspgenx": (grasp["graspgenx"]["host"], grasp["graspgenx"]["port"]),
            "hug": (grasp["hug"]["host"], grasp["hug"]["port"]),
            "occupancy": (view["occupancy"]["host"], view["occupancy"]["port"])}


def test_env_beats_demo_booth_and_arm_overrides_in_every_view(layered, monkeypatch):
    monkeypatch.setenv("CASCADE_BOOTH", "1")

    def load():
        return load_demo_config(camera="usb", arms=["a", "b"], config_dir=layered).as_dict()

    before = load()
    # premise: each layer wins where it should, and arm a's retuning stays in arm a's view
    assert _sidecars(before) == _sidecars(before["arms"][0]["resolved"]) == {
        "graspgenx": ("arm-ggx.invalid", 5556), "hug": ("booth-hug.invalid", 7002),
        "occupancy": ("127.0.0.1", 5557)}
    assert _sidecars(before["arms"][1]["resolved"]) == {
        "graspgenx": ("127.0.0.1", 5556), "hug": ("booth-hug.invalid", 5558),
        "occupancy": ("127.0.0.1", 5557)}
    _private(monkeypatch)
    after = load()
    for view in (after, *(arm["resolved"] for arm in after["arms"])):
        assert _sidecars(view) == {"graspgenx": ("ggx.b41.invalid", 45316),
                                   "hug": ("hug.b41.invalid", 45318),
                                   "occupancy": ("127.0.0.1", 45317)}


def test_no_sidecar_section_is_ever_invented(tmp_path, monkeypatch):
    for kind in ("cameras", "arms", "llm"):
        (tmp_path / kind).mkdir()
    (tmp_path / "demo.yaml").write_text("grasp:\n  backend: obb\n")
    (tmp_path / "cameras" / "usb.yaml").write_text("type: uvc\n")
    (tmp_path / "arms" / "m.yaml").write_text("type: mock\n")
    (tmp_path / "llm" / "mock.yaml").write_text("type: mock\n")
    _private(monkeypatch)
    cfg = load_demo_config(camera="usb", arm="m", config_dir=tmp_path).as_dict()
    for view in (cfg, cfg["arms"][0]["resolved"]):
        assert view["grasp"] == {"backend": "obb"} and "occupancy" not in view


@pytest.mark.parametrize("name", HOST_VARS)
@pytest.mark.parametrize("value", BAD_HOSTS, ids=ascii)
def test_a_malformed_host_is_refused_naming_the_variable(monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match=name):
        _kitchen()
    # ...for a mock-only configuration too: the variable is wrong whatever the profile.
    with pytest.raises(ValueError, match=name):
        load_demo_config()


@pytest.mark.parametrize("value", GOOD_HOSTS)
def test_hostnames_and_ipv4_addresses_are_taken_verbatim(monkeypatch, value):
    _private(monkeypatch, {name: value for name in HOST_VARS})
    cfg = load_demo_config().as_dict()
    for view in (cfg, cfg["arms"][0]["resolved"]):
        assert view["grasp"]["graspgenx"]["host"] == view["grasp"]["hug"]["host"] == value


@pytest.mark.parametrize("value", MALFORMED_PORTS, ids=ascii)
def test_a_malformed_hug_port_is_refused_naming_the_variable(monkeypatch, value):
    monkeypatch.setenv("CASCADE_HUG_PORT", value)
    with pytest.raises(ValueError, match="CASCADE_HUG_PORT"):
        _kitchen()
    with pytest.raises(ValueError, match="CASCADE_HUG_PORT"):
        load_demo_config()


@pytest.mark.parametrize("value", ["1", "65535"])
def test_the_edge_hug_ports_are_accepted(monkeypatch, value):
    monkeypatch.setenv("CASCADE_HUG_PORT", value)
    resolved = _kitchen().as_dict()
    leaves = _flat(resolved)
    assert {leaves[path] for kind, path in _locations(resolved) if kind == "hug.port"} == {int(value)}


def test_empty_means_unset_for_every_variable(monkeypatch):
    # Golden pin (true on 4e896c3 too): empty = unset, like launch.sh's ${VAR:-default}.
    unset = _kitchen().as_dict()
    _private(monkeypatch, {name: "" for name in CONTRACT})
    assert _kitchen().as_dict() == unset


def test_composed_manipulation_gets_them_and_bases_never_do(monkeypatch):
    composed_unset = load_robot_config("mobile_manipulator_mock").as_dict()["domains"]
    base_unset = load_demo_config(base="microduck_isaac").as_dict()
    _private(monkeypatch)
    composed = load_robot_config("mobile_manipulator_mock").as_dict()["domains"]
    manipulation = composed["manipulation"]["resolved"]
    for view in (manipulation, *(arm["resolved"] for arm in manipulation["arms"])):
        assert view["grasp"]["graspgenx"]["host"] == "ggx.b41.invalid"
        assert (view["grasp"]["hug"]["host"], view["grasp"]["hug"]["port"]) == (
            "hug.b41.invalid", 45318)
    assert composed["locomotion"] == composed_unset["locomotion"]
    # Mobile bases have their own explicit variables; the sidecar ones never reach them.
    assert load_demo_config(base="microduck_isaac").as_dict() == base_unset


@pytest.mark.parametrize("kind, name", [("arms", "isaac_kitchen_hug"), ("arms", "isaac_kitchen_gpu"),
                                        ("cameras", "isaac_proof"), ("arms", "so101_mock"),
                                        ("cameras", "mock")])
def test_load_profile_moves_only_the_bridge_port(monkeypatch, kind, name):
    """`load_profile` serves the standalone viewer/recorder, which dial only the bridge;
    a profile's `overrides:` (isaac_kitchen_hug names `grasp.hug`) is resolved by
    load_demo_config, which then applies the variables over it."""
    expected = load_profile(kind, name).as_dict()
    if expected.get("type") == "isaac":
        expected["bridge_port"] = 45311
    _private(monkeypatch)
    assert load_profile(kind, name).as_dict() == expected


# ── the dial itself ───────────────────────────────────────────────────────


class _Dialled(Exception):
    pass


@pytest.fixture
def zmq_spy(monkeypatch):
    """Stand-in zmq/msgpack: record every REQ `connect` endpoint, open nothing (faked
    rather than importorskip'd so the minimal install asserts the dial too)."""
    endpoints = []

    class Socket:
        def setsockopt(self, *args):
            pass

        def connect(self, endpoint):
            endpoints.append(endpoint)
            raise _Dialled(endpoint)

        def close(self, *args, **kwargs):
            pass

    class Context:
        @staticmethod
        def instance():
            return Context()

        def socket(self, kind):
            return Socket()

    zmq = types.ModuleType("zmq")
    zmq.Context = Context
    zmq.REQ, zmq.RCVTIMEO, zmq.SNDTIMEO, zmq.LINGER, zmq.POLLIN, zmq.POLLOUT, zmq.NOBLOCK = range(7)
    zmq.error = types.SimpleNamespace(Again=type("Again", (Exception,), {}))
    zmq.Again = zmq.error.Again
    msgpack_numpy = types.ModuleType("msgpack_numpy")
    msgpack_numpy.patch = lambda: None
    for module_name, module in (("zmq", zmq), ("msgpack", types.ModuleType("msgpack")),
                                ("msgpack_numpy", msgpack_numpy)):
        monkeypatch.setitem(sys.modules, module_name, module)
    return endpoints


def _dial(planner):
    with pytest.raises(_Dialled):
        planner.probe()


def _ggx(cfg):
    from cascade.grasping.graspgenx_backend import GraspGenXPlanner

    return GraspGenXPlanner(cfg)


def _hug(cfg):
    from cascade.grasping.hug_backend import HugPlanner

    return HugPlanner(cfg)


def test_planners_dial_the_endpoint_every_resolved_view_carries(monkeypatch, zmq_spy):
    _private(monkeypatch)
    cfg = _kitchen()
    views = [cfg.grasp, *(Cfg(arm["resolved"]).grasp for arm in cfg.arms)]
    for name in CONTRACT:  # the resolved config must carry the endpoint on its own
        monkeypatch.delenv(name)
    for grasp in views:
        _dial(_ggx(grasp))
        _dial(_hug(grasp))
    assert zmq_spy == ["tcp://ggx.b41.invalid:45316", "tcp://hug.b41.invalid:45318"] * len(views)


def test_an_override_written_after_loading_is_not_undone_at_construction(monkeypatch, zmq_spy):
    """The variables are applied ONCE, by load_demo_config. Re-reading them when the
    planner is built (4e896c3) silently undid the sanctioned runtime override."""
    _private(monkeypatch)
    cfg = _kitchen()
    cfg._data["grasp"]["graspgenx"].update(host="ggx-override.b41.invalid", port=45326)
    cfg._data["grasp"]["hug"].update(host="hug-override.b41.invalid", port=45328)
    _dial(_ggx(cfg.grasp))
    _dial(_hug(cfg.grasp))
    assert zmq_spy == ["tcp://ggx-override.b41.invalid:45326", "tcp://hug-override.b41.invalid:45328"]


@pytest.mark.parametrize("section", [{}, {"timeout_ms": 400}, None], ids=["empty", "partial", "absent"])
def test_a_planner_built_without_load_demo_config_still_honours_the_variables(
        monkeypatch, zmq_spy, section):
    # Backward compatibility: true on 4e896c3 too.
    _private(monkeypatch)
    _dial(_ggx(Cfg({} if section is None else {"graspgenx": dict(section)})))
    _dial(_hug(Cfg({} if section is None else {"hug": dict(section)})))
    assert zmq_spy == ["tcp://ggx.b41.invalid:45316", "tcp://hug.b41.invalid:45318"]


def test_a_section_that_names_its_endpoint_keeps_it(monkeypatch, zmq_spy):
    """A key the section carries is final: a variable fills only what is missing, so the
    two halves of one endpoint can come from different places."""
    _private(monkeypatch)
    _dial(_ggx(Cfg({"graspgenx": {"host": "own.b41.invalid", "port": 45336}})))
    _dial(_hug(Cfg({"hug": {"port": 45338}})))
    _dial(_ggx(Cfg({"graspgenx": {"host": "own.b41.invalid"}})))
    assert zmq_spy == ["tcp://own.b41.invalid:45336", "tcp://hug.b41.invalid:45338",
                       "tcp://own.b41.invalid:45316"]


@pytest.mark.parametrize("name, value", [
    ("CASCADE_HUG_PORT", " 45318"), ("CASCADE_HUG_PORT", "+45318"), ("CASCADE_HUG_PORT", "0"),
    ("CASCADE_GRASPGENX_PORT", "45_316"), ("CASCADE_HUG_HOST", "hug:45318"),
    ("CASCADE_GRASPGENX_HOST", " ggx.b41.invalid")])
@pytest.mark.parametrize("named", [False, True], ids=["section-lacks-it", "section-names-it"])
def test_a_planner_built_directly_refuses_a_malformed_variable(monkeypatch, name, value, named):
    # Same rule as load_demo_config, even when the section makes the variable moot.
    monkeypatch.setenv(name, value)
    section = {"host": "own.b41.invalid", "port": 45336} if named else {}
    build = _hug if "_HUG_" in name else _ggx
    with pytest.raises(ValueError, match=name):
        build(Cfg({"hug" if build is _hug else "graspgenx": section}))


def test_empty_variables_are_unset_for_a_planner_built_directly(monkeypatch, zmq_spy):
    _private(monkeypatch, {name: "" for name in CONTRACT})
    _dial(_ggx(Cfg({"graspgenx": {"port": 45336}})))
    _dial(_hug(Cfg({"hug": {"host": "own.b41.invalid"}})))
    assert zmq_spy == ["tcp://127.0.0.1:45336", "tcp://own.b41.invalid:5558"]
    # ...and the other half of each default, read off the client (nothing dialled)
    ggx, hug = _ggx(Cfg({"graspgenx": {}}))._client, _hug(Cfg({"hug": {}}))._client
    assert [(ggx._host, ggx._port), (hug._host, hug._port)] == [("127.0.0.1", 5556), ("127.0.0.1", 5558)]


def test_the_host_variables_never_reach_the_bridge_or_occupancy_clients(monkeypatch, zmq_spy):
    from cascade.perception.camera_base import CameraError, make_camera
    from cascade.perception.occupancy import OccupancyMap
    from cascade.sim import bridge_client

    _private(monkeypatch)
    cfg = _kitchen()
    dialled = []

    def spy(address, *args, **kwargs):
        dialled.append(tuple(address))
        raise ConnectionRefusedError("spy: no socket is opened in this test")

    monkeypatch.setattr(bridge_client.socket, "create_connection", spy)
    with pytest.raises(CameraError):
        make_camera(Cfg(cfg.cameras[0])).open()
    with pytest.raises(_Dialled):
        OccupancyMap.from_config(cfg.get("occupancy"), workspace_min=[0.0, -0.5, 0.0],
                                 workspace_max=[0.6, 0.5, 0.5])
    assert dialled == [("127.0.0.1", 45311)]
    assert zmq_spy == ["tcp://127.0.0.1:45317"]


# ── scripts/setup_agents.py hands them to every host ──────────────────────

PY = "/usr/bin/python3"
#: every CASCADE_*_PORT / CASCADE_*_HOST is forwarded, not only the six of the contract
FORWARDED = {**PRIVATE, "CASCADE_STREAM_PORT": "45390", "CASCADE_MICRODUCK_BRIDGE_PORT": "45391"}


def _flag_env(command: str) -> dict:
    tokens = shlex.split(command)
    return dict(tokens[i + 1].split("=", 1) for i, token in enumerate(tokens) if token == "--env")


def _endpoint_entries(env: dict) -> dict:
    return {key: value for key, value in env.items() if ENDPOINT_NAME.fullmatch(key)}


def _generated(env: dict) -> dict:
    """The env each generator hands the server, parsed back from its text."""
    import tomllib

    return {
        "codex toml": tomllib.loads(setup_agents.codex_toml_block(PY, env))
        ["mcp_servers"]["cascade"]["env"],
        "claude .mcp.json": json.loads(setup_agents.claude_mcp_json(None, PY, env))
        ["mcpServers"]["cascade"]["env"],
        "claude mcp add": _flag_env(setup_agents.claude_add_command(PY, env)),
        "openclaw mcp add": _flag_env(setup_agents.openclaw_command(PY, env)),
        "mcporter json": json.loads(setup_agents.openclaw_json_block(PY, env))
        ["mcpServers"]["cascade"]["env"],
    }


def _main(monkeypatch, tmp_path, *argv):
    """setup_agents.main() against a private HOME and CODEX_HOME (never the real ones)."""
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "home"))
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "codex"))
    monkeypatch.setattr(sys, "argv", ["setup_agents.py", "--camera", "isaac",
                                      "--arm", "isaac_kitchen_gpu", "--python", PY, *argv])
    return setup_agents.main()


def _written(tmp_path) -> dict:
    import tomllib

    import yaml

    hermes = yaml.safe_load((tmp_path / "home" / ".hermes" / "config.yaml").read_text())
    codex = tomllib.loads((tmp_path / "codex" / "config.toml").read_text())
    return {"hermes yaml": hermes["mcp_servers"]["cascade"]["env"],
            "codex config.toml": codex["mcp_servers"]["cascade"]["env"]}


def test_setup_agents_forwards_every_port_and_host_variable_into_every_entry(
        monkeypatch, tmp_path, capsys):
    _private(monkeypatch, FORWARDED)
    monkeypatch.setenv("UNRELATED_CREDENTIAL", "must-not-forward")
    env = setup_agents.server_env("isaac", "isaac_kitchen_gpu", ":1")
    assert _endpoint_entries(env) == FORWARDED and "UNRELATED_CREDENTIAL" not in env
    for host, got in _generated(env).items():
        assert _endpoint_entries(got) == FORWARDED, host
    # through main(): the two hosts it writes, and the one-liners it prints
    for host in ("hermes", "codex"):
        assert _main(monkeypatch, tmp_path, "--host", host, "--write") == 0
    for host, got in _written(tmp_path).items():
        assert _endpoint_entries(got) == FORWARDED, host
    capsys.readouterr()
    assert _main(monkeypatch, tmp_path, "--host", "openclaw") == 0
    assert _main(monkeypatch, tmp_path, "--host", "claude") == 0
    lines = capsys.readouterr().out.splitlines()
    for prefix in ("openclaw mcp add ", "claude mcp add "):
        [line] = [ln for ln in lines if ln.startswith(prefix)]
        assert _endpoint_entries(_flag_env(line)) == FORWARDED, prefix
    # the operator is told what was copied from this shell
    assert any("CASCADE_HUG_PORT=45318" in ln and "forward" in ln for ln in lines)


def test_setup_agents_invents_nothing_when_no_variable_is_set(monkeypatch, tmp_path):
    # Golden pin (true on 4e896c3 too): no variable set, no endpoint entry anywhere.
    env = setup_agents.server_env("isaac", "isaac_kitchen_gpu", ":1")
    assert _endpoint_entries(env) == {}
    for host, got in _generated(env).items():
        assert _endpoint_entries(got) == {}, host
    for host in ("hermes", "codex"):
        assert _main(monkeypatch, tmp_path, "--host", host, "--write") == 0
    for host, got in _written(tmp_path).items():
        assert _endpoint_entries(got) == {}, host


@pytest.mark.parametrize("name, value", [
    ("CASCADE_HUG_PORT", "abc"), ("CASCADE_BRIDGE_PORT", "0"), ("CASCADE_OCCUPANCY_PORT", " 45317"),
    ("CASCADE_GRASPGENX_HOST", "ggx:45316"), ("CASCADE_HUG_HOST", 'hug"x'),
    ("CASCADE_STREAM_PORT", "45 390")])
def test_setup_agents_refuses_a_value_the_runtime_would_refuse(
        monkeypatch, tmp_path, capsys, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match=name):
        setup_agents.server_env("isaac", "isaac_kitchen_gpu", ":1")
    with pytest.raises(SystemExit) as exc:
        _main(monkeypatch, tmp_path, "--host", "hermes", "--write")
    assert exc.value.code == 2 and name in capsys.readouterr().err
    assert not (tmp_path / "home" / ".hermes" / "config.yaml").exists(), "nothing written"


def test_an_explicit_env_flag_beats_the_inherited_value(monkeypatch):
    # The same precedence as the device variables (test_mcp_device_environment.py).
    monkeypatch.setenv("CASCADE_BRIDGE_PORT", "45311")
    server_env = setup_agents.server_env
    assert server_env("mock", "mock", ":1", extra=["CASCADE_BRIDGE_PORT=45312"])[
        "CASCADE_BRIDGE_PORT"] == "45312"
    assert server_env("mock", "mock", ":1", extra=["CASCADE_BRIDGE_PORT="])["CASCADE_BRIDGE_PORT"] == ""
    # An inherited value the operator overrides is not consulted at all.
    monkeypatch.setenv("CASCADE_HUG_PORT", "not-a-port")
    assert server_env("mock", "mock", ":1", extra=["CASCADE_HUG_PORT=45318"])[
        "CASCADE_HUG_PORT"] == "45318"


def test_an_empty_variable_is_forwarded_empty_as_launch_sh_does(monkeypatch, capsys):
    monkeypatch.setenv("CASCADE_BRIDGE_PORT", "")
    assert setup_agents.server_env("mock", "mock", ":1")["CASCADE_BRIDGE_PORT"] == ""
    assert _registered_env(monkeypatch, capsys)["CASCADE_BRIDGE_PORT"] == ""


def test_with_no_endpoint_variable_the_registrar_needs_no_cascade_import(monkeypatch):
    # Premise: a bare python3 (no PyYAML, so no cascade.config) registers as before.
    monkeypatch.setitem(sys.modules, "cascade.config", None)
    env = setup_agents.server_env("mock", "mock", ":1")
    assert not [k for k in env if ENDPOINT_NAME.fullmatch(k)]


def test_a_value_this_interpreter_cannot_check_is_refused_not_written(monkeypatch, tmp_path, capsys):
    # With a variable set and cascade.config unimportable (a bare python3 without
    # PyYAML), the registrar refuses rather than write a value nobody checked into
    # ~/.hermes or ~/.codex; an explicit --env still passes one as given.
    monkeypatch.setitem(sys.modules, "cascade.config", None)
    monkeypatch.setenv("CASCADE_HUG_PORT", "45318")
    with pytest.raises(ValueError, match=r"CASCADE_HUG_PORT.*--env"):
        setup_agents.server_env("mock", "mock", ":1")
    with pytest.raises(SystemExit) as exc:
        _main(monkeypatch, tmp_path, "--host", "hermes", "--write")
    assert exc.value.code == 2 and "CASCADE_HUG_PORT" in capsys.readouterr().err
    assert not (tmp_path / "home" / ".hermes" / "config.yaml").exists(), "nothing written"
    assert setup_agents.server_env("mock", "mock", ":1", extra=["CASCADE_HUG_PORT=45318"])[
        "CASCADE_HUG_PORT"] == "45318"


def test_the_launcher_and_the_registrar_forward_the_whole_contract(monkeypatch, capsys):
    from cascade.config import ENDPOINT_ENV_VARS

    assert sorted(ENDPOINT_ENV_VARS) == sorted(CONTRACT)
    _private(monkeypatch)
    launched = _registered_env(monkeypatch, capsys)
    registered = setup_agents.server_env("isaac", "isaac_kitchen_gpu", ":1")
    assert {name: launched.get(name) for name in CONTRACT} == PRIVATE
    assert {name: registered.get(name) for name in CONTRACT} == PRIVATE


# ── scripts/serve_hug.py binds what the client dials ──────────────────────


def _serve_module():
    sys.path.insert(0, str(REPO / "scripts"))
    try:
        import serve_hug
    finally:
        sys.path.remove(str(REPO / "scripts"))
    return serve_hug


def _outcome(fn):
    try:
        return ("ok", fn())
    except ValueError as exc:
        return ("refused", "CASCADE_HUG_PORT" in str(exc))


def test_serve_hug_reads_its_port_variable_by_the_clients_rule(monkeypatch):
    from cascade.config import env_port

    serve = _serve_module()
    for raw in ["", "1", "5558", "45318", "65535", *MALFORMED_PORTS]:
        monkeypatch.setenv("CASCADE_HUG_PORT", raw)
        client, server = _outcome(lambda: env_port("CASCADE_HUG_PORT")), _outcome(serve.port_from_env)
        if raw == "":
            assert (client, server) == (("ok", None), ("ok", serve.DEFAULT_PORT))
        else:
            assert server == client, ascii(raw)
            assert client == (("ok", int(raw)) if client[0] == "ok" else ("refused", True)), ascii(raw)


@pytest.mark.parametrize("raw, port", [("45318", 45318), ("", 5558)], ids=["set", "empty"])
def test_serve_hug_binds_the_port_the_variable_names(monkeypatch, raw, port):
    serve = _serve_module()
    bound = []
    monkeypatch.setattr(serve, "serve", lambda engine, host, port, verbose=True: bound.append(port))
    monkeypatch.setenv("CASCADE_HUG_PORT", raw)
    assert serve.main(["--stub", "--quiet"]) == 0
    assert bound == [port]


def test_serve_hug_refuses_a_malformed_variable_unless_port_is_explicit(monkeypatch, capsys):
    serve = _serve_module()
    bound = []
    monkeypatch.setattr(serve, "serve", lambda engine, host, port, verbose=True: bound.append(port))
    monkeypatch.setenv("CASCADE_HUG_PORT", " 45318")
    with pytest.raises(SystemExit) as exc:
        serve.main(["--stub", "--quiet"])
    assert exc.value.code == 2 and "CASCADE_HUG_PORT" in capsys.readouterr().err
    monkeypatch.setenv("CASCADE_HUG_PORT", "abc")
    assert serve.main(["--stub", "--quiet", "--port", "45319"]) == 0
    assert bound == [45319]
