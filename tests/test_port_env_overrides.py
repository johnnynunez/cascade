"""CASCADE_BRIDGE_PORT / _GRASPGENX_PORT / _OCCUPANCY_PORT reach the runtime (backlog B34).

scripts/launch.sh starts the Isaac bridge and the GraspGen-X and occupancy sidecars on
these ports. Before B34 only GraspGenXPlanner read its variable: the bridge clients and
the occupancy client dialled the profile / demo.yaml values (8611 / 5557), so a launch on
private ports left the runtime talking to whatever held the defaults
(docs/LOCAL_RTX_VALIDATION.md, profiling attempt 07). load_demo_config now applies the
three variables LAST, over every layer and every arm's `resolved` view; unset or empty
keeps the configured port; a malformed value raises naming the variable.

No test here opens a socket: every dial is intercepted by a spy before it connects. The
ports are from this item's block (44000-44099).
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import types

import pytest

from conftest import REPO
from cascade.config import Cfg, load_demo_config, load_profile, load_robot_config

LAUNCH = REPO / "scripts/launch.sh"
PORT_VARS = ("CASCADE_BRIDGE_PORT", "CASCADE_GRASPGENX_PORT", "CASCADE_OCCUPANCY_PORT")
PRIVATE = {"CASCADE_BRIDGE_PORT": "44011", "CASCADE_GRASPGENX_PORT": "44016",
           "CASCADE_OCCUPANCY_PORT": "44017"}
KINDS = ("bridge", "graspgenx", "occupancy")
# The Isaac kitchen presenter rig, proof camera included, plus a second Isaac arm so
# there are two `resolved` views built from different `overrides:`.
KITCHEN = dict(cameras=["isaac", "isaac_side", "isaac_proof", "isaac_wrist"],
               arms=["isaac_kitchen_gpu", "isaac_cumotion"], llm="mock")
# Locations that are dialled at runtime; the walker must find each of them.
REQUIRED = {
    "bridge": {"camera", "cameras.0", "cameras.3", "arm", "arms.0", "arms.1",
               "arms.0.resolved.arm", "arms.1.resolved.arm", "arms.0.resolved.camera",
               "arms.1.resolved.cameras.2"},
    "graspgenx": {"grasp.graspgenx", "arms.0.resolved.grasp.graspgenx",
                  "arms.1.resolved.grasp.graspgenx"},
    "occupancy": {"occupancy", "arms.0.resolved.occupancy", "arms.1.resolved.occupancy"},
}


@pytest.fixture(autouse=True)
def _clean_environment(monkeypatch):
    for key in list(os.environ):
        if key in PORT_VARS or key.startswith("CASCADE_MICRODUCK_") or key in {
                "CASCADE_ROBOT", "CASCADE_BASE", "CASCADE_BOOTH", "CASCADE_GRASP_BACKEND",
                "CASCADE_OCCUPANCY", "CASCADE_GRASPGENX_HOST"}:
            monkeypatch.delenv(key, raising=False)


def _private(monkeypatch, ports=PRIVATE):
    for key, value in ports.items():
        monkeypatch.setenv(key, value)


def _kitchen():
    return load_demo_config(**KITCHEN)


def _locations(tree):
    """(kind, owner path, owner dict, key) of every port the runtime dials."""
    found = []

    def walk(node, path):
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "bridge_port":
                    found.append(("bridge", ".".join(path), node, key))
                elif key == "port" and path[-1:] == ("graspgenx",):
                    found.append(("graspgenx", ".".join(path), node, key))
                elif key == "port" and path[-1:] == ("occupancy",):
                    found.append(("occupancy", ".".join(path), node, key))
                walk(value, (*path, str(key)))
        elif isinstance(node, list):
            for i, value in enumerate(node):
                walk(value, (*path, str(i)))

    walk(tree, ())
    return found


def _ports(tree) -> dict:
    out = {kind: {} for kind in KINDS}
    for kind, path, owner, key in _locations(tree):
        out[kind][path] = owner[key]
    return out


def _values(tree) -> dict:
    return {kind: set(ports.values()) for kind, ports in _ports(tree).items()}


# ── the resolved config ───────────────────────────────────────────────────


def test_env_overrides_every_port_in_every_view(monkeypatch):
    unset = _ports(_kitchen().as_dict())
    for kind in KINDS:
        assert REQUIRED[kind] <= set(unset[kind]), kind
    assert _values(_kitchen().as_dict()) == {"bridge": {8611}, "graspgenx": {5556}, "occupancy": {5557}}

    _private(monkeypatch)
    found = _ports(_kitchen().as_dict())
    assert {kind: set(found[kind].values()) for kind in KINDS} == {
        "bridge": {44011}, "graspgenx": {44016}, "occupancy": {44017}}
    # An override moves a port; it never adds or drops a location.
    assert {kind: set(found[kind]) for kind in KINDS} == {kind: set(unset[kind]) for kind in KINDS}


def test_empty_means_unset_and_each_variable_is_independent(monkeypatch):
    expected = _kitchen().as_dict()
    for kind, _, owner, key in _locations(expected):
        if kind == "occupancy":
            owner[key] = 44017
    # Empty = unset, exactly like launch.sh's `${CASCADE_..._PORT:-default}`.
    _private(monkeypatch, {"CASCADE_BRIDGE_PORT": "", "CASCADE_GRASPGENX_PORT": "",
                           "CASCADE_OCCUPANCY_PORT": "44017"})
    assert _kitchen().as_dict() == expected


def test_clearing_the_variables_restores_the_exact_unset_resolution(monkeypatch):
    # In-tree half of the default-path golden; the cross-revision half (origin/main
    # vs this branch, byte-identical JSON of the shipped profiles) is in the B34 PR.
    unset = _kitchen().as_dict()
    _private(monkeypatch)
    assert _kitchen().as_dict() != unset
    for key in PORT_VARS:
        monkeypatch.delenv(key)
    assert _kitchen().as_dict() == unset


@pytest.mark.parametrize("value", ["1", "65535"])
def test_the_edge_ports_are_accepted(monkeypatch, value):
    _private(monkeypatch, {key: value for key in PORT_VARS})
    assert _values(_kitchen().as_dict()) == {kind: {int(value)} for kind in KINDS}


MALFORMED = ["abc", "0", "65536", "70000", "-1", "+44011", " 44011", "44011 ", "44_011",
             "44011.0", "0x1F", "1e3", "  ",
             "\u0664\u0664\u0660\u0661\u0661",   # Arabic-Indic digits: int() accepts them
             "\uff14\uff14\uff10\uff11\uff11"]   # fullwidth digits: int() accepts them


@pytest.mark.parametrize("key", PORT_VARS)
@pytest.mark.parametrize("value", MALFORMED)
def test_a_malformed_port_is_refused_naming_the_variable(monkeypatch, key, value):
    monkeypatch.setenv(key, value)
    with pytest.raises(ValueError, match=key):
        _kitchen()
    # ...for a mock-only configuration too: the variable is wrong whatever the profile.
    with pytest.raises(ValueError, match=key):
        load_demo_config()


@pytest.fixture
def layered_config(tmp_path):
    """demo.yaml < booth.yaml < arm `overrides:` all set ports; the env beats them all."""
    for kind in ("cameras", "arms", "llm"):
        (tmp_path / kind).mkdir()
    (tmp_path / "demo.yaml").write_text(
        "grasp:\n  graspgenx:\n    port: 5556\noccupancy:\n  port: 5557\n")
    (tmp_path / "booth.yaml").write_text("occupancy:\n  port: 6001\n")
    (tmp_path / "cameras" / "sim.yaml").write_text("type: isaac\nbridge_port: 8611\n")
    # An Isaac profile without the key: its consumers fall back to 8611, so the
    # override must plant it.
    (tmp_path / "cameras" / "bare.yaml").write_text("type: isaac\n")
    (tmp_path / "cameras" / "usb.yaml").write_text("type: uvc\n")
    (tmp_path / "arms" / "a.yaml").write_text(
        "type: isaac\nbridge_port: 8611\noverrides:\n"
        "  grasp:\n    graspgenx:\n      port: 7000\n  occupancy:\n    port: 7001\n")
    (tmp_path / "llm" / "mock.yaml").write_text("type: mock\n")
    return tmp_path


def test_env_beats_booth_and_arm_overrides_and_never_creates_a_section(
        layered_config, monkeypatch, tmp_path):
    monkeypatch.setenv("CASCADE_BOOTH", "1")

    def load():
        return load_demo_config(cameras=["sim", "bare", "usb"], arm="a", config_dir=layered_config)

    before = load().as_dict()
    assert before["grasp"]["graspgenx"]["port"] == 7000
    assert before["occupancy"]["port"] == 7001
    assert [c.get("bridge_port") for c in before["cameras"]] == [8611, None, None]
    _private(monkeypatch)
    after = load().as_dict()
    for view in (after, after["arms"][0]["resolved"]):
        assert view["grasp"]["graspgenx"]["port"] == 44016
        assert view["occupancy"]["port"] == 44017
        assert view["arm"]["bridge_port"] == 44011
        assert [c.get("bridge_port") for c in view["cameras"]] == [44011, 44011, None]

    # A section the config does not have stays absent: creating `occupancy:`
    # would ENABLE the map (enabled defaults to true once a section exists).
    monkeypatch.delenv("CASCADE_BOOTH")
    bare = tmp_path / "bare_config"
    for kind in ("cameras", "arms", "llm"):
        (bare / kind).mkdir(parents=True)
    (bare / "demo.yaml").write_text("grasp:\n  backend: obb\n")
    (bare / "cameras" / "usb.yaml").write_text("type: uvc\n")
    (bare / "arms" / "m.yaml").write_text("type: mock\n")
    (bare / "llm" / "mock.yaml").write_text("type: mock\n")
    cfg = load_demo_config(camera="usb", arm="m", config_dir=bare).as_dict()
    for view in (cfg, cfg["arms"][0]["resolved"]):
        assert "occupancy" not in view and view["grasp"] == {"backend": "obb"}
        assert "bridge_port" not in view["arm"] and "bridge_port" not in view["camera"]


def test_load_profile_gives_only_bridge_profiles_the_override(monkeypatch):
    # The standalone viewer/recorder load one camera profile and dial it.
    monkeypatch.setenv("CASCADE_BRIDGE_PORT", "44011")
    assert load_profile("cameras", "isaac_proof").bridge_port == 44011  # via `extends: isaac`
    assert load_profile("arms", "isaac_kitchen_gpu").bridge_port == 44011
    assert "bridge_port" not in load_profile("cameras", "mock")
    assert "bridge_port" not in load_profile("arms", "so101_mock")
    assert load_profile("bases", "microduck_isaac").bridge_port is None


def test_composed_and_mobile_profiles_keep_their_own_port_boundaries(monkeypatch):
    _private(monkeypatch)
    # `_ignore_robot_environment` ignores CASCADE_ROBOT/CASCADE_BASE (robot
    # selection), not the endpoint overrides of the stack this process dials.
    direct = load_demo_config(cameras=["isaac"], arm="isaac_kitchen_gpu",
                              _ignore_robot_environment=True).as_dict()
    assert _values(direct) == {"bridge": {44011}, "graspgenx": {44016}, "occupancy": {44017}}
    composed = load_robot_config("mobile_manipulator_mock").as_dict()["domains"]
    manipulation = _ports(composed["manipulation"]["resolved"])
    assert set(manipulation["graspgenx"].values()) == {44016}
    assert set(manipulation["occupancy"].values()) == {44017}
    assert manipulation["bridge"] == {}  # mock arm and camera: no bridge endpoint invented
    assert _ports(composed["locomotion"]["resolved"]) == {kind: {} for kind in KINDS}

    # Mobile bases have their own explicit variable; the arm ones never reach them.
    assert load_demo_config(base="microduck_isaac").base.bridge_port is None
    monkeypatch.setenv("CASCADE_BRIDGE_PORT", "not-a-port")  # not a mobile variable: unread
    assert load_demo_config(base="microduck_isaac").base.bridge_port is None
    monkeypatch.setenv("CASCADE_MICRODUCK_BRIDGE_PORT", "44021")
    assert load_demo_config(base="microduck_isaac").base.bridge_port == 44021


# ── the dial itself ───────────────────────────────────────────────────────


def test_bridge_clients_dial_the_override(monkeypatch):
    from cascade.control.arm_base import make_arm
    from cascade.perception.camera_base import CameraError, make_camera
    from cascade.sim import bridge_client

    _private(monkeypatch)
    cfg = _kitchen()
    dialled = []

    def spy(address, *args, **kwargs):
        dialled.append(tuple(address))
        raise ConnectionRefusedError("spy: no socket is opened in this test")

    monkeypatch.setattr(bridge_client.socket, "create_connection", spy)
    for camera in cfg.cameras:  # as build_runtime and the standalone viewer construct them
        with pytest.raises(CameraError, match="44011"):
            make_camera(Cfg(camera)).open()
    for arm in cfg.arms:
        with pytest.raises(bridge_client.BridgeError, match="44011"):
            make_arm(Cfg(arm)).connect()
    assert dialled == [("127.0.0.1", 44011)] * (len(KITCHEN["cameras"]) + len(KITCHEN["arms"]))


class _Dialled(Exception):
    pass


@pytest.fixture
def zmq_dials(monkeypatch):
    """Stand-in zmq/msgpack: record every REQ `connect` endpoint, open nothing.

    Faked rather than importorskip'd so the dial is asserted on the minimal
    install too (the grasping extra brings pyzmq)."""
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
    for name, module in (("zmq", zmq), ("msgpack", types.ModuleType("msgpack")),
                         ("msgpack_numpy", msgpack_numpy)):
        monkeypatch.setitem(sys.modules, name, module)
    return endpoints


def test_graspgenx_dials_the_override_carried_by_the_resolved_config(monkeypatch, zmq_dials):
    from cascade.grasping.graspgenx_backend import GraspGenXPlanner

    _private(monkeypatch)
    cfg = _kitchen()
    views = [cfg.grasp, *(Cfg(arm["resolved"]).grasp for arm in cfg.arms)]
    # GraspGenXPlanner also reads its variable at construction (pre-B34
    # behaviour, kept); the resolved config must carry the port on its own.
    monkeypatch.delenv("CASCADE_GRASPGENX_PORT")
    for grasp in views:
        with pytest.raises(_Dialled):
            GraspGenXPlanner(grasp).probe()
    assert zmq_dials == ["tcp://127.0.0.1:44016"] * len(views)


def test_occupancy_dials_the_override(monkeypatch, zmq_dials):
    from cascade.perception.occupancy import OccupancyMap

    _private(monkeypatch)
    cfg = _kitchen()
    with pytest.raises(_Dialled):  # the startup probe is the first round trip
        OccupancyMap.from_config(cfg.get("occupancy"), workspace_min=[0.0, -0.5, 0.0],
                                 workspace_max=[0.6, 0.5, 0.5])
    assert zmq_dials == ["tcp://127.0.0.1:44017"]


# ── the launcher hands them over ──────────────────────────────────────────


def _registered_env(monkeypatch, capsys) -> dict:
    """The `env` launch.sh registers for the MCP server (its Python heredoc, run in-process)."""
    blocks = re.findall(r"<<'PYEOF'[^\n]*\n(.*?)\nPYEOF", LAUNCH.read_text(), re.DOTALL)
    [registration] = [block for block in blocks if '"requestTimeoutMs"' in block]
    monkeypatch.setattr(sys, "argv", ["-", sys.executable, str(REPO), "isaac,isaac_side,isaac_proof",
                                      "isaac_kitchen_gpu", "/fixture/model", "", "isaac",
                                      "/fixture/state", "owner", "0", "auto"])
    exec(compile(registration, str(LAUNCH), "exec"), {})
    return json.loads(capsys.readouterr().out)["env"]


@pytest.mark.parametrize("ports", [{}, PRIVATE, {"CASCADE_OCCUPANCY_PORT": "44017"}],
                         ids=["unset", "private", "occupancy-only"])
def test_launcher_registers_the_mcp_server_with_the_port_overrides(monkeypatch, capsys, ports):
    # The registered `env` is how the launcher hands per-run settings to the
    # MCP server: the gateway that spawns it need not have the launcher's env.
    _private(monkeypatch, ports)
    registered = _registered_env(monkeypatch, capsys)
    assert {key: registered[key] for key in PORT_VARS if key in registered} == ports


def _runtime_check_ports(tmp_path, ports: dict) -> list[str]:
    """Run launch.sh's runtime-check block with a fake python; the ports it saw."""
    source = LAUNCH.read_text()
    begin = source.index('        RUNTIME_CHECK="$(cd "$REPO/models"')
    end = source.index("""        printf '%s\\n' "$RUNTIME_CHECK" | sed""", begin)
    fake_py = tmp_path / "python"
    fake_py.write_text('#!/bin/sh\ncat >/dev/null\n'
                       'echo "ports=${CASCADE_BRIDGE_PORT:-unset},${CASCADE_GRASPGENX_PORT:-unset},'
                       '${CASCADE_OCCUPANCY_PORT:-unset}"\necho "[launch] runtime builds: {}"\n')
    fake_py.chmod(0o755)
    (tmp_path / "repo" / "models").mkdir(parents=True)
    (tmp_path / "state").mkdir(exist_ok=True)
    script = tmp_path / "check.sh"
    script.write_text(
        f'REPO="{tmp_path / "repo"}"; STATE_DIR="{tmp_path / "state"}"; PY="{fake_py}"\n'
        'CAMERAS=isaac; ARM=isaac_kitchen_gpu; DETECTOR=/det.pt; CLASSES=\n'
        'die() { echo "DIE: $*" >&2; exit 1; }\n'
        + source[begin:end] + 'printf "%s\\n" "$RUNTIME_CHECK"\n')
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), **ports}
    result = subprocess.run(["bash", str(script)], env=env, capture_output=True, text=True,
                            timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    line = next(ln for ln in result.stdout.splitlines() if ln.startswith("ports="))
    return line[len("ports="):].split(",")


def test_runtime_check_and_mcp_server_get_the_same_ports(tmp_path, monkeypatch, capsys):
    # The runtime check builds the runtime "with the exact env the MCP server
    # gets" (launch.sh). It inherits CASCADE_*_PORT from the launcher, so it
    # passed against a private bridge while the registered MCP server never
    # got the bridge/occupancy ports and dialled the defaults: a green check
    # for a different stack than the one the robot runs.
    checked = _runtime_check_ports(tmp_path, PRIVATE)
    assert checked == [PRIVATE[key] for key in PORT_VARS]
    _private(monkeypatch)
    registered = _registered_env(monkeypatch, capsys)
    assert [registered.get(key, "unset") for key in PORT_VARS] == checked
