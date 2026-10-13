#!/usr/bin/env python3
"""Configure and run OpenClaw + one shared FR3 MCP runtime + camera companion.

No installation, robot motions, model downloads, bringup changes or Git writes.
The existing native ROS controllers, Cosmos and occupancy server stay external.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import secrets
import shutil
import signal
import socket
import subprocess
import sys
import time
from urllib.parse import urlsplit
from urllib.request import ProxyHandler, build_opener

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from cascade.apps.process_owner import (load_owner, live_records, register_process,
                                       stop_owned, process_identity, _write_json)
from franka_camera_companion import loopback_url

PROFILE = "cascade-franka"
CHAT_TOOLS = (
    "world_state", "camera_snapshot", "get_observation", "localize_object", "locate_pixel",
    "preview_grasp", "grasp_at_pixel", "grasp_object", "pick_and_place", "place_at",
    "place_on_object", "move_home", "open_gripper", "close_gripper", "emergency_stop",
    "reset_stop", "verify_last_action",
)


def private_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with path.open("w") as stream:
        os.chmod(path, 0o600)
        stream.write(text)


def token_file(path):
    if not path.exists():
        private_text(path, secrets.token_urlsafe(32) + "\n")
    token = path.read_text().strip()
    if len(token) < 24:
        raise ValueError(f"Invalid token file: {path}")
    os.chmod(path, 0o600)
    return token


def configuration(state, token, gateway_token, *, context_window, model_base, port_offset=0,
                  full_tool_catalog=False):
    if context_window < 4096:
        raise ValueError("context window must be at least 4096 tokens")
    # The deployed inference endpoint is local. This launcher cannot silently
    # redirect robot images to a remote service or substitute another model.
    origin, suffix = model_base.rstrip("/").rsplit("/", 1)
    if suffix != "v1":
        raise ValueError("model base URL must end in /v1")
    loopback_url(origin)
    ports = {"gateway": 18790 + port_offset, "mcp": 18740 + port_offset,
             "stream": 8090 + port_offset, "camera": 8091 + port_offset}
    if any(not 0 < port < 65536 for port in ports.values()):
        raise ValueError("invalid port offset")
    model = "local-cosmos/cosmos3-edge"
    return {
        "gateway": {"mode": "local", "bind": "loopback", "port": ports["gateway"],
                    "auth": {"mode": "token", "token": gateway_token}},
        "agents": {"defaults": {
            "workspace": str(state / "workspace"), "skills": [],
            "model": {"primary": model}, "timeoutSeconds": 360,
            "heartbeat": {"every": "0m"},
            "models": {model: {"params": {"temperature": 0, "maxTokens": 512,
                "chat_template_kwargs": {"enable_thinking": False}}}},
        }},
        "models": {"providers": {"local-cosmos": {
            "baseUrl": model_base.rstrip("/"), "apiKey": "EMPTY", "api": "openai-completions",
            "models": [{"id": "cosmos3-edge", "name": "Local Cosmos3 Edge",
                        "reasoning": False, "input": ["text", "image"],
                        "contextWindow": context_window, "maxTokens": 512}],
        }}},
        "tools": {"allow": (["cascade__*"] if full_tool_catalog else ["cascade__" + name for name in CHAT_TOOLS]),
                  "toolSearch": {"enabled": False}},
        "cron": {"enabled": False},
        "plugins": {
            "load": {"paths": [str(REPO / "configs/openclaw/franka-plugin")]},
            "entries": {
                "memory-core": {"enabled": False},
                "cascade-franka-prompt": {
                    "enabled": True,
                    "hooks": {"allowConversationAccess": True, "allowPromptInjection": True},
                    "config": {"workspaceDir": str(state / "workspace"),
                               "modelProviderId": "local-cosmos", "modelId": "cosmos3-edge"},
                },
            },
        },
        "mcp": {"servers": {"cascade": {
            "url": f"http://127.0.0.1:{ports['mcp']}/mcp", "transport": "streamable-http",
            "headers": {"Authorization": "Bearer " + token},
            "connectionTimeoutMs": 120000, "requestTimeoutMs": 300000,
            "supportsParallelToolCalls": False,
        }}},
    }, ports


def configure(args, state):
    if live_records(state, load_owner(state, REPO, PROFILE, create=True)):
        raise RuntimeError("Stop this profile before changing its configuration")
    config, ports = configuration(state, token_file(state / "mcp.token"),
        token_file(state / "gateway.token"), context_window=args.context_window,
        model_base=args.model_base, port_offset=args.port_offset,
        full_tool_catalog=getattr(args, "full_tool_catalog", False))
    path = state / "openclaw/openclaw.json"
    if path.exists():
        # Preserve a reviewable copy of the old profile; never rewrite the
        # personal ~/.openclaw config or another checkout's registration.
        shutil.copy2(path, path.with_name(f"openclaw.{time.time_ns()}.backup.json"))
    _write_json(path, config)
    prompt = state / "workspace/AGENTS.md"
    source = REPO / "configs/openclaw/franka-instructions.md"
    if not prompt.exists():
        private_text(prompt, source.read_text())
    settings = {"ports": ports, "mock": args.mock, "repo": str(REPO), "profile": PROFILE}
    _write_json(state / "settings.json", settings)
    print(f"Configured {PROFILE}: {path} (secrets stored privately; no services started)")


def environment(state, settings, python):
    # A developer's OpenClaw profile must not override this dedicated profile.
    env = {k: v for k, v in os.environ.items() if not k.startswith("OPENCLAW_")}
    env.update(OPENCLAW_PROFILE=PROFILE, CASCADE_OPENCLAW_PROFILE=PROFILE,
        OPENCLAW_STATE_DIR=str(state / "openclaw"),
        OPENCLAW_CONFIG_PATH=str(state / "openclaw/openclaw.json"),
        PY=str(python), PYTHONPATH=str(REPO / "src") + (os.pathsep + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else ""),
        CASCADE_STREAM_PORT=str(settings["ports"]["stream"]),
        CASCADE_RUN_DIR=str(state / "runtime"),
        CASCADE_BELIEFS_PATH=str(state / "memory/beliefs.json"),
        CASCADE_GRASP_MEMORY_PATH=str(state / "memory/grasp_memory.json"),
        CASCADE_ENVELOPE_PATH=str(state / "memory/envelope.json"),
        CASCADE_START_STOPPED="1", CASCADE_PREWARM="1", CASCADE_VIEW="0",
        CASCADE_FRANKA_MOCK="1" if settings["mock"] else "0",
        YOLO_OFFLINE="True", ULTRALYTICS_OFFLINE="True")
    # This dedicated launcher selects the Franka profile in its wrapper.
    # Plural/mobile selectors otherwise override CASCADE_ARM inside MCP.
    for key in ("CASCADE_START_HOLDING", "CASCADE_ARMS", "CASCADE_ROBOT", "CASCADE_BASE",
                "CASCADE_GROUNDING_LLM_PROFILE"):
        env.pop(key, None)
    private_bin = REPO / ".openclaw-cli/bin"
    if private_bin.is_dir():
        env["PATH"] = str(private_bin) + os.pathsep + env.get("PATH", "")
    return env


def read_settings(state):
    settings = json.loads((state / "settings.json").read_text())
    if settings.get("repo") != str(REPO) or settings.get("profile") != PROFILE:
        raise ValueError("Settings belong to another checkout/profile; run configure here")
    return settings


def get_json(url):
    with build_opener(ProxyHandler({})).open(url, timeout=3) as response:
        return json.loads(response.read(2 * 1024 * 1024))


def port_open(port):
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=.3):
            return True
    except OSError:
        return False


def python_path(args):
    candidates = [args.python, os.environ.get("PY"), REPO / ".venv-hardware/bin/python",
                  REPO / ".venv/bin/python", sys.executable]
    for value in candidates:
        if value and os.access(value, os.X_OK):
            return Path(value).absolute()
    raise RuntimeError("Choose an installed CASCADE interpreter with --python")


def cli_path(args):
    for value in (args.openclaw, REPO / ".openclaw-cli/bin/openclaw",
                  REPO / ".openclaw-cli/node_modules/.bin/openclaw", shutil.which("openclaw")):
        if value and os.access(value, os.X_OK):
            return str(Path(value).absolute())
    raise RuntimeError("OpenClaw CLI is missing; follow docs/FRANKA_OPENCLAW.md")


def checked_cli(cli, env, *command):
    result = subprocess.run([cli, "--profile", PROFILE, *command], env=env,
                            cwd=REPO, capture_output=True, text=True, timeout=45)
    if result.returncode:
        # Keep tool configuration and tokens out of terminal exceptions.
        raise RuntimeError(f"OpenClaw {' '.join(command[:2])} failed; run that command in this profile for diagnostics")
    return result.stdout


def browser_command(browser, state, url):
    """Use a separate browser profile; never store the authenticated URL."""
    extension = str(REPO / "extensions/chrome")
    return [browser, "--user-data-dir=" + str(state / "browser"),
            "--load-extension=" + extension, "--disable-extensions-except=" + extension,
            "--no-first-run", "--no-default-browser-check", "--new-window", url]


def dashboard(args, state, settings, env, cli):
    if args.no_open:
        return subprocess.call([cli, "--profile", PROFILE, "dashboard", "--no-open"], env=env, cwd=REPO)
    details = json.loads(checked_cli(cli, env, "dashboard", "--json"))
    url = details.get("url", "")
    parsed = urlsplit(url)
    if parsed.scheme != "http" or parsed.hostname != "127.0.0.1" or parsed.port != settings["ports"]["gateway"]:
        raise RuntimeError("OpenClaw returned a dashboard URL for a different gateway")
    browser = args.browser or shutil.which("chromium") or shutil.which("chromium-browser")
    if not browser:
        # Regular Chrome can load the same extension manually once, using
        # chrome://extensions. The generic dashboard handles default browsers.
        return subprocess.call([cli, "--profile", PROFILE, "dashboard"], env=env, cwd=REPO)
    if not (env.get("DISPLAY") or env.get("WAYLAND_DISPLAY")):
        raise RuntimeError("No graphical session; use dashboard --no-open")
    with (state / "browser.log").open("ab") as log:
        os.chmod(log.name, 0o600)
        subprocess.Popen(browser_command(browser, state, url), env=env, cwd=REPO,
                         stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                         start_new_session=True)
    return 0


def probe_model(state):
    from demo_proof import probe_native_tools, request_json
    cfg = json.loads((state / "openclaw/openclaw.json").read_text())
    provider = cfg["models"]["providers"]["local-cosmos"]
    base = provider["baseUrl"]
    # Revalidate a manually edited config before transmitting images/context.
    loopback_url(base.removesuffix("/v1"))
    rows = request_json(base + "/models").get("data", [])
    served = next((row for row in rows if row.get("id") == "cosmos3-edge"), None)
    if served is None:
        raise RuntimeError("The configured endpoint does not serve cosmos3-edge")
    capacity = served.get("max_model_len")
    if capacity and provider["models"][0]["contextWindow"] > int(capacity):
        raise RuntimeError(f"Configured context exceeds served {capacity} tokens; reconfigure it")
    return probe_native_tools(base, "cosmos3-edge")


def terminate_spawned(child, timeout=5):
    """Reap this invocation's private group, including a respawned gateway.

    These children always start a new session. Keep their birth identity
    across wrapper execs; a command-string match is not stable during startup.
    """
    expected_birth = getattr(child, "_cascade_birth", None)

    def signal_group(number):
        current = process_identity(child.pid)
        if current is not None and current["birth"] != expected_birth:
            raise RuntimeError("Refusing cleanup: spawned process identity changed")
        try:
            os.killpg(child.pid, number)
        except ProcessLookupError:
            return False
        return True

    signal_group(signal.SIGTERM)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        child.poll()  # reap an exited leader; its descendants can still own the group
        if not signal_group(0):
            break
        time.sleep(.05)
    signal_group(signal.SIGKILL)
    child.wait(timeout=5)


def spawn(command, state, role, env, owner):
    with (state / (role + ".log")).open("ab") as log:
        os.chmod(log.name, 0o600)
        child = subprocess.Popen(command, cwd=REPO, env=env, stdin=subprocess.DEVNULL,
                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    child._cascade_birth = (process_identity(child.pid) or {}).get("birth")
    # MCP registers itself after interpreter startup. The gateway is recorded
    # after authenticated health succeeds, beyond its CLI/exec startup phase.
    if role not in ("mcp", "gateway_child"):
        try:
            time.sleep(.3)
            if child.poll() is not None:
                raise RuntimeError(f"{role} exited; see {state / (role + '.log')}")
            register_process(state, owner, child.pid, role)
        except BaseException:
            terminate_spawned(child)
            raise
    return child


def wait_runtime(state, settings, child, timeout=120):
    deadline = time.monotonic() + timeout
    url = f"http://127.0.0.1:{settings['ports']['stream']}/state"
    while time.monotonic() < deadline:
        if child.poll() is not None:
            raise RuntimeError(f"MCP runtime exited; see {state / 'mcp.log'}")
        try:
            view = get_json(url)
            cams = view.get("cameras", {})
            if (cams and all(not row.get("error") and row.get("fps", 0) > 0 for row in cams.values())
                    and view.get("stopped") is True and view.get("resume_supported") is True):
                return view
        except (OSError, ValueError):
            pass
        time.sleep(.5)
    raise RuntimeError(f"Camera/STOP/Resume readiness timed out; inspect {state / 'mcp.log'}")


def start(args, state):
    settings = read_settings(state)
    owner = load_owner(state, REPO, PROFILE, create=True)
    if live_records(state, owner):
        raise RuntimeError("This profile has running services; use status or stop first")
    ports = settings["ports"]
    occupied = [port for key, port in ports.items() if (key != "gateway" or not args.no_gateway) and port_open(port)]
    if occupied:
        raise RuntimeError(f"Ports already in use: {occupied}. Stop the old Cascade interface explicitly first")
    python = python_path(args)
    env = environment(state, settings, python)
    cli = None
    if args.no_gateway and not settings["mock"]:
        raise ValueError("--no-gateway is only for the isolated mock smoke")
    if not args.no_gateway:
        cli = cli_path(args)
        checked_cli(cli, env, "config", "validate")
        probe_model(state)  # native tools required; never silently switch models
    children = []
    success = False
    try:
        runtime = spawn(["bash", str(REPO / "scripts/run_franka_http_mcp.sh"),
            "--http", f"127.0.0.1:{ports['mcp']}", "--token-file", str(state / "mcp.token"),
            "--launch-owner", owner["owner"], "--launch-state-dir", str(state)], state, "mcp", env, owner)
        children.append(runtime)
        view = wait_runtime(state, settings, runtime)
        cameras = spawn([str(python), str(REPO / "scripts/franka_camera_companion.py"),
            "--port", str(ports["camera"]), "--source", f"http://127.0.0.1:{ports['stream']}"],
            state, "cameras", env, owner)
        children.append(cameras)
        if cli:
            gateway = spawn([cli, "--profile", PROFILE, "gateway", "run", "--bind", "loopback",
                             "--port", str(ports["gateway"])], state, "gateway_child", env, owner)
            children.append(gateway)
            from demo_proof import wait_gateway
            wait_gateway(60, environment=env, command_prefix=[cli, "--profile", PROFILE])
            if gateway.poll() is not None:
                raise RuntimeError("OpenClaw gateway exited during startup")
            register_process(state, owner, gateway.pid, "gateway_child")
            checked_cli(cli, env, "mcp", "probe", "cascade", "--json")
        success = True
        print(f"Started {PROFILE}; STOP latched; camera(s): {', '.join(view['cameras'])}.")
        print(f"Operator STOP / Resume & Home: http://127.0.0.1:{ports['stream']}/")
        if cli:
            print(f"OpenClaw: http://127.0.0.1:{ports['gateway']}/ (use the dashboard command to sign in)")
        print("No robot motion has been tested by this launch.")
        if cli and not args.no_open:
            try:
                dashboard(args, state, settings, env, cli)
            except (OSError, ValueError, RuntimeError) as exc:
                print(f"Services are running; browser was not opened: {exc}", file=sys.stderr)
    finally:
        if not success:
            # Only processes created by this invocation; never terminate an
            # endpoint discovered by port or process-name matching.
            for child in reversed(children):
                terminate_spawned(child)


def check(args, state):
    settings = read_settings(state)
    result = {"python": str(python_path(args)), "profile": PROFILE,
              "extension": str(REPO / "extensions/chrome"), "ports": settings["ports"],
              "mock": settings["mock"], "checks_are_read_only": True}
    try:
        result["openclaw"] = cli_path(args)
    except RuntimeError as exc:
        result["openclaw_error"] = str(exc)
    owner = load_owner(state, REPO, PROFILE)
    result["running"] = [{"role": r["role"], "pid": r["pid"]} for r in live_records(state, owner)] if owner else []
    print(json.dumps(result, indent=2))
    return int("openclaw_error" in result and not settings["mock"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("configure", "start", "stop", "status", "check", "dashboard"))
    parser.add_argument("--state-dir", type=Path, default=REPO / "runs/franka-openclaw")
    parser.add_argument("--python")
    parser.add_argument("--openclaw")
    parser.add_argument("--browser", help="Chromium executable for the dedicated extension profile")
    parser.add_argument("--context-window", type=int, default=32768)
    parser.add_argument("--model-base", default="http://127.0.0.1:8082/v1")
    parser.add_argument("--port-offset", type=int, default=0, help="isolates a mock smoke; extension fallback uses the default ports")
    parser.add_argument("--mock", action="store_true", help="configure mock FR3 and mock camera")
    parser.add_argument("--full-tool-catalog", action="store_true", help="configure: expose every available MCP tool to the chat model")
    parser.add_argument("--no-gateway", action="store_true", help="mock-only camera/MCP smoke, no model or OpenClaw")
    parser.add_argument("--no-open", action="store_true", help="start: skip browser; dashboard: print its sign-in URL")
    args = parser.parse_args(argv)
    state = args.state_dir.expanduser().resolve()
    if args.action in ("status", "check"):
        return check(args, state)
    if args.action == "dashboard":
        settings = read_settings(state)
        return dashboard(args, state, settings, environment(state, settings, python_path(args)), cli_path(args))
    state.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (state / ".operation.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.action == "configure":
            configure(args, state)
        elif args.action == "start":
            start(args, state)
        else:
            owner = load_owner(state, REPO, PROFILE)
            if owner:
                stop_owned(state, owner, roles={"mcp", "cameras", "gateway_child"})
            print("Stopped owned OpenClaw/MCP/camera services. ROS bringup and model services remain running.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f"[franka-openclaw] {exc}", file=sys.stderr)
        raise SystemExit(1)
