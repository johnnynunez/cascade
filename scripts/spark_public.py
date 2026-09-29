#!/usr/bin/env python3
"""Run a prepared Spark demo and a restricted ngrok visitor with user services."""
from __future__ import annotations

import argparse
import base64
import importlib.util
import json
import os
from pathlib import Path
import pwd
import re
import shlex
import signal
import subprocess
import sys
import time
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

SERVICES = ("paai-spark-demo", "paai-spark-visitor", "paai-spark-ngrok")
CAMERA_ORIGIN = "http://127.0.0.1:8091"
VISITOR_ORIGIN = "http://127.0.0.1:8093"


def run(command, timeout=30, **kwargs):
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout, **kwargs)
    except subprocess.TimeoutExpired:
        raise RuntimeError("The service operation timed out; private output withheld") from None
    if result.returncode:
        raise RuntimeError("The service operation failed; private output withheld")
    return result.stdout


def private_file(path):
    path = Path(path).expanduser()
    if (path.is_symlink() or not path.is_file() or path.stat().st_uid != os.getuid()
            or path.stat().st_mode & 0o077 or path.stat().st_size > 65536):
        raise ValueError("Credential files must be owned by this user and have mode 0600")
    return path


def private_json(path, value):
    if path.is_symlink():
        raise ValueError("Private files must not be symbolic links")
    temporary = path.with_suffix(".tmp")
    with open(temporary, "w", opener=lambda p, flags: os.open(p, flags | os.O_NOFOLLOW, 0o600)) as stream:
        os.fchmod(stream.fileno(), 0o600)
        json.dump(value, stream, indent=2)
        stream.write("\n")
    temporary.chmod(0o600)
    temporary.replace(path)


def root_path(path):
    root = Path(path).resolve(strict=True)
    receipt = json.loads((root / "runs/.install/install.json").read_text())
    if receipt.get("repo") != str(root) or receipt.get("profile") != "spark" or not receipt.get("eula_accepted"):
        raise ValueError("Use the original prepared Spark checkout")
    return root


def private_root(repo):
    directory = repo / "runs/.install/public"
    if directory.is_symlink():
        raise ValueError("The private service directory must not be a symbolic link")
    directory.mkdir(parents=True, mode=0o700, exist_ok=True)
    directory.chmod(0o700)
    return directory


def browser_module(repo):
    sys.path.insert(0, str(repo / "src"))
    spec = importlib.util.spec_from_file_location("spark_browser", repo / "scripts/spark_browser.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def runtime_home(repo):
    return browser_module(repo).runtime_home(repo)


def unit_directory():
    # The user manager belongs to the real user even during a private-HOME test.
    return Path(pwd.getpwuid(os.getuid()).pw_dir) / ".config/systemd/user"


def unit_arg(value, *, expand_dollars=True):
    value = str(value)
    if any(character in value for character in "\n\r\x00"):
        raise ValueError("Service arguments cannot contain line breaks")
    value = value.replace("\\", "\\\\").replace('"', '\\"').replace("%", "%%")
    return '"' + (value.replace("$", "$$") if expand_dollars else value) + '"'


def unit_path(value):
    # Path-valued directives do not use ExecStart's argument quoting.
    value = str(value)
    if not value.startswith("/") or any(character in value for character in "\n\r\x00"):
        raise ValueError("Service paths must be absolute and contain no line breaks or NUL")
    return value.replace("%", "%%")


def units(repo, settings):
    script = repo / "scripts/spark_public.py"
    common = ["/usr/bin/python3", str(script)]
    commands = {
        "paai-spark-demo": [*common, "_serve", "--repo", str(repo)],
        "paai-spark-visitor": ["/usr/bin/python3", str(repo / "deploy/brev/visitor.py"), "--port", "8093",
                              "--camera-origin", CAMERA_ORIGIN, "--repo", str(repo),
                              "--auth-file", str(private_root(repo) / "auth.json")],
        "paai-spark-ngrok": [*common, "_ngrok", "--repo", str(repo)],
    }
    result = {}
    environment = ["/usr/bin/env", "-i", "HOME=" + settings["runtime_home"],
                   "USER=" + pwd.getpwuid(os.getuid()).pw_name, "LANG=C.UTF-8",
                   "PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
                   "PYTHONUNBUFFERED=1", "PYTHONDONTWRITEBYTECODE=1",
                   f"XDG_RUNTIME_DIR=/run/user/{os.getuid()}",
                   f"DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/{os.getuid()}/bus"]
    for name, command in commands.items():
        after = "network-online.target" + (" paai-spark-demo.service" if name != "paai-spark-demo" else "")
        result[name] = (f"# PAAI Spark checkout: {repo}\n[Unit]\nDescription=PAAI Spark {name}\n"
                        f"After={after}\nStartLimitIntervalSec=0\n[Service]\nType=simple\n"
                        f"WorkingDirectory={unit_path(repo)}\nEnvironment={unit_arg('HOME=' + settings['runtime_home'], expand_dollars=False)}\n"
                        "Environment=PYTHONUNBUFFERED=1\nEnvironment=PYTHONDONTWRITEBYTECODE=1\n"
                        f"ExecStart={' '.join(unit_arg(part) for part in environment + command)}\n"
                        "Restart=always\nRestartSec=10\nTimeoutStopSec=420\nKillMode=mixed\n"
                        "UMask=0077\nNoNewPrivileges=true\n[Install]\nWantedBy=default.target\n")
    return result


def owned_units(repo):
    result = {}
    directory = unit_directory()
    for name in SERVICES:
        output = run(["systemctl", "--user", "show", name + ".service", "--no-pager",
                      "--property=LoadState,ActiveState,FragmentPath,DropInPaths"])
        values = dict(line.split("=", 1) for line in output.splitlines() if "=" in line)
        path = directory / (name + ".service")
        fragment = values.get("FragmentPath")
        if values.get("DropInPaths"):
            raise ValueError("A PAAI service has overrides; review its ownership first")
        if fragment or path.exists():
            if (path.is_symlink() or (fragment and Path(fragment) != path)
                    or not path.read_text().startswith(f"# PAAI Spark checkout: {repo}\n")):
                raise ValueError("A PAAI service name belongs to another installation")
            result[name] = {"installed": True, "active": values.get("ActiveState") == "active"}
        elif values.get("LoadState") != "not-found":
            raise ValueError("A PAAI service owner could not be established")
        else:
            result[name] = {"installed": False, "active": False}
    return result


def ngrok_token(repo, token_file, config_file):
    if token_file:
        token = private_file(token_file).read_text().strip()
    else:
        config_file = private_file(config_file)
        # The prepared application has PyYAML; do not print or modify the account config.
        source = ("import json,sys,yaml; c=yaml.safe_load(open(sys.argv[1])); "
                  "print(json.dumps(c.get('agent',{}).get('authtoken') or c.get('authtoken')))")
        token = json.loads(run([str(repo / ".venv/bin/python"), "-c", source, str(config_file)]))
    if not isinstance(token, str) or not token or any(character.isspace() for character in token):
        raise ValueError("The ngrok credential is invalid")
    return token


def enable(repo, args):
    owned = owned_units(repo)
    if run(["loginctl", "show-user", str(os.getuid()), "--property=Linger", "--value"]).strip() != "yes":
        raise RuntimeError("This user's linger is disabled; logout persistence is unavailable")
    if not re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9.-]{0,251}[A-Za-z0-9])?", args.domain or "") or "." not in args.domain:
        raise ValueError("Supply the reserved ngrok domain, without a scheme or path")
    auth = json.loads(private_file(args.auth_file).read_text())
    if (set(auth) != {"username", "password"} or not all(isinstance(v, str) and v for v in auth.values())
            or ":" in auth["username"] or len(auth["password"]) < 12
            or any("\n" in v or "\r" in v for v in auth.values())):
        raise ValueError("Use a username and a password of at least 12 characters")
    executable = Path(args.ngrok).expanduser().resolve(strict=True)
    if not os.access(executable, os.X_OK):
        raise ValueError("The ngrok executable is unavailable")
    token = ngrok_token(repo, args.token_file, args.ngrok_config)
    directory = private_root(repo)
    def changed(path, value):
        return not path.exists() or json.loads(path.read_text()) != value
    auth_changed = changed(directory / "auth.json", auth)
    config = {"version": "3", "agent": {
        "authtoken": token, "web_addr": "127.0.0.1:4043", "update_check": False}}
    config_changed = changed(directory / "ngrok.json", config)
    private_json(directory / "auth.json", auth)
    private_json(directory / "ngrok.json", config)
    run([str(executable), "config", "check", "--config", str(directory / "ngrok.json")])
    settings = {"repo": str(repo), "domain": args.domain, "ngrok": str(executable), "runtime_home": runtime_home(repo)}
    settings_changed = changed(directory / "settings.json", settings)
    private_json(directory / "settings.json", settings)
    directory_units = unit_directory()
    directory_units.mkdir(parents=True, exist_ok=True)
    changed_units = {"paai-spark-visitor"} if auth_changed else set()
    if config_changed or settings_changed:
        changed_units.add("paai-spark-ngrok")
    for name, contents in units(repo, settings).items():
        path = directory_units / (name + ".service")
        previous = path.read_text() if path.exists() else ""
        if previous != contents:
            if path.is_symlink():
                raise ValueError("Service files must not be symbolic links")
            path.write_text(contents)
            path.chmod(0o600)
            changed_units.add(name)
    run(["systemctl", "--user", "daemon-reload"])
    run(["systemctl", "--user", "enable", "--now", *[name + ".service" for name in SERVICES]], timeout=45)
    for name in SERVICES:
        if name not in changed_units:
            continue
        if owned[name]["active"]:
            run(["systemctl", "--user", "restart", name + ".service"], timeout=450)
    return {"enabled": True, "url": "https://" + args.domain,
            "message": "Services enabled. Run check after the demo reports READY."}


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, file, code, message, headers, new_url):
        return None


def public_get(url, authorization=None):
    headers = {"ngrok-skip-browser-warning": "1"}
    if authorization:
        headers["Authorization"] = authorization
    try:
        response = build_opener(NoRedirect()).open(Request(url, headers=headers), timeout=10)
    except HTTPError as error:
        response = error
    with response:
        return response.status, response.read(4 * 1024 * 1024)


def check(repo):
    owned = owned_units(repo)
    result = {"healthy": False, "services": owned}
    if not all(row["installed"] and row["active"] for row in owned.values()):
        return result
    if not browser_module(repo).ready(repo):
        result["reason"] = "The local demo is not READY"
        return result
    directory = private_root(repo)
    settings = json.loads((directory / "settings.json").read_text())
    auth = json.loads((directory / "auth.json").read_text())
    credential = "Basic " + base64.b64encode((auth["username"] + ":" + auth["password"]).encode()).decode()
    origin = "https://" + settings["domain"]
    result["url"] = origin
    if public_get(origin)[0] != 401:
        raise RuntimeError("The public URL did not require authentication")
    for path in ("/openclaw/", "/api/openclaw-bootstrap", "/__openclaw/", "/api/control",
                 "/v1/models", "/state", "/config", "/rpc", "/mcp", "/auth.json"):
        if public_get(origin + path, credential)[0] != 404:
            raise RuntimeError("The public URL did not block a private route")
    for path in ("/", "/visitor.js", "/api/chat"):
        code, body = public_get(origin + path, credential)
        if code != 200:
            raise RuntimeError("The public visitor is unavailable")
        if path == "/api/chat":
            try:
                chat = json.loads(body)
            except (ValueError, UnicodeDecodeError):
                raise RuntimeError("The public attendee chat is not READY") from None
            if (not isinstance(chat, dict) or chat.get("enabled") is not True
                    or chat.get("ready") is not True or chat.get("agent") != "cascade-demo"):
                raise RuntimeError("The public attendee chat is not READY")
        gateway = json.loads((repo / "runs/.launch/profile-cascade-demo/openclaw/openclaw.json").read_text())
        for secret in gateway.get("gateway", {}).get("auth", {}).values():
            if isinstance(secret, str) and len(secret) >= 8 and secret.encode() in body:
                raise RuntimeError("The public visitor disclosed a private credential")
    frames = []
    for index in range(2):
        code, body = public_get(origin + "/api/status", credential)
        rows = json.loads(body).get("cameras", []) if code == 200 else []
        if len(rows) != 3 or {row.get("name") for row in rows} != {"kitchen", "worktop", "side"} or not all(row.get("online") for row in rows):
            raise RuntimeError("The three public cameras are unavailable")
        frames.append({row["name"]: row.get("frame_id") for row in rows})
        if index == 0:
            time.sleep(2)
    if any(frames[0][name] == frames[1][name] for name in frames[0]):
        raise RuntimeError("A public camera is not advancing")
    for name in frames[0]:
        code, body = public_get(origin + f"/snapshot/{name}.jpg", credential)
        if code != 200 or not body.startswith(b"\xff\xd8") or not body.endswith(b"\xff\xd9"):
            raise RuntimeError("A public camera image is incomplete")
    return {**result, "healthy": True, "basic_auth": True, "private_routes_blocked": True,
            "cameras_advancing": True, "attendee_chat_ready": True}


def disable(repo):
    owned = owned_units(repo)
    names = [name + ".service" for name in reversed(SERVICES) if owned[name]["installed"]]
    if names:
        run(["systemctl", "--user", "disable", "--now", *names], timeout=450)
    return {"enabled": False, "services": owned_units(repo)}


def serve(repo):
    def interrupted(_number, _frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    module = browser_module(repo)
    try:
        command = ["/usr/bin/python3", str(repo / "scripts/desktop.py"), "launch", "--repo", str(repo), "--headless", "--no-open"]
        if subprocess.run(command, cwd=repo).returncode:
            raise RuntimeError("The demo did not reach READY")
        failures = 0
        while True:
            time.sleep(10)
            failures = 0 if module.ready(repo) else failures + 1
            if failures >= 3:
                raise RuntimeError("The demo health check failed; restarting the owned stack")
    except KeyboardInterrupt:
        pass
    finally:
        subprocess.run([str(repo / "run.sh"), "down"], cwd=repo, timeout=390)


def ngrok_service(repo):
    directory = private_root(repo)
    settings = json.loads((directory / "settings.json").read_text())
    token = json.loads((directory / "ngrok.json").read_text())["agent"]["authtoken"]
    auth = json.loads((directory / "auth.json").read_text())
    command = [settings["ngrok"], "http", VISITOR_ORIGIN, "--url", "https://" + settings["domain"],
               "--config", str(directory / "ngrok.json"), "--inspect=false", "--log=stdout", "--log-format=json"]
    child = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    def interrupted(_number, _frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        for line in child.stdout:
            for secret in (token, auth["password"], base64.b64encode((auth["username"] + ":" + auth["password"]).encode()).decode()):
                line = line.replace(secret, "[private]")
            print(line.rstrip(), flush=True)
        return child.wait()
    except KeyboardInterrupt:
        return 0
    finally:
        if child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("enable", "check", "disable", "_serve", "_ngrok"))
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--domain")
    parser.add_argument("--auth-file", type=Path)
    parser.add_argument("--ngrok", type=Path)
    credentials = parser.add_mutually_exclusive_group()
    credentials.add_argument("--ngrok-config", type=Path)
    credentials.add_argument("--token-file", type=Path)
    args = parser.parse_args()
    if args.operation == "enable" and (not args.auth_file or not args.ngrok or not args.domain or not (args.ngrok_config or args.token_file)):
        parser.error("enable requires --domain, --auth-file, --ngrok, and --ngrok-config or --token-file")
    repo = root_path(args.repo)
    if args.operation == "_serve":
        serve(repo)
        return 0
    if args.operation == "_ngrok":
        return ngrok_service(repo)
    result = enable(repo, args) if args.operation == "enable" else check(repo) if args.operation == "check" else disable(repo)
    print(json.dumps(result, indent=2))
    if args.operation == "check":
        if result["healthy"]:
            print("PUBLIC READY")
        return 0 if result["healthy"] else 1
    print("PUBLIC ENABLED" if args.operation == "enable" else "PUBLIC STOPPED")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, KeyError, subprocess.SubprocessError) as error:
        print(f"[public] {type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(1)
