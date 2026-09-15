#!/usr/bin/env python3
"""Install an authenticated ngrok visitor service on the prepared Brev host."""
from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path
import pwd
import re
import secrets
import shlex
import shutil
import subprocess
import tarfile
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener, urlopen

HERE = Path(__file__).resolve().parent
DOWNLOAD = "https://bin.ngrok.com/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.tgz"
VISITOR_FILES = ("visitor.py", "visitor.html", "visitor.css", "visitor.js",
                 "visitor_video.py", "visitor-player.js", "staff.html", "staff.css")
VISITOR_ORIGIN = "http://127.0.0.1:8093"
SERVICES = ("paai-visitor", "paai-ngrok")
UNIT_DIRECTORY = Path("/etc/systemd/system")


def visitor_url(tunnels):
    urls = set()
    for row in tunnels:
        if (row.get("proto") != "https" or row.get("config", {}).get("addr")
                not in (VISITOR_ORIGIN, "127.0.0.1:8093")):
            continue
        url = row.get("public_url", "")
        parsed = urlsplit(url)
        if (parsed.scheme != "https" or not parsed.hostname or parsed.username is not None
                or parsed.password is not None or parsed.port not in (None, 443)
                or parsed.path not in ("", "/") or parsed.query or parsed.fragment):
            raise ValueError("The visitor tunnel returned an invalid HTTPS origin")
        urls.add(url.rstrip("/"))
    if len(urls) > 1:
        raise ValueError("More than one public tunnel targets the visitor")
    return next(iter(urls), None)


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, file, code, message, headers, new_url):
        return None


def public_get(url, authorization, deadline):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("Public visitor verification timed out")
    headers = {"ngrok-skip-browser-warning": "1"}
    if authorization:
        headers["Authorization"] = authorization
    try:
        response = build_opener(NoRedirect()).open(
            Request(url, headers=headers), timeout=min(5, remaining))
    except HTTPError as error:
        response = error
    with response:
        body = bytearray()
        while len(body) < 4 * 1024 * 1024:
            if time.monotonic() >= deadline:
                raise TimeoutError("Public visitor verification timed out")
            chunk = response.read1(min(65536, 4 * 1024 * 1024 - len(body)))
            if not chunk:
                break
            body.extend(chunk)
        return response.status, bytes(body)


def verify_visitor(url, auth):
    deadline = time.monotonic() + 30
    authorization = "Basic " + base64.b64encode(
        (auth["username"] + ":" + auth["password"]).encode()).decode()
    for path in ("/", "/staff/"):
        if public_get(url + path, None, deadline)[0] != 401:
            raise RuntimeError("The public visitor did not require authentication")
    status, body = public_get(url + "/", authorization, deadline)
    if status != 200 or body != (HERE / "visitor.html").read_bytes():
        raise RuntimeError("The public URL did not serve the installed visitor page")
    for path, filename in (("/staff/", "staff.html"), ("/staff/style.css", "staff.css")):
        status, body = public_get(url + path, authorization, deadline)
        if status != 200 or body != (HERE / filename).read_bytes():
            raise RuntimeError("The public URL did not serve the installed staff page")
    status, body = public_get(url + "/api/status", authorization, deadline)
    if status != 200:
        raise RuntimeError("The public camera status is unavailable")
    cameras = json.loads(body).get("cameras", [])
    if (len(cameras) != 3 or {row.get("name") for row in cameras} != {"kitchen", "worktop", "side"}
            or not all(row.get("online") is True and type(row.get("frame_id")) is int for row in cameras)):
        raise RuntimeError("The three public cameras are not ready")
    status, body = public_get(url + "/snapshot/worktop.jpg", authorization, deadline)
    if status != 200 or not body.startswith(b"\xff\xd8") or not body.endswith(b"\xff\xd9"):
        raise RuntimeError("The public camera did not return a complete JPEG")
    for path in ("/openclaw/", "/api/openclaw-bootstrap", "/__openclaw/", "/api/control"):
        if public_get(url + path, authorization, deadline)[0] != 404:
            raise RuntimeError("The public visitor did not exclude administrative routes")


def run(command, *, input=None, timeout=30):
    try:
        result = subprocess.run(command, input=input, text=True, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise RuntimeError("Service configuration timed out; private output withheld") from None
    if result.returncode:
        raise RuntimeError("Service configuration failed; private output withheld")
    return result.stdout


def private_json(path, value):
    if path.is_symlink():
        raise ValueError("Private configuration must not be a symbolic link")
    contents = json.dumps(value, indent=2) + "\n"
    try:
        changed = json.loads(path.read_text()) != value
    except (FileNotFoundError, ValueError):
        changed = True
    if changed:
        path.write_text(contents)
    path.chmod(0o600)
    return changed


def process_settings(unit):
    section, values = "", []
    for line in unit.splitlines():
        line = line.strip()
        if line.startswith("["):
            section = line
        elif section == "[Service]" and "=" in line:
            name, value = line.split("=", 1)
            if name not in ("Restart", "RestartSec"):
                values.append((name, value))
    return sorted(values)


def ngrok_config(path, token):
    if path.is_symlink():
        raise ValueError("Private configuration must not be a symbolic link")
    # Preserve ngrok's own token-only YAML when adopting an active tunnel.
    previous = path.read_text() if path.exists() else ""
    native = r"\s*version:\s*['\"]?3['\"]?\s*\nagent:\s*\n[ \t]+authtoken:\s*"
    if re.fullmatch(native + re.escape(token) + r"\s*", previous):
        path.chmod(0o600)
        return False
    return private_json(path, {"version": "3", "agent": {
        "authtoken": token, "web_addr": "127.0.0.1:4040"}})


def deployment_root(root):
    root = root.resolve(strict=True)
    if (os.getuid() == 0 or not re.fullmatch(r"/[A-Za-z0-9_./-]+", str(root))
            or not (root / "data/state/deployment.json").is_file()):
        raise ValueError("Run as the unprivileged owner of a prepared deployment")
    return root


def owned_services(root):
    username = pwd.getpwuid(os.getuid()).pw_name
    prefixes = {
        "paai-visitor": ["/usr/bin/python3", str(root / "tools/visitor/visitor.py")],
        "paai-ngrok": [str(root / "tools/ngrok/ngrok"), "http", VISITOR_ORIGIN],
    }
    services = {}
    for name in SERVICES:
        output = run(["systemctl", "show", name + ".service", "--no-pager",
                      "--property=LoadState,ActiveState,FragmentPath,DropInPaths"])
        properties = dict(line.split("=", 1) for line in output.splitlines() if "=" in line)
        if properties.get("DropInPaths"):
            raise ValueError("Visitor service overrides require an explicit ownership review")
        path = properties.get("FragmentPath")
        installed = bool(path)
        if installed:
            contents = Path(path).read_text()
            entries = dict(line.split("=", 1) for line in contents.splitlines() if "=" in line)
            command = shlex.split(entries.get("ExecStart", ""))
            if (entries.get("User") != username or command[:len(prefixes[name])] != prefixes[name]
                    or Path(path).resolve() != (UNIT_DIRECTORY / (name + ".service")).resolve()):
                raise ValueError("A visitor service name belongs to another deployment")
        elif properties.get("LoadState") != "not-found":
            raise ValueError("The visitor service ownership could not be established")
        services[name] = {"installed": installed,
                          "active": properties.get("ActiveState") == "active",
                          "state": properties.get("ActiveState", "unknown")}
    return services


def current_url():
    try:
        with urlopen("http://127.0.0.1:4040/api/tunnels", timeout=3) as response:
            return visitor_url(json.load(response).get("tunnels", []))
    except (URLError, OSError, ValueError):
        return None


def status(root):
    root = deployment_root(root)
    services = owned_services(root)
    result = {"installed": all(row["installed"] for row in services.values()),
              "healthy": False, "url": None, "services": services}
    if not result["installed"] or not all(row["active"] for row in services.values()):
        return result
    result["url"] = current_url()
    if not result["url"]:
        result["reason"] = "The visitor tunnel is unavailable"
        return result
    try:
        auth_file = root / "data/private/public-visitor/auth.json"
        verify_visitor(result["url"], json.loads(auth_file.read_text()))
        result["healthy"] = True
    except (RuntimeError, OSError, ValueError, KeyError):
        result["reason"] = "The authenticated public visitor health check did not pass"
    return result


def wait_ready(root):
    deadline = time.monotonic() + 45
    while time.monotonic() < deadline:
        result = status(root)
        if result["healthy"]:
            return result
        time.sleep(1)
    raise RuntimeError("The public visitor did not become healthy within its startup deadline")


def service_action(root, operation):
    root = deployment_root(root)
    services = owned_services(root)
    names = [name for name, row in services.items() if row["installed"]]
    if operation == "stop":
        if names:
            run(["sudo", "-n", "systemctl", "stop", *reversed(names)], timeout=45)
        return status(root)
    if operation not in ("start", "restart"):
        raise ValueError("Unsupported visitor service operation")
    if (root / "source/STOP").exists() or (HERE.parents[1] / "STOP").exists():
        raise InterruptedError("Repository STOP exists")
    if len(names) != len(SERVICES):
        raise ValueError("The public visitor is not installed; run install first")
    run(["sudo", "-n", "systemctl", operation, *names], timeout=45)
    return wait_ready(root)


def install(root, token_file, *, local_auth=False, video=False):
    os.umask(0o077)
    root = deployment_root(root)
    if (root / "source/STOP").exists() or (HERE.parents[1] / "STOP").exists():
        raise InterruptedError("Repository STOP exists")
    services = owned_services(root)
    if (token_file.is_symlink() or not token_file.is_file()
            or token_file.stat().st_mode & 0o077 or token_file.stat().st_uid != os.getuid()):
        raise ValueError("The ngrok token file must be private (mode 0600)")
    token = token_file.read_text().strip()
    if not token or any(character.isspace() for character in token):
        raise ValueError("The ngrok token file is invalid")
    if video and not shutil.which("ffmpeg"):
        run(["sudo", "-n", "apt-get", "install", "-y", "--no-install-recommends", "ffmpeg"], timeout=180)
        if not shutil.which("ffmpeg"):
            raise RuntimeError("FFmpeg is required for the public video path")
    private = root / "data/private/public-visitor"
    if not private.resolve().is_relative_to(root):
        raise ValueError("Visitor private configuration must remain inside the deployment")
    private.mkdir(parents=True, mode=0o700, exist_ok=True)
    private.chmod(0o700)
    executable = root / "tools/ngrok/ngrok"
    if not executable.resolve().is_relative_to(root):
        raise ValueError("The ngrok executable must remain inside the deployment")
    if not executable.is_file():
        executable.parent.mkdir(parents=True, exist_ok=True)
        archive = executable.parent / "ngrok.tgz"
        with urlopen(DOWNLOAD, timeout=45) as response, archive.open("wb") as output:
            shutil.copyfileobj(response, output)
        with tarfile.open(archive) as bundle:
            member = bundle.getmember("ngrok")
            if not member.isfile() or member.size > 128 * 1024 * 1024:
                raise ValueError("Unexpected ngrok archive member")
            with bundle.extractfile(member) as source, executable.open("wb") as output:
                shutil.copyfileobj(source, output)
        executable.chmod(0o755)
    config = private / "ngrok.yml"
    config_changed = ngrok_config(config, token)
    run([str(executable), "config", "check", "--config", str(config)])
    auth_file = private / "auth.json"
    if auth_file.is_symlink():
        raise ValueError("Private configuration must not be a symbolic link")
    if not auth_file.exists():
        private_json(auth_file, {"username": "visitor", "password": secrets.token_urlsafe(24)})
    auth_file.chmod(0o600)
    auth = json.loads(auth_file.read_text())
    policy_changed = private_json(private / "policy.json", {"on_http_request": [{"actions": [{"type": "basic-auth",
        "config": {"credentials": [auth["username"] + ":" + auth["password"]]}}]}]})
    visitor = root / "tools/visitor"
    if not visitor.resolve().is_relative_to(root):
        raise ValueError("Visitor source must remain inside the deployment")
    visitor.mkdir(parents=True, exist_ok=True)
    visitor_changed = False
    for name in (*VISITOR_FILES, "public.py"):
        destination = visitor / name
        contents = (HERE / name).read_bytes()
        if destination.is_symlink():
            raise ValueError("Visitor source must not be a symbolic link")
        if not destination.exists() or destination.read_bytes() != contents:
            destination.write_bytes(contents)
            visitor_changed = visitor_changed or name != "public.py"
    username = pwd.getpwuid(os.getuid()).pw_name
    auth_argument = f" --auth-file={auth_file}"
    video_argument = " --video" if video else ""
    policy_argument = "" if local_auth else f" --traffic-policy-file={private}/policy.json"
    commands = {
        "paai-visitor": f"/usr/bin/python3 {visitor}/visitor.py{auth_argument}{video_argument}",
        "paai-ngrok": f"{executable} http {VISITOR_ORIGIN} --config={config}{policy_argument} --inspect=false",
    }
    changed_units, changed_processes = set(), set()
    for name, command in commands.items():
        unit = f"""[Unit]
Description=Physical Agentic AI {name}
After=network-online.target paai-demo-docker.service
Wants=network-online.target
[Service]
User={username}
ExecStart={command}
Restart=always
RestartSec=5
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
StandardOutput=null
StandardError=null
[Install]
WantedBy=multi-user.target
"""
        unit_path = UNIT_DIRECTORY / (name + ".service")
        if unit_path.is_symlink():
            raise ValueError("Visitor service configuration must not be a symbolic link")
        previous = unit_path.read_text() if unit_path.exists() else ""
        if previous != unit:
            run(["sudo", "-n", "tee", str(unit_path)], input=unit)
            changed_units.add(name)
            if process_settings(previous) != process_settings(unit):
                changed_processes.add(name)
    if changed_units:
        run(["sudo", "-n", "systemctl", "daemon-reload"])
    run(["sudo", "-n", "systemctl", "enable", "--now", *SERVICES])
    changed = {"paai-visitor": visitor_changed,
               "paai-ngrok": config_changed or (policy_changed and not local_auth)}
    restarted = []
    for name in SERVICES:
        if services[name]["active"] and (changed[name] or name in changed_processes):
            run(["sudo", "-n", "systemctl", "restart", name], timeout=45)
            restarted.append(name)
    result = wait_ready(root)
    return {**result, "authentication": "local basic auth" if local_auth else "local and ngrok basic auth",
            "visitor_verified": True, "credentials_file": str(auth_file), "restarted": restarted}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", nargs="?", default="install",
                        choices=("install", "start", "status", "restart", "stop"))
    parser.add_argument("--root", type=Path, default=Path("/opt/dlami/nvme/paai-demo"))
    parser.add_argument("--token-file", type=Path)
    parser.add_argument("--local-auth", action="store_true", help="Omit the ngrok policy when the account does not support it; local auth is always enforced")
    parser.add_argument("--video", action="store_true", help="Enable shared H.264 hardware encoding for visitors")
    args = parser.parse_args()
    if args.operation == "install":
        if args.token_file is None:
            parser.error("install requires --token-file")
        result = install(args.root, args.token_file, local_auth=args.local_auth, video=args.video)
    elif args.operation == "status":
        result = status(args.root)
    else:
        result = service_action(args.root, args.operation)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
