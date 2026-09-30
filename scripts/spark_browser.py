#!/usr/bin/env python3
"""Owned Spark camera/visitor processes and a dedicated Chromium window."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import pwd
import select
import shutil
import socket
import subprocess
import sys
import time
from urllib.request import build_opener, ProxyHandler

CHAT_URL = "http://127.0.0.1:8092/"
CAMERA_URL = "http://127.0.0.1:8091"

# The final interpreter acknowledges its own startup over a private inherited
# pipe. Popen may return while /proc/<pid>/cmdline is still empty during exec.
# run_path keeps this interpreter/PID; argv and the script import directory
# match direct `python script.py ...` execution before the application starts.
_SURFACE_BOOTSTRAP = """import os, runpy, sys
ready_fd = int(sys.argv[1])
script = sys.argv[2]
sys.argv = sys.argv[2:]
sys.path[0] = os.path.dirname(os.path.abspath(script))
os.write(ready_fd, b'R')
os.close(ready_fd)
runpy.run_path(script, run_name='__main__')
"""


def runtime_home(repo):
    record = json.loads((Path(repo) / "runs/.install/install.json").read_text())
    return str(Path(record.get("runtime_home", Path.home())).resolve())


def ownership(repo):
    from cascade.apps.process_owner import load_owner, profile_state_dir
    repo = Path(repo).resolve()
    state = profile_state_dir(os.environ.get("CASCADE_LAUNCH_STATE", repo / "runs/.launch"), "cascade-demo")
    return state, load_owner(state, repo, "cascade-demo")


def get_json(url):
    with build_opener(ProxyHandler({})).open(url, timeout=5) as response:
        return json.load(response)


def ready(repo, *, surfaces=True):
    """Require this checkout's live identities and current proof before attach."""
    from cascade.apps.process_owner import is_live, live_records
    try:
        state, owner = ownership(repo)
        if owner is None:
            return False
        proof = json.loads((state / "proof.json").read_text())
        if (proof.get("verified") is not True or proof.get("sim") != "isaac"
                or not is_live(proof.get("process"), owner)):
            return False
        roles = {row["role"] for row in live_records(state, owner)}
        if not {"qwen", "isaac_bridge", "gateway_child"} <= roles:
            return False
        if not surfaces:
            return True
        if not {"cameras", "visitor"} <= roles:
            return False
        cams = get_json(CAMERA_URL + "/state").get("cameras", {})
        if set(cams) != {"kitchen", "worktop", "side"} or not all(row.get("online") for row in cams.values()):
            return False
        chat = get_json(CHAT_URL + "api/chat")
        return chat.get("enabled") is True and chat.get("ready") is True and chat.get("agent") == "cascade-demo"
    except (OSError, ValueError, KeyError):
        return False


def _spawn_surface(command, repo, env, log, *, startup_timeout_s=10):
    """Wait for this child interpreter before admitting its process identity."""
    reader, writer = os.pipe()
    process = None
    try:
        process = subprocess.Popen([command[0], "-c", _SURFACE_BOOTSTRAP, str(writer), *command[1:]],
                                   cwd=repo, env=env, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                                   pass_fds=(writer,))
        os.close(writer)
        writer = None
        readable, _, _ = select.select([reader], [], [], startup_timeout_s)
        if not readable:
            raise RuntimeError(f"Surface interpreter readiness timed out; see {log.name}")
        acknowledgment = os.read(reader, 1)
        if not acknowledgment:
            raise RuntimeError(f"Surface process exited before interpreter readiness; see {log.name}")
        if acknowledgment != b"R":
            raise RuntimeError(f"Invalid surface interpreter readiness acknowledgment; see {log.name}")
        if process.poll() is not None:
            raise RuntimeError(f"Surface process exited before ownership registration; see {log.name}")
        return process
    except BaseException:
        if process is not None:
            if process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        raise
    finally:
        os.close(reader)
        if writer is not None:
            os.close(writer)


def start_surfaces(repo, env):
    from cascade.apps.process_owner import live_records, register_process
    repo = Path(repo).resolve()
    state, owner = ownership(repo)
    if owner is None:
        raise RuntimeError("Spark stack has no ownership receipt")
    created = []
    try:
        for role, port, command in (
            ("cameras", 8091, ["/usr/bin/python3", str(repo / "scripts/spark_cameras.py")]),
            ("visitor", 8092, ["/usr/bin/python3", str(repo / "deploy/brev/visitor.py"),
                                "--port", "8092", "--camera-origin", CAMERA_URL, "--repo", str(repo)]),
        ):
            if live_records(state, owner, role=role):
                continue
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=.2):
                    raise RuntimeError(f"Port {port} belongs to another process; it was preserved")
            except (ConnectionRefusedError, TimeoutError):
                pass
            with (state / (role + ".log")).open("ab") as log:
                process = _spawn_surface(command, repo, env, log)
            created.append(process)
            register_process(state, owner, process.pid, role)
        deadline = time.monotonic() + 90
        while not ready(repo):
            if any(process.poll() is not None for process in created):
                raise RuntimeError(f"Camera/chat startup failed; see {state}/cameras.log and visitor.log")
            if time.monotonic() >= deadline:
                raise RuntimeError(f"Camera/chat readiness timed out; see {state}")
            time.sleep(.5)
    except BaseException:
        for process in created:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=10)
        raise


def browser_plan(repo, env=None, url=CHAT_URL):
    """Keep snap data inside its writable area and personal profiles separate."""
    repo = Path(repo).resolve()
    env = dict(os.environ if env is None else env)
    browser = next((str(path) for name in ("chromium", "chromium-browser")
                    if (path := shutil.which(name))), None)
    if browser is None and Path("/snap/bin/chromium").is_file():
        browser = "/snap/bin/chromium"
    if browser is None:
        raise RuntimeError("Chromium is missing. Run the prerequisite command in docs/DGX_SPARK_SETUP.md")
    actual_home = Path(pwd.getpwuid(os.getuid()).pw_dir)
    # Ubuntu's chromium-browser wrapper also invokes the Chromium snap.
    snap = str(Path(browser).resolve()).startswith("/snap/") or Path(browser).name == "chromium-browser" or browser == "/snap/bin/chromium"
    key = hashlib.sha256(str(repo).encode()).hexdigest()[:12]
    data = actual_home / "snap/chromium/common/paai" / key if snap else repo / "runs/.browser"
    profile, extension = data / "profile", data / "extension"
    if snap:
        env["HOME"] = str(actual_home)
    if snap:
        env["XDG_RUNTIME_DIR"] = f"/run/user/{os.getuid()}"
    else:
        env.setdefault("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}")
    bus = Path(env["XDG_RUNTIME_DIR"]) / "bus"
    if bus.exists():
        if snap:
            env["DBUS_SESSION_BUS_ADDRESS"] = "unix:path=" + str(bus)
        else:
            env.setdefault("DBUS_SESSION_BUS_ADDRESS", "unix:path=" + str(bus))
    command = [browser, "--user-data-dir=" + str(profile), "--load-extension=" + str(extension),
               "--no-first-run", "--no-default-browser-check", "--disable-session-crashed-bubble",
               "--new-window", "--window-size=1500,1000", url]
    if env.get("PAAI_BROWSER_DEBUG") == "1":
        command[1:1] = ["--remote-debugging-port=0", "--remote-debugging-address=127.0.0.1"]
    return command, env, data, extension


def open_browser(repo, url=CHAT_URL):
    """Open the attendee chat, or another local page, in the extension profile."""
    if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        raise RuntimeError("A graphical session is required to open PAAI. Use --no-open for service startup")
    command, env, data, extension = browser_plan(repo, url=url)
    data.mkdir(parents=True, exist_ok=True, mode=0o700)
    extension.mkdir(exist_ok=True, mode=0o700)
    source = Path(repo) / "extensions/chrome"
    for path in source.iterdir():
        if path.is_file() and path.suffix in (".json", ".js", ".mjs", ".html", ".css"):
            target = extension / path.name
            if not target.exists() or target.read_bytes() != path.read_bytes():
                shutil.copyfile(path, target)
    # Snap cannot read a private attempt's hidden Xauthority file.
    if env.get("XAUTHORITY") and Path(env["XAUTHORITY"]).is_file():
        auth = data / "Xauthority"
        shutil.copyfile(env["XAUTHORITY"], auth)
        auth.chmod(0o600)
        env["XAUTHORITY"] = str(auth)
    log_path = data / "browser.log"
    with log_path.open("ab") as log:
        log_path.chmod(0o600)
        process = subprocess.Popen(command, cwd=data, env=env, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    if url == CHAT_URL:
        # Other pages can carry a one-time sign-in fragment: never record them.
        receipt = Path(repo) / "runs/.install/browser.json"
        receipt.write_text(json.dumps({"data": str(data), "profile": str(data / "profile"),
            "extension": str(extension), "url": CHAT_URL, "pid": process.pid,
            "debug_enabled": env.get("PAAI_BROWSER_DEBUG") == "1"}) + "\n")
        receipt.chmod(0o600)
    time.sleep(1)
    if process.poll() not in (None, 0):
        raise RuntimeError(f"Chromium exited {process.returncode}; see {log_path}")
    if url == CHAT_URL:
        print("[desktop] PAAI chat and camera extension opened", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    sys.path.insert(0, str(args.repo / "src"))
    if args.check:
        ok = ready(args.repo)
        print("READY" if ok else "NOT READY")
        return 0 if ok else 1
    open_browser(args.repo)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
