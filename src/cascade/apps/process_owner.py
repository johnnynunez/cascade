"""Local launch ownership. Stdlib-only; never discover owners by scanning ps.

CASCADE_LAUNCH_STATE is a *root*: unprofiled/ or profile-<OpenClaw profile>/
contains owner.json, processes/mcp_*.json and named service receipts. A PID
alone is never authority: both its kernel birth identity and full command
must still match a receipt belonging to this exact repo/state/profile.
"""
from __future__ import annotations

import errno
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import uuid


def profile_state_dir(root, profile: str) -> Path:
    if profile and (not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", profile)):
        raise ValueError("invalid OpenClaw profile name")
    return Path(root).expanduser().resolve() / ("profile-" + profile if profile else "unprofiled")


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    with temp.open("x") as stream:
        os.chmod(temp, 0o600)
        json.dump(data, stream)
        stream.write("\n")
    temp.replace(path)


def load_owner(state_dir, repo, profile: str, *, create: bool = False) -> dict | None:
    state_dir, repo = Path(state_dir).resolve(), Path(repo).resolve()
    path = state_dir / "owner.json"
    expected = {"repo": str(repo), "profile": profile, "state_dir": str(state_dir)}
    if not path.exists():
        if not create:
            return None
        _write_json(path, {**expected, "owner": uuid.uuid4().hex, "schema": 1})
    data = json.loads(path.read_text())
    if (not isinstance(data, dict) or data.get("schema") != 1
            or not re.fullmatch(r"[0-9a-f]{32}", data.get("owner", ""))
            or any(data.get(key) != value for key, value in expected.items())):
        raise ValueError("launch owner does not match repo/state/profile")
    return data


def process_identity(pid: int) -> dict | None:
    if type(pid) is not int or pid <= 1:
        return None
    try:
        if sys.platform.startswith("linux"):
            proc = Path("/proc") / str(pid)
            stat = (proc / "stat").read_text().rsplit(")", 1)[1].split()
            if stat[0] == "Z" or proc.stat().st_uid != os.getuid():
                return None
            birth = Path("/proc/sys/kernel/random/boot_id").read_text().strip() + ":" + stat[19]
            command = (proc / "cmdline").read_bytes().rstrip(b"\0").replace(b"\0", b" ").decode()
        elif sys.platform == "darwin":
            import ctypes
            import struct

            # PROC_PIDTBSDINFO's start timeval has microseconds. ps lstart's
            # calendar seconds can collide when a PID is reused quickly.
            buf = ctypes.create_string_buffer(136)  # struct proc_bsdinfo
            lib = ctypes.CDLL("/usr/lib/libproc.dylib")
            if lib.proc_pidinfo(pid, 3, 0, buf, len(buf)) != len(buf):
                return None
            sec, usec = struct.unpack_from("=QQ", buf.raw, 120)
            result = subprocess.run(["ps", "-ww", "-p", str(pid), "-o", "uid=,stat=,command="],
                                    capture_output=True, text=True, timeout=5)
            parts = result.stdout.strip().split(None, 2)
            if result.returncode or len(parts) != 3 or int(parts[0]) != os.getuid() or parts[1].startswith("Z"):
                return None
            # Refuse a PID that changed between the kernel and command reads.
            if lib.proc_pidinfo(pid, 3, 0, buf, len(buf)) != len(buf) or struct.unpack_from("=QQ", buf.raw, 120) != (sec, usec):
                return None
            birth, command = f"darwin:{sec}:{usec}", parts[2]
        else:
            return None  # no unverified platform fallback for signalling PIDs
        return {"pid": pid, "birth": birth, "command": command} if command else None
    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
        return None


def is_live(record: dict, owner: dict) -> bool:
    if not isinstance(record, dict) or any(record.get(k) != owner.get(k) for k in ("owner", "repo", "profile", "state_dir")):
        return False
    current = process_identity(record.get("pid"))
    if current is None or any(record.get(k) != current[k] for k in ("pid", "birth", "command")):
        return False
    if record.get("role") == "mcp":
        if not re.search(r"(?:^|\s)cascade\.apps\.mcp_server(?:\s|$)", current["command"]):
            return False
        if not re.search(r"(?:^|\s)--launch-owner " + re.escape(owner["owner"]) + r"(?:\s|$)", current["command"]):
            return False
    if record.get("role") == "gateway_child":
        try:
            if record.get("process_group") != record["pid"] or os.getpgid(record["pid"]) != record["pid"]:
                return False
        except ProcessLookupError:
            return False
    return True


def register_process(state_dir, owner: dict, pid: int, role: str, *, run_dir=None) -> dict:
    identity = process_identity(pid)
    if identity is None:
        raise ValueError(f"cannot register dead/unreadable process {pid}")
    if Path(state_dir).resolve() != Path(owner["state_dir"]):
        raise ValueError("wrong owner state directory")
    if not re.fullmatch(r"[a-z][a-z0-9_]*", role):
        raise ValueError("invalid process role")
    record = {**owner, **identity, "role": role, "instance_id": uuid.uuid4().hex, "registered_at": time.time()}
    if role == "gateway_child":
        if os.getpgid(pid) != pid:
            raise ValueError("foreground gateway must own a private process group")
        record["process_group"] = pid
    if run_dir is not None:
        record["run_dir"] = str(Path(run_dir).resolve())
    if not is_live(record, owner):
        raise ValueError("process command does not identify its launch owner")
    name = "processes/mcp_" + record["instance_id"] + ".json" if role == "mcp" else ("gateway.started" if role == "gateway" else role + ".pid")
    _write_json(Path(state_dir) / name, record)
    return record


def records(state_dir) -> list[dict]:
    state = Path(state_dir)
    paths = list((state / "processes").glob("mcp_*.json")) + list(state.glob("*.pid"))
    if (state / "gateway.started").is_file():
        paths.append(state / "gateway.started")
    found = []
    for path in paths:
        try:
            record = json.loads(path.read_text())
            if isinstance(record, dict):
                found.append(record)
        except (OSError, ValueError):
            continue  # legacy bare PID files are deliberately not authority
    return found


def live_records(state_dir, owner: dict, *, role: str | None = None) -> list[dict]:
    return [r for r in records(state_dir) if (role is None or r.get("role") == role) and is_live(r, owner)]


def _run_gateway_stop(command: list[str], *, timeout_s: float = 360.0,
                      cleanup_s: float = 5.0) -> None:
    """Bound the CLI and its respawned children in a dedicated process group.

    OpenClaw's managed stop permits 315 seconds to drain active work and its
    installed systemd unit grants 330 seconds. A 30-second caller deadline
    interrupts a legitimate stop. The group contains only this spawned CLI;
    service-manager-owned gateway processes retain the owner checks below.
    """
    import signal

    proc = subprocess.Popen(command, start_new_session=True)
    handlers = {}

    def interrupted(number, frame):
        raise InterruptedError(f"gateway stop interrupted by signal {number}")

    def group_signal(number):
        try:
            os.killpg(proc.pid, number)
        except ProcessLookupError:
            pass

    try:
        for number in (signal.SIGTERM, signal.SIGHUP):
            handlers[number] = signal.signal(number, interrupted)
        returncode = proc.wait(timeout=timeout_s)
        if returncode:
            raise subprocess.CalledProcessError(returncode, command)
    finally:
        # Finish children even when the node launcher exits before its respawn,
        # or the caller is interrupted. No process-name scan is involved.
        for number in handlers:
            signal.signal(number, signal.SIG_IGN)
        group_signal(signal.SIGTERM)
        deadline = time.monotonic() + cleanup_s
        while time.monotonic() < deadline:
            try:
                os.killpg(proc.pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.05)
        group_signal(signal.SIGKILL)
        try:
            proc.wait(timeout=cleanup_s)
        finally:
            for number, previous in handlers.items():
                signal.signal(number, previous)


def _darwin_group_snapshot(pgid: int) -> dict:
    """Check only an already verified private group; never discover owners."""
    command = ["/bin/ps", "-ww", "-x", "-g", str(pgid), "-o", "pid=,pgid=,stat="]
    snapshot = {"command": command, "monotonic": time.monotonic(), "complete": False,
                "truncated": False, "no_live_members": False, "members": []}
    try:
        # Darwin ignores -g in legacy mode. One -g selector uses KERN_PROC_PGRP;
        # -x includes members without a terminal, and -ww prevents row clipping.
        result = subprocess.run(command, capture_output=True, text=True, timeout=5,
                                env={"COMMAND_MODE": "unix2003", "LC_ALL": "C"})
    except (OSError, UnicodeError, subprocess.TimeoutExpired) as error:
        snapshot.update(error=str(error), errno=getattr(error, "errno", None),
                        truncated=isinstance(error, (UnicodeError, subprocess.TimeoutExpired)))
        for name in ("stdout", "stderr"):
            output = getattr(error, name, None)
            snapshot[name] = output.decode("utf-8", errors="replace") if isinstance(output, bytes) else output
        return snapshot
    snapshot.update(complete=True, returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
    # Darwin ps can exit zero on a sysctl error, with the failure on stderr.
    if result.stderr:
        return snapshot
    if result.returncode == 1 and result.stdout == "":
        snapshot["no_live_members"] = True
        return snapshot
    if result.returncode != 0 or not result.stdout or not result.stdout.endswith("\n"):
        return snapshot
    seen = set()
    for line in result.stdout.splitlines():
        row = re.fullmatch(r"\s*([0-9]+)\s+([0-9]+)\s+([IRSTUZ][+<>AELNSVWXs]*)\s*", line)
        if row is None:
            return snapshot
        pid, group, state = int(row[1]), int(row[2]), row[3]
        if pid <= 1 or group != pgid or pid in seen:
            return snapshot
        seen.add(pid)
        snapshot["members"].append({"pid": pid, "pgid": group, "state": state})
    snapshot["no_live_members"] = all(member["state"].startswith("Z") for member in snapshot["members"])
    return snapshot


def stop_private_gateway(record: dict, owner: dict, *, timeout_s: float = 5) -> None:
    """Stop only a previously verified foreground gateway and its private group."""
    import signal

    if record.get("role") != "gateway_child" or not is_live(record, owner):
        raise ValueError("foreground gateway no longer belongs to this launch owner")
    pid = record["pid"]
    diagnostic = ({"pid": pid, "process_group": pid, "birth": record["birth"],
                   "term_monotonic": time.monotonic()} if sys.platform == "darwin" else None)
    try:
        os.killpg(pid, signal.SIGTERM)
        deadline = time.monotonic() + timeout_s
        while is_live(record, owner) and time.monotonic() < deadline:
            time.sleep(.05)
        # The wrapper can exit before its child. This group was checked above
        # and contains only the foreground gateway started by this checkout.
        current = process_identity(pid)
        if current is not None and any(current[key] != record[key] for key in ("birth", "command")):
            raise ValueError("foreground gateway identity changed while stopping")
        if sys.platform == "darwin":
            # Reap our exited child before signalling a possible zombie-only
            # group. Surviving workers still need the group SIGKILL below.
            diagnostic["waitpid"] = {"monotonic": time.monotonic()}
            try:
                diagnostic["waitpid"]["result"] = os.waitpid(pid, os.WNOHANG)
            except ChildProcessError as error:
                # Another launcher process created this gateway.
                diagnostic["waitpid"].update(errno=error.errno, error=str(error))
            diagnostic["kill_monotonic"] = time.monotonic()
        try:
            os.killpg(pid, signal.SIGKILL)
        except PermissionError as error:
            if sys.platform != "darwin" or error.errno != errno.EPERM:
                raise
            # Darwin can return EPERM for a zombie-only group. A missing leader
            # alone proves nothing: a TERM-ignoring worker may still be alive.
            diagnostic["query"] = _darwin_group_snapshot(pid)
            print("gateway group SIGKILL EPERM: " + json.dumps(diagnostic), file=sys.stderr)
            if not diagnostic["query"]["no_live_members"]:
                raise
    except ProcessLookupError:
        pass


def stop_owned(state_dir, owner: dict, *, dry_run=False, roles=None) -> list[int]:
    import signal

    stopped = []
    # Service-manager shutdown is intentionally separate from direct PID
    # signals. An unowned gateway is never stopped to reap its MCP children.
    for record in sorted(records(state_dir), key=lambda r: r.get("role") in ("gateway", "gateway_child")):
        if roles is not None and record.get("role") not in roles:
            continue
        if not is_live(record, owner):
            continue
        if dry_run:
            print(f"would stop {record['role']} pid={record['pid']}")
            continue
        if record.get("role") == "gateway_child":
            stop_private_gateway(record, owner)
        elif record.get("role") == "gateway":
            prefix = ["openclaw", *(["--profile", owner["profile"]] if owner["profile"] else [])]
            status = subprocess.run([*prefix, "gateway", "status", "--json"], capture_output=True, text=True, timeout=30)
            pid = json.loads(status.stdout).get("service", {}).get("runtime", {}).get("pid") if status.returncode == 0 else None
            if pid != record["pid"] or not is_live(record, owner):
                raise ValueError("gateway no longer belongs to this launch owner; refusing stop")
            _run_gateway_stop([*prefix, "gateway", "stop", "--force"])
        else:
            # Re-check immediately before signalling; a bare/stale PID marker
            # or the same command under a different owner is not sufficient.
            if not is_live(record, owner):
                continue
            try:
                os.kill(record["pid"], signal.SIGTERM)
            except ProcessLookupError:
                pass
        deadline = time.monotonic() + 10
        while is_live(record, owner) and time.monotonic() < deadline:
            time.sleep(0.05)
        if is_live(record, owner):
            raise ValueError(f"{record['role']} pid={record['pid']} did not stop; receipt retained")
        stopped.append(record["pid"])
    return stopped


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--profile", default="")
    parser.add_argument("action", choices=["state-dir", "init", "record", "record-gateway", "owns-gateway", "down"])
    parser.add_argument("--pid", type=int)
    parser.add_argument("--role")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        state = profile_state_dir(args.state_root, args.profile)
        if args.action == "state-dir":
            print(state)
            return 0
        owner = load_owner(state, args.repo, args.profile, create=args.action == "init")
        if owner is None:
            if args.action == "owns-gateway":
                return 1
            if args.action == "down":
                print("no owner receipt for this repo/profile; nothing stopped")
                return 0
            raise ValueError("missing owner receipt: initialize launch ownership first")
        if args.action == "init":
            print(owner["owner"])
        elif args.action == "owns-gateway":
            return 0 if live_records(state, owner, role="gateway") else 1
        elif args.action == "down":
            print(json.dumps({"stopped": stop_owned(state, owner, dry_run=args.dry_run)}))
        else:
            if args.action == "record-gateway":
                prefix = ["openclaw", *(["--profile", args.profile] if args.profile else [])]
                result = subprocess.run([*prefix, "gateway", "status", "--json"], check=True, capture_output=True, text=True, timeout=30)
                args.pid = json.loads(result.stdout).get("service", {}).get("runtime", {}).get("pid")
                args.role = "gateway"
            if not args.pid or not args.role:
                raise ValueError("record needs a live PID and service role")
            print(json.dumps(register_process(state, owner, args.pid, args.role)))
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        print(str(exc), file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
