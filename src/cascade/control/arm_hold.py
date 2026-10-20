"""Arms a teardown left POWERED and HOLDING, remembered across processes.

#424: a teardown whose park did not bring a torque-cutting arm to rest (a dead camera, the e-stop
every MCP SIGTERM latches, a stop mid-park) keeps that arm's torque on, because cutting it would drop
a raised arm. The NEXT process could not see that: it built a fresh standby LazyArm (the dashboard
said `arm_connected: false`), and the reBot drivers' connect() clears motor faults with a RobStride
type-4 STOP frame before enabling (`robstride.clear_motor_faults`: no torque), so the next start,
arm-check or calibrate dropped the raised arm with nobody holding it (#424 review, Important 1).

So the teardown writes one marker per held arm, keyed by the arm's BUS (the CAN channel of either
reBot transport, the serial port of an SO-101): the physical arm, whichever profile drives it next.
Every process reads it:

- a reBot connect (`RebotRSArm`, `RebotRSMotorBridgeArm`) over a marker refuses BEFORE it opens the
  bus (`check_connect`): no frame at all. It goes ahead only on an operator's confirmation written AT
  THAT MOMENT (`confirm`, valid `CONFIRM_TTL_S` and used once): they support the arm;
- the startup check (`scripts/check_rebot_gripper.py`, run by `~/cascade-demo.sh up`) reads the arm
  with no motor enabled and clears the marker when the arm rests folded at park_q
  (`clear_if_at_rest`): the connect's STOP frame then drops nothing;
- a teardown that parks the arm and cuts its torque, or cuts it on a fresh confirmation, clears it;
- the dashboard /state and MCP `world_state` report it (`arm_held`; `arm_connected` true: the arm
  is powered, never "standby").

An SO-101's connect writes no torque-off (`FeetechArm.connect`): it is never refused, but its hold is
reported until a parked torque-off clears it.

Command line (any shell, any working directory):

    python scripts/arm_hold.py list
    python scripts/arm_hold.py confirm 'holding the arm' [--arm can0]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

HOLD_DIR_ENV = "CASCADE_ARM_HOLD_DIR"
#: The operator's words, the same phrase the teardown's torque cut takes
#: (`apps.demo.TORQUE_OFF_CONFIRMATION`): they are holding the arm.
CONFIRMATION = "holding the arm"
#: A confirmation is valid this long after it was written, and only once:
#: it means "I am supporting the arm NOW", never "for this session".
CONFIRM_TTL_S = 120.0
#: The marker's bus key for each torque-cutting backend: (config key, the
#: backend's own default). Both reBot transports drive one arm on one bus.
_BUS = {"so101": ("port", "/dev/ttyACM0"), "rebot_rs": ("channel", "can0"),
        "rebot_rs_mb": ("channel", "can0")}
_ANY = "_any"


class ArmHeldError(RuntimeError):
    """A connect refused before any frame: an earlier session left this arm
    powered and holding, and connecting would cut its torque."""


def hold_dir() -> Path:
    override = os.environ.get(HOLD_DIR_ENV, "").strip()
    return Path(override).expanduser() if override else Path.home() / ".cascade" / "arm_holds"


def hold_key(arm_type, cfg) -> str | None:
    """The bus an arm of profile type `arm_type` drives (`cfg`: its profile),
    or None for a backend whose disconnect never cuts torque (never held)."""
    entry = _BUS.get(str(arm_type or ""))
    if entry is None:
        return None
    name, default = entry
    value = cfg.get(name, default) if cfg is not None else default
    return str(value if value not in (None, "") else default)


def _stem(key: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(key)).strip("_.") or "arm"


def marker_path(key: str) -> Path:
    return hold_dir() / f"{_stem(key)}.json"


def _confirm_path(key: str | None) -> Path:
    return hold_dir() / f"{_ANY if key is None else _stem(key)}.confirm.json"


def _write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True))
    os.replace(tmp, path)


def record(key: str, *, arm: str, reason: str, source: str = "teardown") -> dict:
    """Remember that the arm on `key` stays powered and holding (`reason`).
    Raises when the marker cannot be written: the caller says so loudly."""
    now = time.time()
    data = {"key": str(key), "arm": str(arm), "reason": str(reason), "source": source,
            "pid": os.getpid(), "at": now, "at_text": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now))}
    _write(marker_path(key), data)
    return data


def read(key: str) -> dict | None:
    """The hold marker of `key`, or None. A marker that cannot be read counts
    as a hold (fail closed): only an operator or a measured rest clears it."""
    path = marker_path(key)
    try:
        text = path.read_text()
    except FileNotFoundError:
        return None
    except OSError as exc:
        return {"key": str(key), "reason": f"its hold marker {path} cannot be read ({exc})", "unreadable": True}
    try:
        data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError("not an object")
    except ValueError as exc:
        return {"key": str(key), "reason": f"its hold marker {path} is unreadable ({exc})", "unreadable": True}
    data.setdefault("key", str(key))
    data.setdefault("reason", "held by an earlier session")
    return data


def held(keys) -> list[dict]:
    """The markers of `keys` (None and repeats skipped), in order."""
    out, seen = [], set()
    for key in keys:
        if key is None or key in seen:
            continue
        seen.add(key)
        marker = read(key)
        if marker is not None:
            out.append(marker)
    return out


def all_held() -> list[dict]:
    """Every hold marker on this host (an unreadable one included)."""
    root = hold_dir()
    if not root.is_dir():
        return []
    out = []
    for path in sorted(root.glob("*.json")):
        if path.name.endswith(".confirm.json"):
            continue
        try:
            data = json.loads(path.read_text())
            key = data.get("key") if isinstance(data, dict) else None
        except (OSError, ValueError):
            key = None
        marker = read(key) if key else {"key": path.stem, "reason": f"its hold marker {path} is unreadable",
                                         "unreadable": True}
        if marker is not None:
            out.append(marker)
    return out


def clear(key: str, *, why: str) -> bool:
    """Forget the hold of `key` (why: what made the arm safe). True if one existed."""
    path = marker_path(key)
    try:
        path.unlink()
    except FileNotFoundError:
        return False
    print(f"[cascade] arm on {key}: hold cleared ({why})", file=sys.stderr, flush=True)
    return True


def confirmed(phrase: str) -> bool:
    return str(phrase or "").strip().lower() == CONFIRMATION


def confirm(phrase: str, key: str | None = None) -> Path:
    """The operator's confirmation that they are supporting the arm on `key`
    (None: whichever held arm is connected or cut next), written now and
    valid CONFIRM_TTL_S, once. Raises ValueError for any other phrase."""
    if not confirmed(phrase):
        raise ValueError(f"the confirmation is the phrase {CONFIRMATION!r}")
    path = _confirm_path(key)
    _write(path, {"at": time.time(), "pid": os.getpid(), "key": key})
    return path


def take_confirmation(key: str | None) -> dict | None:
    """A fresh operator confirmation for `key` (or for any arm), consumed: a
    second connect or cut needs a second confirmation. Stale ones are removed."""
    for path in ((_confirm_path(key),) if key is not None else ()) + (_confirm_path(None),):
        try:
            data = json.loads(path.read_text())
        except FileNotFoundError:
            continue
        except (OSError, ValueError):
            data = {}
        try:
            path.unlink()
        except FileNotFoundError:
            continue  # another process took it first
        age = time.time() - float(data.get("at", float("-inf"))) if isinstance(data, dict) else float("inf")
        if 0.0 <= age <= CONFIRM_TTL_S:
            return {**data, "age_s": age}
    return None


def confirm_command(key: str | None = None) -> str:
    """The shell command that confirms, as it works from any directory (the
    MCP server runs from models/ under systemd; its operator reads server.log)."""
    from ..config import PACKAGE_ROOT

    script = PACKAGE_ROOT / "scripts" / "arm_hold.py"
    head = f"{sys.executable} {script}" if script.is_file() else f"{sys.executable} -m cascade.control.arm_hold"
    return f"{head} confirm '{CONFIRMATION}'" + (f" --arm {key}" if key else "")


def refusal(marker: dict, key: str) -> str:
    who = f"pid {marker['pid']}, " if marker.get("pid") is not None else ""
    when = marker.get("at_text") or "an earlier session"
    return (f"the arm on {key} was left POWERED and HOLDING by an earlier session ({marker.get('reason')}; "
            f"{who}{when}). Nothing was sent to it: connecting clears its motor faults with a stop frame that "
            f"cuts its torque, and a raised arm would drop. An operator must support the arm (or lower it to "
            f"rest), then confirm within {CONFIRM_TTL_S:.0f} s: {confirm_command()} -- and retry. With the arm "
            "folded at rest, the startup check (scripts/check_rebot_gripper.py --yes --reference-only, run by "
            "`~/cascade-demo.sh up`) measures it with no motor enabled and clears the hold.")


def check_connect(key: str | None, *, driver: str) -> None:
    """Call FIRST in a torque-cutting connect, before anything opens the bus:
    raise ArmHeldError over a hold unless an operator's fresh confirmation is
    on file (then the hold is cleared and the connect may go ahead)."""
    if key is None:
        return
    marker = read(key)
    if marker is None:
        return
    taken = take_confirmation(key)
    if taken is None:
        raise ArmHeldError(refusal(marker, key))
    clear(key, why=f"an operator confirmed {CONFIRMATION!r} {taken['age_s']:.0f} s ago")
    print(f"[cascade] WARNING: {driver} on {key}: connecting to an arm an earlier session left holding, on "
          "the operator's confirmation: its fault clear cuts torque while they support the arm",
          file=sys.stderr, flush=True)


def clear_if_at_rest(key: str | None, q, park, tol: float, *, log=print) -> bool:
    """The startup check's measured rest (`q`: every joint read with no motor
    enabled) within `tol` of `park`: the arm on `key` drops nothing when its
    torque goes, so its hold is cleared. True if a hold was cleared."""
    if key is None or read(key) is None:
        return False
    import numpy as np

    qv = np.asarray([np.nan if v is None else v for v in q], dtype=float)
    pv = np.asarray(park, dtype=float)
    if qv.shape != pv.shape or not np.all(np.isfinite(qv)) or float(np.max(np.abs(qv - pv))) > float(tol):
        return False
    log(f"[+] the arm on {key} was left holding by an earlier session; it rests folded at park_q "
        f"(max |q - park| {float(np.max(np.abs(qv - pv))):.3f} <= {tol}): hold cleared")
    return clear(key, why="measured folded at rest with no motor enabled")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="arm_hold", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="the arms an earlier session left powered and holding")
    p_confirm = sub.add_parser("confirm", help="you are supporting the arm: the next connect (or a "
                                               f"teardown's torque cut) may go ahead, once, within "
                                               f"{CONFIRM_TTL_S:.0f} s")
    p_confirm.add_argument("phrase", help=f"exactly {CONFIRMATION!r}")
    p_confirm.add_argument("--arm", default=None, help="the held arm's bus (can0, /dev/ttyACM0); default: any")
    args = parser.parse_args(argv)
    if args.cmd == "confirm":
        if not confirmed(args.phrase):
            parser.error(f"the confirmation is the phrase {CONFIRMATION!r} (you are supporting the arm)")
        confirm(args.phrase, args.arm)
        print(f"confirmed: you are supporting {'the arm on ' + args.arm if args.arm else 'the arm'}. The next "
              f"connect of a held arm (or a teardown's torque cut) within {CONFIRM_TTL_S:.0f} s goes ahead "
              "once: its torque drops, keep holding it.")
    markers = all_held()
    if not markers:
        print("no arm is held")
    for m in markers:
        print(f"HELD: arm on {m.get('key')}: {m.get('reason')} ({m.get('at_text', '?')}, pid {m.get('pid', '?')})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
