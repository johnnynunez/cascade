"""Arms a teardown left POWERED and HOLDING, remembered across processes.

#424: a teardown whose park did not bring a torque-cutting arm to rest (a dead camera, the e-stop
every MCP SIGTERM latches, a stop mid-park) keeps that arm's torque on, because cutting it would drop
a raised arm. The NEXT process could not see that: it built a fresh standby LazyArm (the dashboard
said `arm_connected: false`), and the reBot drivers' connect() clears motor faults with a RobStride
type-4 STOP frame before enabling (`robstride.clear_motor_faults`: no torque), so the next start,
arm-check or calibrate dropped the raised arm with nobody holding it (#424 review, Important 1).

So every process that leaves an arm held writes one marker per held arm, keyed by the arm's BUS (the
CAN channel of either reBot transport -- for the SDK one the channel of its hardware YAML -- or the
serial port of an SO-101): the physical arm, whichever profile drives it next. The writers: the
runtime teardown (`apps.demo.shutdown_runtime`), the calibration CLI's cleanup, the arm-check
(`scripts/home_rebot_rs.py`) and the startup check's soft stops, and the bring-up scripts that end
with the motors enabled. Every process reads it:

- a reBot connect (`RebotRSArm`, `RebotRSMotorBridgeArm`) over a marker refuses BEFORE it opens the
  bus (`ConnectGate`): no frame at all. It goes ahead only on an operator's confirmation written AT
  THAT MOMENT, at the robot's own terminal (`scripts/arm_hold.py confirm`, valid `CONFIRM_TTL_S`):
  they support the arm. The confirmation is used, and the marker cleared, only when that connect
  succeeds; a connect that fails (the bus is down) keeps both, so a retry within the window may use
  the same confirmation;
- with no marker, the same connect still measures the arm first, with parameter reads only (no frame
  that can change a motor's torque), before its mode write and fault clear: an arm that is not folded
  at rest may be held by motors an earlier process left ENABLED with no marker (a killed process, an
  older version, a bring-up script), so it is refused like a marker. The RobStride protocol has no
  read that reports whether a motor is enabled (its mode bits come only in status frames, which a
  motor sends in answer to a control frame; motorbridge's `request_feedback` is a no-op for
  RobStride), so the measured pose is the evidence: a disabled raised arm falls to its rest;
- the startup check (`scripts/check_rebot_gripper.py`, run by `~/cascade-demo.sh up`) reads the arm
  with no motor enabled and clears the marker when the arm rests folded at park_q
  (`clear_if_at_rest`): the connect's STOP frame then drops nothing;
- a teardown that parks the arm and cuts its torque, or cuts it on a fresh confirmation, clears it;
- the dashboard /state, MCP `world_state` and `move_to_rest` report it (`arm_held`; `arm_connected`
  true: the arm is powered, never "standby"; never `safe_to_power_off`).

Only the operator confirms. No agent-facing text (a tool error, world_state, /state) names the
command, and `confirm` reads the phrase from the controlling terminal (/dev/tty) of an interactive
session, refusing without one or in an agent's or the robot server's environment (`AGENT_ENV`).

An SO-101's connect writes no torque-off (`FeetechArm.connect`): it is never refused, but its hold is
reported until a parked torque-off clears it.

Command line (the operator, in a terminal on the robot host, any working directory):

    python scripts/arm_hold.py list      # exit 0: nothing held; 3: an arm is held, or it cannot tell
    python scripts/arm_hold.py confirm [--arm can0]   # interactive: type the phrase at the prompt
    python scripts/arm_hold.py revoke    # drop every pending confirmation
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import threading
import time
from pathlib import Path

HOLD_DIR_ENV = "CASCADE_ARM_HOLD_DIR"
#: The operator's words, the same phrase the teardown's torque cut takes
#: (`apps.demo.TORQUE_OFF_CONFIRMATION`): they are holding the arm.
CONFIRMATION = "holding the arm"
#: A confirmation is valid this long after it was written, and is used by
#: one connect (or one teardown cut): it means "I am supporting the arm NOW",
#: never "for this session".
CONFIRM_TTL_S = 120.0
#: How far (rad, every arm joint) from park_q a reBot connect still counts
#: the arm folded at rest; the startup check and the arm-check use the same.
REST_TOL_RAD = 0.12
#: `list`'s exit status when an arm is held, or when it cannot tell.
LIST_HELD_RC = 3
#: The marker's bus key for each torque-cutting backend: (config key, the
#: backend's own default). Both reBot transports drive one arm on one bus;
#: the SDK one (`rebot_rs`) opens the channel of its hardware YAML instead
#: (`_sdk_channel`), the profile's `channel` only as the fallback.
_BUS = {"so101": ("port", "/dev/ttyACM0"), "rebot_rs": ("channel", "can0"),
        "rebot_rs_mb": ("channel", "can0")}
#: Profile types whose connect cuts the torque of a held arm (the reBot
#: drivers' fault clear): their connect refuses over a hold.
CONNECT_CUTS_TORQUE = frozenset({"rebot_rs", "rebot_rs_mb"})
#: Set (non-empty) in the environment of an agent's shell or of the robot
#: server: `confirm` refuses there. A guard on top of the real one (the
#: phrase typed at the controlling terminal of an interactive session).
AGENT_ENV = ("AI_AGENT", "HERMES_AGENT", "HERMES_SESSION_ID", "CLAUDECODE", "CLAUDE_CODE_ENTRYPOINT",
             "CODEX_SANDBOX", "CODEX_SANDBOX_NETWORK_DISABLED", "CODEX_THREAD_ID", "CASCADE_MCP_TOKEN",
             "CASCADE_OPENCLAW_PROFILE")
AGENT_ENV_PREFIXES = ("OPENCLAW_",)
_ANY = "_any"


class ArmHeldError(RuntimeError):
    """A connect refused before any frame that can cut torque: an earlier
    session left this arm powered and holding (or it measures raised), and
    connecting would cut its torque."""


def hold_dir() -> Path:
    override = os.environ.get(HOLD_DIR_ENV, "").strip()
    return Path(override).expanduser() if override else Path.home() / ".cascade" / "arm_holds"


def _installed_sdk_root() -> str | None:
    """reBotArm_control_py's checkout when it is installed rather than named
    by `sdk_path` (located, never imported)."""
    try:
        import importlib.util

        spec = importlib.util.find_spec("reBotArm_control_py")
    except Exception:  # noqa: BLE001 -- absent, blocked or a test double: no answer
        return None
    origin = getattr(spec, "origin", None) if spec is not None else None
    return str(Path(origin).resolve().parent.parent) if origin else None


def _sdk_channel(cfg) -> str | None:
    """The bus the reBot SDK (`rebot_rs`) opens for this profile: the
    `channel` of its hardware YAML (`hw_yaml`, under `<sdk_path>/config/`
    unless absolute: reBotArm_control_py's `load_cfg`, whose default is
    /dev/ttyACM0). The profile's own `channel` is not what the SDK opens, so
    two SDK reBots on can0 and can1 never share a marker (#426 review Minor
    6). None when that file cannot be read (the SDK's connect then fails
    before it opens anything)."""
    if cfg is None:
        return None
    hw = cfg.get("hw_yaml", "rebotarm_rs.yaml")
    if hw in (None, ""):
        return None
    path = Path(str(hw)).expanduser()
    if not path.is_absolute():
        root = cfg.get("sdk_path") or _installed_sdk_root()
        if not root:
            return None
        path = Path(str(root)).expanduser() / "config" / path
    try:
        import yaml

        data = yaml.safe_load(path.read_text())
    except Exception:  # noqa: BLE001 -- unreadable: the profile's channel stands in
        return None
    if not isinstance(data, dict):
        return None
    channel = data.get("channel", "/dev/ttyACM0")
    return str(channel) if channel not in (None, "") else None


def hold_key(arm_type, cfg) -> str | None:
    """The bus an arm of profile type `arm_type` drives (`cfg`: its profile),
    or None for a backend whose disconnect never cuts torque (never held)."""
    kind = str(arm_type or "")
    entry = _BUS.get(kind)
    if entry is None:
        return None
    if kind == "rebot_rs":
        channel = _sdk_channel(cfg)
        if channel:
            return channel
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


def remember(key: str | None, *, arm: str, reason: str, source: str, log=None) -> bool:
    """`record` for a process that leaves an arm held and then exits (a CLI,
    a script): never raises; a marker it could not write is said loudly
    (the next start then cannot see that hold). True when written."""
    if key is None:
        return False
    say = log or (lambda msg: print(msg, file=sys.stderr, flush=True))
    try:
        record(key, arm=arm, reason=reason, source=source)
    except Exception as exc:  # noqa: BLE001 -- the caller's exit goes on; the warning says what is lost
        say(f"[cascade] WARNING: the arm on {key} stays POWERED and HOLDING, but its hold marker could not be "
            f"written ({exc}): the next start cannot see it. Support the arm before any restart.")
        return False
    say(f"[cascade] the arm on {key} stays POWERED and HOLDING ({reason}); remembered in {marker_path(key)}: "
        "the next connect of this arm is refused until an operator supports it and confirms.")
    return True


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


def held_marker(raw) -> dict | None:
    """The hold marker of the arm behind `raw` (its `hold_key`), or None.
    Never materializes a LazyArm."""
    key = getattr(raw, "hold_key", None) if raw is not None else None
    return read(key) if isinstance(key, str) and key else None


def all_held() -> list[dict]:
    """Every hold marker on this host (an unreadable one included). Fails
    closed (#426 review Minor 2): a hold directory that exists but cannot be
    listed is one entry that counts as held."""
    root = hold_dir()
    try:
        names = sorted(os.listdir(root))
    except FileNotFoundError:
        return []
    except OSError as exc:
        return [{"key": "?", "reason": f"the hold directory {root} cannot be read ({exc}): an arm may be held",
                 "unreadable": True}]
    out = []
    for name in names:
        if not name.endswith(".json") or name.endswith(".confirm.json") or name.startswith("."):
            continue
        path = root / name
        try:
            data = json.loads(path.read_text())
            key = data.get("key") if isinstance(data, dict) else None
        except (OSError, ValueError):
            key = None
        marker = read(key) if isinstance(key, str) and key else {
            "key": path.stem, "reason": f"its hold marker {path} is unreadable", "unreadable": True}
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
    """Write the operator's confirmation that they are supporting the arm on
    `key` (None: whichever held arm is connected or cut next): valid
    CONFIRM_TTL_S, used once. Raises ValueError for any other phrase. The
    command line (`main`) is the operator's way in: it reads the phrase at
    the controlling terminal and refuses an agent's environment."""
    if not confirmed(phrase):
        raise ValueError(f"the confirmation is the phrase {CONFIRMATION!r}")
    path = _confirm_path(key)
    _write(path, {"at": time.time(), "pid": os.getpid(), "key": key})
    return path


def _age(data) -> float | None:
    """Seconds since a confirmation was written, or None when it names no
    valid time: a missing, non-numeric or non-finite `at` never raises
    (#426 review Minor 1: a ValueError here aborted a teardown before it
    wrote its markers)."""
    if not isinstance(data, dict):
        return None
    at = data.get("at")
    if isinstance(at, bool) or not isinstance(at, (int, float)):
        return None
    age = time.time() - float(at)
    return age if math.isfinite(age) else None


def _fresh(age) -> bool:
    return age is not None and 0.0 <= age <= CONFIRM_TTL_S


class Confirmation:
    """An operator's fresh confirmation, claimed by one connect or cut (the
    file renamed: no other claimant sees it). `used()` once that connect or
    cut went ahead; `release()` when it failed: a retry within its window may
    claim it again."""

    def __init__(self, path: Path, claimed: Path, data: dict):
        self.path, self.claimed, self.data = path, claimed, data

    @property
    def age_s(self) -> float:
        return float(self.data.get("age_s", 0.0))

    def used(self) -> None:
        try:
            self.claimed.unlink()
        except FileNotFoundError:
            pass

    def release(self) -> None:
        try:
            os.link(self.claimed, self.path)  # back for a retry (its age is checked again then); never
        except OSError:                       # over a newer confirmation
            pass  # a newer one is there, or the directory went: this one is dropped
        self.used()


def claim_confirmation(key: str | None) -> Confirmation | None:
    """Claim a fresh operator confirmation for `key` (or one for any arm).
    Stale, future-dated and unreadable ones are removed on the way."""
    for path in ((_confirm_path(key),) if key is not None else ()) + (_confirm_path(None),):
        claimed = path.with_name(f".{path.name}.claimed.{os.getpid()}.{threading.get_ident()}")
        try:
            os.rename(path, claimed)  # atomic: one claimant per confirmation
        except OSError:
            continue  # none here (or another process took it first)
        try:
            data = json.loads(claimed.read_text())
        except (OSError, ValueError):
            data = None
        age = _age(data)
        if _fresh(age):
            return Confirmation(path, claimed, {**data, "age_s": age})
        try:
            claimed.unlink()
        except OSError:
            pass
    return None


def take_confirmation(key: str | None) -> dict | None:
    """A fresh operator confirmation for `key` (or for any arm), used up at
    once: for an action that follows immediately (a teardown's torque cut).
    Never raises over a malformed file."""
    found = claim_confirmation(key)
    if found is None:
        return None
    found.used()
    return found.data


def pending_confirmations() -> list[dict]:
    """The fresh confirmations on file (key None: any arm), read only."""
    out = []
    try:
        names = sorted(os.listdir(hold_dir()))
    except OSError:
        return out
    for name in names:
        if name.startswith(".") or not name.endswith(".confirm.json"):
            continue
        try:
            data = json.loads((hold_dir() / name).read_text())
        except (OSError, ValueError):
            continue
        age = _age(data)
        if _fresh(age):
            out.append({"key": data.get("key"), "age_s": age, "left_s": CONFIRM_TTL_S - age})
    return out


def revoke_confirmations() -> int:
    """Drop every pending confirmation (an operator command that failed
    before it connected must not leave one usable). The number removed."""
    removed = 0
    try:
        names = os.listdir(hold_dir())
    except OSError:
        return 0
    for name in names:
        if name.endswith(".confirm.json") and not name.startswith("."):
            try:
                (hold_dir() / name).unlink()
                removed += 1
            except OSError:
                pass
    return removed


def confirm_command(key: str | None = None) -> str:
    """The shell command an OPERATOR runs to confirm, as it works from any
    directory (the MCP server runs from models/ under systemd; its operator
    reads server.log). Operator-facing text only: never in a tool error,
    world_state or /state."""
    from ..config import PACKAGE_ROOT

    script = PACKAGE_ROOT / "scripts" / "arm_hold.py"
    head = f"{sys.executable} {script}" if script.is_file() else f"{sys.executable} -m cascade.control.arm_hold"
    return f"{head} confirm" + (f" --arm {key}" if key else "")


#: What every refusal asks for, agent-facing (a tool error): no command.
ASK_THE_OPERATOR = ("Do not retry on your own: ask the operator to support the arm (or lower it to rest) "
                    "and confirm at the robot's own terminal, then retry; an agent cannot confirm.")


def refusal(marker: dict, key: str) -> str:
    who = f"pid {marker['pid']}, " if marker.get("pid") is not None else ""
    when = marker.get("at_text") or "an earlier session"
    return (f"the arm on {key} was left POWERED and HOLDING by an earlier session ({marker.get('reason')}; "
            f"{who}{when}). Nothing was sent to it: connecting clears its motor faults with a stop frame that "
            f"cuts its torque, and a raised arm would drop. {ASK_THE_OPERATOR} Folded at rest, the operator's "
            "startup check measures it with no motor enabled and clears the hold.")


def rest_problem(q, park, tol: float = REST_TOL_RAD) -> str | None:
    """Why the joints `q` (None: no answer) are not the folded rest `park`
    within `tol`, or None when every joint is."""
    import numpy as np

    qv = np.asarray([np.nan if v is None else v for v in q], dtype=float).reshape(-1)
    pv = np.asarray(park, dtype=float).reshape(-1)
    if qv.shape != pv.shape:
        return f"{qv.size} joints read for a {pv.size}-joint rest pose"
    bad = [i + 1 for i in range(qv.size) if not np.isfinite(qv[i])]
    if bad:
        return f"joint(s) {bad} gave no usable reading"
    off = float(np.max(np.abs(qv - pv))) if qv.size else 0.0
    if off > float(tol):
        return f"{off:.3f} rad from park_q (> {float(tol):g})"
    return None


def raised_refusal(key: str, problem: str) -> str:
    return (f"the arm on {key} is not folded at rest ({problem}), and no earlier session recorded why: its "
            "motors may still be ENABLED and holding it (a killed process, an older cascade, a bring-up "
            "script). Only parameter reads were sent and the bus is released: connecting clears its motor "
            f"faults with a stop frame that cuts its torque, and a raised arm would drop. {ASK_THE_OPERATOR}")


class Permit:
    """One connect's go-ahead over a hold or a raised arm: an operator's
    claimed confirmation, and the marker it covers (None: the arm measured
    raised with no marker). `connected()` uses the confirmation and clears
    the marker; `failed()` keeps both (#426 review Important 3)."""

    def __init__(self, key: str, marker: dict | None, confirmation: Confirmation, driver: str):
        self.key, self.marker, self.confirmation, self.driver = key, marker, confirmation, driver

    def connected(self) -> None:
        self.confirmation.used()
        if self.marker is not None:
            clear(self.key, why=f"{self.driver} connected on an operator's confirmation "
                                f"({self.confirmation.age_s:.0f} s old)")

    def failed(self) -> None:
        self.confirmation.release()


def _warn_going_ahead(key: str, driver: str, what: str) -> None:
    print(f"[cascade] WARNING: {driver} on {key}: connecting to {what} on the operator's confirmation: its "
          "fault clear cuts torque while they support the arm", file=sys.stderr, flush=True)


def check_connect(key: str | None, *, driver: str) -> Permit | None:
    """Call FIRST in a torque-cutting connect, before anything opens the bus:
    raise ArmHeldError over a hold unless an operator's fresh confirmation is
    on file. Returns the Permit for that confirmation (None: no hold); the
    caller reports how its connect ended (`ConnectGate` does)."""
    if key is None:
        return None
    marker = read(key)
    if marker is None:
        return None
    found = claim_confirmation(key)
    if found is None:
        raise ArmHeldError(refusal(marker, key))
    _warn_going_ahead(key, driver, "an arm an earlier session left holding")
    return Permit(key, marker, found, driver)


class ConnectGate:
    """The hold gate of one torque-cutting connect (both reBot drivers):

        with arm_hold.ConnectGate(key, driver="rebot_rs") as gate:
            ...open the bus; read the joints with parameter reads only...
            gate.require_rest(q, park)       # before the mode write / fault clear
            ...the rest of the connect...

    Entering refuses over a marker (`check_connect`: nothing is opened).
    `require_rest` refuses an arm that does not measure folded at rest
    unless this connect holds an operator's confirmation. Leaving uses the
    confirmation and clears the marker only when the connect succeeded."""

    def __init__(self, key: str | None, *, driver: str):
        self.key, self.driver, self.permit = key, driver, None

    def __enter__(self) -> "ConnectGate":
        self.permit = check_connect(self.key, driver=self.driver)
        return self

    def require_rest(self, q, park, tol: float = REST_TOL_RAD) -> None:
        problem = rest_problem(q, park, tol)
        if problem is None or self.key is None:
            return
        if self.permit is None:
            found = claim_confirmation(self.key)
            if found is None:
                raise ArmHeldError(raised_refusal(self.key, problem))
            self.permit = Permit(self.key, None, found, self.driver)
        _warn_going_ahead(self.key, self.driver, f"an arm that is not folded at rest ({problem})")

    def confirmed_now(self) -> bool:
        """Whether this connect may cut torque on the operator's word (the
        motorbridge mode-write fallback): its permit, or one claimed now."""
        if self.permit is None and self.key is not None:
            found = claim_confirmation(self.key)
            if found is not None:
                self.permit = Permit(self.key, None, found, self.driver)
        return self.permit is not None

    def __exit__(self, exc_type, exc, tb) -> bool:
        permit, self.permit = self.permit, None
        if permit is not None:
            if exc_type is None:
                permit.connected()
            else:
                permit.failed()
        return False


def clear_if_at_rest(key: str | None, q, park, tol: float, *, log=print) -> bool:
    """The startup check's measured rest (`q`: every joint read with no motor
    enabled) within `tol` of `park`: the arm on `key` drops nothing when its
    torque goes, so its hold is cleared. True if a hold was cleared."""
    if key is None or read(key) is None:
        return False
    if rest_problem(q, park, tol) is not None:
        return False
    import numpy as np

    off = float(np.max(np.abs(np.asarray(q, dtype=float) - np.asarray(park, dtype=float))))
    log(f"[+] the arm on {key} was left holding by an earlier session; it rests folded at park_q "
        f"(max |q - park| {off:.3f} <= {tol}): hold cleared")
    return clear(key, why="measured folded at rest with no motor enabled")


def agent_environment(environ=None) -> str | None:
    """The variable that marks `environ` (default: this process's) as an
    agent's shell or the robot server's, or None."""
    env = os.environ if environ is None else environ
    for name in AGENT_ENV:
        if str(env.get(name, "")).strip():
            return name
    for name, value in env.items():
        if name.startswith(AGENT_ENV_PREFIXES) and str(value).strip():
            return name
    return None


def _confirm_at_terminal(key: str | None) -> int:
    """`confirm`: only a person at the robot's terminal. Refuses (exit 2) in
    an agent's or the robot server's environment, or without an interactive
    session whose controlling terminal can be opened; reads the phrase there,
    never from an argument or a pipe."""
    marker = agent_environment()
    if marker is not None:
        print(f"REFUSED: {marker} is set: this runs in an agent's (or the robot server's) environment. Only the "
              "operator confirms, at a terminal on the robot host. Nothing was written.", file=sys.stderr)
        return 2
    try:
        interactive = sys.stdin.isatty() and sys.stdout.isatty()
    except (AttributeError, ValueError):
        interactive = False
    if not interactive:
        print("REFUSED: confirm needs an interactive terminal (stdin and stdout on a terminal): the operator "
              "types the phrase there. Nothing was written.", file=sys.stderr)
        return 2
    try:
        tty_in = open("/dev/tty", encoding="utf-8")  # noqa: SIM115 -- closed below
        tty_out = open("/dev/tty", "w", encoding="utf-8")  # noqa: SIM115
    except OSError as exc:
        print(f"REFUSED: no controlling terminal ({exc}). Nothing was written.", file=sys.stderr)
        return 2
    with tty_in, tty_out:
        tty_out.write(f"Hold the arm{' on ' + key if key else ''} NOW: its torque drops when it connects (or "
                      f"when a teardown cuts it).\nType {CONFIRMATION!r} to confirm (valid {CONFIRM_TTL_S:.0f} s, "
                      "once): ")
        tty_out.flush()
        answer = tty_in.readline()
    if not confirmed(answer):
        print("not confirmed: nothing was written.", file=sys.stderr)
        return 1
    confirm(CONFIRMATION, key)
    print(f"confirmed: you are supporting {'the arm on ' + key if key else 'the arm'}. The next connect of a held "
          f"arm (or a teardown's torque cut) within {CONFIRM_TTL_S:.0f} s goes ahead once: its torque drops, keep "
          "holding it.")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="arm_hold", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="the arms an earlier session left powered and holding (exit 0: none; "
                                f"{LIST_HELD_RC}: held, or it cannot tell)")
    p_confirm = sub.add_parser("confirm", help="you are supporting the arm: the next connect (or a teardown's "
                                               f"torque cut) may go ahead, once, within {CONFIRM_TTL_S:.0f} s. "
                                               "Interactive: you type the phrase at the prompt")
    p_confirm.add_argument("--arm", default=None, help="the held arm's bus (can0, /dev/ttyACM0); default: any")
    sub.add_parser("revoke", help="drop every pending confirmation")
    args = parser.parse_args(argv)
    if args.cmd == "confirm":
        code = _confirm_at_terminal(args.arm)
        if code:
            return code
    elif args.cmd == "revoke":
        print(f"{revoke_confirmations()} pending confirmation(s) revoked")
    markers = all_held()
    if not markers:
        print("no arm is held")
    for m in markers:
        print(f"HELD: arm on {m.get('key')}: {m.get('reason')} ({m.get('at_text', '?')}, pid {m.get('pid', '?')})")
    for c in pending_confirmations():
        print(f"CONFIRMED: {'the arm on ' + str(c['key']) if c.get('key') else 'any arm'}: an operator confirmed "
              f"{c['age_s']:.0f} s ago (valid {max(0.0, c['left_s']):.0f} s more, once)")
    return LIST_HELD_RC if markers else 0


if __name__ == "__main__":
    sys.exit(main())
