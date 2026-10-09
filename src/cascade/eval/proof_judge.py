"""The outcome judge as a METRIC of the launcher's proof turn (ROADMAP
near-term #6).

`scripts/launch.sh` declares READY on physics alone: `scripts/demo_proof.py`
needs one pick confirmed on the physics channel and the same world reset,
and writes the receipt `<state>/proof.json` plus a per-attempt evidence
directory. This pass runs AFTER that, opt-in (`--judge fake|vlm|grm` or
`CASCADE_JUDGE`; default `off`, and then nothing here runs):

1. reads the receipt (read-only -- `spark_verify.py`, `desktop.py` and
   `spark_browser.py` consume it; this pass never writes it);
2. skips an unverified receipt (robot turn skipped: nothing to calibrate the
   judge against);
3. runs `scripts/judge_run.py` over the proof turn's `pick_and_place` rows of
   the trace the receipt names, in its own process group, under a hard
   wall-clock bound (`CASCADE_JUDGE_TIMEOUT_S`, default 180 s, at most
   1800 s): at the bound the whole group is killed;
4. writes the judge-vs-physics confusion matrix into
   `<evidence_dir>/run-summary.json` (key `judge`, next to a copy of the
   receipt's identity and verdict) and prints ONE banner line.

`fn > 0` is the regression this exists to catch: a physics-confirmed pick
the pictures did not show as progress (the byte-identical-keyframe bug was
exactly that).

Advisory only: the physics verdict is copied, never re-derived; READY and
the launcher's exit status do not depend on anything here; every judge
failure -- bad configuration, a refused or hung endpoint, no pair scored,
the bound expiring, a crash -- reads `unavailable`. The judge that answers
is whatever `judge_run.py` resolves (`eval.judge`, or the complete JSON in
`CASCADE_JUDGE_CONFIG`, with the backend from `--judge`), and its name is in
the record, so a local Qwen number is never read as a GRM or GPT one.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

OFF = "off"
BACKENDS = ("fake", "vlm", "grm")
DEFAULT_TIMEOUT_S = 180.0
MAX_TIMEOUT_S = 1800.0
#: the skill the launcher's proof turn validates (demo_proof.validate_pick_trace)
PROOF_SKILLS = ("pick_and_place",)
SUMMARY_NAME = "run-summary.json"
SCHEMA = "cascade.run_summary/1"
VERDICT_NAME = "judge.json"
LOG_NAME = "judge-run.log"


class PassConfigError(ValueError):
    """`--judge` / `CASCADE_JUDGE` / `CASCADE_JUDGE_TIMEOUT_S` is not valid."""


def resolve_backend(value: str | None) -> str:
    """`off` (also unset/empty) or one of `fake|vlm|grm`. Anything else is an
    error: a judge nobody chose is a number nobody can interpret."""
    v = (value or "").strip().lower()
    if v in ("", OFF):
        return OFF
    if v in BACKENDS:
        return v
    raise PassConfigError(f"judge backend must be off|fake|vlm|grm, got {value!r}")


def resolve_timeout(value) -> float:
    """The pass's wall-clock bound in seconds: unset -> 180; otherwise a
    finite number in (0, 1800]."""
    if value is None or str(value).strip() == "":
        return DEFAULT_TIMEOUT_S
    try:
        t = float(value)
    except (TypeError, ValueError):
        raise PassConfigError(f"CASCADE_JUDGE_TIMEOUT_S must be a number of seconds, got {value!r}") from None
    if not 0 < t <= MAX_TIMEOUT_S:   # also refuses nan and inf (every comparison is False)
        raise PassConfigError(f"CASCADE_JUDGE_TIMEOUT_S must be in (0, {MAX_TIMEOUT_S:g}] s, got {value!r}")
    return t


def _one_line(text) -> str:
    """The banner is ONE line: collapse every newline/whitespace run."""
    return " ".join(str(text).split())


def _read_proof(proof_path: Path) -> tuple[dict | None, bytes | None, str | None]:
    try:
        raw = Path(proof_path).read_bytes()
        proof = json.loads(raw)
    except (OSError, ValueError) as exc:
        return None, None, f"proof receipt unreadable: {type(exc).__name__}"
    if not isinstance(proof, dict):
        return None, None, "proof receipt is not a JSON object"
    return proof, raw, None


def evidence_dir(proof_path: Path, proof: dict | None) -> Path | None:
    """The per-attempt evidence directory the receipt names -- only when it
    sits beside the receipt (the layout demo_proof writes), so the pass never
    writes anywhere a receipt merely points at."""
    name = proof.get("evidence_dir") if isinstance(proof, dict) else None
    if not name:
        return None
    ev = Path(str(name))
    if not ev.is_dir() or ev.resolve().parent != Path(proof_path).resolve().parent:
        return None
    return ev


def _kill_group(child: subprocess.Popen) -> None:
    try:
        os.killpg(child.pid, signal.SIGKILL)
    except (AttributeError, ProcessLookupError, PermissionError):
        child.kill()
    child.wait()


def _last_line(path: Path) -> str:
    try:
        lines = [line for line in path.read_text(errors="replace").splitlines() if line.strip()]
    except OSError:
        return ""
    return lines[-1] if lines else ""


def run_judge_pass(proof_path, backend: str, *, repo, timeout_s: float = DEFAULT_TIMEOUT_S,
                   python: str | None = None, env: dict | None = None) -> dict:
    """The `judge` block of the run summary. Never raises for a judge failure
    and never writes the receipt."""
    started = time.monotonic()
    proof_path = Path(proof_path)
    block: dict = {"backend": backend, "status": "unavailable", "advisory": True,
                   "skills": list(PROOF_SKILLS), "timeout_s": timeout_s, "reason": None}

    def done(status: str, reason: str | None = None) -> dict:
        block.update(status=status, reason=None if reason is None else _one_line(reason),
                     elapsed_s=round(time.monotonic() - started, 3))
        return block

    proof, _, err = _read_proof(proof_path)
    if proof is None:
        return done("unavailable", err)
    if proof.get("verified") is not True:
        return done("skipped", "the proof turn is not physics-verified (robot turn skipped); "
                               "nothing to calibrate the judge against")
    ev = evidence_dir(proof_path, proof)
    if ev is None:
        return done("unavailable", "the proof receipt names no evidence directory beside it")
    trace = Path(str(proof.get("trace") or ""))
    if trace.name != "trace.jsonl" or not trace.is_file():
        return done("unavailable", f"the proof receipt names no readable trace.jsonl ({proof.get('trace')!r})")

    verdict_path, log_path = ev / VERDICT_NAME, ev / LOG_NAME
    verdict_path.unlink(missing_ok=True)   # a failed re-run must not leave the last pass's verdict
    block["log"] = str(log_path)
    cmd = [python or sys.executable, str(Path(repo) / "scripts" / "judge_run.py"), str(trace.parent),
           "--judge", backend, "--skills", ",".join(PROOF_SKILLS), "--json-out", str(verdict_path)]
    with log_path.open("w") as log:
        try:
            child = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                     env=env, start_new_session=True)
        except OSError as exc:
            return done("unavailable", f"cannot start judge_run.py: {exc}")
        block["judge_pid"] = child.pid
        try:
            rc = child.wait(timeout=max(0.0, timeout_s - (time.monotonic() - started)))
        except subprocess.TimeoutExpired:
            _kill_group(child)
            return done("unavailable", f"timed out after {timeout_s:g} s (the bound; the judge process "
                                       "group was killed)")
        except BaseException:
            _kill_group(child)   # Ctrl-C: never leave an orphan judge holding a model endpoint
            raise
    if rc != 0:
        return done("unavailable", f"judge_run.py exited {rc}: {_last_line(log_path) or 'no output'}")
    try:
        verdict = json.loads(verdict_path.read_text())
        confusion = verdict["confusion"]
    except (OSError, ValueError, KeyError, TypeError):
        return done("unavailable", "judge_run.py wrote no verdict")
    steps = verdict.get("steps") or []
    block.update(verdict=str(verdict_path), judge=verdict.get("judge"), mode=verdict.get("mode"), confusion=confusion,
                 final_progress=verdict.get("final_progress"), summary_line=verdict.get("summary_line"),
                 unscored_physics=sum(1 for s in steps if s.get("hop") is None
                                      and s.get("physics") in ("confirmed", "refuted")))
    if not confusion.get("n_scored"):
        first = next((s.get("error") for s in steps if s.get("error")), None)
        return done("unavailable", f"no proof-turn step was scored ({first or 'no pick_and_place row'})")
    return done("ok")


def write_run_summary(proof_path, block: dict) -> Path | None:
    """Merge `proof` (identity + verdict, copied) and `judge` into
    `<evidence_dir>/run-summary.json` atomically; other keys are kept. None
    when the receipt names no evidence directory beside it."""
    proof_path = Path(proof_path).absolute()
    proof, raw, _ = _read_proof(proof_path)
    ev = evidence_dir(proof_path, proof)
    if ev is None:
        return None
    path = ev / SUMMARY_NAME
    summary: dict = {}
    try:
        old = json.loads(path.read_text())
        if isinstance(old, dict):
            summary = old
    except (OSError, ValueError):
        pass
    summary.update(schema=SCHEMA, written_at=time.time(),
                   proof={"path": str(proof_path), "sha256": hashlib.sha256(raw).hexdigest(),
                          **{k: proof.get(k) for k in ("session_id", "verified", "sim", "model", "trace")}},
                   judge=block)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(summary, indent=2) + "\n")
    tmp.replace(path)
    return path


def banner_line(block: dict, summary_path: Path | None = None) -> str:
    """The launcher's ONE `judge:` line."""
    where = f"; {summary_path}" if summary_path else ""
    status = block.get("status")
    if status == "ok":
        line = f"advisory {block.get('summary_line')}"
        if (block.get("confusion") or {}).get("fn"):
            line += " -- fn>0: the pictures missed physics-confirmed progress"
        if block.get("unscored_physics"):
            line += f" -- {block['unscored_physics']} physics-graded step(s) unscored"
        return _one_line(line + f" (physics verdict unchanged{where})")
    if status == "skipped":
        return _one_line(f"skipped ({block.get('reason')}{where})")
    return _one_line(f"unavailable ({block.get('reason')}); advisory only, the physics verdict stands{where}")


def main(argv=None, *, repo: Path | None = None) -> int:
    """CLI for launch.sh: one line on stdout, exit 0 whatever the judge did
    (argparse usage errors aside)."""
    ap = argparse.ArgumentParser(description="Advisory, time-bounded judge pass over the launcher's proof turn.")
    ap.add_argument("--proof", required=True, type=Path, help="the launcher receipt (<state>/proof.json)")
    ap.add_argument("--judge", default=os.environ.get("CASCADE_JUDGE"), help="off|fake|vlm|grm (CASCADE_JUDGE)")
    ap.add_argument("--timeout-s", default=os.environ.get("CASCADE_JUDGE_TIMEOUT_S"),
                    help=f"wall-clock bound, default {DEFAULT_TIMEOUT_S:g} s (CASCADE_JUDGE_TIMEOUT_S)")
    ap.add_argument("--repo", type=Path, default=repo or Path(__file__).resolve().parents[3])
    args = ap.parse_args(argv)
    try:
        try:
            backend = resolve_backend(args.judge)
            timeout_s = resolve_timeout(args.timeout_s)
        except PassConfigError as exc:
            block = {"backend": args.judge, "status": "unavailable", "advisory": True, "reason": str(exc)}
            print(banner_line(block, write_run_summary(args.proof, block)), flush=True)
            return 0
        if backend == OFF:
            print("off (opt in with --judge fake|vlm|grm or CASCADE_JUDGE)", flush=True)
            return 0
        block = run_judge_pass(args.proof, backend, repo=args.repo, timeout_s=timeout_s)
        print(banner_line(block, write_run_summary(args.proof, block)), flush=True)
    except Exception as exc:  # noqa: BLE001 -- advisory: a bug here must never cost READY
        print(banner_line({"status": "unavailable", "reason": f"judge pass error: {type(exc).__name__}: {exc}"}),
              flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
