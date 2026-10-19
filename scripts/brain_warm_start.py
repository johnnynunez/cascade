#!/usr/bin/env python3
"""Warm start for the llama.cpp brain: the first chat after a start reuses OpenClaw's prefix like any later chat.

scripts/serve_nemotron_bolt_llamacpp.sh starts this helper beside llama-server (WARM_START=1, the default).

Every OpenClaw agent request opens with the same prefix: its stable system text and the tool block (cascade's chat
template renders OpenClaw's per-session text AFTER the tools: scripts/chat_templates/). A new chat reuses that
prefix from the previous chat, but right after a start nothing is cached and the first chat re-prefilled all ~13.6k
tokens (13.6 s with the brain alone). So the helper:

1. waits until llama-server answers /health. That is the server's own readiness; the helper never delays it.
2. prefills the RECORDED prefix tokens and copies them into every slot: a raw /completion (n_cmpl = slots,
   n_predict 1). A raw token prompt is no chat: no conversation, transcript or user turn exists anywhere. Two steps:
   - the tokens before OpenClaw's prompt-cache boundary (`<!-- openclaw:attempt:DYNAMIC -->`): the system text and
     the tools, stable by OpenClaw's own contract;
   - then the text after it that the last two OpenClaw sessions shared (date, output directives: ~500 tokens, up to
     the session ids), so the first chat reuses what a new chat reuses from the previous one. The first step's
     context checkpoints (llama.cpp keeps them for this hybrid model) stay: on another day the first chat parts at
     the date and resumes at the boundary.
   Each step is bounded by --timeout (120 s; ~13 s alone). A request that arrives meanwhile waits for a free slot.
3. records those tokens while the server runs, as the server itself tokenized them (no drift, nothing rebuilt):
   - every --record-every seconds GET /slots shows each slot's last prompt (llama-server lists it only with
     LLAMA_SERVER_SLOTS_DEBUG=1, which the serve script sets). That text is detokenized with llama.cpp's clean_spaces
     (" 're" -> "'re"), so it only tells WHEN the prefix changed and where two sessions' system turns part;
   - then, while no slot is busy, the exact tokens: POST /slots/<id>?action=save (--slot-save-path) writes the slot
     (~0.1 s of server time; the file holds the slot's state, its context checkpoints alone ~0.5 GB at 13.6k tokens),
     the helper reads the token list from the file's header and deletes the file at once; /detokenize (plain
     concatenation of the tokens' text) maps the cut points onto those tokens. Once per new prefix (prefixes read
     before are kept in memory); a slot above --max-capture-tokens is left for a shorter one;
   - writes them (0600) to --prefix-file with the system turn of the latest session (to pair it with the next
     session after a restart). A record made with another chat template or model file is not prefilled (the next
     chat replaces it); before the first chat on a host there is nothing to prefill.
   A slot is saved only while the save directory has --min-free-gb free before the save starts (4 GiB; the save
   then writes its slot file on top, deleted right after the read): with less, the changed prefix waits (logged once)
   instead of pushing other writers on the robot host into "disk full". Capture files are deleted right after each
   read and swept at every read of GET /slots (one a timed-out save wrote late).

It exits when the server process ends. Standard library only; the key is read from its file, never from argv.
What this needs from llama-server (LLAMA_SERVER_SLOTS_DEBUG=1, --slot-save-path) widens its API for every holder of
the key: docs/OPENSHELL.md, "Brain warm start".
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import struct
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener

#: OpenClaw's prompt-cache boundary in its system text: what follows may change from chat to chat.
BOUNDARY = "<!-- openclaw:attempt:DYNAMIC -->"
#: How every agent prompt opens and how a turn ends (Nemotron 3.5 Bolt's ChatML).
SYSTEM_TURN, TURN_END = "<|im_start|>system\n", "<|im_end|>"
FORMAT = 2
#: llama.cpp's slot file (llama_state_seq_save_file): magic 'ggsq', version 4, then the packed server tokens.
GGSQ, GGSQ_VERSION, PACKED_VERSION = 0x67677371, 4, 1
CAPTURE_GLOB = "brain-warm-capture-*.bin"
CAPTURE_TRIES = 3   # slot saves per new prefix (one succeeds unless the slot was reused in between)
KNOWN_PREFIXES = 8  # prefixes kept in memory, so OpenClaw agents that alternate never cost another slot save
SESSIONS = 4        # system turns of recent OpenClaw sessions kept to find what sessions share
#: free space a slot save needs before it starts; the save then writes its slot file on top (its context checkpoints
#: alone ~0.5 GB at 13.6k tokens, more when longer), so less than this stays free while the file exists
MIN_FREE_GB = 4.0
_OPENER = build_opener(ProxyHandler({}))   # loopback only: never through an HTTP(S)_PROXY


def log(message: str) -> None:
    print(f"[brain-warm] {message}", file=sys.stderr, flush=True)


def template_id(path: str) -> str:
    """Which chat template rendered the recorded prompt: the file's sha256, or the GGUF's own template."""
    if not path:
        return "gguf"
    return "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()


def model_id(path: str) -> str:
    """Whose token ids the record holds: the served model file (pinned by its manifest), by name and size."""
    p = Path(path)
    return f"{p.name}:{p.stat().st_size}" if path and p.exists() else ""


def openclaw_prefix(prompt) -> str | None:
    """The stable prefix of a rendered OpenClaw request: its system turn up to OpenClaw's boundary, else None."""
    if not isinstance(prompt, str):
        return None
    start, cut = prompt.find(SYSTEM_TURN), prompt.find(BOUNDARY)
    if start < 0 or cut < start or "<|im_start|>" in prompt[:start]:
        return None              # not an agent request, or the system turn is not the first turn
    prefix = prompt[start:cut].rstrip()
    return prefix if len(prefix) > len(SYSTEM_TURN) else None


def system_turn(prompt) -> str | None:
    """An OpenClaw request's text up to the end of its system turn, else None."""
    if openclaw_prefix(prompt) is None:
        return None
    end = prompt.find(TURN_END, prompt.find(BOUNDARY))
    return prompt[:end] if end >= 0 else None


def shared_end(turn: str, other: str) -> int | None:
    """Where the system turns of two OpenClaw sessions part, when past the boundary: the length of the text they
    share, cut back to a word end (ids that merely start alike are not shared). None for one session (its turns are
    alike) or when nothing past the boundary is shared."""
    cut = turn.find(BOUNDARY) + len(BOUNDARY)
    n = len(os.path.commonprefix([turn, other]))
    if n >= min(len(turn), len(other)) or n <= cut:
        return None
    while n > cut and turn[n - 1].isalnum():
        n -= 1
    return n if n > cut else None


def aligned(exact: str, cleaned: str, n: int) -> int | None:
    """The length of the exact text that reads cleaned[:n] (llama.cpp's clean_spaces only drops spaces), else None.
    Both are read from their system turn on (whatever opens the text, such as a BOS, is not compared)."""
    i, j = exact.find(SYSTEM_TURN), cleaned.find(SYSTEM_TURN)
    if i < 0 or j < 0 or n < j:
        return None
    while j < n:
        if i < len(exact) and j < len(cleaned) and exact[i] == cleaned[j]:
            i, j = i + 1, j + 1
        elif i < len(exact) and exact[i] == " ":
            i += 1
        else:
            return None
    return i


def read_record(path: Path, template: str, model: str) -> tuple[dict | None, str]:
    """(record to prefill, why not)."""
    try:
        record = json.loads(path.read_text())
    except FileNotFoundError:
        return None, f"no recorded OpenClaw prefix yet ({path}); the first chat records it"
    except (OSError, ValueError) as e:
        return None, f"unreadable record {path}: {e}"
    if not isinstance(record, dict) or record.get("format") != FORMAT:
        return None, f"unknown record format in {path}"
    if record.get("template") != template or record.get("model") != model:
        return None, f"{path} was recorded with another chat template or model; the next chat records this one"

    def ids(value) -> bool:
        return isinstance(value, list) and all(isinstance(t, int) and not isinstance(t, bool) and t >= 0 for t in value)

    if not (ids(record.get("tokens")) and record["tokens"] and ids(record.setdefault("extension", []))
            and all(isinstance(record.setdefault(k, ""), str) for k in ("text", "shared", "last"))):
        return None, f"{path} holds no prefix tokens"
    return record, ""


def write_record(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump({"format": FORMAT, **record}, stream)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)       # a failed write leaves the old record and nothing else
        raise


def slot_file_tokens(path: Path) -> list[int] | None:
    """The prompt tokens at the head of a llama-server slot file (LLAMA_TOKEN_NULL = -1 marks media), else None."""
    try:
        with path.open("rb") as stream:
            magic, version, n_packed = struct.unpack("<III", stream.read(12))
            if magic != GGSQ or version != GGSQ_VERSION or n_packed < 3:
                return None
            marker, packed_version, count = struct.unpack("<iII", stream.read(12))
            if marker != -1 or packed_version != PACKED_VERSION or count > n_packed - 3:
                return None
            data = stream.read(4 * count)
            return list(struct.unpack(f"<{count}i", data)) if len(data) == 4 * count else None
    except (OSError, struct.error):
        return None


def _start_time(pid: int) -> str | None:
    """Field 22 of /proc/<pid>/stat (unchanged by exec), or None where /proc is missing."""
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
    except (OSError, IndexError):
        return None


def server_alive(pid: int, started: str | None) -> bool:
    """True while the server process runs: not gone, not a zombie, not a recycled PID."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    except OSError:
        return started is None   # no /proc (macOS): kill(0) is all there is
    return fields[0] not in ("Z", "X") and (started is None or fields[19] == started)


class Server:
    def __init__(self, host: str, port: int, key: str):
        self.base, self.key = f"http://{host}:{port}", key

    def call(self, path: str, body: dict | None = None, timeout: float = 10.0):
        """(HTTP status or None when unreachable, decoded JSON or None)."""
        headers = {"Content-Type": "application/json"}
        if self.key:
            headers["Authorization"] = f"Bearer {self.key}"
        data = None if body is None else json.dumps(body).encode()
        try:
            with _OPENER.open(Request(self.base + path, data=data, headers=headers), timeout=timeout) as reply:
                return reply.status, json.loads(reply.read() or b"null")
        except HTTPError as e:
            return e.code, None
        except (URLError, OSError, ValueError):
            return None, None

    def text(self, tokens: list[int]) -> str | None:
        """The exact text of tokens (/detokenize concatenates the tokens' text, no space cleaning)."""
        status, reply = self.call("/detokenize", {"tokens": tokens})
        return reply.get("content") if status == 200 and isinstance(reply, dict) else None


def warm(server: Server, tokens: list[int], slots: int, timeout: float, what: str) -> bool:
    t0 = time.monotonic()
    body = {"prompt": tokens, "n_predict": 1, "n_cmpl": slots, "cache_prompt": True, "temperature": 0.0}
    status, reply = server.call("/completion", body, timeout=timeout)
    took = time.monotonic() - t0
    if status != 200:
        why = "timed out" if status is None and took >= timeout - 0.5 else f"answered {status}"
        log(f"gave up the warm-up after {took:.1f} s ({why}); chats prefill the prefix themselves")
        return False
    results = reply if isinstance(reply, list) else [reply]
    log(f"warmed {len(results)} slot(s) with {what} in {took:.1f} s")
    return True


def prompts_hidden(slots) -> bool:
    """llama-server lists a slot's last prompt only with LLAMA_SERVER_SLOTS_DEBUG set."""
    ran = [s for s in slots if isinstance(s, dict) and s.get("id_task") is not None] if isinstance(slots, list) else []
    return bool(ran) and not any("prompt" in s for s in ran)


def openclaw_slots(slots) -> list[dict]:
    """The slots whose last request came from OpenClaw, oldest request first."""
    found = [s for s in slots if isinstance(s, dict) and system_turn(s.get("prompt")) is not None] \
        if isinstance(slots, list) else []
    return sorted(found, key=lambda s: s["id_task"] if isinstance(s.get("id_task"), int) else -1)   # task 0 is a task


def sweep_captures(save_dir: Path) -> None:
    """Delete capture files: one left by a helper that died mid-capture, or written late after a save timed out."""
    for stale in save_dir.glob(CAPTURE_GLOB):
        stale.unlink(missing_ok=True)


def read_slot(server: Server, slot: dict, save_dir: Path) -> tuple[list[int], str] | None:
    """The tokens and exact text of the slot's prompt, from a slot file deleted at once; None when the slot no longer
    holds the prompt GET /slots showed."""
    name = f"brain-warm-capture-{os.getpid()}.bin"
    try:
        status, _ = server.call(f"/slots/{slot['id']}?action=save", {"filename": name}, timeout=120.0)
        tokens = slot_file_tokens(save_dir / name) if status == 200 else None
    finally:
        (save_dir / name).unlink(missing_ok=True)
    if not tokens:
        return None
    tokens = tokens[:tokens.index(-1)] if -1 in tokens else tokens        # the text before any image
    exact = server.text(tokens)
    if exact is None or aligned(exact, slot["prompt"], len(system_turn(slot["prompt"]) or "")) is None:
        return None
    return tokens, exact


def tokens_within(server: Server, tokens: list[int], limit: str, lo: int = 0) -> int:
    """The longest run of tokens whose exact text lies within limit. A token's text is appended to the text before
    it, so a binary search over the count finds it (~15 /detokenize calls, on llama-server's HTTP threads). A run
    whose text cannot be read counts as not within: the result may be shorter, never longer."""
    hi = len(tokens)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        head = server.text(tokens[:mid])
        lo, hi = (mid, hi) if head is not None and limit.startswith(head) else (lo, mid - 1)
    return lo


def cut_points(server: Server, got: tuple[list[int], str], prompt: str, end: int | None) -> tuple[int, int] | None:
    """(tokens before OpenClaw's boundary, tokens up to the shared end) of a slot read by read_slot."""
    tokens, exact = got
    if BOUNDARY not in exact:
        return None
    n1 = tokens_within(server, tokens, exact[:exact.index(BOUNDARY)].rstrip())
    if not n1:
        return None
    at = aligned(exact, prompt, end) if end else None
    return n1, tokens_within(server, tokens, exact[:at], n1) if at else n1


class Recorder:
    """Step 3: keeps the record equal to what OpenClaw last sent. Each step() reads GET /slots once and, for a changed
    prefix, saves one idle slot to read its exact tokens."""

    def __init__(self, server: Server, *, prefix_file: Path, slot_save_dir: Path, identity: dict,
                 record: dict | None, record_every: float = 10.0, max_capture_tokens: int = 32768,
                 min_free_bytes: int = int(MIN_FREE_GB * 2**30), clock=time.monotonic, free_bytes=None):
        self.server, self.prefix_file, self.save_dir, self.identity = server, prefix_file, slot_save_dir, identity
        self.record_every, self.max_capture_tokens, self.min_free_bytes = record_every, max_capture_tokens, min_free_bytes
        self.clock = clock
        self.free_bytes = free_bytes or (lambda path: shutil.disk_usage(path).free)
        self.seen = (record["text"], record["shared"]) if record else None
        self.known = {self.seen: (record["tokens"], record["extension"])} if record else {}
        self.turns = [record["last"]] if record and record["last"] else []
        self.tries: dict = {}            # slot saves per changed prefix, the latest KNOWN_PREFIXES prefixes
        self.next_read, self.hidden, self.large, self.low = clock(), False, None, None

    def due(self) -> bool:
        return self.clock() >= self.next_read

    def step(self) -> None:
        self.next_read = self.clock() + self.record_every
        sweep_captures(self.save_dir)
        status, slots = self.server.call("/slots")
        if status == 200 and not self.hidden and prompts_hidden(slots):
            self.hidden = True
            log("GET /slots shows no prompts (llama-server without LLAMA_SERVER_SLOTS_DEBUG=1): nothing to record")
        found = openclaw_slots(slots) if status == 200 else []
        for older in found[:-1]:          # the sessions of the other slots, to compare the latest one with
            if (t := system_turn(older["prompt"]) or "") not in self.turns:
                self.turns = (self.turns + [t])[-SESSIONS:]
        if not found:
            return
        slot = found[-1]
        prompt = slot["prompt"]
        turn, text = system_turn(prompt) or "", openclaw_prefix(prompt) or ""
        end = next((e for other in reversed(self.turns) if (e := shared_end(turn, other))), None)
        if turn not in self.turns:
            self.turns = (self.turns + [turn])[-SESSIONS:]
        want = (text, turn[turn.find(BOUNDARY):end] if end else "")
        if not end and self.seen and self.seen[0] == text:
            want = self.seen               # one session so far: keep what earlier sessions shared
        entry, how = self.known.get(want), "read before"
        if entry is None:
            entry, want, how = self._capture(slot, slots, want, text, end)
        if entry and want != self.seen:
            try:
                write_record(self.prefix_file, {**self.identity, "text": want[0], "tokens": entry[0],
                                                "shared": want[1], "extension": entry[1], "last": turn})
            except OSError as e:
                log(f"cannot write {self.prefix_file}: {e}")
            else:
                log(f"recorded the latest OpenClaw prefix ({len(entry[0])} tokens + {len(entry[1])} its sessions "
                    f"share; {how}) for the next start")
                self.seen = want

    def _capture(self, slot: dict, slots: list, want: tuple, text: str, end: int | None):
        """(entry or None, want, how) for a prefix not read before: at most CAPTURE_TRIES slot saves, each while the
        server is idle and the save directory keeps its free space."""
        if (slot.get("n_prompt_tokens") or 0) > self.max_capture_tokens:
            if self.large != want:
                self.large = want
                log(f"a changed OpenClaw prefix sits in a {slot['n_prompt_tokens']}-token slot; "
                    "recording it from a shorter one")
            return None, want, ""
        if any(s.get("is_processing") for s in slots if isinstance(s, dict)):
            self.next_read = self.clock() + min(self.record_every, 2.0)    # save a slot only while idle
            return None, want, ""
        if self.tries.get(want, 0) >= CAPTURE_TRIES or self._disk_low(want):
            return None, want, ""
        n = self.tries.pop(want, 0) + 1
        self.tries = {**dict(list(self.tries.items())[-(KNOWN_PREFIXES - 1):]), want: n}
        t1 = self.clock()
        got = read_slot(self.server, slot, self.save_dir)
        cuts = cut_points(self.server, got, slot["prompt"], end) if got else None
        if not cuts:
            if n == CAPTURE_TRIES:
                log(f"could not read the tokens of a changed OpenClaw prefix {CAPTURE_TRIES} times; "
                    "the next start prefills the previous record")
            return None, want, ""
        if cuts[1] == cuts[0]:            # nothing shared, or it could not be read: the prefix alone
            want = (text, "")
        entry = (got[0][:cuts[0]], got[0][cuts[0]:cuts[1]])
        self.known = {**dict(list(self.known.items())[-(KNOWN_PREFIXES - 1):]), want: entry}
        return entry, want, f"slot {slot['id']} read in {self.clock() - t1:.1f} s"

    def _disk_low(self, want: tuple) -> bool:
        """True while a slot save could fill the disk (logged once per changed prefix); it is retried later."""
        try:
            free, why = self.free_bytes(self.save_dir), ""
        except OSError as e:
            free, why = 0, f" (cannot read it: {e})"
        if free >= self.min_free_bytes:
            self.low = None
            return False
        if self.low != want:
            self.low = want
            log(f"{free / 2**30:.1f} GiB free in {self.save_dir}{why}, under the {self.min_free_bytes / 2**30:.1f} GiB "
                "a slot save needs before it starts: the changed OpenClaw prefix is recorded once there is room")
        return True


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Prefill the recorded OpenClaw prefix into a starting llama-server.")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--server-pid", type=int, required=True, help="the llama-server process (the helper ends with it)")
    p.add_argument("--slots", type=int, required=True, help="llama-server --parallel")
    p.add_argument("--prefix-file", type=Path, required=True, help="the recorded OpenClaw prefix")
    p.add_argument("--slot-save-dir", type=Path, help="llama-server --slot-save-path (needed to record)")
    p.add_argument("--chat-template-file", default="", help="the server's --chat-template-file ('' = the GGUF's)")
    p.add_argument("--model-file", default="", help="the server's --model")
    p.add_argument("--api-key-file", default="", help="0600 file with the server's bearer key")
    p.add_argument("--timeout", type=float, default=120.0, help="bound of each warm-up request, seconds")
    p.add_argument("--record-every", type=float, default=10.0, help="seconds between reads of the latest prompt")
    p.add_argument("--max-capture-tokens", type=int, default=32768, help="larger slots are not saved to record")
    p.add_argument("--min-free-gb", type=float, default=MIN_FREE_GB,
                   help="free space --slot-save-dir needs before a slot save starts, else it waits (the save then "
                        "writes its slot file on top: ~0.5 GB of context checkpoints at 13.6k tokens, more when longer)")
    p.add_argument("--poll", type=float, default=0.5, help=argparse.SUPPRESS)
    args = p.parse_args(argv)

    started = _start_time(args.server_pid)

    def alive() -> bool:
        return server_alive(args.server_pid, started)

    key = Path(args.api_key_file).expanduser().read_text().strip() if args.api_key_file else ""
    identity = {"template": template_id(args.chat_template_file), "model": model_id(args.model_file)}
    server = Server(args.host, args.port, key)
    if args.slot_save_dir:
        sweep_captures(args.slot_save_dir)
    log(f"warm start for {args.host}:{args.port}: {args.slots} slot(s), prefix record {args.prefix_file}")

    t0 = time.monotonic()
    while True:                                       # 1. the server's own readiness
        if not alive():
            return 0
        if server.call("/health", timeout=2.0)[0] == 200:
            break
        time.sleep(args.poll)
    log(f"server ready after {time.monotonic() - t0:.1f} s")

    record, why = read_record(args.prefix_file, **identity)
    if record:                                        # 2. prefill the recorded prefix into every slot
        tokens, extension = record["tokens"], record["extension"]
        if warm(server, tokens, args.slots, args.timeout, f"the recorded OpenClaw prefix ({len(tokens)} tokens)") \
                and extension:
            warm(server, tokens + extension, args.slots, args.timeout,
                 f"what the last OpenClaw sessions shared after it (+{len(extension)} tokens)")
    else:
        log(why)

    if args.slot_save_dir:                            # 3. keep the record equal to what OpenClaw last sent
        recorder = Recorder(server, prefix_file=args.prefix_file, slot_save_dir=args.slot_save_dir,
                            identity=identity, record=record, record_every=args.record_every,
                            max_capture_tokens=args.max_capture_tokens,
                            min_free_bytes=int(args.min_free_gb * 2**30))
        while alive():
            if recorder.due():
                recorder.step()
            time.sleep(args.poll)
    while alive():
        time.sleep(args.poll)
    return 0


if __name__ == "__main__":
    sys.exit(main())
