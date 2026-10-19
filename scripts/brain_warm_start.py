#!/usr/bin/env python3
"""Warm start for the llama.cpp brain: the first chat after a start reuses OpenClaw's prefix like any later chat.

scripts/serve_nemotron_bolt_llamacpp.sh starts this helper beside llama-server (WARM_START=1, the default).

Every OpenClaw agent request opens with the same prefix: its stable system text and the tool block (cascade's chat
template renders OpenClaw's per-session text AFTER the tools: scripts/chat_templates/). llama.cpp reuses that prefix
from chat to chat, but right after a start nothing is cached and the first chat re-prefilled all ~13.6k tokens
(13.6 s with the brain alone). So the helper:

1. waits until llama-server answers /health. That is the server's own readiness; the helper never delays it.
2. prefills the RECORDED prefix tokens once and copies them into every slot: ONE raw /completion (n_cmpl = slots,
   n_predict 1). A raw token prompt is no chat: no conversation, transcript or user turn exists anywhere. Bounded
   by --timeout (120 s; 12.5k tokens take ~13 s alone). A request that arrives meanwhile waits for a free slot.
3. records the prefix while the server runs, from the server's own tokens of the latest OpenClaw request, so it
   never builds or re-tokenizes a prompt (no drift):
   - every --record-every seconds GET /slots shows each slot's last prompt (llama-server lists it only with
     LLAMA_SERVER_SLOTS_DEBUG=1, which the serve script sets). That text is detokenized with llama.cpp's
     clean_spaces (" 're" -> "'re"), so it only tells WHEN OpenClaw's stable text changed;
   - then the exact tokens: POST /slots/<id>?action=save (--slot-save-path) writes the slot, the helper reads the
     token list from the file's header and deletes the file at once. Once per new prefix (prefixes read before are
     kept in memory): the save pauses the server while it writes the slot (~1 GB for 13.6k tokens), so a slot above
     --max-capture-tokens is left for a later, shorter one;
   - it keeps the tokens whose exact text (/detokenize: plain concatenation) lies before OpenClaw's prompt-cache
     boundary (`<!-- openclaw:attempt:DYNAMIC -->`: before it the system text and tools, after it the date and the
     session ids) and writes them (0600) to --prefix-file.
   A record made with another chat template or model file is not prefilled (the next chat replaces it); before the
   first chat on a host there is nothing to prefill.

It exits when the server process ends. Standard library only; the key is read from its file, never from argv.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener

#: OpenClaw's prompt-cache boundary in its system text: what follows changes from chat to chat.
BOUNDARY = "<!-- openclaw:attempt:DYNAMIC -->"
#: How every agent prompt opens (Nemotron 3.5 Bolt's ChatML system turn).
SYSTEM_TURN = "<|im_start|>system\n"
FORMAT = 2
#: llama.cpp's slot file (llama_state_seq_save_file): magic 'ggsq', version 4, then the packed server tokens.
GGSQ, GGSQ_VERSION, PACKED_VERSION = 0x67677371, 4, 1
CAPTURE_GLOB = "brain-warm-capture-*.bin"
CAPTURE_TRIES = 3   # slot saves per changed prefix (one succeeds unless the slot was reused in between)
KNOWN_PREFIXES = 8  # prefixes kept in memory, so OpenClaw agents that alternate never cost another slot save
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
    tokens = record.get("tokens")
    if not (isinstance(tokens, list) and tokens and all(isinstance(t, int) and t >= 0 for t in tokens)
            and isinstance(record.get("text"), str)):
        return None, f"{path} holds no prefix tokens"
    return record, ""


def write_record(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as stream:
        json.dump({"format": FORMAT, **record}, stream)
    os.replace(tmp, path)


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


def warm(server: Server, tokens: list[int], slots: int, timeout: float) -> None:
    t0 = time.monotonic()
    body = {"prompt": tokens, "n_predict": 1, "n_cmpl": slots, "cache_prompt": True, "temperature": 0.0}
    status, reply = server.call("/completion", body, timeout=timeout)
    took = time.monotonic() - t0
    if status != 200:
        why = "timed out" if status is None and took >= timeout - 0.5 else f"answered {status}"
        log(f"gave up the warm-up after {took:.1f} s ({why}); chats prefill the prefix themselves")
        return
    results = reply if isinstance(reply, list) else [reply]
    log(f"warmed {len(results)} slot(s) with the recorded OpenClaw prefix ({len(tokens)} tokens) in {took:.1f} s")


def prompts_hidden(slots) -> bool:
    """llama-server lists a slot's last prompt only with LLAMA_SERVER_SLOTS_DEBUG set."""
    ran = [s for s in slots if isinstance(s, dict) and s.get("id_task") is not None] if isinstance(slots, list) else []
    return bool(ran) and not any("prompt" in s for s in ran)


def latest_openclaw_slot(slots) -> tuple[dict, str] | None:
    """(slot, cleaned prefix text) of the most recent OpenClaw request among the slots."""
    best = None
    for slot in slots if isinstance(slots, list) else []:
        prefix = openclaw_prefix(slot.get("prompt")) if isinstance(slot, dict) else None
        if prefix and (best is None or (slot.get("id_task") or -1) > (best[0].get("id_task") or -1)):
            best = (slot, prefix)
    return best


def capture(server: Server, slot_id: int, save_dir: Path) -> list[int] | None:
    """The slot's exact prompt tokens up to OpenClaw's boundary, from a slot file deleted at once."""
    name = f"brain-warm-capture-{os.getpid()}.bin"
    try:
        status, _ = server.call(f"/slots/{slot_id}?action=save", {"filename": name}, timeout=120.0)
        tokens = slot_file_tokens(save_dir / name) if status == 200 else None
    finally:
        (save_dir / name).unlink(missing_ok=True)
    if not tokens:
        return None
    tokens = tokens[:tokens.index(-1)] if -1 in tokens else tokens        # the text before any image

    def text_of(n: int) -> str | None:                                   # the exact text (no space cleaning)
        status, reply = server.call("/detokenize", {"tokens": tokens[:n]})
        return reply.get("content") if status == 200 and isinstance(reply, dict) else None

    text = text_of(len(tokens))
    if text is None or openclaw_prefix(text) is None:
        return None
    stable = text[:text.index(BOUNDARY)].rstrip()
    # The longest run of the slot's own tokens whose text lies within the stable text: a token's text is appended to
    # the text before it, so a binary search over the count finds it (~15 /detokenize calls, HTTP thread only).
    lo, hi = 0, len(tokens)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        head = text_of(mid)
        if head is None:
            return None
        if stable.startswith(head):
            lo = mid
        else:
            hi = mid - 1
    return tokens[:lo] if lo else None


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
    p.add_argument("--timeout", type=float, default=120.0, help="bound of the warm-up request, seconds")
    p.add_argument("--record-every", type=float, default=10.0, help="seconds between reads of the latest prompt")
    p.add_argument("--max-capture-tokens", type=int, default=32768, help="larger slots are not saved to record")
    p.add_argument("--poll", type=float, default=0.5, help=argparse.SUPPRESS)
    args = p.parse_args(argv)

    started = _start_time(args.server_pid)

    def alive() -> bool:
        return server_alive(args.server_pid, started)

    key = Path(args.api_key_file).expanduser().read_text().strip() if args.api_key_file else ""
    identity = {"template": template_id(args.chat_template_file), "model": model_id(args.model_file)}
    server = Server(args.host, args.port, key)
    if args.slot_save_dir:
        for stale in args.slot_save_dir.glob(CAPTURE_GLOB):       # left by a helper that died mid-capture
            stale.unlink(missing_ok=True)
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
        warm(server, record["tokens"], args.slots, args.timeout)
    else:
        log(why)

    seen = record["text"] if record else None
    known = {seen: record["tokens"]} if record else {}   # prefixes read this run: an alternation saves no slot again
    next_read, hidden, large, tries = time.monotonic(), False, None, {}
    while alive() and args.slot_save_dir:             # 3. keep the record equal to what OpenClaw last sent
        if time.monotonic() >= next_read:
            next_read = time.monotonic() + args.record_every
            status, slots = server.call("/slots")
            if status == 200 and not hidden and prompts_hidden(slots):
                hidden = True
                log("GET /slots shows no prompts (llama-server without LLAMA_SERVER_SLOTS_DEBUG=1): nothing to record")
            found = latest_openclaw_slot(slots) if status == 200 else None
            if found and found[1] != seen:
                slot, text = found
                tokens, how = known.get(text), "read before"
                if tokens is None and (slot.get("n_prompt_tokens") or 0) > args.max_capture_tokens:
                    if large != text:
                        large = text
                        log(f"a changed OpenClaw prefix sits in a {slot['n_prompt_tokens']}-token slot; "
                            "recording it from a shorter one")
                elif tokens is None and tries.get(text, 0) < CAPTURE_TRIES:
                    tries[text] = tries.get(text, 0) + 1
                    t0 = time.monotonic()
                    tokens = capture(server, slot["id"], args.slot_save_dir)
                    how = f"slot {slot['id']} read in {time.monotonic() - t0:.1f} s"
                    if tokens:
                        known = {**dict(list(known.items())[-(KNOWN_PREFIXES - 1):]), text: tokens}
                    elif tries[text] == CAPTURE_TRIES:
                        log(f"could not read the tokens of a changed OpenClaw prefix {CAPTURE_TRIES} times; "
                            "the next start prefills the previous record")
                if tokens:
                    try:
                        write_record(args.prefix_file, {**identity, "text": text, "tokens": tokens})
                    except OSError as e:
                        log(f"cannot write {args.prefix_file}: {e}")
                    else:
                        seen = text
                        log(f"recorded the latest OpenClaw prefix ({len(tokens)} tokens, {how}) for the next start")
        time.sleep(args.poll)
    while alive():
        time.sleep(args.poll)
    return 0


if __name__ == "__main__":
    sys.exit(main())
