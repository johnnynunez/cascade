#!/usr/bin/env python3
"""Warm start for the llama.cpp brain: the first chat after a start reuses OpenClaw's prefix like any later chat.

scripts/serve_nemotron_bolt_llamacpp.sh starts this helper beside llama-server (WARM_START=1, the default).

Every OpenClaw agent request opens with the same prefix: its stable system text and the tool block (cascade's chat
template renders OpenClaw's per-session text AFTER the tools: scripts/chat_templates/). llama.cpp reuses that prefix
from chat to chat, but right after a start nothing is cached and the first chat re-prefilled all ~13.6k tokens
(13.8 s with the brain alone). So the helper:

1. waits until llama-server answers /health. That is the server's own readiness; the helper never delays it.
2. prefills the RECORDED prefix once and copies it into every slot: ONE raw /completion (n_cmpl = slots,
   n_predict 1). A raw prompt is no chat: no conversation, transcript or user turn exists anywhere. Bounded by
   --timeout (120 s; 13.6k tokens take ~14 s alone). A request that arrives meanwhile waits for a free slot, then
   reuses the prefix.
3. records the prefix while the server runs: every --record-every seconds it reads the server's own rendering of
   the latest OpenClaw request (GET /slots, the detokenized prompt; llama-server shows it only with
   LLAMA_SERVER_SLOTS_DEBUG=1, which the serve script sets), keeps the text before OpenClaw's prompt-cache
   boundary (`<!-- openclaw:attempt:DYNAMIC -->`: before it the system text and tools, after it the date and the
   session ids) and writes it (0600) when it changed. The next start prefills exactly what OpenClaw last sent; no
   prefix is ever built here. A record made with another chat template is not prefilled (the next chat replaces
   it); before the first chat on a host there is nothing to prefill.

It exits when the server process ends. Standard library only; the key is read from its file, never from argv.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener

#: OpenClaw's prompt-cache boundary in its system text: what follows changes from chat to chat.
BOUNDARY = "<!-- openclaw:attempt:DYNAMIC -->"
#: How every agent prompt opens (Nemotron 3.5 Bolt's ChatML system turn).
SYSTEM_TURN = "<|im_start|>system\n"
FORMAT = 1
#: A record larger than this is not a prompt prefix (the agent prompt is ~60k characters).
MAX_PREFIX_CHARS = 1_000_000
_OPENER = build_opener(ProxyHandler({}))   # loopback only: never through an HTTP(S)_PROXY


def log(message: str) -> None:
    print(f"[brain-warm] {message}", file=sys.stderr, flush=True)


def template_id(path: str) -> str:
    """Which chat template rendered the recorded text: the file's sha256, or the GGUF's own template."""
    if not path:
        return "gguf"
    return "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()


def openclaw_prefix(prompt) -> str | None:
    """The stable prefix of a rendered OpenClaw request: its system turn up to OpenClaw's boundary, else None."""
    if not isinstance(prompt, str):
        return None
    start, cut = prompt.find(SYSTEM_TURN), prompt.find(BOUNDARY)
    if start < 0 or cut < start or "<|im_start|>" in prompt[:start]:
        return None              # not an agent request, or the system turn is not the first turn
    prefix = prompt[start:cut].rstrip()
    return prefix if len(prefix) > len(SYSTEM_TURN) else None


def usable_prefix(prefix) -> bool:
    return (isinstance(prefix, str) and prefix.startswith(SYSTEM_TURN) and len(prefix) > len(SYSTEM_TURN)
            and BOUNDARY not in prefix and len(prefix) <= MAX_PREFIX_CHARS)


def read_record(path: Path, template: str) -> tuple[str | None, str]:
    """(prefix to prefill, why not)."""
    try:
        record = json.loads(path.read_text())
    except FileNotFoundError:
        return None, f"no recorded OpenClaw prefix yet ({path}); the first chat records it"
    except (OSError, ValueError) as e:
        return None, f"unreadable record {path}: {e}"
    if not isinstance(record, dict) or record.get("format") != FORMAT:
        return None, f"unknown record format in {path}"
    if record.get("template") != template:
        return None, f"{path} was recorded with another chat template; the next chat records this one"
    if not usable_prefix(record.get("prefix")):
        return None, f"{path} holds no stable system prefix"
    return record["prefix"], ""


def write_record(path: Path, template: str, prefix: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as stream:
        json.dump({"format": FORMAT, "template": template, "prefix": prefix}, stream)
    os.replace(tmp, path)


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


def warm(server: Server, prefix: str, slots: int, timeout: float) -> None:
    t0 = time.monotonic()
    body = {"prompt": prefix, "n_predict": 1, "n_cmpl": slots, "cache_prompt": True, "temperature": 0.0}
    status, reply = server.call("/completion", body, timeout=timeout)
    took = time.monotonic() - t0
    if status != 200:
        why = "timed out" if status is None and took >= timeout - 0.5 else f"answered {status}"
        log(f"gave up the warm-up after {took:.1f} s ({why}); chats prefill the prefix themselves")
        return
    results = reply if isinstance(reply, list) else [reply]
    tokens = max((r.get("timings", {}).get("prompt_n", 0) for r in results if isinstance(r, dict)), default=0)
    log(f"warmed {len(results)} slot(s) with the recorded OpenClaw prefix ({tokens} tokens) in {took:.1f} s")


def prompts_hidden(slots) -> bool:
    """llama-server lists a slot's last prompt only with LLAMA_SERVER_SLOTS_DEBUG set."""
    ran = [s for s in slots if isinstance(s, dict) and s.get("id_task") is not None] if isinstance(slots, list) else []
    return bool(ran) and not any("prompt" in s for s in ran)


def latest_openclaw_prefix(slots) -> str | None:
    best = None
    for slot in slots if isinstance(slots, list) else []:
        prefix = openclaw_prefix(slot.get("prompt")) if isinstance(slot, dict) else None
        task = slot.get("id_task", -1) if isinstance(slot, dict) else -1
        if prefix and (best is None or task > best[0]):
            best = (task, prefix)
    return best[1] if best else None


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Prefill the recorded OpenClaw prefix into a starting llama-server.")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--server-pid", type=int, required=True, help="the llama-server process (the helper ends with it)")
    p.add_argument("--slots", type=int, required=True, help="llama-server --parallel")
    p.add_argument("--prefix-file", type=Path, required=True, help="the recorded OpenClaw prefix")
    p.add_argument("--chat-template-file", default="", help="the server's --chat-template-file ('' = the GGUF's)")
    p.add_argument("--api-key-file", default="", help="0600 file with the server's bearer key")
    p.add_argument("--timeout", type=float, default=120.0, help="bound of the warm-up request, seconds")
    p.add_argument("--record-every", type=float, default=10.0, help="seconds between reads of the latest prompt")
    p.add_argument("--poll", type=float, default=0.5, help=argparse.SUPPRESS)
    args = p.parse_args(argv)

    started = _start_time(args.server_pid)

    def alive() -> bool:
        return server_alive(args.server_pid, started)

    key = Path(args.api_key_file).expanduser().read_text().strip() if args.api_key_file else ""
    template = template_id(args.chat_template_file)
    server = Server(args.host, args.port, key)
    log(f"warm start for {args.host}:{args.port}: {args.slots} slot(s), prefix record {args.prefix_file}")

    t0 = time.monotonic()
    while True:                                       # 1. the server's own readiness
        if not alive():
            return 0
        if server.call("/health", timeout=2.0)[0] == 200:
            break
        time.sleep(args.poll)
    log(f"server ready after {time.monotonic() - t0:.1f} s")

    current, why = read_record(args.prefix_file, template)
    if current:                                       # 2. prefill the recorded prefix into every slot
        warm(server, current, args.slots, args.timeout)
    else:
        log(why)

    next_read, hidden = time.monotonic(), False
    while alive():                                    # 3. keep the record equal to what OpenClaw last sent
        if time.monotonic() >= next_read:
            next_read = time.monotonic() + args.record_every
            status, slots = server.call("/slots")
            if status == 200 and not hidden and prompts_hidden(slots):
                hidden = True
                log("GET /slots shows no prompts (llama-server without LLAMA_SERVER_SLOTS_DEBUG=1): nothing to record")
            prefix = latest_openclaw_prefix(slots) if status == 200 else None
            if prefix and prefix != current and usable_prefix(prefix):
                try:
                    write_record(args.prefix_file, template, prefix)
                except OSError as e:
                    log(f"cannot write {args.prefix_file}: {e}")
                else:
                    current = prefix
                    log(f"recorded the latest OpenClaw prefix ({len(prefix)} characters) for the next start")
        time.sleep(args.poll)
    return 0


if __name__ == "__main__":
    sys.exit(main())
