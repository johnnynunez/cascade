"""scripts/brain_warm_start.py: the first chat after a Bolt brain start reuses OpenClaw's prefix, hermetically.

The helper runs beside llama-server (scripts/serve_nemotron_bolt_llamacpp.sh starts it). Once the server answers
/health it prefills the last recorded OpenClaw prefix TOKENS (stable system text + tool block, as the server itself
tokenized them) into every slot with a raw /completion (n_cmpl = slots), then what the last two sessions shared after
OpenClaw's boundary. While the server runs it notices a changed prefix in GET /slots (text detokenized with
clean_spaces, so unusable as a prompt) and reads the exact tokens from a slot file (POST /slots/<id>?action=save),
deleted at once.

Here a fake llama-server on an OS-assigned loopback port answers those endpoints with a character tokenizer (BOS + one
token per character), a lossy /slots text (no space before a quote or a comma, like llama.cpp's clean_spaces) and slot
files in llama.cpp's format, and records every request; a short-lived child process plays the server PID.
"""
from __future__ import annotations

import copy
import importlib.util
import json
import os
import stat
import struct
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest

REPO = Path(__file__).resolve().parents[1]
HELPER = REPO / "scripts/brain_warm_start.py"
TEMPLATE = REPO / "scripts/chat_templates/nemotron-3.5-bolt-openclaw.jinja"
MARK = "<!-- openclaw:attempt:DYNAMIC -->"
KEY = "warm-start-key-7c1"
BOS = 1

STABLE = ("<!-- openclaw:attempt:STABLE -->\nYou are a personal assistant running inside OpenClaw.\n## Tooling\n"
          "Tools policy-filtered.\n<!-- /openclaw:attempt:STABLE -->\n")
TOOLS = [{"type": "function", "function": {
    "name": "cascade__pick_and_place", "description": "Pick an object , e.g. 'red cube' , and place it.",
    "parameters": {"type": "object", "properties": {"object": {"type": "string", "description": "what"}},
                   "required": ["object"]}}},
    {"type": "function", "function": {"name": "cascade__describe_scene", "description": "Describe the table.",
                                      "parameters": {"type": "object", "properties": {}}}}]


def dynamic(session: str, date: str = "2026-10-18") -> str:
    return (f"{MARK}\n## Temporal Context\nCurrent date: {date}\nTime zone: UTC , local\n## Runtime\n"
            f"Runtime: agent=main | session=agent:main:dashboard:{session} | sessionId={session} | "
            "model=cascade/nemotron-3.5-bolt\n<!-- /openclaw:attempt:DYNAMIC -->")


#: What every session of the day shares after OpenClaw's boundary, up to its session ids (exact text).
SHARED = "\n\n" + dynamic("")[:dynamic("").index("dashboard:") + len("dashboard:")]


def openclaw_messages(session: str, order: str = "What do you see on the table?", date: str = "2026-10-18"):
    return [{"role": "system", "content": STABLE + dynamic(session, date)},
            {"role": "user", "content": f"[Sun {date} 08:16 UTC] {order}"}]


def render(**kw) -> str:
    """cascade's Bolt chat template, as llama.cpp renders an OpenClaw request (tests/test_bolt_chat_template.py)."""
    jinja2 = pytest.importorskip("jinja2")
    env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True)
    env.filters["tojson"] = lambda v, **_: json.dumps(v, ensure_ascii=False)
    return env.from_string(TEMPLATE.read_text()).render(add_generation_prompt=True, enable_thinking=False, **kw)


# A hand-rendered OpenClaw prompt in the template's shape, for the tests that need no jinja2. The space before the
# quote and before the comma is what llama.cpp's clean_spaces drops in /slots.
PREFIX = ("<|im_start|>system\n" + STABLE.rstrip() + "\n\n# Tools\n\nYou have access to the following functions:\n\n"
          "<tools>\n<function>\n<name>cascade__describe_scene</name>\n<description>Find the 'red cube' , then"
          "</description>\n</function>\n</tools>\n\n<IMPORTANT>\nReminder: call tools.\n</IMPORTANT>")


def rendered(session: str, order: str = "What do you see on the table?", prefix: str = PREFIX) -> str:
    return (prefix + "\n\n" + dynamic(session) + "<|im_end|>\n<|im_start|>user\n[Sun 2026-10-18 08:16 UTC] " + order
            + "<|im_end|>\n<|im_start|>assistant\n<think></think>")


PERCEPTION = "<|im_start|>user\n<__media__>List every object on the table.<|im_end|>\n<|im_start|>assistant\n"


MEDIA = "<__media__>"


def tokenize(text: str, media: bool = False) -> list[int]:
    """How the fake server tokenizes a prompt: BOS, then one token per character; with media, an image's marker
    becomes the placeholder tokens a slot holds for it (LLAMA_TOKEN_NULL)."""
    if media and MEDIA in text:
        head, tail = text.split(MEDIA, 1)
        return tokenize(head) + [-1] * 4 + [ord(c) for c in tail]
    return [BOS] + [ord(c) for c in text]


def detokenize(tokens) -> str:
    """llama.cpp's /detokenize: the plain concatenation of the pieces (BOS spelled '<s>')."""
    return "".join("<s>" if t == BOS else chr(t) for t in tokens if t >= 0)


def cleaned(text: str) -> str:
    """llama.cpp's clean_spaces as far as the fixtures need it (GET /slots): no space before a quote or a comma."""
    return text.replace(" '", "'").replace(" ,", ",")


def load_module():
    spec = importlib.util.spec_from_file_location("brain_warm_start", HELPER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def slot_file(tokens: list[int]) -> bytes:
    """A slot file as llama-server writes it: 'ggsq', version 4, the packed server tokens, then the state."""
    packed = [-1, 1, len(tokens), *tokens, 0]          # marker, version, tokens, no media keys
    return struct.pack("<III", 0x67677371, 4, len(packed)) + struct.pack(f"<{len(packed)}i", *packed) + b"\0" * 64


class FakeLlamaServer:
    """llama-server's /health, /completion, /slots, /slots/<id>?action=save and /detokenize; requests are recorded."""

    def __init__(self, save_dir: Path, *, unready=0, slots=(), completion_delay=0.0, hide_prompts=False,
                 save_status=200, unreadable=(0, 0), completion_gate=None):
        self.save_dir = save_dir
        self.unready = unready                  # /health answers 503 this many times first ("Loading model")
        self.slots = list(slots)                # each: {"id", "id_task", "text": the exact rendered prompt}
        self.completion_delay = completion_delay
        self.completion_gate = completion_gate  # a threading.Event: /completion answers only once it is set
        self.hide_prompts = hide_prompts        # llama-server without LLAMA_SERVER_SLOTS_DEBUG
        self.save_status = save_status
        self.unreadable = unreadable            # /detokenize fails for token counts strictly between these two
        self.requests: list[dict] = []
        self.lock = threading.Lock()
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def _reply(self, status, payload):
                body = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _record(self, body=None):
                with fake.lock:
                    fake.requests.append({"method": self.command, "path": self.path, "body": body,
                                          "auth": self.headers.get("Authorization"), "t": time.monotonic()})

            def _authorized(self):
                if self.headers.get("Authorization") == f"Bearer {KEY}":
                    return True
                self._reply(401, {"error": "Invalid API Key"})
                return False

            def do_GET(self):
                self._record()
                if self.path == "/health":
                    with fake.lock:
                        loading = fake.unready > 0
                        fake.unready -= 1 if loading else 0
                    self._reply(503 if loading else 200, {"status": "loading model" if loading else "ok"})
                elif self.path == "/slots" and self._authorized():
                    self._reply(200, [fake.slot_json(s) for s in fake.slots])

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                self._record(body)
                if not self._authorized():
                    return
                url = urlparse(self.path)
                if url.path == "/completion":
                    time.sleep(fake.completion_delay)
                    if fake.completion_gate is not None:
                        fake.completion_gate.wait(60)
                    one = {"content": ".", "timings": {"prompt_n": len(body["prompt"]), "cache_n": 0}}
                    self._reply(200, one if body.get("n_cmpl", 1) == 1 else [one] * body["n_cmpl"])
                elif url.path.startswith("/slots/") and parse_qs(url.query).get("action") == ["save"]:
                    slot = next(s for s in fake.slots if s["id"] == int(url.path.rsplit("/", 1)[1]))
                    if fake.save_status == 200:                # "file_text": the slot ran another request since
                        text = slot.get("file_text", slot["text"])
                        (fake.save_dir / body["filename"]).write_bytes(slot_file(tokenize(text, media=True)))
                    self._reply(fake.save_status, {"id_slot": slot["id"], "filename": body["filename"]})
                elif url.path == "/detokenize":
                    if any(tok < 0 for tok in body["tokens"]):  # no text for an image's placeholder tokens
                        self._reply(400, {"error": "invalid token"})
                    elif fake.unreadable[0] < len(body["tokens"]) < fake.unreadable[1]:
                        self._reply(500, {"error": "unreadable"})
                    else:
                        self._reply(200, {"content": detokenize(body["tokens"])})
                else:
                    self._reply(404, {})

        class Quiet(ThreadingHTTPServer):
            def handle_error(self, request, client_address):   # a client that gave up (the timeout test)
                pass

        self.httpd = Quiet(("127.0.0.1", 0), Handler)
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def slot_json(self, slot):
        out = {"id": slot["id"], "id_task": slot.get("id_task"), "is_processing": slot.get("busy", False)}
        if slot.get("text") is not None:
            out["n_prompt_tokens"] = slot.get("n_prompt_tokens", len(tokenize(slot["text"])))
            if not self.hide_prompts:                 # the task's prompt, special tokens spelled out, cleaned
                out["prompt"] = cleaned(detokenize(tokenize(slot["text"])))
        return out

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.httpd.shutdown()
        self.httpd.server_close()

    def seen(self, method, path):
        with self.lock:
            return [r for r in self.requests if r["method"] == method and r["path"].startswith(path)]


@pytest.fixture
def env(tmp_path):
    key = tmp_path / "brain.key"
    key.write_text(KEY + "\n")
    key.chmod(0o600)
    (tmp_path / "template.jinja").write_text("{# a chat template #}")
    (tmp_path / "model.gguf").write_bytes(b"GGUF model")
    (tmp_path / "slots").mkdir(mode=0o700)
    return tmp_path


@pytest.fixture
def run(env):
    """Start the helper against a fake server; stops the 'server' process and the helper at the end."""
    started = []

    def start(fake, *, extra=(), record=True, env_extra=None):
        server = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
        argv = [sys.executable, "-B", str(HELPER), "--port", str(fake.port), "--server-pid", str(server.pid),
                "--slots", "2", "--prefix-file", str(env / "state" / "prefix.json"),
                "--chat-template-file", str(env / "template.jinja"), "--model-file", str(env / "model.gguf"),
                "--api-key-file", str(env / "brain.key"), "--poll", "0.05", "--record-every", "0.1",
                "--min-free-gb", "0",                  # whatever the runner's disk holds (the check: Recorder tests)
                *(("--slot-save-dir", str(env / "slots")) if record else ()), *extra]
        helper = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                  env={**os.environ, **(env_extra or {})})
        started.append((server, helper))
        return {"server": server, "helper": helper, "argv": argv, "prefix_file": env / "state" / "prefix.json"}

    yield start
    for server, helper in started:
        server.kill()
        server.wait()
        try:
            helper.wait(timeout=10)
        except subprocess.TimeoutExpired:
            helper.kill()
            helper.wait()


def stop(proc) -> str:
    """End the 'server' process and return the helper's log once it has exited by itself."""
    proc["server"].kill()
    proc["server"].wait()
    out, err = proc["helper"].communicate(timeout=10)
    assert proc["helper"].returncode == 0, err
    return out + err


def wait_for(predicate, timeout=10.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(0.02)
    return False


def write_record(env: Path, tokens, extension=(), **override):
    m = load_module()
    record = {"template": m.template_id(str(env / "template.jinja")), "model": m.model_id(str(env / "model.gguf")),
              "text": cleaned(PREFIX), "tokens": tokens, "shared": cleaned(SHARED.lstrip()) if extension else "",
              "extension": list(extension), "last": "", **override}
    m.write_record(env / "state" / "prefix.json", record)


def ext(text: str) -> list[int]:
    """The fake server's tokens of text that follows the prefix (no BOS)."""
    return [ord(c) for c in text]


def record_of(proc) -> dict:
    try:
        return json.loads(proc["prefix_file"].read_text())
    except (OSError, ValueError):
        return {}


def test_waits_for_readiness_then_prefills_every_slot_with_the_recorded_tokens(run, env):
    write_record(env, tokenize(PREFIX), ext(SHARED))
    with FakeLlamaServer(env / "slots", unready=5) as fake:
        proc = run(fake)
        assert wait_for(lambda: len(fake.seen("POST", "/completion")) == 2)
        time.sleep(0.3)
        log = stop(proc)
    health = fake.seen("GET", "/health")
    prefix, shared = fake.seen("POST", "/completion")
    assert len(health) == 6 and max(r["t"] for r in health) < prefix["t"]  # 5 x 503 "loading", 200, then the warm-up
    # Raw token prompts (no chat, so no user turn), copied to both slots, one token generated: first the prefix before
    # OpenClaw's boundary, then what its sessions share after it (the first step's checkpoints stay for another day).
    assert prefix["body"] == {"prompt": tokenize(PREFIX), "n_predict": 1, "n_cmpl": 2, "cache_prompt": True,
                              "temperature": 0.0}
    assert shared["body"] == {**prefix["body"], "prompt": tokenize(PREFIX + SHARED)}
    assert prefix["auth"] == shared["auth"] == f"Bearer {KEY}"
    assert log.count("warmed 2 slot(s)") == 2 and KEY not in log and KEY not in " ".join(proc["argv"])


def test_without_a_record_nothing_is_prefilled_and_the_first_openclaw_request_is_recorded_exactly(run, env):
    first = render(messages=openclaw_messages("1e6f762d-0b80"), tools=TOOLS)
    with FakeLlamaServer(env / "slots", slots=[{"id": 0, "id_task": 7, "text": first},
                                               {"id": 1, "id_task": 3, "text": PERCEPTION}]) as fake:
        proc = run(fake)
        assert wait_for(lambda: proc["prefix_file"].exists())
        log = stop(proc)
    assert not fake.seen("POST", "/completion")
    assert "no recorded OpenClaw prefix" in log and "recorded the latest OpenClaw prefix" in log
    assert stat.S_IMODE(proc["prefix_file"].stat().st_mode) == 0o600
    assert list((env / "slots").iterdir()) == []                         # the slot file is gone
    tokens = record_of(proc)["tokens"]
    # Exactly the server's tokens of the system text before OpenClaw's boundary and of the whole tool block: not the
    # /slots text (clean_spaces dropped the space before 'red cube'), nothing per-session.
    assert tokens == tokenize(first[:first.index(MARK)].rstrip())
    assert detokenize(tokens).endswith("</IMPORTANT>") and " 'red cube' ," in detokenize(tokens)
    assert "1e6f762d" not in detokenize(tokens) and "2026-10-18" not in detokenize(tokens)
    # No drift: the next chat (other session ids, another day, another order) starts with exactly those tokens.
    later = render(messages=openclaw_messages("92798453-b3e7", "Reset the scene.", "2026-10-19"), tools=TOOLS)
    assert tokenize(later)[:len(tokens)] == tokens


def test_two_sessions_record_what_they_share_after_the_boundary_as_the_server_tokenized_it(run, env):
    # Rendered by cascade's template: what a new chat reuses from the previous one, up to the session ids. The /slots
    # text dropped the space in "UTC , local"; the record holds the server's own tokens.
    first = render(messages=openclaw_messages("1e6f762d-0b80"), tools=TOOLS)
    second = render(messages=openclaw_messages("92798453-b3e7", "Reset the scene."), tools=TOOLS)
    with FakeLlamaServer(env / "slots", slots=[{"id": 0, "id_task": 7, "text": first},
                                               {"id": 1, "id_task": 9, "text": second}]) as fake:
        proc = run(fake)
        assert wait_for(lambda: proc["prefix_file"].exists())
        time.sleep(0.5)
        log = stop(proc)
    record = record_of(proc)
    stable = first[:first.index(MARK)].rstrip()
    assert record["tokens"] == tokenize(stable)
    assert detokenize(record["extension"]) == SHARED and "UTC , local" in SHARED
    assert record["shared"] == cleaned(SHARED.lstrip())
    assert [r["path"] for r in fake.seen("POST", "/slots/")] == ["/slots/1?action=save"]    # one slot save
    assert "+ 0 its sessions share" not in log
    # No drift: a third session (another order) starts with exactly those tokens; its own ids come right after.
    third = render(messages=openclaw_messages("5a5a5a5a", "Wave."), tools=TOOLS)
    assert tokenize(third)[:len(record["tokens"]) + len(record["extension"])] == record["tokens"] + record["extension"]
    assert third[len(stable) + len(SHARED):].startswith("5a5a5a5a")


def test_shared_text_that_cannot_be_read_is_not_recorded(run, env):
    # /detokenize fails past the prefix: the prefix alone is recorded, its shared text left empty (a later poll
    # retries, a bounded number of times).
    first, second = rendered("1e6f762d"), rendered("92798453")
    with FakeLlamaServer(env / "slots", slots=[{"id": 0, "id_task": 7, "text": first},
                                               {"id": 1, "id_task": 9, "text": second}],
                         unreadable=(len(tokenize(PREFIX)), len(tokenize(second)))) as fake:
        proc = run(fake)
        assert wait_for(lambda: len(fake.seen("POST", "/slots/")) == 3)
        time.sleep(0.5)
        stop(proc)
    record = record_of(proc)
    assert record["tokens"] == tokenize(PREFIX) and record["extension"] == [] and record["shared"] == ""
    assert len(fake.seen("POST", "/slots/")) == 3


def test_session_ids_that_start_alike_are_not_shared(run, env):
    with FakeLlamaServer(env / "slots", slots=[{"id": 0, "id_task": 7, "text": rendered("1e6f762d")},
                                               {"id": 1, "id_task": 9, "text": rendered("1a2b3c4d")}]) as fake:
        proc = run(fake)
        assert wait_for(lambda: proc["prefix_file"].exists())
        stop(proc)
    assert detokenize(record_of(proc)["extension"]) == SHARED      # up to "dashboard:", not "dashboard:1"


def test_follow_ups_of_one_session_share_their_whole_system_turn_and_extend_nothing(run, env):
    with FakeLlamaServer(env / "slots", slots=[{"id": 0, "id_task": 7, "text": rendered("1e6f762d", "First.")},
                                               {"id": 1, "id_task": 9, "text": rendered("1e6f762d", "Then.")}]) as fake:
        proc = run(fake)
        assert wait_for(lambda: proc["prefix_file"].exists())
        time.sleep(0.5)
        stop(proc)
    assert record_of(proc)["tokens"] == tokenize(PREFIX) and record_of(proc)["extension"] == []


def test_the_session_of_the_previous_run_pairs_with_the_first_session_after_a_restart(run, env):
    with FakeLlamaServer(env / "slots", slots=[{"id": 0, "id_task": 2, "text": rendered("1e6f762d")}]) as fake:
        proc = run(fake)                                              # one session: the prefix alone
        assert wait_for(lambda: proc["prefix_file"].exists())
        stop(proc)
    assert record_of(proc)["extension"] == [] and "1e6f762d" in record_of(proc)["last"]
    with FakeLlamaServer(env / "slots", slots=[{"id": 0, "id_task": 3, "text": rendered("92798453")}]) as fake:
        proc = run(fake)                                              # the next start, then a new session
        assert wait_for(lambda: record_of(proc)["extension"])
        stop(proc)
    assert [r["body"]["prompt"] for r in fake.seen("POST", "/completion")] == [tokenize(PREFIX)]
    assert detokenize(record_of(proc)["extension"]) == SHARED
    with FakeLlamaServer(env / "slots") as fake:
        proc = run(fake)                                              # and the start after: both steps
        assert wait_for(lambda: len(fake.seen("POST", "/completion")) == 2)
        stop(proc)
    assert [r["body"]["prompt"] for r in fake.seen("POST", "/completion")] == [tokenize(PREFIX),
                                                                                 tokenize(PREFIX + SHARED)]


def test_a_new_day_records_what_its_sessions_share(run, env):
    write_record(env, tokenize(PREFIX), ext(SHARED))
    new_day = SHARED.replace("2026-10-18", "2026-10-19")
    slots = [{"id": 0, "id_task": 4, "text": rendered("1e6f762d").replace("2026-10-18", "2026-10-19")},
             {"id": 1, "id_task": 5, "text": rendered("92798453").replace("2026-10-18", "2026-10-19")}]
    with FakeLlamaServer(env / "slots", slots=slots) as fake:
        proc = run(fake)
        assert wait_for(lambda: detokenize(record_of(proc).get("extension", [])) == new_day)
        stop(proc)
    assert record_of(proc)["tokens"] == tokenize(PREFIX)


def test_a_slot_that_ran_another_request_before_it_was_saved_is_not_recorded(run, env):
    # GET /slots showed one OpenClaw request; by the time the slot is saved it holds another prompt.
    with FakeLlamaServer(env / "slots", slots=[{"id": 0, "id_task": 4, "text": rendered("aaaa"),
                                                "file_text": rendered("aaaa", prefix=PREFIX.replace("Find", "Seek"))}]) as fake:
        proc = run(fake)
        assert wait_for(lambda: len(fake.seen("POST", "/slots/")) == 3)
        time.sleep(0.3)
        log = stop(proc)
    assert not proc["prefix_file"].exists() and "could not read the tokens" in log


def test_a_lone_session_keeps_what_earlier_sessions_shared(run, env):
    # One session this run, nothing to compare it with: the record keeps the shared text of earlier sessions.
    write_record(env, tokenize(PREFIX), ext(SHARED))
    before = (env / "state" / "prefix.json").stat().st_mtime_ns
    with FakeLlamaServer(env / "slots", slots=[{"id": 0, "id_task": 4, "text": rendered("aaaa")}]) as fake:
        proc = run(fake)
        assert wait_for(lambda: len(fake.seen("GET", "/slots")) >= 4)
        stop(proc)
    assert not fake.seen("POST", "/slots/") and detokenize(record_of(proc)["extension"]) == SHARED
    assert proc["prefix_file"].stat().st_mtime_ns == before      # nothing changed: the record is not rewritten


def test_a_session_whose_system_turn_extends_another_shares_nothing_past_it(run, env):
    # Shared up to the end of one system turn: that includes its session ids, so it is not warmed.
    longer = rendered("aaaa").replace("<|im_end|>", "\nOne more directive.<|im_end|>", 1)
    with FakeLlamaServer(env / "slots", slots=[{"id": 0, "id_task": 4, "text": longer},
                                               {"id": 1, "id_task": 5, "text": rendered("aaaa", "Then.")}]) as fake:
        proc = run(fake)
        assert wait_for(lambda: proc["prefix_file"].exists())
        time.sleep(0.3)
        stop(proc)
    assert record_of(proc)["extension"] == []


def test_image_placeholder_tokens_never_reach_detokenize(run, env):
    # An image in the user turn: the slot file holds LLAMA_TOKEN_NULL for it; only the text before it is read.
    with_image = rendered("aaaa", f"{MEDIA}What is this?")
    with FakeLlamaServer(env / "slots", slots=[{"id": 0, "id_task": 4, "text": with_image}]) as fake:
        proc = run(fake)
        assert wait_for(lambda: proc["prefix_file"].exists())
        stop(proc)
    assert record_of(proc)["tokens"] == tokenize(PREFIX)
    assert all(tok >= 0 for r in fake.seen("POST", "/detokenize") for tok in r["body"]["tokens"])


def test_the_helper_never_goes_through_an_http_proxy(run, env):
    from owned_server import held_dead_port
    write_record(env, tokenize(PREFIX))
    with held_dead_port() as port, FakeLlamaServer(env / "slots") as fake:
        proxy = f"http://127.0.0.1:{port}"
        proc = run(fake, env_extra={"HTTP_PROXY": proxy, "http_proxy": proxy, "NO_PROXY": "", "no_proxy": ""})
        assert wait_for(lambda: fake.seen("POST", "/completion"))
        stop(proc)


def test_a_restart_prefills_what_the_previous_run_recorded(run, env):
    with FakeLlamaServer(env / "slots", slots=[{"id": 1, "id_task": 2, "text": rendered("1e6f762d")}]) as fake:
        proc = run(fake)
        assert wait_for(lambda: proc["prefix_file"].exists())
        stop(proc)
    with FakeLlamaServer(env / "slots", slots=[{"id": 0, "id_task": 9, "text": rendered("1e6f762d", "Then.")}]) as fake:
        proc = run(fake)                                              # the next start, then that session goes on
        assert wait_for(lambda: fake.seen("POST", "/completion"))
        time.sleep(0.5)
        stop(proc)
    assert fake.seen("POST", "/completion")[0]["body"]["prompt"] == tokenize(PREFIX)
    assert not fake.seen("POST", "/slots/")      # the same prefix, the same session: no slot is saved


def test_a_changed_prefix_replaces_the_record_once(run, env):
    write_record(env, tokenize(PREFIX))
    changed = PREFIX.replace("Reminder", "Reminder (new tools)")
    with FakeLlamaServer(env / "slots", slots=[{"id": 0, "id_task": 4, "text": rendered("ab", prefix=changed)}]) as fake:
        proc = run(fake)
        assert wait_for(lambda: record_of(proc).get("tokens") == tokenize(changed))
        time.sleep(0.5)
        stop(proc)
    assert len(fake.seen("POST", "/slots/0?action=save")) == 1


def test_alternating_openclaw_prefixes_cost_one_slot_save_each(run, env):
    # Two agents with their own tools: the record follows the latest one, and a prefix read before is not read again.
    other = PREFIX.replace("Reminder", "Reminder (sub-agent)")
    with FakeLlamaServer(env / "slots", slots=[{"id": 0, "id_task": 4, "text": rendered("aaaa")}]) as fake:
        proc = run(fake)
        assert wait_for(lambda: record_of(proc).get("tokens") == tokenize(PREFIX))
        for n, prefix in enumerate((other, PREFIX, other, PREFIX), start=5):   # each agent: one session
            fake.slots = [{"id": n % 2, "id_task": n, "text": rendered("b" if prefix == other else "aaaa", prefix=prefix)}]
            assert wait_for(lambda p=prefix: record_of(proc).get("tokens") == tokenize(p))
        log = stop(proc)
    assert len(fake.seen("POST", "/slots/")) == 2
    assert log.count("read before") == 3


@pytest.mark.parametrize("override", [{"template": "sha256:" + "0" * 64}, {"model": "other.gguf:1"}],
                         ids=["other-template", "other-model"])
def test_a_record_made_for_another_template_or_model_is_not_prefilled_and_is_replaced(run, env, override):
    write_record(env, tokenize(PREFIX.replace("Tools policy", "Old tools policy")), **override)
    with FakeLlamaServer(env / "slots", slots=[{"id": 0, "id_task": 2, "text": rendered("1e6f762d")}]) as fake:
        proc = run(fake)
        assert wait_for(lambda: record_of(proc).get("tokens") == tokenize(PREFIX))
        log = stop(proc)
    assert not fake.seen("POST", "/completion")
    assert "another chat template or model" in log
    m = load_module()
    assert record_of(proc)["template"] == m.template_id(str(env / "template.jinja"))
    assert record_of(proc)["model"] == m.model_id(str(env / "model.gguf"))


@pytest.mark.parametrize("content", [
    "{not json",
    json.dumps({"format": 1, "template": "x", "prefix": PREFIX}),        # a text record (format 1)
    json.dumps(["a list"]),
    "TOKENS:[]",
    "TOKENS:[1, -1, 2]",
    'TOKENS:[1, "a"]',
], ids=["not-json", "text-record", "not-an-object", "no-tokens", "media-marker", "not-ints"])
def test_an_unusable_record_is_never_prefilled(run, env, content):
    if content.startswith("TOKENS:"):
        write_record(env, json.loads(content[7:]))
    else:
        (env / "state").mkdir()
        (env / "state" / "prefix.json").write_text(content)
    with FakeLlamaServer(env / "slots") as fake:
        proc = run(fake)
        assert wait_for(lambda: len(fake.seen("GET", "/slots")) >= 2)
        stop(proc)
    assert not fake.seen("POST", "/completion")


def test_perception_prompts_and_the_warm_up_itself_are_never_recorded(run, env):
    with FakeLlamaServer(env / "slots", slots=[{"id": 0, "id_task": 9, "text": PREFIX},     # the warm-up's prompt
                                               {"id": 1, "id_task": 8, "text": PERCEPTION},
                                               {"id": 2}]) as fake:                       # never ran a task
        proc = run(fake)
        assert wait_for(lambda: len(fake.seen("GET", "/slots")) >= 3)
        stop(proc)
    assert not proc["prefix_file"].exists() and not fake.seen("POST", "/slots/")


def test_a_server_that_hides_prompts_records_nothing_and_says_why(run, env):
    # llama-server lists a slot's last prompt only with LLAMA_SERVER_SLOTS_DEBUG=1 (the serve script sets it).
    with FakeLlamaServer(env / "slots", slots=[{"id": 0, "id_task": 4, "text": rendered("x")}],
                         hide_prompts=True) as fake:
        proc = run(fake)
        assert wait_for(lambda: len(fake.seen("GET", "/slots")) >= 3)
        log = stop(proc)
    assert not proc["prefix_file"].exists()
    assert log.count("LLAMA_SERVER_SLOTS_DEBUG") == 1


def test_the_latest_openclaw_request_is_recorded(run, env):
    other = PREFIX.replace("cascade__describe_scene", "cascade__wave")
    with FakeLlamaServer(env / "slots", slots=[{"id": 0, "id_task": 4, "text": rendered("aaaa")},
                                               {"id": 1, "id_task": 11, "text": rendered("bbbb", prefix=other)}]) as fake:
        proc = run(fake)
        assert wait_for(lambda: proc["prefix_file"].exists())
        stop(proc)
    assert record_of(proc)["tokens"] == tokenize(other)
    assert [r["path"] for r in fake.seen("POST", "/slots/")] == ["/slots/1?action=save"]


def test_a_slot_is_saved_only_while_the_server_is_idle(run, env):
    # The save runs on llama-server's main loop (~0.1 s): never while a request is being served.
    busy = {"id": 1, "id_task": 5, "text": PERCEPTION, "busy": True}
    with FakeLlamaServer(env / "slots", slots=[{"id": 0, "id_task": 4, "text": rendered("aaaa")}, busy]) as fake:
        proc = run(fake)
        assert wait_for(lambda: len(fake.seen("GET", "/slots")) >= 4)
        assert not fake.seen("POST", "/slots/")
        busy["busy"] = False
        assert wait_for(lambda: proc["prefix_file"].exists())
        stop(proc)
    assert len(fake.seen("POST", "/slots/")) == 1


def test_a_long_slot_is_not_saved_to_record(run, env):
    # Saving pauses the server while it writes the slot: a changed prefix is read from a shorter slot later.
    with FakeLlamaServer(env / "slots", slots=[{"id": 0, "id_task": 4, "text": rendered("aaaa"),
                                                "n_prompt_tokens": 40000}]) as fake:
        proc = run(fake)
        assert wait_for(lambda: len(fake.seen("GET", "/slots")) >= 4)
        log = stop(proc)
    assert not fake.seen("POST", "/slots/") and not proc["prefix_file"].exists()
    assert log.count("40000-token slot") == 1


def test_a_failing_slot_save_is_retried_a_bounded_number_of_times(run, env):
    with FakeLlamaServer(env / "slots", slots=[{"id": 0, "id_task": 4, "text": rendered("aaaa")}],
                         save_status=500) as fake:
        proc = run(fake)
        assert wait_for(lambda: len(fake.seen("GET", "/slots")) >= 8)
        log = stop(proc)
    assert len(fake.seen("POST", "/slots/")) == load_module().CAPTURE_TRIES == 3
    assert "could not read the tokens" in log and not proc["prefix_file"].exists()


def test_without_a_slot_save_dir_nothing_is_recorded(run, env):
    with FakeLlamaServer(env / "slots", slots=[{"id": 0, "id_task": 4, "text": rendered("aaaa")}]) as fake:
        proc = run(fake, record=False)
        assert wait_for(lambda: fake.seen("GET", "/health"))
        time.sleep(0.5)
        stop(proc)
    assert not fake.seen("GET", "/slots") and not proc["prefix_file"].exists()


def test_a_capture_file_left_by_a_dead_helper_is_removed(run, env):
    stale = env / "slots" / "brain-warm-capture-4242.bin"
    stale.write_bytes(b"x" * 1024)
    keep = env / "slots" / "other.bin"
    keep.write_bytes(b"y")
    with FakeLlamaServer(env / "slots") as fake:
        proc = run(fake)
        assert wait_for(lambda: not stale.exists())
        stop(proc)
    assert keep.exists()


def test_a_warm_up_that_outlives_its_bound_is_abandoned_and_recording_goes_on(run, env):
    # The warm-up never answers while the test runs (no wall-clock window to race on a busy runner): recording the
    # newer prefix proves the helper gave the request up at its bound.
    write_record(env, tokenize(PREFIX))
    newer = PREFIX.replace("Tools policy-filtered.", "Tools policy-filtered!")
    gate = threading.Event()
    with FakeLlamaServer(env / "slots", slots=[{"id": 0, "id_task": 5, "text": rendered("cccc", prefix=newer)}],
                         completion_gate=gate) as fake:
        try:
            proc = run(fake, extra=("--timeout", "0.5"))
            assert wait_for(lambda: record_of(proc).get("tokens") == tokenize(newer), timeout=30)
            assert not gate.is_set()
            log = stop(proc)
        finally:
            gate.set()
    assert len(fake.seen("POST", "/completion")) == 1 and "gave up the warm-up" in log and "(timed out)" in log


def test_a_slot_save_that_would_fill_the_disk_waits(run, env):
    with FakeLlamaServer(env / "slots", slots=[{"id": 0, "id_task": 4, "text": rendered("aaaa")}]) as fake:
        proc = run(fake, extra=("--min-free-gb", "1e9"))                     # more than any disk holds
        assert wait_for(lambda: len(fake.seen("GET", "/slots")) >= 4)
        log = stop(proc)
    assert not fake.seen("POST", "/slots/") and not proc["prefix_file"].exists()
    assert log.count("GiB free in") == 1 and "recorded once there is room" in log


def recorder(env, fake, **kw):
    """The helper's recorder (step 3) in this process, against the fake server: one step() per read of GET /slots."""
    m = load_module()
    identity = {"template": m.template_id(str(env / "template.jinja")), "model": m.model_id(str(env / "model.gguf"))}
    record, _ = m.read_record(env / "state" / "prefix.json", **identity)
    return m, m.Recorder(m.Server("127.0.0.1", fake.port, KEY), prefix_file=env / "state" / "prefix.json",
                         slot_save_dir=env / "slots", identity=identity, record=record, **kw)


def test_the_recorder_runs_without_a_subprocess_one_read_per_step(env, capsys):
    now = [100.0]
    with FakeLlamaServer(env / "slots", slots=[{"id": 0, "id_task": 4, "text": rendered("aaaa")}]) as fake:
        m, rec = recorder(env, fake, record_every=10.0, clock=lambda: now[0], free_bytes=lambda path: 2**40)
        assert rec.due()
        rec.step()
        assert not rec.due()
        now[0] += 10.0
        assert rec.due()
        rec.step()
    assert len(fake.seen("GET", "/slots")) == 2 and len(fake.seen("POST", "/slots/")) == 1
    assert json.loads((env / "state" / "prefix.json").read_text())["tokens"] == tokenize(PREFIX)
    assert capsys.readouterr().err.count("recorded the latest OpenClaw prefix") == 1


def test_slot_save_tries_are_kept_for_the_latest_prefixes_only(env):
    with FakeLlamaServer(env / "slots", save_status=500) as fake:             # every save fails
        m, rec = recorder(env, fake, free_bytes=lambda path: 2**40)
        prefixes = [PREFIX.replace("Reminder", f"Reminder {i}") for i in range(m.KNOWN_PREFIXES + 4)]
        for n, prefix in enumerate(prefixes):
            fake.slots = [{"id": 0, "id_task": n, "text": rendered("aaaa", prefix=prefix)}]
            for _ in range(m.CAPTURE_TRIES + 2):
                rec.step()
            assert len(rec.tries) <= m.KNOWN_PREFIXES
    assert len(fake.seen("POST", "/slots/")) == len(prefixes) * m.CAPTURE_TRIES   # each prefix: its bounded tries
    kept = [text for text, _ in rec.tries]
    assert len(kept) == m.KNOWN_PREFIXES and f"Reminder {len(prefixes) - 1}" in kept[-1]
    assert not [text for text in kept if "Reminder 0" in text]                   # the oldest prefixes are dropped


def test_a_slot_is_saved_only_while_the_disk_keeps_room(env, capsys):
    free = [2**20]
    with FakeLlamaServer(env / "slots", slots=[{"id": 0, "id_task": 4, "text": rendered("aaaa")}]) as fake:
        m, rec = recorder(env, fake, min_free_bytes=2**30, free_bytes=lambda path: free[0])
        for _ in range(m.CAPTURE_TRIES + 2):                                  # low disk costs no try
            rec.step()
        assert not fake.seen("POST", "/slots/")
        assert capsys.readouterr().err.count("0.0 GiB free in") == 1           # logged once, not every read
        free[0] = 2**31
        rec.step()
    assert len(fake.seen("POST", "/slots/")) == 1 and (env / "state" / "prefix.json").exists()


def test_min_free_gb_is_the_free_space_a_save_needs_before_it_starts(env, capsys):  # #403 review, Minor 3
    """The check reads the free space BEFORE the save (exactly --min-free-gb free: it saves), and the save then writes
    its slot file on top: the help and the low-disk log say "before", never that the save "leaves" that much."""
    m = load_module()
    with pytest.raises(SystemExit):
        m.main(["--help"])
    usage = " ".join(capsys.readouterr().out.split())
    option = usage[usage.rindex("--min-free-gb MIN_FREE_GB"):]   # its entry under options, not the usage line
    option = option[:option.index("--poll") if "--poll" in option else len(option)]
    assert "free space --slot-save-dir needs before a slot save starts, else it waits" in option, option
    assert "leaves" not in option
    free = [2**20]
    with FakeLlamaServer(env / "slots", slots=[{"id": 0, "id_task": 4, "text": rendered("aaaa")}]) as fake:
        m, rec = recorder(env, fake, min_free_bytes=2**30, free_bytes=lambda path: free[0])
        rec.step()
        log = capsys.readouterr().err
        assert "under the 1.0 GiB a slot save needs before it starts" in log and "leaves" not in log, log
        free[0] = 2**30
        rec.step()
    assert len(fake.seen("POST", "/slots/")) == 1


def test_free_space_that_cannot_be_read_counts_as_low(env):
    def unreadable(path):
        raise OSError("no such device")

    with FakeLlamaServer(env / "slots", slots=[{"id": 0, "id_task": 4, "text": rendered("aaaa")}]) as fake:
        m, rec = recorder(env, fake, free_bytes=unreadable)
        rec.step()
    assert not fake.seen("POST", "/slots/")


def test_a_capture_file_written_late_is_removed_at_the_next_read(env):
    with FakeLlamaServer(env / "slots") as fake:
        m, rec = recorder(env, fake, free_bytes=lambda path: 2**40)
        rec.step()
        late = env / "slots" / "brain-warm-capture-31337.bin"               # a save the helper gave up on, written late
        late.write_bytes(b"x" * 1024)
        keep = env / "slots" / "other.bin"
        keep.write_bytes(b"y")
        rec.step()
    assert not late.exists() and keep.exists()


def test_task_zero_is_the_servers_first_task_not_a_missing_one():
    m = load_module()
    slots = [{"id": 0, "id_task": 0, "prompt": rendered("aaaa")}, {"id": 1, "prompt": rendered("bbbb")}]
    assert [s["id"] for s in m.openclaw_slots(slots)] == [1, 0]            # the slot that ran task 0 is the latest


def test_a_failed_record_write_keeps_the_old_record_and_leaves_no_temporary_file(tmp_path):
    m = load_module()
    target = tmp_path / "state" / "prefix.json"
    m.write_record(target, {"tokens": [1, 2]})
    with pytest.raises(TypeError):
        m.write_record(target, {"tokens": [3], "unwritable": object()})
    assert json.loads(target.read_text())["tokens"] == [1, 2]
    assert [p.name for p in target.parent.iterdir()] == ["prefix.json"]


def test_the_helper_ends_with_the_server_process_even_an_unreaped_one(tmp_path):
    # The serve script execs llama-server under the PID the helper watches; when it dies before its parent reaps it,
    # it is a zombie, and the helper must still end (it holds the script's stdout/stderr).
    server = subprocess.Popen([sys.executable, "-c", "pass"])
    try:
        assert wait_for(lambda: Path(f"/proc/{server.pid}/stat").read_text().split(")")[-1].split()[0] == "Z")
        helper = subprocess.run([sys.executable, "-B", str(HELPER), "--port", "9", "--server-pid", str(server.pid),
                                 "--slots", "2", "--prefix-file", str(tmp_path / "p.json"), "--poll", "0.05"],
                                check=False, capture_output=True, text=True, timeout=10)
    finally:
        server.wait()
    assert helper.returncode == 0, helper.stderr
    assert not (tmp_path / "p.json").exists()


def test_slot_file_tokens_reads_the_head_of_llamacpps_slot_file(tmp_path):
    m = load_module()
    path = tmp_path / "slot.bin"
    path.write_bytes(slot_file([BOS, 60, 61, -1, -1, 62]))
    assert m.slot_file_tokens(path) == [BOS, 60, 61, -1, -1, 62]
    path.write_bytes(b"ggsq" + b"\0" * 8)
    assert m.slot_file_tokens(path) is None
    unpacked = list(range(5, 45))                     # a plain token list, as an older llama.cpp wrote it
    path.write_bytes(struct.pack("<III", 0x67677371, 4, len(unpacked)) + struct.pack(f"<{len(unpacked)}i", *unpacked))
    assert m.slot_file_tokens(path) is None
    for packed in ([-1, 2, 3, 70, 71, 72, 0],          # a server-tokens layout of another version
                   [5, 1, 3, 70, 71, 72, 0]):          # plain tokens that happen to read like a header
        path.write_bytes(struct.pack("<III", 0x67677371, 4, len(packed)) + struct.pack(f"<{len(packed)}i", *packed))
        assert m.slot_file_tokens(path) is None
    assert m.slot_file_tokens(tmp_path / "missing.bin") is None


def test_openclaw_prefix_extraction():
    m = load_module()
    assert m.openclaw_prefix(rendered("x")) == PREFIX
    assert m.openclaw_prefix("<s>" + rendered("x")) == PREFIX          # a BOS the detokenizer spells out
    assert m.openclaw_prefix(PREFIX) is None                            # no boundary: not an OpenClaw request
    assert m.openclaw_prefix(PERCEPTION) is None
    assert m.openclaw_prefix("<|im_start|>user\nhi<|im_end|>\n" + rendered("x")) is None   # not the first turn
    assert m.openclaw_prefix(None) is None
    # The GGUF's own template keeps the per-session text before the tools: only the system text is shared.
    gguf = "<|im_start|>system\n" + STABLE + dynamic("x") + "\n\n# Tools\n<tools></tools><|im_end|>\n"
    assert m.openclaw_prefix(gguf) == "<|im_start|>system\n" + STABLE.rstrip()


def test_identities_name_the_template_and_model_the_server_runs(tmp_path):
    m = load_module()
    a, b = tmp_path / "a.jinja", tmp_path / "b.jinja"
    a.write_text("{{ messages }}")
    b.write_text("{{ messages }}!")
    assert m.template_id(str(a)) != m.template_id(str(b)) and m.template_id(str(a)).startswith("sha256:")
    assert m.template_id("") == "gguf"                                  # CHAT_TEMPLATE_FILE='' : the GGUF's own
    (tmp_path / "x.gguf").write_bytes(b"12345")
    assert m.model_id(str(tmp_path / "x.gguf")) == "x.gguf:5"


def test_records_are_private(tmp_path):
    m = load_module()
    target = tmp_path / "deep" / "state" / "prefix.json"
    m.write_record(target, {"template": "t", "model": "m", "text": "x", "tokens": [1, 2]})
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    assert json.loads(target.read_text()) == {"format": m.FORMAT, "template": "t", "model": "m", "text": "x",
                                              "tokens": [1, 2]}
    assert not [p for p in target.parent.iterdir() if p != target]      # no temporary file left behind


def test_the_real_template_keeps_tools_inside_the_recorded_prefix():
    # Guard on the template the serve script ships: the per-session text still comes AFTER the tool block, so the
    # recorded prefix carries the tools (the ~11k tokens worth prefilling). tests/test_bolt_chat_template.py owns
    # the template's other properties.
    first = render(messages=copy.deepcopy(openclaw_messages("1e6f762d")), tools=TOOLS)
    prefix = load_module().openclaw_prefix(first)
    assert prefix is not None and "<tools>" in prefix and "</tools>" in prefix
    assert os.path.commonprefix([first, render(messages=openclaw_messages("ffff"), tools=TOOLS)]).startswith(prefix)
