"""scripts/brain_warm_start.py: the first chat after a Bolt brain start reuses OpenClaw's prefix, hermetically.

The helper runs beside llama-server (scripts/serve_nemotron_bolt_llamacpp.sh starts it). Once the server answers
/health it prefills the last recorded OpenClaw prefix TOKENS (stable system text + tool block, as the server itself
tokenized them) into every slot with ONE raw /completion (n_cmpl = slots). While the server runs it notices a changed
OpenClaw prefix in GET /slots (text detokenized with clean_spaces, so unusable as a prompt) and reads the exact tokens
from a slot file (POST /slots/<id>?action=save), deleted at once.

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
    return (f"{MARK}\n## Temporal Context\nCurrent date: {date}\n## Runtime\nRuntime: agent=main | "
            f"session=agent:main:dashboard:{session} | sessionId={session} | model=cascade/nemotron-3.5-bolt\n"
            "<!-- /openclaw:attempt:DYNAMIC -->")


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


def tokenize(text: str) -> list[int]:
    """How the fake server tokenizes a prompt: BOS, then one token per character."""
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
                 save_status=200):
        self.save_dir = save_dir
        self.unready = unready                  # /health answers 503 this many times first ("Loading model")
        self.slots = list(slots)                # each: {"id", "id_task", "text": the exact rendered prompt}
        self.completion_delay = completion_delay
        self.hide_prompts = hide_prompts        # llama-server without LLAMA_SERVER_SLOTS_DEBUG
        self.save_status = save_status
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
                    one = {"content": ".", "timings": {"prompt_n": len(body["prompt"]), "cache_n": 0}}
                    self._reply(200, one if body.get("n_cmpl", 1) == 1 else [one] * body["n_cmpl"])
                elif url.path.startswith("/slots/") and parse_qs(url.query).get("action") == ["save"]:
                    slot = next(s for s in fake.slots if s["id"] == int(url.path.rsplit("/", 1)[1]))
                    if fake.save_status == 200:
                        (fake.save_dir / body["filename"]).write_bytes(slot_file(tokenize(slot["text"])))
                    self._reply(fake.save_status, {"id_slot": slot["id"], "filename": body["filename"]})
                elif url.path == "/detokenize":
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
        out = {"id": slot["id"], "id_task": slot.get("id_task"), "is_processing": False}
        if slot.get("text") is not None:
            out["n_prompt_tokens"] = slot.get("n_prompt_tokens", len(tokenize(slot["text"])))
            if not self.hide_prompts:
                out["prompt"] = cleaned(slot["text"])
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

    def start(fake, *, extra=(), record=True):
        server = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
        argv = [sys.executable, "-B", str(HELPER), "--port", str(fake.port), "--server-pid", str(server.pid),
                "--slots", "2", "--prefix-file", str(env / "state" / "prefix.json"),
                "--chat-template-file", str(env / "template.jinja"), "--model-file", str(env / "model.gguf"),
                "--api-key-file", str(env / "brain.key"), "--poll", "0.05", "--record-every", "0.1",
                *(("--slot-save-dir", str(env / "slots")) if record else ()), *extra]
        helper = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
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


def write_record(env: Path, tokens, **override):
    m = load_module()
    record = {"template": m.template_id(str(env / "template.jinja")), "model": m.model_id(str(env / "model.gguf")),
              "text": cleaned(PREFIX), "tokens": tokens, **override}
    m.write_record(env / "state" / "prefix.json", record)


def record_of(proc) -> dict:
    try:
        return json.loads(proc["prefix_file"].read_text())
    except (OSError, ValueError):
        return {}


def test_waits_for_readiness_then_prefills_every_slot_once_with_the_recorded_tokens(run, env):
    write_record(env, tokenize(PREFIX))
    with FakeLlamaServer(env / "slots", unready=5) as fake:
        proc = run(fake)
        assert wait_for(lambda: fake.seen("POST", "/completion"))
        time.sleep(0.3)
        log = stop(proc)
    health = fake.seen("GET", "/health")
    (warm,) = fake.seen("POST", "/completion")
    assert len(health) == 6 and max(r["t"] for r in health) < warm["t"]   # 5 x 503 "loading", 200, then the warm-up
    # One raw token prompt (no chat, so no user turn), copied to both slots, one token generated.
    assert warm["body"] == {"prompt": tokenize(PREFIX), "n_predict": 1, "n_cmpl": 2, "cache_prompt": True,
                            "temperature": 0.0}
    assert warm["auth"] == f"Bearer {KEY}"
    assert "warmed 2 slot(s)" in log and KEY not in log and KEY not in " ".join(proc["argv"])


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


def test_a_restart_prefills_what_the_previous_run_recorded(run, env):
    with FakeLlamaServer(env / "slots", slots=[{"id": 1, "id_task": 2, "text": rendered("1e6f762d")}]) as fake:
        proc = run(fake)
        assert wait_for(lambda: proc["prefix_file"].exists())
        stop(proc)
    with FakeLlamaServer(env / "slots", slots=[{"id": 0, "id_task": 9, "text": rendered("2f00aa11")}]) as fake:
        proc = run(fake)                                              # the next start, then a chat
        assert wait_for(lambda: fake.seen("POST", "/completion"))
        time.sleep(0.5)
        stop(proc)
    assert fake.seen("POST", "/completion")[0]["body"]["prompt"] == tokenize(PREFIX)
    assert not fake.seen("POST", "/slots/")      # the same prefix again: no slot is saved


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
        for n, prefix in enumerate((other, PREFIX, other, PREFIX), start=5):
            fake.slots = [{"id": n % 2, "id_task": n, "text": rendered(f"s{n}", prefix=prefix)}]
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
    write_record(env, tokenize(PREFIX))
    newer = PREFIX.replace("Tools policy-filtered.", "Tools policy-filtered!")
    with FakeLlamaServer(env / "slots", slots=[{"id": 0, "id_task": 5, "text": rendered("cccc", prefix=newer)}],
                         completion_delay=3.0) as fake:
        proc = run(fake, extra=("--timeout", "0.5"))
        t0 = time.monotonic()
        assert wait_for(lambda: record_of(proc).get("tokens") == tokenize(newer))
        assert time.monotonic() - t0 < 2.5                                     # did not wait the 3 s out
        log = stop(proc)
    assert len(fake.seen("POST", "/completion")) == 1 and "gave up" in log


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
    path.write_bytes(struct.pack("<III", 0x67677371, 4, 6) + struct.pack("<6i", 5, 6, 7, 8, 9, 10))  # unpacked tokens
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
