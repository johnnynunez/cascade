"""scripts/brain_warm_start.py: the first chat after a Bolt brain start reuses OpenClaw's prefix, hermetically.

The helper runs beside llama-server (scripts/serve_nemotron_bolt_llamacpp.sh starts it). Once the server answers
/health it prefills the last recorded OpenClaw prefix (stable system text + tool block, as the server itself rendered
it) into every slot with ONE raw /completion (n_cmpl = slots); while the server runs it re-reads that prefix from the
server's own rendering of the latest OpenClaw request (GET /slots). Here a fake llama-server on an OS-assigned
loopback port answers those endpoints and records every request; a short-lived child process plays the server PID.
"""
from __future__ import annotations

import copy
import importlib.util
import json
import os
import stat
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
HELPER = REPO / "scripts/brain_warm_start.py"
TEMPLATE = REPO / "scripts/chat_templates/nemotron-3.5-bolt-openclaw.jinja"
MARK = "<!-- openclaw:attempt:DYNAMIC -->"
KEY = "warm-start-key-7c1"

STABLE = ("<!-- openclaw:attempt:STABLE -->\nYou are a personal assistant running inside OpenClaw.\n## Tooling\n"
          "Tools policy-filtered.\n<!-- /openclaw:attempt:STABLE -->\n")
TOOLS = [{"type": "function", "function": {
    "name": "cascade__pick_and_place", "description": "Pick an object and place it.",
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


# A hand-rendered OpenClaw prompt in the template's shape, for the tests that need no jinja2.
PREFIX = ("<|im_start|>system\n" + STABLE.rstrip() + "\n\n# Tools\n\nYou have access to the following functions:\n\n"
          "<tools>\n<function>\n<name>cascade__describe_scene</name>\n</function>\n</tools>\n\n<IMPORTANT>\n"
          "Reminder: call tools.\n</IMPORTANT>")


def rendered(session: str, order: str = "What do you see on the table?") -> str:
    return (PREFIX + "\n\n" + dynamic(session) + "<|im_end|>\n<|im_start|>user\n[Sun 2026-10-18 08:16 UTC] " + order
            + "<|im_end|>\n<|im_start|>assistant\n<think></think>")


PERCEPTION = "<|im_start|>user\n<__media__>List every object on the table.<|im_end|>\n<|im_start|>assistant\n"


def load_module():
    spec = importlib.util.spec_from_file_location("brain_warm_start", HELPER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeLlamaServer:
    """llama-server's /health, /completion and /slots on 127.0.0.1:<OS port>; every request is recorded."""

    def __init__(self, *, unready=0, slots=(), completion_delay=0.0):
        self.unready = unready            # /health answers 503 this many times first ("Loading model")
        self.slots = list(slots)          # GET /slots payload
        self.completion_delay = completion_delay
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

            def do_GET(self):
                self._record()
                if self.path == "/health":
                    with fake.lock:
                        loading = fake.unready > 0
                        fake.unready -= 1 if loading else 0
                    self._reply(503 if loading else 200, {"status": "loading model" if loading else "ok"})
                elif self.path == "/slots":
                    if self.headers.get("Authorization") != f"Bearer {KEY}":
                        self._reply(401, {"error": "Invalid API Key"})
                    else:
                        self._reply(200, fake.slots)
                else:
                    self._reply(404, {})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                self._record(body)
                if self.path != "/completion" or self.headers.get("Authorization") != f"Bearer {KEY}":
                    self._reply(401, {"error": "Invalid API Key"})
                    return
                time.sleep(fake.completion_delay)
                n = body.get("n_cmpl", 1)
                one = {"content": ".", "timings": {"prompt_n": 4321, "prompt_ms": 4321.0, "cache_n": 0}}
                self._reply(200, one if n == 1 else [one] * n)

        class Quiet(ThreadingHTTPServer):
            def handle_error(self, request, client_address):   # a client that gave up (the timeout test)
                pass

        self.httpd = Quiet(("127.0.0.1", 0), Handler)
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.httpd.shutdown()
        self.httpd.server_close()

    def seen(self, method, path):
        with self.lock:
            return [r for r in self.requests if r["method"] == method and r["path"] == path]


@pytest.fixture
def run(tmp_path):
    """Start the helper against a fake server; yields a starter and stops server process + helper at the end."""
    key = tmp_path / "brain.key"
    key.write_text(KEY + "\n")
    key.chmod(0o600)
    template = tmp_path / "template.jinja"
    template.write_text("{# a chat template #}")
    started = []

    def start(fake, *, prefix_file=None, extra=(), template_file=template):
        server = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
        prefix_file = prefix_file or tmp_path / "state" / "prefix.json"
        argv = [sys.executable, "-B", str(HELPER), "--port", str(fake.port), "--server-pid", str(server.pid),
                "--slots", "2", "--prefix-file", str(prefix_file), "--chat-template-file", str(template_file),
                "--api-key-file", str(key), "--poll", "0.05", "--record-every", "0.1", *extra]
        helper = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        started.append((server, helper))
        return {"server": server, "helper": helper, "argv": argv, "prefix_file": prefix_file}

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


def write_record(path: Path, prefix: str, template_file: Path, **override):
    module = load_module()
    record = {"format": module.FORMAT, "template": module.template_id(str(template_file)), "prefix": prefix, **override}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record))


def test_waits_for_readiness_then_prefills_every_slot_once_with_the_recorded_prefix(run, tmp_path):
    with FakeLlamaServer(unready=5) as fake:
        prefix_file = tmp_path / "state" / "prefix.json"
        write_record(prefix_file, PREFIX, tmp_path / "template.jinja")
        proc = run(fake, prefix_file=prefix_file)
        assert wait_for(lambda: fake.seen("POST", "/completion"))
        time.sleep(0.3)
        log = stop(proc)
    health = fake.seen("GET", "/health")
    (warm,) = fake.seen("POST", "/completion")
    ready_at = max(r["t"] for r in health)
    assert len(health) == 6 and ready_at < warm["t"]          # 5 x 503 "loading", then 200, then the warm-up
    # One raw prompt (no chat, so no user turn), copied to both slots, one token generated.
    assert warm["body"] == {"prompt": PREFIX, "n_predict": 1, "n_cmpl": 2, "cache_prompt": True, "temperature": 0.0}
    assert warm["auth"] == f"Bearer {KEY}"
    assert "warmed 2 slot(s)" in log and KEY not in log and KEY not in " ".join(proc["argv"])


def test_without_a_record_nothing_is_prefilled_and_the_first_openclaw_request_is_recorded(run, tmp_path):
    first = render(messages=openclaw_messages("1e6f762d-0b80"), tools=TOOLS)
    with FakeLlamaServer(slots=[{"id": 0, "id_task": 7, "prompt": first},
                                {"id": 1, "id_task": 3, "prompt": PERCEPTION}]) as fake:
        proc = run(fake)
        assert wait_for(lambda: proc["prefix_file"].exists())
        log = stop(proc)
    assert not fake.seen("POST", "/completion")
    assert "no recorded OpenClaw prefix" in log
    record = json.loads(proc["prefix_file"].read_text())
    assert stat.S_IMODE(proc["prefix_file"].stat().st_mode) == 0o600
    prefix = record["prefix"]
    # The system text before OpenClaw's boundary and the whole tool block, as the server rendered them; nothing of
    # the per-session text (date, session ids) or of the conversation.
    assert prefix.startswith("<|im_start|>system\n" + STABLE.rstrip()) and prefix.endswith("</IMPORTANT>")
    assert "cascade__describe_scene" in prefix and "1e6f762d" not in prefix and "2026-10-18" not in prefix
    assert prefix == first[:first.index(MARK)].rstrip()
    # No drift: the next chat (other session ids, another day, another order) renders that exact prefix first.
    later = render(messages=openclaw_messages("92798453-b3e7", "Reset the scene.", "2026-10-19"), tools=TOOLS)
    assert later.startswith(prefix) and later[len(prefix):].startswith("\n\n" + MARK)


def test_a_restart_prefills_what_the_previous_run_recorded(run, tmp_path):
    with FakeLlamaServer(slots=[{"id": 0, "id_task": 2, "prompt": rendered("1e6f762d")}]) as fake:
        proc = run(fake)
        assert wait_for(lambda: proc["prefix_file"].exists())
        stop(proc)
    with FakeLlamaServer() as fake:                                   # the next brain start
        proc = run(fake, prefix_file=proc["prefix_file"])
        assert wait_for(lambda: fake.seen("POST", "/completion"))
        stop(proc)
    assert fake.seen("POST", "/completion")[0]["body"]["prompt"] == PREFIX


def test_a_record_made_with_another_chat_template_is_not_prefilled_and_is_replaced(run, tmp_path):
    with FakeLlamaServer(slots=[{"id": 0, "id_task": 2, "prompt": rendered("1e6f762d")}]) as fake:
        prefix_file = tmp_path / "state" / "prefix.json"
        write_record(prefix_file, PREFIX.replace("Tools policy", "Old tools policy"), tmp_path / "template.jinja",
                     template="sha256:" + "0" * 64)
        proc = run(fake, prefix_file=prefix_file)
        assert wait_for(lambda: json.loads(prefix_file.read_text())["prefix"] == PREFIX)
        log = stop(proc)
    assert not fake.seen("POST", "/completion")
    assert "another chat template" in log
    assert json.loads(prefix_file.read_text())["template"] == load_module().template_id(str(tmp_path / "template.jinja"))


@pytest.mark.parametrize("record", [
    "{not json",
    json.dumps({"format": 99, "template": "x", "prefix": PREFIX}),
    json.dumps(["a list"]),
], ids=["not-json", "unknown-format", "not-an-object"])
def test_an_unreadable_record_is_never_prefilled(run, tmp_path, record):
    prefix_file = tmp_path / "state" / "prefix.json"
    prefix_file.parent.mkdir(parents=True)
    prefix_file.write_text(record)
    with FakeLlamaServer() as fake:
        proc = run(fake, prefix_file=prefix_file)
        assert wait_for(lambda: len(fake.seen("GET", "/slots")) >= 2)
        stop(proc)
    assert not fake.seen("POST", "/completion")


@pytest.mark.parametrize("prefix", [
    "<|im_start|>user\nhello",                                        # not a system turn
    PREFIX + "\n\n" + MARK + "\n## Runtime\nsession=abc",               # carries per-session text
    "",
], ids=["not-system", "per-session-text", "empty"])
def test_a_record_that_is_not_a_stable_system_prefix_is_never_prefilled(run, tmp_path, prefix):
    prefix_file = tmp_path / "state" / "prefix.json"
    write_record(prefix_file, prefix, tmp_path / "template.jinja")
    with FakeLlamaServer() as fake:
        proc = run(fake, prefix_file=prefix_file)
        assert wait_for(lambda: len(fake.seen("GET", "/slots")) >= 2)
        stop(proc)
    assert not fake.seen("POST", "/completion")


def test_perception_prompts_and_the_warm_up_itself_are_never_recorded(run, tmp_path):
    with FakeLlamaServer(slots=[{"id": 0, "id_task": 9, "prompt": PREFIX},       # the warm-up's own prompt
                                {"id": 1, "id_task": 8, "prompt": PERCEPTION},
                                {"id": 2, "id_task": 1}]) as fake:              # a slot that never ran a task
        proc = run(fake)
        assert wait_for(lambda: len(fake.seen("GET", "/slots")) >= 3)
        stop(proc)
    assert not proc["prefix_file"].exists()


def test_the_latest_openclaw_request_is_recorded(run, tmp_path):
    other = PREFIX.replace("cascade__describe_scene", "cascade__wave")
    with FakeLlamaServer(slots=[{"id": 0, "id_task": 4, "prompt": rendered("aaaa")},
                                {"id": 1, "id_task": 11, "prompt": rendered("bbbb").replace(PREFIX, other)}]) as fake:
        proc = run(fake)
        assert wait_for(lambda: proc["prefix_file"].exists())
        stop(proc)
    assert json.loads(proc["prefix_file"].read_text())["prefix"] == other


def test_a_warm_up_that_outlives_its_bound_is_abandoned_and_recording_goes_on(run, tmp_path):
    prefix_file = tmp_path / "state" / "prefix.json"
    write_record(prefix_file, PREFIX, tmp_path / "template.jinja")
    newer = PREFIX.replace("Tools policy-filtered.", "Tools policy-filtered!")
    with FakeLlamaServer(slots=[{"id": 0, "id_task": 5, "prompt": rendered("cccc").replace(PREFIX, newer)}],
                         completion_delay=3.0) as fake:
        proc = run(fake, prefix_file=prefix_file, extra=("--timeout", "0.5"))
        t0 = time.monotonic()
        assert wait_for(lambda: json.loads(prefix_file.read_text())["prefix"] == newer)
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
                                 "--slots", "2", "--prefix-file", str(tmp_path / "p.json"), "--chat-template-file", "",
                                 "--poll", "0.05"], check=False, capture_output=True, text=True, timeout=10)
    finally:
        server.wait()
    assert helper.returncode == 0, helper.stderr
    assert not (tmp_path / "p.json").exists()


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


def test_template_id_names_the_chat_template_the_server_renders_with(tmp_path):
    m = load_module()
    a, b = tmp_path / "a.jinja", tmp_path / "b.jinja"
    a.write_text("{{ messages }}")
    b.write_text("{{ messages }}!")
    assert m.template_id(str(a)) != m.template_id(str(b)) and m.template_id(str(a)).startswith("sha256:")
    assert m.template_id("") == "gguf"                                  # CHAT_TEMPLATE_FILE='' : the GGUF's own


def test_records_are_private(tmp_path):
    m = load_module()
    target = tmp_path / "deep" / "state" / "prefix.json"
    m.write_record(target, "sha256:x", PREFIX)
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    assert json.loads(target.read_text()) == {"format": m.FORMAT, "template": "sha256:x", "prefix": PREFIX}
    assert not [p for p in target.parent.iterdir() if p != target]      # no temporary file left behind


def test_the_real_template_keeps_tools_inside_the_recorded_prefix():
    # Guard on the template the serve script ships: the per-session text still comes AFTER the tool block, so the
    # recorded prefix carries the tools (the ~11k tokens worth prefilling). tests/test_bolt_chat_template.py owns
    # the template's other properties.
    first = render(messages=copy.deepcopy(openclaw_messages("1e6f762d")), tools=TOOLS)
    prefix = load_module().openclaw_prefix(first)
    assert prefix is not None and "<tools>" in prefix and "</tools>" in prefix
    assert os.path.commonprefix([first, render(messages=openclaw_messages("ffff"), tools=TOOLS)]).startswith(prefix)
