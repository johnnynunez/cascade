"""CPU-only integration at the *installed* Cosmos/vLLM processor boundary.

Run with COSMOS_TEST_PYTHON=<isolated .cosmos-vllm-0.32.0-cu130/bin/python> and
COSMOS_TEST_EXPORT=<validated local export>. Never downloads model weights.
The ordinary app test suite can run the guard tests without vLLM/CUDA.
"""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess

import pytest

REPO = Path(__file__).resolve().parents[1]


def helper():
    spec = importlib.util.spec_from_file_location("cosmos_serving", REPO / "scripts/cosmos_serving.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


FIXTURE_SOURCE = 'REGISTRY = {\n    "keep": "a",\n    "vendored": "b",\n}\n'
FIXTURE_EDIT = ('    "vendored": "b",\n', "")


def compat_fixture(tmp_path, *, results=None, proof_ok=True, after_ok=True, known=True):
    """Inert stand-in for a vLLM wheel file and the child-process smokes.

    Never an inference result: the runner returns canned verdicts and records
    the overlay files each stage would load.
    """
    import hashlib

    site = tmp_path / "site"
    target = site / "pkg/registry.py"
    target.parent.mkdir(parents=True)
    target.write_text(FIXTURE_SOURCE)
    patched = FIXTURE_SOURCE.replace(*FIXTURE_EDIT)
    patch = {"name": "fixture", "upstream": "https://example.invalid/pr/1", "smoke": "registry",
             "files": ({"module": "pkg.registry", "path": "pkg/registry.py",
                        "sha256": hashlib.sha256(FIXTURE_SOURCE.encode()).hexdigest(),
                        "patched_sha256": hashlib.sha256(patched.encode()).hexdigest(),
                        "edits": (FIXTURE_EDIT,)},)}
    stages = []

    def runner(overlay):
        stages.append({name: Path(file).read_text() for name, file in overlay.items()})
        fixed = '"vendored"' not in (stages[-1].get("pkg.registry") or target.read_text())
        ok = {0: fixed, 1: proof_ok, 2: after_ok}.get(len(stages) - 1, True) if results is None else results
        failure = {"ok": False, "error": "AttributeError: fixture", "known": known}
        return {"rope": {"ok": True, "result": {}}, "registry": {"ok": True, "result": {}} if ok else failure}

    kwargs = dict(runner=runner, locate=lambda relative: site / relative, versions_ok=True, patches=(patch,))
    return helper(), target, patched, stages, kwargs


@pytest.mark.parametrize("mode", ["read-only", "version", "source", "unexpected"])
def test_compat_guard_refuses_unknown_or_readonly_writes(tmp_path, mode):
    h, target, _, stages, kwargs = compat_fixture(tmp_path, known=mode != "unexpected")
    kwargs["allow_patch"] = mode != "read-only"
    error = {"read-only": "--setup-only", "version": "Unrecognized", "source": "Unrecognized",
             "unexpected": "Unexpected"}[mode]
    if mode == "version":
        kwargs["versions_ok"] = False
    if mode == "source":
        target.write_text(FIXTURE_SOURCE + "# unknown upstream edit\n")
    before = target.read_text()
    with pytest.raises(RuntimeError, match=error):
        h.ensure_compat(tmp_path / "export", **kwargs)
    assert target.read_text() == before
    assert len(stages) == 1, "no candidate is built for an unrecognized failure"


def test_compat_patch_is_proven_atomic_preserves_uv_hardlinks_and_is_idempotent(tmp_path):
    h, target, patched, stages, kwargs = compat_fixture(tmp_path)
    linked = tmp_path / "wheel-cache.py"
    os.link(target, linked)
    report = h.ensure_compat(tmp_path / "export", allow_patch=True, **kwargs)
    assert report["patches"]["fixture"]["status"] == "patched"
    assert report["smokes"]["registry"]["ok"]
    # 1: installed bytes fail; 2: the exact candidate bytes pass in a fresh
    # process (overlay); 3: the installed file passes after the inode swap.
    assert stages == [{}, {"pkg.registry": patched}, {}]
    assert target.read_text() == patched
    assert linked.read_text() == FIXTURE_SOURCE, "shared wheel bytes were written in place"
    before = target.stat()
    stages.clear()
    again = h.ensure_compat(tmp_path / "export", allow_patch=True, **kwargs)
    assert again["patches"]["fixture"]["status"] == "patched-earlier"
    assert stages == [{}]
    assert (target.stat().st_ino, target.stat().st_mtime_ns) == (before.st_ino, before.st_mtime_ns)


def test_upstream_fixed_behaviour_is_success_without_any_write(tmp_path):
    h, target, _, stages, kwargs = compat_fixture(tmp_path, results=True)
    kwargs["versions_ok"] = False  # a future working release is never rewritten
    report = h.ensure_compat(tmp_path / "export", allow_patch=True, **kwargs)
    assert report["patches"]["fixture"]["status"] == "upstream-ok"
    assert target.read_text() == FIXTURE_SOURCE and len(stages) == 1


def test_failed_candidate_is_never_published(tmp_path):
    h, target, _, stages, kwargs = compat_fixture(tmp_path, proof_ok=False)
    with pytest.raises(RuntimeError, match="nothing written"):
        h.ensure_compat(tmp_path / "export", allow_patch=True, **kwargs)
    assert target.read_text() == FIXTURE_SOURCE
    assert len(stages) == 2
    assert not list(target.parent.glob(".cosmos-compat-*"))


def test_installed_patch_failing_its_reproof_is_rolled_back(tmp_path):
    h, target, _, _, kwargs = compat_fixture(tmp_path, after_ok=False)
    with pytest.raises(RuntimeError, match="Installed compatibility patches failed"):
        h.ensure_compat(tmp_path / "export", allow_patch=True, **kwargs)
    assert target.read_text() == FIXTURE_SOURCE


def test_a_failing_smoke_without_a_patch_is_refused(tmp_path):
    h, target, _, _, kwargs = compat_fixture(tmp_path, results=True)
    runner = kwargs["runner"]

    def broken_rope(overlay):
        result = runner(overlay)
        result["rope"] = {"ok": False, "error": "IndexError: fixture", "known": False}
        return result

    kwargs["runner"] = broken_rope
    with pytest.raises(RuntimeError, match="no known patch.*rope"):
        h.ensure_compat(tmp_path / "export", allow_patch=True, **kwargs)
    assert target.read_text() == FIXTURE_SOURCE


def test_overlay_loads_candidate_bytes_for_modules_and_packages(tmp_path, monkeypatch):
    """The proof imports the candidate file under the real module name, and a
    package keeps its original __path__ for its submodules."""
    import sys

    root = tmp_path / "site"
    (root / "ovlpkg/sub").mkdir(parents=True)
    (root / "ovlpkg/__init__.py").write_text("")
    (root / "ovlpkg/mod.py").write_text("VALUE = 'installed'\n")
    (root / "ovlpkg/sub/__init__.py").write_text("VALUE = 'installed'\n")
    (root / "ovlpkg/sub/leaf.py").write_text("VALUE = 'leaf'\n")
    (tmp_path / "mod.py").write_text("VALUE = 'candidate'\n")
    (tmp_path / "sub.py").write_text("VALUE = 'candidate package'\n")
    monkeypatch.syspath_prepend(str(root))
    h = helper()
    finder = h._Overlay({"ovlpkg.mod": str(tmp_path / "mod.py"), "ovlpkg.sub": str(tmp_path / "sub.py")})
    monkeypatch.setattr(sys, "meta_path", [finder, *sys.meta_path])
    for name in ("ovlpkg", "ovlpkg.mod", "ovlpkg.sub", "ovlpkg.sub.leaf"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    import ovlpkg.mod
    import ovlpkg.sub.leaf

    assert ovlpkg.mod.VALUE == "candidate"
    assert sys.modules["ovlpkg.sub"].VALUE == "candidate package"
    assert ovlpkg.sub.leaf.VALUE == "leaf"
    for name in ("ovlpkg", "ovlpkg.mod", "ovlpkg.sub", "ovlpkg.sub.leaf"):
        sys.modules.pop(name, None)


def test_smokes_report_through_a_fresh_interpreter(tmp_path):
    """The child protocol itself; in an app venv without the Cosmos stack the
    smoke fails, and that verdict (not a crash) reaches the parent."""
    import sys

    result = helper().run_smokes(tmp_path, python=sys.executable, names=("video",))
    assert set(result) == {"video"}
    assert isinstance(result["video"]["ok"], bool)
    if not result["video"]["ok"]:
        assert result["video"]["known"] is False


def installed_probe(code):
    python = os.environ.get("COSMOS_TEST_PYTHON")
    export = os.environ.get("COSMOS_TEST_EXPORT")
    if not python or not export:
        pytest.skip("requires explicit isolated Cosmos interpreter and offline export")
    result = subprocess.run(
        [python, "-B", "-c", code],
        env=dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1"),
        cwd=REPO, capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


def test_unknown_chat_template_is_refused():
    with pytest.raises(RuntimeError, match="Unrecognized"):
        helper().serving_chat_template("# different upstream template")


def test_serving_template_keeps_multimodal_and_history_contract():
    h = helper()
    assert hasattr(h, "serving_chat_template"), "native tool schema must be rendered unambiguously"
    installed_probe('''
import os, runpy, json
from pathlib import Path
from transformers import AutoTokenizer
h = runpy.run_path("scripts/cosmos_serving.py")
path = Path(os.environ["COSMOS_TEST_EXPORT"])
source = (path / "chat_template.jinja").read_text()
candidate = h["serving_chat_template"](source)
t = AutoTokenizer.from_pretrained(path, local_files_only=True)
no_tools = [{"role":"user","content":[{"type":"image"},{"type":"text","text":"Describe."}]}]
kw = dict(tokenize=False, add_generation_prompt=True, enable_thinking=False)
assert t.apply_chat_template(no_tools,chat_template=candidate,**kw) == t.apply_chat_template(no_tools,chat_template=source,**kw)
tools = [{"type":"function","function":{"name":"cascade__inspect","parameters":{"type":"object","properties":{"enabled":{"type":"boolean"}},"required":["enabled"]}}}]
rendered = t.apply_chat_template(no_tools, tools=tools, chat_template=candidate, **kw)
assert json.loads(rendered.split("<tools>",1)[1].split("</tools>",1)[0]) == tools
assert "<parameter=example_parameter_1>" in rendered
assert "<|vision_start|><|image_pad|><|vision_end|>" in rendered
history = [{"role":"assistant","content":"", "tool_calls":[{"type":"function","function":{"name":"cascade__inspect","arguments":{"enabled":True}}}]}, {"role":"tool","content":"{\\"ok\\":true}"}]
assert t.apply_chat_template(history,chat_template=candidate,**kw) == t.apply_chat_template(history,chat_template=source,**kw)
import tempfile
with tempfile.TemporaryDirectory() as tmp:
    try:
        h["ensure_chat_template"](path, tmp)
    except RuntimeError as e:
        assert "--setup-only" in str(e)
    else:
        raise AssertionError("read-only check created a missing template")
    saved = h["ensure_chat_template"](path, tmp, allow_write=True)
    target = Path(saved["path"])
    assert target.read_text() == candidate
    modified = target.stat().st_mtime_ns
    assert h["ensure_chat_template"](path, tmp)["status"] == "reused"
    assert target.stat().st_mtime_ns == modified
assert (path / "chat_template.jinja").read_text() == source
print("JSON tool schema; image/history unchanged; read-only and idempotent template")
''')


def test_installed_native_grammar_covers_bare_functions_and_required_fields():
    # The server's own parser stack at the served strict level (vLLM 0.32.0
    # --tool-strict-level), not the parser class in isolation.
    installed_probe('''
import os, runpy
import xgrammar as xgr
from transformers import AutoTokenizer
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.parser.parser_manager import ParserManager
h = runpy.run_path("scripts/cosmos_serving.py")
p = AutoTokenizer.from_pretrained(os.environ["COSMOS_TEST_EXPORT"], local_files_only=True)
tools = [{"type":"function", "function":{"name":"cascade__report_color", "parameters":{
    "type":"object", "properties":{"color":{"type":"string"}, "observed":{"type":"boolean"}},
    "required":["color","observed"]}}}]
def tag(level):
    cls = ParserManager.get_parser(tool_parser_name=h["TOOL_PARSER"], reasoning_parser_name=h["REASONING_PARSER"],
                                   enable_auto_tools=True, tool_strict_level=level)
    request = ChatCompletionRequest(model="cosmos3-edge", messages=[], tool_choice="auto", tools=tools)
    adjusted = cls(p, tools=tools).adjust_request(request)
    assert request.tools[0].function.strict is None, "grammar must not mutate caller tool definitions"
    return getattr(adjusted.structured_outputs, "structural_tag", None)
assert tag("auto") is None, "vLLM 0.32.0: non-strict auto tools carry no grammar by design"
compiled = xgr.GrammarCompiler(xgr.TokenizerInfo.from_huggingface(p)).compile_structural_tag(tag(h["TOOL_STRICT_LEVEL"]))
for prefix, suffix in [("", ""), ("<tool_call>\\n", "\\n</tool_call>")]:
    good = prefix + "<function=cascade__report_color>\\n<parameter=color>red</parameter>\\n<parameter=observed>true</parameter>\\n</function>" + suffix
    bad = prefix + "<function=cascade__report_color>\\n<parameter=color>red</parameter>\\n</function>" + suffix
    assert xgr.GrammarMatcher(compiled).accept_string(good), good
    assert not xgr.GrammarMatcher(compiled).accept_string(bad), bad
assert xgr.GrammarMatcher(compiled).accept_string("The cube is red."), "auto must still allow prose"
print("served grammar admits complete calls, wrapped or bare, and rejects missing required fields")
''')


def test_installed_wheel_files_carry_exactly_the_proven_patches():
    installed_probe('''
import hashlib, importlib.metadata, runpy
from pathlib import Path
h = runpy.run_path("scripts/cosmos_serving.py")
for patch in h["COMPAT_PATCHES"]:
    for spec in patch["files"]:
        path = Path(str(importlib.metadata.distribution("vllm").locate_file(spec["path"])))
        assert hashlib.sha256(path.read_bytes()).hexdigest() == spec["patched_sha256"], (patch["name"], spec["path"])
print("installed vLLM files hold exactly the backports of vllm#60823 and vllm#62368")
''')


def test_installed_transformers_backend_image_budget_matches_official_pixels():
    # This is the failing startup seam, not AutoProcessor alone (which passes).
    installed_probe('''
import os
import torch
from transformers import AutoProcessor
from vllm.config import ModelConfig
from vllm.multimodal.processing import InputProcessingContext
from vllm.model_executor.models.transformers.multimodal import MultiModalProcessingInfo
path = os.environ["COSMOS_TEST_EXPORT"]
config = ModelConfig(model=path, model_impl="transformers", max_model_len=32768)
reference = AutoProcessor.from_pretrained(path, local_files_only=True)
info = MultiModalProcessingInfo(InputProcessingContext(config, reference.tokenizer))
p = info.get_hf_processor()
counts = p._get_num_multimodal_tokens(image_sizes=[(193, 317)])
image = torch.arange(3*193*317).reshape(3,193,317).remainder(256).to(torch.uint8)
actual = p.image_processor(image, return_tensors="pt")
expected = reference.image_processor(image, return_tensors="pt")
assert torch.equal(actual["image_grid_thw"], expected["image_grid_thw"])
assert torch.equal(actual["pixel_values"], expected["pixel_values"]), "row-major patches silently corrupt HF block-major vision"
assert counts["num_image_patches"] == [actual["pixel_values"].shape[0]]
assert counts["num_image_tokens"] == [int(actual["image_grid_thw"].prod()) // 4]
assert info.get_max_image_tokens() > 0
print("official image budget and block-major pixel parity passed")
''')


def test_installed_vllm_model_config_constructs_exact_dense_hf_architecture():
    installed_probe('''
import os
import torch
from transformers import AutoModel
from transformers.models.cosmos3_edge.configuration_cosmos3_edge import Cosmos3EdgeConfig
from vllm.config import ModelConfig
path = os.environ["COSMOS_TEST_EXPORT"]
c = ModelConfig(model=path, model_impl="transformers", max_model_len=32768)
reference = Cosmos3EdgeConfig.from_pretrained(path, local_files_only=True)
with torch.device("meta"):
    model = AutoModel.from_config(c.hf_config)
assert type(c.hf_config) is Cosmos3EdgeConfig
assert c.hf_config.text_config.num_hidden_layers == reference.text_config.num_hidden_layers
assert c.hf_config.text_config.hidden_act == reference.text_config.hidden_act
assert type(model.get_decoder()).__name__ == "Cosmos3EdgeTextModel"
assert len(model.get_decoder().layers) == reference.text_config.num_hidden_layers
assert all(p.device.type == "meta" for p in model.parameters())
assert "MoE" not in c._architecture, c._architecture
from vllm.transformers_utils import config as registry
assert "cosmos3_edge" not in registry._CONFIG_REGISTRY, "vLLM's Nemotron-H-backed config must not shadow HF's"
custom = ModelConfig(model=path, model_impl="transformers", max_model_len=32768,
                     hf_overrides={"text_config": {"hidden_act": "gelu"}})
assert type(custom.hf_config) is Cosmos3EdgeConfig
assert custom.hf_config.text_config.hidden_act == "gelu"
assert custom.hf_config.text_config.num_hidden_layers == reference.text_config.num_hidden_layers
print("exact dense HF config, overrides and meta-model passed")
''')


def test_installed_video_only_and_mixed_patch_counts_match_real_preprocessing():
    installed_probe('''
import os
import torch
from transformers import AutoProcessor
p = AutoProcessor.from_pretrained(os.environ["COSMOS_TEST_EXPORT"], local_files_only=True)
# No do_resize=False case: every Qwen2-VL-family counter ignores it; CASCADE
# always serves with resizing on.
for shape, options in [((4,63,97), {}),
                        ((3,128,256), {"size": {"shortest_edge":1024,"longest_edge":12288}, "merge_size":1})]:
    frames, height, width = shape
    video = torch.arange(frames*3*height*width).reshape(frames,3,height,width).remainder(256).to(torch.uint8)
    actual = p.video_processor(video, return_tensors="pt", do_sample_frames=False, **options)
    patches = int(actual["video_grid_thw"].prod())
    assert patches == actual["pixel_values_videos"].shape[0]
    expected = [patches // options.get("merge_size", p.video_processor.merge_size)**2]
    for images in (None, [(320,640)]):
        counts = p._get_num_multimodal_tokens(image_sizes=images, video_sizes=[shape], videos_kwargs=options)
        assert counts["num_video_tokens"] == expected, (shape,options,counts,expected)
    # transformers >= 5.18 (PR #48900) counts through the video processor.
    assert p.video_processor.get_num_of_video_patches(*shape, options) == patches
print("video-only, mixed modalities, resize override: pixel-count parity passed")
''')

