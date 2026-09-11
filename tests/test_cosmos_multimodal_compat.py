"""CPU-only integration at the *installed* Cosmos/vLLM processor boundary.

Run with COSMOS_TEST_PYTHON=<isolated .cosmos/bin/python> and
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


def patch_fixture(tmp_path):
    """Inert upstream stand-in to test writes/rollback, never an inference result."""
    import hashlib
    from types import ModuleType

    path = tmp_path / "upstream.py"
    source = "class Owner(object):\n    pass\n"
    path.write_text(source)
    module = ModuleType("upstream")
    module.__file__ = str(path)
    exec(source, module.__dict__)
    args = dict(module=module, owner_name="Owner", method_name="count",
                method_source="    def count(self):\n        return 7\n",
                expected_sha256=hashlib.sha256(source.encode()).hexdigest(),
                version_ok=True, allow_patch=True,
                smoke=lambda: {"count": module.Owner().count()})
    return helper(), path, source, args


@pytest.mark.parametrize("mode", ["read-only", "version", "source"])
def test_multimodal_guard_refuses_unknown_or_readonly_writes(tmp_path, mode):
    h, path, source, args = patch_fixture(tmp_path)
    if mode == "read-only":
        args["allow_patch"] = False
        error = "--setup-only"
    elif mode == "version":
        args["version_ok"] = False
        error = "Unrecognized"
    else:
        path.write_text(source + "# unknown upstream edit\n")
        source = path.read_text()
        error = "Unrecognized"
    with pytest.raises(RuntimeError, match=error):
        h._install_compat_method(**args)
    assert path.read_text() == source
    assert "count" not in args["module"].Owner.__dict__


def test_multimodal_guard_is_atomic_preserves_uv_hardlinks_and_is_idempotent(tmp_path):
    h, path, source, args = patch_fixture(tmp_path)
    linked = tmp_path / "wheel-cache.py"
    os.link(path, linked)
    result = h._install_compat_method(**args)
    assert result["status"] == "patched" and result["count"] == 7
    assert linked.read_text() == source
    before = path.stat()
    args["version_ok"] = False  # future working upstream must not be rewritten
    assert h._install_compat_method(**args)["status"] == "upstream-ok"
    assert path.stat().st_mtime_ns == before.st_mtime_ns
    assert path.stat().st_ino == before.st_ino


def test_failed_candidate_rolls_back_method_without_publishing(tmp_path):
    h, path, source, args = patch_fixture(tmp_path)
    args["method_source"] = '    def count(self):\n        raise RuntimeError("candidate failed")\n'
    with pytest.raises(RuntimeError, match="candidate failed"):
        h._install_compat_method(**args)
    assert path.read_text() == source
    assert "count" not in args["module"].Owner.__dict__
    assert not list(tmp_path.glob(".cosmos-mm-*"))


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
    installed_probe('''
import os
import xgrammar as xgr
from transformers import AutoTokenizer
from vllm.tool_parsers.qwen3_engine_tool_parser import Qwen3EngineToolParser
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
p = AutoTokenizer.from_pretrained(os.environ["COSMOS_TEST_EXPORT"], local_files_only=True)
r = ChatCompletionRequest(model="cosmos3-edge", messages=[], tool_choice="auto", tools=[
    {"type":"function", "function":{"name":"cascade__report_color", "parameters":{
        "type":"object", "properties":{"color":{"type":"string"}, "observed":{"type":"boolean"}},
        "required":["color","observed"]}}}])
parser = Qwen3EngineToolParser(p, tools=r.tools)
tag = parser.get_structural_tag(r)
assert tag is not None, "auto tools bypass native schema grammar"
compiler = xgr.GrammarCompiler(xgr.TokenizerInfo.from_huggingface(p))
compiled = compiler.compile_structural_tag(tag)
for prefix, suffix in [("", ""), ("<tool_call>\\n", "\\n</tool_call>")]:
    good = prefix + "<function=cascade__report_color>\\n<parameter=color>red</parameter>\\n<parameter=observed>true</parameter>\\n</function>" + suffix
    bad = prefix + "<function=cascade__report_color>\\n<parameter=color>red</parameter>\\n</function>" + suffix
    assert xgr.GrammarMatcher(compiled).accept_string(good), good
    assert not xgr.GrammarMatcher(compiled).accept_string(bad), bad
assert r.tools[0].function.strict is None, "grammar must not mutate caller tool definitions"
assert Qwen3EngineToolParser(p,tools=r.tools).get_structural_tag(r.model_copy(update={"model":"different-model"})) is None
print("native grammar admits complete calls and rejects missing required fields")
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
native = ModelConfig(model=path, model_impl="vllm", max_model_len=32768)
assert type(native.hf_config).__module__.startswith("vllm.")
assert native.hf_config.text_config.num_hidden_layers == 2 * reference.text_config.num_hidden_layers
custom = ModelConfig(model=path, model_impl="transformers", max_model_len=32768,
                     hf_overrides={"text_config": {"hidden_act": "gelu"}})
assert type(custom.hf_config) is Cosmos3EdgeConfig
assert custom.hf_config.text_config.hidden_act == "gelu"
assert custom.hf_config.text_config.num_hidden_layers == reference.text_config.num_hidden_layers
print("exact dense HF config, native-backend isolation, overrides and meta-model passed")
''')


def test_installed_video_only_and_mixed_patch_counts_match_real_preprocessing():
    installed_probe('''
import os
import torch
from transformers import AutoProcessor
p = AutoProcessor.from_pretrained(os.environ["COSMOS_TEST_EXPORT"], local_files_only=True)
for shape, options in [((4,63,97), {}), ((1,32,32), {"do_resize": False}),
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
    assert p.video_processor.get_number_of_video_patches(*shape, options) == patches
print("video-only, mixed modalities, resize override, no-resize: pixel-count parity passed")
''')

