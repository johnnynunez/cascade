"""`detector.type: brain` from the config, through `apps.demo._make_detector`.

The brain (the local vision LLM) names and boxes the objects; a box-prompted
SAM (`BoxSegmenter`) masks them. The factory reads the endpoint, model and key
file from the detector block, builds the segmenter from the SAME `segmenter:`
strings as the click `PointSegmenter` (one shared SAM), warms SAM in the
background, and never imports openai or ultralytics while doing so.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np
import pytest

from cascade.apps import demo
from cascade.config import load_demo_config
from cascade.perception.brain_detector import BrainDetector
from cascade.perception.detector import MockDetector
from cascade.perception.segmenter import BoxSegmenter, PointSegmenter
from cascade.types import Frame

REPO = Path(__file__).resolve().parents[1]
PERCEPTION_VARS = ("CASCADE_PERCEPTION_URL", "CASCADE_PERCEPTION_MODEL", "CASCADE_PERCEPTION_KEY_FILE")


@pytest.fixture
def env(monkeypatch, tmp_path):
    """Defaults only: no perception variables, a private HOME, no overlays."""
    for name in (*PERCEPTION_VARS, "CASCADE_BOOTH", "CASCADE_ROBOT", "CASCADE_BASE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    return monkeypatch


@pytest.fixture
def no_warm_up(monkeypatch):
    """Fail loudly if a test that must not warm SAM starts the warm-up."""
    calls = []
    monkeypatch.setattr(demo, "_warm_box_segmenter", lambda seg: calls.append(seg), raising=False)
    return calls


@pytest.fixture
def key_file(env, tmp_path):
    path = tmp_path / "brain.key"
    path.write_text("sk-test-0123\n")
    env.setenv("CASCADE_PERCEPTION_KEY_FILE", str(path))
    return path


def brain_cfg(**detector):
    """The demo config on the Isaac camera (no detector pinned by the camera),
    with the detector block replaced when `detector` is given."""
    cfg = load_demo_config(camera="isaac")
    assert "detector" not in cfg._data["camera"]
    if detector:
        cfg._data["detector"] = dict(detector)
    return cfg


# -- type: brain from configs/demo.yaml -------------------------------------


#: Ruling 24: both repo Qwen serve scripts default to PORT=8080, so the
#: perception default asks that port; the rig's OpenShell instance (:8081)
#: sets CASCADE_PERCEPTION_URL itself (scripts/openshell_rig.serve_mcp_env).
DEFAULT_PERCEPTION_URL = "http://127.0.0.1:8080/v1"
#: Ruling 25: SAM by absolute path under the repo, whatever the server's CWD.
SHIPPED_SAM = str(REPO / "models" / "sam2.1_l.pt")


def test_demo_yaml_builds_the_brain_detector_with_sam_l(key_file):
    det = demo._make_detector(brain_cfg(), warm_up=False)
    assert type(det) is BrainDetector
    assert det._base_url == DEFAULT_PERCEPTION_URL
    assert det._model == "qwen3.8-27b"
    assert det._api_key == "sk-test-0123"
    assert (det._timeout_s, det._max_tokens) == (20.0, 1500)
    assert (det._refresh_s, det._max_cache_age_s, det._scene_change) == (4.0, 30.0, 6.0)
    assert type(det._segmenter) is BoxSegmenter
    assert (det._segmenter._model_path, det._segmenter._device_spec) == (SHIPPED_SAM, "auto")
    assert det._segmenter._model is None, "SAM must load lazily (or in the warm-up thread)"


def test_one_perception_default_on_the_qwen_serve_scripts_port(env):
    """The demo.yaml default, the factory's default for a bare block and the
    PORT both Qwen serve scripts start on are one endpoint (Ruling 24)."""
    assert load_demo_config(camera="isaac").detector.base_url == DEFAULT_PERCEPTION_URL
    assert demo.BRAIN_BASE_URL == DEFAULT_PERCEPTION_URL
    for script in ("serve_qwen_llamacpp.sh", "serve_qwen_vllm.sh"):
        ports = re.findall(r'^PORT="\$\{PORT:-(\d+)\}"$', (REPO / "scripts" / script).read_text(), re.M)
        assert ports == ["8080"], script


def test_the_perception_variables_reach_the_detector(key_file, env):
    env.setenv("CASCADE_PERCEPTION_URL", "http://10.1.2.3:8000/v1")
    env.setenv("CASCADE_PERCEPTION_MODEL", "brain-x")
    det = demo._make_detector(brain_cfg(), warm_up=False)
    assert (det._base_url, det._model) == ("http://10.1.2.3:8000/v1", "brain-x")


def test_building_the_brain_detector_imports_neither_openai_nor_ultralytics(tmp_path):
    """A fresh interpreter: the CI minimal install has neither package, and
    the factory must not need them before the first detect."""
    key = tmp_path / "k"
    key.write_text("k")
    code = (
        "import sys\n"
        "from cascade.config import load_demo_config\n"
        "from cascade.apps import demo\n"
        "det = demo._make_detector(load_demo_config(camera='isaac'), warm_up=False)\n"
        "print(type(det).__name__, type(det._segmenter).__name__,\n"
        "      [m for m in ('openai', 'ultralytics') if m in sys.modules])\n"
    )
    env = {k: v for k, v in os.environ.items()
           if k not in (*PERCEPTION_VARS, "CASCADE_BOOTH", "CASCADE_ROBOT", "CASCADE_BASE")}
    env.update(PYTHONPATH=str(REPO / "src"), HOME=str(tmp_path), CASCADE_PERCEPTION_KEY_FILE=str(key))
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env,
                         cwd=tmp_path, timeout=120, check=False)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().splitlines()[-1] == "BrainDetector BoxSegmenter []"


def test_mock_camera_profile_keeps_the_mock_detector(env, no_warm_up):
    det = demo._make_detector(load_demo_config(camera="mock"))
    assert type(det) is MockDetector
    assert no_warm_up == []


# -- other types --------------------------------------------------------------


def test_yoloe_is_refused_naming_type_brain(env):
    cfg = brain_cfg(type="yoloe", model="${repo}/models/yoloe-11s-seg.pt", device="auto")
    with pytest.raises(ValueError, match="type: brain"):
        demo._make_detector(cfg, warm_up=False)


def test_an_unknown_type_is_refused(env):
    with pytest.raises(ValueError, match="'dino'"):
        demo._make_detector(brain_cfg(type="dino"), warm_up=False)


def test_vlm_stays_the_cosmos_shim(env, no_warm_up):
    from cascade.perception.vlm_detector import VLMDetector

    det = demo._make_detector(brain_cfg(type="vlm"))
    assert isinstance(det, VLMDetector) and isinstance(det, BrainDetector)
    assert (det._base_url, det._model, det._segmenter) == ("http://127.0.0.1:8082/v1", "cosmos3-edge", None)
    assert no_warm_up == []


# -- detector block fields ----------------------------------------------------


def test_numeric_fields_are_cast(key_file):
    det = demo._make_detector(brain_cfg(
        type="brain", base_url="http://b/v1", model="m", api_key_file=str(key_file),
        timeout_s="7.5", max_tokens="900", refresh_s="2", max_cache_age_s="12", scene_change="3.5",
    ), warm_up=False)
    assert (det._timeout_s, det._max_tokens, det._refresh_s, det._max_cache_age_s, det._scene_change) == (
        7.5, 900, 2.0, 12.0, 3.5)
    assert type(det._max_tokens) is int


def test_a_bare_brain_block_takes_the_documented_defaults(env, tmp_path):
    key = tmp_path / "home" / ".cascade" / "nemoclaw" / "llama.key"
    key.parent.mkdir(parents=True)
    key.write_text("from-default-path")
    det = demo._make_detector(brain_cfg(type="brain"), warm_up=False)
    assert (det._base_url, det._model, det._api_key) == (DEFAULT_PERCEPTION_URL, "qwen3.8-27b",
                                                         "from-default-path")
    assert (det._timeout_s, det._max_tokens, det._refresh_s, det._max_cache_age_s, det._scene_change) == (
        20.0, 1500, 4.0, 30.0, 6.0)


def test_a_tilde_key_file_in_a_hand_made_block_is_expanded(env, tmp_path):
    (tmp_path / "home").mkdir()
    (tmp_path / "home" / "brain.key").write_text("tilde-key")
    det = demo._make_detector(brain_cfg(type="brain", api_key_file="~/brain.key"), warm_up=False)
    assert det._api_key == "tilde-key"


def test_a_missing_key_file_sends_no_key_and_warns_once(env, tmp_path, caplog):
    env.setenv("CASCADE_PERCEPTION_KEY_FILE", str(tmp_path / "absent.key"))
    with caplog.at_level(logging.WARNING):
        det = demo._make_detector(brain_cfg(), warm_up=False)
    assert det._api_key == "EMPTY"
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warnings) == 1 and "absent.key" in warnings[0].getMessage()


@pytest.mark.parametrize("segmenter", [{"enabled": False, "model": "sam2.1_l.pt"}, None])
def test_no_enabled_segmenter_gives_a_box_only_brain(key_file, no_warm_up, segmenter):
    cfg = brain_cfg()
    if segmenter is None:
        cfg._data.pop("segmenter")
    else:
        cfg._data["segmenter"] = segmenter
    det = demo._make_detector(cfg)
    assert type(det) is BrainDetector and det._segmenter is None
    assert no_warm_up == []


# -- one SAM for the box and the point segmenter ----------------------------------


@pytest.mark.parametrize("segmenter", [
    None,  # configs/demo.yaml as shipped
    {"enabled": True, "model": "sam2.1_l.pt", "device": None},
    {"enabled": True},
])
def test_box_and_point_segmenters_share_one_model(tmp_path, key_file, segmenter):
    """Ruling 1/7: both segmenters get identical model and device strings, so
    the path-keyed cache holds ONE SAM (and one lock) for both."""
    pytest.importorskip("pinocchio")
    cfg = load_demo_config(camera="mock", arm="mock", llm="mock")
    if segmenter is not None:
        cfg._data["segmenter"] = segmenter
    runtime, arm = demo.build_runtime(cfg, tmp_path / "run", view=False, serve=False)
    try:
        point = runtime._segmenter
    finally:
        demo.shutdown_runtime(runtime, arm)
    cfg._data["camera"].pop("detector")  # the mock camera pins the mock detector
    box = demo._make_detector(cfg, warm_up=False)._segmenter
    assert type(point) is PointSegmenter and type(box) is BoxSegmenter
    assert (box._model_path, box._device_spec) == (point._model_path, point._device_spec)
    assert box._device_spec == "auto"
    assert box._shared is point._shared


def test_shipped_sam_is_one_absolute_repo_path_whatever_the_cwd(tmp_path, key_file, monkeypatch):
    """Ruling 25: configs/demo.yaml names SAM as ${repo}/models/sam2.1_l.pt, so
    a server started outside models/ (.mcp.json, setup_agents, hermes_demo)
    still finds the weights offline, and both segmenters get that identical
    absolute string: one shared cache entry, one loaded SAM."""
    pytest.importorskip("pinocchio")
    monkeypatch.chdir(tmp_path)  # neither the repo root nor models/
    cfg = load_demo_config(camera="mock", arm="mock", llm="mock")
    assert cfg.segmenter.model == SHIPPED_SAM
    runtime, arm = demo.build_runtime(cfg, tmp_path / "run", view=False, serve=False)
    try:
        point = runtime._segmenter
    finally:
        demo.shutdown_runtime(runtime, arm)
    cfg._data["camera"].pop("detector")  # the mock camera pins the mock detector
    box = demo._make_detector(cfg, warm_up=False)._segmenter
    assert type(point) is PointSegmenter and type(box) is BoxSegmenter
    assert box._model_path == point._model_path == SHIPPED_SAM
    assert Path(box._model_path).is_absolute()
    assert Path(box._model_path).relative_to(REPO) == Path("models/sam2.1_l.pt")
    assert box._shared is point._shared


# -- SAM warm-up --------------------------------------------------------------------


def test_the_factory_warms_sam_on_a_daemon_thread_without_blocking(key_file, monkeypatch):
    started, release = threading.Event(), threading.Event()
    seen = []

    def blocking_warm_up(segmenter):
        seen.append((segmenter, threading.current_thread()))
        started.set()
        release.wait(10)

    monkeypatch.setattr(demo, "_warm_box_segmenter", blocking_warm_up, raising=False)
    try:
        det = demo._make_detector(brain_cfg())  # returns while the warm-up is still blocked
        assert started.wait(5), "no warm-up started"
        (segmenter, thread), = seen
        assert segmenter is det._segmenter
        assert thread is not threading.current_thread() and thread.daemon
        assert thread.is_alive()
        # The detector owns the thread: its close() (a shutdown stage) joins it.
        assert det.warm_up_thread is thread and thread.name == "sam-warm-up"
    finally:
        release.set()


def test_warm_up_false_starts_no_warm_up(key_file, no_warm_up):
    det = demo._make_detector(brain_cfg(), warm_up=False)
    assert no_warm_up == []
    assert det.warm_up_thread is None


def test_a_block_without_a_type_is_refused(env):
    """`type: brain` is what configs/demo.yaml ships, not an implicit default."""
    with pytest.raises(ValueError, match="unknown detector type ''"):
        demo._make_detector(brain_cfg(base_url="http://b/v1", model="m"), warm_up=False)
    doc = demo._make_detector.__doc__ or ""
    assert "(the default)" not in doc and "without a `type`" in doc


class _FakeSegmenter:
    def __init__(self, error=None):
        self.calls, self.error = [], error

    def masks(self, frame, boxes):
        self.calls.append((frame, np.asarray(boxes)))
        if self.error is not None:
            raise self.error
        return [None] * len(boxes)


def test_warm_up_runs_one_box_predict_on_a_blank_frame():
    seg = _FakeSegmenter()
    demo._warm_box_segmenter(seg)
    (frame, boxes), = seg.calls
    assert isinstance(frame, Frame) and frame.rgb.dtype == np.uint8 and frame.rgb.ndim == 3
    assert not frame.rgb.any()
    h, w = frame.rgb.shape[:2]
    assert boxes.shape == (1, 4)
    x1, y1, x2, y2 = boxes[0]
    assert 0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h


def test_a_failed_warm_up_warns_once_and_never_raises(caplog):
    seg = _FakeSegmenter(error=ModuleNotFoundError("No module named 'ultralytics'"))
    with caplog.at_level(logging.WARNING):
        demo._warm_box_segmenter(seg)
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warnings) == 1 and "ultralytics" in warnings[0].getMessage()


# -- MCP server: no detector-weights override any more ----------------------------


def test_mcp_server_no_longer_overrides_the_detector_model(env, key_file, monkeypatch):
    """CASCADE_DETECTOR_MODEL used to write a detector weights .pt path into
    detector.model, which is now the brain's served model name."""
    from cascade.apps.mcp_server import McpSkillServer

    for name in ("CASCADE_ARMS", "CASCADE_CAMERAS", "CASCADE_GROUNDING_LLM_PROFILE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("CASCADE_ARM", "mock")
    monkeypatch.setenv("CASCADE_CAMERA", "mock")
    monkeypatch.setenv("CASCADE_DETECTOR_MODEL", "/models/old-detector-weights.pt")
    monkeypatch.setenv("CASCADE_DETECT_CLASSES", "paper cup, red lobster ,")
    built = []

    class Built(Exception):
        pass

    def build(cfg, *args, **kwargs):
        built.append(cfg)
        raise Built

    monkeypatch.setattr(demo, "build_runtime", build)
    with pytest.raises(Built):
        McpSkillServer()._ensure_runtime(_poison=False)
    cfg, = built
    assert cfg.detector.model == "qwen3.8-27b"
    assert cfg.detect_classes == ["paper cup", "red lobster"]
