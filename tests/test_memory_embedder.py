"""Memory embedder (ROADMAP #7): the protocol, the deterministic hash backend
and the optional SigLIP/CLIP backend's contract.

CI and this box have neither `transformers` nor model weights, so the optional
backend is exercised here against a stub `transformers`/`torch` that records
what it was asked (AGENTS.md: do not gate a test on an extra when the contract
can be faked). `tests/test_memory_embedder_weights.py` runs the real model
where the `memory-embed` extra and cached weights exist.

Nothing here is evidence of semantic recall quality with real weights.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import types

import numpy as np
import pytest
from conftest import REPO

try:
    from cascade.memory.embedder import (
        DEFAULT_MODELS,
        EXTRA,
        Embedder,
        EmbedderUnavailable,
        HashEmbedder,
        TransformersEmbedder,
        embedder_config,
        make_embedder,
    )
except ImportError:  # pre-change tree: keep every test collectable so each FAILS on its own
    DEFAULT_MODELS = EXTRA = Embedder = EmbedderUnavailable = HashEmbedder = None
    TransformersEmbedder = embedder_config = make_embedder = None

RED = (0, 0, 220)      # BGR
DARK_RED = (0, 0, 170)
BLUE = (220, 0, 0)
GREEN = (0, 200, 0)
GREY = (128, 128, 128)


def _scene(color=RED, x0=20, y0=45, size=30, h=120, w=160):
    """A grey table with one coloured square (a whole camera frame)."""
    img = np.empty((h, w, 3), np.uint8)
    img[:] = GREY
    img[y0:y0 + size, x0:x0 + size] = color
    return img


def _crop(color=RED, size=40, border=6):
    """An object crop: the object fills the box, a little table around it."""
    img = np.empty((size, size, 3), np.uint8)
    img[:] = GREY
    img[border:size - border, border:size - border] = color
    return img


def _noisy(img, sigma=6.0, seed=0):
    rng = np.random.default_rng(seed)
    return np.clip(img.astype(float) + rng.normal(0, sigma, img.shape), 0, 255).astype(np.uint8)


def _cos(a, b) -> float:
    return float(np.dot(np.asarray(a, float), np.asarray(b, float)))


# ── the deterministic, dependency-free backend ─────────────────────────


def test_hash_embedder_satisfies_the_protocol_and_is_deterministic():
    e1, e2 = HashEmbedder(), HashEmbedder()
    assert isinstance(e1, Embedder)
    assert e1.name == "hash-v1"
    # No shared image-text space: a text hash must never be compared with a
    # colour histogram, and the different sizes make such a mix-up fail loudly.
    assert e1.joint_space is False and e1.text_dim != e1.dim
    img = _scene()
    v1, v2 = e1.embed_image(img), e2.embed_image(img.copy())
    assert v1.shape == (e1.dim,) and v1.dtype == np.float32
    assert np.array_equal(v1, v2)
    assert abs(float(np.linalg.norm(v1)) - 1.0) < 1e-5
    t = e1.embed_text("Grasp the red cube")
    assert t.shape == (e1.text_dim,) and t.dtype == np.float32
    assert abs(float(np.linalg.norm(t)) - 1.0) < 1e-5
    assert np.array_equal(t, e2.embed_text("grasp the RED cube"))


def test_hash_text_embedding_is_identical_across_interpreters():
    """crc32, not Python's salted hash(): a persisted or cross-process vector
    must mean the same thing in the next session."""
    code = (
        "import json, sys; sys.path.insert(0, {src!r}); "
        "from cascade.memory.embedder import HashEmbedder; "
        "print(json.dumps(HashEmbedder().embed_text('grasp the red cube').round(6).tolist()))"
    ).format(src=str(REPO / "src"))
    outs = []
    for seed in ("1", "2"):
        env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
        env["PYTHONHASHSEED"] = seed
        proc = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True,
                              text=True, timeout=120, check=True)
        outs.append(json.loads(proc.stdout.strip().splitlines()[-1]))
    assert outs[0] == outs[1]
    assert np.allclose(outs[0], HashEmbedder().embed_text("grasp the red cube"), atol=1e-6)


def test_hash_image_embedding_separates_object_colours_and_tolerates_noise():
    emb = HashEmbedder()
    red = emb.embed_image(_crop(RED))
    assert _cos(red, emb.embed_image(_noisy(_crop(RED)))) > 0.97
    same_hue = _cos(red, emb.embed_image(_crop(DARK_RED)))
    other = max(_cos(red, emb.embed_image(_crop(BLUE))), _cos(red, emb.embed_image(_crop(GREEN))))
    assert other < 0.6, other
    assert same_hue > other + 0.3, (same_hue, other)


def test_hash_image_embedding_keeps_layout_for_whole_frames():
    emb = HashEmbedder()
    left = emb.embed_image(_scene(x0=20))
    nudged = emb.embed_image(_scene(x0=24))
    right = emb.embed_image(_scene(x0=110))
    assert _cos(left, nudged) > _cos(left, right)
    assert _cos(left, right) < 0.999


def test_hash_text_embedding_matches_inflections_not_unrelated_commands():
    emb = HashEmbedder()
    q = emb.embed_text("grasp the cube")
    assert _cos(q, emb.embed_text("Grasping cubes")) > 0.3
    assert _cos(q, emb.embed_text("wave at the audience")) < 0.15


# ── configuration ───────────────────────────────────────────────────────


def test_make_embedder_selects_backends_and_refuses_unknown_names():
    assert make_embedder({}) is None
    assert make_embedder({"backend": "none"}) is None
    assert isinstance(make_embedder({"backend": "hash"}), HashEmbedder)
    with pytest.raises(ValueError, match="unknown memory embedder backend"):
        make_embedder({"backend": "resnet"})


def test_shipped_config_leaves_the_embedder_off_and_env_overrides_it(monkeypatch):
    from cascade.config import load_demo_config

    cfg = load_demo_config(camera="mock", arm="mock", llm="mock")
    spec = embedder_config(cfg)
    assert spec["backend"] == "none"  # opt-in: the shipped profile is unchanged
    monkeypatch.setenv("CASCADE_MEMORY_EMBEDDER", "hash")
    assert embedder_config(cfg)["backend"] == "hash"
    monkeypatch.setenv("CASCADE_MEMORY_EMBEDDER_MODEL", "google/siglip2-base-patch16-224")
    assert embedder_config(cfg)["model"] == "google/siglip2-base-patch16-224"


# ── the optional SigLIP/CLIP backend ────────────────────────────────────


def test_siglip_backend_without_the_extra_raises_an_explicit_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "transformers", None)  # import -> ImportError
    with pytest.raises(EmbedderUnavailable) as err:
        make_embedder({"backend": "siglip"})
    msg = str(err.value)
    assert EXTRA == "memory-embed"
    assert f"cascade[{EXTRA}]" in msg and "torch" in msg
    assert DEFAULT_MODELS["siglip"].startswith("google/siglip")


class _T:
    """numpy-backed stand-in for a torch tensor: only what the backend calls."""

    def __init__(self, a):
        self.a = np.asarray(a, dtype=np.float32)

    def to(self, *_a, **_k):
        return self

    def detach(self):
        return self

    def float(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.a

    def __getitem__(self, i):
        return _T(self.a[i])


def _stub_transformers(monkeypatch, seen: dict):
    torch = types.ModuleType("torch")

    class _NoGrad:
        def __enter__(self):
            seen["no_grad"] = seen.get("no_grad", 0) + 1

        def __exit__(self, *exc):
            return False

    torch.no_grad = _NoGrad

    class _Processor:
        @classmethod
        def from_pretrained(cls, model, **kw):
            seen["processor"] = (model, kw)
            return cls()

        def __call__(self, images=None, text=None, **kw):
            if images is not None:
                seen.setdefault("images", []).append(np.asarray(images[0]).copy())
                return {"pixel_values": _T(np.zeros((1, 3, 4, 4)))}
            seen.setdefault("texts", []).append((list(text), kw))
            return {"input_ids": _T(np.zeros((1, 64)))}

    class _Model:
        logit_scale = _T([np.log(10.0)])
        logit_bias = _T([-2.0])

        @classmethod
        def from_pretrained(cls, model, **kw):
            seen["model"] = (model, kw)
            return cls()

        def to(self, device):
            seen["device"] = device
            return self

        def eval(self):
            return self

        def get_image_features(self, pixel_values=None):
            return _T([[3.0, 4.0, 0.0]])  # a bare tensor (transformers 4.x)

        def get_text_features(self, **kw):
            seen["text_kwargs"] = sorted(kw)
            # an output object (newer transformers return model outputs)
            return types.SimpleNamespace(pooler_output=_T([[0.0, 0.0, 2.0]]))

    tf = types.ModuleType("transformers")
    tf.AutoModel = _Model
    tf.AutoProcessor = _Processor
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", tf)
    monkeypatch.delenv("CASCADE_DEVICE", raising=False)
    monkeypatch.delenv("CASCADE_REQUIRE_CUDA", raising=False)


def test_transformers_backend_contract_with_a_stub_model(monkeypatch):
    seen: dict = {}
    _stub_transformers(monkeypatch, seen)
    emb = make_embedder({"backend": "siglip", "model": "google/siglip-base-patch16-224",
                         "device": "cpu", "local_files_only": True})
    assert isinstance(emb, TransformersEmbedder) and isinstance(emb, Embedder)
    assert emb.joint_space is True and emb.dim == emb.text_dim == 3
    assert emb.name == "siglip:google/siglip-base-patch16-224"
    assert seen["model"][0] == "google/siglip-base-patch16-224"
    assert seen["model"][1]["local_files_only"] is True
    assert seen["processor"][1]["local_files_only"] is True
    assert seen["device"] == "cpu"

    bgr = np.zeros((8, 8, 3), np.uint8)
    bgr[..., 0] = 255  # pure BLUE in OpenCV's BGR
    v = emb.embed_image(bgr)
    np.testing.assert_allclose(v, [0.6, 0.8, 0.0], atol=1e-6)  # L2-normalised
    sent = seen["images"][-1]
    assert sent[..., 2].min() == 255 and sent[..., 0].max() == 0  # handed over as RGB

    t = emb.embed_text("a red cube")
    np.testing.assert_allclose(t, [0.0, 0.0, 1.0], atol=1e-6)
    texts, kw = seen["texts"][-1]
    assert texts == ["a red cube"]
    assert kw.get("padding") == "max_length" and kw.get("truncation") is True
    assert seen["text_kwargs"] == ["input_ids"]
    assert seen["no_grad"] >= 2
    # SigLIP's own p=0.5 decision boundary: cos = -logit_bias / exp(logit_scale)
    assert emb.text_image_floor == pytest.approx(0.2)


def test_transformers_backend_reports_missing_weights_as_unavailable(monkeypatch):
    seen: dict = {}
    _stub_transformers(monkeypatch, seen)

    def _missing(model, **kw):
        raise OSError(f"{model} is not cached and local_files_only=True")

    sys.modules["transformers"].AutoModel.from_pretrained = staticmethod(_missing)
    with pytest.raises(EmbedderUnavailable, match="could not load"):
        make_embedder({"backend": "clip", "device": "cpu"})
    assert DEFAULT_MODELS["clip"].startswith("openai/clip")
