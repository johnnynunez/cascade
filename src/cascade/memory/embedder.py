"""Memory embedders: the vectors behind visual episodic recall (ROADMAP #7).

An embedder turns a camera image (BGR, the ``Frame.rgb`` convention) or a short
text into a unit vector. ``EpisodicMemory(embedder=...)`` stores frame and
per-object crop vectors in its TurboQuant index; ``SkillLibrary`` can rank its
notes by text similarity. Everything here is OPT-IN: the shipped profile has
``memory.embedder.backend: none`` and the memory then behaves exactly as it did
before this module existed.

Two backends:

* ``hash`` -- :class:`HashEmbedder`, deterministic and dependency-free (numpy +
  OpenCV, both base dependencies). Images: a saturation-weighted, hue-wrapping
  colour histogram plus a coarse spatial layout, so the same object under
  sensor noise matches and a red crop does not match a blue one. Text: crc32-
  hashed, lightly stemmed bag of words ("grasping cubes" ~ "grasp cube"). The
  two live in DIFFERENT spaces (``joint_space = False``, different sizes): a
  colour histogram can never answer "the thing that looked like a mug".
* ``siglip`` / ``clip`` -- :class:`TransformersEmbedder`, a real image-text
  model through Hugging Face ``transformers`` (the ``memory-embed`` extra; torch
  from the host env like every other model here, see ``device.py``). Image and
  text share one space (``joint_space = True``), which is what text queries
  ("looks like X") require. Weights load with ``local_files_only=True`` unless
  the profile explicitly allows a download.

A backend that was requested but cannot run raises :class:`EmbedderUnavailable`
-- it never silently degrades to another backend, because a hash vector
answering a semantic query would look like recall and be noise. Embeddings are
advisory evidence about the past: they never confirm a physical outcome and
never gate motion.
"""

from __future__ import annotations

import os
import re
from typing import Any, Protocol, runtime_checkable
from zlib import crc32

import cv2
import numpy as np

#: The optional dependency group that carries the real image-text backends.
EXTRA = "memory-embed"

#: Default checkpoints per backend. Base-size models: they fit beside the
#: detector on a Jetson and run on CPU at a few frames per second.
DEFAULT_MODELS = {
    "siglip": "google/siglip2-base-patch16-224",
    "clip": "openai/clip-vit-base-patch32",
}

#: Backends `make_embedder` accepts; "none" (or an empty spec) means no embedder.
BACKENDS = ("none", "hash", "siglip", "clip")

ENV_BACKEND = "CASCADE_MEMORY_EMBEDDER"
ENV_MODEL = "CASCADE_MEMORY_EMBEDDER_MODEL"


class EmbedderUnavailable(RuntimeError):
    """A requested embedder cannot run here (missing extra, torch or weights)."""


@runtime_checkable
class Embedder(Protocol):
    """What the memory needs from an embedder.

    ``embed_image`` takes a BGR uint8 image and returns a float32 unit vector
    of length ``dim``; ``embed_text`` returns one of length ``text_dim``.
    ``joint_space`` says whether the two are comparable (text -> image recall).
    ``text_image_floor`` is the cosine below which a text -> image match is
    not a match (None: no calibrated floor); ``text_floor`` is the same for
    text -> text similarity (skill-library notes).
    """

    name: str
    dim: int
    text_dim: int
    joint_space: bool
    text_image_floor: float | None
    text_floor: float

    def embed_image(self, bgr: np.ndarray) -> np.ndarray: ...

    def embed_text(self, text: str) -> np.ndarray: ...


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    n = float(np.linalg.norm(v))
    return (v / n if n > 0 else v).astype(np.float32)


def _as_bgr_u8(img: np.ndarray) -> np.ndarray:
    a = np.asarray(img)
    if a.ndim == 2:
        a = np.repeat(a[:, :, None], 3, axis=2)
    if a.ndim != 3 or a.shape[2] < 3:
        raise ValueError(f"expected an (H, W, 3) BGR image, got shape {a.shape}")
    a = a[:, :, :3]
    if a.dtype != np.uint8:
        a = np.clip(a, 0, 255).astype(np.uint8)
    if a.shape[0] < 1 or a.shape[1] < 1:
        raise ValueError("empty image")
    return np.ascontiguousarray(a)


# ── the deterministic, dependency-free backend ─────────────────────────

_STOPWORDS = frozenset(
    "the a an it its this that those these to of and or in on at by for with from into onto "
    "please can you could would my me your is are be some any "
    "el la los las un una y o por favor lo esa ese esto de del en con".split()
)

def _stem(tok: str) -> str:
    """A deliberately small stemmer: inflections of one word share a stem
    (grasp/grasping/grasped, cube/cubes, box/boxes, place/placed/placing,
    move/moving). Consistency matters more than linguistics here: the same
    rule runs on both sides of every comparison."""
    if len(tok) > 5 and tok.endswith("ies"):
        tok = tok[:-3] + "y"
    elif tok.endswith("ing") and len(tok) - 3 >= 3:
        tok = tok[:-3]
    elif tok.endswith("ed") and len(tok) - 2 >= 3:
        tok = tok[:-2]
    elif tok.endswith("es") and len(tok) - 2 >= 3 and tok[:-2].endswith(("s", "x", "z", "ch", "sh")):
        tok = tok[:-2]
    elif tok.endswith("s") and not tok.endswith("ss") and len(tok) - 1 >= 3:
        tok = tok[:-1]
    if len(tok) >= 4 and tok.endswith("e"):
        tok = tok[:-1]
    return tok


def content_tokens(text: str) -> list[str]:
    """Lower-cased, stemmed content words (stopwords out). Also the identity
    used by action<->object consolidation, so both agree on what one word is."""
    return [_stem(t) for t in re.findall(r"[a-z0-9]+", str(text).lower()) if t not in _STOPWORDS]


class HashEmbedder:
    """Deterministic colour/layout image vectors and hashed-word text vectors.

    No model, no download, identical across processes and platforms (crc32,
    never Python's salted ``hash``). Not a semantic model: it recognises "the
    same colours in the same places" and "the same words", nothing more.
    """

    name = "hash-v1"
    joint_space = False
    #: SEPARATE spaces: the text side is a word hash, the image side a colour
    #: histogram; the different sizes make an accidental comparison fail.
    text_dim = 256
    text_image_floor: float | None = None
    #: Text-text cosine at which two phrasings count as related. A note's key
    #: shares two stemmed content words with a four-word task at ~0.45; one
    #: word with a long task is below 0.3.
    text_floor = 0.3

    _HUE_BINS = 12
    _VALUE_BINS = 3  # black / grey / white for unsaturated pixels
    _GRID = 4  # 4x4 layout cells
    _SIZE = (64, 48)  # every image is reduced to this before binning
    _ACHROMATIC_WEIGHT = 0.35
    _LAYOUT_WEIGHT = 0.6

    def __init__(self):
        nb = self._HUE_BINS + self._VALUE_BINS
        self.dim = nb * (1 + self._GRID * self._GRID)

    def embed_image(self, bgr: np.ndarray) -> np.ndarray:
        img = _as_bgr_u8(bgr)
        small = cv2.resize(img, self._SIZE, interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.float64)
        hue = hsv[:, 0] / 180.0  # OpenCV hue is 0..179: one full turn
        sat = hsv[:, 1] / 255.0
        val = hsv[:, 2] / 255.0
        nh, nv = self._HUE_BINS, self._VALUE_BINS
        n = hue.shape[0]
        feats = np.zeros((n, nh + nv))
        chroma = (sat >= 0.25) & (val >= 0.15)
        # Soft, CIRCULAR hue binning: red sits at both ends of OpenCV's hue
        # range, and sensor noise flips a pure red pixel between 0 and 179.
        pos = hue * nh - 0.5
        lo = np.floor(pos)
        frac = pos - lo
        i0 = lo.astype(int) % nh
        i1 = (i0 + 1) % nh
        rows = np.arange(n)
        w = np.where(chroma, sat, 0.0)
        np.add.at(feats, (rows, i0), w * (1.0 - frac))
        np.add.at(feats, (rows, i1), w * frac)
        # Unsaturated pixels (table, shadows, highlights) by brightness only,
        # down-weighted so a shared background cannot dominate the match.
        vpos = np.clip(val * nv - 0.5, 0.0, nv - 1.0)
        vlo = np.floor(vpos)
        vfrac = vpos - vlo
        v0 = vlo.astype(int)
        v1 = np.minimum(v0 + 1, nv - 1)
        aw = np.where(chroma, 0.0, self._ACHROMATIC_WEIGHT)
        np.add.at(feats, (rows, nh + v0), aw * (1.0 - vfrac))
        np.add.at(feats, (rows, nh + v1), aw * vfrac)
        h, wpx = small.shape[:2]
        g = self._GRID
        cell = (np.arange(h)[:, None] * g // h) * g + (np.arange(wpx)[None, :] * g // wpx)
        layout = np.zeros((g * g, nh + nv))
        np.add.at(layout, cell.reshape(-1), feats)
        glob = feats.sum(0) / n
        layout /= max(n / (g * g), 1.0)
        # Hellinger (square-root) features: cosine on them is the Bhattacharyya
        # coefficient, which does not let one large bin swamp the rest.
        vec = np.concatenate([np.sqrt(glob), self._LAYOUT_WEIGHT * np.sqrt(layout).reshape(-1)])
        return _unit(vec)

    def embed_text(self, text: str) -> np.ndarray:
        # Set semantics (each distinct word once): a note that repeats its
        # skill name in the title and the guard line must not outweigh the
        # words it shares with the task.
        v = np.zeros(self.text_dim, dtype=np.float64)
        for tok in dict.fromkeys(content_tokens(text)):
            h = crc32(tok.encode())
            v[h % self.text_dim] += 1.0 if (h >> 16) & 1 else -1.0
        return _unit(v)


# ── the optional SigLIP / CLIP backend ──────────────────────────────────


def _first_vector(out: Any) -> np.ndarray:
    """Model output -> the first row as numpy. transformers 4.x returns a bare
    tensor from ``get_*_features``; newer releases return a model output whose
    ``pooler_output`` carries the projected embedding."""
    for attr in ("pooler_output", "image_embeds", "text_embeds"):
        val = getattr(out, attr, None)
        if val is not None:
            out = val
            break
    return np.asarray(out[0].detach().float().cpu().numpy(), dtype=np.float64)


def _scalar(param: Any) -> float | None:
    if param is None:
        return None
    try:
        return float(np.asarray(param.detach().float().cpu().numpy()).reshape(-1)[0])
    except Exception:  # noqa: BLE001 -- a missing/odd parameter only costs the floor
        return None


class TransformersEmbedder:
    """SigLIP / CLIP through ``transformers.AutoModel`` + ``AutoProcessor``.

    Construction loads the model and embeds one probe image and one probe text,
    so a broken install or a missing checkpoint fails HERE, at runtime build,
    with :class:`EmbedderUnavailable` -- never at the first recall.
    """

    joint_space = True

    def __init__(self, backend: str, model: str | None = None, *, device: str = "auto",
                 local_files_only: bool = True, text_image_floor: float | None = None,
                 text_floor: float | None = None):
        backend = str(backend).lower()
        if backend not in ("siglip", "clip"):
            raise ValueError(f"TransformersEmbedder backend must be siglip or clip, got {backend!r}")
        self.backend = backend
        self.model_id = str(model or DEFAULT_MODELS[backend])
        self.name = f"{backend}:{self.model_id}"
        try:
            import torch  # noqa: F401 -- probed here so the error names it
            from transformers import AutoModel, AutoProcessor
        except ImportError as e:
            raise EmbedderUnavailable(
                f"memory embedder {backend!r} needs `transformers` and `torch`, which are not "
                f"importable here ({e}). Install the optional extra `cascade[{EXTRA}]` (e.g. "
                f"`uv pip install -e '.[{EXTRA}]'`) plus the torch build for this platform, or "
                "set memory.embedder.backend to `hash` or `none`."
            ) from e
        import torch

        from ..device import resolve_device

        self._torch = torch
        self.device = resolve_device(device, what="memory embedder")
        kw = {"local_files_only": bool(local_files_only)}
        try:
            self._processor = AutoProcessor.from_pretrained(self.model_id, **kw)
            self._model = AutoModel.from_pretrained(self.model_id, **kw).to(self.device).eval()
        except Exception as e:  # noqa: BLE001 -- OSError for missing weights, anything else too
            raise EmbedderUnavailable(
                f"memory embedder {backend!r} could not load {self.model_id!r} "
                f"(local_files_only={kw['local_files_only']}): {e}. Pre-fetch the checkpoint "
                "into the Hugging Face cache, or set memory.embedder.local_files_only: false "
                "to allow a download."
            ) from e
        # Probe both towers now: the dimensions come from the model itself and
        # a model that cannot embed fails the build, not the first recall.
        self.dim = int(self.embed_image(np.zeros((32, 32, 3), np.uint8)).shape[0])
        self.text_dim = int(self.embed_text("a photo").shape[0])
        if self.dim != self.text_dim:
            raise EmbedderUnavailable(
                f"{self.model_id!r} has image dim {self.dim} != text dim {self.text_dim}; "
                "not a joint image-text model")
        if text_image_floor is None:
            # SigLIP is trained with a sigmoid: p(match) = sigmoid(exp(s) * cos + b),
            # so the model's own 0.5 decision boundary is cos = -b / exp(s).
            scale = _scalar(getattr(self._model, "logit_scale", None))
            bias = _scalar(getattr(self._model, "logit_bias", None))
            if scale is not None and bias is not None:
                text_image_floor = -bias / float(np.exp(scale))
            elif backend == "clip":
                text_image_floor = 0.25  # NOT calibrated on this rig; override per profile
        self.text_image_floor = None if text_image_floor is None else float(text_image_floor)
        #: NOT calibrated with real weights (no weights on the CI/dev boxes):
        #: text-text cosines of a contrastive text tower run high; override
        #: with memory.embedder.text_floor after measuring on the GPU host.
        self.text_floor = float(text_floor) if text_floor is not None else 0.85

    def _move(self, batch) -> dict:
        return {k: (v.to(self.device) if hasattr(v, "to") else v) for k, v in dict(batch).items()}

    def embed_image(self, bgr: np.ndarray) -> np.ndarray:
        rgb = np.ascontiguousarray(_as_bgr_u8(bgr)[:, :, ::-1])  # Frame.rgb is BGR
        inputs = self._move(self._processor(images=[rgb], return_tensors="pt"))
        with self._torch.no_grad():
            out = self._model.get_image_features(**inputs)
        return _unit(_first_vector(out))

    def embed_text(self, text: str) -> np.ndarray:
        kw: dict[str, Any] = {"padding": "max_length", "truncation": True}
        text = str(text)
        if self.backend == "siglip":
            # SigLIP was trained on lower-cased text padded to 64 tokens.
            kw["max_length"] = 64
            text = text.lower()
        inputs = self._move(self._processor(text=[text], return_tensors="pt", **kw))
        with self._torch.no_grad():
            out = self._model.get_text_features(**inputs)
        return _unit(_first_vector(out))


# ── configuration ───────────────────────────────────────────────────────


def _get(spec: Any, key: str, default: Any = None) -> Any:
    try:
        val = spec.get(key, default)
    except AttributeError:
        return default
    return default if val is None else val


def embedder_config(cfg) -> dict:
    """``memory.embedder`` from a demo config as a plain dict, with the env
    overrides applied (``CASCADE_MEMORY_EMBEDDER`` = backend,
    ``CASCADE_MEMORY_EMBEDDER_MODEL`` = checkpoint). Absent -> backend none."""
    memory = _get(cfg, "memory", None)
    spec = _get(memory, "embedder", None) if memory is not None else None
    if spec is None:
        out: dict = {}
    elif hasattr(spec, "as_dict"):
        out = spec.as_dict()
    else:
        out = dict(spec)
    out["backend"] = str(out.get("backend") or "none").strip().lower()
    env_backend = os.environ.get(ENV_BACKEND, "").strip().lower()
    if env_backend:
        out["backend"] = env_backend
    env_model = os.environ.get(ENV_MODEL, "").strip()
    if env_model:
        out["model"] = env_model
    return out


def make_embedder(spec) -> Embedder | None:
    """Spec dict -> an embedder, or None for ``backend: none`` / an empty spec.

    Unknown backends are a ValueError; a known backend that cannot run here is
    :class:`EmbedderUnavailable`. Neither ever falls back to another backend.
    """
    backend = str(_get(spec, "backend", "none") or "none").strip().lower()
    if backend in ("", "none", "off", "false"):
        return None
    if backend == "hash":
        return HashEmbedder()
    if backend in ("siglip", "clip"):
        floor = _get(spec, "text_image_floor", None)
        tfloor = _get(spec, "text_floor", None)
        return TransformersEmbedder(
            backend,
            _get(spec, "model", None),
            device=str(_get(spec, "device", "auto")),
            local_files_only=bool(_get(spec, "local_files_only", True)),
            text_image_floor=None if floor is None else float(floor),
            text_floor=None if tfloor is None else float(tfloor),
        )
    raise ValueError(
        f"unknown memory embedder backend {backend!r}; expected one of {', '.join(BACKENDS)}")
