"""Real-weight evaluation of the SigLIP/CLIP memory embedder (ROADMAP #7).

This is the part a stub cannot stand in for, so it SKIPS unless `transformers`
and torch are importable AND the default checkpoint is already in the local
Hugging Face cache: the embedder loads with ``local_files_only=True`` and this
file never downloads anything. On the GPU host:

    uv pip install -e '.[memory-embed]'      # transformers + pillow; torch from the host
    python -c "from huggingface_hub import snapshot_download as s; \
[s(m) for m in ('google/siglip2-base-patch16-224', 'openai/clip-vit-base-patch32')]"
    python -m pytest tests/test_memory_embedder_weights.py -q -rs -s

The images are synthetic flat-colour scenes, so a pass is a sanity check of
the wiring (BGR->RGB, joint space, normalisation, recall through the index),
not a measurement of recall quality on real objects; `-s` prints the cosines
needed to calibrate `text_floor` / `text_image_floor` per checkpoint.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("transformers")
pytest.importorskip("torch")

from cascade.memory import EpisodicMemory  # noqa: E402
from cascade.memory.embedder import EmbedderUnavailable, make_embedder  # noqa: E402

COLOURS = {"red": (0, 0, 220), "green": (0, 200, 0), "blue": (220, 0, 0)}  # BGR


def _crop(bgr, size=224, border=40):
    img = np.full((size, size, 3), 128, np.uint8)
    img[border:size - border, border:size - border] = bgr
    return img


@pytest.fixture(scope="module", params=["siglip", "clip"])
def embedder(request):
    try:
        return make_embedder({"backend": request.param, "local_files_only": True})
    except EmbedderUnavailable as e:
        pytest.skip(f"{request.param}: {e}")


def test_real_embedder_is_a_deterministic_unit_joint_space(embedder):
    assert embedder.joint_space and embedder.dim == embedder.text_dim > 0
    img = _crop(COLOURS["red"])
    a, b = embedder.embed_image(img), embedder.embed_image(img.copy())
    assert np.allclose(a, b, atol=1e-4)
    assert abs(float(np.linalg.norm(a)) - 1.0) < 1e-4
    t = embedder.embed_text("a red cube")
    assert abs(float(np.linalg.norm(t)) - 1.0) < 1e-4
    print(f"\n{embedder.name}: dim={embedder.dim} text_image_floor={embedder.text_image_floor} "
          f"text_floor={embedder.text_floor}")


def test_real_embedder_matches_colour_words_to_colour_crops(embedder):
    images = {c: embedder.embed_image(_crop(bgr)) for c, bgr in COLOURS.items()}
    for colour in COLOURS:
        text = embedder.embed_text(f"a photo of a {colour} cube")
        sims = {c: float(np.dot(text, v)) for c, v in images.items()}
        print(f"{embedder.name} '{colour}': " + ", ".join(f"{c}={s:.3f}" for c, s in sims.items()))
        assert max(sims, key=sims.get) == colour, sims


def test_real_embedder_recalls_the_thing_that_looked_like_x(embedder):
    mem = EpisodicMemory(embedder=embedder)
    for colour, bgr in COLOURS.items():
        mem.add("action", f"localize_object(label={colour} cube) -> ok",
                crops={f"{colour} cube": _crop(bgr)})
    hits = mem.recall_visual("a photo of a blue cube", k=3, min_sim=-1.0)
    print(f"{embedder.name} recall: {[(h['label'], h['score']) for h in hits]}")
    assert hits[0]["label"] == "blue cube", hits
