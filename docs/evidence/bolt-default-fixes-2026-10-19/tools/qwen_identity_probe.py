"""--brain qwen byte-identity probe (throwaway, PR perf/bolt-default-fixes): build the perception detector from
configs/demo.yaml with CASCADE_PERCEPTION_MODEL=qwen3.8-27b, run one inventory and one refer against a recording
fake client (no network), and print the exact requests plus the parsed boxes of a fixed reply as JSON. Run with the
same interpreter on main and on the head (`git archive` copies) and diff the outputs.

  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=<tree>/src CASCADE_PERCEPTION_MODEL=qwen3.8-27b python qwen_identity_probe.py
"""
import json
import os
import sys
import types

import numpy as np

os.environ.setdefault("CASCADE_PERCEPTION_MODEL", "qwen3.8-27b")
os.environ["CASCADE_PERCEPTION_KEY_FILE"] = "/nonexistent"
from cascade.apps import demo  # noqa: E402
from cascade.config import load_demo_config  # noqa: E402
from cascade.perception import brain_detector as bd  # noqa: E402
from cascade.types import Frame  # noqa: E402

REPLY = ('[\n  {"label": "red plush lobster", "bbox_2d": [573, 552, 738, 708]},\n'
         '  {"label": "computer mouse", "bbox_2d": [552, 204, 604, 248]},\n'
         + ",\n".join('  {"label": "white cable", "bbox_2d": [%d, 620, %d, 650]}' % (380 + 20 * k, 430 + 20 * k)
                      for k in range(25)) + "\n]")


class Client:
    def __init__(self):
        self.requests = []
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self.create))

    def create(self, **kw):
        self.requests.append(kw)
        msg = types.SimpleNamespace(content=REPLY, reasoning_content=None)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg, finish_reason="stop")])


cfg = load_demo_config(camera="isaac")
client = Client()
det = demo._make_detector(cfg, warm_up=False)
det._segmenter = None
det._client = client
rgb = np.zeros((72, 128, 3), dtype=np.uint8)
frame = Frame(rgb=rgb, depth_m=None, K=np.eye(3), t=0.0)
inv = det.detect(frame)
ref = det.detect(frame, ["white cup"])
for r in client.requests:
    r["messages"][0]["content"][0]["image_url"]["url"] = r["messages"][0]["content"][0]["image_url"]["url"][:40]
out = {"model": det._model, "requests": client.requests,
       "inventory": [(d.label, [round(float(v), 2) for v in d.bbox]) for d in inv],
       "refer": [(d.label, [round(float(v), 2) for v in d.bbox]) for d in ref],
       "parse_boxes": [(lab, b.tolist()) for lab, b in bd.parse_boxes(REPLY, 1280, 720)]}
json.dump(out, sys.stdout, indent=1, sort_keys=True, default=str)
print()
