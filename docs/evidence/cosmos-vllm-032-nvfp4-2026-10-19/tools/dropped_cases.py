#!/usr/bin/env python3
"""The two smoke cases dropped on transformers 5.19.0, still failing (read-only).

    dropped_cases.py <export dir>   (run with the Cosmos venv python)
"""
import json
import sys
from types import MethodType, SimpleNamespace

import torch
from transformers import AutoProcessor
from transformers.models.cosmos3_edge import modeling_cosmos3_edge as m

out = {}
owner = m.Cosmos3EdgeModel
obj = SimpleNamespace(config=m.Cosmos3EdgeConfig())
obj.get_vision_position_ids = MethodType(owner.get_vision_position_ids, obj)
ids = torch.tensor([[100, 101, 102]])
try:
    owner.get_rope_index(obj, ids, torch.zeros_like(ids), video_grid_thw=torch.empty(0, dtype=torch.long))
    out["rope_1d_empty_video_grid"] = "ok"
except Exception as e:  # noqa: BLE001 - the failure is the result
    out["rope_1d_empty_video_grid"] = f"{type(e).__name__}: {e}"
p = AutoProcessor.from_pretrained(sys.argv[1], local_files_only=True)
v = p.video_processor(torch.zeros((1, 3, 32, 32), dtype=torch.uint8), return_tensors="pt", do_sample_frames=False, do_resize=False)
counts = p._get_num_multimodal_tokens(video_sizes=[(1, 32, 32)], videos_kwargs={"do_resize": False})
out["video_do_resize_false"] = {"counted_tokens": counts["num_video_tokens"], "actual_tokens": int(v["video_grid_thw"].prod()) // 4}
print(json.dumps(out))
