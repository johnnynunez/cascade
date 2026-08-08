"""Reproduce the no-filter ablation WITHOUT touching the repo config.

The doc cites precision 51.8% for "open vocabulary, no filters". That run's
JSON artifact was deleted, and a number in a research note must be backed by
a file, so this reproduces it via the sanctioned runtime-override idiom
(writing into cfg._data) rather than editing configs/demo.yaml.
"""

import importlib.util
import json
import sys
from pathlib import Path

REPO = Path("/home/spark/Projects/demo/wrc_demo")
sys.path.insert(0, str(REPO / "src"))

spec = importlib.util.spec_from_file_location(
    "eval_detector", REPO / "scripts" / "eval_detector.py"
)
ed = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ed)

from wrc_demo.apps import demo as demo_app
from wrc_demo.config import load_demo_config
from wrc_demo.sim.bridge_client import BridgeClient
from wrc_demo.sim.truth import TruthPoseReader

cfg = load_demo_config(cameras=["isaac", "isaac_side"], arm="isaac", llm="mock")
cfg._data.setdefault("stream", {})["mode"] = "off"

# Thresholds that reject nothing: the geometric gate becomes a no-op.
cfg._data["workspace_filter"] = {
    "base_radius_m": 0.0,
    "base_height_m": 0.0,
    "max_extent_m": 99.0,
    "min_extent_m": 0.0,
    "table_z_m": -99.0,
    "max_z_m": 99.0,
    "reach_m": 99.0,
    "max_frame_frac": 1.0,
    "min_height_m": -99.0,
}

client = BridgeClient(port=8611)
client.connect()

rt, arm = demo_app.build_runtime(cfg, Path("/tmp/ablate"), view=False, serve=False)
try:
    report = ed.evaluate(rt, TruthPoseReader(client), 12)
finally:
    demo_app.shutdown_runtime(rt, arm)

ed._print(report)
Path("benchmark/results/perception_nofilter.json").write_text(json.dumps(report, indent=2))
print("[+] wrote benchmark/results/perception_nofilter.json")
