"""What does the prompt-free detector actually report on the Isaac scene?

eval_detector.py scores beliefs against physics truth but does not say which
labels the phantoms carry. That distinction decides whether 48% phantoms is a
detector failure or the scene genuinely containing more than 3 objects.
"""

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, "../src")

from wrc_demo.apps import demo as demo_app
from wrc_demo.config import load_demo_config
from wrc_demo.sim.bridge_client import BridgeClient
from wrc_demo.sim.truth import TruthPoseReader

cfg = load_demo_config(cameras=["isaac", "isaac_side"], arm="isaac", llm="mock")
cfg._data.setdefault("stream", {})["mode"] = "off"

client = BridgeClient(port=8611)
client.connect()
truth = TruthPoseReader(client)
print("PHYSICS TRUTH:")
for k, v in truth.all_poses().items():
    print(f"   {k:26} {[round(float(x), 3) for x in v[:3]]}")

rt, arm = demo_app.build_runtime(cfg, Path("/tmp/labels"), view=False, serve=False)
try:
    frame = rt.observe()
    dets = rt.detector.detect(frame, classes=None)
    print(f"\nRAW DETECTIONS on one frame: {len(dets)}")
    for d in sorted(dets, key=lambda x: -x.conf):
        print(f"   {d.label:28} conf={d.conf:.3f}")
    print("\nlabel histogram:", Counter(d.label for d in dets).most_common())

    rt.execute("get_observation", {})
    seen = [o for o in rt.beliefs.all() if getattr(o, "visible", True)]
    print(f"\nBELIEFS AFTER get_observation: {len(seen)}")
    for o in seen:
        print(f"   {o.label:28} {[round(float(x), 3) for x in o.position[:3]]}")
finally:
    demo_app.shutdown_runtime(rt, arm)
