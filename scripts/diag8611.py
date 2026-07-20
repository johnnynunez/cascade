"""Clean grasp diagnosis for the ORIGINAL wrc_demo scene (port 8611)."""
import sys
from pathlib import Path
sys.path.insert(0, "/home/johnny/Projects/demo/wrc_demo/src")
from wrc_demo.config import load_demo_config
from wrc_demo.apps.demo import build_runtime
from wrc_demo.sim.bridge_client import BridgeClient

cfg = load_demo_config(cameras=["isaac"], arm="isaac", llm="mock")
rt, _ = build_runtime(cfg, Path("/tmp/diag8611"), view=False)
bc = BridgeClient(port=8611); bc.connect()
bc.request({"op": "reset_props"})
import time; time.sleep(1.5)

print("=== 1. OBSERVATION ===")
obs = rt.skill_get_observation()
import json
vis = obs.get("objects_visible", [])
print(json.dumps(vis, indent=1, default=str)[:800])

print("\n=== 2. LOCALIZE pink cube ===")
for label in ("pink cube", "cube", "pink object"):
    try:
        frame, fix = rt._localize(label)
        print(f"  '{label}': pos={[round(float(x),3) for x in fix.position]} extent={[round(float(x),3) for x in fix.extent]}")
        break
    except Exception as e:
        print(f"  '{label}': FAILED {str(e)[:50]}")

print("\n=== 3. GRASP ===")
try:
    res = rt.skill_grasp_object(label)
    print("  result:", {k: res.get(k) for k in ("ok", "held", "error")})
    print("  arm holding:", rt.held_object)
except Exception as e:
    print("  grasp EXCEPTION:", str(e)[:80])
bc.close()
