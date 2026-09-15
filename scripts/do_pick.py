"""Execute a REAL pick-and-place (pink cube -> box) via the runtime directly,
so it doesn't depend on the flaky reflex/LLM path. Prints verified result."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cascade.config import load_demo_config
from cascade.apps.demo import build_runtime
from cascade.sim.bridge_client import BridgeClient

cfg = load_demo_config(cameras=["isaac"], arm="isaac", llm="mock")
rt, _ = build_runtime(cfg, Path("/tmp/pickrun"), view=False)
bc = BridgeClient(port=8611); bc.connect()
bc.request({"op": "reset_props"})
time.sleep(2.0)   # let the recorder capture the initial scene

print(">>> pick_and_place: pink cube -> box", flush=True)
res = rt.skill_pick_and_place("pink cube", "box")
print(">>> result:", {k: res.get(k) for k in ("ok", "stage", "error")}, flush=True)

# verify physically where the cube ended up
code = ('from isaacsim.core.experimental.prims import RigidPrim; '
        'p=RigidPrim("/World_Props/pink_cube").get_world_poses()[0].numpy().reshape(-1)[:3]; '
        'print(f"CUBE FINAL: ({p[0]:.3f},{p[1]:.3f},{p[2]:.3f})")')
r = bc.request({"op": "exec", "code": code})
print(">>>", (r.get("stdout") or r.get("error") or "").strip(), flush=True)
time.sleep(2.0)
bc.close()
