"""Systematic grasp diagnosis for the trophy, via build_runtime()."""
import sys
from pathlib import Path
sys.path.insert(0, "/home/johnny/Projects/demo/wrc_demo_moon/src")
import numpy as np
from wrc_demo.config import load_demo_config
from wrc_demo.apps.demo import build_runtime

cfg = load_demo_config(cameras=["isaac"], arm="isaac", llm="mock")
rt, _ = build_runtime(cfg, Path("/tmp/diag_run"), view=False)


from wrc_demo.sim.bridge_client import BridgeClient
_probe_bc = BridgeClient(port=8611); _probe_bc.connect()


def phys(label="flag_pole"):
    code = (f'from isaacsim.core.experimental.prims import RigidPrim; '
            f'p=RigidPrim("/World_Props/{label}").get_world_poses()[0].numpy().reshape(-1)[:3]; '
            f'print(f"{{p[0]:.3f}},{{p[1]:.3f}},{{p[2]:.3f}}")')
    try:
        r = _probe_bc.request({"op": "exec", "code": code})
        return (r.get("stdout") or r.get("error") or "").strip()
    except Exception as e:
        return f"(phys err {e})"


print("=== reset ===")
try:
    rt.arm.raw.bridge.request({"op": "reset_props"})
except Exception as e:
    print("reset err", e)
print("cube phys before:", phys())

print("\n=== 1. LOCALIZE ===")
frame, fix = rt._localize("cube")
print(f"  perceived position: {[round(float(x),3) for x in fix.position]}")
print(f"  perceived extent:   {[round(float(x),3) for x in fix.extent]}")
print(f"  n_points:           {fix.points.shape[0]}")
print(f"  z range of points:  {round(float(fix.points[:,2].min()),3)} .. {round(float(fix.points[:,2].max()),3)}")

print("\n=== 2. PLAN GRASPS ===")
grasps = rt._plan_grasps(fix, label="cube")
print(f"  n candidates: {len(grasps)}")
for i, g in enumerate(grasps[:5]):
    ap = np.asarray(g.approach, dtype=float)
    print(f"  [{i}] pos={[round(float(x),3) for x in g.position]} "
          f"appr={[round(float(x),2) for x in ap]} "
          f"width={round(float(g.width_m),3)} qual={round(float(getattr(g,'quality',0)),2)}")

print("\n=== 3. EXECUTE GRASP ===")
res = rt.skill_grasp_object("cube")
print("  grasp result:", {k: res.get(k) for k in ("ok", "held", "error")})
print("  cube phys after:", phys())
print("  arm holding:", rt.held_object)
