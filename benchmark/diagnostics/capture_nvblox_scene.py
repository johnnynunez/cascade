"""Read and archive the live, idle kitchen; never commands motion."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root / "src"))
from cascade.config import load_demo_config
from cascade.sim.bridge_client import BridgeClient

p = argparse.ArgumentParser()
p.add_argument("--port", type=int, default=18611)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
a.output.mkdir(exist_ok=False, parents=True)
b = BridgeClient(port=a.port)
b.connect()
metadata = {
    "scope": "Actual idle Isaac kitchen RGB-D captures; no motion commanded. Ground truth is used only to evaluate mapping.",
    "identity": b.request({"op": "ping"}),
    "cameras": {},
}
code = """
import json as _nv_json
from isaacsim.core.experimental.prims import RigidPrim as _NvRP
_nv_out={}
for _nv_name in ('green_cube','pink_cube','orange','lemon','tomato_can'):
    _nv_p = _NvRP('/World_Props/'+_nv_name)
    _nv_pos,_nv_q = _nv_p.get_world_poses()
    _nv_xyz = _nv_pos.numpy()[0].astype(float)
    _nv_xyz[2] -= BASE_Z
    _nv_out[_nv_name]={'position_m':_nv_xyz.tolist(),'quaternion_wxyz':_nv_q.numpy()[0].astype(float).tolist()}
print('NVBLOX_TRUTH '+_nv_json.dumps(_nv_out))
"""
r = b.request({"op": "exec", "code": code})
print("truth result keys", list(r))
out = r.get("stdout", r.get("output", ""))
if not isinstance(out, str):
    raise TypeError("unexpected bridge truth transport")
line = next(x for x in out.splitlines() if x.startswith("NVBLOX_TRUTH "))
metadata["truth"] = json.loads(line.split(" ", 1)[1])
for config_name, name in [
    ("isaac", "cam0"),
    ("isaac_side", "side"),
    ("isaac_proof", "proof"),
]:
    cfg = load_demo_config(camera=config_name)
    # Read the declared camera calibration; bridge serves moving poses only for wrist.
    cc = cfg.camera
    f = b.observation(name)
    assert f.robot_mask is not None and f.depth_m is not None
    T = (
        f.T_base_cam
        if f.T_base_cam is not None
        else np.asarray(cc.extrinsics.T, dtype=float)
    )
    path = a.output / (name + ".npz")
    np.savez_compressed(
        path, depth=f.depth_m, K=f.K, T=T, robot_mask=f.robot_mask, rgb=f.rgb
    )
    metadata["cameras"][name] = {
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "capture": f.capture,
        "shape": list(f.depth_m.shape),
        "masked_pixels": int(f.robot_mask.sum()),
    }
b.close()
(a.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
print(a.output)
