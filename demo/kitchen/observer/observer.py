"""Passive Isaac evidence collection. Importing this module never contacts a bridge."""
from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import textwrap
import time

ROOT = Path(__file__).resolve().parents[3]
PROPS = ("pink_cube", "green_cube", "tomato_can", "lemon")
ALLOWED_PROPS = (*PROPS, "orange")
MARKER = "KITCHEN_OBSERVER "
_codec_spec = importlib.util.spec_from_file_location("cascade_observer_codec",
    ROOT / "demo/kitchen/physics/snapshot_codec.py")
_codec = importlib.util.module_from_spec(_codec_spec)
_codec_spec.loader.exec_module(_codec)


def build_snapshot_code(props=PROPS, camera="cam0"):
    """Return read-only code for the bridge's existing main-loop exec queue.

    Existing rigid-body schemas are required before constructing read views;
    no physics schema, mass, transform, pose, drive, or target is authored.
    """
    if (not isinstance(props, (list, tuple)) or not props or len(props) > len(ALLOWED_PROPS)
            or any(not isinstance(p, str) or p not in ALLOWED_PROPS for p in props)
            or len(set(props)) != len(props)):
        raise ValueError("Expected one to five unique named kitchen props")
    if camera not in ("cam0", "side", "wrist", "proof"):
        raise ValueError("Unknown bridge camera")
    source = '''
import json as _obs_json, time as _obs_time, hashlib as _obs_hash, base64 as _obs_base64
import numpy as _obs_np
from pxr import UsdPhysics as _obs_UP, PhysxSchema as _obs_PX
from isaacsim.core.experimental.prims import RigidPrim as _obs_RP
from isaacsim.core.experimental.utils.backend import use_backend as _obs_backend
from isaacsim.core.simulation_manager import SimulationManager as _obs_SM
def _obs_array(value):
    return _obs_np.asarray(value.numpy() if hasattr(value, "numpy") else value, dtype=float)
def _obs_collect():
    if not _tl.is_playing():
        raise RuntimeError("Observer requires playing physics; USD fallback is forbidden")
    _obs_engine = str(engine).lower()
    if _obs_engine == "newton":
        from isaacsim.physics.newton import acquire_stage as _obs_acquire
        _obs_stage = _obs_acquire()
        if _obs_stage is None or not _obs_stage.initialized:
            raise RuntimeError("Newton authoritative stage is unavailable")
        _obs_st = float(_obs_stage.sim_time)
        _obs_step = int(_obs_stage.simulation_step_count)
        _obs_clock = "newton_stage"
    elif _obs_engine == "physx":
        _obs_st = float(_obs_SM.get_simulation_time())
        _obs_step = int(_obs_SM.get_num_physics_steps())
        _obs_clock = "SimulationManager"
    else:
        raise RuntimeError("Unrecognized physics engine: " + _obs_engine)
    _obs_paths = ["/World_Props/" + name for name in PROPS_LITERAL]
    for _obs_path in _obs_paths:
        _obs_prim = stage.GetPrimAtPath(_obs_path)
        if not _obs_prim or not _obs_prim.HasAPI(_obs_UP.RigidBodyAPI) or not _obs_prim.HasAPI(_obs_PX.PhysxRigidBodyAPI):
            raise RuntimeError("Observer refuses to add missing rigid-body schemas: " + _obs_path)
    _obs_cache = globals().setdefault("_kitchen_passive_observer_views_v1", {})
    # USD path resolution in the constructor does not support the tensor
    # context. Resolve existing views first; force tensor only for state reads.
    for _obs_path in _obs_paths:
        _obs_view = _obs_cache.get(_obs_path)
        if _obs_view is None or not _obs_view.is_physics_tensor_entity_valid():
            _obs_view = _obs_RP(_obs_path, reset_xform_op_properties=False)
            _obs_cache[_obs_path] = _obs_view
    with _obs_backend("tensor", raise_on_unsupported=True, raise_on_fallback=True):
        if not art.is_physics_tensor_entity_valid():
            raise RuntimeError("Articulation tensor view is invalid")
        _obs_q = _obs_array(art.get_dof_positions()).reshape(-1)
        _obs_dq = _obs_array(art.get_dof_velocities()).reshape(-1)
        _obs_lo, _obs_hi = [_obs_array(x).reshape(-1) for x in art.get_dof_limits()]
        if len(GRIP_IDX) != 2:
            raise RuntimeError("Expected two actual gripper joints")
        _obs_gi = list(GRIP_IDX)
        _obs_frac = [float((_obs_q[i]-_obs_lo[i])/(_obs_hi[i]-_obs_lo[i])) for i in _obs_gi]
        _obs_result = {"version":1,"channel":"physics_tensor","engine":_obs_engine,
            "sim_time":_obs_st,"physics_step":_obs_step,"physics_clock":_obs_clock,
            "bridge_loop_step":int(step),"server_monotonic":_obs_time.monotonic(),
            "robot_id":str(args.prim),"base_z":float(BASE_Z),"pose_frame":"robot_base_translation_world_axes",
            "joint_names":list(names),"q":_obs_q.tolist(),"dq":_obs_dq.tolist(),
            "joint_lower":_obs_lo.tolist(),"joint_upper":_obs_hi.tolist(),
            "gripper":{"indices":_obs_gi,"q":_obs_q[_obs_gi].tolist(),"qd":_obs_dq[_obs_gi].tolist(),
                "lower":_obs_lo[_obs_gi].tolist(),"upper":_obs_hi[_obs_gi].tolist(),
                "open_fractions":_obs_frac,"open_fraction":float(_obs_np.mean(_obs_frac))},"props":{}}
        for _obs_name,_obs_path in zip(PROPS_LITERAL,_obs_paths):
            _obs_view = _obs_cache[_obs_path]
            if not _obs_view.is_physics_tensor_entity_valid():
                raise RuntimeError("Rigid-body tensor view is invalid: " + _obs_path)
            _obs_pos,_obs_quat = _obs_view.get_world_poses()
            _obs_vel,_obs_ang = _obs_view.get_velocities()
            _obs_p = _obs_array(_obs_pos).reshape(3).copy()
            _obs_p[2] -= float(BASE_Z)
            _obs_result["props"][_obs_name] = {"position_m":_obs_p.tolist(),
                "orientation_wxyz":_obs_array(_obs_quat).reshape(4).tolist(),
                "linear_velocity_m_s":_obs_array(_obs_vel).reshape(3).tolist(),
                "angular_velocity_rad_s":_obs_array(_obs_ang).reshape(3).tolist(),
                "tensor_device":str(getattr(_obs_pos,"device","unknown"))}
    _obs_frame = _frames.get(CAMERA_LITERAL)
    if _obs_frame is None:
        _obs_result["frame"] = {"camera":CAMERA_LITERAL,"available":False}
    else:
        _obs_capture = _obs_frame.get("proprioception") or {}
        _obs_result["frame"] = {"camera":CAMERA_LITERAL,"available":True,
            "capture_monotonic":_obs_frame.get("t"),"robot_id":_obs_capture.get("robot_id"),
            "producer_q":_obs_capture.get("q"),"producer_time_source":_obs_capture.get("time_source"),
            "jpeg_sha256":_obs_hash.sha256(_obs_base64.b64decode(_obs_frame["rgb_jpeg_b64"])).hexdigest()}
    return _obs_result
_obs_payload = _obs_json.dumps(_obs_collect(), separators=(",",":"), allow_nan=False)
if len((_obs_payload + MARKER_LITERAL).encode("utf-8")) >= 7400:
    raise RuntimeError("Observer snapshot exceeds safe bridge stdout size")
print(MARKER_LITERAL + _obs_payload)
'''
    return textwrap.dedent(source).replace("PROPS_LITERAL", repr(tuple(props))).replace(
        "CAMERA_LITERAL", repr(camera)).replace("MARKER_LITERAL", repr(MARKER))


def unique_json_object(pairs):
    """Reject duplicate records before JSON decoding can overwrite evidence."""
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate observer/configuration JSON key: " + key)
        result[key] = value
    return result


def parse_snapshot_reply(reply):
    if not reply.get("ok"):
        raise RuntimeError(reply.get("error", "Bridge observation failed"))
    stdout = reply.get("stdout", "")
    value = _codec.expand_geometry(json.loads(_codec.snapshot_payload(stdout), object_pairs_hook=unique_json_object))
    if value.get("version") != 1 or value.get("channel") != "physics_tensor":
        raise ValueError("Unexpected observer schema/channel")
    return value


def _json(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def _inside(path):
    resolved = Path(path).resolve()
    if not resolved.is_relative_to(ROOT):
        raise ValueError("Observer artifacts and transcript inputs must stay inside this checkout")
    return resolved


def observe(*, out, duration_s=120, interval_s=.25, host="127.0.0.1", port=8611,
            camera="cam0", transcript=None, order_text=None, timeout_s=5):
    """Bounded passive collection. This is the only function that connects live."""
    from cascade.sim.bridge_client import BridgeClient

    if not 0 < duration_s <= 3600 or not .05 <= interval_s <= 10 or not 0 < timeout_s <= 15:
        raise ValueError("Invalid finite observation budget")
    out = _inside(out)
    out.mkdir(parents=True, exist_ok=False)
    transcript = _inside(transcript) if transcript else None
    code = build_snapshot_code(camera=camera)
    (out / "snapshot-code.py").write_text(code)
    code_sha = hashlib.sha256(code.encode()).hexdigest()
    meta = {"version":1,"kind":"passive physics observer; no actuator commands", "argv":sys.argv,
            "host":host,"port":port,"camera":camera,"duration_s":duration_s,"interval_s":interval_s,
            "order_text_metadata_only":order_text,"transcript_input":str(transcript) if transcript else None,
            "code_sha256":code_sha,"observer_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "started_unix":time.time(),"complete":False}
    _json(out / "metadata.json",meta)
    client = BridgeClient(host=host,port=port,timeout_s=timeout_s)
    count = 0
    first = last = None
    start = time.monotonic()
    requests = (out / "requests.jsonl").open("w")
    wires = (out / "wire.jsonl").open("w")
    samples = (out / "samples.jsonl").open("w")
    def request(payload):
        began = time.monotonic()
        logged = {k:v for k,v in payload.items() if k != "code"}
        if "code" in payload:
            logged["code_sha256"] = code_sha
        try:
            result = client.request(payload,timeout_s=timeout_s)
        except Exception as exc:
            requests.write(json.dumps({"request":logged,"started_monotonic":began,"error":repr(exc)})+"\n")
            requests.flush()
            raise
        requests.write(json.dumps({"request":logged,"started_monotonic":began,"ended_monotonic":time.monotonic(),
            "response_sha256":hashlib.sha256(json.dumps(result,sort_keys=True).encode()).hexdigest()})+"\n")
        requests.flush()
        return result
    def frame(tag):
        result=request({"op":"frame","camera":camera})
        jpeg=base64.b64decode(result["rgb_jpeg_b64"],validate=True)
        (out/f"{tag}-{camera}.jpg").write_bytes(jpeg)
        description={k:v for k,v in result.items() if k not in ("rgb_jpeg_b64","depth_z_b64","robot_pixel_mask")}
        description["jpeg_sha256"]=hashlib.sha256(jpeg).hexdigest()
        _json(out/f"{tag}-frame.json",description)
        return description
    try:
        client.connect()
        meta["ping_before"]=request({"op":"ping"})
        meta["frame_before"]=frame("before")
        next_sample = time.monotonic()
        while time.monotonic()-start < duration_s:
            began=time.monotonic()
            reply=request({"op":"exec","code":code})
            wires.write(json.dumps({"sequence":count,"reply":reply})+"\n");wires.flush()
            physics=parse_snapshot_reply(reply)
            row={"sequence":count,"client_started_monotonic":began,
                 "client_finished_monotonic":time.monotonic(),"physics":physics}
            samples.write(json.dumps(row,allow_nan=False)+"\n");samples.flush()
            if first is None:
                first=row;_json(out/"before-physics.json",row)
            last=row;_json(out/"after-physics.json",row)
            count+=1
            next_sample+=interval_s
            delay=min(next_sample-time.monotonic(),start+duration_s-time.monotonic())
            if delay>0:time.sleep(delay)
        meta["frame_after"]=frame("after")
        meta["ping_after"]=request({"op":"ping"})
        meta["complete"]=bool(first and last)
    except Exception as exc:
        meta["error"]=repr(exc)
        raise
    finally:
        client.close()
        for handle in (requests,wires,samples):handle.close()
        meta.update(samples=count,ended_unix=time.time(),wall_duration_s=time.monotonic()-start)
        if transcript and transcript.is_file():
            raw=transcript.read_bytes()
            (out/"actuation-transcript.txt").write_bytes(raw)
            meta["transcript_sha256"]=hashlib.sha256(raw).hexdigest()
        _json(out/"metadata.json",meta)
    return meta


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",required=True);p.add_argument("--duration",type=float,default=120)
    p.add_argument("--interval",type=float,default=.25);p.add_argument("--timeout",type=float,default=5)
    p.add_argument("--host",default="127.0.0.1");p.add_argument("--port",type=int,default=8611)
    p.add_argument("--camera",choices=("cam0","side","wrist","proof"),default="cam0")
    p.add_argument("--transcript");p.add_argument("--order-text")
    args=p.parse_args()
    result=observe(out=args.out,duration_s=args.duration,interval_s=args.interval,timeout_s=args.timeout,
                   host=args.host,port=args.port,camera=args.camera,transcript=args.transcript,order_text=args.order_text)
    print(json.dumps({"complete":result["complete"],"samples":result["samples"]}))
    return 0 if result["complete"] else 1


if __name__=="__main__":raise SystemExit(main())
