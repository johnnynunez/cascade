"""B16 follow-up: why Newton camera frames go missing during arm motion.

Samples the bridge's frame-publication state at ~10 Hz while the arm streams a
descent/ascent, on one bridge. usage: frame_gap_probe.py <port> <engine> <out.json>
"""
import importlib.util
import json
import math
import sys
import threading
import time
import collections

import numpy as np

port, engine, out = int(sys.argv[1]), sys.argv[2], sys.argv[3]
sys.argv = ["physics_probe"]
spec = importlib.util.spec_from_file_location("pp", "scripts/physics_probe.py")
pp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pp)
from cascade.sim.bridge_client import BridgeClient  # noqa: E402

p = pp.Probe(port, engine)
p.ensure_playing()
p.discover_objects()
spot = pp.OBJECTS["pink_cube"][0]
tcp_z = pp.OBJECTS["pink_cube"][1]
R = pp._yaw_rotation(math.atan2(spot[1], spot[0]))
mon = BridgeClient(port=port, timeout_s=60.0)
mon.connect()
CODE = ("import json\n"
        "fh = _frame_history\n"
        "print(json.dumps({'epoch': _motion_clock_epoch, 'frames': sorted(_frames.keys()),"
        " 'errors': {k: v[:100] for k, v in _camera_frame_errors.items()},"
        " 'n': len(fh._entries), 'amb': len(fh._ambiguous),"
        " 'last_step': (fh._last or {}).get('physics_step'), 'pending': sorted(_pending_camera_publications),"
        " 'diag': _diag()}), flush=True)\n")
DIAG = ("def _diag():\n"
        "    from fractions import Fraction\n"
        "    sensor, _K = _annotators['wrist']\n"
        "    tok, ref, simt = _render_token(sensor)\n"
        "    res = _frame_history.resolve(reference=ref, simulation_time=simt, epoch=_motion_clock_epoch)\n"
        "    if res is not None:\n"
        "        return {'ok': True}\n"
        "    us = Fraction(ref[0] * 1000000 // ref[1], 1000000)\n"
        "    keys = sorted(_frame_history._entries.keys())\n"
        "    near = [k for k in keys if abs(float(k) - float(us)) < 0.2]\n"
        "    ent = {str(k): [_frame_history._entries[k]['reference'][0], _frame_history._entries[k]['simulation_time']] for k in near}\n"
        "    return {'ok': False, 'ref': list(ref), 'simt': simt, 'us_key': str(us), 'in_entries': us in _frame_history._entries,\n"
        "            'near': ent, 'last_key': str(keys[-1]) if keys else None}\n")
samples, stop, phase = [], threading.Event(), ["idle"]


def monitor():
    while not stop.is_set():
        r = mon.request({"op": "exec", "code": DIAG + CODE})
        d = json.loads(r.get("stdout", "").strip().splitlines()[-1])
        d["phase"], d["t"] = phase[0], time.monotonic()
        samples.append(d)
        time.sleep(0.1)


p.home()
assert p.goto(R, (spot[0], spot[1], tcp_z + 0.10), 2.5)
th = threading.Thread(target=monitor, daemon=True)
th.start()
time.sleep(3.0)
for leg, (z_to, dur) in enumerate(((0.03, 3.0), (0.10, 3.0), (0.03, 1.5), (0.10, 1.5))):
    phase[0] = f"moving{leg}"
    ik = p.kin.ik(pp.make_transform(R, np.array([spot[0], spot[1], tcp_z + z_to])), np.asarray(p.arm.get_state().q, float))
    p.arm.stream_to(ik.q, duration_s=dur)
    p.arm.wait_settled(ik.q, tol=0.03, timeout_s=8.0)
    phase[0] = f"idle{leg}"
    time.sleep(2.0)
stop.set()
th.join(timeout=5)
p.home()

by = collections.defaultdict(lambda: collections.Counter())
epochs = []
for s in samples:
    kind = "moving" if s["phase"].startswith("moving") else "idle"
    by[kind]["samples"] += 1
    by[kind]["no_frames"] += not s["frames"]
    for cam, e in s["errors"].items():
        by[kind][f"{cam}: {e}"] += 1
    if not epochs or epochs[-1] != s["epoch"]:
        epochs.append(s["epoch"])
summary = {k: dict(v) for k, v in by.items()}
summary["epoch_changes"] = len(epochs) - 1
summary["ambiguous_max"] = max(s["amb"] for s in samples)
fails = [s["diag"] for s in samples if not s["diag"].get("ok")]
summary["unresolved_now"] = len(fails)
summary["unresolved_examples"] = fails[:4]
json.dump({"engine": engine, "summary": summary, "samples": samples}, open(out, "w"), indent=1)
print(json.dumps({"engine": engine, "summary": summary}, indent=1))
