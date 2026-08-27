"""Did the view cache actually stop the latency growth?

Before the fix, `stress_probe.py` measured probe latency climbing
20 ms -> ~85 ms over 400 calls on an IDLE scene, and a variant of the same
pattern eventually killed the bridge's TCP thread.

The fix caches RigidPrim views on the bridge's globals instead of building a
new one per prop per probe. This re-runs the same measurement and reports
first-vs-last latency. A flat curve means the leak is gone; a rising one
means the cache did not help and the experimental API is the next move.
"""
import statistics
import sys
import time

from cascade.sim.bridge_client import BridgeClient
from cascade.sim.truth import _PROBE, _PROP_ROOTS

c = BridgeClient(port=8611)
c.connect()
probe = _PROBE % {"roots": repr(_PROP_ROOTS)}

# clear any cache from a previous run so the numbers are comparable
c.request({"op": "exec", "code": (
    "globals().pop('_CASCADE_RIGIDPRIM_VIEWS', None)\nprint('cache cleared')\n")})

lat = []
N = 400
for i in range(N):
    t0 = time.monotonic()
    c.request({"op": "exec", "code": probe})
    lat.append((time.monotonic() - t0) * 1000)

first = statistics.median(lat[:25])
last = statistics.median(lat[-25:])
print(f"probes            : {N}")
print(f"median first 25   : {first:6.1f} ms")
print(f"median last 25    : {last:6.1f} ms")
print(f"growth            : {last/first:5.2f}x")
print()
if last / first < 1.5:
    print("PASS  latency is flat -- the view leak is fixed")
else:
    print("FAIL  latency still climbing; the cache did not address it")

r = c.request({"op": "exec", "code": (
    "v = globals().get('_CASCADE_RIGIDPRIM_VIEWS', {})\n"
    "print('cached views:', len(v), sorted(v)[:4])\n")})
print(r.get("stdout", "").strip())
