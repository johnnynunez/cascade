"""B16 attribution: is the green_cube probe-grasp failure the cube or the reach?

Swap the two cubes' positions (via the bridge's place_prop) and run the probe's
own t_grasp on each at the other's spot, on one bridge. Engine-agnostic.
usage: swap_grasp.py <port> <engine> <out.json>
"""
import importlib.util
import json
import sys
import time

port, engine, out = int(sys.argv[1]), sys.argv[2], sys.argv[3]
sys.argv = ["physics_probe"]
spec = importlib.util.spec_from_file_location("pp", "scripts/physics_probe.py")
pp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pp)

p = pp.Probe(port, engine)
p.ensure_playing()
live = p.discover_objects()
assert {"pink_cube", "green_cube"} <= set(live), live
pink0, green0 = pp.OBJECTS["pink_cube"][0], pp.OBJECTS["green_cube"][0]
park = (0.20, -0.20, pink0[2])          # out of the way, reachable, on the table
results = {"engine": engine, "spawns": {"pink_cube": pink0, "green_cube": green0}, "runs": []}

def run(name, where, label):
    p.home()
    other = "green_cube" if name == "pink_cube" else "pink_cube"
    p.teleport(other, park, settle_s=1.0)
    p.teleport(name, where, settle_s=1.5)
    r = p.t_grasp(name)
    r.update({"object": name, "at": list(where), "label": label})
    results["runs"].append(r)
    print(json.dumps(r), flush=True)

run("green_cube", pink0, "green at pink's spot (r=0.227 m)")
run("pink_cube", green0, "pink at green's spot (r=0.340 m)")
run("green_cube", green0, "green at its own spot (control)")
run("pink_cube", pink0, "pink at its own spot (control)")
p.restore()
time.sleep(1.0)
json.dump(results, open(out, "w"), indent=1)
print("report:", out)
