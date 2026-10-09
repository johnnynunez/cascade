"""Golden scenario for the B40 size gate's OFF switch (`memory.size_gate: false`).

`BeliefStore(size_gate=False)` must be byte-identical to the store before B40.
This module replays one fixed scenario -- the side camera's "yellow" view of the
bin next to a yellow prop inside it (the live defect), the bin's two live views
(B32b fixture), same-colour twins (B31), single-observation `update()` calls
and the per-detection path -- into a store and dumps everything the store
decided, exactly (`json.dumps` keeps every float's repr).

The INPUTS are stored, not recomputed (`b40_size_gate_golden/inputs.npz`):
the ray-cast clouds and their OBB centres/extents are computed once, so the
golden does not depend on a platform's BLAS/LAPACK last bits. The OUTPUT
(`golden.json`) was written by the pre-B40 store; `receipt.json` says which.

Regenerate ONLY on a pre-B40 tree (the script refuses otherwise):

    git archive <pre-B40 commit> | tar -x -C /tmp/old
    env -u PYTHONPATH PYTHONPATH=/tmp/old/src python tests/fusion_size_gate_fixture.py \
        --write --source <pre-B40 commit>
"""
from __future__ import annotations

import hashlib
import inspect
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DIR = HERE / "fixtures" / "b40_size_gate_golden"
INPUTS = DIR / "inputs.npz"
GOLDEN = DIR / "golden.json"
RECEIPT = DIR / "receipt.json"
LIVE = HERE / "fixtures" / "b32b_live_bin_clouds" / "clouds.npz"

#: the store configurations the golden covers (the size gate's switch is added
#: by the caller); B32b one-name rule and B31 per-detection baselines included
CONFIGS = {
    "default": {},
    "one_name_rule": {"per_camera_colour": False},
    "per_detection": {"instance_association": False},
}

#: (kind, t, observations); kind "frame" = update_frame, "update" = update().
#: An observation is (key, label, conf, colour, source); its arrays live in
#: inputs.npz as <key>/points (None if absent), <key>/position, <key>/extent,
#: <key>/top_z.
SCENARIO = [
    ("frame", 1.0, [("top_bin", "storage box", 0.80, "orange", "isaac"),
                    ("top_prop", "cube", 0.85, "yellow", "isaac")]),
    # the defect: the side camera sees only the bin, names it "yellow"
    ("frame", 1.5, [("side_bin", "building block", 0.70, "yellow", "isaac_side")]),
    ("frame", 2.0, [("top_bin", "storage box", 0.80, "orange", "isaac"),
                    ("top_prop", "cube", 0.85, "yellow", "isaac")]),
    ("frame", 2.5, [("side_bin", "box", 0.72, "yellow", "isaac_side"),
                    ("side_prop", "passbook", 0.40, "yellow", "isaac_side")]),
    ("frame", 3.0, [("side_bin", "building block", 0.70, "yellow", "isaac_side")]),
    # a localize-style single observation, no camera, no cloud
    ("update", 3.2, [("top_prop_nocloud", "cube", 0.90, "yellow", None)]),
    # the bin's live views (B32b fixture pair 1, in_bin_centre scene)
    ("frame", 3.5, [("live_p1_belief", "gift box", 0.80, "orange", "isaac")]),
    ("frame", 4.0, [("live_p1_obs", "building block", 0.70, "yellow", "isaac_side")]),
    # same-colour twins 5 cm apart, both cameras (B31)
    ("frame", 5.0, [("top_twin1", "cube", 0.90, "red", "isaac"),
                    ("top_twin2", "cube", 0.88, "red", "isaac")]),
    ("frame", 5.5, [("side_twin1", "cube", 0.86, "red", "isaac_side"),
                    ("side_twin2", "cube", 0.87, "red", "isaac_side")]),
    # a single observation WITH a cloud: the bin view through update()
    ("update", 6.0, [("side_bin", "building block", 0.70, "yellow", "isaac_side")]),
    ("frame", 6.5, [("top_bin", "storage box", 0.80, "orange", "isaac"),
                    ("top_prop", "cube", 0.85, "yellow", "isaac")]),
]


def load_inputs() -> dict[str, np.ndarray]:
    z = np.load(INPUTS)
    return {k: z[k] for k in z.files}


def _observation(arrays, key, label, conf, colour, source):
    from cascade.memory.beliefs import FrameObservation

    pts = arrays.get(f"{key}/points")
    return FrameObservation(label, arrays[f"{key}/position"].copy(), conf,
                            extent=arrays[f"{key}/extent"].copy(),
                            top_z=float(arrays[f"{key}/top_z"]), color=colour,
                            points=None if pts is None else pts.copy(), source=source)


def replay(store, arrays) -> list[list[int]]:
    """Feed SCENARIO into `store`; returns, per step, the index (creation
    order) of the belief each observation went to."""
    order: list = []
    out = []
    for kind, t, observations in SCENARIO:
        obs = [_observation(arrays, *o) for o in observations]
        if kind == "frame":
            beliefs = store.update_frame(obs, t=t)
        else:
            beliefs = [store.update(o.label, o.position, o.conf, extent=o.extent, top_z=o.top_z,
                                    t=t, color=o.color, points=o.points, source=o.source)
                       for o in obs]
        step = []
        for b in beliefs:
            idx = next((i for i, x in enumerate(order) if x is b), None)
            if idx is None:
                order.append(b)
                idx = len(order) - 1
            step.append(idx)
        out.append(step)
    return out


def _floats(a):
    return None if a is None else [float(v) for v in np.asarray(a, dtype=float).reshape(-1)]


def canonical(store, steps) -> dict:
    """Everything the store decided, in creation order, floats exact."""
    beliefs = []
    for b in store.all(now=100.0):
        pts = None if b.points is None else np.ascontiguousarray(np.asarray(b.points, dtype=np.float32))
        beliefs.append({
            "label": b.label, "aliases": sorted(b.aliases), "color": b.color,
            "source_colors": sorted(b.source_colors.items()),
            "position": _floats(b.position), "extent": _floats(b.extent),
            "top_z": None if b.top_z is None else float(b.top_z), "conf": float(b.conf),
            "observations": int(b.observations),
            "first_seen_t": float(b.first_seen_t), "last_seen_t": float(b.last_seen_t),
            "points_sha256": None if pts is None else hashlib.sha256(pts.tobytes()).hexdigest(),
            "points_n": None if pts is None else int(pts.shape[0]),
        })
    return {"steps": steps, "beliefs": beliefs}


def dump(store_factory, arrays) -> dict:
    out = {}
    for name, kw in CONFIGS.items():
        store = store_factory(**kw)
        out[name] = canonical(store, replay(store, arrays))
    return out


def as_text(doc) -> str:
    return json.dumps(doc, sort_keys=True, indent=1)


# ── (re)generation, pre-B40 trees only ─────────────────────────────────────


def _make_inputs() -> dict[str, np.ndarray]:
    """Ray-cast the live in-bin scene (bridge prop 5 x 5 x 8 cm where the live
    run had it, (0.1833, -0.1584)) and twins, both calibrated cameras, exact
    masks at 320 x 180 (tests/test_colour_identity_beliefs.py), each cloud cut
    to <= 384 points the way the store remembers it."""
    sys.path.insert(0, str(HERE))
    import test_colour_identity_beliefs as tcb
    from cascade.memory.beliefs import BeliefStore
    from cascade.perception.grounding import oriented_bbox

    arrays: dict[str, np.ndarray] = {}

    def put(key, pts, keep_points=True):
        pts = np.asarray(pts, dtype=float)
        c, e, _ = oriented_bbox(pts)
        arrays[f"{key}/position"] = np.asarray(c, dtype=float)
        arrays[f"{key}/extent"] = np.asarray(e, dtype=float)
        arrays[f"{key}/top_z"] = np.asarray(float(pts[:, 2].max()))
        if keep_points:
            arrays[f"{key}/points"] = np.asarray(BeliefStore._remembered_cloud(pts), dtype=np.float32)

    prop = [((0.1833, -0.1584, 0.04), (0.05, 0.05, 0.08))]
    for cam, tag in (("isaac", "top"), ("isaac_side", "side")):
        frame, masks = tcb._render(cam, {"bin": tcb.BIN, "prop": prop})
        put(f"{tag}_bin", tcb._cloud(cam, frame, masks["bin"]))
        put(f"{tag}_prop", tcb._cloud(cam, frame, masks["prop"]))
        frame, masks = tcb._render(cam, {"t1": tcb._cube(0.24, 0.04), "t2": tcb._cube(0.24, 0.09)})
        put(f"{tag}_twin1", tcb._cloud(cam, frame, masks["t1"]))
        put(f"{tag}_twin2", tcb._cloud(cam, frame, masks["t2"]))
    frame, masks = tcb._render("isaac", {"bin": tcb.BIN, "prop": prop})
    put("top_prop_nocloud", tcb._cloud("isaac", frame, masks["prop"]), keep_points=False)
    live = np.load(LIVE)
    for key in ("p1_obs", "p1_belief"):
        put(f"live_{key}", live[key])
    return arrays


def main() -> int:
    from cascade.memory.beliefs import BeliefStore

    if "size_gate" in inspect.signature(BeliefStore).parameters:
        print("refusing: this tree already has the B40 size gate; the golden must come "
              "from a pre-B40 store", file=sys.stderr)
        return 2
    if "--write" not in sys.argv or "--source" not in sys.argv:
        print(f"{__doc__}\n    (add --source '<the pre-B40 commit the store came from>')")
        return 1
    import cascade

    source = sys.argv[sys.argv.index("--source") + 1]
    DIR.mkdir(parents=True, exist_ok=True)
    arrays = _make_inputs()
    np.savez_compressed(INPUTS, **arrays)
    arrays = load_inputs()  # replay from what was written, exactly as the test does
    text = as_text(dump(BeliefStore, arrays))
    GOLDEN.write_text(text)
    RECEIPT.write_text(json.dumps({
        "what": "BeliefStore decisions on SCENARIO (tests/fusion_size_gate_fixture.py), "
                "written by the pre-B40 store; B40's size_gate=False must reproduce them byte for byte",
        "store_source": source,
        "command": "env -u PYTHONPATH PYTHONPATH=<git archive of store_source>/src "
                   "python tests/fusion_size_gate_fixture.py --write --source <store_source>",
        "inputs_npz_sha256": hashlib.sha256(INPUTS.read_bytes()).hexdigest(),
        "golden_json_sha256": hashlib.sha256(GOLDEN.read_bytes()).hexdigest(),
    }, indent=1) + "\n")
    print(f"wrote {INPUTS.name}, {GOLDEN.name}, {RECEIPT.name} from {cascade.__file__}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
