"""Ground-truth object poses from the Isaac Sim bridge (Pigey's best channel).

Pigey's closed loop needs to answer "did the object actually move?" from an
observation the actuator does not control.  In simulation there is a channel
strictly better than perception: the physics state itself.  Reading a prop's
world transform out of the USD stage is exact, costs no detector pass, and
cannot flicker the way YOLOE labels do (the repo's own ROADMAP lists sim
perception flakiness as an open campaign -- misses the banana on some boots,
label-flickers the soup can).

This is a *verification* channel only.  It is never fed to the planner or the
grasp pipeline: the agent still perceives the world through cameras exactly as
it would on the real rig, so nothing here leaks privileged state into the
policy.  It only adjudicates, after the fact, whether a claimed effect
happened -- the sim equivalent of a human checking the table.

On the real rig this simply returns None and ``PostconditionChecker`` falls
back to the belief store, so the same code path runs in both worlds.

Implementation note: the bridge's ``exec`` op returns **stdout only** (see
``_run_exec_jobs`` in scripts/isaac_bridge.py -- it redirects stdout into the
response and has no ``result`` channel).  So the probe prints JSON and we
parse it back.  That deliberately requires no bridge-side change, which means
it works against an already-running Isaac session instead of demanding a
restart.
"""

from __future__ import annotations

import json
import math
import re
import time

#: Prop container prims in the demo scene, most specific first.
_PROP_ROOTS = ("/World_Props", "/World/props", "/World")

#: Minimal ES->EN token map so the physics channel survives a Spanish command.
#: Only colours and prop nouns -- this is label matching, not translation.
_ES_EN = {
    "cubo": "cube", "caja": "box", "cuenco": "bowl", "bol": "bowl",
    "taza": "cup", "vaso": "cup", "plato": "plate", "botella": "bottle",
    "platano": "banana", "juguete": "toy", "bloque": "block", "papelera": "bin",
    "rosa": "pink", "verde": "green", "amarillo": "yellow", "amarilla": "yellow",
    "rojo": "red", "roja": "red", "azul": "blue", "naranja": "orange",
    "morado": "purple", "morada": "purple", "blanco": "white", "blanca": "white",
    "negro": "black", "negra": "black",
}

#: Probe executed on the sim main thread between steps.  Kept to pure reads:
#: authoring from here would race hydra (the bridge's own warning).
#:
#: CRITICAL: dynamic props must be read through ``RigidPrim``, NOT through
#: ``UsdGeom.Xformable.ComputeLocalToWorldTransform``.  PhysX keeps live body
#: poses in fabric and does not write them back to the USD attributes during
#: simulation, so the Xformable path returns the *authored* spawn transform --
#: verified on the live rig, where both cubes reported (0, 0, 0) while
#: RigidPrim correctly reported (0.17, 0.15, 0.04) and (0.30, 0.16, 0.04).
#: Reading the authored value would make every displacement check compare a
#: pose against itself and silently confirm nothing.
_PROBE = """
import json as _json
import omni.usd as _usd
from pxr import UsdGeom as _UsdGeom, Usd as _Usd, UsdPhysics as _UsdPhysics

_stage = _usd.get_context().get_stage()
_out = {}
_dynamic = []
for _prim in _stage.Traverse():
    _path = _prim.GetPath().pathString
    if not _path.startswith(%(roots)s):
        continue
    if _prim.GetTypeName() not in ("Mesh", "Xform"):
        continue
    _name = _path.rsplit("/", 1)[-1]
    if _prim.HasAPI(_UsdPhysics.RigidBodyAPI):
        _dynamic.append((_name, _path))
        continue
    try:
        _m = _UsdGeom.Xformable(_prim).ComputeLocalToWorldTransform(_Usd.TimeCode.Default())
        _t = _m.ExtractTranslation()
        _out[_name] = [round(float(_t[0]), 5), round(float(_t[1]), 5), round(float(_t[2]), 5)]
    except Exception:
        continue

if _dynamic:
    try:
        from isaacsim.core.prims import RigidPrim as _RigidPrim
        # Cache the views on the bridge's globals. Building a fresh RigidPrim
        # per prop on EVERY probe leaks views against the same prims -- over an
        # hour of verification polling that is thousands of them, and it
        # degrades the bridge: probe latency grew 20 ms -> 85 ms over 400 calls
        # and the TCP thread eventually died while the process stayed alive
        # (docs/BRIDGE_DEGRADATION.md). The views are stable for a given prim
        # path, so they are safe to reuse.
        _cache = globals().setdefault("_CASCADE_RIGIDPRIM_VIEWS", {})
        for _name, _path in _dynamic:
            try:
                _view = _cache.get(_path)
                if _view is None:
                    _view = _RigidPrim(_path)
                    _cache[_path] = _view
                _pos, _ = _view.get_world_poses()
                _out[_name] = [round(float(v), 5) for v in _pos[0]]
            except Exception:
                _cache.pop(_path, None)   # drop a stale view, retry next probe
                continue
    except Exception:
        pass

print("CASCADE_TRUTH_POSES " + _json.dumps(_out))
"""


def _normalize(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


#: Poses outside this half-extent (metres, base frame) are not physics, they
#: are a body read mid-explosion. Observed live: a prop reported
#: [-11.8, -10.6, -122.1] -- a 123 m "displacement" -- during an ablation run.
#: A verification channel that hands such a value to the checker turns a
#: numerical fault into a confident verdict, which is worse than no channel.
_SANE_RADIUS_M = 5.0


def _is_sane(xyz) -> bool:
    """Reject non-finite or physically impossible positions."""
    try:
        vals = [float(v) for v in xyz]
    except (TypeError, ValueError):
        return False
    if len(vals) != 3:
        return False
    return all(math.isfinite(v) and abs(v) <= _SANE_RADIUS_M for v in vals)


class TruthPoseReader:
    """Reads world-space prop positions from the running sim.

    Results are cached for ``ttl_s`` so a single verification pass costs one
    bridge round-trip even when several objects are checked.  Every failure
    mode (no bridge, sim closed, probe error) degrades to ``None``.
    """

    def __init__(self, client, ttl_s: float = 0.25, roots: tuple[str, ...] = _PROP_ROOTS):
        self._client = client
        self.ttl_s = float(ttl_s)
        self._roots = tuple(roots)
        self._cache: dict[str, list[float]] = {}
        self._cache_t = 0.0
        #: count of poses rejected as physically impossible (diagnostic: a
        #: non-zero value means PhysX handed us a body mid-explosion)
        self.rejected = 0

    # ── public API (matches PostconditionChecker's object_pose hook) ─────

    def __call__(self, label: str):
        return self.pose(label)

    def pose(self, label: str):
        """World position of the prop best matching ``label``, or None."""
        if not label:
            return None
        poses = self._poses()
        if not poses:
            return None
        want = _normalize(label)
        if want in poses:
            return poses[want]
        # token overlap: "pink cube" -> pink_cube, "the cube" -> pink_cube
        want_tokens = set(want.split("_"))
        # The user talks to this robot in whatever language they like, and the
        # sim prims are English. Without this, a Spanish command silently loses
        # the ONLY independent verification channel: "cubo rosa" matched no
        # prim, physics returned None, and the check fell back to the belief
        # the skill had just written -- confirming a cube that was 30 cm from
        # the bin (live rig, 2026-07-31).
        want_tokens |= {_ES_EN.get(t, t) for t in want_tokens}
        best, best_score = None, 0
        for key, xyz in poses.items():
            score = len(want_tokens & set(key.split("_")))
            if score > best_score:
                best, best_score = xyz, score
        # A single shared token is not identification: "pink cube" overlaps
        # "green_cube" on {cube}, so when the pink one is missing (dropped as
        # an impossible pose, or absent from the scene) this used to hand back
        # the GREEN cube's position and the checker would confirm against the
        # wrong object entirely. Demand either an exact key match (handled
        # above) or more than one shared token.
        if best_score < 2:
            return None
        return best

    def all_poses(self) -> dict:
        return dict(self._poses())

    # ── internals ────────────────────────────────────────────────────────

    def _poses(self) -> dict:
        now = time.monotonic()
        if self._cache and (now - self._cache_t) < self.ttl_s:
            return self._cache
        raw = self._probe()
        if raw is None:
            return self._cache  # keep the last good reading rather than lying
        # Drop insane poses INSTEAD of caching them: a body caught mid-solver
        # reports things like [-11.8, -10.6, -122.1], and passing that to the
        # checker converts a numerical fault into a confident verdict about a
        # 123 m "displacement". Silence is the correct answer here -- the
        # checker already degrades to the next channel when a pose is missing.
        clean = {}
        for k, v in raw.items():
            if _is_sane(v):
                clean[_normalize(k)] = v
            else:
                self.rejected += 1
        self._cache = clean
        self._cache_t = now
        return self._cache

    def _probe(self) -> dict | None:
        code = _PROBE % {"roots": repr(self._roots)}
        try:
            resp = self._client.request({"op": "exec", "code": code})
        except Exception:
            return None
        stdout = str(resp.get("stdout") or "")
        marker = "CASCADE_TRUTH_POSES "
        idx = stdout.rfind(marker)
        if idx < 0:
            return None
        try:
            return json.loads(stdout[idx + len(marker):].splitlines()[0])
        except (json.JSONDecodeError, IndexError):
            return None


def make_truth_pose_fn(arm):
    """Build a truth-pose callable from an arm backend, if it is a sim arm.

    Returns None for real hardware (and for mock arms), which makes
    ``PostconditionChecker`` fall back to perception -- the same behaviour the
    physical rig will have.

    PITFALL (cost a test regression): ``LazyArm.__getattr__`` materializes the
    real arm on any non-underscore attribute access, so a bare
    ``getattr(arm, "client", None)`` here powers up the CAN bus / motors as a
    side effect of *setting up verification*.  That is precisely what LazyArm
    exists to prevent.  Never probe an unmaterialized LazyArm: if it has not
    been used yet there is nothing to verify against anyway.
    """
    if type(arm).__name__ == "LazyArm" and not getattr(arm, "connected", False):
        return None
    client = arm.__dict__.get("client") or arm.__dict__.get("_client")
    if client is None or not hasattr(client, "request"):
        return None
    try:
        reader = TruthPoseReader(client)
        reader.all_poses()  # one probe now, so a broken wiring fails loudly here
        return reader
    except Exception:
        return None
