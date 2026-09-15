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
        return _match_label(label, poses)

    def all_poses(self) -> dict:
        return dict(self._poses())

    # ── internals ────────────────────────────────────────────────────────

    def _poses(self) -> dict:
        now = time.monotonic()
        if self._cache and (now - self._cache_t) < self.ttl_s:
            return self._cache
        raw = self._probe()
        if raw is None:
            # The TTL-valid cache already returned above. A failed fresh probe
            # cannot extend an expired pose's lifetime as current physics.
            self._cache = {}
            self._cache_t = 0.0
            return {}
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


class MujocoTruthReader:
    """Prop poses straight out of a live MuJoCo world (`sim/mujoco_world.py`).

    The MuJoCo twin of ``TruthPoseReader``: same call surface, same label
    matching (exact normalized name, else >= 2 shared tokens with the ES->EN
    map), same sanity gate. Reads ``data.xpos`` of every free-jointed body
    under the world's lock, so a read never sees a half-stepped state.

    Deliberately holds NO reference that keeps the world alive: it looks the
    world up by path on every read (``mujoco_world.peek``). Verification must
    never be the thing that owns a physics world -- if the arm disconnects
    and the world is dropped, this reader reports None and the checker
    degrades to the next channel, exactly like a closed Isaac bridge.
    """

    def __init__(self, scene_path: str, ttl_s: float = 0.0):
        self._scene = str(scene_path)
        self.ttl_s = float(ttl_s)
        self.rejected = 0
        self._cache: dict[str, list[float]] = {}
        self._cache_t = 0.0

    def __call__(self, label: str):
        return self.pose(label)

    def pose(self, label: str):
        if not label:
            return None
        poses = self._poses()
        if not poses:
            return None
        return _match_label(label, poses)

    def all_poses(self) -> dict:
        return dict(self._poses())

    @property
    def live(self) -> bool:
        from . import mujoco_world

        return mujoco_world.peek(self._scene) is not None

    def _poses(self) -> dict:
        from . import mujoco_world

        now = time.monotonic()
        if self.ttl_s > 0 and self._cache and (now - self._cache_t) < self.ttl_s:
            return self._cache
        world = mujoco_world.peek(self._scene)
        if world is None:
            return {}
        clean = {}
        for name in world.free_body_names():
            xyz = world.body_pos(name)
            if xyz is None:
                continue
            if _is_sane(xyz):
                clean[_normalize(name)] = xyz
            else:
                self.rejected += 1
        self._cache, self._cache_t = clean, now
        return clean


#: Query words that name nothing in particular. "the red object" carries one
#: bit of identity (red); "object"/"thing"/"cube-shaped item" add none.
_GENERIC = {"object", "objects", "thing", "things", "item", "items", "the", "a",
            "an", "one", "this", "that", "el", "la", "los", "las", "objeto", "cosa"}

#: Colour tokens, post ES->EN mapping. A colour is a strong identifier on a
#: table of distinct-colour props (the demo's case), unlike a shape noun.
_COLOURS = {"red", "green", "blue", "pink", "yellow", "orange", "purple",
            "white", "black", "brown", "grey", "gray", "cyan", "magenta"}

_NOUN_ALIASES = {"tin": "can"}


def _label_tokens(name: str) -> set[str]:
    tokens = {_ES_EN.get(t, t) for t in _normalize(name).split("_")
              if t and t not in _GENERIC}
    return {_NOUN_ALIASES.get(t, t) for t in tokens}


def _match_label(label: str, poses: dict):
    """Shared label -> pose resolution.

    Rules, in order (see TruthPoseReader for the incidents behind them):
      1. exact normalized name, then the same name without a leading article;
      2. a unique best match with >= 2 shared content tokens; candidates
         must retain every non-generic query token, including noun aliases
         ("tomato tin" vs tomato_can); a single
         shared token is NOT identification ("pink cube" vs green_cube share
         {cube} -- returning the green cube confirmed against the wrong
         object entirely);
      3. a shared COLOUR that is unique among the candidates: "the red
         object" names exactly one body on a table with one red thing. The
         query's other words are generic ("object"), so demanding a second
         shared token here would drop the ONLY independent channel for the
         demo's own headline command ("pick and place the red object" ->
         body `red_cube`: measured, channel fell back to belief).
    """
    want = _normalize(label)
    if want in poses:
        return poses[want]
    article, _, name = want.partition("_")
    if article in {"the", "a", "an"} and name in poses:
        # Only remove the article. Ordinals, colors and other qualifiers
        # still belong to the object's identity, including one-word names.
        return poses[name]
    want_tokens = _label_tokens(want)
    candidates = []
    for key, xyz in poses.items():
        tokens = _label_tokens(key)
        if not want_tokens.issubset(tokens):
            continue
        candidates.append((tokens, xyz))
    best, best_score = [], 0
    for tokens, xyz in candidates:
        score = len(want_tokens & tokens)
        if score > best_score:
            best, best_score = [xyz], score
        elif score == best_score:
            best.append(xyz)
    if best_score >= 2:
        return best[0] if len(best) == 1 else None
    colours = want_tokens & _COLOURS
    if len(colours) == 1:
        colour = next(iter(colours))
        hits = [xyz for tokens, xyz in candidates if colour in tokens]
        if len(hits) == 1:
            return hits[0]
    return None


class LazyTruthPoseFn:
    """A truth channel that binds to the sim on FIRST USE, not at startup.

    Why: ``build_runtime`` attaches the verifier once, at startup. Under the
    MCP server the arm is a LazyArm that only materializes on the first
    motion command, so at attach time there is no bridge client (Isaac) and
    no loaded world (MuJoCo) to bind to -- ``make_truth_pose_fn`` correctly
    returns None, and every chat-driven pick then verifies against the
    belief store only. That is the demo's headline verification silently
    off in exactly the mode the demo is shown in.

    This object resolves the real reader lazily: each call first checks
    whether a reader can now be built (cheap: a dict lookup / registry peek,
    never a motor power-up -- it refuses to touch an unmaterialized LazyArm,
    same rule as ``make_truth_pose_fn``), caches it once found, and forgets it
    again if the sim goes away. ``PostconditionChecker`` sees a plain
    ``object_pose(label)`` callable either way.

    When explicitly given the warmed camera rig and Isaac arm profile, it
    can bind through a matching camera-owned client before any arm use. That
    supplies a physics pre-state without materializing the lazy actuator.
    """

    def __init__(self, arm, *, camera_rig=None, arm_cfg=None):
        self._arm = arm
        self._camera_rig = camera_rig
        self._arm_cfg = arm_cfg
        self._reader = None

    def _resolve(self):
        if self._reader is not None:
            live = getattr(self._reader, "live", True)
            if live:
                return self._reader
            self._reader = None
        try:
            self._reader = _matching_isaac_camera_reader(self._arm, self._camera_rig, self._arm_cfg)
            if self._reader is None:
                self._reader = make_truth_pose_fn(self._arm)
        except Exception:
            self._reader = None
        return self._reader

    def __call__(self, label: str):
        reader = self._resolve()
        return reader(label) if reader is not None else None

    def all_poses(self) -> dict:
        reader = self._resolve()
        return reader.all_poses() if reader is not None else {}

    @property
    def bound(self) -> bool:
        return self._resolve() is not None


def _matching_isaac_camera_reader(arm, rig, cfg):
    """Borrow an already-open, identity-matched camera transport for truth.

    Isaac exists before its LazyArm materializes. CameraRig owns this client's
    connection and teardown; verification never calls an arm factory, connect,
    resume, reset, target or gripper method.
    """
    if rig is None or cfg is None or cfg.get("type") != "isaac":
        return None
    kind = type(arm).__name__
    if kind == "LazyArm":
        if arm.__dict__.get("_profile_type") != "isaac":
            return None
    elif kind != "IsaacArm":
        return None
    endpoint = (str(cfg.get("bridge_host", "127.0.0.1")), int(cfg.get("bridge_port", 8611)))
    robot_id = cfg.get("bridge_robot_id")
    if not robot_id:
        return None
    for stream in rig:
        camera = getattr(stream, "_camera", None)
        if type(camera).__name__ != "IsaacCamera":
            continue
        client = camera.__dict__.get("_client")
        if client is None or getattr(client, "_addr", None) != endpoint:
            continue
        frame = stream.latest()
        capture = getattr(frame, "capture", None)
        if not isinstance(capture, dict):
            continue
        state = capture.get("proprioception") or {}
        stamp = capture.get("t")
        if (capture.get("backend") != "isaac" or capture.get("source") != endpoint
                or type(stamp) not in (int, float) or not math.isfinite(stamp) or stamp < 0
                or type(state.get("version")) is not int or state.get("version") != 1
                or state.get("backend") != "isaac"
                or state.get("joint_convention") != "asset"
                or state.get("robot_id") != robot_id
                or state.get("t") != stamp
                or state.get("time_source") != "physics_loop_monotonic"):
            continue
        return TruthPoseReader(client, ttl_s=0.0)
    return None


def _mujoco_scene_of(arm) -> str | None:
    """The MJCF path a (materialized) MuJoCo arm simulates, else None."""
    real = arm
    if type(arm).__name__ == "LazyArm":
        real = arm.__dict__.get("_arm")  # never __getattr__: it materializes
    if real is None or type(real).__name__ != "MujocoArm":
        return None
    if getattr(real, "world", None) is None:
        return None  # not connected
    return str(real.mjcf_path)


def make_truth_pose_fn(arm):
    """Build a truth-pose callable from an arm backend, if it is a sim arm.

    Isaac (bridge client) and MuJoCo (shared world) both qualify. Returns
    None for real hardware (and for mock arms), which makes
    ``PostconditionChecker`` fall back to perception -- the same behaviour the
    physical rig will have.

    PITFALL (cost a test regression): ``LazyArm.__getattr__`` materializes the
    real arm on any non-underscore attribute access, so a bare
    ``getattr(arm, "client", None)`` here powers up the CAN bus / motors as a
    side effect of *setting up verification*.  That is precisely what LazyArm
    exists to prevent.  Never probe an unmaterialized LazyArm: if it has not
    been used yet there is nothing to verify against anyway (and
    ``LazyTruthPoseFn`` re-asks later, once it has).
    """
    if type(arm).__name__ == "LazyArm" and not getattr(arm, "connected", False):
        # Unmaterialized MuJoCo arm: the ARM has no world yet, but a rendered
        # camera (`type: mujoco`, opened at prewarm) may already hold the very
        # world the arm will attach to. Peeking at it costs no motor power-up
        # and binds the physics channel BEFORE the first motion, so the
        # pre-motion snapshot is physics too -- otherwise it falls back to a
        # belief restored from disk and the displacement check compares two
        # different channels (see PostconditionChecker._comparable_start).
        # Only when the answer is unambiguous: one live world.
        if getattr(arm, "profile_type", None) == "mujoco":
            from . import mujoco_world

            worlds = mujoco_world.live_worlds()
            if len(worlds) == 1:
                reader = MujocoTruthReader(worlds[0].path)
                return reader if reader.live else None
        return None
    scene = _mujoco_scene_of(arm)
    if scene is not None:
        reader = MujocoTruthReader(scene)
        return reader if reader.live else None
    real = arm.__dict__.get("_arm") if type(arm).__name__ == "LazyArm" else arm
    if real is None:
        return None
    client = real.__dict__.get("client") or real.__dict__.get("_client")
    if client is None or not hasattr(client, "request"):
        return None
    try:
        reader = TruthPoseReader(client)
        reader.all_poses()  # one probe now, so a broken wiring fails loudly here
        return reader
    except Exception:
        return None
