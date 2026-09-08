"""One MuJoCo world shared by every backend that simulates the same scene.

WHY THIS EXISTS. The MuJoCo arm (`control/mujoco_arm.py`) steps physics on
its own `MjModel`/`MjData`; a rendered camera (`perception/mujoco_camera.py`)
needs to draw THAT state, and the physics-truth verification channel
(`sim/truth.py`) needs to read prop poses out of THAT state. Three objects,
one world. If each loaded the MJCF for itself the camera would render a scene
in which the arm never moves and the prop never leaves the table -- the
verification channel would then "refute" every successful grasp, which is
worse than having no channel at all.

So the world is a process-wide registry keyed by the resolved MJCF path.
Whoever asks first loads the model; everyone else attaches to the same
`MjData`. The physics OWNER (the arm) steps; readers (cameras, truth) only
look. One lock per world guards the data because MuJoCo's C structs are not
reentrant: the camera stream thread renders while the motion thread steps,
and without the lock a render sees half-stepped state (and, worse, a step
can run while `mj_forward` from a reader mutates derived fields).

Build order is deliberately NOT assumed. `build_runtime` constructs arms
before cameras, but the MCP server wraps the arm in a LazyArm that only
materializes on the first motion command -- so in that mode the CAMERA is
the first to load the world and the arm attaches later. Both orders are
pinned by `tests/test_mujoco_world.py`.

The Warp engine is out of scope here on purpose: its authoritative state is
the device-side `mjw.Data`, and a camera would need `get_data_into()` to
mirror it back every frame. `engine: warp` therefore keeps its private model
and the rendered camera refuses to attach to it (with a message), rather
than silently rendering a frozen home pose.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

logger = logging.getLogger(__name__)


class MujocoWorld:
    """A loaded MJCF + its live state + the lock that guards both."""

    def __init__(self, mjcf_path: str):
        import mujoco

        self.mj = mujoco
        self.path = str(Path(mjcf_path).resolve())
        self.model = mujoco.MjModel.from_xml_path(self.path)
        self.data = mujoco.MjData(self.model)
        #: RLock: the arm takes it around every step AND from wait_settled,
        #: which itself calls get_state (also locked) -- a plain Lock deadlocks.
        self.lock = threading.RLock()
        self.refs = 0
        #: Set by the physics owner once the home pose is realized. Readers
        #: that render before this see the MJCF's default pose, which is the
        #: honest answer ("nobody has posed the arm yet"), not an error.
        self.realized = False
        mujoco.mj_forward(self.model, self.data)

    # ── ground truth ────────────────────────────────────────────────────

    def body_pos(self, name: str):
        """World position of a body, or None if the MJCF has no such body."""
        bid = self.mj.mj_name2id(self.model, self.mj.mjtObj.mjOBJ_BODY, name)
        if bid < 0:
            return None
        with self.lock:
            return [float(v) for v in self.data.xpos[bid]]

    def free_body_names(self) -> list[str]:
        """Bodies hanging off a free joint -- the props, in every scene this
        repo generates (`sim/demo_scene.py`) and in MuJoCo Menagerie scenes.
        The arm's links are excluded by construction (hinges), and so is the
        world body, so this is exactly "the things a grasp can move"."""
        out = []
        m, mj = self.model, self.mj
        for j in range(m.njnt):
            if m.jnt_type[j] == mj.mjtJoint.mjJNT_FREE:
                out.append(mj.mj_id2name(m, mj.mjtObj.mjOBJ_BODY, int(m.jnt_bodyid[j])))
        return out


_WORLDS: dict[str, MujocoWorld] = {}
_REGISTRY_LOCK = threading.Lock()


def acquire(mjcf_path: str) -> MujocoWorld:
    """Attach to the world for `mjcf_path`, loading it on first use."""
    key = str(Path(mjcf_path).resolve())
    with _REGISTRY_LOCK:
        world = _WORLDS.get(key)
        if world is None:
            world = MujocoWorld(key)
            _WORLDS[key] = world
            logger.info("mujoco world loaded: %s", Path(key).name)
        world.refs += 1
        return world


def release(world: MujocoWorld) -> None:
    """Detach; the world is dropped when the last holder lets go, so a
    reconnect (or the next test) starts from a fresh MJCF load."""
    with _REGISTRY_LOCK:
        world.refs = max(0, world.refs - 1)
        if world.refs == 0 and _WORLDS.get(world.path) is world:
            del _WORLDS[world.path]
            logger.info("mujoco world dropped: %s", Path(world.path).name)


def peek(mjcf_path: str) -> MujocoWorld | None:
    """The live world for `mjcf_path` if someone holds it, without attaching.
    This is what the truth channel uses: verification must never be the
    thing that loads (and therefore *owns*) a physics world."""
    key = str(Path(mjcf_path).resolve())
    with _REGISTRY_LOCK:
        return _WORLDS.get(key)


def live_worlds() -> list[MujocoWorld]:
    with _REGISTRY_LOCK:
        return list(_WORLDS.values())
