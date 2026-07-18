"""Build a Pinocchio-ready URDF string from the RS arm's USD asset.

The USD asset (assets/usd/RS-rebot-dev-arm) is the single source of truth
for the robot model: Isaac Sim consumes it directly, and host-side FK/IK
needs the same joint tree in Pinocchio, which has no USD reader. usd-core
ships no aarch64 wheels (this repo runs on DGX Spark), so this parses the
converter-generated .usda text directly -- the physics layer is flat,
machine-written USD ASCII and everything kinematic lives in Physics*Joint
defs plus per-link PhysicsMassAPI attributes.

USD physics joint -> URDF joint math: with A = SE3(localRot0, localPos0)
(joint frame in parent body) and B = SE3(localRot1, localPos1) (joint frame
in child body), the child pose is X(q) = A * Motion_axis(q) * B^-1, which is
the URDF form X(q) = origin * Motion_a(q) with origin = A * B^-1 and the
axis a = R_B @ axis expressed in the child/joint frame. localPos1 must be
zero for that to hold exactly (true for all converter output; asserted).
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np

_JOINT_DEF = re.compile(r'def\s+Physics(\w+)Joint\s+"([^"]+)"')
_SCOPE_DEF = re.compile(r'(?:def|over)\s+[\w:]*\s*"([^"]+)"')
_REF_PATH = re.compile(r"@([^@]+\.usda)@")
_VEC = re.compile(r"[-+0-9.eE]+")

_AXES = {"X": np.array([1.0, 0, 0]), "Y": np.array([0, 1.0, 0]), "Z": np.array([0, 0, 1.0])}


def _floats(s: str) -> list[float]:
    return [float(v) for v in _VEC.findall(s)]


def _attr(block: str, name: str) -> str | None:
    m = re.search(rf"{re.escape(name)}\s*=\s*([^\n]+)", block)
    return m.group(1).strip() if m else None


def _quat_to_R(w: float, x: float, y: float, z: float) -> np.ndarray:
    """usda quatf text order is (w, x, y, z)."""
    n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def _R_to_rpy(R: np.ndarray) -> tuple[float, float, float]:
    """URDF rpy: extrinsic X-Y-Z, i.e. R = Rz(y) @ Ry(p) @ Rx(r)."""
    sy = math.hypot(R[0, 0], R[1, 0])
    if sy < 1e-9:  # pitch at +-90 deg: fold yaw into roll
        return math.atan2(-R[1, 2], R[1, 1]), math.atan2(-R[2, 0], sy), 0.0
    return (math.atan2(R[2, 1], R[2, 2]),
            math.atan2(-R[2, 0], sy),
            math.atan2(R[1, 0], R[0, 0]))


def _block(text: str, start: int) -> str:
    """The { ... } body following `start` (skips the (metadata) header)."""
    i = text.index("{", start)
    depth = 0
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1:j]
    raise ValueError("unbalanced braces in usda")


def _find_physics_layer(path: Path) -> str:
    """Text of the (sub)layer that defines the Physics*Joint prims.

    Accepts the asset root .usda or the physics layer itself; follows
    ASCII payload/reference/subLayer paths breadth-first. Binary .usd
    payloads (geometry) are never joint sources and are skipped.
    """
    queue, seen = [path.resolve()], set()
    while queue:
        p = queue.pop(0)
        if p in seen or not p.exists():
            continue
        seen.add(p)
        text = p.read_text(errors="ignore")
        if _JOINT_DEF.search(text):
            return text
        queue += [(p.parent / rel).resolve() for rel in _REF_PATH.findall(text)]
    raise ValueError(f"no PhysicsJoint prims reachable from {path}")


def _parse_joints(text: str) -> list[dict]:
    joints = []
    for m in _JOINT_DEF.finditer(text):
        body = _block(text, m.end())
        get = lambda name, default=None: _attr(body, name) or default
        j = {
            "name": m.group(2),
            "type": m.group(1).lower(),  # revolute | prismatic | fixed
            "body0": get("physics:body0", "").strip("<>"),
            "body1": get("physics:body1", "").strip("<>"),
            "pos0": np.array(_floats(get("physics:localPos0", "(0,0,0)"))),
            "pos1": np.array(_floats(get("physics:localPos1", "(0,0,0)"))),
            "rot0": _quat_to_R(*_floats(get("physics:localRot0", "(1,0,0,0)"))),
            "rot1": _quat_to_R(*_floats(get("physics:localRot1", "(1,0,0,0)"))),
            "axis": get('physics:axis', '"Z"').strip('"'),
            "lower": get("physics:lowerLimit"),
            "upper": get("physics:upperLimit"),
            "effort": get("urdf:limit:effort"),
            "velocity": get("newton:velocityLimit"),
        }
        # rel lines look like `prepend rel physics:body0 = </path>`
        for k in ("body0", "body1"):
            mm = re.search(rf"physics:{k}\s*=\s*<([^>]*)>", body)
            j[k] = mm.group(1) if mm else ""
        joints.append(j)
    return joints


def _parse_mass_props(text: str) -> dict[str, dict]:
    """Leaf prim name -> PhysicsMassAPI attrs, via brace-scoped scan."""
    props: dict[str, dict] = {}
    stack: list[str] = []
    pending: str | None = None
    for line in text.splitlines():
        s = line.strip()
        d = _SCOPE_DEF.match(s)
        if d:
            pending = d.group(1)
        if "{" in s:
            stack.append(pending or "")
            pending = None
        for attr, key in (("physics:mass", "mass"),
                          ("physics:centerOfMass", "com"),
                          ("physics:diagonalInertia", "inertia"),
                          ("physics:principalAxes", "axes")):
            if stack and f"{attr} " in s:
                props.setdefault(stack[-1], {})[key] = _floats(s.split("=", 1)[1])
        if "}" in s and stack:
            stack.pop()
    return props


def urdf_xml_from_usd(usd_path: str | Path) -> str:
    """USD asset (root or physics layer) -> URDF XML string.

    Joint frames, limits (USD degrees -> rad), efforts, velocities and
    PhysicsMassAPI inertials all carry over; visual/collision geometry is
    intentionally dropped (Pinocchio FK/IK does not need meshes).
    """
    text = _find_physics_layer(Path(usd_path))
    joints = _parse_joints(text)
    mass_props = _parse_mass_props(text)

    robot = ET.Element("robot", name=Path(usd_path).stem)
    links_done: set[str] = set()

    def add_link(name: str):
        if name in links_done:
            return
        links_done.add(name)
        link = ET.SubElement(robot, "link", name=name)
        mp = mass_props.get(name)
        if not mp or "mass" not in mp:
            return
        inertial = ET.SubElement(link, "inertial")
        com = mp.get("com", [0, 0, 0])
        rpy = _R_to_rpy(_quat_to_R(*mp["axes"])) if "axes" in mp else (0, 0, 0)
        ET.SubElement(inertial, "origin",
                      xyz=" ".join(f"{v:.9g}" for v in com),
                      rpy=" ".join(f"{v:.9g}" for v in rpy))
        ET.SubElement(inertial, "mass", value=f"{mp['mass'][0]:.9g}")
        ixx, iyy, izz = mp.get("inertia", [0, 0, 0])
        ET.SubElement(inertial, "inertia", ixx=f"{ixx:.9g}", iyy=f"{iyy:.9g}",
                      izz=f"{izz:.9g}", ixy="0", ixz="0", iyz="0")

    for j in joints:
        # The articulation anchor (fixed joint from the asset's default
        # prim onto base_link) has no URDF counterpart: its body0 is the
        # root prim itself, so base_link stays the URDF root link.
        if j["body0"].count("/") <= 1:
            continue
        parent, child = j["body0"].rsplit("/", 1)[1], j["body1"].rsplit("/", 1)[1]
        if np.linalg.norm(j["pos1"]) > 1e-9:
            raise ValueError(f"{j['name']}: nonzero localPos1 unsupported")
        add_link(parent)
        add_link(child)

        origin_R = j["rot0"] @ j["rot1"].T
        el = ET.SubElement(robot, "joint", name=j["name"], type=j["type"])
        ET.SubElement(el, "origin",
                      xyz=" ".join(f"{v:.9g}" for v in j["pos0"]),
                      rpy=" ".join(f"{v:.9g}" for v in _R_to_rpy(origin_R)))
        ET.SubElement(el, "parent", link=parent)
        ET.SubElement(el, "child", link=child)
        if j["type"] == "fixed":
            continue
        axis = j["rot1"] @ _AXES[j["axis"]]
        ET.SubElement(el, "axis", xyz=" ".join(f"{v:.9g}" for v in axis))
        to_si = math.radians if j["type"] == "revolute" else float
        ET.SubElement(
            el, "limit",
            lower=f"{to_si(float(j['lower'] or 0)):.9g}",
            upper=f"{to_si(float(j['upper'] or 0)):.9g}",
            effort=f"{float(j['effort'] or 100):.9g}",
            velocity=f"{to_si(float(j['velocity'] or (2 if j['type'] == 'prismatic' else 180))):.9g}",
        )

    return ET.tostring(robot, encoding="unicode")
