#!/usr/bin/env python3
"""Extract a fixed-base ARM chain from a full-robot URDF.

    python scripts/extract_arm_urdf.py full.urdf --root torso_link \
        --tip right_wrist_yaw_link -o arm.urdf

Why this exists: cascade's kinematics slice the FIRST `n_joints` of the
model (see control/kinematics.py), which is correct for the shipped
tabletop arms because their URDFs order the arm chain first. A humanoid
URDF breaks that convention twice -- a floating base joint (nq += 7) and
legs declared before the arms -- so the honest fix is a derived,
reviewed asset: an arm-only URDF whose root is the humanoid's torso and
whose joints are exactly the chain root->tip, in order. This script
produces that document deterministically; the output is vendored with a
PROVENANCE.md naming the upstream file, commit and this command, so a
re-vendor is a reviewed event exactly like every other asset here.

The output keeps each kept element's original XML (attributes, limits,
inertials, mesh references untouched -- Pinocchio's buildModelFromXML
never opens meshes, so `package://` URIs in visual/collision blocks are
inert text). Everything not on the chain is dropped, including the
floating base, so the root link becomes a fixed base by URDF convention.
"""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def _req(el: ET.Element, tag: str, attr: str) -> str:
    sub = el.find(tag)
    val = sub.get(attr) if sub is not None else None
    if not val:
        raise SystemExit(f"joint {el.get('name')!r} lacks <{tag} {attr}=...>")
    return val


def extract_arm(urdf_text: str, root_link: str, tip_link: str,
                robot_name: str | None = None,
                tcp_offset: tuple[float, float, float] | None = None) -> str:
    robot = ET.fromstring(urdf_text)
    if robot.tag != "robot":
        raise SystemExit("input is not a URDF (<robot> root not found)")

    links = {el.get("name") or "": el for el in robot.findall("link")}
    joints = list(robot.findall("joint"))
    by_child: dict[str, ET.Element] = {}
    for j in joints:
        child = j.find("child")
        if child is not None and child.get("link"):
            by_child[child.get("link") or ""] = j

    for name in (root_link, tip_link):
        if name not in links:
            raise SystemExit(
                f"link {name!r} not in the URDF; links: {sorted(links)[:40]}...")

    # Walk tip -> root through the child->parent joint map.
    chain: list[ET.Element] = []
    cur = tip_link
    seen = set()
    while cur != root_link:
        if cur in seen:
            raise SystemExit(f"cycle at link {cur!r}")
        seen.add(cur)
        j = by_child.get(cur)
        if j is None:
            raise SystemExit(
                f"no joint has child {cur!r}; {tip_link!r} does not descend "
                f"from {root_link!r}")
        chain.append(j)
        cur = _req(j, "parent", "link")
    chain.reverse()

    out = ET.Element("robot", {
        "name": robot_name or f"{robot.get('name', 'robot')}_{tip_link}",
    })
    out.append(links[root_link])
    for j in chain:
        out.append(j)
        out.append(links[_req(j, "child", "link")])

    if tcp_offset is not None:
        # A humanoid arm often ends mid-forearm (H1 gen 1 has no wrist and
        # its last link IS the forearm), so grasp planning needs a tool
        # frame at the physical tip. A fixed joint adds that frame without
        # touching kinematic structure (nq unchanged).
        x, y, z = tcp_offset
        tcp = ET.SubElement(out, "link", {"name": "tcp_link"})
        _ = tcp
        j = ET.SubElement(out, "joint", {"name": "tcp_joint", "type": "fixed"})
        ET.SubElement(j, "origin", {"xyz": f"{x} {y} {z}", "rpy": "0 0 0"})
        ET.SubElement(j, "parent", {"link": tip_link})
        ET.SubElement(j, "child", {"link": "tcp_link"})

    ET.indent(out, space="  ")
    return ET.tostring(out, encoding="unicode", xml_declaration=True) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("urdf", help="full-robot URDF path")
    p.add_argument("--root", required=True, help="new base link (e.g. torso_link)")
    p.add_argument("--tip", required=True, help="end-effector link")
    p.add_argument("--name", default=None, help="output <robot name=...>")
    p.add_argument("--tcp-offset", default=None,
                   help="x,y,z of a fixed tcp_link appended after --tip "
                        "(for arms whose last link is the forearm)")
    p.add_argument("-o", "--output", default=None, help="output path (default: stdout)")
    args = p.parse_args(argv)

    text = Path(args.urdf).read_text()
    tcp = None
    if args.tcp_offset:
        tcp = tuple(float(v) for v in args.tcp_offset.split(","))
        if len(tcp) != 3:
            p.error("--tcp-offset needs x,y,z")
    out = extract_arm(text, args.root, args.tip, args.name, tcp_offset=tcp)
    if args.output:
        Path(args.output).write_text(out)
        n_joints = out.count("<joint ")
        print(f"wrote {args.output}: root={args.root} tip={args.tip} "
              f"joints={n_joints}", file=sys.stderr)
    else:
        print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
