#!/usr/bin/env python3
"""Download the robot assets that are deliberately NOT vendored.

    python scripts/fetch_robot_assets.py --list
    python scripts/fetch_robot_assets.py so101
    python scripts/fetch_robot_assets.py so101 --with-urdf-meshes

WHAT IS AND IS NOT IN GIT
-------------------------
Kinematics need only URDF *text*: Pinocchio's `buildModelFromXML` never opens a
mesh, so FK, IK, joint limits and the whole safety layer work from a ~16 KB file.
Those are vendored (assets/urdf/<robot>/, with a PROVENANCE.md each).

Meshes and MuJoCo MJCFs are tens of megabytes and only rendering and contact
simulation need them, so they are fetched on demand. Keeping them out of git is
not only about size: CI's `actions/checkout` does not fetch LFS content by
default, so an LFS-backed model file arrives as a pointer and breaks every
kinematics test in a way that looks like a code bug.

Every download is pinned to a COMMIT, never a branch. `main` moving under a
robot description silently changes the arm's geometry, and a re-vendor is
supposed to be a reviewed event -- `tests/test_kinematics_so101.py` pins the
facts the arm profiles depend on for exactly that reason.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RAW = "https://raw.githubusercontent.com"

SO101_MESHES = [
    "base_motor_holder_so101_v1.stl",
    "base_so101_v2.stl",
    "motor_holder_so101_base_v1.stl",
    "motor_holder_so101_wrist_v1.stl",
    "moving_jaw_so101_gripper_part0_v1.stl",
    "moving_jaw_so101_gripper_part1_v1.stl",
    "moving_jaw_so101_gripper_v1.stl",
    "moving_jaw_so101_v1.stl",
    "rotation_pitch_so101_v1.stl",
    "sts3215_03a_no_horn_v1.stl",
    "sts3215_03a_v1.stl",
    "under_arm_so101_v1.stl",
    "upper_arm_so101_v1.stl",
    "waveshare_mounting_plate_so101_v2.stl",
    "wrist_roll_follower_so101_camera_mount.stl",
    "wrist_roll_follower_so101_gripper_part0_v1.stl",
    "wrist_roll_follower_so101_gripper_v1.stl",
    "wrist_roll_follower_so101_v1.stl",
    "wrist_roll_pitch_so101_v2.stl",
]


@dataclass
class Robot:
    name: str
    description: str
    repo: str
    commit: str
    license: str
    #: (path within the upstream repo, destination relative to assets/)
    files: list[tuple[str, str]]
    #: mesh basenames, fetched from `mesh_src` into `mesh_dest`
    meshes: list[str] = field(default_factory=list)
    mesh_src: str = ""
    mesh_dest: str = ""
    #: where the vendored URDF wants its meshes, for --with-urdf-meshes
    urdf_mesh_dest: str = ""


ROBOTS: dict[str, Robot] = {
    "so101": Robot(
        name="so101",
        description="The Robot Studio SO-101 — MuJoCo MJCF + STL meshes "
                    "(the URDF is already vendored)",
        repo="google-deepmind/mujoco_menagerie",
        commit="da76818e269b82289eba39808e2fb91d679d6994",
        license="Apache-2.0",
        files=[
            ("robotstudio_so101/so101.xml", "mjcf/so101/so101.xml"),
            ("robotstudio_so101/scene.xml", "mjcf/so101/scene.xml"),
            ("robotstudio_so101/scene_box.xml", "mjcf/so101/scene_box.xml"),
            ("robotstudio_so101/LICENSE", "mjcf/so101/LICENSE"),
            ("robotstudio_so101/README.md", "mjcf/so101/README.md"),
        ],
        meshes=SO101_MESHES,
        mesh_src="robotstudio_so101/assets",
        mesh_dest="mjcf/so101/assets",
        urdf_mesh_dest="urdf/so101/assets",
    ),
}


def fetch(url: str, dest: Path, force: bool = False) -> bool:
    """-> True if downloaded, False if skipped. Writes atomically."""
    if dest.exists() and not force:
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f)
    except urllib.error.HTTPError as e:
        tmp.unlink(missing_ok=True)
        raise SystemExit(f"HTTP {e.code} fetching {url}\n"
                         f"The pinned commit may have been rewritten upstream.") from e
    except urllib.error.URLError as e:
        tmp.unlink(missing_ok=True)
        raise SystemExit(f"network error fetching {url}: {e.reason}") from e
    # Replace only after a complete download, so an interrupted run leaves no
    # half-file that the next run would skip as "already there".
    tmp.replace(dest)
    return True


def link_or_copy(src: Path, dest: Path) -> None:
    """Hardlink `src` to `dest`, falling back to a copy across filesystems."""
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        import os

        os.link(src, dest)
    except (OSError, AttributeError):
        shutil.copy2(src, dest)


def fetch_robot(robot: Robot, assets: Path, force: bool, urdf_meshes: bool) -> None:
    base = f"{RAW}/{robot.repo}/{robot.commit}"
    print(f"{robot.name}: {robot.repo} @ {robot.commit[:12]} ({robot.license})")
    got = skipped = 0
    for src, dest in robot.files:
        if fetch(f"{base}/{src}", assets / dest, force):
            got += 1
            print(f"  + {dest}")
        else:
            skipped += 1
    for mesh in robot.meshes:
        dest = assets / robot.mesh_dest / mesh
        if fetch(f"{base}/{robot.mesh_src}/{mesh}", dest, force):
            got += 1
            print(f"  + {robot.mesh_dest}/{mesh}")
        else:
            skipped += 1
    if urdf_meshes and robot.urdf_mesh_dest:
        for mesh in robot.meshes:
            link_or_copy(assets / robot.mesh_dest / mesh,
                         assets / robot.urdf_mesh_dest / mesh)
        print(f"  = linked {len(robot.meshes)} meshes into {robot.urdf_mesh_dest}/ "
              f"(only needed to VIEW the URDF; kinematics never load meshes)")
    print(f"  {got} downloaded, {skipped} already present")

    written = assets / robot.mesh_dest
    stamp = written.parent / "FETCHED"
    stamp.parent.mkdir(parents=True, exist_ok=True)
    stamp.write_text(
        f"repo: {robot.repo}\ncommit: {robot.commit}\nlicense: {robot.license}\n"
        f"fetched-by: scripts/fetch_robot_assets.py\n"
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("robot", nargs="*", help="robot name(s); omit with --list")
    p.add_argument("--list", action="store_true", help="show what can be fetched")
    p.add_argument("--force", action="store_true", help="re-download existing files")
    p.add_argument("--with-urdf-meshes", action="store_true",
                   help="also place the meshes beside the vendored URDF (only "
                        "needed to VIEW it; FK/IK never read them)")
    p.add_argument("--assets", default=str(REPO / "assets"),
                   help="assets directory (default: <repo>/assets)")
    args = p.parse_args(argv)

    if args.list or not args.robot:
        print("available robots:\n")
        for r in ROBOTS.values():
            n_files = len(r.files) + len(r.meshes)
            print(f"  {r.name:12s} {r.description}")
            print(f"  {'':12s} {r.repo} @ {r.commit[:12]}, {n_files} files, {r.license}")
        if not args.robot:
            print("\nnothing fetched; name a robot (e.g. `so101`)")
        return 0

    unknown = [n for n in args.robot if n not in ROBOTS]
    if unknown:
        p.error(f"unknown robot(s) {unknown}; known: {sorted(ROBOTS)}")

    assets = Path(args.assets)
    for name in args.robot:
        fetch_robot(ROBOTS[name], assets, args.force, args.with_urdf_meshes)
    return 0


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


if __name__ == "__main__":
    sys.exit(main())
