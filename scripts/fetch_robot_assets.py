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
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

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

PIPER_MESHES = [
    "base_link.stl", "link1.stl", "link2.stl", "link2_0.obj", "link2_1.obj",
    "link2_10.obj", "link2_11.obj", "link2_12.obj", "link2_13.obj", "link2_14.obj",
    "link2_15.obj", "link2_16.obj", "link2_17.obj", "link2_18.obj", "link2_19.obj",
    "link2_2.obj", "link2_20.obj", "link2_21.obj", "link2_22.obj", "link2_23.obj",
    "link2_24.obj", "link2_25.obj", "link2_26.obj", "link2_27.obj", "link2_28.obj",
    "link2_29.obj", "link2_3.obj", "link2_30.obj", "link2_31.obj", "link2_4.obj",
    "link2_5.obj", "link2_6.obj", "link2_7.obj", "link2_8.obj", "link2_9.obj",
    "link2_gray.stl", "link2_red.stl", "link3.stl", "link3_0.obj", "link3_1.obj",
    "link3_10.obj", "link3_11.obj", "link3_12.obj", "link3_13.obj", "link3_14.obj",
    "link3_15.obj", "link3_16.obj", "link3_17.obj", "link3_18.obj", "link3_19.obj",
    "link3_2.obj", "link3_20.obj", "link3_21.obj", "link3_22.obj", "link3_23.obj",
    "link3_24.obj", "link3_25.obj", "link3_26.obj", "link3_27.obj", "link3_28.obj",
    "link3_3.obj", "link3_4.obj", "link3_5.obj", "link3_7.obj", "link3_8.obj",
    "link3_9.obj", "link4.stl", "link4_2.obj", "link4_3.obj", "link4_4.obj",
    "link4_5.obj", "link4_6.obj", "link5.stl", "link5_2.obj", "link5_3.obj",
    "link5_4.obj", "link5_5.obj", "link5_6.obj", "link5_7.obj", "link5_8.obj",
    "link6.stl", "link7.stl", "link8.stl", "linke2_dark_gray.stl",
]

H1_MESHES = [
    "left_ankle_link.stl", "left_elbow_link.stl", "left_hip_pitch_link.stl",
    "left_hip_roll_link.stl", "left_hip_yaw_link.stl", "left_knee_link.stl",
    "left_shoulder_pitch_link.stl", "left_shoulder_roll_link.stl",
    "left_shoulder_yaw_link.stl", "logo_link.stl", "pelvis.stl",
    "right_ankle_link.stl", "right_elbow_link.stl", "right_hip_pitch_link.stl",
    "right_hip_roll_link.stl", "right_hip_yaw_link.stl", "right_knee_link.stl",
    "right_shoulder_pitch_link.stl", "right_shoulder_roll_link.stl",
    "right_shoulder_yaw_link.stl", "torso_link.stl",
]

# Exactly the `file="..."` set referenced by franka_fr3/fr3.xml at the pinned
# commit (36 files: STL collision + OBJ visual per link).
FR3_MESHES = [
    "link0.stl", "link0_0.obj", "link0_1.obj", "link0_2.obj", "link0_3.obj",
    "link0_4.obj", "link0_5.obj", "link0_6.obj",
    "link1.obj", "link1.stl", "link2.obj", "link2.stl",
    "link3.stl", "link3_0.obj", "link3_1.obj",
    "link4.stl", "link4_0.obj", "link4_1.obj",
    "link5.stl", "link5_0.obj", "link5_1.obj", "link5_2.obj",
    "link6.stl", "link6_0.obj", "link6_1.obj", "link6_2.obj", "link6_3.obj",
    "link6_4.obj", "link6_5.obj", "link6_6.obj", "link6_7.obj",
    "link7.stl", "link7_0.obj", "link7_1.obj", "link7_2.obj", "link7_3.obj",
]
# franka_emika_panda/hand.xml's meshes (the Franka Hand is the same part on
# the Panda and the FR3).
PANDA_HAND_MESHES = [
    "hand.stl", "hand_0.obj", "hand_1.obj", "hand_2.obj", "hand_3.obj",
    "hand_4.obj", "finger_0.obj", "finger_1.obj",
]

def compose_fr3_hand_scene(mjcf_dir: Path) -> Path:
    """Write `fr3_hand.xml` = fr3.xml with the Franka Hand attached to link7,
    and `scene_hand.xml` = the Menagerie scene around it.

    Uses MuJoCo's <attach> (3.1+): hand.xml is declared as an <asset><model>
    and attached inside a <frame> under link7 posed so the hand lands exactly
    where franka_description mounts it -- `fr3_joint8` (0 0 0.107) then
    `fr3_hand_joint` (rpy 0 0 -pi/4). BUT hand.xml's own body carries
    quat="0 0 0 1" (180 deg about z) which <attach> KEEPS (panda.xml avoids
    this by restating the body header), so the frame must absorb it:
    yaw(-45) * yaw(-180) = yaw(+135), quat (cos 67.5, 0, 0, sin 67.5).
    MEASURED: with the naive -45 deg frame the hand was 180 deg off the URDF
    at every q; with +135 deg, 50 random q agree to 0.0 mm / 3e-8 in R.
    `tests/test_kinematics_fr3.py` pins that agreement.

    Attached names get the prefix `fh_` (fh_hand, fh_finger_joint1/2,
    fh_actuator8).

    PATH TRAP (mujoco 3.12, measured): `<model file=...>` is resolved by
    prepending the top-level model's directory STRING to the reference --
    twice when that top-level path is itself relative, so loading
    "assets/mjcf/fr3/scene_hand.xml" from the repo root looks for
    assets/mjcf/fr3/assets/mjcf/fr3/hand.xml and fails, while the very same
    file loads from an ABSOLUTE path (any cwd). `../fr3/hand.xml` is the
    reference that works from the absolute form. Every loader in this repo
    (`sim/mujoco_world.acquire`, the arm, the camera) resolves the MJCF path
    to absolute before `MjModel.from_xml_path`, so the runtime never hits
    the relative form; the fetcher's own self-check below uses it too.
    """
    fr3 = (mjcf_dir / "fr3.xml").read_text()
    anchor = '<site name="attachment_site" pos="0 0 0.107"/>'
    if anchor not in fr3 or "<asset>" not in fr3:
        raise SystemExit("fr3.xml changed upstream (no attachment_site/asset); "
                         "update compose_fr3_hand_scene")
    mount = (
        anchor + "\n"
        '                      <frame name="hand_mount" pos="0 0 0.107" '
        'quat="0.3826834 0 0 0.9238795">\n'
        '                        <attach model="franka_hand" body="hand" prefix="fh_"/>\n'
        "                      </frame>"
    )
    out = fr3.replace(anchor, mount, 1)
    out = out.replace("<asset>", '<asset>\n    <model name="franka_hand" file="../fr3/hand.xml"/>', 1)
    out = out.replace('<mujoco model="fr3">', '<mujoco model="fr3_hand">', 1)
    out = out.replace("<mujoco model=\"fr3_hand\">",
                      "<mujoco model=\"fr3_hand\">\n  <!-- GENERATED by scripts/fetch_robot_assets.py "
                      "(compose_fr3_hand_scene): Menagerie franka_fr3/fr3.xml + the Franka Hand "
                      "(franka_emika_panda/hand.xml) attached where franka_description mounts it. "
                      "Do not edit; re-run the fetcher. -->", 1)
    (mjcf_dir / "fr3_hand.xml").write_text(out)

    scene = (mjcf_dir / "scene.xml").read_text().replace('file="fr3.xml"', 'file="fr3_hand.xml"')
    scene = scene.replace('<mujoco model="fr3 scene">', '<mujoco model="fr3 hand scene">')
    (mjcf_dir / "scene_hand.xml").write_text(scene)
    print(f"  = composed {mjcf_dir / 'fr3_hand.xml'} + scene_hand.xml (Franka Hand on link7)")
    # self-check when mujoco is importable: the composed model must load and
    # carry the 7 arm joints + 2 finger joints + the hand's tendon actuator.
    try:
        import mujoco  # noqa: PLC0415
    except ImportError:
        return mjcf_dir / "scene_hand.xml"
    m = mujoco.MjModel.from_xml_path(str((mjcf_dir / "scene_hand.xml").resolve()))
    names = [m.joint(i).name for i in range(m.njnt)]
    if m.nq != 9 or "fh_finger_joint1" not in names or m.nu != 8:
        raise SystemExit(f"composed FR3+hand model is wrong: nq={m.nq} nu={m.nu} joints={names}")
    print(f"  = verified: nq={m.nq} nu={m.nu} (7 arm + 2 fingers, 7 + 1 actuators)")
    return mjcf_dir / "scene_hand.xml"


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
    #: meshes that live under a DIFFERENT upstream directory than `mesh_src`
    #: (source dir, basename), fetched into `mesh_dest` alongside the others
    extra_meshes: list[tuple[str, str]] = field(default_factory=list)
    #: optional step after fetching (e.g. compose a scene from the pieces)
    post: "Callable[[Path], object] | None" = None


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
    # Both pinned to the SAME commit as so101: verified 2026-09-03 that
    # agilex_piper/ and unitree_h1/ are blob-for-blob identical between this
    # pin and Menagerie main (e4049d0a3bfd), so one pin covers the whole set.
    "piper": Robot(
        name="piper",
        description="AgileX PiPER 6-DoF arm + parallel gripper -- MuJoCo MJCF "
                    "+ meshes (the URDF is already vendored from piper_ros)",
        repo="google-deepmind/mujoco_menagerie",
        commit="da76818e269b82289eba39808e2fb91d679d6994",
        license="MIT",
        files=[
            ("agilex_piper/piper.xml", "mjcf/piper/piper.xml"),
            ("agilex_piper/scene.xml", "mjcf/piper/scene.xml"),
            ("agilex_piper/LICENSE", "mjcf/piper/LICENSE"),
            ("agilex_piper/README.md", "mjcf/piper/README.md"),
        ],
        meshes=PIPER_MESHES,
        mesh_src="agilex_piper/assets",
        mesh_dest="mjcf/piper/assets",
        urdf_mesh_dest="",  # piper_ros URDF uses package:// URIs; meshes not linkable
    ),
    "h1": Robot(
        name="h1",
        description="Unitree H1 humanoid (gen 1, 4-DoF arms) -- MuJoCo MJCF + "
                    "meshes (arm-only URDF is vendored under urdf/h1/)",
        repo="google-deepmind/mujoco_menagerie",
        commit="da76818e269b82289eba39808e2fb91d679d6994",
        license="BSD-3-Clause",
        files=[
            ("unitree_h1/h1.xml", "mjcf/h1/h1.xml"),
            ("unitree_h1/scene.xml", "mjcf/h1/scene.xml"),
            ("unitree_h1/LICENSE", "mjcf/h1/LICENSE"),
            ("unitree_h1/README.md", "mjcf/h1/README.md"),
        ],
        meshes=H1_MESHES,
        mesh_src="unitree_h1/assets",
        mesh_dest="mjcf/h1/assets",
        urdf_mesh_dest="",
    ),
    # Franka FR3. Menagerie's franka_fr3 is the BARE arm (attachment_site at
    # the flange, no hand); the Franka Hand model lives in franka_emika_panda
    # (hand.xml: two slide fingers, coupled by an equality, ONE tendon
    # actuator). Both are fetched and `scene_hand.xml` composes them with
    # MuJoCo's <attach> at the same pose franka_description mounts the hand
    # (0 0 0.107 from link7, yawed -45 deg) -- see compose_fr3_hand_scene().
    "fr3": Robot(
        name="fr3",
        description="Franka Research 3 arm (franka_fr3) + Franka Hand "
                    "(franka_emika_panda/hand.xml) -- MuJoCo MJCF + meshes; "
                    "the arm+hand URDF is vendored under urdf/fr3/",
        repo="google-deepmind/mujoco_menagerie",
        commit="da76818e269b82289eba39808e2fb91d679d6994",
        license="Apache-2.0 (fr3) / BSD-3-Clause (panda hand)",
        files=[
            ("franka_fr3/fr3.xml", "mjcf/fr3/fr3.xml"),
            ("franka_fr3/scene.xml", "mjcf/fr3/scene.xml"),
            ("franka_fr3/LICENSE", "mjcf/fr3/LICENSE"),
            ("franka_fr3/README.md", "mjcf/fr3/README.md"),
            ("franka_emika_panda/hand.xml", "mjcf/fr3/hand.xml"),
            ("franka_emika_panda/LICENSE", "mjcf/fr3/LICENSE.panda_hand"),
        ],
        meshes=FR3_MESHES,
        mesh_src="franka_fr3/assets",
        mesh_dest="mjcf/fr3/assets",
        urdf_mesh_dest="",
        extra_meshes=[("franka_emika_panda/assets", m) for m in PANDA_HAND_MESHES],
        post=lambda assets: compose_fr3_hand_scene(assets / "mjcf" / "fr3"),
    ),
}


def fetch(url: str, dest: Path, force: bool = False,
          attempts: int = 4) -> bool:
    """-> True if downloaded, False if skipped. Writes atomically.

    VERIFIES the byte count against Content-Length and retries: a dropped
    TLS connection ends the stream early but `copyfileobj` returns without
    error, and promoting that file ships a truncated mesh that surfaces much
    later as `stl_decoder: ... has wrong size` at MjModel load (observed with
    piper's link7.stl, 80492 of 91984 bytes). GitHub's raw endpoint always
    sends Content-Length, so a missing header is not treated as failure.
    """
    if dest.exists() and not force:
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    last_err: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
                expect = r.headers.get("Content-Length")
                shutil.copyfileobj(r, f)
                got = f.tell()
            if expect is not None and got != int(expect):
                raise OSError(
                    f"truncated read: {got} of {expect} bytes"
                )
            # Replace only after a VERIFIED-complete download, so an
            # interrupted run leaves no half-file that the next run would
            # skip as "already there".
            tmp.replace(dest)
            return True
        except urllib.error.HTTPError as e:
            tmp.unlink(missing_ok=True)
            raise SystemExit(f"HTTP {e.code} fetching {url}\n"
                             f"The pinned commit may have been rewritten upstream.") from e
        except (urllib.error.URLError, OSError) as e:
            # URLError covers DNS/TLS setup; OSError covers mid-stream resets
            # and the truncation above. Both are transient on flaky links, so
            # retry with a short backoff instead of dying on the first one.
            last_err = e
            tmp.unlink(missing_ok=True)
            if attempt < attempts:
                time.sleep(1.5 * attempt)
    raise SystemExit(f"network error fetching {url} "
                     f"after {attempts} attempts: {last_err}")


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
    for src_dir, mesh in robot.extra_meshes:
        dest = assets / robot.mesh_dest / mesh
        if fetch(f"{base}/{src_dir}/{mesh}", dest, force):
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
    if robot.post is not None:
        robot.post(assets)

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
