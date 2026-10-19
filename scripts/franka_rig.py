#!/usr/bin/env python3
"""Franka FR3 checks for scripts/launch.sh (--sim none --arm fr3 --cameras d435i).

    franka_rig.py bringup [--arm fr3] [--timeout 30]   # native franka_ros2 bringup answers
    franka_rig.py ready [--port 8090] [--camera d435i] [--timeout 300]
                                                     # robot server: camera streaming, STOP latched,
                                                     # Resume & Home available
    franka_rig.py cpus                               # this process's CPU affinity, when narrower than the host

All read-only. `bringup` runs inside the ROS 2 environment (scripts/ros_env.sh,
or a shell that sourced the setup): it waits for the arm's joint states, the
arm controller's trajectory action and the Franka Gripper actions named in the
arm profile, and for the arm controller to be ACTIVE. It starts, switches and
commands nothing: ROS bringup stays a separate step (scripts/bringup_franka.sh).
`ready` only reads the robot server's live view (`/state`).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from urllib.request import ProxyHandler, build_opener

REPO = Path(__file__).resolve().parents[1]


def _resolve(namespace: str, name: str) -> str:
    """ros2_arm.Ros2Arm._resolve: namespace-prefix relative names."""
    if name.startswith("/"):
        return name
    return f"/{namespace}/{name}" if namespace else f"/{name}"


def bringup_targets(arm: dict) -> dict:
    """What a native bringup must offer this arm profile (configs/arms/fr3.yaml):
    the joint-state topics, every action the driver calls, and the arm
    controller that owns the trajectory action."""
    ns = str(arm.get("ros_namespace") or "").strip("/")
    topics = [_resolve(ns, str(arm["ros_state_topic"]))]
    if arm.get("ros_gripper_state_topic"):
        topics.append(_resolve(ns, str(arm["ros_gripper_state_topic"])))
    actions = [_resolve(ns, str(arm[key])) for key in (
        "ros_trajectory_action", "ros_gripper_action", "ros_gripper_move_action", "ros_gripper_grasp_action")
        if arm.get(key)]
    controller = None
    if arm.get("ros_trajectory_action"):
        controller = str(arm["ros_trajectory_action"]).strip("/").split("/")[0]
    return {"topics": list(dict.fromkeys(topics)), "actions": list(dict.fromkeys(actions)),
            "controller": controller}


def load_arm(name: str) -> dict:
    sys.path.insert(0, str(REPO / "src"))
    from cascade.config import load_profile

    return load_profile("arms", name).as_dict()


def check_bringup(targets: dict, timeout_s: float) -> dict:
    """Wait (read-only) until the bringup offers every target. Raises
    RuntimeError naming what is missing."""
    import rclpy
    from sensor_msgs.msg import JointState

    rclpy.init()
    node = rclpy.create_node("cascade_franka_bringup_check")
    seen: dict[str, bool] = {topic: False for topic in targets["topics"]}
    subs = [node.create_subscription(JointState, topic, (lambda _msg, t=topic: seen.__setitem__(t, True)), 10)
            for topic in targets["topics"]]
    deadline = time.monotonic() + timeout_s
    missing: list[str] = []
    controller_state = None
    try:
        while True:
            rclpy.spin_once(node, timeout_sec=0.2)
            services = {name for name, _ in node.get_service_names_and_types()}
            missing = [f"no JointState on {t}" for t, ok in seen.items() if not ok]
            missing += [f"no action server {a}" for a in targets["actions"]
                        if f"{a}/_action/send_goal" not in services]
            if not missing and targets.get("controller"):
                controller_state = _controller_state(node, targets["controller"], deadline)
                if controller_state != "active":
                    missing.append(f"controller {targets['controller']} is {controller_state or 'not loaded'}")
            if not missing or time.monotonic() >= deadline:
                break
        if missing:
            raise RuntimeError("; ".join(missing))
        return {**targets, "controller_state": controller_state, "ros_domain_id": os.environ.get("ROS_DOMAIN_ID", "0")}
    finally:
        for sub in subs:
            node.destroy_subscription(sub)
        node.destroy_node()
        rclpy.shutdown()


def _controller_state(node, controller: str, deadline: float) -> str | None:
    """The controller manager's state for `controller` (list_controllers)."""
    import rclpy
    from controller_manager_msgs.srv import ListControllers

    client = node.create_client(ListControllers, "/controller_manager/list_controllers")
    try:
        if not client.wait_for_service(timeout_sec=max(0.1, min(5.0, deadline - time.monotonic()))):
            return "unknown (no /controller_manager/list_controllers)"
        future = client.call_async(ListControllers.Request())
        rclpy.spin_until_future_complete(node, future, timeout_sec=max(0.1, min(5.0, deadline - time.monotonic())))
        result = future.result()
        if result is None:
            return "unknown (list_controllers timed out)"
        return next((c.state for c in result.controller if c.name == controller), None)
    finally:
        node.destroy_client(client)


def runtime_ready(view: dict, camera: str | None = None) -> list[str]:
    """Why the robot server's live view is NOT the Franka operator contract
    yet (empty = ready): every camera streaming without error, the software
    STOP latched, and Resume & Home available on the operator page."""
    problems = []
    cams = view.get("cameras") or {}
    if not cams:
        problems.append("no camera announced")
    if camera and camera not in cams:
        problems.append(f"camera {camera} not announced")
    for name, row in cams.items():
        if row.get("error"):
            problems.append(f"camera {name}: {row['error']}")
        elif not (row.get("fps") or 0) > 0:
            problems.append(f"camera {name} has no frames yet")
    if view.get("stopped") is not True:
        problems.append("software STOP is not latched")
    if view.get("resume_supported") is not True:
        problems.append("Resume & Home is not available")
    return problems


def wait_ready(port: int, timeout_s: float, camera: str | None = None, *, get=None, sleep=time.sleep) -> dict:
    url = f"http://127.0.0.1:{port}/state"
    if get is None:
        opener = build_opener(ProxyHandler({}))

        def _get(u):
            with opener.open(u, timeout=3) as response:
                return json.loads(response.read(2 * 1024 * 1024))
        get = _get
    deadline = time.monotonic() + timeout_s
    problems = ["live view not answering"]
    while True:
        try:
            view = get(url)
            problems = runtime_ready(view, camera)
            if not problems:
                return {"cameras": sorted(view.get("cameras") or {}), "stopped": True, "resume_supported": True,
                        "operator_page": f"http://127.0.0.1:{port}/"}
        except (OSError, ValueError) as exc:
            problems = [f"live view {url} not answering ({exc})"]
        if time.monotonic() >= deadline:
            raise RuntimeError("; ".join(problems))
        sleep(0.5)


def cpu_ranges(cpus) -> str:
    """{0,1,2,3,8} -> '0-3,8' (systemd CPUAffinity= / taskset -c syntax)."""
    out, run = [], []
    for cpu in sorted(cpus):
        if run and cpu == run[-1] + 1:
            run.append(cpu)
            continue
        if run:
            out.append(f"{run[0]}-{run[-1]}" if len(run) > 1 else str(run[0]))
        run = [cpu]
    if run:
        out.append(f"{run[0]}-{run[-1]}" if len(run) > 1 else str(run[0]))
    return ",".join(out)


def narrowed_affinity(affinity=None, online=None) -> str:
    """This process's CPU list when it is narrower than the host's (e.g. the
    launcher ran under `taskset -c 0-7`), else ''."""
    affinity = set(os.sched_getaffinity(0) if affinity is None else affinity)
    online = set(range(os.cpu_count() or 0)) if online is None else set(online)
    return cpu_ranges(affinity) if affinity and affinity < online else ""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("bringup", help="wait for the native franka_ros2 bringup (read-only)")
    b.add_argument("--arm", default="fr3")
    b.add_argument("--timeout", type=float, default=30.0)
    r = sub.add_parser("ready", help="wait for the robot server's operator contract (read-only)")
    r.add_argument("--port", type=int, default=int(os.environ.get("CASCADE_STREAM_PORT") or 8090))
    r.add_argument("--camera", default=None)
    r.add_argument("--timeout", type=float, default=300.0)
    sub.add_parser("cpus", help="print this process's CPU list when narrower than the host's")
    args = ap.parse_args(argv)
    try:
        if args.cmd == "bringup":
            result = check_bringup(bringup_targets(load_arm(args.arm)), args.timeout)
        elif args.cmd == "ready":
            result = wait_ready(args.port, args.timeout, args.camera)
        else:
            print(narrowed_affinity())
            return 0
    except (RuntimeError, ImportError, OSError, KeyError) as exc:
        print(f"[franka] {args.cmd}: {exc}", file=sys.stderr)
        return 3
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
