"""FR3 web task runner. Startup checks never activate or move the robot.

Source franka_ros2_ws/install/setup.bash, then run with .venv-hardware/bin/python.
--check reports readiness. Camera settings and the mounting transform come
only from configs/cameras/d435i.yaml. The existing camera monitor must be stopped
before starting this runner, because the RealSense has one pipeline owner.
"""
import argparse
import json
import os
import re
from pathlib import Path
import signal
import threading
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--port', type=int, default=8090)
    args = parser.parse_args()
    blockers = []
    try:
        urllib.request.urlopen('http://127.0.0.1:8082/health', timeout=3).close()
    except Exception as exc:
        blockers.append(f'Cosmos service unavailable: {exc}')
    controllers = {}
    import rclpy
    from controller_manager_msgs.srv import ListControllers
    rclpy.init()
    node = rclpy.create_node('cascade_fr3_preflight')
    try:
        client = node.create_client(ListControllers, '/controller_manager/list_controllers')
        if not client.wait_for_service(timeout_sec=5):
            blockers.append('Real ROS controller manager unavailable.')
        else:
            future = client.call_async(ListControllers.Request())
            rclpy.spin_until_future_complete(node, future, timeout_sec=5)
            if not future.done() or future.result() is None:
                blockers.append('Controller status timed out.')
            else:
                controllers = {c.name: c.state for c in future.result().controller}
        if controllers.get('fr3_arm_controller') != 'active':
            blockers.append('fr3_arm_controller is not active.')
        for broadcaster in ('joint_state_broadcaster', 'franka_robot_state_broadcaster'):
            if controllers.get(broadcaster) != 'active':
                blockers.append(f'{broadcaster} is not active.')
    finally:
        node.destroy_node()
        rclpy.shutdown()
    rt_threads = []
    for process in Path('/proc').glob('[0-9]*'):
        try:
            if '/controller_manager/ros2_control_node' not in (process / 'cmdline').read_bytes().decode():
                continue
            for thread in (process / 'task').iterdir():
                tid = int(thread.name)
                if os.sched_getscheduler(tid) in (os.SCHED_FIFO, os.SCHED_RR):
                    rt_threads.append(tid)
        except (OSError, ValueError):
            continue
    if not rt_threads:
        blockers.append('No real-time scheduled controller threads; administrator setup required.')
    report = {'ready': not blockers, 'controllers': controllers,
              'realtime_threads': rt_threads, 'blockers': blockers}
    print(json.dumps(report, indent=2), flush=True)
    if args.check or blockers:
        return 0 if not blockers else 2

    from cascade.config import load_demo_config
    from cascade.apps.demo import build_runtime, shutdown_runtime
    from cascade.agent.llm import make_llm
    from cascade.agent.orchestrator import AgentOrchestrator
    from cascade.agent.reflex import FastPlanner, ExperienceMemory, parse_command
    run_dir = ROOT / 'runs' / 'fr3-prepared' / 'session'
    run_dir.mkdir(parents=True, exist_ok=True)
    os.environ['CASCADE_BELIEFS_PATH'] = str(run_dir / 'beliefs.json')
    os.chdir(ROOT / 'models')
    cfg = load_demo_config(arm='fr3', camera='d435i', llm='local_cosmos')
    cfg._data['arm']['ros_state_topic'] = '/franka/joint_states'
    cfg._data['stream'].update(mode='eager', host='127.0.0.1', port=args.port)
    cfg._data['grasp']['backend'] = 'obb'
    runtime, arm = build_runtime(cfg, run_dir, serve=True)
    agent = AgentOrchestrator(make_llm(cfg.llm), runtime, max_steps=8,
                              fast_planner=FastPlanner(ExperienceMemory(run_dir / 'experience.json')))
    stopped = threading.Event()

    def task(text):
        runtime.current_task = text
        try:
            # Joint commands must never fall through to the LLM's retry/recovery
            # loop: repeating a relative rotation would double the requested angle.
            plan = parse_command(text)
            if re.search(r'\b(joint|rotate|rotation)\b', text, re.I):
                if plan is None or plan.intent != 'rotate_joint':
                    runtime.memory.add('outcome', 'Unsupported rotation wording. Use: rotate joint 7 10 degrees. No motion sent.')
                    return
                name, arguments = plan.calls[0]
                result = runtime.execute(name, arguments)
                runtime.memory.add('outcome', json.dumps(result))
                print(json.dumps(result), flush=True)
                return
            runtime.arm.get_state()  # fail before model planning on stale feedback
            result = agent.run_task(text)
            runtime.memory.add('outcome', result.summary)
            print(result, flush=True)
        except Exception as exc:
            runtime.memory.add('outcome', f'Command blocked: {exc}')
            raise
        finally:
            runtime.current_task = None

    def shutdown(*_):
        runtime.arm.stop()
        stopped.set()

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    runtime.stream_server.set_task_fn(task)
    runtime.stream_server.set_cancel_fn(runtime.arm.stop)
    print(f'FR3 task chat: http://127.0.0.1:{args.port}/', flush=True)
    try:
        stopped.wait()
    finally:
        shutdown_runtime(runtime, arm)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
