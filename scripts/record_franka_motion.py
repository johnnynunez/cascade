#!/usr/bin/env python3
r"""Read-only Franka motion evidence: native state, controller state and feedback.

Run from the repository after sourcing ROS and the Franka workspace:

    source /opt/ros/jazzy/setup.bash
    source /home/robot/franka_ros2_ws/install/setup.bash
    ROS_DOMAIN_ID=31 .venv-hardware/bin/python scripts/record_franka_motion.py \
        --output runs/real_franka_bf16/motion-diagnostic.jsonl --rate 100

Ctrl-C/SIGTERM flush pending native samples and close the JSONL file. This
script only subscribes; it never sends a robot goal, command or service call.
Interval peaks include every received native sample, even between saved states.
Receipt timestamps use monotonic and UTC clocks; ROS header/robot timestamps
are recorded separately without assuming they share the wall clock.
"""

from __future__ import annotations

import argparse
from array import array
from collections.abc import Mapping
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import signal
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cascade.control.franka_state import ROBOT_MODE_NAMES, _active_errors


def message_dict(value):
    """Serialize ROS messages without importing ROS in the sampling helpers."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(k): message_dict(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, array, bytes, bytearray)):
        return [message_dict(v) for v in value]
    # Generated ROS fixed-size numeric fields are NumPy arrays/scalars.
    # Their tolist() conversion avoids loading NumPy in this recorder.
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return message_dict(tolist())
    fields = getattr(value, "get_fields_and_field_types", None)
    if callable(fields):
        return {name: message_dict(getattr(value, name)) for name in fields()}
    if hasattr(value, "__dict__"):
        return {name: message_dict(v) for name, v in vars(value).items()}
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _vector(v) -> list[float]:
    return [float(v.x), float(v.y), float(v.z)]


class IntervalPeaks:
    """Abs-component and norm peaks across all incoming native samples."""

    def __init__(self):
        self.samples = 0
        self.wrenches = {}
        self.joint_torque_abs = []

    def add(self, msg) -> None:
        self.samples += 1
        for name in ("o_f_ext_hat_k", "k_f_ext_hat_k"):
            wrench = getattr(msg, name).wrench
            peak = self.wrenches.setdefault(name, {
                "force_abs_xyz_n": [0.0, 0.0, 0.0],
                "torque_abs_xyz_nm": [0.0, 0.0, 0.0],
                "force_norm_n": 0.0,
                "torque_norm_nm": 0.0,
            })
            for kind, units in (("force", "n"), ("torque", "nm")):
                vector = _vector(getattr(wrench, kind))
                peak[f"{kind}_abs_xyz_{units}"] = [
                    max(old, abs(new)) for old, new in
                    zip(peak[f"{kind}_abs_xyz_{units}"], vector)
                ]
                peak[f"{kind}_norm_{units}"] = max(
                    peak[f"{kind}_norm_{units}"], math.sqrt(sum(v * v for v in vector)))
        external = list(msg.tau_ext_hat_filtered.effort)
        if len(self.joint_torque_abs) != len(external):
            self.joint_torque_abs = [0.0] * len(external)
        self.joint_torque_abs = [max(old, abs(float(new))) for old, new in
                                 zip(self.joint_torque_abs, external)]

    def snapshot(self) -> dict:
        return {
            "native_samples": self.samples,
            "cartesian_wrench": message_dict(self.wrenches),
            "joint_external_torque_abs_nm": self.joint_torque_abs.copy(),
        }


class MotionRecorder:
    """Single-executor sampler. Files are flushed after every saved record."""

    def __init__(self, output, rate_hz=100.0, clock=time.monotonic, ros_clock=None):
        if not math.isfinite(rate_hz) or not 0 < rate_hz <= 1000:
            raise ValueError("rate must be positive and at most 1000 Hz")
        self.output = output
        self.period_s = 1.0 / rate_hz
        self.clock = clock
        self.ros_clock = ros_clock
        self.last_native_write = None
        self.last_controller_write = None
        self.last_native = None
        self.last_native_receipt = None
        self.last_controller = None
        self.last_controller_receipt = None
        self.last_signature = None
        self.last_robot_time = None
        self.max_native_time_gap_s = 0.0
        self.interval_max_native_time_gap_s = 0.0
        self.peaks = IntervalPeaks()
        self.counts = {"native_samples": 0, "robot_state_records": 0,
                       "controller_records": 0, "action_feedback_records": 0}

    def write(self, event, payload=None, msg=None, now=None):
        now = self.clock() if now is None else now
        record = {
            "event": event,
            "recorded_monotonic_s": now,
            "recorded_utc": datetime.now(timezone.utc).isoformat(),
        }
        if self.ros_clock is not None:
            record["recorded_ros_time_ns"] = int(self.ros_clock())
        if msg is not None and hasattr(msg, "header"):
            record["ros_header"] = message_dict(msg.header)
        if payload:
            record.update(payload)
        self.output.write(json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n")
        self.output.flush()

    def _write_native(self, now, reason):
        msg = self.last_native
        if msg is None or not self.peaks.samples:
            return
        state = message_dict(msg)
        state["robot_mode_name"] = ROBOT_MODE_NAMES.get(msg.robot_mode, "Unknown")
        # Boolean fields and active-name lists serve different inspection uses.
        state["current_error_names"] = list(_active_errors(msg.current_errors))
        state["last_motion_error_names"] = list(_active_errors(msg.last_motion_errors))
        measured = msg.measured_joint_state
        self.write("robot_state", {
            "reason": reason,
            "sample_received_monotonic_s": self.last_native_receipt,
            "q": list(measured.position),
            "dq": list(measured.velocity),
            "tau_j": list(measured.effort),
            "state": state,
            "interval_peaks": self.peaks.snapshot(),
            "interval_max_native_time_gap_s": self.interval_max_native_time_gap_s,
        }, msg=msg, now=now)
        self.counts["robot_state_records"] += 1
        self.last_native_write = now
        self.peaks = IntervalPeaks()
        self.interval_max_native_time_gap_s = 0.0

    def native(self, msg):
        now = self.clock()
        self.counts["native_samples"] += 1
        self.peaks.add(msg)
        robot_time = float(msg.time)
        if self.last_robot_time is not None:
            gap = max(0.0, robot_time - self.last_robot_time)
            self.max_native_time_gap_s = max(self.max_native_time_gap_s, gap)
            self.interval_max_native_time_gap_s = max(self.interval_max_native_time_gap_s, gap)
        self.last_robot_time = robot_time
        self.last_native, self.last_native_receipt = msg, now
        signature = (int(msg.robot_mode), _active_errors(msg.current_errors),
                     _active_errors(msg.last_motion_errors))
        changed = self.last_signature is not None and signature != self.last_signature
        self.last_signature = signature
        if self.last_native_write is None or changed or now - self.last_native_write >= self.period_s:
            self._write_native(now, "mode_or_fault_change" if changed else "sample")

    def controller(self, msg):
        now = self.clock()
        self.last_controller, self.last_controller_receipt = msg, now
        if self.last_controller_write is None or now - self.last_controller_write >= self.period_s:
            self.write("controller_state", {"state": message_dict(msg)}, msg=msg, now=now)
            self.last_controller_write = now
            self.counts["controller_records"] += 1

    def action_feedback(self, msg):
        self.write("action_feedback", {"feedback": message_dict(msg)}, msg=msg.feedback)
        self.counts["action_feedback_records"] += 1

    def finish(self):
        now = self.clock()
        self._write_native(now, "shutdown_flush")
        if (self.last_controller is not None and
                self.last_controller_receipt != self.last_controller_write):
            self.write("controller_state", {
                "reason": "shutdown_flush",
                "sample_received_monotonic_s": self.last_controller_receipt,
                "state": message_dict(self.last_controller),
            }, msg=self.last_controller, now=now)
            self.counts["controller_records"] += 1
        self.write("session_stop", {"counts": self.counts.copy(),
                                    "max_native_time_gap_s": self.max_native_time_gap_s}, now=now)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True,
                        help="New JSONL file; existing recordings are never overwritten")
    parser.add_argument("--rate", type=float, default=100.0,
                        help="Native/controller saved-state limit in Hz (default 100)")
    parser.add_argument("--duration", type=float, default=None,
                        help="Optional recording duration in seconds")
    parser.add_argument("--robot-state-topic",
                        default="/franka_robot_state_broadcaster/robot_state")
    parser.add_argument("--controller-state-topic", default="/fr3_arm_controller/controller_state")
    parser.add_argument("--action-feedback-topic",
                        default="/fr3_arm_controller/follow_joint_trajectory/_action/feedback")
    args = parser.parse_args(argv)
    if not math.isfinite(args.rate) or not 0 < args.rate <= 1000:
        parser.error("--rate must be positive and at most 1000 Hz")
    if args.duration is not None and (not math.isfinite(args.duration) or args.duration <= 0):
        parser.error("--duration must be finite and positive")
    return args


def main(argv=None):
    args = parse_args(argv)
    import rclpy
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from rclpy.signals import SignalHandlerOptions
    from control_msgs.action import FollowJointTrajectory
    from control_msgs.msg import JointTrajectoryControllerState
    from franka_msgs.msg import FrankaRobotState

    stop = threading.Event()
    old_handlers = {sig: signal.signal(sig, lambda *_: stop.set())
                    for sig in (signal.SIGINT, signal.SIGTERM)}
    node = None
    output = None
    recorder = None
    try:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        output = args.output.open("x", encoding="utf-8", buffering=1)
        rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
        node = rclpy.create_node("cascade_franka_motion_recorder", enable_rosout=False,
                                 start_parameter_services=False)
        recorder = MotionRecorder(output, rate_hz=args.rate,
                                  ros_clock=lambda: node.get_clock().now().nanoseconds)
        recorder.write("session_start", {
            "pid": os.getpid(), "ros_domain_id": os.environ.get("ROS_DOMAIN_ID"),
            "rate_hz": args.rate,
            "topics": {"robot_state": args.robot_state_topic,
                       "controller_state": args.controller_state_topic,
                       "action_feedback": args.action_feedback_topic},
        })
        # Best effort matches native/sensor publishers without demanding
        # retransmissions from a controller running its real-time loop.
        qos = QoSProfile(depth=100, reliability=ReliabilityPolicy.BEST_EFFORT)
        node.create_subscription(FrankaRobotState, args.robot_state_topic, recorder.native, qos)
        node.create_subscription(JointTrajectoryControllerState, args.controller_state_topic,
                                 recorder.controller, qos)
        node.create_subscription(FollowJointTrajectory.Impl.FeedbackMessage,
                                 args.action_feedback_topic, recorder.action_feedback, qos)
        print(f"Recording read-only motion feedback to {args.output.resolve()}", flush=True)
        started = time.monotonic()
        while rclpy.ok() and not stop.is_set():
            if args.duration is not None and time.monotonic() - started >= args.duration:
                break
            rclpy.spin_once(node, timeout_sec=0.2)
    finally:
        try:
            if recorder is not None:
                recorder.finish()
        finally:
            if node is not None:
                node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
            if output is not None:
                output.close()
            for sig, handler in old_handlers.items():
                signal.signal(sig, handler)


if __name__ == "__main__":
    main()
