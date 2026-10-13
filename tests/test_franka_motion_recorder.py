"""Native peaks survive sampling; fault records bypass the sampling interval."""

import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np


_path = Path(__file__).resolve().parents[1] / "scripts" / "record_franka_motion.py"
_spec = importlib.util.spec_from_file_location("franka_motion_recorder", _path)
recorder_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(recorder_module)


def _msg(mode=2, force=0.0, torque=0.0, external=0.0, reflex=False, stamp=0.0):
    vector = lambda x: NS(x=x, y=0.0, z=0.0)
    wrench = NS(wrench=NS(force=vector(force), torque=vector(torque)))
    return NS(robot_mode=mode, time=stamp,
              current_errors=NS(cartesian_reflex=reflex),
              last_motion_errors=NS(cartesian_reflex=reflex),
              measured_joint_state=NS(position=[0.0] * 7, velocity=[0.0] * 7, effort=[1.0] * 7),
              tau_ext_hat_filtered=NS(effort=[external] * 7),
              o_f_ext_hat_k=wrench, k_f_ext_hat_k=wrench,
              header=NS(stamp=NS(sec=7, nanosec=42), frame_id="fr3_link0"))


class _Clock:
    now = 0.0

    def __call__(self):
        return self.now


class _Output(io.StringIO):
    flush_count = 0

    def flush(self):
        self.flush_count += 1

    def records(self):
        return [json.loads(line) for line in self.getvalue().splitlines()]


def test_short_native_spike_survives_100hz_saved_state():
    clock, output = _Clock(), _Output()
    recorder = recorder_module.MotionRecorder(output, clock=clock, rate_hz=100)
    recorder.native(_msg())
    for i in range(1, 11):
        clock.now = i * 0.001
        recorder.native(_msg(force=-100 if i == 3 else 1,
                             torque=12 if i == 3 else 0,
                             external=-5 if i == 3 else 0,
                             stamp=clock.now))
    records = output.records()
    assert len(records) == 2
    assert records[-1]["state"]["o_f_ext_hat_k"]["wrench"]["force"]["x"] == 1
    peaks = records[-1]["interval_peaks"]
    assert peaks["native_samples"] == 10
    assert peaks["cartesian_wrench"]["o_f_ext_hat_k"]["force_abs_xyz_n"] == [100, 0, 0]
    assert peaks["cartesian_wrench"]["o_f_ext_hat_k"]["force_norm_n"] == 100
    assert peaks["cartesian_wrench"]["k_f_ext_hat_k"]["torque_norm_nm"] == 12
    assert peaks["joint_external_torque_abs_nm"] == [5] * 7
    assert output.flush_count == 2


def test_fault_and_mode_changes_save_immediately_with_peaks():
    clock, output = _Clock(), _Output()
    recorder = recorder_module.MotionRecorder(output, clock=clock, rate_hz=100)
    recorder.native(_msg())
    clock.now = 0.001
    recorder.native(_msg(force=45, stamp=0.001))
    clock.now = 0.002
    recorder.native(_msg(mode=4, force=30, reflex=True, stamp=0.002))
    records = output.records()
    assert len(records) == 2
    fault = records[-1]
    assert fault["reason"] == "mode_or_fault_change"
    assert fault["state"]["robot_mode_name"] == "Reflex"
    assert fault["state"]["current_error_names"] == ["cartesian_reflex"]
    assert fault["interval_peaks"]["cartesian_wrench"]["o_f_ext_hat_k"]["force_norm_n"] == 45
    assert fault["ros_header"]["stamp"] == {"sec": 7, "nanosec": 42}
    assert fault["recorded_utc"].endswith("+00:00")


def test_shutdown_flushes_pending_native_samples_and_counts():
    clock, output = _Clock(), _Output()
    recorder = recorder_module.MotionRecorder(output, clock=clock, rate_hz=100)
    recorder.native(_msg(stamp=0))
    clock.now = 0.005
    recorder.native(_msg(force=9, stamp=0.005))
    recorder.finish()
    records = output.records()
    assert records[-2]["reason"] == "shutdown_flush"
    assert records[-2]["interval_peaks"]["native_samples"] == 1
    assert records[-1]["event"] == "session_stop"
    assert records[-1]["counts"]["native_samples"] == 2
    assert records[-1]["max_native_time_gap_s"] == 0.005


def test_controller_and_action_feedback_preserve_control_fields():
    clock, output = _Clock(), _Output()
    recorder = recorder_module.MotionRecorder(output, clock=clock)
    controller = NS(reference=NS(positions=[1]), feedback=NS(positions=[0.8]),
                    error=NS(positions=[0.2]), output=NS(effort=[4]),
                    header=NS(stamp=NS(sec=1, nanosec=2)))
    recorder.controller(controller)
    action = NS(goal_id=NS(uuid=[3] * 16),
                feedback=NS(desired=NS(positions=[1]), actual=NS(positions=[0.8]),
                            error=NS(positions=[0.2]), header=controller.header))
    recorder.action_feedback(action)
    records = output.records()
    assert records[0]["state"]["output"]["effort"] == [4]
    assert records[1]["feedback"]["goal_id"]["uuid"] == [3] * 16
    assert records[1]["feedback"]["feedback"]["error"]["positions"] == [0.2]


def test_ros_fixed_size_numeric_fields_serialize_without_nan():
    value = NS(ddq_d=np.array([1.0, float("nan")]), time=np.float64(1.25))
    assert recorder_module.message_dict(value) == {"ddq_d": [1.0, None], "time": 1.25}
