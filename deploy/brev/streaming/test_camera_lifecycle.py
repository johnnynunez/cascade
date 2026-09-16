"""Local lifecycle and shutdown checks; no simulator, GPU or remote calls."""

import ast
from contextlib import ExitStack, redirect_stdout
import io
import json
import os
from pathlib import Path
import signal
import shutil
import socketserver
import subprocess
import sys
import tempfile
import threading
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from camera_stream_lifecycle import CameraVideoLifecycle, camera_video_from_environment
from camera_stream_profile import StreamProfile


CAMPAIGN = Path(__file__).resolve().parents[1]
BRIDGE = CAMPAIGN.parents[1] / "scripts/isaac_bridge.py"


class FakeOwner:
    def __init__(self, events):
        self.events = events
        self.state = "stopped"
        self.start_error = self.stop_error = self.restart_error = None
        self.sensors = None

    def start(self, sensors, *, verify_rtsp_isolation):
        verify_rtsp_isolation((8554, 8555, 8556))
        self.sensors = sensors
        self.events.append("attach")
        self.state = "attached"
        if self.start_error is not None:
            raise self.start_error

    def restart_camera(self, name):
        self.events.append(("restart", name, threading.current_thread().ident))
        if self.restart_error is not None:
            raise self.restart_error

    def stop(self):
        self.events.append("video.stop")
        if self.stop_error is not None:
            error, self.stop_error = self.stop_error, None
            raise error
        self.state = "stopped"

    def status(self):
        return {"state": self.state, "live_verified": False,
                "cameras": {"kitchen": {"render_product": "/owned/video"}}}


class LifecycleTests(unittest.TestCase):
    def setUp(self):
        self.events = []
        self.now = 50.0
        self.guard_error = None
        self.owner = FakeOwner(self.events)
        self.video = CameraVideoLifecycle(
            StreamProfile("100.80.90.10", "http://100.80.90.10:8092"),
            owner_factory=self.factory, verify_isolation=self.guard, clock=lambda: self.now,
        )
        self.sensors = {name: object() for name in ("cam0", "side", "proof", "wrist")}

    def factory(self, profile):
        self.events.append("factory")
        return self.owner

    def guard(self, ports):
        self.assertEqual(ports, (8554, 8555, 8556))
        self.events.append("guard")
        if self.guard_error is not None:
            raise self.guard_error

    def test_absent_empty_and_disabled_config_import_no_owner_or_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "video.json"
            path.write_text(json.dumps({"schema": 1, "enabled": False}))
            with patch.dict(sys.modules, {"camera_streams": None, "network_guard": None}):
                for env in ({}, {"PAAI_CAMERA_VIDEO_CONFIG": ""},
                            {"PAAI_CAMERA_VIDEO_CONFIG": str(path)}):
                    with self.subTest(env=env):
                        self.assertIsNone(camera_video_from_environment(env))

    def test_enabled_config_defers_native_creation_and_accepts_selected_ports(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "video.json"
            path.write_text(json.dumps({"schema": 1, "enabled": True,
                "tailnet_ipv4": "100.80.90.10", "visitor_origin": "http://100.80.90.10:8092",
                "rtsp_ports": [8557, 8558, 8559], "http_port": 8890, "media_port": 8190}))
            with patch.dict(sys.modules, {"camera_streams": None, "network_guard": None}):
                video = camera_video_from_environment({"PAAI_CAMERA_VIDEO_CONFIG": str(path)})
            self.assertEqual(video.profile.rtsp_ports, (8557, 8558, 8559))
            self.assertEqual(video.status()["state"], "pending")

    def test_config_rejects_unknown_schema_implicit_enable_paths_and_unbounded_input(self):
        invalid = [[], {"schema": True, "enabled": True}, {"schema": 1},
                   {"schema": 1, "enabled": "true"},
                   {"schema": 1, "enabled": False, "admission_path": "/tmp/forged"},
                   {"schema": 1, "enabled": True},
                   {"schema": 1, "enabled": True, "rtsp_ports": "8554"}]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "video.json"
            for value in invalid:
                path.write_text(json.dumps(value))
                with self.subTest(value=value), self.assertRaises(ValueError):
                    camera_video_from_environment({"PAAI_CAMERA_VIDEO_CONFIG": str(path)})
            path.write_text(" " * 8193)
            with self.assertRaisesRegex(ValueError, "8192"):
                camera_video_from_environment({"PAAI_CAMERA_VIDEO_CONFIG": str(path)})
        with self.assertRaisesRegex(ValueError, "absolute"):
            camera_video_from_environment({"PAAI_CAMERA_VIDEO_CONFIG": "relative.json"})

    def test_start_checks_guard_before_construction_and_borrows_unchanged_sensors(self):
        native_ids = {name: id(sensor) for name, sensor in self.sensors.items()}
        result = self.video.start(self.sensors)
        self.assertEqual(self.events, ["guard", "factory", "guard", "attach"])
        self.assertIs(self.owner.sensors, self.sensors)
        self.assertEqual(native_ids, {name: id(sensor) for name, sensor in self.sensors.items()})
        self.assertEqual(result["state"], "attached")
        self.assertIs(result["live_verified"], False)
        with self.assertRaises(RuntimeError):
            self.video.start(self.sensors)

    def test_guard_denial_creates_no_native_resources_and_does_not_throw_to_physics(self):
        self.guard_error = RuntimeError("admission expired")
        result = self.video.start(self.sensors)
        self.assertEqual(self.events, ["guard"])
        self.assertEqual(result["state"], "failed")
        self.assertIn("admission expired", result["error"])
        self.assertFalse(self.video.poll())

    def test_partial_start_and_cleanup_failure_are_visible_and_retry_at_shutdown(self):
        self.owner.start_error = RuntimeError("partial attach")
        self.owner.stop_error = RuntimeError("detach failed")
        result = self.video.start(self.sensors)
        self.assertEqual(result["state"], "failed")
        self.assertIn("partial attach", result["error"])
        self.assertIn("detach failed", result["cleanup_error"])
        self.video.stop()
        self.assertEqual(self.owner.state, "stopped")
        self.assertEqual(self.events.count("video.stop"), 2)

    def test_periodic_freshness_check_runs_at_five_seconds_and_failure_stops_only_video(self):
        self.video.start(self.sensors)
        self.events.clear()
        self.now = 54.99
        self.assertFalse(self.video.poll())
        self.assertEqual(self.events, [])
        self.now = 55.0
        self.assertFalse(self.video.poll())
        self.assertEqual(self.events, ["guard"])
        self.now = 60.0
        self.guard_error = RuntimeError("changed host identity")
        self.assertTrue(self.video.poll())
        self.assertEqual(self.events, ["guard", "guard", "video.stop"])
        self.assertEqual(self.video.status()["state"], "failed")
        self.guard_error = None
        self.now = 100.0
        self.assertFalse(self.video.poll())
        with self.assertRaisesRegex(RuntimeError, "reconciliation"):
            self.video.restart_camera("kitchen")
        self.assertEqual(self.events, ["guard", "guard", "video.stop"])

    def test_restart_rechecks_admission_and_preserves_sensor_mapping(self):
        self.video.start(self.sensors)
        self.events.clear()
        self.video.restart_camera("side")
        self.assertEqual(self.events, ["guard", ("restart", "side", threading.main_thread().ident)])
        self.assertIs(self.owner.sensors, self.sensors)
        with self.assertRaises(ValueError):
            self.video.restart_camera("../kitchen")

    def test_restart_failure_stops_all_owned_outputs_without_automatic_retry(self):
        self.video.start(self.sensors)
        self.events.clear()
        self.owner.restart_error = RuntimeError("writer attach failed")
        result = self.video.restart_camera("kitchen")
        self.assertEqual(result["state"], "failed")
        self.assertEqual(self.events[-1], "video.stop")
        self.assertFalse(self.video.poll())

    def test_guard_denial_on_restart_never_reaches_writer(self):
        self.video.start(self.sensors)
        self.events.clear()
        self.guard_error = RuntimeError("revoked")
        self.assertEqual(self.video.restart_camera("side")["state"], "failed")
        self.assertEqual(self.events, ["guard", "video.stop"])

    def test_native_operations_are_rejected_on_socket_thread(self):
        errors = []
        def wrong_thread():
            for action in (lambda: self.video.start(self.sensors), self.video.poll,
                           lambda: self.video.restart_camera("side"), self.video.stop):
                try:
                    action()
                except RuntimeError as exc:
                    errors.append(str(exc))
        worker = threading.Thread(target=wrong_thread)
        worker.start()
        worker.join(timeout=2)
        self.assertFalse(worker.is_alive())
        self.assertEqual(len(errors), 4)
        self.assertTrue(all("main thread" in error for error in errors))
        self.assertEqual(self.events, [])


class BridgeBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tree = ast.parse(BRIDGE.read_text(), filename=str(BRIDGE))
        cls.functions = {node.name: node for node in cls.tree.body if isinstance(node, ast.FunctionDef)}

    def setUp(self):
        self.events = []
        self.ns = {"__name__": "bridge_lifecycle_fixture", "os": os, "json": json,
                   "sys": sys, "_REPO_ROOT": str(BRIDGE.parents[1]),
                   "threading": threading, "socketserver": socketserver,
                   "_camera_video": None, "_camera_video_startup_error": None,
                   "_shutdown_requested": False, "_exec_lock": threading.Lock(), "_exec_jobs": []}
        nodes = [self.functions[name] for name in (
            "_request_shutdown", "_bridge_should_stop", "_camera_video_status", "_run_exec_jobs")]
        nodes.append(next(node for node in self.tree.body if isinstance(node, ast.ClassDef) and node.name == "Handler"))
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(BRIDGE), "exec"), self.ns)

    def lifecycle(self, *, video=True, initial_stop=False, update=None, stop_error=None,
                  log_error=None, startup_error=None, config_error=None, playing=False,
                  iterations=2, guard=None):
        owner = FakeOwner(self.events)
        owner.stop_error = stop_error
        owner.start_error = startup_error
        clock = lambda: 50.0 + ticks[0] * 5.0
        streams = CameraVideoLifecycle(
            StreamProfile("100.80.90.10", "http://100.80.90.10:8092"),
            owner_factory=lambda _: owner, verify_isolation=guard or (lambda _: None), clock=clock)
        ticks = [0]
        def app_update():
            ticks[0] += 1
            self.events.append("app.update")
            if update is not None:
                update()
        def configure():
            if config_error is not None:
                raise config_error
            return streams
        def remove_log(consumer):
            self.events.append("log.remove")
            if log_error is not None:
                raise log_error
        fake_signal = SimpleNamespace(SIGTERM=signal.SIGTERM)
        previous = object()
        handlers = [previous]
        def install(signum, handler):
            old, handlers[0] = handlers[0], handler
            self.events.append("signal.install" if handler is not previous else "signal.restore")
            return old
        fake_signal.signal = install
        self.ns.update(
            signal=fake_signal, ExitStack=ExitStack, _REQUIRE_CUDA=False,
            _tl=SimpleNamespace(is_playing=lambda: playing), _was_playing=True,
            app=SimpleNamespace(is_running=lambda: ticks[0] < iterations, update=app_update,
                                close=lambda: self.events.append("app.close")),
            app_utils=SimpleNamespace(stop=lambda: self.events.append("app_utils.stop")),
            server=SimpleNamespace(shutdown=lambda: self.events.append("server.shutdown"),
                                   server_close=lambda: self.events.append("server.close")),
            _gpu_log_consumer=object(),
            omni=SimpleNamespace(log=SimpleNamespace(get_log=lambda: SimpleNamespace(remove_message_consumer=remove_log))),
            CAM_DEFS={name: (object(), object()) for name in ("cam0", "side", "proof", "wrist")},
            step=0, args=SimpleNamespace(cam_every=4), _state_lock=threading.Lock(),
            _targets={"q": None, "grip_frac": None, "stopped": True},
            _update_wrist_cam=lambda: self.events.append("wrist.update"),
            _refresh_frames=lambda: self.events.append("frames.refresh"),
        )
        start = next(i for i, node in enumerate(self.tree.body)
                     if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name)
                        and target.id == "_previous_sigterm" for target in node.targets))
        body = ast.Module(body=self.tree.body[start:], type_ignores=[])
        module = ModuleType("camera_stream_lifecycle")
        module.camera_video_from_environment = configure
        env = {"PAAI_CAMERA_VIDEO_CONFIG": "/fixture/video.json"} if video else {}
        with patch.dict(os.environ, env, clear=True), patch.object(sys, "path", list(sys.path)), \
                patch.dict(sys.modules, {"camera_stream_lifecycle": module}), \
                patch("os.path.lexists", return_value=initial_stop), redirect_stdout(io.StringIO()):
            exec(compile(body, str(BRIDGE), "exec"), self.ns)
        self.assertIs(handlers[0], previous)
        return streams, owner

    def test_opt_out_leaves_camera_owner_unused_and_runs_original_loop(self):
        streams, owner = self.lifecycle(video=False)
        self.assertEqual(streams.status()["state"], "pending")
        self.assertIsNone(owner.sensors)
        self.assertEqual(self.events.count("app.update"), 2)
        self.assertNotIn("video.stop", self.events)
        self.assertEqual(self.events[-2:], ["app.close", "signal.restore"])

    def test_owner_starts_with_initialized_cameras_and_stops_before_application(self):
        streams, owner = self.lifecycle()
        self.assertEqual(self.events[1], "attach")
        self.assertEqual(set(owner.sensors), set(self.ns["CAM_DEFS"]))
        for name, sensor in owner.sensors.items():
            self.assertIs(sensor, self.ns["CAM_DEFS"][name][0])
        self.assertEqual(self.events[-7:], ["video.stop", "server.shutdown", "server.close", "log.remove",
                                             "app_utils.stop", "app.close", "signal.restore"])

    def test_cleanup_error_cannot_skip_server_log_app_or_signal_restoration(self):
        with self.assertRaisesRegex(RuntimeError, "detach failure"):
            self.lifecycle(stop_error=RuntimeError("detach failure"))
        self.assertEqual(self.events[-7:], ["video.stop", "server.shutdown", "server.close", "log.remove",
                                             "app_utils.stop", "app.close", "signal.restore"])

    def test_log_cleanup_error_still_closes_application(self):
        with self.assertRaisesRegex(RuntimeError, "log cleanup"):
            self.lifecycle(log_error=RuntimeError("log cleanup"))
        self.assertEqual(self.events[-3:], ["app_utils.stop", "app.close", "signal.restore"])

    def test_sigterm_only_requests_exit_and_cleans_up_on_main_thread(self):
        def terminate():
            before = list(self.events)
            self.ns["_request_shutdown"](signal.SIGTERM, None)
            self.assertEqual(self.events, before)
        self.lifecycle(update=terminate)
        self.assertEqual(self.events.count("app.update"), 1)
        self.assertLess(self.events.index("video.stop"), self.events.index("app.close"))

    def test_stop_marker_prevents_attachment_and_any_update(self):
        self.lifecycle(initial_stop=True)
        self.assertNotIn("attach", self.events)
        self.assertNotIn("app.update", self.events)
        self.assertTrue(self.ns["_shutdown_requested"])
        self.assertIn("app.close", self.events)

    def test_stop_marker_is_latched_and_never_removed(self):
        with tempfile.TemporaryDirectory() as directory:
            stop = Path(directory) / "STOP"
            stop.touch()
            lexists = os.path.lexists
            with patch("os.path.lexists", side_effect=lambda path: lexists(stop)) as check:
                self.assertTrue(self.ns["_bridge_should_stop"]())
                check.assert_called_with("/data/acceptance/STOP")
            self.assertTrue(stop.exists())
            with patch("os.path.lexists", return_value=False):
                self.assertTrue(self.ns["_bridge_should_stop"]())

    def test_start_interruption_keeps_owner_for_final_cleanup(self):
        with self.assertRaises(KeyboardInterrupt):
            self.lifecycle(startup_error=KeyboardInterrupt())
        self.assertIn("attach", self.events)
        self.assertEqual(self.events[-7:], ["video.stop", "server.shutdown", "server.close", "log.remove",
                                             "app_utils.stop", "app.close", "signal.restore"])

    def test_bad_optional_config_is_visible_while_original_loop_keeps_running(self):
        self.lifecycle(config_error=ValueError("invalid optional profile"))
        status = self.ns["_camera_video_status"]()
        self.assertEqual(status["state"], "failed")
        self.assertIn("invalid optional profile", status["error"])
        self.assertEqual(self.events.count("app.update"), 2)
        self.assertNotIn("attach", self.events)

    def test_private_restart_dispatch_queues_work_onto_main_thread(self):
        called = []
        video = SimpleNamespace(restart_camera=lambda name: called.append((name, threading.get_ident())),
                                status=lambda: {"state": "attached", "live_verified": False})
        self.ns["_camera_video"] = video
        handler = self.ns["Handler"].__new__(self.ns["Handler"])
        results, queued = [], threading.Event()
        class Jobs(list):
            def append(self, job):
                super().append(job)
                queued.set()
        self.ns["_exec_jobs"] = Jobs()
        worker = threading.Thread(target=lambda: results.append(handler._dispatch(
            {"op": "camera_video", "action": "restart", "camera": "side"})))
        with patch("os.path.lexists", return_value=False):
            worker.start()
            self.assertTrue(queued.wait(timeout=1))
            self.assertEqual(called, [])
            self.ns["_run_exec_jobs"]()
            worker.join(timeout=2)
        self.assertFalse(worker.is_alive())
        self.assertEqual(called, [("side", threading.main_thread().ident)])
        self.assertTrue(results[0]["ok"])

    def test_shutdown_rejects_already_queued_restart_and_new_requests(self):
        called = []
        holder, done = {}, threading.Event()
        self.ns["_exec_jobs"].append((lambda: called.append("unsafe restart"), holder, done))
        self.ns["_shutdown_requested"] = True
        with patch("os.path.lexists", return_value=False):
            self.ns["_run_exec_jobs"]()
            handler = self.ns["Handler"].__new__(self.ns["Handler"])
            response = handler._dispatch({"op": "camera_video", "action": "restart", "camera": "side"})
        self.assertTrue(done.is_set())
        self.assertFalse(holder["resp"]["ok"])
        self.assertFalse(response["ok"])
        self.assertEqual(called, [])
        self.assertEqual(self.ns["_exec_jobs"], [])

    def test_playing_loop_retains_native_camera_cadence_with_optional_video(self):
        self.lifecycle(playing=True, iterations=8)
        self.assertEqual(self.events.count("app.update"), 8)
        self.assertEqual(self.events.count("wrist.update"), 2)
        self.assertEqual(self.events.count("frames.refresh"), 2)
        for index, event in enumerate(self.events):
            if event == "wrist.update":
                self.assertEqual(self.events[index + 1:index + 3], ["app.update", "frames.refresh"])

    def test_playing_loop_continues_after_guard_failure_and_keeps_perception_cadence(self):
        def guard(_ports):
            if "app.update" in self.events:
                raise RuntimeError("host receipt expired")
        streams, owner = self.lifecycle(playing=True, iterations=8, guard=guard)
        self.assertEqual(self.events.count("attach"), 1)
        self.assertEqual(self.events.count("app.update"), 8)
        self.assertEqual(self.events.count("frames.refresh"), 2)
        self.assertIn("host receipt expired", streams.status()["error"])
        self.assertEqual(owner.state, "stopped")

    def test_playing_sigterm_cancels_jobs_queued_during_final_render_before_cleanup(self):
        holder, done, called = {}, threading.Event(), []
        def terminate():
            self.ns["_exec_jobs"].append((lambda: called.append("motion"), holder, done))
            self.ns["_request_shutdown"](signal.SIGTERM, None)
        self.lifecycle(playing=True, iterations=8, update=terminate)
        self.assertEqual(self.events.count("app.update"), 1)
        self.assertNotIn("frames.refresh", self.events)
        self.assertEqual(called, [])
        self.assertTrue(done.is_set())
        self.assertFalse(holder["resp"]["ok"])
        self.assertEqual(self.ns["_exec_jobs"], [])

    def test_shutdown_between_request_check_and_queue_lock_cannot_leave_a_late_job(self):
        namespace = self.ns
        class ShutdownLock:
            def __enter__(self):
                namespace["_shutdown_requested"] = True
            def __exit__(self, *_):
                return False
        self.ns["_exec_lock"] = ShutdownLock()
        handler = self.ns["Handler"].__new__(self.ns["Handler"])
        with patch("os.path.lexists", return_value=False):
            result = handler._on_main(lambda: {"ok": True})
        self.assertFalse(result["ok"])
        self.assertEqual(self.ns["_exec_jobs"], [])

    def test_optional_import_resolves_packaged_modules_in_clean_isolated_interpreter(self):
        block = ast.unparse(self.tree.body[-1].body[0])
        modules = ("camera_stream_lifecycle", "camera_stream_profile", "camera_streams", "network_guard")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "deploy/brev/streaming"
            target.mkdir(parents=True)
            for name in modules:
                shutil.copy2(CAMPAIGN / "streaming" / (name + ".py"), target)
            config = root / "video.json"
            config.write_text(json.dumps({"schema": 1, "enabled": False}))
            program = '''
import contextlib, importlib, io, json, os, pathlib, sys
data = json.load(sys.stdin)
target = pathlib.Path(data["root"]) / "deploy/brev/streaming"
assert str(target) not in sys.path
assert not any(name in sys.modules for name in data["modules"])
ns = {"os": os, "sys": sys, "json": json, "_REPO_ROOT": data["root"],
      "_bridge_should_stop": lambda: False, "_camera_video_status": lambda: {},
      "_camera_video": None, "_camera_video_startup_error": None}
initial_path = list(sys.path)
os.environ.pop("PAAI_CAMERA_VIDEO_CONFIG", None)
exec(compile(data["block"], "bridge-opt-in", "exec"), ns)
assert sys.path == initial_path
assert not any(name in sys.modules for name in data["modules"])
os.environ["PAAI_CAMERA_VIDEO_CONFIG"] = data["config"]
with contextlib.redirect_stdout(io.StringIO()):
    exec(compile(data["block"], "bridge-opt-in", "exec"), ns)
assert ns["_camera_video_startup_error"] is None, ns["_camera_video_startup_error"]
assert ns["_camera_video"] is None
assert sys.path[0] == str(target)
paths = {}
for name in data["modules"]:
    module = importlib.import_module(name)
    paths[name] = str(pathlib.Path(module.__file__).resolve())
    assert pathlib.Path(paths[name]).parent == target
assert not any(name.split(".")[0] in {"omni", "isaacsim", "pxr"} for name in sys.modules)
print(json.dumps({"modules": paths, "native_imports": False, "isolated": bool(sys.flags.isolated)}))
'''
            result = subprocess.run([sys.executable, "-I", "-B", "-c", program],
                input=json.dumps({"block": block, "root": str(root), "config": str(config), "modules": modules}),
                capture_output=True, text=True, timeout=10, cwd=root)
            self.assertEqual(result.returncode, 0, result.stderr)
            proof = json.loads(result.stdout)
            self.assertEqual(set(proof["modules"]), set(modules))
            self.assertTrue(proof["isolated"])
            self.assertFalse(proof["native_imports"])

    def test_retained_stop_exits_complete_bridge_before_any_native_import(self):
        with tempfile.TemporaryDirectory() as directory:
            stop = Path(directory) / "STOP"
            stop.write_text("Retained operator stop\n")
            program = '''
import builtins, json, os, pathlib, sys
source, stop = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
sys.argv = [str(source)]
original_import, original_lexists = builtins.__import__, os.path.lexists
native, checked = [], []
def guarded_import(name, *args, **kwargs):
    if name.split(".")[0] in {"isaacsim", "omni", "pxr", "carb", "numpy", "isaac_runtime"}:
        native.append(name)
        raise AssertionError("Native import attempted before retained STOP admission")
    return original_import(name, *args, **kwargs)
def fixture_lexists(path):
    if path == "/data/acceptance/STOP":
        checked.append(path)
        return original_lexists(stop)
    return original_lexists(path)
def refuse_removal(*args, **kwargs):
    raise AssertionError("Bridge must not clear a retained STOP")
builtins.__import__, os.path.lexists = guarded_import, fixture_lexists
os.unlink = os.remove = refuse_removal
try:
    exec(compile(source.read_text(), str(source), "exec"), {"__name__": "__main__", "__file__": str(source)})
except SystemExit as exc:
    reason = str(exc)
else:
    raise AssertionError("Bridge did not exit with a retained STOP")
assert "retained STOP" in reason and "/data/acceptance/STOP" in reason, reason
assert checked == ["/data/acceptance/STOP"], checked
assert not native, native
assert stop.read_text() == "Retained operator stop\\n"
print(json.dumps({"reason": reason, "native_imports": native, "stop_retained": stop.exists()}))
'''
            result = subprocess.run([sys.executable, "-I", "-B", "-c", program, str(BRIDGE), str(stop)],
                                    capture_output=True, text=True, timeout=10, cwd=directory)
            self.assertEqual(result.returncode, 0, result.stderr)
            proof = json.loads(result.stdout)
            self.assertEqual(proof["native_imports"], [])
            self.assertTrue(proof["stop_retained"])
            self.assertTrue(stop.exists())


if __name__ == "__main__":
    unittest.main()
