"""Bounded mobile skill dispatch, distinct from manipulation semantics."""
from __future__ import annotations

import base64
import copy
import threading
import time
import uuid
from collections import deque


MOTION_SKILLS = frozenset({"walk_velocity", "walk_distance", "turn"})
SYSTEM_PROMPT = """You control a base-only mobile robot, not an arm.
Use only the supplied capability tools and exact base names. There is no
TCP, gripper, arm IK, table-frame grasping, obstacle avoidance or navigation map.
Read observations before commanding bounded body-frame velocities in SI units.
An ACK/execution_ok is not physical success: only independent confirmed evidence
can establish an outcome. Kinematic mocks are software fixtures, never physics.
Respect faults/stops; reset_stop grants new permission only and never resumes a
command. Do not reset automatically after a safety fault. Call one tool per turn.
Use task_done(success=false) when the requested outcome is refuted or unverified.
"""


def _spec(name, description, properties=None, required=()):
    return {"name": name, "description": description,
            "parameters": {"type": "object", "properties": properties or {},
                           "required": list(required), "additionalProperties": False}}


_BASE = {"base": {"type": "string", "description": "Exact base name; omitted selects primary."}}
TOOL_SPECS = [
    _spec("list_bases", "Read configured base metadata and capabilities. Does not connect or actuate."),
    _spec("get_base_state", "Read measured base state, identity and clock. Kinematic mocks are not physics.", _BASE),
    _spec("walk_velocity", "Bounded body-frame SI velocity. Only independently confirmed execution is ok.",
          {**_BASE, **{k: {"type": "number"} for k in ("vx", "vy", "wz", "duration_s")}},
          ("vx", "vy", "wz", "duration_s")),
    _spec("walk_distance", "Travel a signed measured forward distance (metres); profile-owned policy command, bounded deadline, independent verification. Does not promise velocity tracking.",
          {**_BASE, "distance_m": {"type": "number"}}, ("distance_m",)),
    _spec("turn", "Turn by measured yaw (radians); bounded deadline and independent verification.",
          {**_BASE, "angle_rad": {"type": "number"}}, ("angle_rad",)),
    _spec("stop_navigation", "Priority cancellation to zero twist; pending/active dispatch is latched until reset_stop. Does not disable balance torque."),
    _spec("emergency_stop", "Invalidate pending/active commands and latch all bases. ACK is not physical rest confirmation."),
    _spec("reset_stop", "Clear permission only; never replay motion or reset the scene. Refused during an active call."),
    _spec("get_observation", "Passive measured base state; no arm frame or invented camera image.", _BASE),
    _spec("recall_memory", "Read this session's action/outcome memory.", {"query": {"type": "string"}}),
    _spec("task_memory", "Recent actions and verdicts. No camera means no visual evidence.",
          {"k": {"type": "integer"}, "new_task": {"type": "boolean"}}),
    _spec("verify_last_action", "Read independent motion verdicts and bounded post-ACK stop observations, bound to exact receipts. ACKs remain unverified."),
    _spec("task_done", "End task honestly; unverified motion cannot count as success.",
          {"success": {"type": "boolean"}, "summary": {"type": "string"}}, ("success", "summary")),
]


def tool_specs_for_profiles(profiles):
    from ..sim.mobile_frames import camera_profiles

    capabilities = set().union(*(set(p["capabilities"]) for p in profiles))
    cameras = [camera_profiles(p) for p in profiles]  # validate ALL, including secondary bases
    specs = copy.deepcopy([s for s in TOOL_SPECS
                           if s["name"] not in MOTION_SKILLS or s["name"] in capabilities])
    if any(cameras):
        selectors = {**_BASE, "camera": {"type": "string", "description": "Exact configured camera name."}}
        specs.append(_spec("camera_snapshot", "Passive source-bound RGB JPEG. No depth, TCP or navigation map.", selectors))
        next(s for s in specs if s["name"] == "get_observation")["parameters"]["properties"] = copy.deepcopy(selectors)
    return specs


def _without_images(value):
    """Trace/memory already own the exact JPEG sidecar; don't duplicate base64."""
    if isinstance(value, dict):
        return {k: _without_images(v) for k, v in value.items() if not k.endswith("_jpeg_b64")}
    if isinstance(value, list):
        return [_without_images(v) for v in value]
    return copy.deepcopy(value)


def _unverified(reason):
    return {"status": "unverified", "reason": reason}



class _StopTruth:
    """Independent passive states; the checker owns the ACK/temporal boundary.

    None is a real channel failure, never a healthy temporal exclusion.
    """
    def __init__(self, profile):
        from ..sim.base_truth import BaseTruthReader
        self.reader = BaseTruthReader(profile)
        self.job = None

    def __call__(self):
        state = self.reader()
        if state is None:
            raise RuntimeError(self.reader.last_error or "missing independent stop state")
        return state

    def close(self):
        self.reader.close()


class _StopObserver:
    """One coordinator + checker's one sampler per base, ONE latest mailbox.

    The priority path never calls begin/finish/close or waits on their locks.
    An uncooperative injected/native reader is quarantined, never multiplied.
    All production IO has the explicit profile timeout. No motion is executed.
    """
    def __init__(self, profile, current, publish):
        from ..agent.base_effects import BasePostconditionChecker
        self.reader = _StopTruth(profile)
        self.checker = BasePostconditionChecker(
            self.reader, limits=profile["verifier"], support_contract=profile.get("support_contract"))
        self.limits = copy.deepcopy(profile["verifier"])
        if profile["timeout_s"] > self.limits["read_timeout_s"]:
            raise ValueError("stop reader timeout_s exceeds verifier read_timeout_s")
        self._current, self._publish = current, publish
        self._condition = threading.Condition()
        self._pending = None
        self._closed = False
        self._thread = None
        self._quarantined = False
        self.error = None

    def submit(self, job):
        with self._condition:
            if self._closed or not self._current(job):
                return
            if self._pending is not None and self._pending["serial"] > job["serial"]:
                return
            self._pending = copy.deepcopy(job)  # replace, never queue an unbounded backlog
            if self._thread is None:
                self._thread = threading.Thread(target=self._run, name=f"mobile-stop-{job['base']}", daemon=True)
                self._thread.start()
            self._condition.notify()

    def _observe(self, job):
        if self._quarantined:
            return _unverified("stop observer quarantined after reader/checker failure")
        self.reader.job = job
        deadline = job["ack_monotonic_s"] + self.limits["max_wall_duration_s"]
        # Baseline acquisition and settling share ONE sampler, attempt budget
        # and ACK-anchored deadline. No prewarm observation is discarded/reused.
        boundary = {key: job["ack"][key] for key in
                    ("robot_id", "source", "epoch", "generation", "model_identity_sha256")}
        boundary["ack_monotonic_s"] = job["ack_monotonic_s"]
        token = self.checker.begin(job["skill"], {"base": job["base"]}, stop_boundary=boundary,
                                   is_current=lambda: not self._closed and self._current(job))
        verdict = self.checker.finish(token, copy.deepcopy(job["ack"]))
        if (verdict.get("evidence", {}).get("channel_failed") or
                verdict.get("reason") == "reader_timeout" or
                verdict.get("reason", "").startswith("verifier_error:")):
            self._quarantined = True
        if time.monotonic() > deadline:
            return {**verdict, "observed_status": verdict["status"], "observed_reason": verdict["reason"],
                    **_unverified("post-ACK observation wall deadline")}
        return verdict

    def _run(self):
        try:
            while True:
                with self._condition:
                    self._condition.wait_for(lambda: self._closed or self._pending is not None)
                    if self._closed:
                        return
                    job, self._pending = self._pending, None
                try:
                    verdict = self._observe(job)
                except Exception as exc:
                    self._quarantined = True
                    verdict = _unverified(f"stop observer failed: {exc}")
                self._publish(job, verdict)
        finally:
            try:
                self.checker.close()
            except Exception as exc:
                self.error = f"stop checker close failed: {exc}"

    def close(self):
        with self._condition:
            self._closed = True
            self._pending = None
            self._condition.notify()
        if self._thread is None:
            self.checker.close()
        else:
            self._thread.join(self.limits["max_wall_duration_s"] + 2 * self.limits["read_timeout_s"])
            if self._thread.is_alive():
                self.error = "stop observer did not close within wall bound; quarantined"
        return self.error


class MobileSkillRuntime:
    robot_mode = "mobile"
    motion_skills = MOTION_SKILLS

    def __init__(self, base_rig, memory, trace, cfg, *, checkers=None):
        self.base_rig, self.memory, self.trace, self.cfg = base_rig, memory, trace, cfg
        self.tool_specs = tool_specs_for_profiles(cfg.bases)
        self._capabilities = {p["name"]: set(p["capabilities"]) & base_rig.get(p["name"]).capabilities
                              for p in cfg.bases}
        for p in cfg.bases:
            if "walk_distance" in p["capabilities"] and "walk_distance" not in self._capabilities[p["name"]]:
                raise ValueError("walk_distance requires an implemented explicit distance-control profile")
        self.checkers = dict(checkers or {})
        self.verifier_errors = {}
        self.stop_observers = {}
        self.stop_verifier_errors = {}
        self._stop_latest = {}
        self._disabled_checkers = set()
        self.observation_readers = {}
        self.observation_errors = {}
        self.frame_readers = {}
        self._visual_history = deque(maxlen=64)
        self._frame_lock = threading.Lock()
        self._last_camera = None
        self._history = deque(maxlen=128)
        self._episode = []
        self._task_id = uuid.uuid4().hex
        self._stop_obligations = {}  # exact receipt/base, separate from immutable ACKs
        self._gate = threading.Lock()
        self._active = False
        self._resetting = False
        self._latched = False
        self._serial = 0
        self.current_task = self.current_tier = self.last_path = None
        self.stream_server = self.viewer = self.last_frame = None
        self.effects = None
        self._connections = {name: threading.Lock() for name in base_rig.names}
        self._connected = set()
        self._record_lock = threading.Lock()
        self._closed = False

    def _connect(self, name):
        with self._connections[name]:
            if self._closed:
                raise RuntimeError("mobile runtime closed")
            if name not in self._connected:
                self.base_rig.get(name).connect()
                if self._closed:
                    self.base_rig.get(name).disconnect()
                    raise RuntimeError("runtime closed during connection")
                self._connected.add(name)
        return self.base_rig.get(name)

    def passive_state(self, name):
        """The base's own validated state for a whole-body frame chain.

        The same passive read as ``get_base_state`` (a kinematic mock connects
        on first read); never a motion authorization or an independent pose.
        """
        return self._connect(name).get_state()

    def execute(self, name, args=None, *, admission_check=None):
        started = time.monotonic()
        with self._gate:
            task_id = self._task_id
        receipt_id = uuid.uuid4().hex
        arguments = {} if args is None else args
        try:
            spec = next((s for s in self.tool_specs if s["name"] == name), None)
            if spec is None:
                raise ValueError(f"unsupported mobile tool: {name!r}")
            schema = spec["parameters"]
            if not isinstance(arguments, dict):
                raise ValueError("arguments must be an object")
            if set(arguments) - schema["properties"].keys() or set(schema["required"]) - arguments.keys():
                raise ValueError("unknown or missing mobile arguments")
            if self._closed:
                raise RuntimeError("mobile runtime closed")
            if admission_check is not None and not callable(admission_check):
                raise ValueError("admission_check must be a passive callable")
            result = self._dispatch(name, arguments, task_id=task_id, admission_check=admission_check)
        except Exception as exc:
            result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        result.setdefault("task_id", task_id)
        result.setdefault("receipt_id", receipt_id)
        before_keyframe = None
        if result.get("image_before_jpeg_b64"):
            before_keyframe = "keyframes/mobile_" + uuid.uuid4().hex + "_before.jpg"
            (self.trace.run_dir / before_keyframe).write_bytes(base64.b64decode(result["image_before_jpeg_b64"], validate=True))
        keyframe = None
        if result.get("image_jpeg_b64"):
            keyframe = "keyframes/mobile_" + uuid.uuid4().hex + ".jpg"
            (self.trace.run_dir / keyframe).write_bytes(base64.b64decode(result["image_jpeg_b64"], validate=True))
        with self._record_lock:
            self.trace.record(name, arguments, _without_images(result), (time.monotonic() - started) * 1000,
                              keyframe_before=before_keyframe, keyframe_after=keyframe,
                              tier=self.current_tier, context={"robot_mode": "mobile", "task": self.current_task})
            if name in MOTION_SKILLS or (not result.get("ok") and name not in {"task_done", "task_memory"}):
                receipt = {**copy.deepcopy(result.get("postcondition") or _unverified(result.get("error", "execution failed"))),
                           "skill": name, "base": result.get("base"),
                           "task_id": task_id, "receipt_id": result["receipt_id"],
                           "execution_ok": result.get("execution_ok") is True, "ok": result.get("ok") is True}
                self._history.append(receipt)
                if task_id == self._task_id:
                    self._episode.append(receipt)
            # Do not recursively store task_memory's copy of all older events.
            if name in MOTION_SKILLS or name in {"emergency_stop", "stop_navigation", "reset_stop"}:
                self.memory.add("action", f"{name}: {result.get('postcondition', {}).get('status', result.get('ok'))}",
                                data=_without_images(result))
        return result

    def _dispatch(self, name, args, *, task_id, admission_check=None):
        if name == "list_bases":
            return {"ok": True, "bases": [
                {"name": n, **b.metadata, "capabilities": sorted(self._capabilities[n]),
                 "admission": self.cfg.bases[i].get("admission", "pending_physical_admission")}
                for i, (n, b) in enumerate(self.base_rig.bases.items())]}
        if name in {"emergency_stop", "stop_navigation"}:
            return self.stop(latch=name == "emergency_stop", _task_id=task_id)
        if name == "reset_stop":
            return self.reset_stop()
        if name == "verify_last_action":
            with self._record_lock:
                recent = copy.deepcopy(list(self._history)[-5:])
            with self._gate:
                for entry in self._stop_latest.values():
                    if entry["pending"] and time.monotonic() >= entry["deadline_monotonic_s"]:
                        entry.update(pending=False, status="unverified", reason="post-ACK observation wall deadline",
                                     physical_stop_verified=False)
                stops = copy.deepcopy(list(self._stop_latest.values()))
            recent = (recent + stops)[-5:]
            return {"ok": True, "recent": recent, "stop_verifications": stops,
                    "digest": "\n".join(f"{p['skill']}: {p['status']} — {p['reason']}" for p in recent),
                    "contradictions": [p for p in recent if p["status"] == "refuted"]}
        if name == "recall_memory":
            return {"ok": True, "memory": self.memory.digest()}
        if name == "task_memory":
            k = args.get("k", 4)
            if type(k) is not int or not 0 <= k <= 128:
                raise ValueError("k must be an integer in [0,128]")
            new_task = args.get("new_task", False)
            if type(new_task) is not bool:
                raise ValueError("new_task must be boolean")
            if new_task:
                self.begin_task()
            events = self.memory.events(("action",))
            with self._frame_lock:
                now = time.monotonic()
                while self._visual_history and now - self._visual_history[0]["t"] > self.memory.frame_horizon_s:
                    self._visual_history.popleft()
                frames = copy.deepcopy(list(self._visual_history)[-k:] if k else [])
            current = {"available": False}
            images = [{"image_jpeg_b64": base64.b64encode(f["jpeg"]).decode(), "frame": f["frame"], "caption": f["text"]} for f in frames]
            primary = self.base_rig.names[0]
            if primary in self.frame_readers:
                fresh = self._camera_observation(primary)
                current = {"available": fresh["ok"], "frame": fresh.get("frame"), "error": fresh.get("error")}
                if fresh["ok"]:
                    images.append({"image_jpeg_b64": fresh["image_jpeg_b64"], "frame": fresh["frame"], "caption": "current view"})
            return {"ok": True, "frames": len(frames), "current_view": current, "frame_images": images,
                    "steps_recorded": ([{"action": f["text"], "frame": f["frame"], "verdict": f["verdict"]} for f in frames]
                                       if frames else [{"action": e.text, "result": copy.deepcopy(e.data)}
                                                      for e in (events[-k:] if k else [])]),
                    "note": "Saved images are history, not fresh evidence or a physical success verdict."}
        if name == "task_done":
            if type(args["success"]) is not bool:
                raise ValueError("success must be boolean")
            unresolved = self.unverified_actions(task_id=task_id)
            success = args["success"] and not unresolved
            return {"ok": success, "task_complete": True, "success": success,
                    "summary": args["summary"], "unverified": unresolved}
        selected = args.get("base", self.base_rig.names[0])
        if not isinstance(selected, str):
            raise ValueError("base must be an exact nonempty name")
        self.base_rig.get(selected)
        if name in MOTION_SKILLS:
            if name not in self._capabilities[selected]:
                raise ValueError(f"base {selected!r} does not support {name}")
            return self._motion(name, args, selected, task_id=task_id, admission_check=admission_check)
        if name == "camera_snapshot":
            return self._camera_observation(selected, args.get("camera"))
        if selected in self.observation_readers:
            reader = self.observation_readers[selected]
            state = reader() if reader is not None else None
            if state is None:
                raise RuntimeError(self.observation_errors.get(selected) or
                                   getattr(reader, "last_error", None) or "independent state unavailable")
        else:
            state = self._connect(selected).get_state()
        state = state.as_dict()
        visual = {"frame": None}
        if name == "get_observation" and selected in self.frame_readers:
            # State and camera are independent, NOT a synchronized pose packet.
            visual = self._camera_observation(selected, args.get("camera"), expected_state=state)
        elif "camera" in args:
            return {"ok": False, "error": "selected base has no camera"}
        return {"ok": True, "base": selected, "state": state, **visual,
                "observation_note": "World pose and body velocity are base feedback, not a TCP/table frame. "
                                    "Kinematic mock feedback never confirms physical motion."}

    def _camera_observation(self, selected, camera=None, *, caption="observation", verdict="", expected_state=None):
        reader = self.frame_readers.get(selected)
        if reader is None:
            return {"ok": False, "base": selected, "frame": None, "error": "selected base has no camera"}
        camera = next(iter(reader.cameras)) if camera is None else camera
        frame = reader(camera)
        if frame is None:
            return {"ok": False, "base": selected, "frame": None, "error": reader.last_error or "camera unavailable"}
        meta = {"base": selected, **frame.as_dict()}
        if expected_state is not None and any(meta[k] != expected_state[k] for k in ("robot_id", "source", "epoch")):
            reader.close()  # no silent rebinding after a world reset
            return {"ok": False, "base": selected, "frame": None, "error": "camera/state identity mismatch; rebuild readers"}
        text = f"{caption}: {selected}/{camera} source={meta['source']} epoch={meta['epoch']} step={meta['step']}"
        with self._frame_lock:
            self._last_camera = (selected, camera)
            self._visual_history.append({"t": time.monotonic(), "frame": copy.deepcopy(meta),
                                         "jpeg": frame.jpeg, "text": text, "verdict": verdict})
        self.memory.add("observation", text, data={"frame": copy.deepcopy(meta), "verdict": verdict}, thumb_jpeg=frame.jpeg)
        return {"ok": True, "base": selected, "frame": meta, "image_jpeg_b64": base64.b64encode(frame.jpeg).decode()}

    def _motion(self, name, args, selected, *, task_id, admission_check=None):
        with self._gate:
            if self._active or self._resetting or self._latched or self._closed or task_id != self._task_id:
                return {"ok": False, "execution_ok": False, "error": "motion active, stopped or closed",
                        "postcondition": _unverified("motion not admitted")}
            self._active = True
            self._invalidate_stop_observations("new motion admitted")
            serial = self._serial
        checker = self.checkers.get(selected) if selected not in self._disabled_checkers else None
        token = None
        postcondition = _unverified(self.verifier_errors.get(selected, "independent verifier unavailable"))
        try:
            base = self._connect(selected)
            camera_before = (self._camera_observation(selected, caption="before " + name)
                             if selected in self.frame_readers else None)
            if checker is not None:
                try:
                    token = checker.begin(name, copy.deepcopy(args))
                except Exception as exc:
                    postcondition = _unverified(f"verifier begin failed: {exc}")
                    self._retire_checker(selected, checker, postcondition["reason"])
            with self._gate:
                cancelled = serial != self._serial or self._closed
            if cancelled:
                result = {"ok": False, "execution_ok": False, "error": "cancelled before dispatch"}
            else:
                values = {k: v for k, v in args.items() if k != "base"}
                if admission_check is not None:
                    values["admission_check"] = admission_check
                result = getattr(base, name)(**values)
            # A checker cannot mutate away execution errors/uncertain delivery.
            if token is not None:
                try:
                    verdict = checker.finish(token, copy.deepcopy(result))
                    if (not isinstance(verdict, dict) or verdict.get("status") not in
                            {"confirmed", "refuted", "unverified"} or not isinstance(verdict.get("reason"), str)):
                        raise ValueError("invalid verifier verdict")
                    postcondition = copy.deepcopy(verdict)
                except Exception as exc:
                    postcondition = _unverified(f"verifier finish failed: {exc}")
                    self._retire_checker(selected, checker, postcondition["reason"])
            if base.metadata["measurement_kind"] == "kinematic_mock" and postcondition["status"] == "confirmed":
                postcondition = _unverified("kinematic mock cannot confirm physical outcomes")
            with self._gate:
                if serial != self._serial:
                    result.update(execution_ok=False, error=result.get("error", "cancelled during execution/verification"))
                    if postcondition["status"] == "confirmed":
                        postcondition = {**postcondition, "observed_status": "confirmed",
                                         **_unverified("command receipt superseded during verification")}
            result.update(base=selected, postcondition=postcondition,
                          outcome=postcondition["status"],
                          ok=(result.get("execution_ok") is True and not result.get("error")
                              and not result.get("delivery_uncertain") and postcondition["status"] == "confirmed"))
            if camera_before is not None:
                camera_after = self._camera_observation(selected, caption="after " + name, verdict=postcondition["status"])
                result["camera_evidence"] = {
                    phase: {k: v for k, v in capture.items() if k != "image_jpeg_b64"}
                    for phase, capture in (("before", camera_before), ("after", camera_after))}
                result["image_before_jpeg_b64"] = camera_before.get("image_jpeg_b64")
                result["image_jpeg_b64"] = camera_after.get("image_jpeg_b64")
        except BaseException:
            self.stop(_record_obligation=False)
            if checker is not None:
                self._retire_checker(selected, checker, "execution interrupted")
            raise
        finally:
            with self._gate:
                self._active = False
        return result

    def _retire_checker(self, name, checker, reason):
        self._disabled_checkers.add(name)
        self.verifier_errors[name] = reason
        try:
            checker.close()
        except Exception as exc:
            self.verifier_errors[name] += f"; checker close failed: {exc}"

    def configure_stop_observer(self, profile):
        name = profile["name"]
        try:
            if profile["type"] != "isaac" or not isinstance(profile.get("verifier"), dict):
                raise ValueError("physical stop reader and explicit verifier limits unavailable")
            self.stop_observers[name] = _StopObserver(profile, self._stop_current, self._publish_stop)
        except Exception as exc:
            self.stop_verifier_errors[name] = str(exc)

    def _invalidate_stop_observations(self, reason):
        # Called with _gate held. Completed verdicts remain historical evidence,
        # but no longer assert the current stop/permission state.
        self._serial += 1
        for name, entry in self._stop_latest.items():
            key = (entry["receipt_id"], name)
            obligation = self._stop_obligations.get(key)
            if obligation is not None and obligation["pending"]:
                self._stop_obligations[key] = {**obligation, "pending": False, "superseded": True,
                                               "status": "unverified", "physical_stop_verified": False,
                                               "reason": "stop receipt superseded: " + reason}
            self._stop_latest[name] = {**entry, "pending": False, "superseded": True,
                                      "status": "unverified", "reason": reason, "physical_stop_verified": False}

    def _stop_current(self, job):
        with self._gate:
            entry = self._stop_latest.get(job["base"], {})
            return (not self._closed and job["task_id"] == self._task_id and job["serial"] == self._serial
                    and time.monotonic() < job["deadline_monotonic_s"]
                    and entry.get("receipt_id") == job["receipt_id"])

    def _publish_stop(self, job, verdict):
        verdict = copy.deepcopy(verdict)
        if time.monotonic() >= job["deadline_monotonic_s"]:
            verdict.setdefault("observed_status", verdict["status"])
            verdict.setdefault("observed_reason", verdict["reason"])
            verdict.update(status="unverified", reason="post-ACK observation wall deadline")
        completed = {**job, "postcondition": verdict, "status": verdict["status"],
                     "reason": verdict["reason"], "pending": False,
                     "physical_stop_verified": verdict["status"] == "confirmed"}
        with self._gate:
            current = (not self._closed and job["task_id"] == self._task_id and self._serial == job["serial"] and
                       self._stop_latest.get(job["base"], {}).get("receipt_id") == job["receipt_id"])
            if current:
                self._stop_latest[job["base"]] = completed
                key = (job["receipt_id"], job["base"])
                if key in self._stop_obligations:
                    self._stop_obligations[key] = completed
            else:
                completed.update(superseded=True, status="unverified", physical_stop_verified=False,
                                 reason="completion discarded: stop receipt superseded")
        # Separate event: never mutate the original ACK or a failed motion.
        with self._record_lock:
            self.trace.record("stop_verification", {"receipt_id": job["receipt_id"], "base": job["base"]},
                              completed, (time.monotonic() - job["ack_monotonic_s"]) * 1000,
                              context={"robot_mode": "mobile", "phase": "post_ack"})
            self.memory.add("outcome", f"stop {job['receipt_id']}: {completed['status']}", data=copy.deepcopy(completed))

    def stop(self, *, latch=True, _task_id=None, _record_obligation=True):
        # No init, connection, motion or trace lock is taken before invalidation.
        # Latch pending dispatch too: a nonlatched stop between preflight and
        # SafeBase admission would otherwise let an old command start afterwards.
        receipt_id = uuid.uuid4().hex
        with self._gate:
            task_id = self._task_id if _task_id is None else _task_id
            effective_latch = latch or self._active or self._latched
            self._latched = effective_latch
            self._invalidate_stop_observations("new stop receipt")
            serial = self._serial
            # Publish a pending obligation BEFORE IO: task_done cannot race the ACK.
            if _record_obligation and not self._closed and task_id == self._task_id:
                for name in self.base_rig.names:
                    self._stop_obligations[(receipt_id, name)] = {
                        "task_id": task_id, "receipt_id": receipt_id, "base": name,
                        "skill": "emergency_stop" if latch else "stop_navigation",
                        "status": "unverified", "reason": "stop delivery/physical observation pending",
                        "physical_stop_verified": False, "pending": True}
        result = self.base_rig.stop(latch=effective_latch)
        ack_time = time.monotonic()
        for name, ack in result["bases"].items():
            observer = self.stop_observers.get(name)
            valid = (ack.get("ok") is True and not ack.get("error") and not ack.get("delivery_uncertain")
                     and type(ack.get("generation")) is int and ack["generation"] >= 0
                     and type(ack.get("latched")) is bool and (not effective_latch or ack["latched"])
                     and all(isinstance(ack.get(k), str) and ack[k] and ack[k] == ack[k].strip()
                             for k in ("robot_id", "source", "epoch")))
            reason = self.stop_verifier_errors.get(name, "post-ACK independent observation pending")
            if not valid:
                reason = "stop ACK failed or has no exact physical receipt; " + str(ack.get("error", ""))
            job = {"receipt_id": receipt_id, "task_id": task_id, "serial": serial, "base": name,
                   "skill": "emergency_stop" if latch else "stop_navigation", "ack": copy.deepcopy(ack),
                   "ack_monotonic_s": ack_time,
                   "deadline_monotonic_s": ack_time + (observer.limits["max_wall_duration_s"] if observer else 0.),
                   "phase": "post_ack", "pending": bool(valid and observer),
                   "status": "unverified", "reason": reason, "physical_stop_verified": False}
            with self._gate:
                if self._closed or task_id != self._task_id or serial != self._serial:
                    continue
                self._stop_latest[name] = job
                if (receipt_id, name) in self._stop_obligations:
                    self._stop_obligations[(receipt_id, name)] = job
            if valid and observer is not None:
                observer.submit(job)
        return {**result, "latched": effective_latch, "outcome": "unverified",
                "receipt_id": receipt_id, "task_id": task_id, "serial": serial, "ack_monotonic_s": ack_time,
                "physical_stop_verified": False,
                "postcondition": _unverified("stop ACK invalidates commands; physical rest not independently measured"),
                "note": "Zero travel intent is not a balance/physical rest attestation. Reset clears permission, never resumes commands."}

    def reset_stop(self):
        with self._gate:
            if self._closed or self._active or self._resetting:
                return {"ok": False, "error": "motion/startup/reset active or runtime closed"}
            self._resetting = True
            self._invalidate_stop_observations("reset_stop invalidated stop observation")
            serial = self._serial
        results = {}
        try:
            for name in self.base_rig.names:
                try:
                    results[name] = self._connect(name).reset_stop()
                except Exception as exc:
                    results[name] = {"ok": False, "error": str(exc)}
            ok = all(r.get("ok") is True for r in results.values())
            with self._gate:
                raced = serial != self._serial
                if ok and not raced:
                    self._latched = False
            if raced or not ok:
                self.stop(_record_obligation=False)
            return {"ok": ok and not raced, "bases": results, "latched": self._latched,
                    "note": "Permission only; no motion replay, scene reset or fault recovery."}
        finally:
            with self._gate:
                self._resetting = False

    def begin_task(self):
        # Only an explicit new-task boundary clears obligations. task_done never does.
        with self._record_lock:
            with self._gate:
                if self._active or self._resetting:
                    raise ValueError("cannot begin a new task during an active action")
                self._task_id = uuid.uuid4().hex
                self._invalidate_stop_observations("explicit new task boundary")
                self._stop_obligations.clear()
                self._episode.clear()
        self.memory.reset_frames()
        with self._frame_lock:
            self._visual_history.clear()

    def unverified_actions(self, *, task_id=None):
        with self._record_lock:
            with self._gate:
                if task_id is not None and task_id != self._task_id:
                    return ["task boundary changed during request"]
                episode = tuple(self._episode)
                obligations = tuple(self._stop_obligations.values())
                active = self._active or self._resetting
        # Formatting/history traversal is never on the priority-stop gate.
        unresolved = [f"{r['skill']}: {r['status']} ({r['reason']})" for r in episode if not r["ok"]]
        if active:
            unresolved.append("action still in flight")
        unresolved.extend(
            f"{r['skill']} [{r['task_id']}/{r['receipt_id']}/{r['base']}]: "
            f"{r['status']} ({r['reason']})"
            for r in obligations if not r["physical_stop_verified"])
        return unresolved

    def frame_jpeg(self):
        with self._frame_lock:
            selected = self._last_camera
        if selected is None:
            return None
        reader = self.frame_readers.get(selected[0])
        frame = reader.cached(selected[1]) if reader else None
        return frame.jpeg if frame else None

    def close(self):
        with self._gate:
            self._closed = True
        stopped = self.stop(_record_obligation=False)
        errors = []
        for name, observer in self.stop_observers.items():
            try:
                error = observer.close()
                if error:
                    errors.append(f"stop observer {name}: {error}")
            except Exception as exc:
                errors.append(f"stop observer {name}: {exc}")
        for name, checker in self.checkers.items():
            try:
                checker.close()
            except Exception as exc:
                errors.append(f"checker {name}: {exc}")
        for name, reader in self.observation_readers.items():
            if reader is not None:
                try:
                    reader.close()
                except Exception as exc:
                    errors.append(f"reader {name}: {exc}")
        for name, reader in self.frame_readers.items():
            try:
                reader.close()
            except Exception as exc:
                errors.append(f"camera {name}: {exc}")
        disconnected = self.base_rig.close()
        return {"stop": stopped, **disconnected, "errors": errors,
                "ok": disconnected.get("ok") is True and not errors}
