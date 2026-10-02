"""Independent mobile postconditions; no actuation and no actor-owned evidence.

All physical thresholds are explicit caller inputs. CPU tests of this module
are software tests, not a calibration or a physics acceptance campaign.
"""
from __future__ import annotations

from collections import Counter, deque
from copy import deepcopy
from dataclasses import dataclass, field
import math
import threading
import time
import uuid

from ..control.mobile_base import BaseState, finite_real, identifier, nonnegative_int
from ..control.mobile_support import digest, support_contract as validate_support_contract


LIMIT_KEYS = frozenset({
    "sample_interval_s", "read_timeout_s", "max_wall_duration_s", "settle_timeout_s",
    "settle_window_s", "min_motion_samples", "min_settle_samples", "max_samples", "max_history",
    "max_state_age_s", "max_sample_gap_s", "max_position_abs_m", "max_linear_speed_m_s",
    "max_angular_speed_rad_s", "min_height_m", "max_tilt_rad", "min_progress_ratio",
    "max_progress_ratio", "translation_tolerance_m", "rotation_tolerance_rad",
    "max_lateral_drift_m", "max_heading_drift_rad", "stop_linear_speed_m_s",
    "stop_angular_speed_rad_s", "stop_drift_m", "stop_drift_rad",
})


def _validate_limits(limits):
    if not isinstance(limits, dict) or set(limits) != LIMIT_KEYS:
        raise ValueError("verifier limits require exactly: " + ", ".join(sorted(LIMIT_KEYS)))
    integers = {"min_motion_samples", "min_settle_samples", "max_samples", "max_history"}
    result = {key: (nonnegative_int(value, key) if key in integers else finite_real(value, key))
              for key, value in limits.items()}
    if any(value <= 0 for value in result.values()):
        raise ValueError("all verifier limits must be positive")
    if not (result["sample_interval_s"] <= result["read_timeout_s"] <= result["settle_timeout_s"]
            <= result["max_wall_duration_s"]):
        raise ValueError("require sample_interval <= read_timeout <= settle_timeout <= max_wall_duration")
    if (result["min_motion_samples"] < 2 or result["min_settle_samples"] < 2 or
            result["max_samples"] < 1 + result["min_motion_samples"] + result["min_settle_samples"]):
        raise ValueError("insufficient sample counts/budget")
    if not result["min_progress_ratio"] <= 1 <= result["max_progress_ratio"]:
        raise ValueError("progress ratios must bracket one")
    if (result["max_tilt_rad"] >= math.pi/2 or result["rotation_tolerance_rad"] >= math.pi or
            result["max_heading_drift_rad"] >= math.pi):
        raise ValueError("invalid angular tolerances")
    if (result["stop_linear_speed_m_s"] > result["max_linear_speed_m_s"] or
            result["stop_angular_speed_rad_s"] > result["max_angular_speed_rad_s"]):
        raise ValueError("stop thresholds must stay inside plausibility limits")
    return result


@dataclass
class _Window:
    token: str
    skill: str
    args: dict
    started: float
    deadline: float
    finished: float | None = None
    samples: list = field(default_factory=list)
    attempts: int = 0
    error: str | None = None
    duplicates: int = 0
    late_reads: int = 0
    rejected: list = field(default_factory=list)
    last_generation: int | None = None
    last_state: dict | None = None
    first_generation: int | None = None
    stop_boundary: dict | None = None
    is_current: object = None
    temporal_pending: list = field(default_factory=list)
    channel_failed: bool = False
    read_started: float | None = None
    changed: threading.Event = field(default_factory=threading.Event)
    first: threading.Event = field(default_factory=threading.Event)
    done: threading.Event = field(default_factory=threading.Event)
    cancel: threading.Event = field(default_factory=threading.Event)
    lock: threading.Lock = field(default_factory=threading.Lock)


class BasePostconditionChecker:
    """One bounded observation window at a time, with a separately owned reader.

    Reader calls must themselves be bounded and close must unblock outstanding
    I/O. At most ONE sampler is created; an uncooperative reader is quarantined
    rather than spawning replacement workers. Python cannot kill a stuck callable.
    ``finish`` never reads ``result.measured`` or any actuator target.
    Physical readers must independently admit the configured support registry
    against the producer's hello contract hash. A model digest copied into an
    arbitrary registry does not prove that registry belongs to the model.
    """

    def __init__(self, reader, *, limits: dict, support_contract=None):
        self._limits = _validate_limits(limits)
        self._support_contract = validate_support_contract(support_contract)
        if not callable(reader):
            raise ValueError("reader must be callable")
        self._reader = reader
        self._history = deque(maxlen=self._limits["max_history"])
        self._gate = threading.Lock()
        self._active = None
        self._worker = None
        self._closed = False

    def begin(self, skill: str, args: dict, *, stop_boundary=None, is_current=None) -> str:
        """Acquire a baseline with the same sampler that observes the outcome.

        A stop_boundary is the exact ACK fence (robot_id/source/epoch/model/generation
        plus local ack_monotonic_s), for stop skills ONLY. Valid pre-ACK states
        consume attempts/cadence but not the baseline. The wall deadline is
        anchored to that ACK, never restarted by baseline acquisition or finish.
        is_current is an optional passive receipt-supersession check, not evidence.
        """
        boundary = deepcopy(stop_boundary)
        if boundary is not None:
            keys = {"ack_monotonic_s", "robot_id", "source", "epoch", "generation", "model_identity_sha256"}
            if (skill not in ("stop", "stop_navigation", "emergency_stop") or
                    not isinstance(boundary, dict) or set(boundary) != keys):
                raise ValueError("stop_boundary requires an exact stop ACK fence")
            for key in ("robot_id", "source", "epoch"):
                identifier(boundary[key], key)
            digest(boundary["model_identity_sha256"])
            nonnegative_int(boundary["generation"], "generation")
            stamp = finite_real(boundary["ack_monotonic_s"], "ack_monotonic_s")
            if not 0 <= stamp <= time.monotonic():
                raise ValueError("stop ACK must use a nonfuture local monotonic clock")
        if is_current is not None and (boundary is None or not callable(is_current)):
            raise ValueError("is_current requires stop_boundary and a callable")
        with self._gate:
            if self._closed:
                raise RuntimeError("checker is closed")
            if self._active is not None or (self._worker and self._worker.is_alive()):
                raise RuntimeError("an observation window is already active")
            now = time.monotonic()
            anchor = boundary["ack_monotonic_s"] if boundary is not None else now
            op = _Window(uuid.uuid4().hex, skill, deepcopy(args), now,
                         anchor + self._limits["max_wall_duration_s"],
                         stop_boundary=boundary, is_current=is_current)
            self._active = op
            self._worker = threading.Thread(target=self._sample, args=(op,),
                                            name="base-postcondition-sampler", daemon=True)
            self._worker.start()
        if boundary is not None:
            self._wait_for_stop_baseline(op)
        elif not op.first.wait(self._limits["read_timeout_s"]):
            with op.lock:
                if not op.done.is_set():
                    self._reject(op, "reader_timeout")
            op.cancel.set()
        return op.token

    def _wait_for_stop_baseline(self, op):
        # No additional reader/thread/retry: observe the ONE sampler's progress.
        # A sequence of healthy pending reads is not one blocked read.
        while True:
            op.changed.clear()
            with op.lock:
                if op.first.is_set() or op.done.is_set():
                    return
                deadline = op.deadline
                if op.read_started is not None:
                    deadline = min(deadline, op.read_started + self._limits["read_timeout_s"])
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    if deadline < op.deadline:
                        self._reject(op, "reader_timeout")
                    else:
                        op.error = op.error or "wall_deadline before post-ACK baseline"
                    op.cancel.set()
                    return
            op.changed.wait(remaining)

    def _sample(self, op):
        try:
            while not op.cancel.is_set():
                started = time.monotonic()
                with op.lock:
                    deadline = op.deadline
                    if op.finished is not None:
                        deadline = min(deadline, op.finished + self._limits["settle_timeout_s"])
                    if started >= deadline:
                        if started >= op.deadline:
                            op.error = op.error or "wall_deadline"
                        break
                    if op.attempts >= self._limits["max_samples"]:
                        op.error = op.error or "sample_limit"
                        break
                    if op.stop_boundary is not None and op.error:
                        break
                    if op.is_current is not None and not op.is_current():
                        op.error = op.error or "stop receipt superseded or expired"
                        break
                    op.attempts += 1
                    op.read_started = started
                    op.changed.set()
                try:
                    value = self._reader()
                    ended = time.monotonic()
                    with op.lock:
                        if ended - started > self._limits["read_timeout_s"]:
                            self._reject(op, "reader_timeout")
                        deadline = op.deadline if op.finished is None else min(
                            op.deadline, op.finished + self._limits["settle_timeout_s"])
                        if ended > deadline or op.cancel.is_set():
                            op.late_reads += 1
                        else:
                            self._accept(op, value, started, ended)
                except Exception as exc:
                    with op.lock:
                        if time.monotonic() - started > self._limits["read_timeout_s"]:
                            self._reject(op, "reader_timeout")
                        self._reject(op, "reader_error: " + str(exc)[:200])
                finally:
                    with op.lock:
                        op.read_started = None
                        op.changed.set()
                if op.stop_boundary is not None and op.error:
                    break
                if op.cancel.wait(self._limits["sample_interval_s"]):
                    break
        except Exception as exc:  # noqa: BLE001 - never lose a sampler failure outside the receipt
            with op.lock:
                self._reject(op, "sampler_error: " + str(exc)[:200])
        finally:
            op.done.set()
            op.changed.set()

    @staticmethod
    def _reject(op, reason):
        op.error = op.error or reason
        op.channel_failed = True
        op.rejected.append({"attempt": op.attempts, "reason": reason})

    def _accept(self, op, value, started, ended):
        if value is None:
            self._reject(op, "missing_state")
            return
        try:
            if not isinstance(value, BaseState):
                raise ValueError("reader must return BaseState")
            # Defensive copy and full validation, including quaternion norm. A
            # misbehaving reader must not mutate an already accepted snapshot.
            state = BaseState.from_dict(value.as_dict()).as_dict()
        except (TypeError, ValueError) as exc:
            self._reject(op, "invalid_state: " + str(exc)[:200])
            return
        if state["received_monotonic_s"] > ended:
            self._reject(op, "future receipt clock")
            return
        age = state["producer_age_s"] + ended - state["received_monotonic_s"]
        if age > self._limits["max_state_age_s"]:
            self._reject(op, "stale independent state")
            return
        if (max(abs(v) for v in state["position_world"]) > self._limits["max_position_abs_m"]
                or math.hypot(*state["linear_velocity_world"]) > self._limits["max_linear_speed_m_s"]
                or math.hypot(*state["angular_velocity_body"]) > self._limits["max_angular_speed_rad_s"]):
            self._reject(op, "state exceeds numeric plausibility bounds")
            return
        boundary = op.stop_boundary
        pending = (boundary is not None and
                   state["received_monotonic_s"] - state["producer_age_s"] < boundary["ack_monotonic_s"])
        if boundary is not None:
            for key in ("robot_id", "source", "epoch", "model_identity_sha256"):
                if state[key] != boundary[key]:
                    self._reject(op, f"post-ACK {key} mismatch")
                    return
            # Temporal exclusion cannot conceal an invalid/unsafe observation.
            if pending and (state["controller_status"] in ("fault", "disabled") or state["fallen"] or
                            state["position_world"][2] < self._limits["min_height_m"] or
                            self._tilt(state) > self._limits["max_tilt_rad"]):
                self._reject(op, "unsafe pre-ACK state: controller/posture fault")
                return
        current = op.is_current is None or op.is_current()
        duplicate = False
        if op.last_state is not None:
            previous = op.last_state
            if state["received_monotonic_s"] < previous["received_monotonic_s"]:
                self._reject(op, "local receipt clock regressed")
                return
            for key in ("robot_id", "source", "epoch", "measurement_kind", "joint_names", "model_identity_sha256"):
                if state[key] != previous[key]:
                    self._reject(op, f"conflicting {key}")
                    return
            generation_budget = 2 if op.skill in ("walk_velocity", "turn") else 1
            if current and (state["generation"] < op.last_generation or
                            state["generation"] > op.first_generation + generation_budget):
                self._reject(op, "generation regressed or unrelated operation invalidated window")
                return
            ds = state["step"] - previous["step"]
            dt = state["sim_time_s"] - previous["sim_time_s"]
            duplicate = ds == 0 and dt == 0
            if duplicate:
                if state["controller_status"] in ("fault", "disabled"):
                    self._reject(op, "controller fault/disabled without a new completed step")
                    return
                physical = ("position_world", "orientation_wxyz", "linear_velocity_world",
                            "angular_velocity_body", "joint_positions", "joint_velocities", "contacts", "fallen", "support")
                if any(state[key] != previous[key] for key in physical):
                    self._reject(op, "conflicting values in duplicate completed step")
                    return
            else:
                if ds <= 0 or dt <= 0:
                    self._reject(op, "physics clock did not advance consistently")
                    return
                if dt > self._limits["max_sample_gap_s"]:
                    self._reject(op, "simulation sampling gap exceeds limit")
                    return
                displacement = math.dist(state["position_world"], previous["position_world"])
                angular_bound = self._limits["max_angular_speed_rad_s"] * dt
                if (displacement > self._limits["max_linear_speed_m_s"] * dt or
                        self._attitude_distance(previous, state) > angular_bound or angular_bound >= math.pi):
                    self._reject(op, "interstep pose exceeds plausibility/aliasing bounds")
                    return
        if not current:
            if state["controller_status"] in ("fault", "disabled") or state["fallen"]:
                self._reject(op, "controller/posture fault during superseded read")
                return
            op.error = op.error or "stop receipt superseded during independent read"
            op.cancel.set()
            return
        if boundary is not None and state["generation"] != boundary["generation"]:
            self._reject(op, "post-ACK generation mismatch")
            return
        if op.first_generation is None:
            op.first_generation = state["generation"]
        op.last_generation = state["generation"]
        op.last_state = state
        if duplicate:
            op.duplicates += 1
        if pending and boundary is not None:
            op.temporal_pending.append({"attempt": op.attempts, "state": state,
                                        "capture_margin_s": state["received_monotonic_s"] -
                                        state["producer_age_s"] - boundary["ack_monotonic_s"]})
            return
        if duplicate:
            return
        if op.samples:
            accepted_gap = state["sim_time_s"] - op.samples[-1]["state"]["sim_time_s"]
            if accepted_gap > self._limits["max_sample_gap_s"]:
                self._reject(op, "accepted simulation sampling gap exceeds limit")
                return
            if accepted_gap * self._limits["max_angular_speed_rad_s"] >= math.pi:
                self._reject(op, "accepted interstep pose exceeds aliasing bound")
                return
        phase = ("before" if not op.samples else
                 "after" if op.finished is not None and started >= op.finished else "during")
        op.samples.append({"phase": phase, "read_started_monotonic_s": started,
                           "observed_monotonic_s": ended, "state": state})
        op.first.set()

    def finish(self, token: str, result: dict) -> dict:
        with self._gate:
            op = self._active
            if op is None or token != op.token:
                raise ValueError("unknown or already finished verification token")
            with op.lock:
                if op.finished is not None:
                    raise ValueError("verification is already finishing")
                op.finished = time.monotonic()
                deadline = min(op.deadline, op.finished + self._limits["settle_timeout_s"])
        op.done.wait(max(0., deadline - time.monotonic()))
        op.cancel.set()
        self._worker.join(timeout=self._limits["read_timeout_s"])
        with op.lock:
            if self._worker.is_alive():
                self._reject(op, "reader_timeout")
            if time.monotonic() >= op.deadline:
                op.error = op.error or "wall_deadline"
            verdict = self._receipt(op, result)
            try:
                self._evaluate(op, verdict)
            except Exception as exc:
                verdict.update(status="unverified", reason="verifier_error: " + str(exc)[:200])
        with self._gate:
            self._history.append(deepcopy(verdict))
            self._active = None
        return verdict

    def _receipt(self, op, result):
        failed = (not isinstance(result, dict) or
                  ("execution_ok" in result and result["execution_ok"] is not True) or
                  bool(result.get("error")) or bool(result.get("delivery_uncertain")) or
                  ("execution_ok" not in result and result.get("ok") is False))
        return {"status": "unverified", "reason": op.error or "missing independent evidence",
                "skill": op.skill, "args": deepcopy(op.args), "limits": deepcopy(self._limits),
                "execution_failed": failed, "metrics": {},
                # Protocol metadata delimits evidence; actor poses are NEVER copied.
                "admission": {key: deepcopy(result.get("ack", {}).get(key)) for key in
                              ("ok", "accepted", "latched", "error", "delivery_uncertain", "robot_id", "source", "epoch", "model_identity_sha256", "generation",
                               "start_sim_time_s", "end_sim_time_s")}
                if isinstance(result, dict) and isinstance(result.get("ack"), dict) else {},
                "completion": {key: deepcopy(result.get("stop_ack", {}).get(key)) for key in
                               ("ok", "latched", "error", "delivery_uncertain", "robot_id", "source", "epoch", "model_identity_sha256", "generation")}
                if isinstance(result, dict) and isinstance(result.get("stop_ack"), dict) else {},
                "evidence": {"samples": deepcopy(op.samples), "attempts": op.attempts,
                             "started_monotonic_s": op.started,
                             "wall_deadline_monotonic_s": op.deadline,
                             "stop_boundary": deepcopy(op.stop_boundary),
                             "channel_failed": op.channel_failed,
                             "temporal_pending": deepcopy(op.temporal_pending),
                             "finished_monotonic_s": op.finished, "provenance": None,
                             "duplicates": op.duplicates, "late_reads": op.late_reads,
                             "rejected": deepcopy(op.rejected)}}

    def _evaluate(self, op, verdict):
        if op.error or not op.samples:
            return verdict
        states = [entry["state"] for entry in op.samples]
        verdict["evidence"]["provenance"] = {
            key: states[0][key] for key in ("robot_id", "source", "epoch", "measurement_kind", "model_identity_sha256")}
        if states[0]["measurement_kind"] == "kinematic_mock":
            verdict["reason"] = "kinematic_mock is not physical evidence"
            return verdict
        if not any(entry["phase"] == "before" for entry in op.samples):
            return verdict
        if (op.skill in ("walk_velocity", "turn") and
                sum(entry["phase"] == "during" for entry in op.samples) < self._limits["min_motion_samples"]):
            verdict["reason"] = "insufficient advancing samples during execution"
            return verdict
        effect_states = states
        if op.skill in ("walk_velocity", "turn"):
            try:
                effect_states = self._admitted_states(op, verdict, states)
            except (ValueError, TypeError, KeyError) as exc:
                verdict["reason"] = "unbound command evidence: " + str(exc)
                return verdict
        metrics = self._measure(effect_states)
        settle_status, settle_reason, settle_metrics = self._settle(op)
        metrics.update(settle_metrics)
        verdict["metrics"] = metrics
        status, reason = self._motion_verdict(op.skill, op.args, metrics)
        if settle_status == "unverified" or (status == "confirmed" and settle_status == "refuted"):
            status, reason = settle_status, settle_reason
        # Entire observed episode retains forbidden support and unavailable
        # channels. Swing/flight during locomotion does not require ground load;
        # its terminal settle window does. Zero-twist balance requires support
        # throughout the causally admitted interval as well as terminal rest.
        zero_twist = op.skill == "walk_velocity" and all(op.args.get(k) == 0 for k in ("vx", "vy", "wz"))
        balance_steps = {s["step"] for s in effect_states} if zero_twist else set()
        support_results = [self._support(s, require_load=s["step"] in balance_steps) for s in states]
        verdict["evidence"]["support_contract"] = deepcopy(self._support_contract)
        verdict["evidence"]["support_checks"] = [dict(step=s["step"], status=r[0], reason=r[1])
                                                     for s, r in zip(states, support_results)]
        support_failure = next((r for r in support_results if r[0] == "refuted"), None)
        if support_failure is None and status == "confirmed":
            support_failure = next((r for r in support_results if r[0] == "unverified"), None)
        if support_failure is not None:
            status, reason = support_failure
        for state in states:
            if state["fallen"]:
                status, reason = "refuted", "fallen in independently observed window"
                break
            if state["position_world"][2] < self._limits["min_height_m"]:
                status, reason = "refuted", "height below upright limit"
                break
            if self._tilt(state) > self._limits["max_tilt_rad"]:
                status, reason = "refuted", "tilt exceeds upright limit"
                break
            if state["controller_status"] in ("fault", "disabled"):
                status, reason = "refuted", "controller fault/disabled observed"
                break
        if verdict["execution_failed"] and status == "confirmed":
            status, reason = "unverified", "execution failed; favorable final state cannot repair execution"
        verdict.update(status=status, reason=reason)
        return verdict

    def _admitted_states(self, op, verdict, states):
        """Clip to measured admission/completion boundaries, never interpolate.

        A pre-admission sample may supply the baseline ONLY at the exact
        admission clock. Otherwise use the first admitted independent sample,
        within the already configured sampling-gap bound. Missing travel is
        not extrapolated. All pre-samples remain available for anomaly checks.
        """
        ack, stop = verdict["admission"], verdict["completion"]
        for receipt in (ack, stop):
            if (receipt.get("ok") is not True or receipt.get("latched") is not False or
                    receipt.get("error") or receipt.get("delivery_uncertain")):
                raise ValueError("missing/failed admission or completion receipt")
            for key in ("robot_id", "source", "epoch", "model_identity_sha256"):
                if receipt.get(key) != states[0][key]:
                    raise ValueError(f"receipt {key} mismatch")
        generation = nonnegative_int(ack.get("generation"), "admission generation")
        completed = nonnegative_int(stop.get("generation"), "completion generation")
        if generation < 1 or completed != generation + 1 or ack.get("accepted") is not True:
            raise ValueError("invalid admission/completion generation")
        start = finite_real(ack.get("start_sim_time_s"), "admission clock")
        end = finite_real(ack.get("end_sim_time_s"), "completion clock")
        if start < 0 or end <= start:
            raise ValueError("invalid admission interval")
        if op.skill == "walk_velocity":
            try:
                duration = finite_real(op.args.get("duration_s"), "duration_s")
                for key in ("vx", "vy", "wz"):
                    finite_real(finite_real(op.args.get(key), key) * duration, "requested effect")
                if duration <= 0 or not math.isclose(end - start, duration, rel_tol=1e-9, abs_tol=1e-12):
                    raise ValueError("admission duration differs from caller intent")
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError("invalid caller intent: " + str(exc)) from exc
        if any(s["generation"] not in (generation, completed) for s in states if s["sim_time_s"] > start):
            raise ValueError("unrelated generation after admission")
        if states[-1]["generation"] != completed:
            raise ValueError("missing independent completion generation after outcome")
        effect = [s for s in states if start <= s["sim_time_s"] <= end and
                  (s["generation"] == generation or
                   (s["sim_time_s"] == start and s["generation"] == generation - 1))]
        # A turn may deliberately stop early. The first completed-generation
        # sample bounds that end, not any later quiet-window motion.
        stopped = next((s for s in states if s["generation"] == completed and
                        start < s["sim_time_s"] <= end), None)
        if stopped is not None:
            effect = [s for s in effect if s["sim_time_s"] < stopped["sim_time_s"]] + [stopped]
        if (len(effect) < self._limits["min_motion_samples"] or
                effect[0]["sim_time_s"] - start > self._limits["max_sample_gap_s"]):
            raise ValueError("no sufficiently close independent admission baseline")
        if stopped is None and end - effect[-1]["sim_time_s"] > self._limits["max_sample_gap_s"]:
            raise ValueError("no sufficiently close independent completion sample")
        verdict["evidence"]["effect_interval"] = {
            "admitted_start_sim_time_s": start, "admitted_end_sim_time_s": end,
            "baseline_step": effect[0]["step"], "baseline_sim_time_s": effect[0]["sim_time_s"],
            "last_step": effect[-1]["step"], "last_sim_time_s": effect[-1]["sim_time_s"],
            "generation": generation, "completion_generation": completed}
        return effect

    def _settle(self, op):
        # producer_age includes conservative transport latency; this is a lower
        # bound on capture time in the LOCAL clock, not a remote-clock subtraction.
        boundary = op.stop_boundary["ack_monotonic_s"] if op.stop_boundary is not None else op.finished
        after = [e["state"] for e in op.samples if e["phase"] == "after" and
                 e["read_started_monotonic_s"] >= boundary and
                 e["state"]["received_monotonic_s"] - e["state"]["producer_age_s"] >= boundary]
        window = []
        for state in reversed(after):
            window.insert(0, state)
            if (len(window) >= self._limits["min_settle_samples"] and
                    window[-1]["sim_time_s"] - window[0]["sim_time_s"] >= self._limits["settle_window_s"]):
                break
        duration = window[-1]["sim_time_s"] - window[0]["sim_time_s"] if window else 0.
        metrics = {"settle_samples": len(window), "settle_sim_duration_s": duration}
        if len(window) < self._limits["min_settle_samples"] or duration < self._limits["settle_window_s"]:
            return "unverified", "insufficient advancing states captured after outcome", metrics
        speed = max(math.hypot(*s["linear_velocity_world"]) for s in window)
        angular = max(math.hypot(*s["angular_velocity_body"]) for s in window)
        drift = sum(math.dist(a["position_world"], b["position_world"]) for a, b in zip(window, window[1:]))
        rotation = sum(self._attitude_distance(a, b) for a, b in zip(window, window[1:]))
        metrics.update(settle_max_linear_speed_m_s=speed, settle_max_angular_speed_rad_s=angular,
                       settle_drift_m=drift, settle_drift_rad=rotation)
        if (any(s["controller_status"] != "ready" for s in window) or
                speed > self._limits["stop_linear_speed_m_s"] or
                angular > self._limits["stop_angular_speed_rad_s"] or
                drift > self._limits["stop_drift_m"] or rotation > self._limits["stop_drift_rad"]):
            return "refuted", "did not settle: active controller, residual velocity or pose drift", metrics
        support = [self._support(s, require_load=True) for s in window]
        failure = next((r for r in support if r[0] == "refuted"), None)
        if failure is None:
            failure = next((r for r in support if r[0] == "unverified"), None)
        if failure is not None:
            return failure[0], failure[1], metrics
        return "confirmed", "advancing post-outcome states prove supported measured rest", metrics

    def _support(self, state, *, require_load):
        contract, observed = self._support_contract, state["support"]
        if contract is None or observed is None:
            return "unverified", "missing independent solved support contract/evidence"
        if contract["model_identity_sha256"] != state["model_identity_sha256"]:
            return "unverified", "support model identity differs from admitted contract"
        if observed["status"] != "known":
            return "unverified", "solved support unavailable: " + observed["reason"]
        gravity = contract["gravity_world_m_s2"]
        up = tuple(-x / math.hypot(*gravity) for x in gravity)
        robot, feet, ground = (set(contract[k]) for k in ("robot_shapes", "foot_shapes", "ground_shapes"))
        loaded_sole = False
        for contact in observed["contacts"]:
            a, b = contact["shape_a"], contact["shape_b"]
            if (a in robot) == (b in robot):
                continue  # Self contact / contacts between external bodies are not support.
            body, other, sign = (a, b, -1.) if a in robot else (b, a, 1.)
            if contact["normal_force_n"] <= 0:
                continue
            if body not in feet or other not in ground:
                return "refuted", "forbidden external robot contact in independent support evidence"
            force_up = sign * sum(x*y for x, y in zip(contact["force_on_b_world_n"], up))
            normal_up = sign * sum(x*y for x, y in zip(contact["normal_a_to_b_world"], up))
            loaded_sole |= force_up > 0 and normal_up > 0
        if require_load and not loaded_sole:
            return "refuted", "no admitted sole has positive upward solved reaction"
        return "confirmed", "complete solved contacts satisfy the phase support contract"

    @staticmethod
    def _attitude_distance(before, after):
        dot = sum(x*y for x, y in zip(before["orientation_wxyz"], after["orientation_wxyz"]))
        return 2 * math.acos(min(1., abs(dot)))

    @staticmethod
    def _yaw(state):
        w, x, y, z = state["orientation_wxyz"]
        return math.atan2(2 * (w*z + x*y), 1 - 2 * (y*y + z*z))

    @staticmethod
    def _tilt(state):
        _, x, y, _ = state["orientation_wxyz"]
        return math.acos(max(-1., min(1., 1 - 2 * (x*x + y*y))))

    @classmethod
    def _measure(cls, states):
        body = [0., 0.]
        angle = 0.
        path = 0.
        rotation_path = 0.
        for before, after in zip(states, states[1:]):
            yaw = cls._yaw(before)
            delta = cls._yaw(after) - yaw
            dyaw = math.atan2(math.sin(delta), math.cos(delta))
            heading = yaw + dyaw / 2
            dx, dy = (after["position_world"][i] - before["position_world"][i] for i in (0, 1))
            body[0] += math.cos(heading)*dx + math.sin(heading)*dy
            body[1] += -math.sin(heading)*dx + math.cos(heading)*dy
            angle += dyaw
            path += math.hypot(dx, dy)
            rotation_path += abs(dyaw)
        return {"body_displacement_m": body, "yaw_change_rad": angle,
                "path_length_m": path, "rotation_path_rad": rotation_path,
                "min_height_m": min(s["position_world"][2] for s in states),
                "max_tilt_rad": max(cls._tilt(s) for s in states),
                "max_linear_speed_m_s": max(math.hypot(*s["linear_velocity_world"]) for s in states),
                "max_angular_speed_rad_s": max(math.hypot(*s["angular_velocity_body"]) for s in states),
                "sim_duration_s": states[-1]["sim_time_s"] - states[0]["sim_time_s"]}

    def _motion_verdict(self, skill, args, metrics):
        if skill in ("stop", "stop_navigation", "emergency_stop"):
            return "confirmed", "independent rest check required"
        try:
            if skill == "turn":
                value = finite_real(args["angle_rad"], "angle_rad")
                if abs(value) > math.pi:
                    raise ValueError("ambiguous requested turn")
            elif skill == "walk_velocity":
                values = {key: finite_real(args[key], key) for key in ("vx", "vy", "wz", "duration_s")}
                if values["duration_s"] <= 0:
                    raise ValueError("duration must be positive")
                for key in ("vx", "vy", "wz"):
                    finite_real(values[key] * values["duration_s"], "requested effect")
        except (TypeError, ValueError, KeyError, OverflowError):
            return "unverified", "invalid caller intent"
        if skill == "turn":
            try:
                target = args["angle_rad"]
                if abs(target) <= self._limits["rotation_tolerance_rad"]:
                    return "unverified", "requested turn below configured resolution"
                if abs(metrics["yaw_change_rad"] - target) > self._limits["rotation_tolerance_rad"]:
                    return "refuted", "turn no effect, wrong sign or outside angle tolerance"
                if metrics["path_length_m"] > self._limits["max_lateral_drift_m"]:
                    return "refuted", "unrequested translation during turn"
                return "confirmed", "independent measured yaw matches requested angle"
            except (KeyError, TypeError):
                return "unverified", "invalid caller intent"
        if skill != "walk_velocity":
            return "unverified", "unsupported skill"
        try:
            expected = [args[key] * args["duration_s"] for key in ("vx", "vy", "wz")]
        except (KeyError, TypeError):
            return "unverified", "invalid caller intent"
        if not any(expected):
            if (metrics["path_length_m"] > self._limits["max_lateral_drift_m"] or
                    metrics["rotation_path_rad"] > self._limits["max_heading_drift_rad"]):
                return "refuted", "balance drift during zero twist"
            if (metrics["max_linear_speed_m_s"] > self._limits["stop_linear_speed_m_s"] or
                    metrics["max_angular_speed_rad_s"] > self._limits["stop_angular_speed_rad_s"]):
                return "refuted", "balance velocities exceed rest limits"
        measured = [*metrics["body_displacement_m"], metrics["yaw_change_rad"]]
        for axis, want, got in zip(("vx", "vy", "wz"), expected, measured):
            tol = self._limits["rotation_tolerance_rad" if axis == "wz" else "translation_tolerance_m"]
            drift = self._limits["max_heading_drift_rad" if axis == "wz" else "max_lateral_drift_m"]
            if want == 0:
                if abs(got) > drift:
                    return "refuted", f"unrequested {axis} drift"
            elif abs(want) * self._limits["min_progress_ratio"] <= tol:
                return "unverified", "requested effect below configured resolution"
            else:
                progress = math.copysign(1., want) * got
                if progress <= 0:
                    return "refuted", f"no effect or wrong sign on {axis}"
                if progress < abs(want) * self._limits["min_progress_ratio"] - tol:
                    return "refuted", f"insufficient progress on {axis}"
                if progress > abs(want) * self._limits["max_progress_ratio"] + tol:
                    return "refuted", f"excess progress on {axis}"
        return "confirmed", "independent measured motion matches caller intent"

    def history(self) -> list[dict]:
        with self._gate:
            return deepcopy(list(self._history))

    def digest(self) -> str:
        history = self.history()
        counts = Counter(item["status"] for item in history)
        return "; ".join(f"{key}={counts[key]}" for key in ("confirmed", "refuted", "unverified"))

    def close(self):
        with self._gate:
            if self._closed:
                return
            self._closed = True
            if self._active is not None:
                with self._active.lock:
                    self._active.error = self._active.error or "checker_closed"
                self._active.cancel.set()
        closer = getattr(self._reader, "close", None)
        try:
            if closer is not None:
                closer()
        finally:
            if self._worker is not None:
                self._worker.join(timeout=self._limits["read_timeout_s"])
