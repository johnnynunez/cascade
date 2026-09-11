#!/usr/bin/env python3
"""Read-only CASCADE camera recorder; image holds preserve wall-clock time.

Bridge acquisition timestamps are CLIENT receipt times, not sensor timestamps.
Recording an image/state is not evidence of physical task success. No robot or
camera-pose command is ever sent. Tests inject explicitly synthetic fixtures;
the CLI always uses the real BridgeClient.
"""
from __future__ import annotations

import hashlib
import json
import math
import queue
import subprocess
import sys
import textwrap
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

TITLE = "DGX Spark / Isaac Sim Newton / CASCADE"


@dataclass
class Config:
    output: Path
    duration: float = 30.0
    fps: float = 10.0
    camera: str = "side"
    secondary: str | None = None
    width: int = 1920
    height: int = 1080
    host: str = "127.0.0.1"
    port: int = 8611
    timeout_s: float = 5.0
    controller_label: str = "diagnostic observation"
    stop_file: Path | None = None
    status_file: Path | None = None


def json_value(value):
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


class Evidence:
    def __init__(self, path):
        self.file = path.open("x", encoding="utf-8", buffering=1)
        self.lock = threading.Lock()

    def emit(self, event, **fields):
        row = {"event": event, "logged_wall_time_ns": time.time_ns(),
               "logged_monotonic_ns": time.monotonic_ns(), **fields}
        with self.lock:
            self.file.write(json.dumps(json_value(row), allow_nan=False) + "\n")

    def close(self):
        self.file.close()


def bridge_factory(**kwargs):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from cascade.sim.bridge_client import BridgeClient
    return BridgeClient(**kwargs)


class Acquisition(threading.Thread):
    """One owner for the bridge socket; never blocks the media clock."""
    def __init__(self, cfg, evidence, client_factory):
        super().__init__(daemon=True, name="cascade-readonly-acquisition")
        self.cfg, self.evidence = cfg, evidence
        self.client_factory = client_factory
        self.samples = queue.Queue(maxsize=8)
        self.stop_event = threading.Event()
        self.attempts = self.good = self.errors = 0
        self.dropped = 0
        self.failure = None

    def run(self):
        client = None
        connected = False
        try:
            client = self.client_factory(host=self.cfg.host, port=self.cfg.port,
                                         timeout_s=self.cfg.timeout_s)
            while not self.stop_event.is_set():
                if not connected:
                    client.connect()
                    ping = client.request({"op": "ping"})
                    self.evidence.emit("bridge_ping", response=ping)
                    if not ping.get("ok") or ping.get("engine") != "newton":
                        raise RuntimeError("refusing Newton banner: bridge ping is not Newton")
                    connected = True
                self.attempts += 1
                begin = time.monotonic_ns()
                camera_info, images = {}, {}
                try:
                    names = list(dict.fromkeys(n for n in
                                 (self.cfg.camera, self.cfg.secondary) if n))
                    for name in names:
                        start = time.monotonic_ns()
                        image, depth, K = client.frame(name)
                        received = time.monotonic_ns()
                        if (not isinstance(image, np.ndarray) or image.dtype != np.uint8
                                or image.ndim != 3 or image.shape[2] != 3
                                or min(image.shape[:2]) == 0):
                            raise ValueError(f"invalid BGR uint8 frame from {name}")
                        images[name] = image.copy()
                        camera_info[name] = {
                            "request_monotonic_ns": start,
                            "received_monotonic_ns": received,
                            "received_wall_time_ns": time.time_ns(),
                            "shape": list(image.shape), "K": json_value(K),
                            "depth_present": depth is not None,
                            "bgr_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
                        }
                    state_start = time.monotonic_ns()
                    state = client.state()
                    for key in ("q", "dq"):
                        values = np.asarray(state[key], dtype=float)
                        if values.ndim != 1 or not values.size or not np.isfinite(values).all():
                            raise ValueError(f"invalid measured state {key}")
                    if not math.isfinite(float(state["gripper_pos"])):
                        raise ValueError("invalid measured gripper_pos")
                    ready = time.monotonic_ns()
                    sample = {"acquisition_id": self.attempts, "ok": True,
                              "request_monotonic_ns": begin,
                              "ready_monotonic_ns": ready,
                              "received_wall_time_ns": time.time_ns(),
                              "cameras": camera_info, "state": json_value(state),
                              "state_request_monotonic_ns": state_start,
                              "state_received_monotonic_ns": ready}
                    self.good += 1
                    self.evidence.emit("acquisition", **sample)
                    try:
                        self.samples.put_nowait({**sample, "images": images})
                    except queue.Full:
                        self.dropped += 1
                        self.evidence.emit("acquisition_pixels_dropped",
                                           acquisition_id=self.attempts, reason="encoder_backlog")
                except Exception as exc:
                    self.errors += 1
                    self.evidence.emit("acquisition", acquisition_id=self.attempts,
                                       ok=False, error=str(exc), cameras=camera_info,
                                       request_monotonic_ns=begin)
                    # A socket makefile cannot be reused after a timeout.
                    client.close()
                    connected = False
                self.stop_event.wait(max(0.0, 1 / self.cfg.fps -
                                         (time.monotonic_ns() - begin) / 1e9))
        except Exception as exc:
            self.failure = str(exc)
            self.evidence.emit("acquisition_fatal", error=str(exc))
            try:
                self.samples.put_nowait({"fatal": str(exc)})
            except queue.Full:
                pass  # Fatal error is also in JSONL; queued evidence remains valid.
        finally:
            if client is not None:
                client.close()


def letterbox(image, width, height):
    scale = min(width / image.shape[1], height / image.shape[0])
    w, h = max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale))
    canvas = np.zeros((height, width, 3), np.uint8)
    x, y = (width - w) // 2, (height - h) // 2
    canvas[y:y+h, x:x+w] = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
    return canvas


def read_status(path):
    status = {"path": str(path) if path else None, "text": "", "verdict": None,
              "read_wall_time_ns": time.time_ns()}
    if path is None:
        return status
    try:
        with Path(path).open("rb") as stream:
            import os
            stat = os.fstat(stream.fileno())
            if stat.st_size > 1_000_000:
                raise ValueError("status file exceeds 1 MB")
            raw = json.load(stream)
        if not isinstance(raw, dict):
            raise ValueError("status must be a JSON object")
        status.update(raw=raw, mtime_ns=stat.st_mtime_ns,
                      age_s=(time.time_ns()-stat.st_mtime_ns)/1e9,
                      text=str(raw.get("text", raw.get("description", ""))),
                      verdict=raw.get("verdict"))
    except (OSError, ValueError) as exc:
        status.update(error=str(exc), text=f"Status unavailable: {exc}")
    return status


def render(cfg, sample, index, age_s, repeated, status):
    canvas = np.zeros((cfg.height, cfg.width, 3), np.uint8)
    canvas[:] = (21, 19, 17)
    names = list(sample["images"])
    tile_width = cfg.width // len(names)
    for i, name in enumerate(names):
        canvas[150:cfg.height-220, i*tile_width:(i+1)*tile_width] = letterbox(
            sample["images"][name], tile_width, cfg.height-370)
        cv2.putText(canvas, f"Camera: {name} (full frame, letterboxed)",
                    (i*tile_width+18, 142), cv2.FONT_HERSHEY_SIMPLEX,
                    min(0.65, cfg.width / 1700), (235, 215, 170), 1, cv2.LINE_AA)
    lines = [TITLE, f"Controller label: {cfg.controller_label}",
             f"WALL TIME {index/cfg.fps:7.2f}s | frame {index+1} | "
             f"source age {age_s:.3f}s | {'HOLD (no newer acquisition)' if repeated else 'ACQUIRED'}"]
    for row, text in enumerate(lines):
        cv2.putText(canvas, text, (18, 35 + row*38), cv2.FONT_HERSHEY_SIMPLEX,
                    min(0.8, cfg.width / 1600), (240, 240, 240), 1, cv2.LINE_AA)
    q = np.array2string(np.asarray(sample["state"]["q"]), precision=3)
    external_verdict = ("NOT PROVIDED" if status["verdict"] is None else
                        json.dumps(status["verdict"], ensure_ascii=False))
    text_lines = textwrap.wrap(status["text"] or "No external status description supplied", 120)
    if len(text_lines) > 2:
        text_lines = [text_lines[0], text_lines[1][:110] + "... [JSONL]"]
    footer = [f"Measured q: {q} | gripper: {sample['state']['gripper_pos']:.4f}",
              "Physical task verdict: NOT EVALUATED by this recorder",
              *text_lines,
              f"External verdict (unvalidated): {external_verdict}"]
    for row, text in enumerate(footer):
        cv2.putText(canvas, text, (18, cfg.height-188 + row*36), cv2.FONT_HERSHEY_SIMPLEX,
                    min(0.7, cfg.width / 1700), (230, 230, 230), 1, cv2.LINE_AA)
    return canvas


class FFmpegEncoder:
    """Close stdin and wait for the MP4 trailer; stderr never fills a PIPE."""
    def __init__(self, output, size, fps):
        import shutil
        import subprocess
        executable = shutil.which("ffmpeg")
        if executable is None:
            raise RuntimeError("ffmpeg is required (libx264); no silent codec fallback")
        self.log_path = output.with_suffix(".ffmpeg.log")
        self.log = self.log_path.open("xb")
        width, height = size
        command = [executable, "-hide_banner", "-nostdin", "-n", "-f", "rawvideo",
                   "-pixel_format", "bgr24", "-video_size", f"{width}x{height}",
                   "-framerate", str(fps), "-i", "pipe:0", "-an", "-c:v", "libx264",
                   "-preset", "veryfast", "-crf", "20", "-threads", "2",
                   "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output)]
        try:
            self.process = subprocess.Popen(command, stdin=subprocess.PIPE,
                                            stdout=subprocess.DEVNULL, stderr=self.log)
        except Exception:
            self.log.close()
            raise
        self.closed = False

    def write(self, frame):
        try:
            assert self.process.stdin is not None
            self.process.stdin.write(np.ascontiguousarray(frame).tobytes())
        except BrokenPipeError as exc:
            raise RuntimeError(f"ffmpeg failed; inspect {self.log_path}") from exc

    def close(self):
        if not self.closed:
            self.closed = True
            try:
                assert self.process.stdin is not None
                self.process.stdin.close()
            except BrokenPipeError:
                pass
            try:
                code = self.process.wait(timeout=60)
            except subprocess.TimeoutExpired as exc:
                self.process.kill()  # Only the encoder Popen owned by this instance.
                self.process.wait()
                raise RuntimeError(f"encoder finalization timed out; see {self.log_path}") from exc
            finally:
                self.log.close()
            if code:
                raise RuntimeError(f"ffmpeg exit {code}; inspect {self.log_path}")
        return self.process.returncode


def stop_reason(cfg, stop_event):
    if stop_event is not None and stop_event.is_set():
        return "signal"
    if cfg.stop_file is not None and Path(cfg.stop_file).exists():
        return "stop_file"
    return None


def wait_until(deadline_ns, cfg, stop_event):
    while True:
        reason = stop_reason(cfg, stop_event)
        if reason:
            return reason
        remaining = (deadline_ns - time.monotonic_ns()) / 1e9
        if remaining <= 0:
            return None
        time.sleep(min(remaining, 0.02))


def record(cfg, *, client_factory=bridge_factory, encoder_factory=FFmpegEncoder,
           stop_event=None):
    validate_config(cfg)
    output = Path(cfg.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    evidence = Evidence(output.with_suffix(".jsonl"))
    evidence.emit("metadata", schema_version=1, title=TITLE, output=str(output),
                  duration_requested_s=cfg.duration, fps=cfg.fps,
                  cameras=[n for n in (cfg.camera, cfg.secondary) if n],
                  controller_label=cfg.controller_label, read_only=True,
                  timestamp_semantics="client receipt; no sensor timestamp exposed by BridgeClient.frame",
                  timing_policy="CFR wall-clock slots; hold latest acquisition at or before slot; no pose interpolation",
                  physical_task_verdict="not_evaluated")
    worker = Acquisition(cfg, evidence, client_factory)
    encoder = None
    worker.start()
    count = repeated_count = 0
    previous = None
    deferred = None
    start = end = 0
    reason = "duration"
    failure = None
    encoder_returncode = None
    used_ids = set()
    wall_start = None
    try:
        try:
            sample = worker.samples.get(timeout=4 * cfg.timeout_s + 5)
        except queue.Empty as exc:
            raise RuntimeError("no valid camera/state acquisition within startup budget") from exc
        if "fatal" in sample:
            raise RuntimeError(sample["fatal"])
        if encoder_factory is None:
            raise RuntimeError("encoder not configured")
        encoder = encoder_factory(output, (cfg.width, cfg.height), cfg.fps)
        start = time.monotonic_ns()
        wall_start = time.time_ns()
        evidence.emit("recording_started", start_monotonic_ns=start,
                      start_wall_time_ns=wall_start)
        frame_total = math.ceil(cfg.duration * cfg.fps)
        for index in range(frame_total):
            target = start + round(index * 1e9 / cfg.fps)
            interrupted = wait_until(target, cfg, stop_event)
            if interrupted:
                reason = interrupted
                break
            while True:
                if deferred is None:
                    try:
                        deferred = worker.samples.get_nowait()
                    except queue.Empty:
                        break
                if "fatal" in deferred:
                    raise RuntimeError(deferred["fatal"])
                if deferred["ready_monotonic_ns"] > target:
                    break
                sample, deferred = deferred, None
            repeated = sample["acquisition_id"] == previous
            age_s = (target - sample["ready_monotonic_ns"]) / 1e9
            status = read_status(cfg.status_file)
            encoder.write(render(cfg, sample, index, age_s, repeated, status))
            evidence.emit("frame", frame_index=index, frame_count=index+1,
                          pts_s=index/cfg.fps, scheduled_monotonic_ns=target,
                          wall_time_ns=wall_start + target-start,
                          acquisition_id=sample["acquisition_id"], repeated=repeated,
                          source_age_s=age_s, q=sample["state"]["q"],
                          external_status=status,
                          dq=sample["state"]["dq"], gripper_pos=sample["state"]["gripper_pos"])
            previous = sample["acquisition_id"]
            used_ids.add(previous)
            repeated_count += int(repeated)
            count += 1
        if reason == "duration":
            reason = wait_until(start + round(cfg.duration * 1e9), cfg, stop_event) or reason
        end = time.monotonic_ns()
    except BaseException as exc:
        failure = exc
        reason = "error"
        end = time.monotonic_ns()
    finally:
        worker.stop_event.set()
        worker.join(timeout=4 * cfg.timeout_s + 5)
        if worker.failure:
            failure = failure or RuntimeError(worker.failure)
        if worker.is_alive():
            failure = failure or RuntimeError("acquisition worker did not stop within its timeout budget")
        if encoder is not None:
            try:
                encoder_returncode = encoder.close()
            except Exception as exc:
                failure = failure or exc
        if count == 0:
            failure = failure or RuntimeError("no video frames were encoded")
    summary = {"frame_count": count, "repeated_frame_count": repeated_count,
               "recording_ok": failure is None, "error": str(failure) if failure else None,
               "output": str(output), "start_wall_time_ns": wall_start,
               "stop_reason": "error" if failure else reason,
               "acquisition_count": worker.good, "acquisition_error_count": worker.errors,
               "unique_acquisitions_used": len(used_ids),
               "acquisition_pixels_dropped": worker.dropped,
               "worker_stopped": not worker.is_alive(), "encoder_returncode": encoder_returncode,
               "wall_duration_s": (end-start)/1e9 if start else 0,
               "media_duration_s": count/cfg.fps,
               "finalization_duration_s": (time.monotonic_ns()-end)/1e9,
               "physical_task_verdict": "not_evaluated"}
    overrun = max(0.0, summary["wall_duration_s"] - cfg.duration) if reason == "duration" else 0.0
    summary["encoding_overrun_s"] = overrun
    summary["timing_warning"] = (
        "Encoder slower than wall clock; media retains original receipt-time slots, "
        "encoding finished late (not a faster physical trajectory)." if overrun > 1/cfg.fps else None)
    evidence.emit("summary", **summary)
    if not worker.is_alive():
        evidence.close()
    if failure:
        raise failure
    return summary


def validate_config(cfg):
    for name in ("duration", "fps", "timeout_s"):
        value = getattr(cfg, name)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if cfg.width < 640 or cfg.height < 600 or cfg.width % 2 or cfg.height % 2:
        raise ValueError("video size must be even, at least 640x600 (no camera cropping)")
    if Path(cfg.output).suffix != ".mp4":
        raise ValueError("output must end in .mp4")
    if not cfg.camera or cfg.secondary == "":
        raise ValueError("camera names must not be empty")
    outputs = {Path(cfg.output).resolve(), Path(cfg.output).with_suffix(".jsonl").resolve(),
               Path(cfg.output).with_suffix(".ffmpeg.log").resolve()}
    for path in (cfg.status_file, cfg.stop_file):
        if path is not None and Path(path).resolve() in outputs:
            raise ValueError("status/stop file must not be an output artifact")


def parse_args(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=(
        "Read-only DGX Spark camera recorder. No robot motion. H.264 MP4 + JSONL + "
        "ffmpeg stderr log. Slow acquisitions hold the last image at wall-clock FPS."))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--duration", required=True, type=float, help="wall-clock seconds")
    parser.add_argument("--fps", type=float, default=10)
    parser.add_argument("--camera", default="side", help="any existing bridge camera name")
    parser.add_argument("--secondary", help="optional second camera (for example cam0)")
    parser.add_argument("--status-file", type=Path,
                        help="read JSON text/description and explicit verdict only; never derive success from ok")
    parser.add_argument("--stop-file", type=Path,
                        help="stop when this path exists; close encoder normally to retain MP4 trailer")
    parser.add_argument("--controller-label", default="diagnostic observation",
                        help="explicit user-supplied label, not an inferred controller identity")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8611)
    parser.add_argument("--timeout-s", type=float, default=5)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    cfg = Config(**vars(parser.parse_args(argv)))
    try:
        validate_config(cfg)
    except ValueError as exc:
        parser.error(str(exc))
    return cfg


def main(argv=None):
    import signal
    cfg = parse_args(argv)
    stop = threading.Event()
    old_handlers = {sig: signal.signal(sig, lambda signum, frame: stop.set())
                    for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        summary = record(cfg, stop_event=stop)
        print(json.dumps(summary, allow_nan=False), flush=True)
        if summary["timing_warning"]:
            print(f"TIMING WARNING: {summary['timing_warning']}", file=sys.stderr)
        print(f"MP4: {cfg.output}\nEvidence: {cfg.output.with_suffix('.jsonl')}\n"
              f"Encoder stderr: {cfg.output.with_suffix('.ffmpeg.log')}", file=sys.stderr)
        return 0
    except Exception as exc:
        print(f"Recording failed: {exc}", file=sys.stderr, flush=True)
        return 1
    finally:
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    raise SystemExit(main())
