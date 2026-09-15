#!/usr/bin/env python3
"""Discover Isaac assets without importing Kit or assuming a source build."""
from __future__ import annotations

import importlib.metadata
import math
import os
from pathlib import Path
import tomllib


def setup_physics(simulation_manager, *, dt: float, device: str, require_cuda: bool = False) -> None:
    """Configure the requested backend once. A CUDA failure is never a CPU retry."""
    if require_cuda and not str(device).startswith("cuda:"):
        raise RuntimeError(f"GPU demo requires an explicit CUDA physics device; received {device!r}")
    simulation_manager.setup_simulation(dt=dt, device=device)


def physics_device_identity(simulation_manager, *, require_cuda: bool = False) -> dict:
    """Attest actual backend data allocation, independently of requested env values."""
    engine = str(simulation_manager.get_active_physics_engine()).lower()
    device = str(simulation_manager.get_device())
    scenes = simulation_manager.get_physics_scenes()
    view = simulation_manager.get_physics_simulation_view()
    if not scenes or view is None or not view.is_valid:
        raise RuntimeError("Cannot attest GPU physics without registered scenes and a valid live tensor view")
    tensor_device = str(view.device)
    ordinal = int(view.device_ordinal)
    cuda_context_present = bool(view.cuda_context)
    dynamics = all(scene.get_enabled_gpu_dynamics() for scene in scenes) if engine == "physx" else None
    broadphases = [str(scene.get_broadphase_type()) for scene in scenes] if engine == "physx" else []
    gpu = (engine in ("physx", "newton") and device.startswith("cuda:")
           and tensor_device.startswith("cuda:") and ordinal >= 0 and cuda_context_present
           and device == tensor_device == f"cuda:{ordinal}"
           and (engine != "physx" or (dynamics and all(b == "GPU" for b in broadphases))))
    if require_cuda and not gpu:
        raise RuntimeError(f"GPU physics required; CUDA readback mismatch: engine={engine}, "
                           f"device={device}, tensor={tensor_device}, ordinal={ordinal}, "
                           f"context={cuda_context_present}, dynamics={dynamics}, broadphase={broadphases}")
    attestation = {"required": require_cuda, "backend": engine, "device": device,
                   "tensor_device": tensor_device, "tensor_device_ordinal": ordinal,
                   "cuda_context_present": cuda_context_present, "gpu_dynamics": dynamics,
                   "broadphase": broadphases[0] if len(set(broadphases)) == 1 else broadphases,
                   "cpu_fallback_allowed": False}
    return {"physics_device": device, "physics_tensor_device": tensor_device,
            "physics_gpu": bool(gpu), "gpu_attestation": attestation}


class GpuPhysicsLogGuard:
    """Latch explicit CPU contact fallback and buffer truncation from native logs.

    The callback only stores text; it never logs or calls Kit from its log thread.
    The simulator's main thread calls check() before publishing readiness/stepping.
    """
    def __init__(self):
        import threading
        self._lock = threading.Lock()
        self.failures = []

    def on_message(self, channel, level, module, filename, function, line, message, *rest):
        value = str(message)
        lower = value.lower()
        physics = any(word in str(channel).lower() + " " + lower for word in ("physx", "physics", "newton", "mjwarp", "collision"))
        fallback = any(word in lower for word in ("fall back to cpu", "fallback to cpu", "falling back to cpu", "failed to cook gpu-compatible"))
        overflow = "overflow" in lower and any(word in lower for word in ("contact", "constraint", "pair"))
        if physics and (fallback or overflow):
            with self._lock:
                if len(self.failures) < 20:
                    self.failures.append({"channel": str(channel), "message": value[:2000]})

    def check(self):
        with self._lock:
            if self.failures:
                raise RuntimeError("GPU physics/contact failure: " + self.failures[0]["message"])


def _check_kit(path: Path) -> Path:
    try:
        with path.open("rb") as stream:
            version = tomllib.load(stream).get("package", {}).get("version")
    except tomllib.TOMLDecodeError as exc:
        raise RuntimeError(f"invalid Isaac experience TOML: {path}") from exc
    if version != "6.1.0":
        raise RuntimeError(f"Isaac experience package.version=6.1.0 required; {path} declares {version!r}")
    # Keep the release/apps spelling: source builds symlink .kit files (or
    # apps itself) into source/apps. Kit anchors ${app}/../extsDeprecated
    # and extscache to the supplied path; resolve() loses the built tree.
    return path.absolute()


def find_experience(engine: str, *, release=None, package_roots=None) -> Path:
    if engine not in ("newton", "physx"):
        raise ValueError(f"unknown physics engine: {engine}")
    name = "isaacsim.exp.full.newton.kit" if engine == "newton" else "isaacsim.exp.full.kit"
    release = release or os.environ.get("ISAACSIM_PATH")
    if release:
        # Explicit source roots are authoritative. A missing/stale app must
        # not silently select a wheel belonging to another installation.
        return _check_kit(Path(release).expanduser() / "apps" / name)
    roots = []
    if package_roots is None:
        try:
            distribution = importlib.metadata.distribution("isaacsim")
            roots.append(Path(str(distribution.locate_file("isaacsim"))))
        except importlib.metadata.PackageNotFoundError:
            pass
    else:
        roots.extend(Path(root) for root in package_roots)
    for root in roots:
        candidate = root / "apps" / name
        if candidate.is_file():
            return _check_kit(candidate)
    raise FileNotFoundError(
        f"Isaac {engine} experience {name} not found; install isaacsim[all,extscache]==6.1.0.0 "
        "in the Isaac Python environment, or set ISAACSIM_PATH to a complete release"
    )


def ensure_time_code_range(stage, *, duration_s: float = 24 * 60 * 60) -> dict:
    """Give static robot assets a usable playback range without replacing animation.

    Isaac 6.1 warns that startTimeCode == endTimeCode loops one frame and
    prevents time-accumulating sensors/controllers from producing output.
    This changes timeline metadata only, never a body pose or physics limit.
    """
    duration_s = float(duration_s)
    if not math.isfinite(duration_s) or duration_s <= 0:
        raise ValueError("Playback duration must be finite and positive")
    start = float(stage.GetStartTimeCode())
    end = float(stage.GetEndTimeCode())
    rate = float(stage.GetTimeCodesPerSecond())
    if not all(math.isfinite(v) for v in (start, end, rate)) or rate <= 0:
        raise ValueError("Invalid stage time-code metadata")
    changed = end <= start
    if changed:
        stage.SetStartTimeCode(start)
        stage.SetEndTimeCode(start + duration_s * rate)
        end = float(stage.GetEndTimeCode())
        if not math.isfinite(end) or end <= start:
            raise RuntimeError("Stage did not accept a non-degenerate playback range")
    return {"changed": changed, "start": start, "end": end,
            "time_codes_per_second": rate}


def physics_timestep_identity(simulation_manager) -> dict:
    """Read actual configured physics timing on Kit's main thread before serving.

    get_physics_dt() alone silently returns 1/60 when its scene is absent.
    Require actual scene objects and agreement across the registered scenes.
    No requested argument, default timestep, or stage timeline is a readback.
    """
    scenes = simulation_manager.get_physics_scenes()
    if not scenes:
        raise RuntimeError("Cannot publish physics timestep without registered physics scenes")
    def valid(value):
        if isinstance(value, (bool, str, bytes)):
            raise RuntimeError("Physics timestep readback must be numeric")
        try:
            dt = float(value)
        except (TypeError, ValueError) as exc:
            raise RuntimeError("Physics timestep readback must be numeric") from exc
        if not math.isfinite(dt) or not 0 < dt <= 1:
            raise RuntimeError("Physics timestep readback must be finite and in (0, 1] seconds")
        return dt
    dt = valid(simulation_manager.get_physics_dt())
    for scene in scenes:
        scene_dt = valid(scene.get_dt())
        if not math.isclose(dt, scene_dt, rel_tol=1e-6, abs_tol=1e-12):
            raise RuntimeError("Registered physics scenes disagree on actual timestep")
    return {"physics_dt_s": dt, "physics_dt_source": "SimulationManager.get_physics_dt",
            "physics_dt_scene_count": len(scenes)}


def installation_info() -> dict:
    import sys

    try:
        version = importlib.metadata.version("isaacsim")
    except importlib.metadata.PackageNotFoundError:
        version = None
    if version is not None and version != "6.1.0.0":
        raise RuntimeError(f"Isaac Sim 6.1.0.0 required; selected Python has {version}")
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError("Isaac Sim 6.1.0.0 requires Python 3.12")
    release = os.environ.get("ISAACSIM_PATH")
    if release:
        root, selected = Path(release).expanduser().resolve(), Path(sys.executable).resolve()
        embedded = {p.resolve() for p in (root / "kit/python/bin").glob("python*") if p.is_file()}
        if not selected.is_relative_to(root) and selected not in embedded:
            raise RuntimeError("selected Python does not belong to ISAACSIM_PATH; use that release's python.sh")
        layout, version = "source", "6.1.0"
    else:
        layout = "wheel"
        if version is None:
            raise RuntimeError("Isaac Sim 6.1.0.0 is not installed in the selected Python")
    return {"layout": layout, "version": version, "python": sys.executable, "newton_experience": str(find_experience("newton"))}


if __name__ == "__main__":
    import argparse
    import json
    import sys

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", required=True)
    parser.parse_args()
    try:
        print(json.dumps(installation_info()))
    except (RuntimeError, FileNotFoundError) as error:
        print(str(error), file=sys.stderr)
        sys.exit(3)
