#!/usr/bin/env python3
"""Run a measured Newton CPU screw experiment, then open its 3-D replay.

The viewer replays solver-produced body states; it does not animate a fake
screw or run CASCADE/OpenClaw. Screw-thread geometry is an idealized constraint.
Use the isolated Newton environment, not CASCADE's application environment.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import math
from pathlib import Path
import signal
import sys
import time
import uuid
import webbrowser

import numpy as np

REPO = Path(__file__).resolve().parents[1]
FIELDS = ("time_s", "phase", "screw_turns", "axial_mm", "applied_torque_nm",
          "motor_torque_nm", "seating_contact_force_n", "tip_contact_force_n", "completed", "verified")


def write_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def simulate(demo, output, *, max_sim_seconds=30.0):
    """Persist measured frames and fail closed if the experiment is incomplete."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    report = {"verified": False, "display_mode": "replay_of_solver_states"}
    write_json(output / "summary.json", report)
    states, rows = [], []
    try:
        if not math.isfinite(max_sim_seconds) or max_sim_seconds <= 0:
            raise RuntimeError("simulation time budget must be finite and positive")
        if str(demo.model.device) != "cpu":
            raise RuntimeError("this demonstration requires the CPU device")
        previous_time = -math.inf
        previous_phase = None
        while True:
            current = float(demo.sim_time)
            if not math.isfinite(current) or current <= previous_time:
                raise RuntimeError("simulation clock is not advancing monotonically")
            state = np.array(demo.state.body_q.numpy(), copy=True)
            if state.shape != (demo.model.body_count, 7) or not np.isfinite(state).all():
                raise RuntimeError("invalid or non-finite measured body state")
            metrics = demo.metrics()
            row = {key: metrics[key] for key in FIELDS if key != "time_s"}
            row["time_s"] = current
            # This also rejects accidental NumPy objects and non-finite metrics.
            json.dumps(row, allow_nan=False)
            rows.append(row)
            states.append(state)
            if row["phase"] != previous_phase:
                print(f"[newton-screw] t={current:.2f}s phase={row['phase']}", flush=True)
                previous_phase = row["phase"]
            if row["completed"] is True:
                break
            if current >= max_sim_seconds:
                raise RuntimeError("simulation budget exhausted before completion")
            previous_time = current
            demo.step()
        report.update(metrics, frames=len(rows), recorded_states_finite=True)
        if report.get("verified") is not True:
            raise RuntimeError("physical acceptance was not satisfied; refusing a success replay")
        times = np.asarray([row["time_s"] for row in rows])
        body_q = np.stack(states)
        np.savez_compressed(output / "states.npz", times=times, body_q=body_q)
        with (output / "measurements.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        write_json(output / "measurements.json", rows)
        report["states_sha256"] = hashlib.sha256((output / "states.npz").read_bytes()).hexdigest()
        report["measurements_sha256"] = hashlib.sha256((output / "measurements.json").read_bytes()).hexdigest()
        write_json(output / "summary.json", report)
        return {"body_q": body_q, "rows": rows, "summary": report}
    except Exception as exc:
        write_json(output / "summary.json", {"verified": False, "error": str(exc),
                                             "recorded_frames": len(rows)})
        raise


def make_viewer(*, port=8766, record_to_viser=None):
    """Use Newton's native viewer while explicitly restricting its listener."""
    module = importlib.import_module("newton.viewer")
    viser = importlib.import_module("viser")

    class LocalModule:
        @staticmethod
        def ViserServer(**kwargs):
            return viser.ViserServer(host="127.0.0.1", verbose=False, **kwargs)

        def __getattr__(self, name):
            return getattr(viser, name)

    class LoopbackViewer(module.ViewerViser):
        @classmethod
        def _get_viser(cls):
            return LocalModule()

    viewer = LoopbackViewer(port=port, label="Newton CPU · SO-101 · Tornillo",
                            verbose=False, share=False, record_to_viser=record_to_viser)
    viewer._port = viewer._server.get_port()
    return viewer


def camera_view(viewer, target, *, close=False):
    """Aim the existing Newton camera at the fixture, including future clients."""
    wp = importlib.import_module("warp")
    target = np.asarray(target, dtype=float)
    offset = np.asarray((0.085, -0.12, 0.075) if close else (0.52, -0.64, 0.38))
    position = target + offset
    direction = target - position
    yaw = math.degrees(math.atan2(direction[1], direction[0]))
    pitch = math.degrees(math.atan2(direction[2], np.linalg.norm(direction[:2])))
    viewer.set_camera(wp.vec3(*position), pitch=pitch, yaw=yaw)


def replay(demo, recording, output, *, port=8766, open_browser=True, serve_seconds=None):
    """Display only recorded physical states, at their measured simulation times."""
    viewer = make_viewer(port=port, record_to_viser=str(output / "episode.viser"))
    server = viewer._server
    rows = recording["rows"]
    states = recording["body_q"]
    render_state = demo.model.state()  # separate display buffer; never passed to the solver
    target = getattr(demo, "camera_target", None)
    if target is None:
        target = np.mean(states[-1, :, :3], axis=0)
    old_signals = {}
    stop = False

    def request_stop(*_):
        nonlocal stop
        stop = True

    try:
        for sig in (signal.SIGTERM, signal.SIGINT):
            old_signals[sig] = signal.signal(sig, request_stop)
        viewer.set_model(demo.model)
        camera_view(viewer, target)
        # One exact episode is serialized, then recording stops before looping.
        for row, state in zip(rows, states, strict=True):
            render_state.body_q.assign(state)
            viewer.begin_frame(row["time_s"])
            viewer.log_state(render_state)
            viewer.end_frame()
        viewer.save_recording()

        server.gui.configure_theme(dark_mode=True, show_share_button=False,
                                   brand_color=(40, 170, 130))
        server.gui.main_panel.dock_right()
        server.gui.add_markdown("## SO-101 · Apretar un tornillo\n"
                                "**Newton en CPU — simulación física grabada**\n\n"
                                "Mesa, soporte, herramienta y cabeza con colisiones activas. "
                                "El asiento se resuelve por contacto.\n\n"
                                "Rosca por restricción helicoidal y acoplamiento torsional idealizado. "
                                "No se simula contacto entre filetes ni control por LLM.")
        server.gui.add_markdown("**Ensayo completo: verificado por el núcleo físico.** "
                                "El visor reproduce los estados calculados; no reintegra la física.")
        progress = server.gui.add_slider("Tiempo del ensayo (s)", min=0.0,
                                        max=float(rows[-1]["time_s"]),
                                        step=float(demo.frame_dt), initial_value=0.0,
                                        disabled=True)
        info = server.gui.add_markdown("")
        playing = server.gui.add_checkbox("Reproducir", initial_value=True)
        rate = server.gui.add_slider("Velocidad", min=0.25, max=2.0, step=0.25, initial_value=1.0)
        overview = server.gui.add_button("Vista del brazo")
        detail = server.gui.add_button("Acercar al tornillo")
        restart = server.gui.add_button("Repetir desde el inicio")
        server.gui.add_markdown(f"Resultados locales: `{output}`")
        index = 0
        rewind = False

        @overview.on_click
        def _overview(_):
            camera_view(viewer, target)

        @detail.on_click
        def _detail(_):
            camera_view(viewer, getattr(demo, "screw_target", target), close=True)

        @restart.on_click
        def _restart(_):
            nonlocal rewind
            rewind = True
            playing.value = True

        url = f"http://127.0.0.1:{server.get_port()}"
        write_json(output / "viewer.json", {"url": url, "display_mode": "replay",
                                            "recording": str(output / "episode.viser"),
                                            "pid": __import__("os").getpid()})
        print(f"SCREW_DEMO_READY {url} evidence={output}", flush=True)
        if open_browser:
            webbrowser.open(url)
        started = time.monotonic()
        while not stop and viewer.is_running():
            tick = time.monotonic()
            if serve_seconds is not None and tick - started >= serve_seconds:
                break
            if rewind:
                index, rewind = 0, False
            row = rows[index]
            render_state.body_q.assign(states[index])
            viewer.begin_frame(row["time_s"])
            viewer.log_state(render_state)
            viewer.end_frame()
            progress.value = row["time_s"]
            info.content = (
                f"### Fase: {row['phase']}\n\n"
                f"Giro del tornillo: **{row['screw_turns']:.3f} vueltas**\n\n"
                f"Avance axial: **{row['axial_mm']:.3f} mm**\n\n"
                f"Par del motor: **{row['motor_torque_nm']:.4f} N·m**\n\n"
                f"Par del acoplamiento ideal: **{row['applied_torque_nm']:.4f} N·m**\n\n"
                f"Contacto cabeza–soporte: **{row['seating_contact_force_n']:.3f} N**\n\n"
                f"Contacto punta–cabeza: **{row['tip_contact_force_n']:.3f} N**"
            )
            if playing.value:
                if index == len(rows) - 1:
                    playing.value = False
                else:
                    index += 1
            delay = float(demo.frame_dt) / float(rate.value)
            time.sleep(max(0.001, delay - (time.monotonic() - tick)))
    finally:
        viewer.close()
        for sig, handler in old_signals.items():
            signal.signal(sig, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true", help="simulate and verify, without opening a server")
    parser.add_argument("--no-open", action="store_true")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-sim-seconds", type=float, default=30.0)
    parser.add_argument("--serve-seconds", type=float)
    parser.add_argument("--disable-drive", action="store_true", help="negative control: should refuse success")
    parser.add_argument("--disengaged", action="store_true", help="negative control: should refuse success")
    args = parser.parse_args()
    if not 0 <= args.port <= 65535:
        parser.error("port must be between 0 and 65535")
    output = (args.output_dir or REPO / "runs/newton-screw" / uuid.uuid4().hex).resolve()
    sys.path.insert(0, str(REPO / "benchmark/diagnostics"))
    warp = importlib.import_module("warp")
    warp.config.log_level = warp.LOG_WARNING
    scene = importlib.import_module("newton_screw_scene")
    assert scene.__file__ is not None
    demo = scene.ScrewDemo(enable_drive=not args.disable_drive, disengaged=args.disengaged)
    recording = simulate(demo, output, max_sim_seconds=args.max_sim_seconds)
    provenance = {
        "newton": importlib.import_module("newton").__version__,
        "warp": importlib.import_module("warp").__version__,
        "device": str(demo.model.device),
        "python": sys.executable,
        "body_labels": list(demo.model.body_label),
        "joint_labels": list(demo.model.joint_label),
        "shape_labels": list(demo.model.shape_label),
        "shape_flags": demo.model.shape_flags.numpy().tolist(),
        "source_sha256": {str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in (Path(__file__), Path(scene.__file__))},
    }
    write_json(output / "provenance.json", provenance)
    print(json.dumps(recording["summary"], indent=2), flush=True)
    print(f"[newton-screw] evidence: {output}", flush=True)
    if not args.headless:
        replay(demo, recording, output, port=args.port,
               open_browser=not args.no_open, serve_seconds=args.serve_seconds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
