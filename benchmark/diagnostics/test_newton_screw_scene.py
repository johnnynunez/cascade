"""Task-level physics tests. Use the isolated Newton interpreter, no mocks."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]


def scene_class():
    path = ROOT / "benchmark/diagnostics/newton_screw_scene.py"
    assert path.is_file(), "The actual Newton screw scene has not been implemented"
    spec = importlib.util.spec_from_file_location("screw_scene", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ScrewDemo


def test_tightens_with_measured_rotation_and_axial_travel(tmp_path):
    demo = scene_class()()
    initial = demo.state.body_q.numpy().copy()
    rows = []
    metrics = demo.metrics()
    while demo.sim_time < 24:
        demo.step()
        metrics = demo.metrics()
        rows.append(metrics)
        for field in ("body_q", "body_qd", "joint_q", "joint_qd"):
            assert np.isfinite(getattr(demo.state, field).numpy()).all(), field
        if metrics["completed"]:
            break
    (tmp_path / "physics.json").write_text(json.dumps(rows, indent=2))
    print(json.dumps(metrics, indent=2))
    assert metrics["verified"] is True
    assert metrics["screw_turns"] > 1.5
    assert 3.8 < metrics["axial_mm"] < 5.0
    assert metrics["max_pitch_error_mm"] < 0.15
    assert 0.005 < metrics["peak_motor_torque_nm"] <= demo.motor_limit + 1e-5
    assert metrics["held_after_drive_off_s"] >= 0.5
    assert abs(metrics["screw_speed_rad_s"]) < 0.02, "Completion must wait for physical rest, not a timer"
    assert 0.01 < metrics["peak_seating_contact_force_n"] < 100.0, "Reject inflated contact loads from artificial axial inertia"
    assert metrics["seating_contact_force_n"] > 0.01
    assert np.linalg.norm(demo.state.body_q.numpy()[demo.screw_body, :3]
                          - initial[demo.screw_body, :3]) > 0.0038
    assert metrics["tool_travel_m"] > 0.02
    assert metrics["engaged_steps"] > 0
    assert metrics["solver"] == "Newton SolverMuJoCo / MuJoCo Warp CPU"


@pytest.mark.parametrize("case", ["no_drive", "misaligned"])
def test_negative_control_does_not_tighten(tmp_path, case):
    demo = scene_class()(enable_drive=case != "no_drive", disengaged=case == "misaligned")
    rows = []
    for _ in range(300):
        demo.step()
        rows.append(demo.metrics())
    (tmp_path / f"{case}.json").write_text(json.dumps(rows, indent=2))
    final = rows[-1]
    assert not any(row["completed"] or row["verified"] for row in rows)
    assert max(abs(row["screw_turns"]) for row in rows) < 0.05
    assert max(abs(row["axial_mm"]) for row in rows) < 0.15
    assert final["tool_travel_m"] > 0.02, "The arm must still perform its approach"
    if case == "no_drive":
        assert final["engaged_steps"] > 0, "Remove drive, not engagement"
        assert final["peak_motor_torque_nm"] == 0.0
    else:
        assert final["peak_motor_torque_nm"] > 0.005, "The detached driver must really be powered"
        assert final["engaged_steps"] == 0
        assert final["peak_coupling_torque_nm"] == 0.0


def test_scene_has_workbench_tool_and_head_colliders():
    import newton

    demo = scene_class()()
    flags = demo.model.shape_flags.numpy()
    labels = [name.rsplit("/", 1)[-1] for name in demo.model.shape_label]
    required = ["bench", "driver_case", "driver_bit", "screw_head", "screw_shaft"]
    required += [name for name in labels if name.startswith("fixture_")]
    for name in required:
        assert flags[labels.index(name)] & int(newton.ShapeFlags.COLLIDE_SHAPES), name
    assert not flags[labels.index("visual_thread_only")] & int(newton.ShapeFlags.COLLIDE_SHAPES)


def test_same_workbench_geometry_really_stops_a_falling_body():
    import newton
    from newton.solvers import SolverMuJoCo
    import warp as wp

    cls = scene_class()
    fixture = cls.__new__(cls)  # Use the actual shared geometry builder, no arm required.
    fixture._fixture_collisions = True
    builder = newton.ModelBuilder()
    fixture._add_fixture(builder, np.array([0.24, 0.0, 0.035]))
    probe = builder.add_body(xform=wp.transform(wp.vec3(-0.05, 0.1, 0.12), wp.quat_identity()))
    builder.add_shape_box(probe, hx=0.015, hy=0.015, hz=0.015)
    model = builder.finalize(device="cpu")
    state, nxt = model.state(), model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    solver = SolverMuJoCo(model, use_mujoco_cpu=False, use_mujoco_contacts=False,
                          nconmax=128, njmax=512, update_data_interval=1)
    pipeline = newton.CollisionPipeline(model, rigid_contact_max=128)
    contacts = pipeline.contacts()
    control = model.control()
    for _ in range(360):
        state.clear_forces()
        pipeline.collide(state, contacts)
        solver.step(state, nxt, control, contacts, 1 / 240)
        state, nxt = nxt, state
    assert abs(float(state.body_q.numpy()[probe, 2]) - 0.015) < 0.001
    assert np.linalg.norm(state.body_qd.numpy()[probe]) < 0.002


def test_removing_seating_contacts_removes_the_stop(tmp_path):
    demo = scene_class()(fixture_collisions=False)
    rows = []
    for _ in range(480):
        demo.step()
        rows.append(demo.metrics())
    (tmp_path / "no_seating_contacts.json").write_text(json.dumps(rows, indent=2))
    assert not any(row["verified"] for row in rows), "An invisible joint stop must not certify seating"
    assert max(row["axial_mm"] for row in rows) > 4.8
