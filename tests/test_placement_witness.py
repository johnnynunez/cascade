"""Independent placement verdicts on synthetic records; no task admission."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from cascade.eval.placement import PlacementPolicy, seal_placement_row, verify_placement_window


def policy():
    return PlacementPolicy("a"*64, "episode", (0,), (1,), (2,), .1, (0., 0., -9.81), .002, 1, 2, 4)


def row(step, p):
    return {"model_identity_sha256": p.model_identity_sha256, "epoch": p.epoch, "policy_sha256": p.sha256,
        "solver_step": step, "constraint_time_s": (step-1)*.002, "advanced_time_s": step*.002,
        "phase": "euler_constraint_before_integration", "coverage": "all_native_contact_candidates",
        "coupling_admission": p.constraint_recipe,
        "native_ngeom": 4, "ncon": 1, "nefc": 3, "native_warnings": [], "external_forces_zero": True,
        "bodies": {name: {"body_id": getattr(p, name+"_body_id"), "position_m": pos, "linear_velocity_m_s": [0., 0., 0.],
                           "angular_velocity_rad_s": [0., 0., 0.]}
                   for name, pos in (("object", [0., 0., .15]), ("support", [0., 0., 0.]))},
        "geometries": [{"id": i, "position_m": pos, "bound_radius_m": radius}
                       for i, pos, radius in ((0, [0., 0., .15], .05), (1, [0., 0., 0.], .2), (2, [1., 0., 0.], .08))],
        "contacts": [{"index": 0, "geom_a": 1, "geom_b": 0, "efc_address": 0, "dimension": 3,
                      "distance_m": -.0001, "position_m": [0., 0., .1],
                      "frame_rows": [[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]],
                      "wrench_on_b_contact": [.981, 0., 0., 0., 0., 0.], "force_on_b_world_n": [0., 0., .981]}]}


def window(p=None):
    p = p or policy()
    return [row(step, p) for step in range(1, 252)]


def verify(rows, p=None):
    return verify_placement_window([seal_placement_row(r) for r in rows], p or policy(),
                                   first_solver_step=1, last_solver_step=251)


def test_explicit_complete_window_has_release_support_but_no_containment_credit():
    rows = window()
    old = deepcopy(rows)
    result = verify(rows)
    assert result["status"] == "confirmed", result
    assert result["containment"] == "unverified"
    assert result["measured"]["window_sim_s"] == .5
    assert rows == old


def test_force_on_b_orientation_is_not_assumed_from_geometry_order():
    rows = window()
    for r in rows:
        c = r["contacts"][0]
        c.update(geom_a=0, geom_b=1, force_on_b_world_n=[0., 0., -.981],
                 frame_rows=[[0., 0., -1.], [1., 0., 0.], [0., -1., 0.]])
    assert verify(rows)["status"] == "confirmed"


@pytest.mark.parametrize("fault", ["no_support", "wrong_support", "downward", "too_much_force",
    "robot_contact", "robot_too_close", "object_moving", "support_moving", "rotating", "drifting"])
def test_one_observed_violation_refutes_whole_window_without_hunting_a_suffix(fault):
    rows = window()
    r = rows[3]
    if fault == "no_support":
        r.update(ncon=0, contacts=[])
    elif fault == "wrong_support":
        r["contacts"][0]["geom_a"] = 3
    elif fault == "downward":
        r["contacts"][0].update(force_on_b_world_n=[0., 0., -.981],
            frame_rows=[[0., 0., -1.], [1., 0., 0.], [0., -1., 0.]])
    elif fault == "too_much_force":
        r["contacts"][0].update(force_on_b_world_n=[0., 0., 9.81], wrench_on_b_contact=[9.81, 0., 0., 0., 0., 0.])
    elif fault == "robot_contact":
        extra = deepcopy(r["contacts"][0]); extra.update(index=1, geom_a=2)
        r["contacts"].append(extra); r["ncon"] = 2
    elif fault == "robot_too_close":
        r["geometries"][2]["position_m"] = [0., 0., .27]
    elif fault in ("object_moving", "support_moving"):
        r["bodies"][fault.split("_")[0]]["linear_velocity_m_s"][0] = .02
    elif fault == "rotating":
        r["bodies"]["object"]["angular_velocity_rad_s"][0] = .2
    else:
        r["bodies"]["object"]["position_m"][0] = .004
    assert verify(rows)["status"] == "refuted"


@pytest.mark.parametrize("fault", ["missing_solve", "duplicate_solve", "epoch", "model", "policy",
    "clock", "phase", "truncated_contacts", "contact_index", "warning", "external_force",
    "missing_geom", "duplicate_geom", "force_frame", "inactive_force", "nan", "bool_step", "constraint",
    "body_identity", "geometry_inventory"])
def test_incomplete_or_inconsistent_native_evidence_is_unverified(fault):
    rows = window()
    r = rows[100]
    if fault == "missing_solve": rows.pop(100)
    elif fault == "duplicate_solve": r["solver_step"] -= 1
    elif fault in ("epoch", "phase"): r[fault] = "other"
    elif fault == "model": r["model_identity_sha256"] = "b"*64
    elif fault == "policy": r["policy_sha256"] = "b"*64
    elif fault == "clock": r["advanced_time_s"] += .01
    elif fault == "truncated_contacts": r["ncon"] += 1
    elif fault == "contact_index": r["contacts"][0]["index"] = 1
    elif fault == "warning": r["native_warnings"] = [1]
    elif fault == "external_force": r["external_forces_zero"] = False
    elif fault == "missing_geom": r["geometries"].pop()
    elif fault == "duplicate_geom": r["geometries"].append(r["geometries"][0])
    elif fault == "force_frame": r["contacts"][0]["force_on_b_world_n"][2] *= -1
    elif fault == "inactive_force": r["contacts"][0]["efc_address"] = -1
    elif fault == "nan": r["bodies"]["object"]["position_m"][0] = "nan"
    elif fault == "bool_step": r["solver_step"] = True
    elif fault == "constraint": r["contacts"][0]["efc_address"] = r["nefc"]
    elif fault == "body_identity": r["bodies"]["object"]["body_id"] = 3
    elif fault == "geometry_inventory": r["native_ngeom"] += 1
    assert verify(rows)["status"] == "unverified"


def test_snapshot_mutation_and_insufficient_rest_never_pass():
    p = policy()
    rows = [seal_placement_row(r) for r in window(p)]
    rows[-1]["contacts"] = []
    assert verify_placement_window(rows, p, first_solver_step=1, last_solver_step=251)["status"] == "unverified"
    rows = [seal_placement_row(r) for r in window(p)[:100]]
    assert verify_placement_window(rows, p, first_solver_step=1, last_solver_step=100)["status"] == "unverified"


@pytest.mark.parametrize("updates", [{"object_geoms": [0]}, {"robot_geoms": (0,)}, {"support_geoms": ()},
    {"object_mass_kg": True}, {"gravity_world_m_s2": (0., 0., 0.)}, {"rest_s": -.5},
    {"force_fraction_max": .5}, {"epoch": ""}, {"model_identity_sha256": "unknown"}])
def test_policy_requires_explicit_disjoint_immutable_binding(updates):
    with pytest.raises(ValueError):
        replace(policy(), **updates)


def recorder_fixture(tmp_path, monkeypatch, **budgets):
    # Exercise the recorder's real lifecycle and archive code without importing
    # an SDK, constructing a model, or treating these records as physical data.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from benchmark.vab.placement_witness import PlacementRecorder
    contact = row(1, policy())["contacts"][0]
    m = SimpleNamespace(nbody=4, ngeom=4, njnt=2, nu=0, neq=0, ntendon=0, nplugin=0,
        opt=SimpleNamespace(integrator=0, gravity=np.array([0., 0., -9.81]), timestep=.002),
        body_parentid=np.array([0, 0, 0, 0]), body_jntnum=np.array([0, 1, 1, 0]),
        body_jntadr=np.array([-1, 0, 1, -1]), jnt_type=np.array([0, 0]),
        jnt_dofadr=np.array([0, 6]), jnt_stiffness=np.zeros(2), dof_frictionloss=np.zeros(12),
        body_gravcomp=np.zeros(4), jnt_bodyid=np.array([1, 2]),
        actuator_trntype=np.zeros(0, dtype=int), actuator_trnid=np.zeros((0, 2), dtype=int),
        body_mass=np.array([0., .1, 1., 1.]), geom_bodyid=np.array([1, 2, 3, 0]),
        pair_geom1=np.zeros(0, dtype=int), pair_geom2=np.zeros(0, dtype=int),
        geom_contype=np.array([1, 1, 1, 0]), geom_conaffinity=np.array([1, 1, 1, 0]),
        geom_type=np.full(4, 6), geom_size=np.array([[.02,.02,.02],[.1,.1,.1],[.04,.04,.04],[1.,1.,1.]]),
        geom_pos=np.zeros((4,3)), geom_quat=np.tile([1.,0.,0.,0.],(4,1)),
        geom_rbound=np.array([.05, .2, .08, 0.]))
    d = SimpleNamespace(time=0., warning=SimpleNamespace(number=np.zeros(8, dtype=int)),
        xpos=np.array([[0., 0., 0.], [0., 0., .15], [0., 0., 0.], [1., 0., 0.]]),
        geom_xpos=np.array([[0., 0., .15], [0., 0., 0.], [1., 0., 0.], [0., 0., 0.]]),
        xmat=np.tile(np.eye(3).ravel(),(4,1)), geom_xmat=np.tile(np.eye(3).ravel(),(4,1)),
        ncon=1, nefc=3, qpos=np.zeros(14), qvel=np.zeros(12), ctrl=np.zeros(1),
        xfrc_applied=np.zeros((4, 6)), qfrc_applied=np.zeros(12),
        contact=[SimpleNamespace(geom1=1, geom2=0, efc_address=0, dim=3, dist=-.0001,
                                 frame=np.array(contact["frame_rows"]).ravel(), pos=np.array([0., 0., .1]))])
    def wrench(_m, _d, _i, out): out[:] = contact["wrench_on_b_contact"]
    mj = SimpleNamespace(mjtIntegrator=SimpleNamespace(mjINT_EULER=0),
        mjtJoint=SimpleNamespace(mjJNT_FREE=0), mjtObj=SimpleNamespace(mjOBJ_BODY=1),
        mjtGeom=SimpleNamespace(mjGEOM_BOX=6),
        mjtTrn=SimpleNamespace(mjTRN_JOINT=0),
        mj_name2id=lambda *_: 3, mj_contactForce=wrench,
        mju_quat2Mat=lambda out,_: out.__setitem__(slice(None),np.eye(3).ravel()),
        mj_objectVelocity=lambda *_: None)
    for callback in ("control", "passive", "sensor", "contactfilter", "act_dyn", "act_gain", "act_bias", "time"):
        setattr(mj, "get_mjcb_"+callback, lambda: None)
    monkeypatch.setitem(sys.modules, "mujoco", mj)
    class Sim:
        calls = 0
        model = SimpleNamespace(_model=m)
        data = SimpleNamespace(_data=d)
        def step(self, *args, **kwargs):
            self.calls += 1
            d.time += .002
            return "ordinary-step"
    sim = Sim()
    original = sim.step
    env = SimpleNamespace(sim=sim, _obj_body_id={"can": 1, "basket": 2})
    recorder = PlacementRecorder(env, tmp_path/"placement", model_identity_sha256="a"*64,
        epoch="episode", object_name="can", support_name="basket", robot_root_body="robot0_link0", **budgets)
    return recorder, sim, original, m, d


def test_optional_box_geometry_keeps_compiled_shapes_and_same_solve_raw_rotations(tmp_path, monkeypatch):
    import json
    recorder, sim, original, m, d = recorder_fixture(tmp_path, monkeypatch, record_box_geometry=True)
    assert sim.calls == 0
    manifest = json.loads((recorder.out/'box-geometry.json').read_text())
    assert [b['geometry_id'] for b in manifest['object_boxes']] == [0]
    assert [b['geometry_id'] for b in manifest['support_boxes']] == [1]
    sim.step()
    row = json.loads((recorder.out/'solves.jsonl').read_text())
    assert row['box_geometry']['inventory_sha256'] == recorder.box_inventory.sha256
    assert [v['id'] for v in row['box_geometry']['rotations_world']] == [0,1]
    hulls = recorder.box_inventory.object_hulls(row, recorder.policy)
    assert np.allclose(hulls[0].min(axis=0), [-.02,-.02,.13])
    assert np.allclose(hulls[0].max(axis=0), [.02,.02,.17])
    d.geom_xmat[0,0] = 2.
    assert row['box_geometry']['rotations_world'][0]['rotation'][0] == 1.
    assert recorder.close()['ok'] and sim.step == original and sim.calls == 1


@pytest.mark.parametrize('field',['geom_type','geom_size','geom_pos','geom_quat'])
def test_compiled_box_geometry_changes_refuse_before_next_solve(tmp_path, monkeypatch, field):
    recorder, sim, original, m, _ = recorder_fixture(tmp_path, monkeypatch, record_box_geometry=True)
    getattr(m,field).flat[0] += 1
    with pytest.raises(RuntimeError,match='geometry changed'): sim.step()
    assert sim.calls == 0
    assert not recorder.close()['ok'] and sim.step == original


def test_box_geometry_never_silently_substitutes_a_bound_for_an_unsupported_shape(tmp_path, monkeypatch):
    from benchmark.vab.placement_witness import PlacementRecorder
    recorder, sim, original, m, _ = recorder_fixture(tmp_path, monkeypatch)
    assert recorder.close()['ok']
    m.geom_type[0] = 2
    env = SimpleNamespace(sim=sim,_obj_body_id={'can':1,'basket':2})
    with pytest.raises(ValueError,match='all colliders to be boxes'):
        PlacementRecorder(env,tmp_path/'unsupported',model_identity_sha256='a'*64,epoch='another',
            object_name='can',support_name='basket',robot_root_body='robot0_link0',record_box_geometry=True)
    assert sim.calls == 0 and sim.step == original and not (tmp_path/'unsupported').exists()


def test_recorder_reads_only_after_exactly_one_existing_solve_and_restores_hook(tmp_path, monkeypatch):
    import json
    recorder, sim, original, m, d = recorder_fixture(tmp_path, monkeypatch)
    saved = {key: getattr(d, key).copy() for key in ("qpos", "qvel", "ctrl", "xfrc_applied", "qfrc_applied")}
    assert sim.calls == recorder.steps == 0
    for _ in range(251): assert sim.step() == "ordinary-step"
    assert sim.calls == recorder.steps == 251
    rows = [json.loads(s) for s in (recorder.out/"solves.jsonl").read_text().splitlines()]
    result = verify_placement_window(rows, recorder.policy, first_solver_step=1, last_solver_step=251)
    assert result["status"] == "confirmed", result
    for key, value in saved.items(): np.testing.assert_array_equal(getattr(d, key), value)
    assert recorder.close()["ok"] and sim.step == original
    assert sim.calls == 251


@pytest.mark.parametrize("fault", ["max_solves", "max_bytes", "model_change", "clock_change", "capture_error", "hook_change"])
def test_recorder_faults_preserve_failed_closure_and_do_not_replay_steps(tmp_path, monkeypatch, fault):
    budgets = {fault: 1} if fault in ("max_solves", "max_bytes") else {}
    recorder, sim, original, m, d = recorder_fixture(tmp_path, monkeypatch, **budgets)
    if fault == "max_solves": sim.step()
    elif fault == "model_change": m.geom_rbound[0] += .01
    elif fault == "clock_change": m.opt.timestep = .003
    elif fault == "capture_error":
        def fail(*_): raise RuntimeError("failed contact capture")
        recorder._capture = fail
    elif fault == "hook_change":
        foreign = lambda: None
        sim.step = foreign
        assert not recorder.close()["ok"] and sim.step is foreign
        assert sim.calls == 0
        return
    with pytest.raises(RuntimeError): sim.step()
    expected_calls = 0 if fault in ("model_change", "clock_change") else 1
    assert sim.calls == expected_calls
    with pytest.raises(RuntimeError): sim.step()
    assert sim.calls == expected_calls
    assert not recorder.close()["ok"] and sim.step == original


@pytest.mark.parametrize("fault", ["equality", "tendon", "plugin", "object_actuator", "site_actuator",
    "spring", "frictionloss", "gravitycomp", "callback", "unowned_advance"])
def test_hidden_couplings_or_unowned_advances_refuse_before_another_solve(tmp_path, monkeypatch, fault):
    recorder, sim, original, m, d = recorder_fixture(tmp_path, monkeypatch)
    if fault in ("equality", "tendon", "plugin"):
        setattr(m, {"equality": "neq", "tendon": "ntendon", "plugin": "nplugin"}[fault], 1)
    elif fault in ("object_actuator", "site_actuator"):
        m.nu = 1
        m.actuator_trntype = np.array([0 if fault == "object_actuator" else 4])
        m.actuator_trnid = np.array([[0, -1]])
    elif fault == "spring": m.jnt_stiffness[0] = 1.
    elif fault == "frictionloss": m.dof_frictionloss[0] = 1.
    elif fault == "gravitycomp": m.body_gravcomp[1] = 1.
    elif fault == "callback": recorder.mj.get_mjcb_passive = lambda: (lambda: None)
    elif fault == "unowned_advance": d.time = .002
    with pytest.raises((ValueError, RuntimeError)): sim.step()
    assert sim.calls == 0
    assert not recorder.close()["ok"] and sim.step == original
