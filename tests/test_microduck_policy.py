"""Contract tests; optional local official artifacts never trigger downloads."""
import ast
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

JOINTS = (
    "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
    "neck_pitch", "head_pitch", "head_yaw", "head_roll", "right_hip_yaw",
    "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
)
HOME = np.array([0., -.0873, -.4579, -.0049, .4530, .3491, .3491,
                 0., 0., 0., .0873, .4579, .0049, -.4530], dtype=np.float32)


def module():
    name = "cascade.control.microduck_policy"
    assert importlib.util.find_spec(name) is not None, "MicroDuck policy contract missing"
    return importlib.import_module(name)


def test_layout_uses_exact_home_and_raw_previous_action():
    m = module()
    q = HOME + np.linspace(-.2, .2, 14, dtype=np.float32)
    dq = np.arange(14, dtype=np.float32) / 10
    previous = np.arange(14, dtype=np.float32) - 7
    command = np.arange(13, dtype=np.float32) / 20
    got = m.observation(q, dq, [1, 2, 3], [0, 0, -1], previous, command)
    expected = np.concatenate(([1, 2, 3, 0, 0, -1], q - HOME, dq, previous, command))
    assert got.shape == (1, 61) and got.dtype == np.float32
    np.testing.assert_array_equal(got[0], expected.astype(np.float32))
    assert m.POLICY_JOINTS == JOINTS
    assert not m.HOME_Q.flags.writeable


def test_joint_mapping_reorders_and_omits_mouth_and_passive():
    m = module()
    names = ("mouth", *reversed(JOINTS), "passive_LF_wheel")
    values = np.r_[999., np.arange(14)[::-1], 777.]
    np.testing.assert_array_equal(m.policy_order(names, values), np.arange(14))
    for bad_names, bad_values in [(names[:-2], values[:-2]),
                                 (names + (JOINTS[0],), np.r_[values, 0]),
                                 (names, values[:-1])]:
        with pytest.raises(ValueError):
            m.policy_order(bad_names, bad_values)


@pytest.mark.parametrize("index,value", [(0, [0]*15), (1, [0]*13), (2, [0]*4),
    (3, [0, np.nan, -1]), (4, [np.inf]*14), (5, [0]*3), (0, [1e100]*14)])
def test_observation_rejects_invalid_dimensions_and_nonfinite(index, value):
    m = module()
    args = [HOME, np.zeros(14), np.zeros(3), [0, 0, -1], np.zeros(14), np.zeros(13)]
    args[index] = value
    with pytest.raises(ValueError):
        m.observation(*args)


def fake_runtime(monkeypatch, *, inputs=None, outputs=None, result=None):
    descriptor = lambda name, shape, dtype="tensor(float)": SimpleNamespace(name=name, shape=shape, type=dtype)
    class Session:
        def __init__(self, data, **kwargs):
            assert isinstance(data, bytes), "hash-verified bytes must be executed, not reopened path"
            assert kwargs["providers"] == ["CPUExecutionProvider"]
        def get_inputs(self):
            return inputs if inputs is not None else [descriptor("obs", [1, 61])]
        def get_outputs(self):
            return outputs if outputs is not None else [descriptor("actions", [1, 14])]
        def run(self, names, feed):
            assert names == ["actions"]
            assert set(feed) == {"obs"}
            # No external observation normalizer: the graph owns normalization.
            np.testing.assert_array_equal(feed["obs"], np.arange(61, dtype=np.float32)[None])
            return [result if result is not None else np.arange(14, dtype=np.float32)[None]]
    monkeypatch.setitem(sys.modules, "onnxruntime", SimpleNamespace(
        InferenceSession=Session, SessionOptions=SimpleNamespace))
    return descriptor


def model_file(tmp_path):
    p = tmp_path / "fixture.onnx"
    p.write_bytes(b"unit-test-session-boundary-not-a-real-model")
    return p, hashlib.sha256(p.read_bytes()).hexdigest()


def test_runner_keeps_raw_history_scales_once_and_resets(monkeypatch, tmp_path):
    m = module()
    fake_runtime(monkeypatch)
    runner = m.MicroduckPolicy(*model_file(tmp_path), action_scale=.9)
    got = runner.infer(np.arange(61, dtype=np.float32)[None])
    np.testing.assert_array_equal(got, np.arange(14, dtype=np.float32))
    np.testing.assert_array_equal(runner.previous_action, got)
    np.testing.assert_allclose(runner.targets(got), HOME + .9 * got)
    got[:] = 100
    assert runner.previous_action[0] == 0
    history = runner.previous_action
    history[:] = 10
    assert runner.previous_action[0] == 0
    runner.reset()
    np.testing.assert_array_equal(runner.previous_action, np.zeros(14))


def test_preview_does_not_commit_cancelled_raw_history(monkeypatch, tmp_path):
    m = module()
    fake_runtime(monkeypatch)
    runner = m.MicroduckPolicy(*model_file(tmp_path))
    raw = runner.preview(np.arange(61, dtype=np.float32)[None])
    np.testing.assert_array_equal(raw, np.arange(14, dtype=np.float32))
    np.testing.assert_array_equal(runner.previous_action, np.zeros(14, np.float32))
    runner.commit(raw)
    raw[:] = 99
    np.testing.assert_array_equal(runner.previous_action, np.arange(14, dtype=np.float32))


def test_robotd_target_profile_first_reset_and_filtered_anchor():
    m = module()
    pipeline = m.MicroduckTargets('robotd-targets-v1')
    home = np.array([0., -.0873, -.4579, -.0049, .4530, .3491, .3491,
                     0., 0., 0., .0873, .4579, .0049, -.4530], dtype=np.float64)
    first = np.linspace(-.25, .25, 14, dtype=np.float32)
    second = -first
    expected_first = home + .9 * first.astype(np.float64)
    np.testing.assert_array_equal(pipeline.preview(first), expected_first)
    assert pipeline.committed is None
    pipeline.commit(first)
    expected = np.array([
        a * (h + .9 * float(x)) + (1. - a) * prev
        for i, (h, x, prev) in enumerate(zip(home, second, expected_first))
        for a in [.5 if 5 <= i < 9 else .7]
    ])
    staged = pipeline.preview(second)
    np.testing.assert_array_equal(staged, expected)
    # Discarding/repeating previews cannot move the accepted filter anchor.
    pipeline.preview(first * 100)
    np.testing.assert_array_equal(pipeline.preview(second), expected)
    np.testing.assert_array_equal(pipeline.committed, expected_first)
    pipeline.commit(second)
    staged[:] = 100
    detached = pipeline.committed
    detached[:] = 200
    np.testing.assert_array_equal(pipeline.committed, expected)
    # The anchor is the filtered result, not the previous unfiltered target.
    expected_next = np.array([
        a * (h + .9 * float(x)) + (1. - a) * prev
        for i, (h, x, prev) in enumerate(zip(home, first, expected))
        for a in [.5 if 5 <= i < 9 else .7]
    ])
    np.testing.assert_array_equal(pipeline.preview(first), expected_next)
    pipeline.reset()
    assert pipeline.committed is None
    np.testing.assert_array_equal(pipeline.preview(first), expected_first)


def test_filtered_policy_discard_commit_raw_observation_and_identity(monkeypatch, tmp_path):
    m = module()
    fake_runtime(monkeypatch)
    runner = m.MicroduckPolicy(*model_file(tmp_path), target_profile='robotd-targets-v1')
    obs = np.arange(61, dtype=np.float32)[None]
    raw = runner.preview(obs)
    staged = runner.targets(raw)
    assert runner.committed_targets is None
    np.testing.assert_array_equal(runner.previous_action, np.zeros(14, np.float32))
    runner.targets(-raw)  # withdrawn candidate: no commit
    np.testing.assert_array_equal(runner.targets(raw), staged)
    runner.commit(raw)
    np.testing.assert_array_equal(runner.committed_targets, staged)
    np.testing.assert_array_equal(runner.previous_action, raw)
    next_obs = m.observation(HOME, np.zeros(14), np.zeros(3), [0, 0, -1],
                             runner.previous_action, np.zeros(13))
    np.testing.assert_array_equal(next_obs[0, 34:48], raw)
    contract = runner.target_contract
    assert contract['policy_sha256'] == runner.sha256
    assert contract['profile'] == 'robotd-targets-v1'
    assert contract['upstream_commit'] == m.ROBOTD_SOURCE
    assert contract['physical_admission'] is False
    contract['profile'] = 'changed'
    assert runner.target_contract['profile'] == 'robotd-targets-v1'
    # Immediate inference returns raw output; committed_targets is its target,
    # while targets(raw) always previews the next slot.
    next_target = runner.targets(raw)
    runner.infer(obs)
    np.testing.assert_array_equal(runner.committed_targets, next_target)
    runner.reset()
    assert runner.committed_targets is None
    np.testing.assert_array_equal(runner.previous_action, np.zeros(14, np.float32))
    np.testing.assert_array_equal(runner.targets(raw), staged)


@pytest.mark.parametrize('profile,scale', [('unknown',None), ('robotd-targets-v1',1.),
    ('robotd-targets-v1',True), ('direct-v1',np.nan), ('direct-v1',0)])
def test_target_profile_refuses_ambiguous_configuration_before_runtime(profile, scale):
    with pytest.raises(ValueError):
        module().MicroduckTargets(profile, scale)


def test_invalid_filtered_commit_preserves_both_histories(monkeypatch, tmp_path):
    m = module()
    fake_runtime(monkeypatch)
    runner = m.MicroduckPolicy(*model_file(tmp_path), target_profile='robotd-targets-v1')
    raw = np.arange(14, dtype=np.float32)
    runner.commit(raw)
    target = runner.committed_targets
    for bad in [np.full(14, np.nan, np.float32), np.zeros(14), np.zeros(15, np.float32)]:
        with pytest.raises(ValueError):
            runner.commit(bad)
        np.testing.assert_array_equal(runner.previous_action, raw)
        np.testing.assert_array_equal(runner.committed_targets, target)


def test_direct_default_retains_raw_only_commit_and_float32_mapping(monkeypatch, tmp_path):
    m = module()
    fake_runtime(monkeypatch)
    runner = m.MicroduckPolicy(*model_file(tmp_path), action_scale=1e100)
    raw = np.full(14, np.finfo(np.float32).max, np.float32)
    runner.commit(raw)  # Legacy raw history accepts finite action independently of targets.
    np.testing.assert_array_equal(runner.previous_action, raw)
    with pytest.raises(ValueError):
        runner.targets(raw)
    direct = m.MicroduckTargets()
    raw = np.linspace(-.25, .25, 14, dtype=np.float32)
    assert direct.preview(raw).dtype == np.float32
    np.testing.assert_array_equal(direct.preview(raw), HOME + raw)


@pytest.mark.parametrize('bad', [np.zeros(14), np.zeros(15, np.float32),
                               np.full(14, np.nan, np.float32), True])
def test_commit_rejects_non_raw_output_without_mutating_history(monkeypatch, tmp_path, bad):
    m = module()
    fake_runtime(monkeypatch)
    runner = m.MicroduckPolicy(*model_file(tmp_path))
    with pytest.raises(ValueError, match='action'):
        runner.commit(bad)
    np.testing.assert_array_equal(runner.previous_action, np.zeros(14, np.float32))


@pytest.mark.parametrize("kind", ["input_shape", "input_name", "input_type", "recurrent", "output_shape", "output_type"])
def test_rejects_unknown_or_recurrent_contract(monkeypatch, tmp_path, kind):
    m = module()
    desc = lambda name, shape, dtype="tensor(float)": SimpleNamespace(name=name, shape=shape, type=dtype)
    inputs = [desc("obs", [1, 61])]
    outputs = [desc("actions", [1, 14])]
    if kind == "input_shape": inputs[0].shape = [1, 51]
    if kind == "input_name": inputs[0].name = "state"
    if kind == "input_type": inputs[0].type = "tensor(double)"
    if kind == "recurrent": inputs.append(desc("h_in", [1, 128]))
    if kind == "output_shape": outputs[0].shape = [1, 15]
    if kind == "output_type": outputs[0].type = "tensor(double)"
    fake_runtime(monkeypatch, inputs=inputs, outputs=outputs)
    with pytest.raises(ValueError):
        m.MicroduckPolicy(*model_file(tmp_path))


@pytest.mark.parametrize("obs", [np.zeros(61, dtype=np.float32), np.zeros((1,61)),
                                np.full((1,61), np.nan, dtype=np.float32)])
def test_infer_rejects_bad_input_before_session(monkeypatch, tmp_path, obs):
    m = module()
    fake_runtime(monkeypatch)
    runner = m.MicroduckPolicy(*model_file(tmp_path))
    with pytest.raises(ValueError): runner.infer(obs)


@pytest.mark.parametrize("result", [np.zeros((1, 15), dtype=np.float32),
    np.full((1,14), np.nan, dtype=np.float32), np.zeros((1,14))])
def test_bad_output_never_updates_history(monkeypatch, tmp_path, result):
    m = module()
    fake_runtime(monkeypatch, result=result)
    runner = m.MicroduckPolicy(*model_file(tmp_path))
    with pytest.raises(ValueError): runner.infer(np.arange(61, dtype=np.float32)[None])
    np.testing.assert_array_equal(runner.previous_action, np.zeros(14))


def test_hash_scale_and_targets_fail_closed(monkeypatch, tmp_path):
    m = module()
    fake_runtime(monkeypatch)
    path, digest = model_file(tmp_path)
    for bad in ["a"*64, "", " " + digest, digest.upper(), "z"*64]:
        with pytest.raises(ValueError): m.MicroduckPolicy(path, bad)
    for scale in [np.nan, np.inf, 0, -1, True]:
        with pytest.raises(ValueError): m.MicroduckPolicy(path, digest, scale)
    runner = m.MicroduckPolicy(path, digest)
    for action in [np.zeros(15), np.full(14, np.inf), np.zeros((1, 14))]:
        with pytest.raises(ValueError): runner.targets(action)


def test_imports_are_lazy_and_offline():
    module()
    src = Path(__file__).resolve().parents[1] / "src"
    code = """
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'onnxruntime','onnx','torch','mujoco','isaacsim','urllib','requests'}:
            raise AssertionError('unexpected optional/network import: ' + fullname)
sys.meta_path.insert(0, Block())
import cascade.control.microduck_policy
import cascade.control.microduck_actuator
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                            env={**os.environ, "PYTHONPATH": str(src), "CUDA_VISIBLE_DEVICES": "-1"})
    assert result.returncode == 0, result.stderr


def reference_root():
    value = os.environ.get("MICRODUCK_REFERENCE_ROOT")
    if not value:
        pytest.skip("set MICRODUCK_REFERENCE_ROOT to the pinned local policy research directory")
    root = Path(value)
    assert root.is_dir(), root
    return root


def test_all_official_onnx_match_source_observation_and_direct_cpu():
    m = module()
    root = reference_root()
    import onnxruntime as ort
    audit = json.loads((root / "weights-static-audit.json").read_text())
    source = root / "sources/microduck_rl/scripts/infer_policy.py"
    admission = json.loads((Path(__file__).resolve().parents[1] / "assets/microduck/manifest.json").read_text())
    admitted = {row["path"]: row for row in admission["files"]}
    assert hashlib.sha256(source.read_bytes()).hexdigest() == admitted["microduck_rl/scripts/infer_policy.py"]["sha256"]
    for entry in audit["artifacts"]:
        assert entry["sha256"] == admitted["microduck-policies/" + entry["file"]]["sha256"]
    # Extract ONLY the official pure assembly method, no simulator import/startup.
    tree = ast.parse(source.read_text())
    method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "get_observations")
    env = {"np": np}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), env)
    rng = np.random.default_rng(476)
    fixtures = [("home_zero", HOME.copy(), np.zeros(14), np.zeros(3), [0,0,-1], np.zeros(14), np.zeros(13)),
                ("moving_raw_history", HOME + rng.normal(0,.1,14), rng.normal(0,.2,14),
                 [.1,-.2,.3], [.1,.2,-np.sqrt(.95)], rng.normal(0,.3,14), rng.normal(0,.1,13))]
    assert audit["count"] == len(audit["artifacts"]) == 10
    for entry in audit["artifacts"]:
        path = root / "sources/microduck-policies" / entry["file"]
        runner = m.MicroduckPolicy(path, entry["sha256"])
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        ref = ort.InferenceSession(path.read_bytes(), sess_options=opts, providers=["CPUExecutionProvider"])
        assert ref.get_providers() == ["CPUExecutionProvider"]
        for name, q, dq, gyro, gravity, previous, command in fixtures:
            upstream = SimpleNamespace(
                get_base_ang_vel=lambda: np.asarray(gyro, dtype=np.float32),
                use_projected_gravity=True,
                get_projected_gravity=lambda: np.asarray(gravity, dtype=np.float32),
                get_joint_pos_relative=lambda: np.asarray(q, dtype=np.float32) - HOME,
                get_joint_vel=lambda: np.asarray(dq, dtype=np.float32),
                last_action=np.asarray(previous, dtype=np.float32), command=np.asarray(command, dtype=np.float32))
            obs_ref = env["get_observations"](upstream)[None]
            obs = m.observation(q, dq, gyro, gravity, previous, command)
            np.testing.assert_array_equal(obs, obs_ref, err_msg=name)
            expected = ref.run(["actions"], {"obs": obs_ref})[0][0]
            actual = runner.infer(obs)
            np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6, err_msg=entry["file"]+name)
            assert np.isfinite(actual).all()
        runner.reset()
        np.testing.assert_array_equal(runner.previous_action, np.zeros(14))
