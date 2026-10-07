"""The Unitree H2 policy bundle contract: pins, layout and fail-closed drift checks."""
import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest

from cascade.config import CONFIG_DIR
from cascade.control import h2_policy_contract as contract_module
from cascade.control.h2_policy_contract import (H2PolicyContract, TermHistory, load_bundle_manifest,
                                                load_bundle_yaml, vendored_path)

POLICY_SHA = "0a47182bbe203eddc2b7a9185f822c14d3799723f8b8392c7a2895d25d449c83"


@pytest.fixture(scope="module")
def documents():
    manifest = load_bundle_manifest()
    return (load_bundle_yaml(vendored_path(manifest, "IO_descriptors.yaml")),
            load_bundle_yaml(vendored_path(manifest, "env.yaml")), manifest)


@pytest.fixture(scope="module")
def contract():
    return H2PolicyContract.from_bundle()


def test_vendored_bundle_files_match_their_pins_and_the_isaac_sim_policy_digest(documents):
    _, _, manifest = documents
    files = manifest["files"]
    assert files["policy.pt"]["sha256"] == POLICY_SHA and files["policy.pt"]["vendored"] is False
    for name in ("env.yaml", "IO_descriptors.yaml"):
        path = CONFIG_DIR.parent / "assets" / "h2" / "bundle" / name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == files[name]["sha256"]
        assert path.read_text().splitlines()[1].strip() == "# SPDX-License-Identifier: Apache-2.0"
    assert files["H2.usda"]["vendored"] is False
    assert manifest["asset_root"].startswith("https://") and manifest["model"] == {
        "input_width": 255, "output_width": 14, "format": "TorchScript"}


def test_contract_reproduces_the_exported_layout(contract):
    assert contract.model_sha256 == POLICY_SHA
    assert len(contract.joint_names) == 31 and len(contract.policy_joint_names) == 14
    assert len(contract.held_joint_names) == 17 and "waist_yaw_joint" in contract.held_joint_names
    assert set(contract.policy_joint_names).isdisjoint(contract.held_joint_names)
    assert [(t.name, t.width, t.history_length) for t in contract.observation_terms] == [
        ("base_ang_vel", 3, 5), ("projected_gravity", 3, 5), ("generated_commands", 3, 5),
        ("joint_pos_rel", 14, 5), ("joint_vel_rel", 14, 5), ("last_action", 14, 5)]
    assert contract.observation_width == 255
    assert contract.observation_terms[0].scale == pytest.approx(0.2)
    assert contract.observation_terms[4].scale == pytest.approx(0.05)
    assert (contract.physics_dt, contract.control_dt, contract.decimation) == (0.005, 0.02, 4)
    assert contract.command_ranges == {"lin_vel_x": (-0.5, 0.5), "lin_vel_y": (-0.5, 0.5), "ang_vel_z": (-1.0, 1.0)}
    assert contract.action_scale == 0.5
    # offsets are the default standing pose of the commanded joints
    assert contract.action_offset[contract.policy_joint_names.index("left_knee_joint")] == pytest.approx(0.3)
    assert contract.action_offset[contract.policy_joint_names.index("left_ankle_pitch_joint")] == pytest.approx(-0.2)
    assert contract.gains["left_knee_joint"].stiffness == 300.0 and contract.gains["left_knee_joint"].group == "legs"
    assert contract.gains["left_ankle_roll_joint"].effort_limit == 19.0
    assert contract.gains["waist_yaw_joint"].group == "waist" and contract.gains["head_yaw_joint"].group == "head"
    assert contract.init_root_pos[2] == pytest.approx(1.015)
    assert contract.fall_tilt_rad == pytest.approx(math.radians(30)) and contract.fall_pelvis_height_m == 0.615
    layout = contract.observation_layout()
    assert layout[0] == ("base_ang_vel", 0, 0, 3) and layout[-1] == ("last_action", 4, 241, 255)
    assert all(b[2] == a[3] for a, b in zip(layout, layout[1:]))


def test_sample_scaling_history_order_and_action_decode_follow_isaac_sim(contract):
    zeros14 = [0.0] * 14
    sample = contract.scaled_sample(base_ang_vel=[1.0, 2.0, 3.0], projected_gravity=[0, 0, -1],
                                    command=[0.5, 0.0, -1.0], joint_pos=contract.action_offset,
                                    joint_vel=[2.0] * 14, last_action=zeros14)
    assert sample.shape == (51,) and sample.dtype == np.float32
    assert sample[:3] == pytest.approx([0.2, 0.4, 0.6])        # base_ang_vel * 0.2
    assert sample[6:9] == pytest.approx([0.5, 0.0, -1.0])      # commands unscaled
    assert sample[9:23] == pytest.approx(zeros14)              # joint_pos_rel against the offsets
    assert sample[23:37] == pytest.approx([0.1] * 14)          # joint_vel * 0.05
    history = TermHistory(contract)
    first = history.append(sample)
    assert first.shape == (255,) and first[:15] == pytest.approx(np.tile(sample[:3], 5))  # backfilled
    second = contract.scaled_sample(base_ang_vel=[5.0, 0, 0], projected_gravity=[0, 0, -1], command=[0, 0, 0],
                                    joint_pos=contract.action_offset, joint_vel=zeros14, last_action=zeros14)
    expanded = history.append(second)
    # oldest -> newest: four old frames of the first sample, then the new one
    assert expanded[:12] == pytest.approx(np.tile(sample[:3], 4)) and expanded[12:15] == pytest.approx(second[:3])
    history.reset()
    assert history.append(second)[:15] == pytest.approx(np.tile(second[:3], 5))
    with pytest.raises(ValueError, match="nonfinite"):
        contract.scaled_sample(base_ang_vel=[math.nan, 0, 0], projected_gravity=[0, 0, -1], command=[0, 0, 0],
                               joint_pos=contract.action_offset, joint_vel=zeros14, last_action=zeros14)
    with pytest.raises(ValueError):
        history.append(sample[:-1])
    targets = contract.decode_action([0.0] * 14)
    assert targets == {n: pytest.approx(o) for n, o in zip(contract.policy_joint_names, contract.action_offset)}
    raw = [40.0] + [0.0] * 13   # clipped to +10 before scaling
    assert contract.decode_action(raw)["left_hip_pitch_joint"] == pytest.approx(-0.1 + 0.5 * 10.0)
    for bad in ([0.0] * 13, [math.inf] + [0.0] * 13):
        with pytest.raises(ValueError):
            contract.decode_action(bad)
    held = contract.held_targets()
    assert set(held) == set(contract.held_joint_names) and held["left_shoulder_pitch_joint"] == 0.0
    assert contract.command_within_training_ranges(0.5, -0.5, 1.0)
    assert not contract.command_within_training_ranges(0.51, 0.0, 0.0)


@pytest.mark.parametrize("drift", ["term_order", "history", "extra_term", "action_shape", "offset", "joint_count",
                                   "timing", "gains_missing", "manifest_width", "action_joint_order"])
def test_any_drift_from_the_exported_bundle_is_refused(documents, drift):
    descriptor, env, manifest = (copy.deepcopy(d) for d in documents)
    policy = descriptor["observations"]["policy"]
    action = next(a for a in descriptor["actions"] if a["name"] == "joint_position_action")
    if drift == "term_order":
        policy[0], policy[1] = policy[1], policy[0]
    elif drift == "history":
        policy[2]["overloads"]["history_length"] = 4
    elif drift == "extra_term":
        policy.append(copy.deepcopy(policy[0]))
    elif drift == "action_shape":
        action["shape"] = [13]
    elif drift == "offset":
        action["offset"][0] += 0.05
    elif drift == "joint_count":
        descriptor["articulations"]["robot"]["joint_names"].append("extra_joint")
    elif drift == "timing":
        env["sim"]["dt"] = 0.004
    elif drift == "gains_missing":
        env["scene"]["robot"]["actuators"]["legs"]["stiffness"].pop(".*_knee_joint")
    elif drift == "manifest_width":
        manifest["model"]["input_width"] = 204
    elif drift == "action_joint_order":
        j = action["joint_names"]
        j[0], j[1] = j[1], j[0]
    with pytest.raises(ValueError):
        H2PolicyContract.from_documents(descriptor, env, manifest)


def test_tampered_vendored_file_and_executable_yaml_tags_are_refused(tmp_path, documents):
    _, _, manifest = documents
    config = tmp_path / "configs"
    (config / "h2").mkdir(parents=True)
    (config / "h2" / "bundle.json").write_text(json.dumps(manifest))
    bundle = tmp_path / "assets" / "h2" / "bundle"
    bundle.mkdir(parents=True)
    for name in ("env.yaml", "IO_descriptors.yaml"):
        (bundle / name).write_bytes((CONFIG_DIR.parent / "assets" / "h2" / "bundle" / name).read_bytes())
    assert H2PolicyContract.from_bundle(config).observation_width == 255
    path = bundle / "IO_descriptors.yaml"
    path.write_bytes(path.read_bytes().replace(b"history_length: 5", b"history_length: 4", 1))
    with pytest.raises(ValueError, match="pinned SHA-256"):
        H2PolicyContract.from_bundle(config)
    evil = tmp_path / "evil.yaml"
    evil.write_text("x: !!python/object/apply:os.system ['echo pwned']\n")
    with pytest.raises(ValueError, match="refusing to construct python tag"):
        load_bundle_yaml(evil)
    assert load_bundle_yaml(Path(CONFIG_DIR.parent / "assets" / "h2" / "bundle" / "env.yaml"))["terminations"]["base_orientation"]["params"][
        "asset_cfg"]["joint_ids"] == {"__slice__": (None, None, None)}
    bad = json.loads((CONFIG_DIR / "h2" / "bundle.json").read_text())
    bad["files"]["policy.pt"]["sha256"] = "nothex"
    (config / "h2" / "bundle.json").write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        contract_module.load_bundle_manifest(config)
