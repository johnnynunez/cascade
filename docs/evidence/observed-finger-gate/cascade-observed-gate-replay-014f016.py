"""Independent offline replay of historical joint paths on new observed scenes."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np

BASE = Path('/home/johnny/Projects/demo/cascade-lab/CODEX_NEWTON')
REPO = BASE / 'cascade-gripper-scene'
sys.path.insert(0, str(REPO / 'src'))
from cascade.config import load_demo_config
from cascade.control.kinematics import Kinematics
from cascade.control.motion_profile import min_jerk, nominal_profile
from cascade.types import Frame, RobotState

SNAPSHOT = Path('/tmp/cascade-observed-gate-snapshot-014f016')
MODULE = SNAPSHOT / 'src/cascade/grasping/observed_scene.py'
spec = importlib.util.spec_from_file_location('cascade.grasping.observed_scene', MODULE)
gate_module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = gate_module
spec.loader.exec_module(gate_module)
gate_module.ROOT = REPO  # Verify immutable collision source hashes against materialized URDF and collision STL assets.
geometry = gate_module.FingerGeometry()
cfg = load_demo_config(camera='isaac', arm='isaac_kitchen_gpu', llm='mock')
kin = Kinematics(cfg.arm.model, cfg.arm.ee_frame, 6, cfg.arm.joint_signs)
FRAMES, MASKS = BASE / 'passive-scene-05', BASE / 'passive-masks-05b'
CAPTURE_RECEIPT = json.loads((FRAMES / 'receipt.json').read_text())
MASK_RECEIPT = json.loads((MASKS / 'receipt.json').read_text())
EVIDENCE = BASE / 'native-evidence-04-analysis/grasp-evidence-04'

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

report = {
    'scope': 'Historical launch04 joint paths checked against NEW passive05 observed depth. This is a counterfactual discrete surface veto, not a replay of unavailable launch04 full camera depth, continuous safety, or physical acceptance.',
    'source_commit': '014f01699800bb9da7b3127e2517047f55890bdf', 'module_sha256': digest(MODULE), 'geometry_sha256': geometry.sha256,
    'capture_receipt_sha256': digest(FRAMES / 'receipt.json'),
    'mask_receipt_sha256': digest(MASKS / 'receipt.json'),
    'script_sha256': digest(Path(__file__)),
    'uncertainty_m': .0001, 'actuator_calls': 0, 'rows': [],
}
for label, prefix in [('green cube', '1790818438'), ('orange', '1790818825')]:
    ep = next(EVIDENCE.glob(prefix + '*.json'))
    evidence = json.loads(ep.read_text())
    profiles = {}
    for phase in ('home', 'pregrasp', 'descent'):
        commands = [e for e in evidence['events'] if e['kind'] == 'isaac_joint_target_sent' and e['phase'] == phase]
        target = next(e['data'] for e in evidence['events'] if e['kind'] == 'move_target' and e['phase'] == phase)
        goal = np.asarray(target['q'])
        mix = min_jerk(1 / len(commands))
        start = (np.asarray(commands[0]['data']['q_local']) - mix * goal) / (1 - mix)
        waypoints = list(nominal_profile(start, goal, target['duration_s'], 30))
        assert len(waypoints) == len(commands)
        error = max(float(np.max(np.abs(w.q - np.asarray(c['data']['q_local'])))) for w, c in zip(waypoints, commands))
        assert error < 1e-12
        profiles[phase] = (start, goal, target['duration_s'], len(commands), error)
    for query in MASK_RECEIPT['queries']:
        if query['query'] != label or not query['ok']:
            continue
        metadata_path = FRAMES / query['input_metadata']
        assert digest(metadata_path) == query['input_sha256']
        assert digest(MASKS / query['npz']) == query['sha256']
        meta = json.loads(metadata_path.read_text())
        frame_path = metadata_path.with_suffix('.npz')
        assert digest(frame_path) == meta['npz_sha256']
        with np.load(frame_path, allow_pickle=False) as data:
            f = Frame(rgb=data['rgb'], depth_m=data['depth_m'], K=data['K'],
                      T_base_cam=data['T_base_cam'], robot_mask=data['robot_mask'],
                      t=meta['frame_t'], capture=meta['capture'], depth_source='sensor')
        with np.load(MASKS / query['npz'], allow_pickle=False) as data:
            mask = data['target_mask']
        # BridgeClient stores endpoint as a tuple; JSON serializes it as a
        # list. Restore only that documented container type for the replay.
        source = tuple(f.capture['source'])
        f.capture['source'] = source
        epoch = f.capture['proprioception']['producer_epoch']
        robot = f.capture['proprioception']['robot_id']
        raw = CAPTURE_RECEIPT['initial_state']
        # Match IsaacArm.get_state's local endpoint binding. The raw state
        # and these frames were saved by the same original BridgeClient.
        assert source == ('127.0.0.1', CAPTURE_RECEIPT['port'])
        state = RobotState(q=np.asarray(raw['q']) * cfg.arm.joint_signs,
                           physics_clock={**raw['physics_clock'], 'source': source},
                           gripper_joints=raw['gripper_joints'])
        before = time.perf_counter()
        scene = gate_module.ObservedScene(f, mask, f.T_base_cam, source=source, robot_id=robot, epoch=epoch)
        gate = gate_module.ObservedFingerGate(scene, geometry, kin, state, uncertainty_m=.0001)
        build_s = time.perf_counter() - before
        row = {'query': label, 'input_frame': metadata_path.name,
               'input_sha256': digest(metadata_path), 'attempt_sha256': digest(ep),
               'build_wall_s': build_s, 'observed_points': len(scene.points), 'phases': {}}
        for phase, (start, goal, duration, targets, error) in profiles.items():
            before = time.perf_counter()
            conflict = gate.profile(start, goal, duration, 30)
            row['phases'][phase] = {'wall_s': time.perf_counter() - before,
                                    'targets': targets, 'reconstruction_error_rad': error,
                                    'first_conflict': conflict}
        report['rows'].append(row)
        print(json.dumps(row), flush=True)
primary = [r for r in report['rows'] if '-cam0.' in r['input_frame']]
assert len(primary) == 4
assert all(r['phases']['descent']['first_conflict'] is not None for r in primary if r['query'] == 'orange')
assert all(r['phases']['descent']['first_conflict'] is None for r in primary if r['query'] == 'green cube')
report['primary_counterexample_checks_pass'] = True
out = BASE / 'native-evidence-04-analysis/observed-gate-replay05-014f016.json'
out.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
