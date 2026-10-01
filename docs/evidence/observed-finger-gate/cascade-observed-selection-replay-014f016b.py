"""Independent offline replay of historical joint paths on new observed scenes."""
import ast
from types import SimpleNamespace
import subprocess
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
from cascade.types import Frame, RobotState, Grasp, SafetyViolation
from cascade.safety.harness import SafetyHarness, SafetyLimits
from cascade.grasping.selector import select_grasp

SNAPSHOT = Path('/tmp/cascade-observed-gate-snapshot-014f016')
MODULE = SNAPSHOT / 'src/cascade/grasping/observed_scene.py'
spec = importlib.util.spec_from_file_location('cascade.grasping.observed_scene', MODULE)
gate_module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = gate_module
spec.loader.exec_module(gate_module)
gate_module.ROOT = REPO  # Verify immutable collision source hashes against materialized URDF and collision STL assets.
geometry = gate_module.FingerGeometry()
PIN = '014f01699800bb9da7b3127e2517047f55890bdf'
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip() == PIN
runtime_path = REPO / 'src/cascade/skills/runtime.py'
runtime_tree = ast.parse(runtime_path.read_text())
skill = next(n for n in ast.walk(runtime_tree) if isinstance(n, ast.FunctionDef) and n.name == 'skill_grasp_object')
names = ['_scene_cancel', '_scene_duration', '_scene_path', '_vet']
functions = [n for n in skill.body if isinstance(n, ast.FunctionDef) and n.name in names]
assert [n.name for n in functions] == names
vet_code = compile(ast.Module(body=functions, type_ignores=[]), str(runtime_path), 'exec')
cfg = load_demo_config(camera='isaac', arm='isaac_kitchen_gpu', llm='mock')
kin = Kinematics(cfg.arm.model, cfg.arm.ee_frame, 6, cfg.arm.joint_signs)
FRAMES, MASKS = BASE / 'passive-scene-05', BASE / 'passive-masks-05b'
CAPTURE_RECEIPT = json.loads((FRAMES / 'receipt.json').read_text())
MASK_RECEIPT = json.loads((MASKS / 'receipt.json').read_text())
EVIDENCE = BASE / 'native-evidence-04-analysis/grasp-evidence-04'

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

report = {
    'scope': 'Counterfactual selection: unchanged historical launch04 floor-adjusted candidates evaluated against NEW passive05 depth and production YOLOE masks with actual select_grasp, exact AST-extracted runtime vet functions and production kinematics/harness. No live Runtime, robot transport or planner calls. Discrete observed finger surfaces only; not physical acceptance.',
    'runtime_sha256': digest(runtime_path), 'vet_ast_functions': names,
    'source_commit': '014f01699800bb9da7b3127e2517047f55890bdf', 'module_sha256': digest(MODULE), 'geometry_sha256': geometry.sha256,
    'capture_receipt_sha256': digest(FRAMES / 'receipt.json'),
    'mask_receipt_sha256': digest(MASKS / 'receipt.json'),
    'script_sha256': digest(Path(__file__)),
    'uncertainty_m': .0001, 'actuator_calls': 0, 'rows': [],
}
for label, prefix in [('green cube', '1790818438'), ('orange', '1790818825')]:
    ep = next(EVIDENCE.glob(prefix + '*.json'))
    evidence = json.loads(ep.read_text())
    candidate_rows = next(e['data']['grasps'] for e in evidence['events'] if e['kind'] == 'floor_adjusted_candidates')
    candidates = [Grasp(**{k: np.asarray(v) if k in ('position', 'rotation', 'approach') else v for k, v in r.items()}) for r in candidate_rows]
    historical_selected = next(e['data'] for e in evidence['events'] if e['kind'] == 'selected')
    seed = np.asarray(next(e['data']['seed_q'] for e in evidence['events'] if e['kind'] == 'selection_input'))
    assert np.array_equal(seed, cfg.arm.home_q)
    harness = SafetyHarness(SafetyLimits.from_config(cfg.safety), kinematics=kin)
    events = []
    def event(name, **kw):
        kw = json.loads(json.dumps(kw, default=lambda value: value.tolist() if isinstance(value, np.ndarray) else value.item()))
        events.append({'kind': name, **kw})
    environment = {'__package__': 'cascade.skills', 'np': np, 'SafetyViolation': SafetyViolation,
                   'harness': harness, 'scene_halt_generation': harness._halt_generation,
                   'scene_gate': None, '_seed': seed, 'gcfg': cfg.grasp,
                   'exempt_r': float(cfg.grasp.exempt_radius_m), 'grasp_evidence': SimpleNamespace(event=event),
                   'self': SimpleNamespace(kin=kin, arm=SimpleNamespace(raw=SimpleNamespace(motion_rate_hz=30.)))}
    exec(vet_code, environment)
    def select():
        return select_grasp(candidates, kin, seed, max_width_m=float(cfg.arm.gripper.max_width_m),
                            pregrasp_offset_m=float(cfg.grasp.pregrasp_offset_m), validate=environment['_vet'],
                            jaw_fixed_tip_m=cfg.arm.gripper.get('jaw_fixed_tip_m'), jaw_close_dir=cfg.arm.gripper.get('jaw_close_dir'))
    baseline = select()
    baseline_error = max(float(np.max(np.abs(baseline[i]-historical_selected[key]))) for i,key in [(1,'q_pre'),(2,'q_grasp')])
    assert baseline_error < 1e-12, baseline_error
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
        environment['scene_gate'] = gate
        events.clear()
        before = time.perf_counter()
        try:
            chosen, q_pre, q_grasp = select()
            selected_idx = next(i for i,g in enumerate(candidates) if np.array_equal(g.position, chosen.position) and g.quality == chosen.quality)
            row['selection'] = {'ok': True, 'candidate_index': selected_idx, 'quality': chosen.quality,
                                'position': chosen.position.tolist(), 'rotation': chosen.rotation.tolist(),
                                'q_pre': q_pre.tolist(), 'q_grasp': q_grasp.tolist()}
        except Exception as exc:
            row['selection'] = {'ok': False, 'error_type': type(exc).__name__, 'error': str(exc)}
        row['selection'].update({'wall_s': time.perf_counter()-before, 'vet_rejection_events': events.copy(),
                                 'baseline_historical_q_max_error_rad': baseline_error,
                                 'candidate_count': len(candidates)})
        report['rows'].append(row)
        print(json.dumps(row), flush=True)
primary = [r for r in report['rows'] if '-cam0.' in r['input_frame']]
assert len(primary) == 4
assert all(r['phases']['descent']['first_conflict'] is not None for r in primary if r['query'] == 'orange')
assert all(r['phases']['descent']['first_conflict'] is None for r in primary if r['query'] == 'green cube')
report['primary_counterexample_checks_pass'] = True
report['all_selections_found_alternative'] = all(row['selection']['ok'] for row in report['rows'])
out = BASE / 'native-evidence-04-analysis/observed-selection-replay05b-014f016.json'
out.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
