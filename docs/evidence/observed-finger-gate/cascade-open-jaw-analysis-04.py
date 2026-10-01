from pathlib import Path
import hashlib
import json
import numpy as np

BASE = Path('/home/johnny/Projects/demo/cascade-lab/CODEX_NEWTON')
ROOT = BASE / 'native-evidence-04-20261001'
PROOF = ROOT / 'runs/.launch/profile-cascade-demo/cascade-proof-deb2b1349ac4441bbd58d89a69148616'
EVIDENCE = ROOT / 'runs/native-acceptance-20261001/grasp-evidence-04'
OUT = BASE / 'native-evidence-04-analysis/open-jaw-tracking.json'
CASES = [('green', 1, '1790818438140538464-610516-fa6fc2a00d6d464b86c256879ad42f5a.json'),
         ('orange', 2, '1790818825075088880-610516-d41f14baed4042529174daeb0546ebc9.json')]

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

result = {'scope': 'Individual physical finger feedback while the command is open, bound to original phase timestamps; numerical/tracking calibration only. No object truth supplied to planning. Observed extrema are not a universal bound.',
          'script_sha256': digest(Path(__file__)), 'cases': []}
for name, number, evidence_name in CASES:
    source = EVIDENCE / evidence_name
    evidence = json.loads(source.read_text())
    witness_path = PROOF / f'case-{number}/physics/samples.jsonl'
    witness = [json.loads(line) for line in witness_path.read_text().splitlines()]
    start = evidence['started_monotonic_s']
    phase_starts = {e['phase']: start + e['elapsed_s'] for e in evidence['events'] if e['kind'] == 'phase'}
    case = {'object': name, 'attempt': evidence['attempt_id'],
            'witness_sha256': digest(witness_path), 'attempt_sha256': digest(source), 'phases': {}}
    for phase, after in [('home', 'pregrasp'), ('pregrasp', 'descent'), ('descent', 'close')]:
        selected = [s for s in witness if phase_starts[phase] <= s['physics']['server_monotonic'] < phase_starts[after]]
        assert selected
        q = np.asarray([s['physics']['gripper']['q'] for s in selected])
        hi = np.asarray([s['physics']['gripper']['upper'] for s in selected])
        contact = np.asarray([s['physics']['contacts']['jaw_contact_counts'] for s in selected])
        assert np.isfinite(q).all() and q.shape[1] == 2
        row = {'samples': len(selected), 'first_sequence': selected[0]['sequence'],
               'last_sequence': selected[-1]['sequence'],
               'minimum_position_m': q.min(axis=0).tolist(), 'maximum_position_m': q.max(axis=0).tolist(),
               'position_range_um': (np.ptp(q, axis=0) * 1e6).tolist(),
               'minimum_error_from_open_um': ((q - hi).min(axis=0) * 1e6).tolist(),
               'maximum_error_from_open_um': ((q - hi).max(axis=0) * 1e6).tolist(),
               'max_absolute_error_from_open_um': (np.abs(q - hi).max(axis=0) * 1e6).tolist(),
               'samples_with_contact_on_target': int(np.any(contact > 0, axis=1).sum()),
               'target_contact_scope': 'These contact sensors observe the requested target only; neighbor contact absence is not established.'}
        no_contact = ~np.any(contact > 0, axis=1)
        if no_contact.any():
            row['no_target_contact_max_absolute_error_um'] = (np.abs(q[no_contact] - hi[no_contact]).max(axis=0) * 1e6).tolist()
        case['phases'][phase] = row
    result['cases'].append(case)
OUT.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
print(json.dumps(result, indent=2))
