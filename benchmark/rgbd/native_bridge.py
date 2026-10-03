#!/usr/bin/env python3
"""Bounded planar-reference bridge; check-only never imports or opens Kit.

The visual fixture is authored before the ordinary camera/export/play sequence.
This benchmark does not admit locomotion or replace the production bridge.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
SOURCES = (
    'benchmark/rgbd/__init__.py',
    'benchmark/rgbd/planar_reference.py',
    'benchmark/rgbd/native_bridge.py',
    'benchmark/rgbd/live_reference.py',
)
BOARD_PATH = '/World/RgbdMetricBoard'


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      allow_nan=False, ensure_ascii=True).encode('ascii')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bind_bridge():
    sys.path[:0] = [str(REPO), str(REPO / 'scripts'), str(REPO / 'src')]
    import isaac_microduck_bridge as bridge
    if Path(bridge.__file__).resolve() != REPO / 'scripts/isaac_microduck_bridge.py':
        raise RuntimeError('foreign bridge module')
    bridge.bind_repo()
    return bridge


def reference_identity(path, file_sha256, model_sha256):
    """Pin the actual prior document AND its canonical recipe, never a label."""
    from cascade.sim.microduck_newton import strict_json
    if sha(path) != file_sha256:
        raise ValueError('reference identity file SHA mismatch')
    value = strict_json(Path(path).read_bytes())
    if (value['model_identity_sha256'] != model_sha256
            or hashlib.sha256(canonical(value['recipe'])).hexdigest() != model_sha256):
        raise ValueError('reference model identity mismatch')
    if not value['recipe']['native'] or not value['recipe']['bam']:
        raise ValueError('missing reference native recipe')
    return value


def extend_admission(admission, reference):
    """Canonical model builder rehashes these files after backend open."""
    admission['source_sha256'].update({p: sha(REPO / p) for p in SOURCES})
    admission['planar_reference_identity'] = reference
    return admission


def usd_value(value):
    """Stable content values, not Python addresses of freshly wrapped USD ops."""
    if type(value) is float and not math.isfinite(value):
        # USD's unlimited joint bounds legitimately contain infinity. Preserve
        # it as typed diagnostic content, never emit nonstandard JSON numbers.
        return {'type': 'float', 'nonfinite': str(value)}
    if value is None or type(value) in (str, bool, float, int):
        return value
    if isinstance(value, dict):
        return {key: usd_value(item) for key, item in value.items()}
    if hasattr(value, 'isExplicit') and hasattr(value, 'explicitItems'):
        return {'type': type(value).__name__, 'isExplicit': value.isExplicit,
                **{key: usd_value(list(getattr(value, key))) for key in
                   ('explicitItems', 'prependedItems', 'appendedItems',
                    'addedItems', 'deletedItems', 'orderedItems')}}
    if hasattr(value, '__iter__'):
        return {'type': type(value).__name__, 'items': [usd_value(item) for item in value]}
    text = str(value)
    if ' object at 0x' in text:
        raise ValueError('USD snapshot cannot serialize object content: ' + type(value).__name__)
    return {'type': type(value).__name__, 'text': text,
            **({'resolvedPath': value.resolvedPath} if hasattr(value, 'resolvedPath') else {})}


def stage_snapshot(stage, *, exclude_path=BOARD_PATH):
    """Composed values/metadata, connections, targets and every time sample.

    Only the new subtree is excluded. An existing similarly named prim is not.
    No secondary live stage is opened in Kit.
    """
    rows = {}
    for prim in stage.TraverseAll():
        path = str(prim.GetPath())
        if exclude_path is not None and (path == exclude_path or path.startswith(exclude_path + '/')):
            continue
        rows[path] = {
            'type': prim.GetTypeName(), 'active': prim.IsActive(),
            'defined': prim.IsDefined(), 'schemas': list(prim.GetAppliedSchemas()),
            'metadata': usd_value(prim.GetAllMetadata()),
            'attributes': {a.GetName(): {
                'default': usd_value(a.Get()), 'metadata': usd_value(a.GetAllMetadata()),
                'connections': list(map(str, a.GetConnections())),
                'samples': [[t, usd_value(a.Get(t))] for t in a.GetTimeSamples()],
            } for a in prim.GetAttributes()},
            'relationships': {r.GetName(): {
                'targets': list(map(str, r.GetTargets())),
                'metadata': usd_value(r.GetAllMetadata()),
            } for r in prim.GetRelationships()},
        }
    return {'stage_metadata': usd_value(stage.GetPseudoRoot().GetAllMetadata()), 'prims': rows}


def author_checked_board(stage):
    from benchmark.rgbd.planar_reference import author_visual_board, digest, visual_quads
    if stage.GetPrimAtPath(BOARD_PATH):
        raise ValueError('reference board already exists')
    before = stage_snapshot(stage)
    fixture = author_visual_board(stage, path=BOARD_PATH)
    after = stage_snapshot(stage)
    if before != after:
        raise ValueError('visual authoring changed existing stage state')
    from pxr import UsdGeom
    prims = [p for p in stage.TraverseAll() if str(p.GetPath()) == BOARD_PATH
             or str(p.GetPath()).startswith(BOARD_PATH + '/')]
    if any(any(word in schema.lower() for word in ('physics', 'collision', 'drive', 'articulation'))
           for p in prims for schema in p.GetAppliedSchemas()):
        raise ValueError('reference fixture contains a physics schema')
    meshes = [p for p in prims if p.IsA(UsdGeom.Mesh)]
    quads = visual_quads()
    if len(meshes) != len(quads):
        raise ValueError('reference visual geometry count mismatch')
    for prim, quad in zip(meshes, quads):
        points = UsdGeom.Mesh(prim).GetPointsAttr().Get()
        if len(points) != 4 or any(abs(float(x) - y) > 1e-7
                                  for p, q in zip(points, quad['points_m']) for x, y in zip(p, q)):
            raise ValueError('reference authored points mismatch')
    return {**fixture, 'existing_prims': len(before['prims']),
            'existing_stage_sha256': digest(before), 'existing_stage_unchanged': True,
            'visual_meshes': len(meshes), 'no_added_physics_schemas': True}


def native_invariance(native, reference):
    """Exact native values, including ALL shape labels. No geometry filtering."""
    expected = reference['recipe']
    differences = [f'native.{key}' for key, value in expected['native'].items()
                   if key not in native or canonical(native[key]) != canonical(value)]
    actual_bam = native.get('bam', {})
    differences += [f'bam.{key}' for key, value in expected['bam'].items()
                    if key not in actual_bam or canonical(actual_bam[key]) != canonical(value)]
    return {'passed': not differences, 'differences': differences,
            'reference_model_identity_sha256': reference['model_identity_sha256'],
            'checked_native_fields': sorted(expected['native']),
            'checked_bam_fields': sorted(expected['bam']),
            'physical_admission': False}


def reference_backend(base):
    """Explicit dependency seam; the parent remains the sole physics owner."""
    class ReferenceBackend(base):
        def _create_camera(self, stage):
            self._checkpoint()
            self.receipt['planar_fixture'] = author_checked_board(stage)
            self._checkpoint()
            return super()._create_camera(stage)

        def open(self):
            super().open()
            self._checkpoint()
            check = native_invariance(self.receipt, self.admission['planar_reference_identity'])
            self.receipt['planar_physics_invariance'] = check
            # Preserve the diagnostic even when no model/cache/listener is admitted.
            (self.args.out / 'planar-physics-invariance.json').write_bytes(canonical(check) + b'\n')
            if not check['passed']:
                raise ValueError('native physics/calibration recipe changed: ' + ', '.join(check['differences']))
    return ReferenceBackend


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--reference-identity', type=Path, required=True)
    parser.add_argument('--reference-identity-sha256', required=True)
    parser.add_argument('--reference-model-sha256', required=True)
    reference_args, bridge_argv = parser.parse_known_args(argv)
    bridge = bind_bridge()
    args = bridge.parse_args(bridge_argv)
    try:
        if not args.camera_rgbd:
            raise ValueError('planar benchmark requires explicit camera-rgbd')
        reference = reference_identity(reference_args.reference_identity,
            reference_args.reference_identity_sha256, reference_args.reference_model_sha256)
        admission = extend_admission(bridge.admit(args), reference)
        if args.check_only:
            print(json.dumps({'ok': True, 'physical_acceptance': False,
                'reference_model_identity_sha256': reference['model_identity_sha256'],
                'source_sha256': admission['source_sha256'], 'no_native_run': True}))
            return 0
        from cascade.apps.signal_stop import SignalRequest, StopSignals
        from cascade.sim.microduck_newton import KitNewtonBackend
        with StopSignals(protect_registration=True) as signals:
            try:
                result = bridge.run(args, admission, signals=signals,
                                    backend_factory=reference_backend(KitNewtonBackend))
            except SignalRequest as exc:
                return 128 + exc.signum
        print(json.dumps({k: result[k] for k in ('completed', 'physical_acceptance', 'steps',
                                                'policy_evaluations', 'teardown_errors')}))
        return result['exit_code']
    except (ValueError, OSError, RuntimeError, ImportError, KeyError, TypeError) as exc:
        print(json.dumps({'ok': False, 'error': f'{type(exc).__name__}: {exc}', 'physical_acceptance': False}))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
