#!/usr/bin/env python3
"""Opt-in Ground texture benchmark; no native execution in check-only mode."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from benchmark.rgbd import native_bridge as common
from benchmark.rgbd.ground_texture import (
    GROUND, GroundTextureBoard, MATERIAL, author_ground_texture, png_bytes,
    scene_bindings, texture_rectangles, validate_ground_texture,
)
from benchmark.rgbd.planar_reference import digest

TEXTURE = 'benchmark/rgbd/assets/ground_reference.png'
SOURCES = (*common.SOURCES, 'benchmark/rgbd/ground_texture.py',
           'benchmark/rgbd/ground_native_bridge.py', TEXTURE)
METADATA_KEY = 'cascade:rgbdGroundReference'


def texture_descriptor():
    """Metric content without a camera, depth map or path-derived label."""
    board = GroundTextureBoard()
    data = (REPO / TEXTURE).read_bytes()
    if data != png_bytes(board):
        raise ValueError('pinned Ground PNG differs from its metric descriptor')
    return {'board': board.description(), 'board_sha256': board.sha256,
            'texture_sha256': hashlib.sha256(data).hexdigest(),
            'rectangles_sha256': digest(texture_rectangles(board)),
            'texture_source': TEXTURE, 'texture_bytes': len(data)}


def board_from_fixture(fixture):
    """Reject a producer/caller mismatch rather than infer another board."""
    expected = texture_descriptor()
    if any(fixture.get(key) != value for key, value in expected.items()):
        raise ValueError('producer Ground metric descriptor does not match the consumer')
    return GroundTextureBoard()


def extend_admission(admission, reference):
    descriptor = texture_descriptor()
    admission['source_sha256'].update({p: common.sha(REPO / p) for p in SOURCES})
    admission['planar_reference_identity'] = reference
    admission['ground_reference_descriptor'] = descriptor
    return admission


def author_checked_ground(stage, expected):
    if expected != texture_descriptor():
        raise ValueError('Ground PNG/descriptor changed after admission')
    board = board_from_fixture(expected)
    fixture = author_ground_texture(stage, REPO / TEXTURE, board=board)
    # String JSON avoids Vt/container coercion. The consumed USD scene binds
    # descriptor and PNG content, in addition to shader connections and ST.
    stage.GetPrimAtPath(MATERIAL).SetCustomDataByKey(
        METADATA_KEY, common.canonical(expected).decode('ascii'))
    fixture.update(scene_bindings(stage))
    fixture.update(expected)
    validate_ground_texture(stage, REPO / TEXTURE, board=board, receipt=fixture)
    return fixture


def ground_snapshot(stage):
    """Actual surface/appearance, separate from later camera/Render authoring.

    Native physics is still compared in full. This smaller USD checkpoint only
    excludes unrelated prims that the ordinary camera/bootstrap must create.
    Ground/material values, connections, bindings and time samples remain exact.
    """
    from pxr import Usd, UsdGeom

    snapshot = common.stage_snapshot(stage, exclude_path=None)
    prims = snapshot['prims']
    required = (GROUND, '/World/GroundMaterial', MATERIAL)
    if any(path not in prims for path in required):
        raise ValueError('Ground surface/material disappeared during bootstrap')
    selected = {path: row for path, row in prims.items()
                if path in required or path.startswith(MATERIAL + '/')}
    ancestors = {}
    prim = stage.GetPrimAtPath(GROUND).GetParent()
    while prim and not prim.IsPseudoRoot():
        # Relationship strengths/collections and metadata can override the
        # Ground's material without changing its own prim or transform. Child
        # prim additions are not properties in this composed prim snapshot.
        ancestors[str(prim.GetPath())] = prims[str(prim.GetPath())]
        prim = prim.GetParent()
    return {'surface_prims': selected, 'ancestors': ancestors,
            'world_from_ground': common.usd_value(UsdGeom.Xformable(
                stage.GetPrimAtPath(GROUND)).ComputeLocalToWorldTransform(Usd.TimeCode.Default())),
            'meters_per_unit': UsdGeom.GetStageMetersPerUnit(stage),
            'up_axis': str(UsdGeom.GetStageUpAxis(stage))}


def reference_backend(base, *, author=None, descriptor=None):
    # Explicit benchmark variant dependencies; no global recipe substitution.
    author = author_checked_ground if author is None else author
    descriptor = texture_descriptor if descriptor is None else descriptor
    class GroundReferenceBackend(base):
        def _create_camera(self, stage):
            self._checkpoint()
            self.receipt['ground_reference_fixture'] = author(
                stage, self.admission['ground_reference_descriptor'])
            self._ground_stage = stage
            self._ground_authored = ground_snapshot(stage)
            self._checkpoint()
            result = super()._create_camera(stage)
            self._checkpoint()
            self._check_ground('after_camera')
            return result

        def _check_ground(self, phase):
            observed = ground_snapshot(self._ground_stage)
            differences = [key for key, value in self._ground_authored.items()
                           if observed.get(key) != value]
            check = {'passed': not differences, 'phase': phase,
                     'authored_sha256': digest(self._ground_authored),
                     'observed_sha256': digest(observed), 'differences': differences,
                     'physical_admission': False}
            self.receipt.setdefault('ground_appearance_checks', {})[phase] = check
            (self.args.out / ('ground-appearance-' + phase + '.json')).write_bytes(
                common.canonical(check) + b'\n')
            if differences:
                raise ValueError('Ground appearance/geometry changed during bootstrap: '
                                 + ', '.join(differences))

        def open(self):
            super().open()
            self._checkpoint()
            self._check_ground('after_bootstrap')
            # Check source bytes again after bootstrap; the ordinary identity
            # builder will independently rehash SOURCES, including the PNG.
            if descriptor() != self.admission['ground_reference_descriptor']:
                raise ValueError('Ground PNG/descriptor changed during bootstrap')
            check = common.native_invariance(self.receipt, self.admission['planar_reference_identity'])
            self.receipt['planar_physics_invariance'] = check
            (self.args.out / 'planar-physics-invariance.json').write_bytes(common.canonical(check) + b'\n')
            if not check['passed']:
                raise ValueError('native physics/calibration recipe changed: ' + ', '.join(check['differences']))
    return GroundReferenceBackend


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--reference-identity', type=Path, required=True)
    parser.add_argument('--reference-identity-sha256', required=True)
    parser.add_argument('--reference-model-sha256', required=True)
    reference_args, bridge_argv = parser.parse_known_args(argv)
    bridge = common.bind_bridge()
    args = bridge.parse_args(bridge_argv)
    try:
        if not args.camera_rgbd:
            raise ValueError('Ground benchmark requires explicit camera-rgbd')
        reference = common.reference_identity(reference_args.reference_identity,
            reference_args.reference_identity_sha256, reference_args.reference_model_sha256)
        admission = extend_admission(bridge.admit(args), reference)
        if args.check_only:
            print(json.dumps({'ok': True, 'physical_acceptance': False,
                'reference_model_identity_sha256': reference['model_identity_sha256'],
                'source_sha256': admission['source_sha256'],
                'ground_reference_descriptor': admission['ground_reference_descriptor'],
                'no_native_run': True}))
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
