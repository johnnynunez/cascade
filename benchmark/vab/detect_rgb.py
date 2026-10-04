"""Detect predicted 2D regions in a closed public RGB archive; no simulator."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

from benchmark.vab.reconstruct_rgb import read_archive, verify_archive
from cascade.perception.rgb_detection import detect_rgb_view


def detect_archive(directory, detector_factory):
    """Validate every capture before constructing a detector, then analyze once."""
    captures, artifacts = read_archive(directory)
    detector = detector_factory()
    results = []
    for capture in captures:
        if capture['status'] == 'unverified':
            results.append(dict(capture))
            continue
        results.append({'file': capture['file'], 'status': 'analyzed',
                        'views': [detect_rgb_view(view, detector) for view in capture['views']]})
    verify_archive(directory, artifacts)
    return {'schema': 'cascade.vab-offline-rgb-detections.v1', 'captures': results,
            'artifact_sha256': artifacts, 'physics_steps_advanced': 0,
            'detector_instances_constructed': 1, 'simulator_models_constructed': 0,
            'object_identity_verified': False, 'position_error_m': None,
            'complete_geometry_verified': False, 'physical_admission': False}


def weights_digest(path):
    if path.is_symlink() or not path.is_file() or not 1024**2 <= path.stat().st_size <= 512*1024**2:
        raise ValueError('local bounded detector checkpoint required')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def create_detector(weights, expected_sha256, device):
    # Admission is repeated after archive validation, at the actual loading
    # boundary, as well as before and after the complete analysis.
    if (not weights.is_absolute() or not weights.name.endswith('-pf.pt')
            or weights_digest(weights) != expected_sha256):
        raise ValueError('exact local prompt-free checkpoint required')
    from cascade.perception.detector import OpenVocabDetector
    return OpenVocabDetector(str(weights.resolve()), device=device, conf=.25, prompt_free=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--weights-sha256', required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda:0'], default='cpu')
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('new detection output required')
    if (not args.weights.is_absolute() or not args.weights.name.endswith('-pf.pt')
            or weights_digest(args.weights) != args.weights_sha256):
        raise ValueError('exact local prompt-free checkpoint required')
    import cv2
    versions = {name: importlib.metadata.version(name) for name in ('ultralytics', 'torch', 'numpy')}
    versions['cv2'] = cv2.__version__
    def factory():
        return create_detector(args.weights, args.weights_sha256, args.device)
    result = detect_archive(args.directory, factory)
    if weights_digest(args.weights) != args.weights_sha256:
        raise ValueError('detector checkpoint changed during analysis')
    result['detector_recipe'] = {'weights_sha256': args.weights_sha256, 'device': args.device,
        'minimum_score': .25, 'vocabulary': 'checkpoint prompt-free',
        'versions': versions}
    with args.output.open('x') as stream:
        stream.write(json.dumps(result, allow_nan=False, indent=2)+'\n')


if __name__ == '__main__':
    main()
