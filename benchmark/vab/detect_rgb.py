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


def create_detector(weights, expected_sha256, device, *, base_url, model, api_key_file=None):
    """The brain detector (the brain names and boxes, SAM masks) with the
    admitted SAM checkpoint. Admission is repeated after archive validation,
    at the actual loading boundary, as well as before and after the complete
    analysis. The brain's answer cache is off (`scene_change=-1`: every frame
    is a new scene), so each archived view gets its own brain answer and a
    failed brain call yields [] for that view, never another view's boxes."""
    if not weights.is_absolute() or weights_digest(weights) != expected_sha256:
        raise ValueError('exact local SAM checkpoint required')
    from cascade.perception.brain_detector import BrainDetector
    from cascade.perception.segmenter import BoxSegmenter
    return BrainDetector(base_url=base_url, model=model, api_key_file=api_key_file,
                         segmenter=BoxSegmenter(model_path=str(weights.resolve()), device=device),
                         refresh_s=0.0, scene_change=-1.0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('output', type=Path)
    from cascade.apps.demo import BRAIN_BASE_URL, BRAIN_KEY_FILE, BRAIN_MODEL
    parser.add_argument('--weights', type=Path, required=True, help='absolute SAM checkpoint (sam2.1_l.pt)')
    parser.add_argument('--weights-sha256', required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda:0'], default='cpu')
    parser.add_argument('--base-url', default=BRAIN_BASE_URL, help='brain endpoint (OpenAI-compatible)')
    parser.add_argument('--model', default=BRAIN_MODEL, help='served brain model')
    parser.add_argument('--api-key-file', default=BRAIN_KEY_FILE, help='file holding the brain api key')
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('new detection output required')
    if not args.weights.is_absolute() or weights_digest(args.weights) != args.weights_sha256:
        raise ValueError('exact local SAM checkpoint required')
    import cv2
    versions = {name: importlib.metadata.version(name) for name in ('ultralytics', 'torch', 'numpy', 'openai')}
    versions['cv2'] = cv2.__version__
    def factory():
        return create_detector(args.weights, args.weights_sha256, args.device, base_url=args.base_url,
                               model=args.model, api_key_file=args.api_key_file)
    result = detect_archive(args.directory, factory)
    if weights_digest(args.weights) != args.weights_sha256:
        raise ValueError('detector checkpoint changed during analysis')
    result['detector_recipe'] = {'weights_sha256': args.weights_sha256, 'device': args.device,
        'brain_base_url': args.base_url, 'brain_model': args.model,
        'vocabulary': 'brain inventory (open world)', 'versions': versions}
    with args.output.open('x') as stream:
        stream.write(json.dumps(result, allow_nan=False, indent=2)+'\n')


if __name__ == '__main__':
    main()
