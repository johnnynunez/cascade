#!/usr/bin/env python3
"""Explicit binary Ground authoring with an authenticated consumer contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from benchmark.rgbd import ground_native_bridge as ground
from benchmark.rgbd import native_bridge as common
from benchmark.rgbd.binary_layout_a import BinaryLayoutABoard
from benchmark.rgbd.binary_reference import canonical_detector_json
from benchmark.rgbd.checker_accuracy import AccuracyBoard, canonical_consumer_json
from benchmark.rgbd.ground_texture import (
    MATERIAL,
    author_ground_texture,
    png_bytes,
    scene_bindings,
    texture_rectangles,
    validate_ground_texture,
)
from benchmark.rgbd.planar_reference import digest

TEXTURE = "benchmark/rgbd/assets/ground_binary_layout_a.png"
CONSUMER_SOURCE = "benchmark/rgbd/assets/binary_layout_a_consumer.json"
ACCURACY_CONSUMER_SOURCE = "benchmark/rgbd/assets/checker_accuracy_consumer.json"
SOURCES = (
    *ground.SOURCES,
    "benchmark/rgbd/binary_reference.py",
    "benchmark/rgbd/binary_layout_a.py",
    "benchmark/rgbd/homography_support.py",
    "benchmark/rgbd/layout_a_native_bridge.py",
    "benchmark/rgbd/checker_accuracy.py",
    TEXTURE,
    CONSUMER_SOURCE,
    "benchmark/rgbd/observation_health.py",
    "src/cascade/agent/base_effects.py",
)
GENERATOR_METADATA = "cascade:rgbdBinaryLayoutAAuthoringEnvironment"


def source_inputs(accuracy=False):
    return (*SOURCES, ACCURACY_CONSUMER_SOURCE) if accuracy else SOURCES


def load_consumer(path, expected_sha256, *, accuracy=False):
    path = Path(path)
    source = ACCURACY_CONSUMER_SOURCE if accuracy else CONSUMER_SOURCE
    if (
        path.resolve() != (REPO / source).resolve()
        or path.is_symlink()
        or path.stat().st_size > 32768
        or common.sha(path) != expected_sha256
    ):
        raise ValueError("consumer detector source/hash mismatch")
    validate = canonical_consumer_json if accuracy else canonical_detector_json
    return validate(path.read_text())


def declared_board(consumer_json, *, accuracy=False):
    return (AccuracyBoard(consumer_json) if accuracy
            else BinaryLayoutABoard(consumer_detector_json=consumer_json))


def texture_descriptor(consumer_json, *, accuracy=False):
    board = declared_board(consumer_json, accuracy=accuracy)
    data = (REPO / TEXTURE).read_bytes()
    if data != png_bytes(board):
        raise ValueError("layout A PNG differs from frozen codebook/geometry")
    return {
        "board": board.description(),
        "board_sha256": board.sha256,
        "texture_sha256": hashlib.sha256(data).hexdigest(),
        "rectangles_sha256": digest(texture_rectangles(board)),
        "texture_source": TEXTURE,
        "texture_bytes": len(data),
        "consumer_detector_sha256": hashlib.sha256(consumer_json.encode()).hexdigest(),
    }


def board_from_fixture(fixture, consumer_json, *, accuracy=False):
    expected = texture_descriptor(consumer_json, accuracy=accuracy)
    if any(fixture.get(k) != v for k, v in expected.items()):
        raise ValueError(
            "producer layout A fixture does not match consumer declaration"
        )
    board = declared_board(consumer_json, accuracy=accuracy)
    board.require_consumer_implementation()
    return board


def generator_record():
    import cv2
    from benchmark.rgbd.binary_reference import detector_recipe

    root = Path(cv2.__file__).parent
    return {
        "role": "bitmap_authoring_only_not_consumer_detection",
        "opencv_version": cv2.__version__,
        "dictionary_bytes_sha256": detector_recipe()["dictionary_bytes_sha256"],
        "module_content_sha256": {
            str(p.relative_to(root)): common.sha(p)
            for p in sorted(root.rglob("*"))
            if p.is_file()
            and (p.suffix == ".py" or ".so" in p.name or ".pyd" in p.name)
        },
    }


def author_checked_layout(stage, expected, consumer_json, *, accuracy=False):
    if expected != texture_descriptor(consumer_json, accuracy=accuracy):
        raise ValueError("layout A descriptor changed after admission")
    board = declared_board(consumer_json, accuracy=accuracy)
    fixture = author_ground_texture(stage, REPO / TEXTURE, board=board)
    generator = generator_record()
    material = stage.GetPrimAtPath(MATERIAL)
    material.SetCustomDataByKey(
        ground.METADATA_KEY, common.canonical(expected).decode("ascii")
    )
    material.SetCustomDataByKey(
        GENERATOR_METADATA, common.canonical(generator).decode("ascii")
    )
    fixture.update(scene_bindings(stage))
    fixture.update(expected)
    fixture["generator"] = generator
    validate_ground_texture(stage, REPO / TEXTURE, board=board, receipt=fixture)
    return fixture


def extend_admission(admission, reference, consumer_json, *, accuracy=False):
    admission["source_sha256"].update({p: common.sha(REPO / p) for p in source_inputs(accuracy)})
    admission["planar_reference_identity"] = reference
    admission["ground_reference_descriptor"] = texture_descriptor(consumer_json, accuracy=accuracy)
    return admission


def reference_backend(base, consumer_path, consumer_sha256, *, accuracy=False):
    # Every recheck reads the admitted bytes. No monkeypatch or mutable global
    # detector override; the producer never claims to have run the consumer.
    def declaration():
        return load_consumer(consumer_path, consumer_sha256, accuracy=accuracy)

    return ground.reference_backend(
        base,
        author=lambda stage, expected: author_checked_layout(
            stage, expected, declaration(), accuracy=accuracy
        ),
        descriptor=lambda: texture_descriptor(declaration(), accuracy=accuracy),
    )


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--reference-identity", type=Path, required=True)
    parser.add_argument("--reference-identity-sha256", required=True)
    parser.add_argument("--reference-model-sha256", required=True)
    parser.add_argument("--consumer-detector", type=Path, required=True)
    parser.add_argument("--consumer-detector-sha256", required=True)
    parser.add_argument("--checker-accuracy", action="store_true",
                        help="Explicit schema5 consumer; never reinterpret schema4 captures")
    ref, bridge_argv = parser.parse_known_args(argv)
    bridge = common.bind_bridge()
    args = bridge.parse_args(bridge_argv)
    try:
        if not args.camera_rgbd:
            raise ValueError("layout A benchmark requires explicit camera-rgbd")
        consumer = load_consumer(ref.consumer_detector, ref.consumer_detector_sha256,
                                 accuracy=ref.checker_accuracy)
        reference = common.reference_identity(
            ref.reference_identity,
            ref.reference_identity_sha256,
            ref.reference_model_sha256,
        )
        admission = extend_admission(bridge.admit(args), reference, consumer,
                                     accuracy=ref.checker_accuracy)
        if args.check_only:
            print(
                json.dumps(
                    {
                        "ok": True,
                        "physical_acceptance": False,
                        "reference_model_identity_sha256": reference[
                            "model_identity_sha256"
                        ],
                        "source_sha256": admission["source_sha256"],
                        "ground_reference_descriptor": admission[
                            "ground_reference_descriptor"
                        ],
                        "generator": generator_record(),
                        "no_native_run": True,
                    }
                )
            )
            return 0
        from cascade.apps.signal_stop import SignalRequest, StopSignals
        from cascade.sim.microduck_newton import KitNewtonBackend

        with StopSignals(protect_registration=True) as signals:
            try:
                result = bridge.run(
                    args,
                    admission,
                    signals=signals,
                    backend_factory=reference_backend(
                        KitNewtonBackend,
                        ref.consumer_detector,
                        ref.consumer_detector_sha256,
                        accuracy=ref.checker_accuracy,
                    ),
                )
            except SignalRequest as exc:
                return 128 + exc.signum
        print(
            json.dumps(
                {
                    k: result[k]
                    for k in (
                        "completed",
                        "physical_acceptance",
                        "steps",
                        "policy_evaluations",
                        "teardown_errors",
                    )
                }
            )
        )
        return result["exit_code"]
    except (ValueError, OSError, RuntimeError, ImportError, KeyError, TypeError) as exc:
        print(
            json.dumps(
                {
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "physical_acceptance": False,
                }
            )
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
