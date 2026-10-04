"""One frozen synthetic corpus, once per baseline/candidate; CPU only.

prepare materializes all images and convergence diagnostics before any
localizer. evaluate refuses to overwrite evidence, records every case and
never tunes the declared transformations, detector flags, or metric gates.
"""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from . import binary_layout_a as layout
from .checker_accuracy import AccuracyBoard, detect_accuracy_corners
from .checker_saddle import SaddleBoard, detect_saddle_corners
from .checker_corpus import filtered, raster, read_bound_json
from .planar_reference import apply_homography, fit_homography, reference_from_corners

SPEC_SHA = "dd062605ed97a30a6fcab49ee9fd1b25e5d237509898c199f35b9e203942280e"
TRUTH_SHA = "758f187caee602f3422af47b7fa04414a274de049bfec3778d6554f8a34d8f10"
ASSETS = Path(__file__).parent / "assets"
SADDLE_SPEC_SHA = "842b79d2c37c173eead2b263bf71daa67383136bda8fcf41f01efe9bb6b83599"
SADDLE_TRUTH_SHA = "192f0779e51bcc4697a82af1ce77a38eb8afa615523ba80553e982d3121fd2a7"


def profile(variant):
    if variant == "accuracy":
        return "checker_accuracy", SPEC_SHA, TRUTH_SHA, 48
    if variant == "saddle":
        return "checker_saddle", SADDLE_SPEC_SHA, SADDLE_TRUTH_SHA, 51
    raise ValueError("explicit known checker corpus variant required")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def source_manifest():
    result = {p.name: sha(p) for p in sorted(Path(__file__).parent.glob("*.py"))}
    result["assets/checker_saddle_consumer.json"] = sha(
        ASSETS / "checker_saddle_consumer.json"
    )
    return result


def inputs(variant="accuracy"):
    stem, spec_sha, truth_sha, count = profile(variant)
    if variant == "saddle":
        SaddleBoard(
            (ASSETS / "checker_saddle_consumer.json").read_text()
        ).require_consumer_implementation()
    spec = read_bound_json(ASSETS / (stem + "_corpus_spec.json"), spec_sha)
    truth = read_bound_json(ASSETS / (stem + "_corpus_truth.json"), truth_sha)
    if truth["geometry"] != layout.BinaryLayoutABoard().description():
        raise ValueError("frozen geometry/consumer code differs")
    if truth["spec_sha256"] != spec_sha or len(truth["cases"]) != count:
        raise ValueError("frozen corpus contract differs")
    return spec, truth


def prepare(out, variant="accuracy"):
    spec, truth = inputs(variant)
    _, spec_sha, truth_sha, count = profile(variant)
    out.mkdir(parents=True, exist_ok=False)
    source = source_manifest()
    write(
        out / "inputs-before.json",
        {"source": source, "spec_sha256": spec_sha, "truth_sha256": truth_sha},
    )
    cells = layout.layout_rectangles(layout.BinaryLayoutABoard())
    dark = [r["local_xy_bounds_m"] for r in cells if r["rgb8"] == [0, 0, 0]]
    assets, convergence = [], []
    for p in range(0, count, 3):
        base = truth["cases"][p]
        t0 = time.monotonic()
        rgb16 = raster(base["H"], dark, samples=16)
        rgb32 = raster(base["H"], dark, samples=32)
        for offset in range(3):
            case = truth["cases"][p + offset]
            assert case["H"] == base["H"]
            image16 = filtered(rgb16, case["filter"])
            image32 = filtered(rgb32, case["filter"])
            delta = image16.astype(float) - image32.astype(float)
            name = f"case-{p + offset:02}.npz"
            np.savez_compressed(out / name, rgb16=image16, rgb32=image32)
            assets.append(
                {
                    "id": case["id"],
                    "file": name,
                    "sha256": sha(out / name),
                    "rgb16_sha256": hashlib.sha256(image16.tobytes()).hexdigest(),
                    "rgb32_sha256": hashlib.sha256(image32.tobytes()).hexdigest(),
                }
            )
            convergence.append(
                {
                    "id": case["id"],
                    "absolute_mean_rgb8": float(np.abs(delta).mean()),
                    "absolute_max_rgb8": float(np.abs(delta).max()),
                    "different_pixels": int((delta != 0).any(axis=2).sum()),
                }
            )
        print(
            json.dumps(
                {
                    "prepared_projection_phase": base["id"],
                    "wall_s": time.monotonic() - t0,
                }
            ),
            flush=True,
        )
    after = source_manifest()
    write(
        out / "convergence.json",
        {
            "scope": "16versus32 quadrature; no detector. Descriptive error, no unregistered threshold.",
            "cases": convergence,
        },
    )
    write(
        out / "prepared.json",
        {
            "schema": 1,
            "spec_sha256": spec_sha,
            "truth_sha256": truth_sha,
            "source_before": source,
            "source_after": after,
            "source_unchanged": source == after,
            "cases": assets,
            "native_images_read": 0,
            "detector_calls": 0,
        },
    )
    if source != after:
        raise ValueError("source changed during raster preparation")


def evaluate(out, variant="accuracy"):
    import cv2

    cv2.setNumThreads(1)
    _, truth = inputs(variant)
    _, spec_sha, truth_sha, count = profile(variant)
    prepared = json.loads((out / "prepared.json").read_text())
    if (
        not prepared["source_unchanged"]
        or source_manifest() != prepared["source_after"]
        or prepared["spec_sha256"] != spec_sha
        or prepared["truth_sha256"] != truth_sha
    ):
        raise ValueError("prepared source/spec binding differs")
    for entry in prepared["cases"]:
        if sha(out / entry["file"]) != entry["sha256"]:
            raise ValueError("prepared image changed")
    result = out / "observations.jsonl"
    # Exclusive creation guards against accidentally re-evaluating and replacing
    # an earlier result. Failure midway is retained; no automatic retry.
    baseline, candidate = layout.BinaryLayoutABoard(), AccuracyBoard()
    variants = (
        ("baseline", baseline, lambda im: layout.detect_layout_corners(im, baseline)),
        ("accuracy", candidate, lambda im: detect_accuracy_corners(im, candidate)[0]),
    )
    if variant == "saddle":
        saddle = SaddleBoard()
        variants = (
            variants[1],
            ("saddle", saddle, lambda im: detect_saddle_corners(im, saddle)[0]),
        )
    counts = {name: 0 for name, _, _ in variants}
    with result.open("x") as stream:
        for case, asset in zip(truth["cases"], prepared["cases"], strict=True):
            if case["id"] != asset["id"]:
                raise ValueError("prepared case ordering differs")
            with np.load(out / asset["file"], allow_pickle=False) as data:
                rgb = data["rgb16"].copy()
            for name, board, detect in variants:
                row = {
                    "case": case["id"],
                    "variant": name,
                    "flags": 2 if name == "baseline" else 34,
                    "board_sha256": board.sha256,
                    "case_set": case.get("set", "original_corpus"),
                    "rgb_sha256": asset["rgb16_sha256"],
                    "reference_passed": False,
                }
                t0 = time.monotonic()
                try:
                    corners = detect(rgb)
                    row["corners_uv"] = corners.tolist()
                    error = corners - np.array(case["corners_uv"])
                    row["truth_residual_vectors_px"] = error.tolist()
                    row["truth_rms_px"] = float(np.sqrt((error**2).sum(axis=1).mean()))
                    row["truth_max_px"] = float(np.linalg.norm(error, axis=1).max())
                    fit = [i for i in range(35) if (i // 7 + i % 7) % 2 == 0]
                    held = [i for i in range(35) if i not in fit]
                    h = fit_homography(corners[fit], board.corners_xy()[fit])
                    vectors = (
                        apply_homography(np.linalg.inv(h), board.corners_xy()) - corners
                    )
                    row["fit_ids"], row["held_ids"] = fit, held
                    row["fit_residual_vectors_px"] = vectors[fit].tolist()
                    row["held_residual_vectors_px"] = vectors[held].tolist()
                    row["held_rms_px"] = float(
                        np.sqrt((vectors[held] ** 2).sum(axis=1).mean())
                    )
                    row["held_max_px"] = float(
                        np.linalg.norm(vectors[held], axis=1).max()
                    )
                    ref = reference_from_corners(
                        corners, board, rgb_sha256=row["rgb_sha256"]
                    )
                    row["reference"] = asdict(ref)
                    row["reference_passed"] = True
                    counts[name] += 1
                except Exception as exc:
                    row["error_type"], row["error"] = type(exc).__name__, str(exc)
                row["wall_s"] = time.monotonic() - t0
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                stream.flush()
                print(
                    json.dumps(
                        {k: row[k] for k in ("case", "variant", "reference_passed")}
                        | {"error": row.get("error")}
                    ),
                    flush=True,
                )
    after = source_manifest()
    write(
        out / "evaluation.json",
        {
            "schema": 1,
            "cases_per_variant": count,
            "variant": variant,
            "counts_reference_passed": counts,
            "source_before": prepared["source_after"],
            "source_after": after,
            "source_unchanged": after == prepared["source_after"],
            "observations_sha256": sha(result),
            "native_images_read": 0,
            "physical_admission": False,
            "spec_sha256": spec_sha,
            "truth_sha256": truth_sha,
        },
    )
    if after != prepared["source_after"]:
        raise ValueError("source changed during evaluation")


def negatives(out, variant="accuracy"):
    """Predeclared synthetic adversaries; preserve even unexpected acceptance."""
    import cv2
    from . import binary_reference as binary
    from .checker_corpus import project

    cv2.setNumThreads(1)
    spec, truth = inputs(variant)
    prepared = json.loads((out / "prepared.json").read_text())
    if source_manifest() != prepared["source_after"]:
        raise ValueError("source changed before synthetic adversaries")
    case = truth["cases"][0]
    if case["id"] != (
        "fronto/phase0/area" if variant == "accuracy" else "fronto/newphase0/area"
    ):
        raise ValueError("negative base differs")
    asset = prepared["cases"][0]
    if sha(out / asset["file"]) != asset["sha256"]:
        raise ValueError("negative base image changed")
    with np.load(out / asset["file"], allow_pickle=False) as data:
        original = data["rgb16"].copy()
    board = AccuracyBoard() if variant == "accuracy" else SaddleBoard()
    detect = detect_accuracy_corners if variant == "accuracy" else detect_saddle_corners
    rows = layout.layout_rectangles(board.geometry)
    results = out / "negatives.jsonl"
    with results.open("x") as stream:
        for kind in spec["negative_cases"]:
            rgb = original.copy()
            if kind == "reflect_x":
                rgb = np.ascontiguousarray(rgb[:, ::-1])
            elif kind in (
                "duplicate_tag",
                "foreign_tag",
                "missing_tag",
                "one_payload_bit_corruption",
            ):
                cells = {r["label"]: r["rgb8"][0] for r in rows}
                if kind == "missing_tag":
                    for key in cells:
                        if key.startswith("tag_0_"):
                            cells[key] = 255
                elif kind == "one_payload_bit_corruption":
                    cells["tag_0_2_2"] = 255 - cells["tag_0_2_2"]
                else:
                    bits = cv2.aruco.generateImageMarker(
                        binary._api()[1],
                        1 if kind == "duplicate_tag" else 49,
                        6,
                        borderBits=1,
                    )
                    for r in range(6):
                        for c in range(6):
                            cells[f"tag_0_{r}_{c}"] = int(bits[r, c])
                rgb = raster(
                    case["H"],
                    [r["local_xy_bounds_m"] for r in rows if cells[r["label"]] == 0],
                )
            elif kind in ("missing_checker_corner", "half_board_occlusion"):
                bounds = (
                    [0.039, 0.039, 0.057, 0.057]
                    if kind == "missing_checker_corner"
                    else [0, 0, 0.096, 0.144]
                )
                a, b, c, d = bounds
                quad = project(case["H"], [[a, b], [c, b], [c, d], [a, d]])
                lo = np.floor(quad.min(axis=0)).astype(int)
                hi = np.ceil(quad.max(axis=0)).astype(int)
                rgb[lo[1] : hi[1], lo[0] : hi[0]] = 127
            elif kind == "gaussian_sigma2_kernel13":
                rgb = filtered(rgb, {"gaussian_sigma_px": 2.0, "kernel": 13})
            elif kind == "downscale_one_eighth":
                rgb = cv2.resize(rgb, (80, 60), interpolation=cv2.INTER_AREA)
            else:
                raise ValueError("undeclared negative")
            row = {
                "case": kind,
                "rgb_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
                "reference_passed": False,
                "expected_rejection": True,
            }
            try:
                ref = reference_from_corners(
                    detect(rgb, board)[0],
                    board,
                    rgb_sha256=row["rgb_sha256"],
                )
                row["reference_passed"] = True
                row["reference"] = asdict(ref)
            except Exception as exc:
                row["error_type"], row["error"] = type(exc).__name__, str(exc)
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
            print(
                json.dumps({k: row[k] for k in ("case", "reference_passed")}),
                flush=True,
            )
    if source_manifest() != prepared["source_after"]:
        raise ValueError("source changed during synthetic adversaries")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("prepare", "evaluate", "negatives"))
    parser.add_argument("out", type=Path)
    parser.add_argument("--variant", choices=("accuracy", "saddle"), default="accuracy")
    args = parser.parse_args()
    {"prepare": prepare, "evaluate": evaluate, "negatives": negatives}[args.operation](
        args.out, args.variant
    )


if __name__ == "__main__":
    main()
