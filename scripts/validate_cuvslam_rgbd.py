"""Native cuVSLAM estimator exercise with upstream synthetic camera images.

No hardware, simulator, actuator, observed obstacle map or navigation admission.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from cascade.sensing import BufferedSensorProvider, SensorHub
from cascade.sensing.hub import SensorDescriptor
from cascade.sensing.models import MeasurementMetadata, ObservationEnvelope, RgbdPayload
from cascade.spatial.cuvslam import CuVslamSpatialDomain


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def close_owners(report, domain, hub):
    """Attempt both cleanups and retain a failed close as a failed exercise."""
    for name, close in (("domain_close", domain.close), ("hub_close", lambda: hub.close(2.))):
        try:
            report[name] = close()
        except Exception as exc:
            report[name] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    report["ok"] = bool(report["ok"] and report["domain_close"].get("ok") is True
                        and report["hub_close"].get("ok") is True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdk-checkout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    if options.output.exists():
        parser.error("output already exists; preserve the previous receipt")
    import cuvslam
    sys.path.insert(0, str(options.sdk_checkout.resolve() / "python/test"))
    import data_gen
    binding = Path(cuvslam._core.__file__).resolve()
    binding_hash = file_hash(binding)
    report = {"kind": "native_sdk_synthetic_rgbd_replay", "physical_admission": False,
              "actuator_commands": 0, "version": cuvslam.get_version(),
              "binding_path": str(binding), "binding_sha256": binding_hash,
              "library_sha256": file_hash(binding.parent / "libcuvslam.so"),
              "generator_sha256": file_hash(options.sdk_checkout / "python/test/data_gen.py"),
              "results": [], "ok": False}
    calibration = hashlib.sha256(json.dumps({"width": 640, "height": 480,
        "intrinsics": [320., 0., 320., 0., 320., 240., 0., 0., 1.],
        "pixel_center_offset_uv": [0., 0.]} , sort_keys=True).encode()).hexdigest()
    report["calibration_sha256"] = calibration
    report["script_sha256"] = file_hash(__file__)
    descriptor = SensorDescriptor("rgbd", "synthetic-camera", "upstream-pattern", "rgbd",
        "optical", "synthetic-seconds", "synthetic", calibration_id=calibration,
        max_age_s=5., read_timeout_s=1.)
    provider = BufferedSensorProvider(descriptor)
    hub = SensorHub()
    hub.register(provider)
    domain = CuVslamSpatialDomain("space", "synthetic-camera", hub,
        sensor_domain="sensors", sensor_id="rgbd", map_id="synthetic-plane",
        map_epoch="native-validation", map_frame_id="local-map",
        binding_sha256=binding_hash, max_gap_s=.1, timeout_s=30.)
    cameras = data_gen.generate_stereo_camera(640, 480, baseline=.25)
    generator = data_gen.ImageGenerator(cameras, steps=100)
    start_distance = generator.get_start_distance()

    def capture(index, *, warmup=False):
        images, distance = generator.generate_zoomed_images(index)
        rgb = np.repeat(images[0][..., None], 3, axis=2)
        depth = np.full((480, 640), start_distance - distance, dtype="<f4")
        payload = RgbdPayload(MeasurementMetadata("optical", calibration), 640, 480,
            rgb.tobytes(), depth.tobytes(), (320., 0., 320., 0., 320., 240., 0., 0., 1.))
        observation = ObservationEnvelope("upstream-pattern", "rgbd", "replay-session",
            0 if warmup else index + 1, "synthetic-seconds", .5 if warmup else 1. + index / 30.,
            time.monotonic(), 0., None, "synthetic", payload)
        provider.publish(observation)
        assert hub.read("rgbd") is observation
        return {"epoch": observation.epoch, "sequence": observation.sequence,
                "capture_sha256": observation.sha256}, distance

    try:
        args, _ = capture(0, warmup=True)
        start = time.monotonic()
        result = domain.execute("warmup_localization", args)
        report["warmup"] = {"seconds": time.monotonic() - start, "result": result}
        assert result["ok"], result
        for index in range(12):
            args, distance = capture(index)
            start = time.monotonic()
            result = domain.execute("track_capture", args)
            report["results"].append({"frame": index, "known_forward_m": distance,
                "seconds": time.monotonic() - start, "result": result})
            assert result["ok"], result
            assert result["physical_admission"] is False
            assert result["pose"]["position_error_m"] is None
            assert np.isfinite(result["pose"]["translation_m"]).all()
        last = report["results"][-1]
        report["final_translation_error_m"] = float(np.linalg.norm(
            np.array(last["result"]["pose"]["translation_m"]) - [0., 0., last["known_forward_m"]]))
        assert report["final_translation_error_m"] < .1, report["final_translation_error_m"]
        report["stop"] = domain.stop()
        report["post_stop"] = domain.execute("get_localization", {})
        assert not report["post_stop"]["ok"]
        report["ok"] = True
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        close_owners(report, domain, hub)
        with options.output.open("x") as output:
            output.write(json.dumps(report, indent=2) + "\n")
        print(json.dumps({key: report[key] for key in ("ok", "domain_close", "hub_close")}))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
