"""Content identity for one opened MicroDuck native recipe, not an outcome.

Paths, ports, output directories and elapsed/bootstrap counters are deliberately
absent. Consumers pin the digest independently; learning it from an untrusted
hello would only identify whichever model happened to answer that endpoint.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path, PurePosixPath
import re


IDENTITY_KEY = "model_identity_sha256"


def digest_token(value):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("model identity requires an exact lowercase SHA256")
    return value


def canonical_bytes(value):
    """No nonfinite values, repr coercion, SDK objects or incidental ordering."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def support_contract_digest(value, model_identity_sha256=None):
    """Bind the registry itself, not merely its claimed model digest."""
    from ..control.mobile_support import support_contract

    contract = support_contract(value)
    if contract is None:
        raise ValueError("explicit support_contract is required for a physical endpoint")
    if model_identity_sha256 is not None and contract[IDENTITY_KEY] != model_identity_sha256:
        raise ValueError("support_contract model_identity_sha256 mismatch")
    return hashlib.sha256(canonical_bytes(contract)).hexdigest()


def _file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _relative(value):
    if (not isinstance(value, str) or not value or "\\" in value
            or PurePosixPath(value).is_absolute() or ".." in PurePosixPath(value).parts
            or PurePosixPath(value).as_posix() != value):
        raise ValueError("identity content path must be canonical and relative")
    return value


def _hashes(mapping):
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError("identity requires nonempty source hashes")
    return {_relative(k): digest_token(v) for k, v in mapping.items()}


def build_model_identity(admission, native, *, repo, runtime_scene):
    """Build after backend.open, from admitted bytes and measured native fields.

Rehash inputs after bootstrap, so a file changed while Kit initialized cannot
inherit its earlier admission. This is recipe/provenance binding, not proof of
locomotion, fidelity of a solver, contact support or any physical outcome.
"""
    bundle = Path(admission["bundle"]).resolve()
    receipt = admission["receipt"]
    receipt_hash = digest_token(admission["asset_receipt_sha256"])
    if _file_digest(bundle / "receipt.json") != receipt_hash:
        raise ValueError("bundle receipt changed after admission")
    outputs = {}
    for row in receipt["outputs"]:
        name = _relative(row["path"])
        if name in outputs or set(row) != {"path", "size", "sha256"}:
            raise ValueError("invalid or duplicate identity bundle member")
        path = bundle / name
        if path.is_symlink() or not path.resolve().is_relative_to(bundle):
            raise ValueError("identity bundle member escapes admitted root")
        digest = digest_token(row["sha256"])
        if (type(row["size"]) is not int or row["size"] < 0
                or path.stat().st_size != row["size"] or _file_digest(path) != digest):
            raise ValueError("bundle member changed after admission: " + name)
        outputs[name] = dict(row)
    if not outputs:
        raise ValueError("identity requires complete bundle contents")
    sources = _hashes(admission["source_sha256"])
    for relative, expected in sources.items():
        if _file_digest(Path(repo) / relative) != expected:
            raise ValueError("runtime source changed during bootstrap: " + relative)
    layers = []
    for row in native["consumed_asset_layers"]:
        path = Path(row["path"]).resolve()
        name = path.relative_to(bundle).as_posix()
        if name not in outputs or row["sha256"] != outputs[name]["sha256"]:
            raise ValueError("consumed layer is not bound to bundle content")
        layers.append({"path": name, "sha256": row["sha256"]})
    if not layers or len({r["path"] for r in layers}) != len(layers):
        raise ValueError("identity requires distinct consumed asset layers")
    root = _relative(receipt["usd_path"])
    if (root not in outputs or outputs[root]["sha256"] != admission["asset_sha256"]
            or root not in {r["path"] for r in layers}):
        raise ValueError("consumed root USD is not the admitted asset")
    scene = Path(runtime_scene).read_bytes()
    if hashlib.sha256(scene).hexdigest() != native["runtime_layer_sha256"]:
        raise ValueError("authored runtime scene changed after backend open")
    # USD exported by the backend references the verified bundle root by an
    # absolute file path. Normalize only that admitted prefix, not scene data.
    scene = scene.replace(str(bundle).encode(), b"cascade-admitted-bundle:")
    bam = native["bam"]
    if (bam["params"] != admission["bam_params"]
            or _hashes(bam["source_sha256"]) != _hashes(admission["bam_source_sha256"])):
        raise ValueError("effective BAM differs from admitted parameters/source")
    if bam["device"] != native["physics_device"] or bam["newton_version"] != native["newton_version"]:
        raise ValueError("BAM and solver identities differ")
    bam_fields = ("implementation", "revision", "source_sha256", "device", "newton_version",
                  "params", "q_indices", "dof_indices", "friction_reference", "env_dof_stride",
                  "mechanical_damping", "armature")
    native_fields = ("solver", "newton_version", "warp_version", "actual_physics_dt",
                     "physics_dt_s", "physics_device", "gpu_attestation", "configuration",
                     "native_labels", "native_model_properties", "native_body_properties", "disabled_source_actuators",
                     "initialization", "support_contract", "support_extraction", "runtime_versions")
    recipe = {
        "schema": "cascade.microduck.effective-model.v1", "engine": "newton",
        "bundle_receipt_sha256": receipt_hash, "bundle_outputs": [outputs[k] for k in sorted(outputs)],
        "consumed_asset_layers": sorted(layers, key=lambda r: r["path"]),
        "runtime_scene_sha256": hashlib.sha256(scene).hexdigest(),
        "asset_sha256": digest_token(admission["asset_sha256"]),
        "policy_sha256": digest_token(admission["policy_sha256"]),
        "policy_dt_s": .020, "limits": admission["limits"],
        "source_sha256": sources, "bam_config_sha256": digest_token(admission["bam_config_sha256"]),
        "bam": {k: bam[k] for k in bam_fields},
        "native": {k: native[k] for k in native_fields},
    }
    # Copy by canonical serialization: callers cannot change a nested receipt
    # after its digest is published, and all values have one JSON meaning.
    payload = canonical_bytes(recipe)
    return {IDENTITY_KEY: hashlib.sha256(payload).hexdigest(),
            "recipe": json.loads(payload),
            "support_contract": {**copy.deepcopy(native["support_contract"]),
                                 IDENTITY_KEY: hashlib.sha256(payload).hexdigest()}}
