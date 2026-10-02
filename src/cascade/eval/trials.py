"""Bind external benchmark outcomes to independent, episode-specific evidence.

An upstream success bit is not physical admission. The verifier callback is a
trusted integration boundary: this module checks its provenance and coverage,
not the physics behind its measurements.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Callable, Mapping
from types import MappingProxyType


def digest_json(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def require_digest(value: str, name: str = "digest") -> None:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")


@dataclass(frozen=True)
class Artifact:
    path: str
    sha256: str

    def __post_init__(self):
        if not isinstance(self.path, str) or not self.path:
            raise ValueError("artifact path is required")
        require_digest(self.sha256, "artifact sha256")

    @classmethod
    def read(cls, path: str | Path) -> "Artifact":
        path = Path(path).resolve()
        return cls(str(path), hashlib.sha256(path.read_bytes()).hexdigest())


@dataclass(frozen=True)
class EpisodeBinding:
    trial_id: str
    configuration_sha256: str
    model_identity_sha256: str
    epoch: str
    first_step: int
    last_step: int
    first_sim_time_s: float
    last_sim_time_s: float
    first_snapshot_sha256: str
    last_snapshot_sha256: str

    def __post_init__(self):
        for name in ("trial_id", "epoch"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{name} is required")
        for name in ("configuration_sha256", "model_identity_sha256",
                     "first_snapshot_sha256", "last_snapshot_sha256"):
            require_digest(getattr(self, name), name)
        if any(type(v) is not int or v < 0 for v in (self.first_step, self.last_step)):
            raise ValueError("physical steps must be nonnegative integers")
        if self.last_step <= self.first_step:
            raise ValueError("a verification window must advance physical steps")
        if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0
               for v in (self.first_sim_time_s, self.last_sim_time_s)):
            raise ValueError("physical times must be finite and nonnegative")
        if self.last_sim_time_s <= self.first_sim_time_s:
            raise ValueError("a verification window must advance physical time")
        if self.first_snapshot_sha256 == self.last_snapshot_sha256:
            raise ValueError("distinct physical snapshots are required")


@dataclass(frozen=True)
class ExternalEpisode:
    backend: str
    source_revision: str
    record_id: str
    record_sha256: str
    artifact: Artifact
    benchmark_success: bool | None
    binding: EpisodeBinding | None = None

    def __post_init__(self):
        if not self.backend or not self.record_id:
            raise ValueError("backend and record_id are required")
        if re.fullmatch(r"[0-9a-f]{40}", self.source_revision) is None:
            raise ValueError("source_revision must be a full Git commit")
        require_digest(self.record_sha256, "record_sha256")
        if self.benchmark_success is not None and type(self.benchmark_success) is not bool:
            raise ValueError("benchmark_success must be bool or None")
        if self.binding is not None and self.binding.trial_id != self.record_id:
            raise ValueError("physical binding must identify this exact external record")


@dataclass(frozen=True)
class IndependentEvidence:
    binding: EpisodeBinding
    external_artifact_sha256: str
    external_record_sha256: str
    channel: str
    artifact: Artifact
    checks: Mapping[str, str]

    def __post_init__(self):
        require_digest(self.external_artifact_sha256)
        require_digest(self.external_record_sha256)
        if not self.channel:
            raise ValueError("an independent measurement channel is required")
        if not isinstance(self.checks, Mapping):
            raise ValueError("checks must be a mapping")
        object.__setattr__(self, "checks", MappingProxyType(dict(self.checks)))
        if any(not key or value not in {"confirmed", "refuted", "unverified"}
               for key, value in self.checks.items()):
            raise ValueError("checks require named three-state verdicts")


@dataclass(frozen=True)
class TrialVerdict:
    status: str
    reason: str
    benchmark_success: bool | None
    independent_channel: str | None = None


def verify_episode(
    episode: ExternalEpisode,
    *,
    required_checks: frozenset[str],
    independent_verifier: Callable[[ExternalEpisode], IndependentEvidence] | None = None,
) -> TrialVerdict:
    """Check receipt binding; never manufacture absent physical evidence."""
    def result(status, reason, channel=None):
        return TrialVerdict(status, reason, episode.benchmark_success, channel)

    if not required_checks or any(not isinstance(key, str) or not key for key in required_checks):
        raise ValueError("at least one explicit independent check is required")
    if episode.binding is None:
        return result("unverified", "external result has no physical episode binding")
    if independent_verifier is None:
        return result("unverified", "independent verifier unavailable")
    try:
        if Artifact.read(episode.artifact.path) != episode.artifact:
            return result("unverified", "external result artifact changed after import")
        evidence = independent_verifier(episode)
        if not isinstance(evidence, IndependentEvidence):
            return result("unverified", "invalid independent evidence schema")
        if (evidence.binding != episode.binding
                or evidence.external_artifact_sha256 != episode.artifact.sha256
                or evidence.external_record_sha256 != episode.record_sha256):
            return result("unverified", "independent evidence belongs to a different episode or artifact")
        if evidence.channel == episode.backend or evidence.artifact.sha256 == episode.artifact.sha256:
            return result("unverified", "benchmark outcome is not an independent measurement")
        if Artifact.read(evidence.artifact.path) != evidence.artifact:
            return result("unverified", "independent evidence artifact is missing or changed")
        if any(value == "refuted" for value in evidence.checks.values()):
            return result("refuted", "independent measurement refuted", evidence.channel)
        if any(evidence.checks.get(key) != "confirmed" for key in required_checks):
            return result("unverified", "required independent checks missing or unverified", evidence.channel)
        if episode.benchmark_success is None:
            return result("unverified", "upstream benchmark success unavailable", evidence.channel)
        if not episode.benchmark_success:
            return result("refuted", "upstream benchmark criterion failed", evidence.channel)
        return result("confirmed", "upstream outcome and bound independent checks confirmed", evidence.channel)
    except Exception as exc:
        return result("unverified", f"independent verifier failed: {type(exc).__name__}")
