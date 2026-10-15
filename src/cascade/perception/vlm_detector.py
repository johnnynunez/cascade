"""Deprecated: `VLMDetector` is `brain_detector.BrainDetector` with the old defaults.

Kept for one release so `detector.type: vlm` and existing imports keep working:
cosmos3-edge on :8082, no segmenter (box-only detections), and the old
constructor kwargs. New code uses `BrainDetector`, which reads its api key
from a file; only this shim still accepts a literal `api_key`.
"""

from __future__ import annotations

from .brain_detector import BrainDetector


class VLMDetector(BrainDetector):
    """`BrainDetector` with the old cosmos3-edge defaults and constructor."""

    def __init__(self, base_url: str = "http://127.0.0.1:8082/v1", model: str = "cosmos3-edge",
                 api_key: str | None = None, timeout_s: float = 20.0, conf: float | None = None,
                 **kwargs):
        # `conf` is accepted and ignored: brain boxes carry conf 1.0.
        super().__init__(base_url=base_url, model=model, timeout_s=timeout_s, **kwargs)
        if api_key is not None:
            self._api_key = str(api_key)
