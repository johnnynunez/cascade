import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from wrc_demo.config import load_demo_config  # noqa: E402

MODEL = REPO / "assets" / "usd" / "RS-rebot-dev-arm" / "00-arm-rs_asm-v3.usda"


def has_pinocchio() -> bool:
    try:
        import pinocchio  # noqa: F401

        return True
    except ImportError:
        return False


needs_pin = pytest.mark.skipif(
    not has_pinocchio() or not MODEL.exists(),
    reason="pinocchio or RS USD model not available",
)


@pytest.fixture
def demo_cfg():
    return load_demo_config(camera="mock", arm="mock", llm="mock")


@pytest.fixture
def rng():
    return np.random.default_rng(42)
