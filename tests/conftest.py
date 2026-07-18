import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from wrc_demo.config import load_demo_config  # noqa: E402

URDF = REPO / "assets" / "urdf" / "00-arm-rs_asm-v3" / "urdf" / "00-arm-rs_asm-v3.urdf"
USD = REPO / "assets" / "usd" / "RS-rebot-dev-arm" / "00-arm-rs_asm-v3.usda"
# Assets are authored in the mirrored joint convention; the SDK and every q
# constant in this repo are local (q_local = -q_asset). See usd_model.
JOINT_SIGNS = [-1, -1, -1, -1, -1, -1]


def has_pinocchio() -> bool:
    try:
        import pinocchio  # noqa: F401

        return True
    except ImportError:
        return False


needs_pin = pytest.mark.skipif(
    not has_pinocchio() or not URDF.exists(),
    reason="pinocchio or RS model not available",
)


@pytest.fixture
def demo_cfg():
    return load_demo_config(camera="mock", arm="mock", llm="mock")


@pytest.fixture
def rng():
    return np.random.default_rng(42)
