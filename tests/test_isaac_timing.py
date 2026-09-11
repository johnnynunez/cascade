"""The booth must not loop a zero-length USD animation timeline."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import isaac_runtime


class Stage:
    def __init__(self, start=0.0, end=0.0, rate=24.0):
        self.start, self.end, self.rate = start, end, rate

    def GetStartTimeCode(self):
        return self.start

    def GetEndTimeCode(self):
        return self.end

    def GetTimeCodesPerSecond(self):
        return self.rate

    def SetStartTimeCode(self, value):
        self.start = value

    def SetEndTimeCode(self, value):
        self.end = value


def timing(stage, **kwargs):
    assert hasattr(isaac_runtime, "ensure_time_code_range"), "Missing non-degenerate booth timeline setup"
    return isaac_runtime.ensure_time_code_range(stage, **kwargs)


def test_degenerate_timeline_gets_duration_in_stage_time_codes():
    stage = Stage(start=12.0, end=12.0, rate=60.0)
    result = timing(stage, duration_s=120.0)
    assert result["changed"] is True
    assert stage.start == 12.0
    assert stage.end - stage.start == 120.0 * 60.0


def test_existing_animation_range_is_preserved():
    stage = Stage(start=5.0, end=25.0, rate=24.0)
    result = timing(stage, duration_s=120.0)
    assert result["changed"] is False
    assert (stage.start, stage.end, stage.rate) == (5.0, 25.0, 24.0)


@pytest.mark.parametrize("duration", [0.0, -1.0, float("inf"), float("nan")])
def test_invalid_duration_is_not_authored(duration):
    stage = Stage()
    with pytest.raises(ValueError):
        timing(stage, duration_s=duration)
    assert (stage.start, stage.end) == (0.0, 0.0)
