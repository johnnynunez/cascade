"""The kitchen keeps the event retry cap through profile inheritance."""

import pytest

from cascade.config import load_demo_config


@pytest.mark.parametrize("arm", ["isaac_kitchen", "isaac_kitchen_gpu"])
@pytest.mark.parametrize("booth", [False, True])
def test_kitchen_retry_budget(arm, booth, monkeypatch):
    if booth:
        monkeypatch.setenv("CASCADE_BOOTH", "1")
    config = load_demo_config(camera="mock", arm=arm, llm="mock")
    assert config.grasp.max_pick_attempts == 2
    assert config.arms[0]["resolved"]["grasp"]["max_pick_attempts"] == 2
