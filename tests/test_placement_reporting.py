"""The host must receive an explicit verdict after the motion routine returns."""
from copy import deepcopy

import pytest

from cascade.agent.effects import Postcondition
from cascade.config import load_demo_config


@pytest.fixture
def runtime(tmp_path):
    from cascade.apps.demo import build_runtime, shutdown_runtime

    cfg = load_demo_config(arms=["mock"], camera="mock", llm="mock")
    runtime, arm = build_runtime(cfg, tmp_path / "run", view=False, serve=False)
    try:
        yield runtime
    finally:
        shutdown_runtime(runtime, arm)


@pytest.mark.parametrize("verdict", ["unverified", "confirmed", "refuted", "crashed"])
def test_placement_reply_follows_verifier_and_preserves_motion_evidence(runtime, verdict):
    calls = []
    motion = {
        "ok": True, "picked": "orange", "destination": "open box",
        "destination_kind": "configured_point", "placed_at": [.30, -.14, .104],
        "grip_verified": True,
        "return_home": {"attempted": True, "ok": False, "error": "home path unavailable"},
        "next_action": "Report this result and await the next user order.",
    }
    runtime.skill_pick_and_place = lambda **args: (calls.append(args) or deepcopy(motion))

    def verify(*args, **kwargs):
        if verdict == "crashed":
            raise RuntimeError("physics unavailable")
        return Postcondition(
            skill="pick_and_place", kind="relocated", status=verdict,
            evidence="Containment and release were not established.", channel="physics",
        )

    runtime.effects.verify = verify
    result = runtime.execute("pick_and_place", {"object": "orange", "destination": "open box"})
    assert calls == [{"object": "orange", "destination": "open box"}]
    for key in ("picked", "destination", "destination_kind", "placed_at", "grip_verified", "return_home"):
        assert result[key] == motion[key]
    if verdict in {"unverified", "crashed"}:
        assert result["ok"] is True and result["verified"] is False
        assert result["postcondition"]["status"] == "unverified"
        assert result["verification_note"]
        assert result["next_action"].startswith("Start the reply with 'Placement unverified.'")
        assert "does not confirm" in result["next_action"]
        assert "return_home failure" in result["next_action"]
        assert "without another movement" in result["next_action"]
    elif verdict == "confirmed":
        assert result["verified"] is True
        assert result["next_action"] == motion["next_action"]
    else:
        assert result["ok"] is False
        assert "Do not start another pick" in result["next_action"]
        assert "Placement unverified" not in result["next_action"]
