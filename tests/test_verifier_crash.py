"""A crashing postcondition VERIFIER must not leave a self-reported `ok`
standing as verified.

Measured gap: `execute()` wrapped the Pigey check in `except Exception:
pass`, so a verifier failure (camera hiccup, truth channel down, a bug in
the checker) silently switched the closed loop OFF -- the skill's own
`ok: True` went out as the final word with no `verified` flag at all. Now
the crash becomes an UNVERIFIED postcondition that names the cause.
"""
from __future__ import annotations

from cascade.config import load_demo_config


def test_verifier_crash_becomes_unverified_not_silent_ok(tmp_path, capsys):
    from cascade.apps.demo import build_runtime, shutdown_runtime

    cfg = load_demo_config(arms=["mock"], camera="mock", llm="mock")
    runtime, arm = build_runtime(cfg, tmp_path / "run", view=False, serve=False)
    try:
        assert runtime.effects is not None, "the mock stack must run the postcondition checker"
        real_verify = runtime.effects.verify

        def _boom(*a, **k):
            raise RuntimeError("truth channel down")

        runtime.effects.verify = _boom
        res = runtime.execute("move_home", {})
        assert res["ok"] is True, res
        assert res.get("verified") is False, res
        assert "truth channel down" in res.get("verification_note", ""), res
        assert res["postcondition"]["status"] == "unverified"
        assert "postcondition verifier for move_home raised" in capsys.readouterr().err

        # control: with the checker intact the same skill is judged normally
        runtime.effects.verify = real_verify
        res2 = runtime.execute("move_home", {})
        assert res2["ok"] is True
        assert "verification_note" not in res2 or "crashed" not in res2.get("verification_note", "")
    finally:
        shutdown_runtime(runtime, arm)


def test_mutation_guard_verifier_result_is_not_a_plain_pass():
    """Static check that the swallow did not come back: the verify call in
    execute() must be followed by an annotate, never a bare `pass`."""
    import inspect

    from cascade.skills import runtime as rt

    src = inspect.getsource(rt.SkillRuntime.execute)
    i = src.index("self.effects.verify(")
    window = src[i : i + 900]
    assert "except Exception:\n                pass" not in window
    assert "annotate_result(result, pc)" in window
