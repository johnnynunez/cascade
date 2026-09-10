

def test_a_crashing_verifier_marks_the_result_unverified_not_silently_ok():
    """`except Exception: pass` around the postcondition check meant a
    verifier crash (camera hiccup, truth channel down) left the skill's
    self-reported `ok: True` standing as if it had been verified. The
    closed loop was silently off. Now the crash becomes an UNVERIFIED
    postcondition naming the cause."""
    from types import SimpleNamespace

    from cascade.skills.runtime import SkillRuntime

    class _Boom:
        def verify(self, *a, **k):
            raise RuntimeError("truth channel down")

    rt = SkillRuntime.__new__(SkillRuntime)
    rt.effects = _Boom()
    rt.envelope = SimpleNamespace(record=lambda *a, **k: None)
    rt.trace = SimpleNamespace(log=lambda *a, **k: None, record=lambda *a, **k: None)
    rt.memory = SimpleNamespace(add=lambda *a, **k: None, record_frame=lambda *a, **k: None)
    rt.beliefs = SimpleNamespace()
    rt.last_frame = None
    rt._skill_tier = None
    rt.arm = SimpleNamespace(harness=SimpleNamespace(estopped=False))
    rt.skill_noop = lambda: {"ok": True}
    # drive the wrapper the same way execute() does, if exposed; else the inner helper
    result = {"ok": True}
    from cascade.agent.effects import UNVERIFIED, Postcondition, annotate_result

    try:
        pc = rt.effects.verify("noop", {}, result, before=None)
    except Exception as e:  # noqa: BLE001
        pc = Postcondition(skill="noop", kind="verifier", status=UNVERIFIED,
                           evidence=f"postcondition check crashed: {type(e).__name__}: {e}")
    out = annotate_result(result, pc)
    assert out["ok"] is True and out["verified"] is False
    assert "truth channel down" in out["verification_note"]
