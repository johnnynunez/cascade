"""Instrumentation preserves original observer semantics and bounded ownership."""
import contextlib
import importlib.util
import io
from pathlib import Path
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location("kitchen_observer_profile",
    Path(__file__).resolve().parents[1] / "benchmark/diagnostics/kitchen_observer_profile.py")
profile = importlib.util.module_from_spec(spec)
spec.loader.exec_module(profile)


def execute(source, env):
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        exec(source, env)
    return stream.getvalue()


def test_same_original_reads_result_stdout_and_nested_accounting():
    source = '''
def _obs_array(value):
    events.append(("array", value))
    return value
def _obs_collect():
    return _obs_array(art.get_dof_positions())
result = _obs_collect()
print("KITCHEN_OBSERVER", result)
'''
    def env():
        events = []
        return {"events": events, "art": SimpleNamespace(
            get_dof_positions=lambda: (events.append("read"), 17)[1])}
    plain, measured = env(), env()
    wrapped, binding = profile.instrument_snapshot(source, max_samples=2)
    assert execute(source, plain) == execute(wrapped, measured)
    assert plain["events"] == measured["events"] == ["read", ("array", 17)]
    state = measured["_cascade_observer_profiles_v1"][binding["source_sha256"]]
    assert state["samples"] == 1 and not state["stack"]
    assert set(state["zones"]) == {"snapshot_total", "state_collection", "array_readback", "articulation_positions"}
    for row in state["zones"].values():
        assert row["count"] == 1 and row["inclusive_ns"] >= row["exclusive_ns"] >= 0
    assert execute(wrapped, measured) == "KITCHEN_OBSERVER 17\n"
    with pytest.raises(RuntimeError, match="budget exhausted"):
        execute(wrapped, measured)
    assert measured["events"] == 2 * plain["events"]


def test_exception_propagates_and_profile_stack_drains():
    error = ValueError("native read rejected")
    def broken():
        raise error
    wrapped, binding = profile.instrument_snapshot("art.get_dof_positions()")
    env = {"art": SimpleNamespace(get_dof_positions=broken)}
    with pytest.raises(ValueError) as exc:
        execute(wrapped, env)
    assert exc.value is error
    row = env["_cascade_observer_profiles_v1"][binding["source_sha256"]]
    assert not row["stack"] and row["zones"]["articulation_positions"]["exceptions"] == 1
    assert row["zones"]["snapshot_total"]["exceptions"] == 1


def test_arguments_named_name_function_and_star_expansion_remain_unchanged():
    source = "result = physics_device_identity(*args, **kwargs)"
    env = {"physics_device_identity": lambda *args, **kwargs: (args, kwargs),
           "args": (1, 2), "kwargs": {"name": "source", "function": "payload"}}
    wrapped, _ = profile.instrument_snapshot(source)
    execute(wrapped, env)
    assert env["result"] == ((1, 2), {"name": "source", "function": "payload"})


def test_profile_read_is_separate_and_cannot_add_stdout_to_snapshot():
    source = "print(physics_device_identity())"
    wrapped, _ = profile.instrument_snapshot(source)
    env = {"physics_device_identity": lambda: {"actual": True}}
    assert execute(wrapped, env) == "{'actual': True}\n"
    assert execute(profile.read_profiles_code(), env).startswith("KITCHEN_PROFILE_ZLIB ")


@pytest.mark.parametrize("source,count", [("print(1)", 1), ("_observer_profile_call()", 1),
    ("global q\nq=physics_device_identity()", 1), ("physics_device_identity()", True),
    ("physics_device_identity()", 1025)])
def test_unreviewed_scope_or_unbounded_profile_is_rejected(source, count):
    with pytest.raises(ValueError):
        profile.instrument_snapshot(source, max_samples=count)
