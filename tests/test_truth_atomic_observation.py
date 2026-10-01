"""Optional authority executes the same read-only probe, with no USD fallback."""
import copy
from contextlib import nullcontext
from types import ModuleType, SimpleNamespace as S
import sys

import numpy as np
import pytest

from test_truth_probe_readonly import probe_bridge


@pytest.fixture
def atomic_bridge(probe_bridge, monkeypatch):
    state, reader = probe_bridge
    reader.pose("pink cube")  # Existing legacy API, constructor behavior unchanged.
    cls = type(state.views[0])
    state.valid_handle = True
    monkeypatch.setattr(cls, "is_physics_handle_valid", lambda self: state.valid_handle, raising=False)
    backend = ModuleType("isaacsim.core.experimental.utils.backend")
    def use_backend(name, **kw):
        assert name == "tensor" and kw == dict(raise_on_unsupported=True, raise_on_fallback=True)
        return nullcontext()
    backend.use_backend = use_backend
    monkeypatch.setitem(sys.modules, backend.__name__, backend)
    monkeypatch.setattr(sys.modules["pxr"].UsdGeom, "GetStageMetersPerUnit", lambda stage: 1., raising=False)
    state.clock = dict(version=1, engine="physx", clock="SimulationManager", epoch="epoch",
        robot_id="/robot", sim_time=1., physics_step=120, physics_dt_s=1/120)
    state.q = np.arange(8, dtype=float)
    state.art_valid = True
    tensor = lambda value: S(numpy=lambda: np.asarray(value))
    state.art = S(is_physics_tensor_entity_valid=lambda: state.art_valid,
        get_dof_positions=lambda: tensor([state.q]),
        get_world_poses=lambda: (tensor([[1., 2., 3.]]), tensor([[1., 0., 0., 0.]])))
    reader._client.bridge_globals.update(art=state.art, ARM_IDX=list(range(6)),
        names=[f"joint{i}" for i in range(1, 7)]+["left", "right"],
        _motion_clock_snapshot=lambda: copy.deepcopy(state.clock))
    reader._client._addr = ("127.0.0.1", 8697)
    original = reader._client.request
    state.requests = []
    def request(packet, **kwargs):
        state.requests.append(kwargs)
        return original(packet)
    reader._client.request = request
    return state, reader


def test_same_probe_adds_optional_atomic_metadata_and_bounded_fresh_read(atomic_bridge):
    state, reader = atomic_bridge
    result = reader.observation(("pink cube", "pink object"))
    assert result["position_m"] == [.17, .15, .04]
    assert result["resolved_name"] == "pink_cube"
    assert result["resolved_path"] == "/World_Props/pink_cube"
    assert result["q_asset"] == list(range(6))
    assert result["joint_convention"] == "asset"
    assert result["joint_names"] == [f"joint{i}" for i in range(1, 7)]
    assert result["joint_indices"] == list(range(6))
    assert result["base_position_world"] == [1., 2., 3.]
    assert result["base_orientation_wxyz"] == [1., 0., 0., 0.]
    assert result["physics_clock"] == state.clock
    assert result["source"] == ("127.0.0.1", 8697)
    assert state.requests == [{"timeout_s": 1.}]
    assert not state.writes and state.authored_dynamic_reads == 0


@pytest.mark.parametrize("which", ["before", "after", "articulation"])
def test_invalid_physics_handles_cannot_authorize_even_if_legacy_pose_exists(atomic_bridge, monkeypatch, which):
    state, reader = atomic_bridge
    if which == "articulation": state.art_valid = False
    else:
        calls = iter([False, True] if which == "before" else [True, False])
        monkeypatch.setattr(type(state.views[0]), "is_physics_handle_valid", lambda self: next(calls))
    assert reader.observation(("pink cube",)) is None
    assert reader.pose("pink cube") == [.17, .15, .04]
    assert not state.writes


def test_clock_changed_during_read_has_no_authority(atomic_bridge):
    state, reader = atomic_bridge
    def moving_clock():
        state.clock["physics_step"] += 1
        return copy.deepcopy(state.clock)
    reader._client.bridge_globals["_motion_clock_snapshot"] = moving_clock
    assert reader.observation(("pink cube",)) is None


def prim(path):
    return S(GetPath=lambda: S(pathString=path), GetTypeName=lambda: "Xform", HasAPI=lambda api: True)


@pytest.mark.parametrize("names", [("orange", "orange"), ("pink-cube", "pink_cube")])
@pytest.mark.parametrize("failed_one", [False, True])
def test_inventory_identity_ambiguity_survives_failed_pose_read(atomic_bridge, monkeypatch, names, failed_one):
    state, reader = atomic_bridge
    paths = ["/World_Props/a/"+names[0], "/World_Props/b/"+names[1]]
    stage = S(Traverse=lambda: iter([prim(path) for path in paths]))
    monkeypatch.setattr(sys.modules["omni.usd"], "get_context", lambda: S(get_stage=lambda: stage))
    cls = type(state.views[0]); original_init = cls.__init__; original_read = cls.get_world_poses
    def init(self, path, **kwargs):
        original_init(self, path, **kwargs); self.path = path
    def read(self):
        if failed_one and self.path == paths[0]: raise RuntimeError("one body read failed")
        return original_read(self)
    monkeypatch.setattr(cls, "__init__", init); monkeypatch.setattr(cls, "get_world_poses", read)
    assert reader.observation((names[1],)) is None


def test_canonical_unique_but_alias_ambiguous_is_unknown(atomic_bridge, monkeypatch):
    state, reader = atomic_bridge
    paths = ["/World_Props/"+name for name in ("orange", "red_cube", "red_can")]
    stage = S(Traverse=lambda: iter([prim(path) for path in paths]))
    monkeypatch.setattr(sys.modules["omni.usd"], "get_context", lambda: S(get_stage=lambda: stage))
    assert reader.observation(("orange", "red object")) is None
    assert reader.observation(("orange", "orange fruit"))["resolved_name"] == "orange"


def test_failed_alias_bodies_still_make_inventory_ambiguous(atomic_bridge, monkeypatch):
    state, reader = atomic_bridge
    paths = ["/World_Props/"+name for name in ("orange", "red_cube", "red_can")]
    stage = S(Traverse=lambda: iter([prim(path) for path in paths]))
    monkeypatch.setattr(sys.modules["omni.usd"], "get_context", lambda: S(get_stage=lambda: stage))
    cls = type(state.views[0]); original = cls.__init__
    def init(self, path, **kw):
        if "red_" in path: raise RuntimeError("body pose unavailable")
        original(self, path, **kw)
    monkeypatch.setattr(cls, "__init__", init)
    assert reader.observation(("orange", "red object")) is None
    assert reader.observation(("orange",))["resolved_name"] == "orange"


def test_failed_forced_read_cannot_reuse_atomic_cache(atomic_bridge):
    state, reader = atomic_bridge
    assert reader.observation(("pink cube",)) is not None
    reader.ttl_s = 1000
    reader._client.request = lambda *a, **kw: (_ for _ in ()).throw(TimeoutError("no fresh snapshot"))
    assert reader.observation(("pink cube",)) is None
