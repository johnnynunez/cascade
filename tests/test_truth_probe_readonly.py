"""Execute the actual bridge probe against views with Isaac's authoring defaults."""
import contextlib
import io
import sys
from types import ModuleType, SimpleNamespace

import pytest

from cascade.sim.truth import TruthPoseReader


@pytest.fixture
def probe_bridge(monkeypatch):
    state = SimpleNamespace(poses=[[.17, .15, .04]], writes=[], views=[],
                            failed_read=False, authored_dynamic_reads=0)

    class Prim:
        def __init__(self, path, dynamic):
            self.path, self.dynamic = path, dynamic

        def GetPath(self):
            return SimpleNamespace(pathString=self.path)

        def GetTypeName(self):
            return "Xform"

        def HasAPI(self, api):
            return self.dynamic

    props = [Prim("/World_Props/pink_cube", True), Prim("/World/box", False)]

    class Xformable:
        def __init__(self, prim):
            self.prim = prim

        def ComputeLocalToWorldTransform(self, time_code):
            if self.prim.dynamic:
                state.authored_dynamic_reads += 1
                return SimpleNamespace(ExtractTranslation=lambda: [9., 9., 9.])
            return SimpleNamespace(ExtractTranslation=lambda: [.3, -.2, .01])

    class RigidPrim:
        def __init__(self, path, *, reset_xform_properties=True,
                     prepare_contact_sensors=True):
            # These defaults mirror the installed Isaac RigidPrim API: an
            # apparently observational construction otherwise authors USD.
            if reset_xform_properties:
                state.writes.append((path, "normalize_xform_ops"))
            if prepare_contact_sensors:
                state.writes.append((path, "set_sleep_threshold_zero"))
            state.views.append(self)

        def get_world_poses(self):
            if state.failed_read:
                raise RuntimeError("physics view unavailable")
            return state.poses, None

    def module(name, **attrs):
        mod = ModuleType(name)
        mod.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, mod)
        return mod

    usd = module("omni.usd", get_context=lambda: SimpleNamespace(
        get_stage=lambda: SimpleNamespace(Traverse=lambda: iter(props))))
    module("omni", usd=usd)
    module("pxr", UsdGeom=SimpleNamespace(Xformable=Xformable),
           Usd=SimpleNamespace(TimeCode=SimpleNamespace(Default=lambda: None)),
           UsdPhysics=SimpleNamespace(RigidBodyAPI=object()))
    prims = module("isaacsim.core.prims", RigidPrim=RigidPrim)
    core = module("isaacsim.core", prims=prims)
    module("isaacsim", core=core)

    class Client:
        def __init__(self):
            self.bridge_globals = {}

        def request(self, packet):
            assert packet["op"] == "exec"
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                exec(packet["code"], self.bridge_globals)
            return {"ok": True, "stdout": output.getvalue()}

    return state, TruthPoseReader(Client(), ttl_s=0)


def test_truth_probe_reads_live_physics_without_authoring_and_reuses_views(probe_bridge):
    state, reader = probe_bridge
    assert reader.pose("pink cube") == [.17, .15, .04]
    assert state.writes == []
    assert state.authored_dynamic_reads == 0
    assert len(state.views) == 1
    state.poses = [[.3, -.2, .08]]
    assert reader.pose("pink cube") == [.3, -.2, .08]
    assert reader.pose("box") == [.3, -.2, .01]
    assert len(state.views) == 1
    assert state.writes == []


def test_failed_live_read_never_uses_authored_spawn_or_expired_pose(probe_bridge):
    state, reader = probe_bridge
    assert reader.pose("pink cube") == [.17, .15, .04]
    state.failed_read = True
    assert reader.pose("pink cube") is None
    assert state.authored_dynamic_reads == 0
    assert state.writes == []
    state.failed_read = False
    state.poses = [[.2, .1, .06]]
    assert reader.pose("pink cube") == [.2, .1, .06]
    assert len(state.views) == 2  # A failed view is replaced, not reused.
    assert state.writes == []
