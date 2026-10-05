"""Real initialization orchestration and OS signals, with inert SDK doubles."""
import os
import signal
from types import SimpleNamespace as NS
from unittest.mock import MagicMock
import sys

import pytest

from cascade.apps.signal_stop import SignalRequest, StopSignals
from cascade.sim.microduck_newton import KitNewtonBackend, synchronize_camera_authoring


@pytest.mark.parametrize('signum', [signal.SIGINT, signal.SIGTERM])
def test_authoring_update_unwinds_before_sensor_attachment(signum):
    native = NS(initialized=False, simulation_step_count=0, sim_time=0.)
    timeline = NS(is_stopped=lambda: True, get_current_time=lambda: 0.)
    manager = NS(get_num_physics_steps=lambda: 0, get_simulation_time=lambda: 0.)
    swallowed = []
    def update():
        try:
            os.kill(os.getpid(), signum)
        except SignalRequest:
            swallowed.append(True)
    with StopSignals() as signals:
        with pytest.raises(SignalRequest) as exc:
            synchronize_camera_authoring(NS(update=update), timeline, manager, native,
                                         checkpoint=lambda: signals.checkpoint(persistent=True))
        assert exc.value.signum == signum
    assert swallowed == [True]


@pytest.mark.parametrize('signum', [signal.SIGINT, signal.SIGTERM])
@pytest.mark.parametrize('site', ['camera', 'export', 'play'])
@pytest.mark.parametrize('integrator_profile', ['sdk-default', 'euler-v1'])
def test_initialize_fences_camera_and_play_callbacks(tmp_path, monkeypatch, signum, site, integrator_profile):
    """Execute _initialize itself; a consumed callback cannot reach the next phase."""
    import cascade.sim.microduck_newton as native
    modules = {}
    names = ['warp', 'newton', 'mujoco', 'mujoco_warp', 'omni', 'omni.usd', 'omni.timeline',
             'pxr', 'isaacsim', 'isaacsim.core', 'isaacsim.core.simulation_manager',
             'isaacsim.core.version', 'isaacsim.core.experimental',
             'isaacsim.core.experimental.utils', 'isaacsim.physics', 'isaacsim.physics.newton',
             'isaac_runtime', 'convert_microduck']
    for name in names:
        module = modules[name] = MagicMock(name=name)
        monkeypatch.setitem(sys.modules, name, module)
        if '.' in name:
            parent, attr = name.rsplit('.', 1)
            setattr(modules[parent], attr, module)
    stage = modules['omni.usd'].get_context.return_value.get_stage.return_value
    stage.GetUsedLayers.return_value = []
    monkeypatch.setattr(native, 'neutralize_asset_actuation', lambda stage, kind, root_path=None: [])
    import cascade.control.newton_bam as bam
    loaded = []
    monkeypatch.setattr(bam, 'load_pinned_bam', lambda root, sdk_recipe=None: loaded.append((root, sdk_recipe)))
    events = []
    from cascade.sim.microduck_integrator import contract
    prim = modules['pxr'].UsdPhysics.Scene.Define.return_value.GetPrim.return_value
    prim.GetPath.return_value = '/World/PhysicsScene'
    attr = prim.GetAttribute.return_value
    attr.GetTypeName.return_value = 'token'
    attr.Get.return_value = 'euler'
    schema_ready = False
    def setup_physics(*args, **kwargs):
        nonlocal schema_ready
        schema_ready = True  # The pinned SimulationManager attaches MjcSceneAPI here.
    def get_attribute(name):
        assert schema_ready and name == 'mjc:option:integrator'
        return attr
    modules['isaac_runtime'].setup_physics.side_effect = setup_physics
    prim.GetAttribute.side_effect = get_attribute
    backend = KitNewtonBackend(NS(out=tmp_path, device='cuda:0', integrator_profile=integrator_profile,
                                  bam_source_root=tmp_path / 'bam-sources'),
                              {'asset': 'inert.usda', 'bundle': str(tmp_path), 'receipt': {'outputs': []},
                               'integrator_contract':contract(integrator_profile)}, None)
    backend.app = MagicMock()
    backend.app._app.get_extension_manager.return_value.is_extension_enabled.return_value = False
    def callback(name):
        if integrator_profile == 'euler-v1':
            attr.Set.assert_called_once_with('euler')
        else:
            attr.Set.assert_not_called()
        events.append(name)
        if site == name:
            try:
                os.kill(os.getpid(), signum)
            except SignalRequest:
                events.append('swallowed')
    backend._create_camera = lambda stage: callback('camera')
    def export(path):
        from pathlib import Path
        Path(path).write_text('inert scene')
        callback('export')
    stage.GetRootLayer.return_value.Export.side_effect = export
    modules['isaacsim.core.experimental.utils'].app.play.side_effect = lambda **kw: callback('play')
    with StopSignals() as signals:
        backend.signals = signals
        with pytest.raises(SignalRequest) as exc:
            backend._initialize()
        assert exc.value.signum == signum
    assert events == {'camera': ['camera', 'swallowed'],
                      'export': ['camera', 'export', 'swallowed'],
                      'play': ['camera', 'export', 'play', 'swallowed']}[site]
    modules['isaacsim.physics.newton'].acquire_stage.assert_not_called()
    assert backend.ns is None
