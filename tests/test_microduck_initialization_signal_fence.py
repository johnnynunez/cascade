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
def test_initialize_fences_camera_and_play_callbacks(tmp_path, monkeypatch, signum, site):
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
    monkeypatch.setattr(native, 'disable_source_actuators', lambda stage: [])
    events = []
    backend = KitNewtonBackend(NS(out=tmp_path, device='cuda:0'),
                              {'asset': 'inert.usda', 'bundle': str(tmp_path), 'receipt': {'outputs': []}}, None)
    backend.app = MagicMock()
    backend.app._app.get_extension_manager.return_value.is_extension_enabled.return_value = False
    def callback(name):
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
