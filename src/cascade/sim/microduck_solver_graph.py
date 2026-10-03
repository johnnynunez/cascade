"""Optional SDK solver capture; all BAM and admission checks remain on host.

Reviewed NewtonStage.step_sim warms once, captures simulate(dt) and launches
once; subsequent calls launch once. simulate copies state arrays instead of
swapping their objects in graph mode. Enable only after model preparation and
initial HOME, so no bootstrap graph retains pre-preparation buffers.
"""
from __future__ import annotations

from cascade.sim.microduck_newton import sha256

STAGE_SHA256 = '08352d2a0e767611461b7dfe75a06ab24a36605f0c806ddba03ff211b1bef47a'


class SolverGraphContract:
    def __init__(self, stage, *, enabled, wp, dt, source_path, sdk_recipe=None):
        if type(enabled) is not bool:
            raise ValueError('solver graph selection must be explicit boolean')
        if stage.cfg.use_cuda_graph or stage.graph is not None:
            raise RuntimeError('bootstrap graph must be disabled before model preparation')
        self.enabled, self.stage, self.dt = enabled, stage, dt
        self.sdk_recipe = sdk_recipe
        expected = STAGE_SHA256
        if sdk_recipe is not None:
            from .microduck_sdk import INTERNAL_SOURCE_SHA256, require_recipe
            require_recipe(sdk_recipe)
            expected = INTERNAL_SOURCE_SHA256['isaacsim.physics.newton.impl.newton_stage']
        self._captured = None
        self._bindings = []
        self.source_sha256 = None
        if not enabled:
            return
        if sha256(source_path) != expected:
            raise RuntimeError('solver graph requires the reviewed exact NewtonStage source')
        self.source_sha256 = expected
        # Initial allocation/copy only; no joint/pose writes in an episode.
        stage.state_temp = stage.model.state()
        stage.state_temp.assign(stage.state_0)
        stage._kernels_compiled = False
        stage.cfg.use_cuda_graph = True
        for name in ('state_0', 'state_1', 'state_temp', 'control', 'contacts'):
            obj = getattr(stage, name)
            if obj is None:
                raise RuntimeError('solver graph requires complete stable state/control/contact buffers')
            self._bindings.append((stage, name, obj))
            self._bindings.extend((obj, field, value) for field, value in vars(obj).items()
                                  if isinstance(value, wp.array))

    def check(self):
        stage = self.stage
        if stage.cfg.use_cuda_graph is not self.enabled:
            raise RuntimeError('solver graph mode changed')
        if not self.enabled:
            if stage.graph is not None:
                raise RuntimeError('unexpected graph in uncaptured solver mode')
            return
        if any(getattr(obj, key, None) is not value for obj, key, value in self._bindings):
            raise RuntimeError('solver graph state/control/contact buffers replaced')
        if stage.graph is not None:
            if stage._graph_capture_dt != self.dt:
                raise RuntimeError('solver graph captured a different timestep')
            if self.sdk_recipe is not None and stage._graph_capture_substeps != 1:
                raise RuntimeError('solver graph captured a different substep count')
            if self._captured is None:
                self._captured = stage.graph
        if self._captured is not None and stage.graph is not self._captured:
            raise RuntimeError('solver graph unexpectedly replaced or discarded')

    def telemetry(self):
        return {'enabled': self.enabled, 'captured': self._captured is not None,
                'stage_source_sha256': self.source_sha256,
                'captured_dt_s': self.dt if self._captured is not None else None,
                'host_bam_and_admission': True}
