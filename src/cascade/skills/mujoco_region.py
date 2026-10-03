"""Optional geometric destination selection; no motion or release authority."""
from __future__ import annotations

import time

import numpy as np

from ..safety.trajectory import PLAN_BUDGET_S
from ..types import SkillError, make_transform
from . import mujoco_withdrawal
from .place_geometry import plan as plan_geometry


def select(runtime):
    from .runtime import _PostPlaceRetreatPlanError
    from ..sim.mujoco_placement import Region, state_digest
    from ..sim.truth import _match_label
    cfg = runtime.cfg.arm.get('mj_delivery_area')
    if cfg is None:
        return None
    deadline = time.monotonic()+PLAN_BUDGET_S
    region = Region.parse(cfg)
    if region.version == 2:
        bounds = np.asarray(region.bounds_xy_m)
        limits = runtime.arm.harness.limits
        if ((bounds[0] < limits.workspace_min[:2]).any()
                or (bounds[1] > limits.workspace_max[:2]).any()):
            raise _PostPlaceRetreatPlanError('configured region exceeds this arm base-frame workspace')
    raw = runtime.arm.raw
    from ..control.lazy_arm import LazyArm
    backend = raw.__dict__.get('_arm') if isinstance(raw, LazyArm) else raw
    if backend is None:
        raise _PostPlaceRetreatPlanError('configured region cannot activate an unmaterialized backend')
    world = getattr(backend, 'world', None)
    history = getattr(world, 'placement_history', None)
    if history is None or history.arm is not backend or history.region != region:
        raise _PostPlaceRetreatPlanError('configured region lacks its bound native geometry observer')
    generation = runtime.arm.harness._halt_generation
    def guard():
        runtime.arm.harness._check_halt_generation(generation)
        if runtime.arm.harness.estopped or runtime.arm.harness.halted is not None:
            raise _PostPlaceRetreatPlanError('region planning is stopped; keeping the grasp')
        if time.monotonic() >= deadline:
            raise _PostPlaceRetreatPlanError('region planning deadline expired; keeping the grasp')
    rejected = []
    with world.lock:
        history.guard()
        if region.version == 2 and history.prefix_faults:
            raise _PostPlaceRetreatPlanError('previous region placement was disturbed; keeping the grasp')
        snapshot = state_digest(world.data)
        epoch = history.epoch
        lower, upper = history.geometry()
        name = _match_label(runtime._held_det_label or runtime.held_object, {n: n for n in history.objects})
        if name is None:
            raise _PostPlaceRetreatPlanError('held object has no unambiguous region geometry')
        geoms = history.objects[name]
        body, _, _ = history.bodies[name]
        center = history.scratch.xpos[body].copy()
        lo, hi = lower[geoms].min(axis=0), upper[geoms].max(axis=0)
        offsets = np.array([lo[:2]-center[:2], hi[:2]-center[:2]])
        q = world.data.qpos[raw._qadr].copy()
        tcp = runtime.kin.fk(q)
        offset = center-tcp[:3, 3]
        clearance = float(runtime.arm.harness.limits.table_clearance)
        capacity_inventory = []
        capacity_layouts = {}
        if region.version == 2:
            from .placement_packing import pack_rows
            for obj, obj_geoms in history.objects.items():
                obj_lo, obj_hi = lower[obj_geoms].min(axis=0), upper[obj_geoms].max(axis=0)
                capacity_inventory.append({'id': history.bodies[obj][0], 'lower': obj_lo[:2], 'upper': obj_hi[:2],
                    'fixed': obj != name and (obj in history.confirmed_prefix or region.contains(obj_lo, obj_hi))})
            try:
                layouts = pack_rows(region.bounds_xy_m, capacity_inventory,
                                    margin=region.planning_margin_m, separation=clearance)
            except ValueError as exc:
                raise _PostPlaceRetreatPlanError('region capacity unavailable: '+str(exc)) from exc
            candidates = []
            for layout in layouts:
                planned = np.asarray(layout['footprints'][body])
                point = tuple((planned.mean(axis=0)+center[:2]-(lo[:2]+hi[:2])/2).tolist())
                if point not in capacity_layouts:
                    candidates.append(point)
                    capacity_layouts[point] = layout
            guard()
        else:
            candidates = region.candidates(offsets, clearance)
        if not candidates:
            raise _PostPlaceRetreatPlanError('measured payload footprint cannot fit in the configured region')
        gcfg = runtime.cfg.grasp
        table = float(runtime.cfg.safety.get('table_z', 0.))
        release_z = runtime._supported_release_height(table, table+float(gcfg.get('release_height_m', .05)))
        z_cap = float(gcfg.get('topdown_carry_z_max', gcfg.get('topdown_z_max', .15)))-.005
        release_z = min(release_z, z_cap)
        for x, y in candidates:
            guard()
            target = np.array([x-offset[0], y-offset[1], release_z])
            try:
                geometry = plan_geometry(runtime, q, target, x=x, y=y, release_z=release_z, z_cap=z_cap)
                preview = mujoco_withdrawal.prepare(runtime, make_transform(geometry.rotation, target),
                    carry_goals=([geometry.lift.q] if geometry.lift is not None else [])
                                + [geometry.pre.q, geometry.low.q], deadline=deadline)
                if preview is None:
                    raise SkillError('region requires the configured release/home collision planner')
                rotation = preview.data.geom_xmat.reshape(-1, 3, 3)
                centers = preview.data.geom_xpos + np.einsum('gij,gj->gi', rotation, preview.model.geom_aabb[:, :3])
                half = np.einsum('gij,gj->gi', np.abs(rotation), preview.model.geom_aabb[:, 3:])
                projected_lo, projected_hi = (centers-half)[geoms].min(axis=0), (centers+half)[geoms].max(axis=0)
                if not region.contains(projected_lo, projected_hi):
                    raise SkillError('predicted oriented footprint leaves the region')
                final_capacity = None
                if region.version == 2:
                    bounds = np.asarray(region.bounds_xy_m)
                    margin = region.planning_margin_m
                    if ((projected_lo[:2] < bounds[0]+margin).any()
                            or (projected_hi[:2] > bounds[1]-margin).any()):
                        raise SkillError('predicted oriented footprint violates planned boundary margin')
                    inventory = [dict(obj) for obj in capacity_inventory]
                    for obj in inventory:
                        if obj['id'] == body:
                            obj.update(lower=projected_lo[:2], upper=projected_hi[:2], fixed=True)
                    residual = pack_rows(region.bounds_xy_m, inventory, margin=margin, separation=clearance)
                    if not residual:
                        raise SkillError('oriented release leaves insufficient whole-inventory capacity')
                    final_capacity = {'scope': 'all free bodies; current measured footprint reservations',
                        'planning_margin_m': margin, 'bodies': {str(history.bodies[n][0]): n for n in history.objects},
                        'initial_layout': capacity_layouts[(x, y)], 'after_oriented_release': residual[0],
                        'future_task_admission': False}
                for other, other_geoms in history.objects.items():
                    if other == name:
                        continue
                    other_lo, other_hi = lower[other_geoms].min(axis=0), upper[other_geoms].max(axis=0)
                    gap = np.maximum(other_lo[:2]-projected_hi[:2], projected_lo[:2]-other_hi[:2]).max()
                    if gap < clearance:
                        raise SkillError('destination does not preserve physical object separation')
                guard()
                history.guard()
                if history.epoch != epoch or state_digest(world.data) != snapshot or time.monotonic() >= deadline:
                    raise SkillError('region preview changed identity/state or exceeded its deadline')
                return {'destination': region.name, 'destination_kind': 'configured_region',
                        'region': region.as_dict(), 'target': [x, y], 'model_sha256': history.identity,
                        'epoch': epoch, 'generation': generation, 'physical_task_verdict': False,
                        'planning_method': ('interior row capacity and existing full carry/release/home checks'
                            if region.version == 2 else 'measured footprint edge packing and existing full carry/release/home checks'),
                        'capacity': final_capacity,
                        'object_separation_m': clearance, 'rejected_candidates': rejected}
            except SkillError as exc:
                rejected.append({'target': [x, y], 'reason': str(exc)})
    raise _PostPlaceRetreatPlanError('no feasible unoccupied destination in region: '+repr(rejected))


def explicit_context(arm, label, args):
    """Registry-only lookup for a top-level exact-point request, not nested place."""
    from ..control.lazy_arm import LazyArm
    from ..control.mujoco_arm import MujocoArm
    from ..sim.truth import _match_label
    raw = arm.raw
    backend = raw.__dict__.get('_arm') if isinstance(raw, LazyArm) else raw
    if not isinstance(backend, MujocoArm) or backend._engine is None:
        return None
    history = getattr(backend.world, 'placement_history', None)
    if history is None or history.arm is not backend:
        return None
    with history.world.lock:
        history.guard()
        target = np.array([args.get('x'), args.get('y')], dtype=float)
        if target.shape != (2,) or not np.isfinite(target).all():
            return None
        bounds = np.asarray(history.region.bounds_xy_m)
        if (target >= bounds[0]).all() and (target <= bounds[1]).all():
            return None
        name = _match_label(label or '', {n: n for n in history.confirmed_prefix})
        if name is None:
            return None
        with arm.harness._observation_lock:
            if arm.harness.estopped or arm.harness.halted is not None:
                raise SkillError('explicit region-obligation request is stopped')
            generation = arm.harness._halt_generation
            cancellation_token = arm.harness._observation_cancel_generation
        return history, {'object': name, 'target_xy_m': target.tolist(), 'epoch': history.epoch,
                         'model_sha256': history.identity, 'request_step': history.step,
                         'request_final_time_s': float(history.world.data.time), 'tool': 'place_at',
                         'generation': generation, 'cancellation_token': cancellation_token}, arm.harness


def finish_explicit(context, result):
    if context is None:
        return
    history, request, harness = context
    try:
        result['region_obligation'] = history.retire_explicit(request, result, harness=harness)
    except Exception as exc:
        result['region_obligation'] = {'retired': False, 'reason': f'{type(exc).__name__}: {exc}'}
