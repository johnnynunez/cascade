"""Lab-only PhysX smoke of the Isaac Lab MicroDuck USD under Isaac Sim 6.2 (headless).

Loads the admitted bundle USD under /World/MicroDuck with a ground plane, selects the PhysX
engine, plays N physics steps at 0.005 s and records the articulation layout and the root
trajectory. The USD's zero-gain drives are left untouched, so the robot is unactuated and
simply settles/collapses: this probes that PhysX parses and steps the asset, nothing more.
No BAM, no policy, no CASCADE owner, no locomotion or physical claim."""
import json, sys, time
from pathlib import Path
from isaacsim import SimulationApp
BUNDLE = Path('/home/johnny/Projects/demo/cascade-lab/MICRODUCK/external/isaaclab-usd-allcollisions-v1')
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent / 'physx-smoke-01'
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 400
OUT.mkdir(parents=True, exist_ok=False)
app = SimulationApp({"headless": True, "physics_gpu": 0})
report = {'physical_acceptance': False, 'engine': 'physx', 'bundle_receipt_sha256': (BUNDLE / 'receipt.sha256').read_text().strip(),
          'asset': str(BUNDLE / 'usd/microduck_allcollisions.usd'), 'steps_requested': STEPS, 'errors': []}
try:
    import omni.usd, omni.timeline, carb
    from pxr import Gf, UsdGeom, UsdPhysics, PhysxSchema
    from isaacsim.core.simulation_manager import SimulationManager as SM
    from isaacsim.core.version import get_version
    report['isaacsim_version'] = get_version()
    SM.switch_physics_engine('physx')
    omni.usd.get_context().new_stage()
    stage = omni.usd.get_context().get_stage()
    UsdGeom.SetStageMetersPerUnit(stage, 1.); UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.Xform.Define(stage, '/World')
    root = UsdGeom.Xform.Define(stage, '/World/MicroDuck').GetPrim()
    root.GetReferences().AddReference(str(BUNDLE / 'usd/microduck_allcollisions.usd'))
    actuators = [p for p in stage.Traverse() if p.GetTypeName() == 'NewtonActuator']
    for p in actuators:
        p.SetActive(False)   # Newton-only prims; PhysX has no consumer for them
    report['newton_actuator_prims_deactivated'] = len(actuators)
    ground = UsdGeom.Plane.Define(stage, '/World/Ground') if hasattr(UsdGeom, 'Plane') else None
    import omni.physx.scripts.physicsUtils as pu
    pu.add_ground_plane(stage, '/World/GroundPlane', 'Z', 25.0, Gf.Vec3f(0.0), Gf.Vec3f(0.5))
    scene = UsdPhysics.Scene.Define(stage, '/World/PhysicsScene')
    scene.CreateGravityDirectionAttr(Gf.Vec3f(0., 0., -1.)); scene.CreateGravityMagnitudeAttr(9.81)
    physx_scene = PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim())
    physx_scene.CreateTimeStepsPerSecondAttr(200)
    physx_scene.CreateEnableGPUDynamicsAttr(True)
    joints = [p for p in stage.Traverse() if p.IsA(UsdPhysics.RevoluteJoint) and str(p.GetPath()).startswith('/World/MicroDuck/')]
    roots = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
    report['revolute_joints'] = len(joints); report['articulation_roots'] = [str(p.GetPath()) for p in roots]
    report['drive_sample'] = {k: UsdPhysics.DriveAPI.Get(joints[0], 'angular').GetPrim().GetAttribute(k).Get() for k in ('drive:angular:physics:stiffness', 'drive:angular:physics:damping', 'drive:angular:physics:maxForce')}
    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    app.update()
    from isaacsim.core.prims import Articulation
    art = Articulation('/World/MicroDuck/Geometry/trunk_base')
    trajectory = []
    xform_cache = UsdGeom.XformCache()
    trunk = stage.GetPrimAtPath('/World/MicroDuck/Geometry/trunk_base')
    t0 = time.monotonic()
    for i in range(STEPS):
        app.update()
        if i % 20 == 0 or i == STEPS - 1:
            xform_cache.Clear()
            m = xform_cache.GetLocalToWorldTransform(trunk)
            trajectory.append({'update': i, 'root_z_m': float(m.ExtractTranslation()[2])})
    report['wall_s'] = time.monotonic() - t0
    try:
        report['articulation'] = {'dof_count': int(art.num_dof), 'dof_names': list(art.dof_names)} if hasattr(art, 'num_dof') else 'unavailable'
    except Exception as exc:
        report['articulation'] = f'unavailable: {type(exc).__name__}: {exc}'
    report['root_trajectory'] = trajectory
    report['steps_completed'] = STEPS
    timeline.stop()
except Exception as exc:
    import traceback
    report['errors'].append(f'{type(exc).__name__}: {exc}'); report['traceback'] = traceback.format_exc()[-2000:]
finally:
    (OUT / 'physx-smoke.json').write_text(json.dumps(report, indent=1, default=str) + '\n')
    print('PHYSX_SMOKE', json.dumps({k: report.get(k) for k in ('errors', 'revolute_joints', 'articulation_roots', 'steps_completed', 'wall_s')}, default=str)[:600])
    app.close()
