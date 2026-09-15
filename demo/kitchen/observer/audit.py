"""Offline audit of passive physics records. Tool/LLM success is never consulted."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


class VerifiedColliderGeometry:
    """Body-local baked points from one SHA-256-verified geometry artifact."""

    def __init__(self, *, vertices_m, body_name, receipt):
        self.vertices_m=np.asarray(vertices_m,dtype=float).copy()
        self.vertices_m.setflags(write=False)
        self.body_name=body_name
        self.receipt=receipt


def load_collider_geometry(path, *, expected_sha256, expected_scene_config_sha256=None):
    """Verify and load explicit authored collider geometry; no simulator reads."""
    path=Path(path).resolve()
    root=Path(__file__).resolve().parents[3]
    if not path.is_relative_to(root):
        raise ValueError("Collider geometry input must stay inside this checkout")
    raw=path.read_bytes()
    actual=hashlib.sha256(raw).hexdigest()
    if actual!=expected_sha256:
        raise ValueError("Collider geometry SHA-256 does not match the independently supplied digest")
    obj=json.loads(raw)
    if (obj.get('version')!=1 or obj.get('frame')!='body_local' or obj.get('units')!='m'
            or obj.get('collision_approximation')!='convexHull' or not obj.get('body_name')):
        raise ValueError("Unsupported collider geometry schema, frame, units or approximation")
    vertices=np.asarray(obj['vertices_m'],dtype=float)
    if (vertices.ndim!=2 or vertices.shape[1]!=3 or not 4<=len(vertices)<=10000
            or not np.isfinite(vertices).all() or np.linalg.matrix_rank(vertices-vertices[0])<3):
        raise ValueError("Collider points must be a finite three-dimensional Nx3 solid")
    baked=np.ascontiguousarray(vertices,dtype='<f4')
    if not np.array_equal(baked.astype(float),vertices):
        raise ValueError("Collider artifact must contain exact baked float32 point values")
    vertex_sha=hashlib.sha256(baked.tobytes()).hexdigest()
    if vertex_sha!=obj.get('vertices_f32_sha256'):
        raise ValueError("Collider vertex-array digest does not match the artifact")
    provenance=obj.get('provenance')
    if not isinstance(provenance,dict) or not provenance:
        raise ValueError("Collider geometry artifact must identify its provenance")
    if (expected_scene_config_sha256 is not None and
            provenance.get('scene_config_sha256')!=expected_scene_config_sha256):
        raise ValueError("Collider provenance does not match the expected active scene configuration")
    return VerifiedColliderGeometry(vertices_m=vertices,body_name=obj['body_name'],receipt={
        'method':'projected_baked_convex_vertices','file':str(path),'file_sha256':actual,
        'vertices_f32_sha256':vertex_sha,'vertex_count':len(vertices),'body_name':obj['body_name'],
        'frame':'body_local','units':'m','collision_approximation':'convexHull','provenance':provenance,
        'scene_config_binding_verified':expected_scene_config_sha256 is not None,
        'half_height_argument_used':False,'upright_tilt_limit_applied':False})


def projected_support_extents(vertices_m, orientations_wxyz):
    """Downward body-origin extent of rotated vertices, in metres.

    A linear height functional attains its minimum on a convex hull's vertices;
    no analytic ellipsoid or widened geometric tolerance is substituted.
    """
    vertices=np.asarray(vertices_m,dtype=float)
    quats=np.asarray(orientations_wxyz,dtype=float)
    if (vertices.ndim!=2 or vertices.shape[1]!=3 or quats.ndim!=2 or quats.shape[1]!=4
            or not np.isfinite(vertices).all() or not np.isfinite(quats).all()):
        raise ValueError("Invalid collider vertices or wxyz orientations")
    norms=np.linalg.norm(quats,axis=1)
    if not (np.abs(norms-1)<1e-3).all():
        raise ValueError("Support projection requires valid unit orientations")
    w,x,y,z=(quats/norms[:,None]).T
    world_z_rows=np.column_stack((2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)))
    return -(world_z_rows@vertices.T).min(axis=1)


def audit_records(records, *, object_name, target_xy, support_top_z, object_half_height,
                  xy_tolerance_m=.04, z_tolerance_m=.005, settle_sim_s=.5,
                  min_final_samples=4, max_linear_speed_m_s=.02, max_angular_speed_rad_s=.2,
                  max_position_spread_m=.005, min_open_fraction=.9, max_tilt_deg=5,
                  min_lift_m=.015, min_xy_displacement_m=.02, max_frame_age_s=2,
                  collider_geometry=None):
    """Require a stable, released object supported at the requested destination.

    support_top_z and object_half_height are collider geometry in the observer's
    robot-base frame. The centre-height rule is limited to upright (or inverted)
    symmetric objects. An explicitly verified collider_geometry instead projects
    the baked convex vertices with the actual quaternion, retaining tolerances.
    """
    result={"pass":False,"physics_pass":False,"camera_pass":False,"checks":{},"metrics":{},
            "object":object_name,"target_xy":list(target_xy),"support_top_z":support_top_z,
            "object_half_height":object_half_height,"scope":"physical destination and camera audit; chat transcript is separate evidence",
            "support_geometry":{"method":"upright_half_height","half_height_argument_used":True,
                                "upright_tilt_limit_applied":True},
            "criteria":{"xy_tolerance_m":xy_tolerance_m,"z_tolerance_m":z_tolerance_m,
                "settle_sim_s":settle_sim_s,"min_final_samples":min_final_samples,
                "max_linear_speed_m_s":max_linear_speed_m_s,"max_angular_speed_rad_s":max_angular_speed_rad_s,
                "max_position_spread_m":max_position_spread_m,"min_open_fraction":min_open_fraction,
                "max_tilt_deg":max_tilt_deg,"min_lift_m":min_lift_m,
                "min_xy_displacement_m":min_xy_displacement_m,"max_frame_age_s":max_frame_age_s},"limitations":[
                "Support test uses upright collider half-height; tilted objects fail the declared tilt bound.",
                "Actual open jaws and settled support pose establish release here; no contact-force sensor is claimed."]}
    checks=result["checks"]
    try:
        if collider_geometry is not None:
            if not isinstance(collider_geometry,VerifiedColliderGeometry) or collider_geometry.body_name!=object_name:
                raise ValueError("Explicit verified collider geometry must belong to the audited object")
            result['support_geometry']=dict(collider_geometry.receipt)
            result['limitations'][0]=("Support uses the recorded authored convex collider vertices transformed by the measured wxyz "
                "orientation. Engine cooking may simplify that collider; runtime cooked-hull equality is not claimed.")
        if (not np.isfinite([*target_xy,support_top_z,object_half_height,*result["criteria"].values()]).all()
                or object_half_height<=0 or len(target_xy)!=2 or xy_tolerance_m<=0 or xy_tolerance_m>.04
                or settle_sim_s<=0 or min_final_samples<2 or not 0<min_open_fraction<=1):
            raise ValueError("Invalid geometry, tolerance or release thresholds")
        samples=[r.get("physics",r) for r in records]
        if len(samples)<min_final_samples:
            raise ValueError("Insufficient physics samples")
        checks["physics_channel"]=all(s.get("channel")=="physics_tensor" for s in samples)
        checks["same_robot_and_engine"]=len({(s["robot_id"],s["engine"]) for s in samples})==1
        checks["same_observed_prop_set"]=len({tuple(sorted(s["props"])) for s in samples})==1
        clocks=np.asarray([s["sim_time"] for s in samples],dtype=float)
        steps=np.asarray([s["physics_step"] for s in samples],dtype=float)
        if not np.isfinite(clocks).all() or not np.isfinite(steps).all():
            raise ValueError("Non-finite physics clock")
        checks["physics_clock_advances"]=bool(np.isfinite(clocks).all() and np.isfinite(steps).all()
            and (np.diff(clocks)>=0).all() and (np.diff(steps)>=0).all()
            and clocks[-1]-clocks[0]>=settle_sim_s and steps[-1]>steps[0])
        # Fail if any observed robot/prop sample contains non-finite physical state.
        finite=True; joint_bounds=True
        for s in samples:
            q=np.asarray(s["q"],float);dq=np.asarray(s["dq"],float)
            lo=np.asarray(s["joint_lower"],float);hi=np.asarray(s["joint_upper"],float)
            margin=np.full(q.shape,.02)
            margin[np.asarray(s["gripper"]["indices"],int)]=.002
            finite &= bool(np.isfinite(q).all() and np.isfinite(dq).all())
            joint_bounds &= bool(q.shape==lo.shape==hi.shape==dq.shape and
                                 (q>=lo-margin).all() and (q<=hi+margin).all())
            for p in s["props"].values():
                for k in ("position_m","orientation_wxyz","linear_velocity_m_s","angular_velocity_rad_s"):
                    finite &= bool(np.isfinite(np.asarray(p[k],float)).all())
        checks["finite_all_observed_state"]=finite
        checks["actual_joints_within_limits"]=joint_bounds
        if not finite:
            raise ValueError("Non-finite observed physical state")
        # Include the immediately preceding sample so the audited trailing window
        # spans at least settle_sim_s, rather than silently accepting a short tail.
        start=max(0,int(np.searchsorted(clocks,clocks[-1]-settle_sim_s,side="right"))-1)
        tail=samples[start:]
        tail_times=clocks[start:]
        checks["final_window_long_enough"]=bool(len(tail)>=min_final_samples
            and len(np.unique(steps[start:]))>=min_final_samples and tail_times[-1]-tail_times[0]>=settle_sim_s-1e-9)
        props=[s["props"][object_name] for s in samples]
        poses=np.asarray([p["position_m"] for p in props],float)
        final_positions=poses[start:]
        velocities=np.asarray([p["linear_velocity_m_s"] for p in props[start:]],float)
        angular=np.asarray([p["angular_velocity_rad_s"] for p in props[start:]],float)
        quat=np.asarray([p["orientation_wxyz"] for p in props[start:]],float)
        quat_norm=np.linalg.norm(quat,axis=1)
        checks["valid_final_orientations"]=bool(np.isfinite(quat_norm).all() and
                                                (np.abs(quat_norm-1)<1e-3).all())
        normalized=quat/np.maximum(quat_norm[:,None],1e-12)
        local_z_dot_world_z=1-2*(normalized[:,1]**2+normalized[:,2]**2)
        tilt=np.degrees(np.arccos(np.clip(np.abs(local_z_dot_world_z),0,1)))
        if collider_geometry is None:
            checks["upright_support_geometry"]=bool((tilt<=max_tilt_deg).all())
            support_extents=np.full(len(tail),object_half_height)
        else:
            support_extents=projected_support_extents(collider_geometry.vertices_m,quat)
            checks['projected_support_geometry']=bool(np.isfinite(support_extents).all())
            result['support_geometry']['final_samples']=[{'sample_index':start+i,'sim_time':float(tail_times[i]),
                'downward_extent_m':float(extent),'expected_body_origin_z_m':float(support_top_z+extent),
                'observed_body_origin_z_m':float(final_positions[i,2]),
                'lowest_vertex_world_z_m':float(final_positions[i,2]-extent)} for i,extent in enumerate(support_extents)]
        errors=np.linalg.norm(final_positions[:,:2]-np.asarray(target_xy),axis=1)
        heights=np.abs(final_positions[:,2]-(support_top_z+support_extents))
        speeds=np.linalg.norm(velocities,axis=1)
        spins=np.linalg.norm(angular,axis=1)
        spread=np.linalg.norm(final_positions-final_positions[-1],axis=1)
        fractions=[]; gripper_consistent=True
        for s in tail:
            gi=np.asarray(s["gripper"]["indices"],int)
            if gi.shape!=(2,) or len(set(gi))!=2:
                raise ValueError("Expected two distinct actual gripper joint indices")
            aq=np.asarray(s["q"],float)[gi]
            alo=np.asarray(s["joint_lower"],float)[gi]
            ahi=np.asarray(s["joint_upper"],float)[gi]
            if not np.isfinite(ahi-alo).all() or not (ahi>alo).all():
                raise ValueError("Invalid gripper limits")
            frac=(aq-alo)/(ahi-alo)
            fractions.append(frac)
            gripper_consistent &= bool(np.allclose(aq,s["gripper"]["q"],atol=1e-8,rtol=0) and
                np.allclose(frac,s["gripper"]["open_fractions"],atol=1e-7,rtol=0))
        fractions=np.asarray(fractions,float)
        checks["gripper_readback_matches_actual_q"]=gripper_consistent
        checks["released_actual_jaws"]=bool(fractions.ndim==2 and fractions.shape[1]==2
                                            and np.isfinite(fractions).all() and (fractions>=min_open_fraction).all())
        checks["destination_xy"]=bool((errors<=xy_tolerance_m).all())
        checks["supported_z"]=bool((heights<=z_tolerance_m).all())
        checks["settled_linear_speed"]=bool((speeds<=max_linear_speed_m_s).all())
        checks["settled_angular_speed"]=bool((spins<=max_angular_speed_rad_s).all())
        checks["stable_final_position"]=bool((spread<=max_position_spread_m).all())
        lift=float(poses[:,2].max()-poses[0,2])
        displacement=float(np.linalg.norm(poses[-1,:2]-poses[0,:2]))
        checks["observed_lift"]=lift>=min_lift_m
        checks["observed_xy_displacement"]=displacement>=min_xy_displacement_m
        result["metrics"]={"samples":len(samples),"final_samples":len(tail),
            "simulation_elapsed_s":float(clocks[-1]-clocks[0]),"final_window_sim_s":float(tail_times[-1]-tail_times[0]),
            "lift_m":lift,"xy_displacement_m":displacement,"final_xy_error_m":float(errors[-1]),
            "max_final_xy_error_m":float(errors.max()),"max_final_support_z_error_m":float(heights.max()),
            "max_final_linear_speed_m_s":float(speeds.max()),"max_final_angular_speed_rad_s":float(spins.max()),
            "max_final_position_spread_m":float(spread.max()),"max_final_tilt_deg":float(tilt.max()),
            "min_final_actual_open_fraction":float(fractions.min())}
        result["physics_pass"]=all(checks.values())
        frames=[s.get("frame",{}) for s in samples]
        frame_available=all(f.get("available") for f in frames)
        camera_checks={"frames_available":frame_available}
        if frame_available:
            capture=np.asarray([f["capture_monotonic"] for f in frames],float)
            server=np.asarray([s["server_monotonic"] for s in samples],float)
            ages=server-capture
            camera_checks.update(
                frame_clock_advances=bool(np.isfinite(capture).all() and (np.diff(capture)>=0).all() and capture[-1]>capture[0]),
                frames_recent=bool((ages>=-.01).all() and (ages<=max_frame_age_s).all()),
                frame_robot_binding=all(f.get("robot_id")==s["robot_id"] and
                    f.get("producer_time_source")=="physics_loop_monotonic" for f,s in zip(frames,samples)),
                visual_changes_during_motion=len({f.get("jpeg_sha256") for f in frames})>1)
            result["metrics"].update(max_frame_age_s=float(ages.max()) if np.isfinite(ages).all() else None,
                unique_jpeg_hashes=len({f.get("jpeg_sha256") for f in frames}))
        result["camera_checks"]=camera_checks
        result["camera_pass"]=all(camera_checks.values())
        result["pass"]=result["physics_pass"] and result["camera_pass"]
        result["failed_checks"]=[k for k,v in checks.items() if not v]+[k for k,v in camera_checks.items() if not v]
    except (KeyError,ValueError,TypeError,IndexError) as exc:
        result["error"]=str(exc)
        result["failed_checks"]=[k for k,v in checks.items() if not v]+["malformed_or_insufficient_evidence"]
    # Preserve a writable failure receipt even for malformed numeric input.
    def json_finite(value):
        if isinstance(value,float) and not np.isfinite(value):
            return None
        if isinstance(value,dict):
            return {k:json_finite(v) for k,v in value.items()}
        if isinstance(value,(list,tuple)):
            return [json_finite(v) for v in value]
        return value
    return json_finite(result)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory",type=Path);p.add_argument("--object",required=True)
    p.add_argument("--target",type=float,nargs=2,required=True,metavar=("X","Y"))
    p.add_argument("--support-top-z",type=float,required=True);p.add_argument("--object-half-height",type=float,required=True)
    p.add_argument("--settle-sim-seconds",type=float,default=.5);p.add_argument("--output",type=Path)
    p.add_argument("--collider-vertices",type=Path,help="Optional explicit baked convex collider artifact")
    p.add_argument("--collider-sha256",help="Independently supplied SHA-256 of --collider-vertices")
    a=p.parse_args()
    if bool(a.collider_vertices)!=bool(a.collider_sha256):
        p.error("--collider-vertices and --collider-sha256 must be provided together")
    geometry=load_collider_geometry(a.collider_vertices,expected_sha256=a.collider_sha256) if a.collider_vertices else None
    records=[json.loads(line) for line in (a.directory/"samples.jsonl").read_text().splitlines() if line.strip()]
    result=audit_records(records,object_name=a.object,target_xy=a.target,support_top_z=a.support_top_z,
                         object_half_height=a.object_half_height,settle_sim_s=a.settle_sim_seconds,
                         collider_geometry=geometry)
    if (a.directory/"metadata.json").is_file():
        meta=json.loads((a.directory/"metadata.json").read_text())
        result["metadata"]={k:meta.get(k) for k in ("complete","code_sha256","observer_sha256","transcript_sha256","ping_before","ping_after")}
        if not meta.get("complete"):
            result["pass"]=False;result["failed_checks"].append("observation_incomplete")
        before,after=meta.get("ping_before",{}),meta.get("ping_after",{})
        if before.get("scene_config_sha256")!=after.get("scene_config_sha256"):
            result["pass"]=False;result["failed_checks"].append("scene_identity_changed")
        if geometry is not None:
            expected=geometry.receipt['provenance'].get('scene_config_sha256')
            binding=bool(expected and before.get('scene_config_sha256')==expected==after.get('scene_config_sha256'))
            result['support_geometry']['scene_config_binding_verified']=binding
            if not binding:
                result['pass']=False;result['failed_checks'].append('collider_scene_identity_not_proven')
    elif geometry is not None:
        result['pass']=False;result['failed_checks'].append('collider_scene_identity_not_proven')
    target=a.output or (a.directory/"audit.json")
    root=Path(__file__).resolve().parents[3]
    if not target.resolve().is_relative_to(root):
        raise ValueError("Audit output must stay inside this checkout")
    target.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps({k:result[k] for k in ("pass","physics_pass","camera_pass","failed_checks")}))
    return 0 if result["pass"] else 1


if __name__=="__main__":raise SystemExit(main())
