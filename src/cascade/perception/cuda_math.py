"""Strict CUDA tensor geometry for the GPU demo, with explicit device receipts.

Point clouds and masks stay on CUDA through perception. Only compact geometry,
labels, browser images and serialized memory cross back to the host. The public
CPU implementation remains the reference and is used outside strict GPU mode.
"""
from __future__ import annotations
from functools import lru_cache
import json
import os
from pathlib import Path
import threading
import time

_lock = threading.Lock()
_stages = {}
_linalg_init_lock = threading.Lock()
_linalg_ready_devices = set()


def enabled():
    return os.environ.get("CASCADE_REQUIRE_CUDA", "0") == "1"


def _initialize_cuda_linalg(torch, device):
    """Publish CUDA linalg readiness only after one serialized first use.

    PyTorch's lazy CUDA linalg dispatcher replaces its stubs on first use.
    Concurrent watcher/foreground calls can both enter the old stub and
    trigger ``lazy wrapper should be called at most once``. lru_cache alone
    does not serialize misses, so context() gates them with this lock.
    One CUDA eigh loads the shared linalg dispatch library, including the
    SVD/least-squares paths used below. Failed initialization stays failed
    for this caller; nothing falls back to a CPU decomposition.
    """
    with _linalg_init_lock:
        key = str(device)
        if key in _linalg_ready_devices:
            return
        probe = torch.eye(3, dtype=torch.float64, device=device)
        torch.linalg.eigh(probe)
        torch.cuda.synchronize(device)
        _linalg_ready_devices.add(key)


@lru_cache(maxsize=1)
def context():
    import torch
    from ..device import resolve_device
    device = torch.device(resolve_device("auto", what="perception tensor geometry"))
    if device.type != "cuda" or not torch.cuda.is_available() or not torch.version.cuda or torch.version.hip:
        raise RuntimeError("Perception tensor geometry requires NVIDIA CUDA; CPU fallback is forbidden")
    _initialize_cuda_linalg(torch, device)
    return torch, device


def tensor(value, *, dtype=None):
    torch, device = context()
    result = torch.as_tensor(value, dtype=dtype or torch.float64, device=device)
    if result.device.type != "cuda":
        raise RuntimeError("Perception tensor allocation did not use CUDA")
    return result


def record(stage, *values):
    torch, device = context()
    if not values or any(value.device.type != "cuda" for value in values):
        raise RuntimeError(f"Perception stage {stage} used a CPU tensor")
    with _lock:
        previous = _stages.get(stage, {})
        _stages[stage] = {"calls": previous.get("calls", 0) + 1,
            "devices": sorted({str(value.device) for value in values}),
            "shapes": [list(value.shape) for value in values],
            "last_monotonic": time.monotonic()}
        directory = os.environ.get("CASCADE_GPU_PERCEPTION_EVIDENCE_DIR")
        if directory and (not previous or _stages[stage]["calls"] % 25 == 0):
            from ..apps.process_owner import process_identity
            identity = process_identity(os.getpid())
            if identity is None:
                raise RuntimeError("Cannot bind CUDA perception evidence to this live process")
            destination = Path(directory)
            destination.mkdir(parents=True, exist_ok=True)
            payload = {"pid": os.getpid(), "required_cuda": True, "cpu_fallback_allowed": False,
                "birth": identity["birth"], "written_monotonic": time.monotonic(),
                "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
                "device": str(device), "gpu_name": torch.cuda.get_device_name(device),
                "cuda_allocated_bytes": torch.cuda.memory_allocated(device),
                "stages": dict(_stages)}
            path = destination / f"perception-{os.getpid()}.json"
            temp = path.with_suffix(".tmp")
            temp.write_text(json.dumps(payload, indent=2) + "\n")
            temp.replace(path)


def numpy_result(value):
    """Copy a completed small geometry result or image for its host consumer."""
    return value.detach().cpu().numpy()


def segment_mask(mask, shape):
    """Keep learned masks on CUDA while restoring camera resolution."""
    import torch.nn.functional as F
    torch, _ = context()
    mask = tensor(mask, dtype=torch.float32)
    if tuple(mask.shape) != tuple(shape):
        mask = F.interpolate(mask[None, None], size=shape, mode="nearest")[0, 0]
    result = mask.to(torch.bool)
    record("point_segmentation_mask", mask, result)
    return result


def sample_indices(size, count):
    torch, device = context()
    generator = torch.Generator(device=device).manual_seed(0)
    return torch.randperm(size, generator=generator, device=device)[:count]


def bbox_mask(shape, bbox, *, fraction=1., inclusive=False):
    torch, device = context()
    h, w = shape
    x0, y0, x1, y1 = [float(v) for v in bbox]
    if fraction != 1.:
        cx,cy=(x0+x1)/2,(y0+y1)/2
        hw,hh=(x1-x0)*fraction/2,(y1-y0)*fraction/2
        x0,x1,y0,y1=cx-hw,cx+hw,cy-hh,cy+hh
    mask=torch.zeros((h,w),dtype=torch.bool,device=device)
    edge=int(inclusive)
    mask[max(int(y0),0):min(int(y1)+edge,h),max(int(x0),0):min(int(x1)+edge,w)]=True
    record("bbox_mask", mask)
    return mask


def mask_fraction(mask):
    torch,_=context()
    mask=tensor(mask,dtype=torch.bool)
    return float(mask.to(torch.float64).mean()) if mask.numel() else 0.


def valid_mask(depth, mask):
    torch,_=context()
    depth=tensor(depth,dtype=torch.float32)
    valid=tensor(mask,dtype=torch.bool) & torch.isfinite(depth) & (depth>0)
    record("valid_depth_mask", depth, valid)
    return depth,valid


def median_depth(depth, u, v, radius):
    """Median valid sensor depth with the probe's original pixel window."""
    torch, _ = context()
    h, w = depth.shape[:2]
    patch = tensor(depth[max(0, v-radius):min(h, v+radius+1),
                         max(0, u-radius):min(w, u+radius+1)], dtype=torch.float32)
    valid = patch[torch.isfinite(patch) & (patch > 0)]
    record("probe_depth_reduction", patch, valid)
    return float(torch.quantile(valid, .5)) if valid.numel() else None


def depth_statistics(depth):
    """Reduce a full sensor frame on CUDA; only four summary scalars return."""
    torch, _ = context()
    depth = tensor(depth, dtype=torch.float32)
    valid = depth[torch.isfinite(depth) & (depth > 0)]
    record("scene_depth_statistics", depth, valid)
    return {"valid_fraction": round(valid.numel()/depth.numel(), 3),
        "min_m": round(float(valid.amin()), 3) if valid.numel() else None,
        "median_m": round(float(torch.quantile(valid, .5)), 3) if valid.numel() else None,
        "max_m": round(float(valid.amax()), 3) if valid.numel() else None}


def backproject(frame, mask, max_points=4000, depth_band=(.05,.95)):
    from ..types import SkillError
    torch,_=context()
    if not frame.has_depth:
        raise SkillError("no depth available; cannot lift mask to 3D")
    mask=tensor(mask,dtype=torch.bool)
    depth=tensor(frame.depth_m,dtype=torch.float32)
    ys,xs=torch.nonzero(mask,as_tuple=True)
    zs=depth[ys,xs]
    valid=zs>0
    ys,xs,zs=ys[valid],xs[valid],zs[valid]
    if not zs.numel():
        return tensor([]).reshape(0,3)
    band=torch.quantile(zs.to(torch.float64),tensor(depth_band))
    keep=(zs>=band[0]) & (zs<=band[1])
    ys,xs,zs=ys[keep],xs[keep],zs[keep]
    if zs.numel()>max_points:
        index=sample_indices(zs.numel(),max_points)
        ys,xs,zs=ys[index],xs[index],zs[index]
    K=tensor(frame.K)
    points=torch.stack(((xs-K[0,2])/K[0,0]*zs,(ys-K[1,2])/K[1,1]*zs,zs),dim=-1)
    record("depth_quantile_backprojection",mask,depth,points)
    return points


def transform(T, points):
    transform=tensor(T);cloud=tensor(points)
    result=cloud @ transform[:3,:3].T + transform[:3,3]
    record("cloud_transform",cloud,result)
    return result


def oriented_bbox(points):
    torch,_=context();points=tensor(points)
    centroid=points.mean(dim=0);centered=points-centroid
    covariance=centered.T @ centered / (points.shape[0]-1)
    eigenvalues,eigenvectors=torch.linalg.eigh(covariance)
    axes=eigenvectors[:,torch.argsort(eigenvalues,descending=True)]
    projection=centered @ axes
    lower,upper=projection.amin(dim=0),projection.amax(dim=0)
    center=centroid+axes @ ((lower+upper)/2)
    extents=upper-lower
    record("pca_oriented_bbox",points,covariance,axes,center,extents)
    return tuple(numpy_result(value) for value in (center,extents,axes))


def recentre(center, points, extents, camera):
    torch,_=context();points=tensor(points)
    if points.shape[0]<20:return center
    camera=tensor(camera)[:3]
    direction=points.mean(dim=0)-camera
    norm=torch.linalg.vector_norm(direction)
    if float(norm)<1e-6:return center
    direction=direction/norm;relative=points-camera
    projected=relative @ direction
    near_s=projected[projected<=torch.quantile(projected,.1)].mean()
    average=relative.mean(dim=0)
    lateral=average-(average @ direction)*direction
    near=camera+lateral+direction*near_s
    size=tensor(extents).amin().clamp(.01,.30)
    result=near+direction*(size/2)
    record("view_ray_recentering",points,projected,result)
    return numpy_result(result)


def refine_upright_cylinder(center, points):
    """Fit an observed upright cylindrical shell without a known radius/pose.

    The visible half of a can is thin along the viewing direction. Its
    smallest PCA extent is consequently not its diameter, so the generic
    view-ray heuristic can leave a jaw over the lid. Fit only side pixels,
    require an upright long axis and a well-conditioned, sufficiently broad
    circular arc, and preserve the old centre when evidence is insufficient.
    Height, grasp depth, control limits and physical verification are unchanged.
    """
    torch, _ = context()
    cloud = tensor(points)
    report = {"accepted": False, "device": str(cloud.device)}

    def reject(reason):
        return center, {**report, "reason": reason}

    if cloud.ndim != 2 or cloud.shape[1] != 3 or cloud.shape[0] < 128:
        return reject("insufficient_side_points")
    if not bool(torch.isfinite(cloud).all()):
        return reject("non_finite_points")
    relative = cloud - cloud.mean(dim=0)
    covariance = relative.T @ relative / (cloud.shape[0] - 1)
    _, axes = torch.linalg.eigh(covariance)
    if float(axes[2, -1].abs()) < .94:
        return reject("upright_long_axis_not_observed")
    heights = cloud[:, 2]
    band = torch.quantile(heights, tensor([.2, .8]))
    if float(band[1] - band[0]) < .025:
        return reject("insufficient_observed_height")
    side = cloud[(heights > band[0]) & (heights < band[1])]
    if side.shape[0] < 64:
        return reject("insufficient_side_points")
    origin = side[:, :2].mean(dim=0)
    xy = side[:, :2] - origin
    scale = xy.square().sum(dim=1).mean().sqrt()
    if float(scale) < 1e-5:
        return reject("degenerate_side_points")
    normalized = xy / scale
    matrix = torch.cat((2 * normalized, torch.ones_like(normalized[:, :1])), dim=1)
    singular = torch.linalg.svdvals(matrix)
    condition = singular[0] / singular[-1].clamp_min(1e-12)
    report["condition"] = float(condition)
    if float(condition) > 60:
        return reject("ill_conditioned_arc")
    fitted = torch.linalg.lstsq(matrix, normalized.square().sum(dim=1), driver="gels").solution
    radius_squared = fitted[2] + fitted[:2].square().sum()
    if not bool(torch.isfinite(fitted).all()) or float(radius_squared) <= 0:
        return reject("invalid_circle")
    circle_center = origin + fitted[:2] * scale
    radius = radius_squared.sqrt() * scale
    report["radius_m"] = float(radius)
    if not .005 <= float(radius) <= .15:
        return reject("unsupported_observed_radius")
    residual = torch.linalg.vector_norm(side[:, :2] - circle_center, dim=1) - radius
    rms = residual.square().mean().sqrt()
    p95 = torch.quantile(residual.abs(), .95)
    angles = torch.sort(torch.atan2(side[:, 1] - circle_center[1], side[:, 0] - circle_center[0])).values
    gaps = torch.diff(torch.cat((angles, angles[:1] + 2 * torch.pi)))
    arc = 2 * torch.pi - gaps.amax()
    report.update(radial_rms_m=float(rms), radial_p95_m=float(p95),
                  visible_arc_deg=float(arc * (180 / torch.pi)), side_points=int(side.shape[0]))
    record("cylinder_surface_fit", cloud, side, matrix, fitted, residual, angles)
    if float(arc) < 2 * torch.pi / 3:
        return reject("insufficient_visible_arc")
    tolerance = min(.003, float(radius) * .08)
    if float(rms) > tolerance or float(p95) > 2 * tolerance:
        return reject("non_cylindrical_or_noisy_surface")
    result = tensor(center).clone()
    if float(torch.linalg.vector_norm(circle_center - result[:2])) > float(radius) * 2:
        return reject("unsupported_center_correction")
    result[:2] = circle_center
    report["accepted"] = True
    return numpy_result(result), report


def hsv8(bgr):
    """OpenCV's 8-bit HSV integer equations evaluated entirely on CUDA."""
    torch,device=context();pixels=tensor(bgr,dtype=torch.int64)
    b,g,r=pixels.unbind(dim=-1)
    value=pixels.amax(dim=-1);diff=value-pixels.amin(dim=-1)
    x=torch.arange(256,dtype=torch.float64,device=device).clamp_min(1)
    sdiv=torch.round((255<<12)/x).to(torch.int64);sdiv[0]=0
    hdiv=torch.round((180<<12)/(6*x)).to(torch.int64);hdiv[0]=0
    saturation=(diff*sdiv[value]+(1<<11))>>12
    hue=torch.where(value==r,g-b,torch.where(value==g,b-r+2*diff,r-g+4*diff))
    hue=(hue*hdiv[diff]+(1<<11))>>12
    hue=torch.where(hue<0,hue+180,hue)
    result=torch.stack((hue,saturation,value),dim=-1)
    record("bgr_hsv",pixels,result)
    return result


def mask_color(bgr, mask, max_px=4000):
    from .colors import classify_hsv
    torch,_=context()
    if mask is None:return None
    selected=tensor(mask,dtype=torch.bool)
    ys,xs=torch.nonzero(selected,as_tuple=True)
    if not ys.numel():return None
    if ys.numel()>max_px:
        index=sample_indices(ys.numel(),max_px);ys,xs=ys[index],xs[index]
    pixels=tensor(bgr,dtype=torch.uint8)[ys,xs]
    hsv=hsv8(pixels).to(torch.float64)
    median=torch.quantile(hsv,.5,dim=0);hues=hsv[:,0]
    if float(hues.std(correction=0))>40 and float(((hues<20)|(hues>160)).to(torch.float64).mean())>.6:
        median[0]=(torch.quantile((hues+90)%180,.5)-90)%180
    record("mask_color_reduction",selected,pixels,hsv,median)
    return classify_hsv(*[float(v) for v in median])


def changed_fraction(a,b,threshold=25):
    torch,_=context();a=tensor(a);b=tensor(b)
    if a.shape!=b.shape or not a.numel():return 0.
    gray_a=a if a.ndim==2 else a[...,:3].mean(dim=2)
    gray_b=b if b.ndim==2 else b[...,:3].mean(dim=2)
    changed=(gray_a.to(torch.int16)-gray_b.to(torch.int16)).abs()>threshold
    result=changed.to(torch.float64).mean()
    record("visual_change_reduction",a,b,changed,result)
    return float(result)


def pixel_geometry(frame, mask, transform_matrix):
    from ..types import SkillError
    from .pixel_target import MIN_REGION_PX
    torch,_=context();depth,valid=valid_mask(frame.depth_m,mask)
    count=int(valid.sum())
    if count<MIN_REGION_PX:raise SkillError(f"only {count} pixels with valid depth in the mask; too few to localize an object")
    ys,xs=torch.nonzero(valid,as_tuple=True);zs=depth[ys,xs].to(torch.float64);K=tensor(frame.K)
    points=torch.stack(((xs-K[0,2])/K[0,0]*zs,(ys-K[1,2])/K[1,1]*zs,zs),dim=-1)
    cloud=transform(transform_matrix,points)
    center,extents,axes=oriented_bbox(cloud)
    bbox=torch.stack((xs.amin(),ys.amin(),xs.amax(),ys.amax())).to(torch.float32)
    record("pixel_mask_geometry",valid,points,cloud,bbox)
    return valid,cloud,center,extents,axes,numpy_result(bbox)


def remember_cloud(points):
    torch,_=context();points=tensor(points,dtype=torch.float32).reshape(-1,3)
    if points.shape[0]>384:points=points[sample_indices(points.shape[0],384)]
    result=points.clone();record("belief_cloud_sampling",points,result);return result


def shift_cloud(points, shift):
    points=tensor(points);result=points+tensor(shift)
    record("belief_cloud_translation",points,result);return result


def box_cloud(center, half, top_z, bottom_z):
    """Sample the existing box prior on CUDA; this is prior geometry, not truth."""
    torch, device = context()
    half = tensor(half)
    center = tensor(center)
    generator = torch.Generator(device=device).manual_seed(0)
    faces = []
    for axis in range(3):
        for sign in (-1., 1.):
            face = (torch.rand((60, 3), device=device, dtype=torch.float64, generator=generator)-.5)*2*half
            face[:, axis] = sign*half[axis]
            faces.append(face)
    points = torch.cat(faces)
    points[:, 2] = (points[:, 2]+(top_z+bottom_z)/2).clamp(bottom_z, top_z)
    points[:, :2] += center[:2]
    record("remembered_box_prior_sampling", points)
    return points


def rim_grasp_width(points,obj_top_z,band_m=.006,up=None):
    torch,_=context()
    if points is None or len(points)<60:return None
    points=tensor(points);axis=tensor([0.,0.,1.] if up is None else up)
    axis=axis/torch.linalg.vector_norm(axis).clamp_min(1e-15)
    origin=points.mean(dim=0);height=(points-origin)@axis
    band=points[(height-height.amax()).abs()<band_m]
    if len(band)<24:return None
    centre=band[:,:2].mean(dim=0);radius=torch.linalg.vector_norm(band[:,:2]-centre,dim=1)
    outer,inner=float(torch.quantile(radius,.99)),float(torch.quantile(radius,.01))
    record("rim_cloud_geometry",points,band,radius)
    if outer<1e-6 or inner/outer<.5 or outer-inner<=1e-4:return None
    wall=outer-inner;grasp_z=float(obj_top_z-.012)
    deep=points[(points[:,2]-grasp_z).abs()<.0025]
    if len(deep)>=24:
        radius=torch.linalg.vector_norm(deep[:,:2]-centre,dim=1)
        deep_outer,deep_inner=float(torch.quantile(radius,.99)),float(torch.quantile(radius,.01))
        if deep_outer>deep_inner>0:outer,inner=deep_outer,deep_inner;wall=outer-inner
    return wall,grasp_z,outer,inner,numpy_result(centre)
