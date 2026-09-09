"""Warp kernels for the `warp` occupancy backend (dense projective TSDF with
carving + exact separable Euclidean distance transform).

Kept in a module WITHOUT `from __future__ import annotations`: Warp resolves
kernel argument annotations by evaluating them, and stringified annotations
make `wp` unresolvable at that point. Importing this module requires
`warp-lang`; occupancy_backends imports it lazily.
"""

import warp as wp

wp.config.quiet = True


@wp.kernel
def k_integrate(depth: wp.array2d(dtype=wp.float32), width: int, height: int,
                fx: float, fy: float, cx: float, cy: float,
                R_cb: wp.mat33, t_cb: wp.vec3, origin: wp.vec3, voxel: float,
                trunc: float, max_w: float,
                tsdf: wp.array3d(dtype=wp.float32), weight: wp.array3d(dtype=wp.float32)):
    """One thread per voxel: project the voxel centre into the depth image and
    fuse the truncated signed distance. Voxels between camera and surface get
    +trunc (FREE) -- that is the carving; voxels behind the surface are left
    untouched (unknown)."""
    i, j, k = wp.tid()
    p_base = origin + wp.vec3(float(i), float(j), float(k)) * voxel
    p_cam = R_cb * p_base + t_cb
    z = p_cam[2]
    if z <= 0.05:
        return
    u = fx * p_cam[0] / z + cx
    v = fy * p_cam[1] / z + cy
    ui = int(wp.round(u))
    vi = int(wp.round(v))
    if ui < 0 or vi < 0 or ui >= width or vi >= height:
        return
    d = depth[vi, ui]
    if d <= 0.0:
        return
    sdf = d - z
    if sdf < -trunc:
        return
    sdf = wp.min(sdf, trunc)
    w_old = weight[i, j, k]
    tsdf[i, j, k] = (tsdf[i, j, k] * w_old + sdf) / (w_old + 1.0)
    weight[i, j, k] = wp.min(w_old + 1.0, max_w)


@wp.kernel
def k_seed(tsdf: wp.array3d(dtype=wp.float32), weight: wp.array3d(dtype=wp.float32),
           big: float, out: wp.array3d(dtype=wp.float32)):
    """Squared-distance seed: 0 on occupied voxels (observed and tsdf <= 0),
    `big` elsewhere."""
    i, j, k = wp.tid()
    if weight[i, j, k] > 0.0 and tsdf[i, j, k] <= 0.0:
        out[i, j, k] = 0.0
    else:
        out[i, j, k] = big


# 1-D squared EDT along each axis. Brute force O(n^2) per line: n <= ~200
# for a tabletop grid, no per-thread scratch memory needed, and the three
# passes compose into the exact 3-D squared Euclidean distance (the
# separability of min_p f(p) + (q-p)^2 over axes).

@wp.kernel
def k_edt_x(src: wp.array3d(dtype=wp.float32), dst: wp.array3d(dtype=wp.float32),
            nx: int, big: float):
    j, k = wp.tid()
    for q in range(nx):
        best = big
        for p in range(nx):
            f = src[p, j, k]
            if f < big:
                dq = float(q - p)
                val = f + dq * dq
                if val < best:
                    best = val
        dst[q, j, k] = best


@wp.kernel
def k_edt_y(src: wp.array3d(dtype=wp.float32), dst: wp.array3d(dtype=wp.float32),
            ny: int, big: float):
    i, k = wp.tid()
    for q in range(ny):
        best = big
        for p in range(ny):
            f = src[i, p, k]
            if f < big:
                dq = float(q - p)
                val = f + dq * dq
                if val < best:
                    best = val
        dst[i, q, k] = best


@wp.kernel
def k_edt_z(src: wp.array3d(dtype=wp.float32), dst: wp.array3d(dtype=wp.float32),
            nz: int, big: float):
    i, j = wp.tid()
    for q in range(nz):
        best = big
        for p in range(nz):
            f = src[i, j, p]
            if f < big:
                dq = float(q - p)
                val = f + dq * dq
                if val < best:
                    best = val
        dst[i, j, q] = best
