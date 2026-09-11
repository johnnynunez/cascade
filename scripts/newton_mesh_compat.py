"""Keep Newton's external-contact meshes compilable in MuJoCo.

Newton supplies both contacts and body inertias. MuJoCo still computes an
unused solid-mesh inertia while compiling its bookkeeping geoms; microscopic
CoACD fragments can fail that calculation. Only those exact failures may use
shell inertia, with vertices/faces, collision masks and body inertias unchanged.
This is not a workaround for meshes used by MuJoCo's own contact generator.
"""
from __future__ import annotations

from contextvars import ContextVar
import math
import re


_VOLUME_ERROR = re.compile(r"mesh volume is too small: (.*?) \. Try setting inertia to shell")


_EXTERNAL_CONTACTS = ContextVar("cascade_newton_external_contacts", default=False)


def compile_newton_spec(spec, compiler=None, *args, external_contacts=False, emit=None, **kwargs):
    import mujoco

    compiler = compiler or mujoco.MjSpec.compile
    changed = []
    seen = set()
    try:
        while True:
            try:
                model = compiler(spec, *args, **kwargs)
            except ValueError as exc:
                match = _VOLUME_ERROR.search(str(exc))
                if match is None or match[1] in seen:
                    raise
                name = match[1]
                meshes = [mesh for mesh in spec.meshes if mesh.name == name]
                geoms = [geom for geom in spec.geoms if geom.meshname == name]
                if len(meshes) != 1 or not geoms:
                    raise
                if spec.compiler.inertiafromgeom == mujoco.mjtInertiaFromGeom.mjINERTIAFROMGEOM_TRUE:
                    raise  # Geometric inertia must not override Newton's explicit values.
                for geom in geoms:
                    body = geom.parent
                    if not external_contacts and (geom.contype or geom.conaffinity):
                        raise  # Native MuJoCo collision meshes are outside this contract.
                    if body.name != "world" and (
                        not body.explicitinertial or not math.isfinite(body.mass) or body.mass <= 0
                    ):
                        raise  # Shell inertia would change an inferred physical mass model.
                mesh = meshes[0]
                changed.append((mesh, mesh.inertia))
                seen.add(name)
                mesh.inertia = mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL
            else:
                if emit:
                    for mesh, _ in changed:
                        emit(mesh.name)
                return model
    except BaseException:
        for mesh, inertia in changed:
            mesh.inertia = inertia
        raise


def install_newton_mesh_guard(*, solver_class=None):
    """Scope the fallback to Newton conversions using external contacts.

    Collision filter masks remain populated even with external contacts, so
    the solver's actual mode, not its masks, establishes the contract.
    Context-local state prevents leakage into other threads/nested solvers.
    No third-party source file is edited.
    """
    import functools
    import mujoco

    if solver_class is None:
        from newton.solvers import SolverMuJoCo

        solver_class = SolverMuJoCo
    conversion = getattr(solver_class, "_convert_to_mjc", None)
    if not callable(conversion):
        raise RuntimeError("Unrecognized Newton MuJoCo conversion interface")

    if not getattr(conversion, "_cascade_external_contact_scope", False):
        @functools.wraps(conversion)
        def scoped(self, *args, **kwargs):
            token = _EXTERNAL_CONTACTS.set(getattr(self, "_use_mujoco_contacts", None) is False)
            try:
                return conversion(self, *args, **kwargs)
            finally:
                _EXTERNAL_CONTACTS.reset(token)

        scoped._cascade_external_contact_scope = True
        solver_class._convert_to_mjc = scoped

    original = mujoco.MjSpec.compile
    if getattr(original, "_cascade_newton_mesh_guard", False):
        return

    def report(name):
        print(f"[bridge] Newton mesh compiler: shell bookkeeping inertia for {name}; "
              "external contacts, vertices and explicit body inertia unchanged", flush=True)

    @functools.wraps(original)
    def guarded(spec, *args, **kwargs):
        if not _EXTERNAL_CONTACTS.get():
            return original(spec, *args, **kwargs)
        return compile_newton_spec(spec, original, *args, external_contacts=True, emit=report, **kwargs)

    guarded._cascade_newton_mesh_guard = True
    mujoco.MjSpec.compile = guarded

