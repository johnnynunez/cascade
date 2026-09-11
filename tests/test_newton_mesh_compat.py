"""Real MuJoCo compiler checks; no Isaac/GPU/hardware simulation claimed."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


def tiny_mesh_spec():
    mujoco = pytest.importorskip("mujoco")
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="explicit_body")
    body.mass = 1.0
    body.inertia = [0.01, 0.02, 0.02]
    body.explicitinertial = True
    vertices = np.array([[0, 0, 0], [1e-5, 0, 0], [0, 1e-5, 0], [0, 0, 1e-5]])
    faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
    mesh = spec.add_mesh(name="tiny", uservert=vertices.ravel(), userface=faces.ravel())
    geom = body.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname="tiny", contype=0, conaffinity=0)
    return spec, mesh, body, geom


def test_external_mesh_compiles_without_changing_geometry_or_body_inertia():
    spec, mesh, body, _ = tiny_mesh_spec()
    vertices = np.array(mesh.uservert, copy=True)
    faces = np.array(mesh.userface, copy=True)
    mass, inertia = body.mass, np.array(body.inertia, copy=True)
    with pytest.raises(ValueError, match="mesh volume is too small"):
        spec.compile()
    assert importlib.util.find_spec("newton_mesh_compat"), "missing guarded Newton compiler adapter"
    from newton_mesh_compat import compile_newton_spec

    model = compile_newton_spec(spec)

    assert model.body_mass[1] == mass
    np.testing.assert_array_equal(model.body_inertia[1], inertia)
    np.testing.assert_array_equal(mesh.uservert, vertices)
    np.testing.assert_array_equal(mesh.userface, faces)
    assert np.count_nonzero(model.geom_contype) == 0
    assert np.count_nonzero(model.geom_conaffinity) == 0


def test_installed_guard_is_idempotent_and_reaches_the_real_compiler(monkeypatch):
    import mujoco
    import newton_mesh_compat as compat

    class SolverBoundary:
        def __init__(self, native):
            self._use_mujoco_contacts = native

        def _convert_to_mjc(self, spec, *, solver=None):
            self.solver_argument = solver
            return spec.compile()

    original = mujoco.MjSpec.compile
    monkeypatch.setattr(mujoco.MjSpec, "compile", original)
    compat.install_newton_mesh_guard(solver_class=SolverBoundary)
    installed = mujoco.MjSpec.compile
    conversion = SolverBoundary._convert_to_mjc
    compat.install_newton_mesh_guard(solver_class=SolverBoundary)
    assert mujoco.MjSpec.compile is installed
    assert SolverBoundary._convert_to_mjc is conversion
    spec, mesh, body, geom = tiny_mesh_spec()
    geom.contype = geom.conaffinity = 1  # Newton preserves filter masks even for external contacts.
    solver = SolverBoundary(False)
    assert solver._convert_to_mjc(spec, solver="Newton").body_mass[1] == body.mass
    assert solver.solver_argument == "Newton"
    assert geom.contype == geom.conaffinity == 1
    mesh.inertia = mujoco.mjtMeshInertia.mjMESH_INERTIA_CONVEX
    with pytest.raises(ValueError, match="mesh volume is too small"):
        SolverBoundary(True)._convert_to_mjc(spec)
    with pytest.raises(ValueError, match="mesh volume is too small"):
        spec.compile()  # Scope must not leak into other users of MuJoCo.


def test_explicit_external_contacts_preserve_filter_masks():
    from newton_mesh_compat import compile_newton_spec

    spec, mesh, body, geom = tiny_mesh_spec()
    geom.contype = geom.conaffinity = 1
    vertices = np.array(mesh.uservert, copy=True)
    model = compile_newton_spec(spec, external_contacts=True)
    assert model.geom_contype.tolist() == [1]
    assert model.geom_conaffinity.tolist() == [1]
    assert model.body_mass[1] == body.mass
    np.testing.assert_array_equal(mesh.uservert, vertices)



@pytest.mark.parametrize("masks", [(1, 0), (0, 1)])
def test_native_mujoco_contact_meshes_are_not_reinterpreted(masks):
    from newton_mesh_compat import compile_newton_spec

    spec, mesh, _, geom = tiny_mesh_spec()
    geom.contype, geom.conaffinity = masks
    inertia = mesh.inertia
    with pytest.raises(ValueError, match="mesh volume is too small"):
        compile_newton_spec(spec)
    assert mesh.inertia == inertia


@pytest.mark.parametrize("infer", ["body", "compiler"])
def test_inferred_physical_inertia_is_not_changed(infer):
    import mujoco
    from newton_mesh_compat import compile_newton_spec

    spec, mesh, body, _ = tiny_mesh_spec()
    if infer == "body":
        body.explicitinertial = False
    else:
        spec.compiler.inertiafromgeom = mujoco.mjtInertiaFromGeom.mjINERTIAFROMGEOM_TRUE
    inertia = mesh.inertia
    with pytest.raises(ValueError, match="mesh volume is too small"):
        compile_newton_spec(spec)
    assert mesh.inertia == inertia


def test_unknown_later_failure_restores_prior_mesh_changes():
    import mujoco
    from newton_mesh_compat import compile_newton_spec

    spec, mesh, _, _ = tiny_mesh_spec()
    inertia = mesh.inertia
    calls = 0

    def compiler(spec):
        nonlocal calls
        calls += 1
        if calls == 1:
            return mujoco.MjSpec.compile(spec)
        raise ValueError("unrelated compiler failure")

    with pytest.raises(ValueError, match="unrelated compiler failure"):
        compile_newton_spec(spec, compiler)
    assert calls == 2
    assert mesh.inertia == inertia

