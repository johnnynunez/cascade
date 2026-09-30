"""Fit the audited Cocina_Asier_01 room around the calibrated task surface."""
from pathlib import Path

from pxr import Gf, UsdGeom, UsdLux

from own_kitchen import DEFAULT_COUNTER, _box, _material


def author_kitchen(stage, path="/Kitchen", *, counter=None):
    counter = DEFAULT_COUNTER if counter is None else counter
    if any(abs(float(counter[key][i]) - DEFAULT_COUNTER[key][i]) > 1e-9
           for key in ("min", "max") for i in range(3)):
        raise ValueError("Cocina Asier must retain the calibrated counter bounds and top at z=0")
    if stage.GetPrimAtPath(path):
        raise ValueError("Kitchen root already exists")
    root = UsdGeom.Xform.Define(stage, path).GetPrim()
    root.SetCustomDataByKey("design", "paai-cocina-asier-v1")
    root.SetCustomDataByKey("provenance", "Owner-authorized room; audited CC0 wood and original replacement surfaces")
    asset = Path(__file__).resolve().parent / "scene/cocina_asier/cocina_asier.usdc"
    if not asset.is_file():
        raise ValueError("Cocina Asier asset missing; run python scripts/kitchen_assets.py")
    architecture = stage.DefinePrim(path + "/Architecture", "Xform")
    architecture.GetReferences().AddReference(str(asset), "/CocinaAsier")
    # Physical closed blinds, with an opaque backing, fill the actual window.
    mat = _material(stage, path + "/WindowTreatment/Ivory", (.73, .70, .64), roughness=.64)
    _box(stage, path + "/WindowTreatment/Backing", (-2.041, .709, .867), (.01, 1.73, 1.09), mat)
    for pane, x, y, z, width, height in (
        ("L", -2.005, .2909, .86555, .75, .9395),
        ("R", -1.973, 1.12705, .86555, .7465, .8895),
    ):
        # Put each closed blind within the actual aperture, in front of the
        # pane but behind its frame/handle. An RTX transparent pane otherwise
        # still shows a dark reflection instead of the closed treatment.
        pane_name = "Plane_1" if pane == "L" else "Plane1"
        glass = stage.GetPrimAtPath(path + "/Architecture/Room/window__Window__window" + pane + "__" + pane_name)
        UsdGeom.Imageable(glass).CreateVisibilityAttr("invisible")
        _box(stage, path + f"/WindowTreatment/{pane}Backing", (x-.004, y, z), (.004, width, height), mat)
        for i in range(26):
            center = z-height/2 + height*(i+.5)/26
            _box(stage, path + f"/WindowTreatment/{pane}Slat{i}", (x, y, center),
                 (.010, width, height/26+.003), mat)
        _box(stage, path + f"/WindowTreatment/{pane}TopRail", (x+.005, y, z+height/2), (.018, width, .025), mat)
    # The warm analytic rig preserves the prior task illumination and requires
    # no photographed environment map. All source-export lights were removed.
    dome = UsdLux.DomeLight.Define(stage, path + "/Lighting/Ambient")
    dome.CreateIntensityAttr(650)
    dome.CreateColorAttr(Gf.Vec3f(1, .985, .955))
    ceiling = UsdLux.RectLight.Define(stage, path + "/Lighting/CeilingSoftbox")
    ceiling.CreateWidthAttr(3.6)
    ceiling.CreateHeightAttr(4.3)
    ceiling.CreateNormalizeAttr(True)
    ceiling.CreateIntensityAttr(22500)
    ceiling.CreateColorAttr(Gf.Vec3f(1, .978, .94))
    UsdGeom.Xformable(ceiling).AddTranslateOp().Set(Gf.Vec3d(.72, -.04, 1.705))
    from own_kitchen_props import author_bowl_of_oranges
    author_bowl_of_oranges(stage, path + "/DecorativeBowl", position=(.7514, .0106, 0), diameter=.3023)
    return root
