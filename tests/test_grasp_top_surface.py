"""A top-only observation must not turn a tall tabletop prop into an edge pinch."""
import numpy as np
import pytest
from cascade.grasping.obb_grasp import plan_grasps_from_fix
from cascade.types import Detection, ObjectFix


@pytest.mark.parametrize("low,high,grasp_z", [
    (.05, .05, .0425),       # only the 50 mm top is visible
    (0, .05, .0425),         # full side observation of that same object
    (.06, .08, .077),        # resolved elevated bottom remains elevated
    (.004, .004, .005),      # thin object still respects the table clamp
])
def test_visible_height_and_support_prior(low, high, grasp_z):
    points = np.array([[x,y,z] for x in [.18,.22] for y in [.08,.12] for z in [low,high]])
    fix = ObjectFix("box", points.mean(0), points,
        Detection("box", .95, np.array([0,0,10,10])), np.array([.04,.04,high-low]), np.eye(3))
    grasps = plan_grasps_from_fix(fix, table_z=0, depth_fraction=.15)
    assert grasps
    assert all(g.position[2] == pytest.approx(grasp_z) for g in grasps)
