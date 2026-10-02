"""Optional motion candidates; runtime execution remains gated by SafeArm."""


class PlanningError(RuntimeError):
    """The requested planner is unavailable or did not produce a valid candidate."""


def make_motion_planner(config):
    """Construct an explicitly selected planner; never substitute another backend.

    The optional runtime binding also uses this factory. A returned plan alone
    is not authority to move a robot; see docs/CUMOTION.md.
    """
    if config.get("type") != "cumotion":
        raise PlanningError("motion planner type must explicitly be 'cumotion'")
    from .cumotion import CumotionPlanner

    return CumotionPlanner(config)
