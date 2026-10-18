"""Backend-neutral jaw-stage rules: a close never opens the jaws.

Shared by the runtime's two-stage close (skills/runtime.py) and the reBot
drivers' own two-stage closes (rebot_rs_arm.py, rebot_rs_mb_arm.py,
robstride.close_two_stage_capped). Positions are in the profile's units,
`open_pos` .. `closed_pos` either way round.
"""
from __future__ import annotations


def opens_past(target: float, reading: float | None, open_pos: float, closed_pos: float) -> bool:
    """`target` is more open than `reading` (two jaw positions in the
    profile's units, `open_pos` .. `closed_pos` either way round). A close
    stage that does is never sent: from the reBot's stage-2 seat (0.93 rad)
    the rigid close's scout stage (3.1 rad) would open the jaws on whatever
    they hold first (#379 review A). None (no reading) is not judged here:
    each caller decides what unknown jaws mean. Polarity-aware like
    `robstride.clamp_to_travel`; a jaw with no travel (open == closed) never opens."""
    if reading is None:
        return False
    return (float(target) - float(reading)) * (float(open_pos) - float(closed_pos)) > 0.0


def closing_stages(stages, reading: float | None, open_pos: float, closed_pos: float,
                   log=None) -> tuple:
    """The `stages` of a two-stage close that are sent, each a tuple whose
    first item is the stage's jaw target: a stage more open than `reading`
    (the jaws when the close starts) or than the stage before it is skipped
    (`log` gets one line for it), so a close never opens the jaws. From
    open jaws every stage stays, exactly as before. With no reading (None)
    the first stage stays too: the caller decides what unknown jaws mean
    (the RS gripper guard references them by closing first and refuses an
    opening it cannot read, rs_gripper.py)."""
    kept, limit = [], reading
    for stage in stages:
        target = stage[0]
        if limit is not None and opens_past(target, limit, open_pos, closed_pos):
            if log is not None:
                log(f"close stage to {float(target):.3f} skipped: the jaws are already at "
                    f"{float(limit):.3f}, and a close never opens them")
            continue
        kept.append(stage)
        limit = target
    return tuple(kept)
