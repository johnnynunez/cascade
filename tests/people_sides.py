"""Test helper: a runtime whose rig declares no side of the table as a visitor side.

The shipped profiles treat every side as one where visitors may stand
(`safety.people.visitor_sides`), so the people guard refuses every throw
before it starts (tests/test_people_guard_runtime.py). Tests that pin the
throw's OWN mechanics (its grasp, its honest verdicts, its refusals before and
during the swing) run with no visitor side, the way a rig that measured its
surroundings free of people would.
"""

from __future__ import annotations

import dataclasses


def no_visitor_side(runtime) -> None:
    """Clear `safety.people.visitor_sides` on every arm harness of `runtime`."""
    for harness in runtime._arm_harnesses():
        if harness is None:
            continue
        limits = harness.limits
        people = dataclasses.replace(limits.people, visitor_sides=frozenset())
        harness.limits = dataclasses.replace(limits, people=people)
