"""Explicit mounted fixture authoring variants; no SDK or device discovery."""
from dataclasses import dataclass


@dataclass(frozen=True)
class SeatingRecipe:
    name: str
    center_xy_m: tuple[float, float]
    ik_margin_rad: float
    intersect_position_control_range: bool


LEGACY_RECIPE = "factory_m20_fixed_axis_v1"
MARGIN_RECIPE = "factory_m20_fixed_axis_margin_v2"
_RECIPES = (
    SeatingRecipe(LEGACY_RECIPE, (.24, 0.), .001, False),
    SeatingRecipe(MARGIN_RECIPE, (.23, 0.), .025, True),
)


def seating_recipe(name: str) -> SeatingRecipe:
    for recipe in _RECIPES:
        if name == recipe.name:
            return recipe
    raise ValueError("unknown fixed-axis Factory M20 authoring recipe")
