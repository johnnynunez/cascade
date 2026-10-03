"""Explicit Factory SDK selections; never discover or fall back at runtime."""
import json
from pathlib import Path


INTERNAL_SDK_RECIPE = "isaacsim_48b2d951_newton_1_6_1rc1"
INTERNAL_PINS = Path(__file__).with_name("factory_internal_sdk_pins.json")


def validate_sdk_recipe(recipe):
    if recipe is not None and recipe != INTERNAL_SDK_RECIPE:
        raise ValueError("unknown explicit Factory SDK recipe")


def selected_pins(default, recipe=None):
    validate_sdk_recipe(recipe)
    return dict(default) if recipe is None else json.loads(INTERNAL_PINS.read_text())
