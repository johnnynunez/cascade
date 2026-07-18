"""Isaac Sim bridge: run the identical agentic demo against a simulated rig.

The sim is just another camera profile + arm profile (`--cameras isaac
--arm isaac`); SkillRuntime, safety harness, reflex tier and the livestream
dashboard are unchanged. See scripts/isaac_bridge.py for the sim-side
server (runs inside Isaac Sim's Python).
"""

from .bridge_client import BridgeClient, BridgeError

__all__ = ["BridgeClient", "BridgeError"]
