# SPDX-License-Identifier: Apache-2.0
"""Recursive agent harnessing (Harness², arXiv:2610.14224) for CASCADE's built-in agent.

Improves the agent WITHOUT touching model weights or any safety value: a Propose–Probe–Compose
step edits the advice text around the model (rules, tool guidance, triggered skills), executes
every candidate on the same task on a real CASCADE stack, selects blind, gates adoption on
CASCADE's own verifier and records evidence-backed lessons in a per-family library.

Entry points: :class:`Harnessing` (one step per task), :class:`AgentHarness` (the editable
surface), :class:`CascadeExecutor` (rollouts), ``scripts/harness2_improve.py`` (CLI).
"""
from .executor import CascadeExecutor, Rollout, TaskSpec
from .improver import ClaudeImprover, CodexImprover, OpenAICompatImprover, ScriptedImprover, make_improver
from .library import Library, scope_key
from .recursion import Harnessing, Ledger, StepResult
from .surface import LOCKED_TOOLS, AgentHarness, Skill, lint_text

__all__ = ["AgentHarness", "CascadeExecutor", "ClaudeImprover", "CodexImprover", "Harnessing", "LOCKED_TOOLS",
           "Ledger", "Library", "OpenAICompatImprover", "Rollout", "ScriptedImprover", "Skill", "StepResult",
           "TaskSpec", "lint_text", "make_improver", "scope_key"]
