"""Which environment variables reach the MCP server a registration writes (B63).

A stdio MCP host (OpenClaw, Hermes, Codex, Claude) starts
``python -m cascade.apps.mcp_server`` with the ``env`` its registration names,
not with the environment of the shell that registered it. A runtime switch an
operator exported -- ``CASCADE_GRASP_EXECUTOR=vla``, a private
``CASCADE_VLA_PORT``, ``CASCADE_BOOTH=1`` -- therefore reaches the robot runtime
only if the registration copies it. Before B63 ``scripts/launch.sh`` copied a
fixed list of 18 names and ``scripts/setup_agents.py`` its own (four device
names plus every ``CASCADE_*_PORT`` / ``_HOST``), so
``CASCADE_GRASP_EXECUTOR=vla ./run.sh`` registered a server on the analytic
executor while the launcher's own runtime check, which inherits the shell,
saw ``vla``.

This module is the ONE list both registrations read (`forwarded_env`). Every
``CASCADE_*`` variable the runtime (``src/cascade``) reads is either

* in `FORWARDED`: copied VERBATIM when it is present in the registering shell
  (an empty value stays empty, exactly what the server would have read there),
  never invented when absent; a value the registration derives from its own
  flags still wins (launch.sh ``--occupancy none`` / ``--graspgenx none`` /
  ``--headless``, setup_agents ``--env`` / ``--detect-classes`` /
  ``--hide-tools``); or
* in `NOT_FORWARDED`, with a category from `CATEGORIES` and the reason.

``tests/test_mcp_env_forwarding.py`` scans ``src/cascade`` and fails when the
runtime reads a ``CASCADE_*`` variable listed in neither, or when an entry
here is no longer read. Credentials are never forwarded: a persisted MCP
config (``~/.hermes/config.yaml``, ``.mcp.json``, OpenClaw's profile) must not
hold one.

Stdlib only, no package-relative imports: launch.sh's registration heredoc
and setup_agents.py (under a bare python3) both import it.
"""
from __future__ import annotations

import os
from collections.abc import Iterable, Mapping

#: forwarded verbatim when present, in this order (name -> what it switches).
#: The first 18 are launch.sh's pre-B63 list in its order, so a registration
#: that sets none of the others is byte-identical to before.
FORWARDED: dict[str, str] = {
    "CASCADE_GRASP_MEMORY_PATH": "grasp-outcome memory store (private per profile / test)",
    "CASCADE_ENVELOPE_PATH": "learned envelope store",
    "CASCADE_BELIEFS_PATH": "persisted belief store",
    "CASCADE_BELIEFS": "belief persistence on/off",
    "CASCADE_GRASP_BACKEND": "grasp planner (obb|graspgenx|hug|camera_frame)",
    "CASCADE_GRASPGENX_PORT": "GraspGen-X sidecar port (B34)",
    "CASCADE_GRASPGENX_HOST": "GraspGen-X sidecar host (B41)",
    "CASCADE_BRIDGE_PORT": "Isaac bridge port (B34)",
    "CASCADE_OCCUPANCY_PORT": "occupancy sidecar port (B34)",
    "CASCADE_HUG_PORT": "HUG sidecar port (B41)",
    "CASCADE_HUG_HOST": "HUG sidecar host (B41)",
    "CASCADE_GRASP_EVIDENCE_DIR": "grasp evidence capture directory",
    "CASCADE_OBSERVED_FINGER_GATE": "observed-finger closing gate on/off",
    "CASCADE_KITCHEN_CAMERA_RENDERER": "kitchen camera renderer (isaac|ovrtx)",
    "CUDA_VISIBLE_DEVICES": "CUDA device visibility (empty = no GPU, kept verbatim)",
    "CUDA_DEVICE_ORDER": "CUDA ordinal mapping",
    "CASCADE_DEVICE": "model device for every profile",
    "CASCADE_REQUIRE_CUDA": "fail instead of falling back off CUDA",
    # ---- forwarded since B63 ----
    "CASCADE_OCCUPANCY": "occupancy map on/off (launch.sh --occupancy none still writes 0)",
    "CASCADE_GRASP_EXECUTOR": "grasp executor (analytic|vla, B49)",
    "CASCADE_VLA_PORT": "VLA policy server port (B49)",
    "CASCADE_BOOTH": "booth tuning overlay (configs/booth.yaml)",
    "CASCADE_DETECTOR_MODEL": "detector weights (launch.sh writes its own resolution first)",
    "CASCADE_DETECT_CLASSES": "detector vocabulary (--detect-classes wins)",
    "CASCADE_HIDE_TOOLS": "tools delisted from the catalog (--hide-tools wins)",
    "CASCADE_STREAM": "dashboard stream mode (0|eager|lazy|off)",
    "CASCADE_STREAM_PORT": "dashboard port",
    "CASCADE_VIEW": "camera viewer window (launch.sh derives it for sim runs)",
    "CASCADE_MJ_VIEW": "MuJoCo viewer window (launch.sh derives it for MuJoCo runs)",
    "CASCADE_PREWARM": "perception pre-warm at server start",
    "CASCADE_EXTERNAL_VIEW_URL": "dashboard link to an external live view",
    "CASCADE_EPISODIC": "episodic index persistence on/off (B43)",
    "CASCADE_EPISODIC_PATH": "episodic index store",
    "CASCADE_PROGRAMS": "programs tier on/off (B42)",
    "CASCADE_PROGRAMS_PATH": "program store",
    "CASCADE_MEMORY_EMBEDDER": "memory embedder backend",
    "CASCADE_MEMORY_EMBEDDER_MODEL": "memory embedder model",
    "CASCADE_MOTION_EVIDENCE_DIR": "motion evidence capture directory",
    "CASCADE_GPU_PERCEPTION_EVIDENCE_DIR": "GPU perception evidence directory",
    "CASCADE_MICRODUCK_ASSET_SHA256": "MicroDuck base asset pin (base profiles only)",
    "CASCADE_MICRODUCK_POLICY_SHA256": "MicroDuck base policy pin (base profiles only)",
    "CASCADE_MICRODUCK_MODEL_IDENTITY_SHA256": "MicroDuck base model pin (base profiles only)",
    "CASCADE_MICRODUCK_BRIDGE_PORT": "MicroDuck base bridge port (base profiles only)",
    "CASCADE_MICRODUCK_ENGINE": "MicroDuck base physics engine (base profiles only)",
    "CASCADE_MICRODUCK_DEVICE": "MicroDuck base physics device (base profiles only)",
}

#: why a variable the runtime reads is not copied from the registering shell
CATEGORIES: dict[str, str] = {
    "registration": "the registration writes it itself, from its own flags",
    "selection": "it would replace the rig the registration names",
    "internal": "the server derives it per process",
    "cli": "only the demo CLI's own brain loop reads it; the MCP host is the brain",
    "launcher": "only launcher-side tools read it",
    "transport": "only the HTTP transport reads it; registrations start stdio",
    "secret": "a credential; a persisted MCP config must never hold one",
}

#: name -> (category, reason)
NOT_FORWARDED: dict[str, tuple[str, str]] = {
    "CASCADE_CAMERAS": ("registration", "the camera rig: launch.sh --cameras / setup_agents --camera write it"),
    "CASCADE_ARM": ("registration", "the arm: launch.sh --arm / setup_agents --arm write it"),
    "CASCADE_OPENCLAW_PROFILE": ("registration", "launch.sh always writes the profile it registered under; the "
                                 "server checks it against --launch-owner, which only launch.sh passes"),
    "CASCADE_ARMS": ("selection", "the server prefers it over CASCADE_ARM, so an inherited list would silently "
                     "replace the arm the registration names; pass several arms with --arm a,b"),
    "CASCADE_CAMERA": ("selection", "legacy single-camera selector, read only when CASCADE_CAMERAS is unset; "
                       "every registration writes CASCADE_CAMERAS"),
    "CASCADE_ROBOT": ("selection", "selects a composed robot instead of the arm+camera rig the registration "
                      "names; set it explicitly (setup_agents --env) for such a server"),
    "CASCADE_BASE": ("selection", "selects a mobile base instead of the arm+camera rig the registration names; "
                     "set it explicitly (setup_agents --env) for such a server"),
    "CASCADE_RUN_DIR": ("internal", "per-process trace directory: the server derives a fresh one (mcp_<pid>, "
                        "or a new one per launch-owned process); an inherited one would merge sessions"),
    "CASCADE_LLM": ("cli", "brain profile of the demo CLI / dashboard runner; the MCP server hardcodes "
                    "llm=mock because its host is the brain"),
    "CASCADE_CODEX_BIN": ("cli", "codex executable for the demo CLI's codex_astra brain; the MCP server runs "
                          "no brain"),
    "CASCADE_PREMOTION_CHECK": ("cli", "the advisory pre-motion critic lives in the demo CLI's orchestrator; "
                                "the MCP server builds no orchestrator"),
    "CASCADE_JUDGE": ("launcher", "the advisory judge pass (scripts/judge_proof.py -> cascade.eval.proof_judge) "
                      "runs in the launcher after the proof, never in the MCP server"),
    "CASCADE_JUDGE_TIMEOUT_S": ("launcher", "bound of the launcher's judge pass (cascade.eval.proof_judge); "
                                "never read by the MCP server"),
    "CASCADE_MCP_SSE_KEEPALIVE_S": ("transport", "SSE keepalive of `mcp_server --http`; both registrations "
                                    "start the stdio transport"),
    "CASCADE_MCP_TOKEN": ("secret", "bearer secret of `mcp_server --http` only; stdio servers never read it "
                          "and it must not be persisted (use --token-file)"),
}


def forwarded_env(environ: Mapping[str, str] | None = None, skip: Iterable[str] = ()) -> dict[str, str]:
    """Every `FORWARDED` variable present in `environ` (default: this process),
    verbatim, in `FORWARDED` order. Absent ones are not invented and nothing
    outside `FORWARDED` is copied. Names in `skip` (the registration sets them
    from its own flags) are left out."""
    source = os.environ if environ is None else environ
    skipped = set(skip)
    return {name: source[name] for name in FORWARDED if name in source and name not in skipped}
