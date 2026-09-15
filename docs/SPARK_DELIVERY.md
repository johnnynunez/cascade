# DGX Spark demo delivery

For the installation walkthrough, start with [DGX Spark setup](DGX_SPARK_SETUP.md).
This document covers installer behavior and acceptance checks in more detail.

## Target and certification boundary

One independent installation per DGX Spark, running Linux / DGX OS with a
working NVIDIA driver. The production target is **Isaac Sim v6.1.0**, reusing
a complete source/standalone release or installing
`isaacsim[all,extscache]==6.1.0.0` under Python 3.12, **Cosmos3-Edge** as the
local brain, and **OpenClaw** as the attendee chat. The bridge retains its
Newton physics default. Standalone Newton CPU tests on macOS supplement
the development suite; they do not replace this Isaac runtime.

**Installation is not GPU certification.** Installer boundary tests and
metadata-only checks of an existing Spark source release do not certify Isaac
rendering, Cosmos inference, GPU memory coexistence or physical manipulation
in simulation. Do not label the event build certified until the cold-start
checklist below passes on a representative Spark with retained live evidence.

## One command per machine

After these changes are published to the chosen repository ref:

```bash
curl -fsSL https://raw.githubusercontent.com/johnnynunez/cascade/main/scripts/bootstrap.sh | bash -s -- --accept-eula
```

For an event, use an immutable tested ref rather than a moving `main`, and
pass that same ref to the installer with `--ref`. The installer must exist
at that ref before the remote command can use it. To exercise it from an
existing checkout:

```bash
bash scripts/install.sh --dir "$PWD" --profile spark --accept-eula
```

`--accept-eula` records the operator's explicit acceptance of the NVIDIA
Omniverse license. The script does not assume acceptance, install or replace
GPU drivers, reinstall DGX OS, or weaken device permissions. A missing host
prerequisite stops with a diagnostic; it is not silently replaced by MuJoCo
or a hosted model.

The default path must complete the entire chain:

1. Obtain the checkout and create its isolated CASCADE environment.
2. Validate and reuse an existing Isaac 6.1 release without package writes,
   or install the complete pinned package set in `.isaacsim`. Resolve the
   Newton experience from the selected release, not another Python environment.
3. Install the pinned OpenClaw CLI in the checkout-local prefix.
4. Prepare the separately versioned Cosmos stack and model snapshot; start
   the reasoner and require native OpenAI tool calls, not XML in prose.
5. Start Isaac with the shipped robot/scene and the configured sidecars.
6. Register CASCADE in the dedicated OpenClaw profile and check tools.
7. Run brain -> pick -> physics verdict -> reset in ONE persistent chat
   session. Read the reset from the exact world that performed the pick.
8. Only then print READY and open the OpenClaw chat. Desktop shortcuts are
   registered after successful preparation, before starting this proof chain.

The initial download / model export / shader warmup is preparation work,
not an instantaneous launch. Repeat runs reuse caches and environments.
Prepare the machines before attendees arrive rather than starting N large
downloads simultaneously on the venue network.

## Desktop installation and launch

From an existing checkout, register the two Linux shortcuts without starting
anything or accepting any license:

```bash
python3 scripts/desktop.py register --repo "$PWD" --dry-run  # no writes
python3 scripts/desktop.py register --repo "$PWD"            # register only
```

Registration is also performed automatically after successful Spark
preparation, including `--prepare-only`. It writes checkout-specific,
idempotent entries to `${XDG_DATA_HOME:-$HOME/.local/share}/applications`
and the configured XDG desktop directory when enabled. Other applications
are preserved. `--desktop-dir PATH` overrides only the desktop destination,
not the application-menu destination. A desktop environment may require
the operator to choose **Allow Launching** once; registration does not bypass
desktop trust or install an autostart service.

- **Install CASCADE (Spark)** opens a terminal, asks for explicit NVIDIA EULA
  consent with a default-cancel GUI prompt (or the exact terminal text
  `I agree`), then runs the installer and simulation proof. Cancel makes no
  install/launch writes. Absence of a usable prompt is not consent.
- **CASCADE (Spark)** uses the recorded Isaac interpreter and the existing
  installer supervisor for local Cosmos, then runs the launcher/proof chain.
  Missing explicit consent, a non-Spark receipt or a missing app environment
  fails rather than installing packages or substituting a different demo.
- CLI equivalents are `python3 scripts/desktop.py install --repo "$PWD"`
  and `python3 scripts/desktop.py launch --repo "$PWD"`. For interactive
  consent followed by preparation only, add `--prepare-only` to `install`.
  Either action accepts `--dry-run`, which starts nothing, writes nothing
  and does not prompt or accept a license.

The terminal streams installer/launcher output and retains the complete
stream in `runs/.install/desktop-*/progress.log`. The latest operation,
log path, timestamps and exit code are in `runs/.install/desktop-latest.json`.
Service-specific output is in the launch-state directory described below.
Duplicate clicks are refused while an operation is running (exit 75).
Failures retain a nonzero exit code; Ctrl-C, TERM and terminal HUP use the
controller's private-child shutdown path. Never infer READY from exit 0 of
registration or preparation. A valid current physical proof remains required.

## Reusing an existing Isaac source release

For example, select the already-built Spark release explicitly before a
metadata-only check or installation:

```bash
export ISAACSIM_PATH="$HOME/Projects/isaac/IsaacSim/_build/linux-aarch64/release"
bash scripts/install_isaac.sh --dir "$PWD" --check  # no EULA acceptance or Kit startup
# Only after the operator has reviewed and accepted the EULA:
bash scripts/install.sh --dir "$PWD" --profile spark --accept-eula --prepare-only
```

Selection order is explicit `ISAACSIM_PYTHON_EXE`, explicit `ISAACSIM_PATH`,
the executable `.isaacsim/bin/python`, then known source/standalone locations
(`$HOME/Projects/isaac/IsaacSim/_build/linux-<arch>/release`, `$HOME/isaacsim`,
legacy Omniverse package directories and `/isaac-sim`). An explicitly selected
broken release fails closed, without silently installing wheels instead.
Source reuse never runs pip/uv against the embedded Python or changes Kit's
packages. Application and Cosmos dependencies remain in their own environments.

Metadata preflight strips inherited Python/loader overrides. Actual Isaac
startup uses `scripts/isaac_launch.py`, preserving display/device selection
but removing the caller's virtualenv, `PYTHONEXE`, library paths and profiling
injection. The adapter forwards termination to the source wrapper's private
child group; a source `python.sh` is not itself an exec wrapper.

Successful installation records checkout-bound explicit consent, source
identity and the selected Isaac environment in `runs/.install/install.json`,
plus `runs/.install/env.sh` for shell reuse. An ambient
`OMNI_KIT_ACCEPT_EULA=YES` does not grant consent. Legacy receipts without
explicit consent must be replaced through an operator-approved installation;
do not fabricate a receipt. `run.sh` restores the recorded shell environment
for low-level checks/down/launch, but unlike the desktop controller its launch
path expects Cosmos already running. Use the desktop launch action for the
supervised full-stack bring-up.

Cold Newton collision preprocessing and shader work can take more than ten
minutes. Isaac startup has a visible 1200-second budget (`ISAAC_WAIT_S` may
override it), with periodic real log excerpts. This timeout is not a readiness
shortcut: even borrowed bridges must answer health checks with finite joint
state, and managed Spark mode requires the Newton engine.

### Existing systemd service budgets

The desktop controller, Cosmos supervisor and Isaac adapter create private
process sessions for signal ownership, not new cgroups. Their subprocesses
retain the caller's systemd/cgroup resource budget. They do not apply `ulimit`,
write service memory limits or raise global limits. A separate OpenClaw service
has its own budget; an environment variable in the desktop shell cannot raise
that service's `MemoryMax` or an ancestor slice's lower limit.

When an operator authorizes a task-specific budget (for example, 48 GiB per
demo service), the deployment operator must apply and verify it on the exact
demo unit and check effective ancestor limits before cold start. Keep the
override scoped to that unit and retained across service restarts. No desktop
registration, EULA consent or passing metadata check authorizes changing
system-wide limits. Resource-limit changes and real systemd execution remain
a separate deployment step, not something the installer silently escalates.

## Isolation and defaults

The control stack is independent of the default Isaac simulator. Upgrading
OpenClaw/Cosmos must not change Isaac's Python packages. They communicate
over the loopback model API and simulator bridge, not cross-environment
imports. The current stable version matrix is intentionally resolved and
pinned **per environment**, rather than selecting every dependency's highest
version inside one incompatible environment:

- CASCADE/MCP + perception: `.venv`; torch **2.14.0+cu130** and torchvision
  **0.29.0+cu130** on Spark (the latest stable app pair).
- Cosmos backend: `.cosmos`; plain vLLM **0.29.0**, its required torch
  **2.13.0**, and transformers **5.17.0**. Independent of the app and Isaac
  torch ABIs. This path is not vLLM-Omni; a separate Omni investigation or
  `.cosmos-omni` environment does not change the delivered backend.
- Isaac Sim: existing **6.1.0** source/standalone release, or `.isaacsim`
  with official **6.1.0.0** / Python **3.12** wheels. The wheel core requires
  torch **2.11.0**; only the managed wheel installer selects
  **2.11.0+cu130**. Source releases keep their embedded package versions.
  Do not force the app's torch2.14 into either Isaac environment.
- OpenClaw CLI: `.openclaw-cli/bin/openclaw`, version 2026.9.3.
- OpenClaw profile: `cascade-demo`; separate workspace, no personal skill
  collection and an explicit CASCADE tool allowlist. Both the default and
  main-agent workspace point to `workspace` beside the selected profile's
  config file, not a transient checkout `runs/` directory. Runtime grasp,
  envelope and belief memories default to the profile-owned launch state,
  unless the operator explicitly supplies their paths.
- Profile gateway: loopback port 18790 by default. Existing unprofiled
  development uses 18789. Ports are local to each Spark; no fleet controller
  or cross-machine shared session is needed.
- Model alias: `cosmos3-edge`, with a native tool-call inference probe.
  The in-process `Cosmos3EdgeClient` is not the OpenClaw host adapter.
- Cosmos serving pins the current stable compatible matrix above and HF
  revision `a9d944e2c6a1bf9f48b92ad16348e70c5f1836ba`. The reasoner is
  exported with canonical HF keys, staging, checksums and a completion
  manifest; an incomplete export is never silently reused. A behavioral
  no-video RoPE probe guards the known 5.17.0 empty-tensor compatibility fix.

A listening port is not ownership or readiness. The installer and launcher
share `cascade.apps.process_owner`: `CASCADE_LAUNCH_STATE` is a root (default
`runs/.launch`), with `profile-cascade-demo/` for Spark and `unprofiled/` when
no OpenClaw profile is selected. Cosmos logs and its JSON `cosmos.pid` receipt,
as well as `proof.json`, live inside that effective profile directory.
Receipts bind repo/profile/owner to the PID, kernel birth identity and exact
command. Cosmos is registered only after health and its final exec; a reused
external endpoint is never adopted. Legacy numeric PID files are ignored.
Stopped processes may retain inert receipts for diagnosis: use `live_records`,
not file existence, to determine liveness. A failed installer must not delete
a receipt that another invocation has replaced.

The launcher authenticates against the selected gateway profile and waits for
semantic health across configuration reloads. It does not kill an unrelated
service occupying the port. It updates only the selected CASCADE MCP entry,
not other tool servers.

Development on a Mac remains explicit: MuJoCo plus an already authenticated
OpenAI/Anthropic model. That must be identified as the effective provider,
never reported as a Cosmos run. The stock MuJoCo demo uses color-threshold
perception and a small scene; it does not demonstrate arbitrary semantic
object recognition.

## Evidence and failure behavior

The launcher writes a per-attempt directory below its launch state directory
and a `proof.json` receipt. The receipt names the provider/model, simulator,
OpenClaw profile/session, trace and reset props. A new attempt invalidates the
previous success before dependency/startup checks. An old trace or old green receipt
is not evidence for a new launch.

Physical success requires all of:

- the expected tool was actually executed, with an explicit zero failure
  count on motion turns;
- the effective model matches the selected model;
- the expected object has a current trace with `ok: true`, `verified: true`,
  `postcondition.status: confirmed` and `postcondition.channel: physics`;
- the trace belongs to the explicitly identified MCP runtime and session,
  not another recent run under the same checkout;
- reset succeeds in that same trace/world, includes the manipulated prop
  and is followed by the requested `world_state` tool;
- Isaac reset reports finite measured positions for every configured spawn,
  with distance within 0.02 m; empty, partial or contradictory replies fail.

Explicit `--no-robot-turn` launches are **STARTED / UNVERIFIED**, not READY.
The optional visual outcome judge is a separate metric; a judge outage cannot
invent a physics success or erase valid physics evidence.

## Cold-start acceptance on a Spark

Run this with no inherited CASCADE/Isaac/Cosmos environments:

1. Confirm the exact source ref, Linux/aarch64 host, functioning NVIDIA
   driver, disk space, display and network access to the package/model hosts.
2. Run the one-command installer and retain its full logs and exit code.
3. Confirm installed Isaac **6.1.0.0**, the selected Newton experience, real
   RGB-D frames, finite robot state and props resting on the table.
4. Confirm Cosmos model snapshot, GPU device, native tool-call probe,
   OpenClaw profile isolation and absence of a hidden cloud fallback.
5. Require the launcher's physical proof and verified reset to pass.
6. In the attendee chat, move each supported scene object in Spanish or
   English, verify the physical result, ask what was moved, and reset.
7. Try an absent object, an invalid reference and an unreachable target.
   Expect an honest refusal, never substitution of another object or a
   fabricated success. Test emergency stop separately in simulation.
8. Repeat the installer: no duplicate services, no source reset, no repeated
   downloads of completed artifacts, same provider and scene.
9. Stop and relaunch; verify the next attendee starts from the spawn layout.
10. From the real desktop session, verify the install/launch shortcuts, EULA
    cancellation, progress/error window, duplicate-click protection and a
    cold launch with the recorded Isaac source. Keep the personal gateway
    on 18789 untouched; the dedicated demo gateway defaults to 18790.

"Any object" is bounded by segmentation/identity, reach, jaw geometry and the
available skills. Same-color duplicates, arbitrary new assets and arbitrary
natural-language actions need their own acceptance cases. A two-cube test is
not proof of those capabilities.

## Authoritative version evidence

- Release: https://github.com/isaac-sim/IsaacSim/releases/tag/v6.1.0
- Package metadata: https://pypi.org/pypi/isaacsim/6.1.0.0/json
- Model: https://huggingface.co/nvidia/Cosmos3-Edge
- OpenClaw installer: https://docs.openclaw.ai/install/installer

The published 6.1.0.0 aarch64 wheel was inspected and includes
`isaacsim/apps/isaacsim.exp.full.newton.kit`. The versioned 6.1 documentation
URLs were not yet reachable during implementation, and the `latest` Python
installation page still described 6.0; neither is used to silently downgrade
the user's requested release.
