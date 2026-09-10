# DGX Spark demo delivery

## Target and certification boundary

One independent installation per DGX Spark, running Linux / DGX OS with a
working NVIDIA driver. The production target is **Isaac Sim v6.1.0**, installed
as `isaacsim[all,extscache]==6.1.0.0` under Python 3.12, **Cosmos3-Edge** as the
local brain, and **OpenClaw** as the attendee chat. The bridge retains its
Newton physics default. Standalone Newton CPU tests on macOS supplement
the development suite; they do not replace this Isaac runtime.

**GPU certification is pending.** There was no Spark available during this
implementation. Installer boundary tests, upstream wheel inspection and a
real OpenClaw/OpenAI/MuJoCo rehearsal on macOS do not certify Isaac rendering,
Cosmos inference, GPU memory coexistence or physical manipulation on Spark.
Do not label the event build certified until the cold-start checklist below
passes on a representative Spark.

## One command per machine

After these changes are published to the chosen repository ref:

```bash
curl -fsSL https://raw.githubusercontent.com/johnnynunez/cascade/main/scripts/bootstrap.sh | bash -s -- --accept-eula
```

For an event, use an immutable tested ref rather than a moving `main`, and
pass that same ref to the installer with `--ref`. This development session
does not publish a branch, tag or release. Until publication, exercise the
installer from the local checkout:

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
2. Install the complete pinned Isaac Sim package set, not merely import its
   metapackage. Resolve the Newton experience from the installed package.
3. Install the pinned OpenClaw CLI in the checkout-local prefix.
4. Prepare the separately versioned Cosmos stack and model snapshot; start
   the reasoner and require native OpenAI tool calls, not XML in prose.
5. Start Isaac with the shipped robot/scene and the configured sidecars.
6. Register CASCADE in the dedicated OpenClaw profile and check tools.
7. Run brain -> pick -> physics verdict -> reset in ONE persistent chat
   session. Read the reset from the exact world that performed the pick.
8. Only then print READY and open the OpenClaw chat.

The initial download / model export / shader warmup is preparation work,
not an instantaneous launch. Repeat runs reuse caches and environments.
Prepare the machines before attendees arrive rather than starting N large
downloads simultaneously on the venue network.

## Isolation and defaults

The control stack is independent of the default Isaac simulator. Upgrading
OpenClaw/Cosmos must not change Isaac's Python packages. They communicate
over the loopback model API and simulator bridge, not cross-environment
imports. The current stable version matrix is intentionally resolved and
pinned **per environment**, rather than selecting every dependency's highest
version inside one incompatible environment:

- CASCADE/MCP + perception: `.venv`; torch **2.14.0+cu130** and torchvision
  **0.29.0+cu130** on Spark (the latest stable app pair).
- Cosmos backend: `.cosmos`; vLLM **0.29.0**, its required torch **2.13.0**,
  and transformers **5.17.0**. Independent of the app and Isaac torch ABIs.
- Isaac Sim: `.isaacsim`; official **6.1.0.0** / Python **3.12**. Its core
  requires torch **2.11.0**; the installer selects **2.11.0+cu130**. Do not
  force torch2.14 into it. The launcher accepts `ISAACSIM_PYTHON_EXE`.
- OpenClaw CLI: `.openclaw-cli/bin/openclaw`, version 2026.9.3.
- OpenClaw profile: `cascade-demo`; separate workspace, no personal skill
  collection and an explicit CASCADE tool allowlist.
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
previous success before querying the brain. An old trace or old green receipt
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
