# PAAI on NVIDIA DGX Spark

**Next action:** run [Step 1](#1-check-the-spark) below (a few seconds). The six quick-path
steps take about 25 min with warm download caches and 30–90 min on an uncached machine,
plus cold CUDA builds.

PAAI (Physical Agentic AI) installs the Build a Claw kitchen demo: Isaac Sim 6.1 with PhysX
CUDA, real GraspGen-X inference, local Qwen Q4, OpenClaw and the PAAI camera extension in
Chromium. No model API key is needed. The kitchen is Cocina_Asier_01 (owner-authorized
geometry, CC0 wood, an original orange and fruit platter); the installer checks the archive
and every extracted asset against pinned SHA-256 hashes ([scene notice](../demo/scene/NOTICE.md)).
Capability status: [ROADMAP](ROADMAP.md).

## Validated release and scope

The commands pin `ff8d58be1ae8e000bbecb19cd8db1551ac54be73` (reviewed 1 October 2026).
The last fresh installation was measured on the earlier source
`9cf540279f09df8197eda6541311b41edb455243` on a GB10 Spark, 30 September 2026
([release receipt](../benchmark/results/spark_clean_delivery_20260930.json),
[UI screenshot](../benchmark/results/images/spark_clean_three_cameras_20260930.png)):

| Stage | Measured result |
| --- | --- |
| Install and read-only check | `PREPARED`, both exit 0; install **12 min 21 s** |
| First headless desktop launch | Exit 0 and `spark_verify.py` READY; **12 min 12 s** |
| Native visitor chat campaign | **5/5** objects passed independent physical placement and reset checks |
| Real Chromium and shipped extension | Connected chat, enabled controls and three advancing **1280 × 720** camera views |
| Full stop and second headless launch | New session and processes, READY again in **9 min 10 s** |

These used reused download/uv caches and are not guarantees. Chromium ran under Xvfb; the
GNOME app grid and a visible Isaac editor were not exercised.

Installed Spark defaults: PhysX CUDA, Qwen Q4, required GraspGen-X, three camera views,
`dt=1/120`, six bridge iterations between nominal camera polls and the observed-finger gate
on. Isaac nominal joint targets are 30 Hz with the legacy 50 Hz safety edges retained (not
measured throughput). Occupancy/nvblox is off; JEv is not part of the installer or presenter
runtime; Newton stays an explicit engine option. Explicit supported overrides still apply.
Short demo and recovery: [presenter card](BOOTH_RUNBOOK.md#9-simulated-kitchen-on-dgx-spark-presenter-card).

## Quick path

Use a normal account on the Spark. The default install folder is
`$HOME/paai-spark`. For a fresh install, the destination **must not exist**;
the installer creates it. Do not create an empty folder first.
To keep an existing installation, first run
`export PAAI_INSTALL_DIR="$HOME/paai-spark-new"` (a path that does not yet exist) and keep it
exported for every later step. An intentional retry may reuse the checkout of an earlier
attempt. The commands never change your HOME, the driver or the operating system.

Run each numbered block once and wait for its success line and exit code 0. Never repeat
Start just to see more output. If a command fails, stop and read
[recovery](#if-something-goes-wrong).

### 1. Check the Spark

A few seconds. DGX OS provides the NVIDIA driver and CUDA toolkit; the check also requires Git
LFS and a compiler. Chromium, `gio` and `gnome-terminal` are needed only for the Step 5
launcher: required in a graphical session, a non-blocking `WARNING` over SSH. A missing system
package needs the [administrator command](#system-prerequisites).

```bash
bash -o pipefail -c 'curl -fsSL --retry 3 https://raw.githubusercontent.com/johnnynunez/cascade/ff8d58be1ae8e000bbecb19cd8db1551ac54be73/scripts/spark_prerequisites.py | python3 -'
```

Success: `PREREQUISITES_OK`.

### 2. Install

Before your first installation, review the [NVIDIA Isaac Sim / Omniverse
licenses](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/common/legal.html).
**`--accept-eula` records your explicit acceptance for this checkout.**
An agent must have the operator's explicit consent for this Isaac version.
Previous explicit acceptance in the current task remains valid. If that consent is
already supplied, proceed to the install command; do not fetch the license
page or ask for consent again. `OMNI_KIT_ACCEPT_EULA=YES` alone does not grant consent.

The source URL and `--ref` select the same commit. This downloads the runtime, model and
assets, then builds the local CUDA model server. Needs at least 150 GiB free. Time: 12 min
with warm caches, 30–90 min uncached plus cold CUDA builds. The log shows each stage.

```bash
bash -o pipefail -c 'curl -fsSL --retry 3 https://raw.githubusercontent.com/johnnynunez/cascade/ff8d58be1ae8e000bbecb19cd8db1551ac54be73/scripts/bootstrap.sh | env -u ISAACSIM_PATH -u ISAACSIM_PYTHON_EXE -u CASCADE_QWEN_MODEL -u CASCADE_QWEN_MMPROJ -u LLAMA_SERVER -u LLAMA_DIR bash -s -- --ref ff8d58be1ae8e000bbecb19cd8db1551ac54be73 --profile spark --accept-eula --prepare-only --dir "${PAAI_INSTALL_DIR:-$HOME/paai-spark}" 2>&1 | tee "${PAAI_INSTALL_DIR:-$HOME/paai-spark}-install.log"'
```

Success: `PREPARED`. No demo services or manipulation tests have run; `--prepare-only`
installs and checks files and certifies no grasp or placement.

### 3. Check the installation

Read-only: starts no services, accepts no license.

```bash
cd "${PAAI_INSTALL_DIR:-$HOME/paai-spark}" && bash scripts/install.sh --profile spark --dir "$PWD" --check && printf 'CHECKED\n'
```

Success: `CHECKED`. The receipt is `runs/.install/install.json`.

### 4. Start and prove the demo

Starts the dedicated `cascade-demo` OpenClaw agent and moves the simulated robot through two
placement checks and resets. About 12 min; cold shader and collision preparation can take
longer, and the physical checks can be silent for minutes. Wait for the command to finish; do
not start another copy. Timeouts: model health 30 min, Isaac startup 20 min.

```bash
(
  set -o pipefail
  cd "${PAAI_INSTALL_DIR:-$HOME/paai-spark}" || exit
  python3 scripts/desktop.py launch --repo "$PWD" --headless --no-open 2>&1 |
    tee "${PAAI_INSTALL_DIR:-$HOME/paai-spark}-launch.log" | awk '/^\[desktop\]/ { print; fflush() }'
) && python3 "${PAAI_INSTALL_DIR:-$HOME/paai-spark}/scripts/spark_verify.py" --repo "${PAAI_INSTALL_DIR:-$HOME/paai-spark}" --expected-ref ff8d58be1ae8e000bbecb19cd8db1551ac54be73
```

Success: one `READY` summary with both placements, cameras and resets passing. It checks the
[current proof](#verify-the-result), health and live process ownership. Full output stays in
`${PAAI_INSTALL_DIR:-$HOME/paai-spark}-launch.log`. Do not repeat this block for more output.

### 5. Open the demo in one click

On the Spark desktop, open the app grid and click **PAAI (Spark)**. It starts the stack or
attaches to the running one, then opens a dedicated Chromium profile with the camera extension
loaded on the PAAI demo page (`http://127.0.0.1:8092`: chat to the OpenClaw `cascade-demo`
agent plus three live cameras). No extension setup, token or permission prompt. This page is
the demo's own UI, not OpenClaw's; see [Open OpenClaw](#open-openclaw).

With several installations, every app-grid entry is named **PAAI (Spark)**. Open the exact
registered launcher for your `PAAI_INSTALL_DIR` instead (also the desktop-automation target):

```bash
gio launch "${PAAI_INSTALL_DIR:-$HOME/paai-spark}/runs/.install/paai-spark.desktop"
```

Success: connected chat and advancing Worktop (`cam0`), Side (`side`) and Kitchen (`proof`)
views, usually within seconds. Write normal English in the chat. First inspect the table:

> What can you see on the table?

Then send one order and wait for it to finish:

> Could you put the green cube in the green square?

Check that the cube is released inside the square. To reset, send:

> Let's start over.

After reset, try the second order:

> Please put the orange in the open box.

Check that the orange is released inside the box. Reset before the next visitor.

### 6. Stop cleanly

Wait for the last request to finish. This stops only this checkout's stack, then checks that
no owned process remains. Up to a minute.

```bash
bash -e <<'BASH'
cd "${PAAI_INSTALL_DIR:-$HOME/paai-spark}"
./run.sh down
remaining=$(./run.sh down --dry-run)
printf '%s\n' "$remaining"
case "$remaining" in *'would stop'*) exit 1 ;; esac
printf 'STOPPED\n'
BASH
```

Success: `STOPPED`, with no `would stop` lines. Installed files stay. Next time, click
**PAAI (Spark)**. The rest of this page is reference; the Bonus Track is optional.

## System prerequisites

Linux `aarch64`, glibc 2.35 or newer, an NVIDIA driver compatible with CUDA 13 and a CUDA
toolkit with `nvcc` (DGX OS ships both; [DGX Spark software
versions](https://docs.nvidia.com/dgx/dgx-spark/release-notes.html)). The setup check reports
missing host components; it does not install them.

If Step 1 reports missing packages, an administrator runs this one command, then repeats
Step 1. Skip it when Step 1 already passes.

```bash
sudo apt-get update && sudo apt-get install -y git git-lfs curl python3 ca-certificates build-essential libgomp1 libglib2.0-bin gnome-terminal chromium-browser && printf 'PACKAGES_READY\n'
```

1. **Browser.** Use Chromium: branded Google Chrome 137+ [does not support the extension
   loading flag](https://groups.google.com/a/chromium.org/g/chromium-extensions/c/1-g8EFx2BBY/m/S0ET5wPjCAAJ).
   The launcher handles the snap's profile and file paths. A desktop-file manager may require
   **Allow Launching** for an untrusted icon.
2. **Disk.** At least 150 GiB free in the home volume (also check `XDG_CACHE_HOME` /
   `UV_CACHE_DIR` if overridden). Model and projector: 18.85 GB; kitchen archive about 30 MB
   (50 MB extracted).
3. **Network.** Outbound HTTPS (with redirects) to GitHub and Git LFS, Hugging Face and its
   file hosts, PyPI, `pypi.nvidia.com`, `download.pytorch.org`, `astral.sh`, `openclaw.ai`,
   Node.js and the npm registry.
4. **Clean shell.** Unset `ISAACSIM_PATH`, `ISAACSIM_PYTHON_EXE`, `CASCADE_QWEN_MODEL`,
   `CASCADE_QWEN_MMPROJ`, `LLAMA_DIR` and `LLAMA_SERVER` from another installation; the quick
   path uses private runtimes in this checkout.

## Installed files

Paths are relative to `$HOME/paai-spark`.

| Component | Location |
| --- | --- |
| App, Python 3.12 and CUDA perception packages | `.venv/` |
| Isaac Sim `isaacsim[all,extscache]==6.1.0.0`, Python 3.12 | `.isaacsim/` |
| GraspGen-X CUDA model, pinned source and full checkpoints | `.graspgenx/`, `.graspgenx-src/` |
| OpenClaw 2026.9.3 and private Node.js | `.openclaw-cli/` |
| Pinned Qwen Q4 model and BF16 vision projector | `models/qwen3.8-27b/` |
| Pinned llama.cpp CUDA build and receipt | `.llama.cpp/build/` |
| Robot Git LFS files and perception weights | `assets/`, `models/` |
| Verified Cocina Asier room and original task props | `demo/scene/cocina_asier/`, `demo/scene/props/` |
| Source, package and consent receipt; reusable environment | `runs/.install/install.json`, `runs/.install/env.sh` |

The installer also uses `$HOME/.local/bin/uv`, `$HOME/.cache/uv` and
`$HOME/.cache/cascade/installers`, obtains Python 3.12 when needed and registers the desktop
launcher. A rerun reuses verified files.

The consent receipt belongs to the absolute checkout path in `install.json`. A moved checkout
needs installation to record the operator's acceptance at the new path (prior explicit
authorization for this Isaac version need not be asked again). Do not edit consent receipts.
`--dry-run` and `--check` grant no consent.

## Verify the result

The installation receipt must show the checkout's `repo`, the pinned `source_commit`,
`source_dirty: false`, `profile: "spark"`, `brain: "qwen"` and `eula_accepted: true`.

Step 4 validates `runs/.install/desktop-latest.json` and
`runs/.launch/profile-cascade-demo/proof.json` while the demo runs:

1. Health has `"ok": true`; the desktop receipt has `action: "launch"` and `exit_code: 0`.
2. `proof.json` has `verified: true`, `sim: "isaac"` and model `Qwen/Qwen3.8-27B`. On a fresh
   start, `attached: false` in the desktop receipt and proof `started_at` between its
   `started_at` and `finished_at` prove this launch ran the proof.
3. Reopening the launcher records `attached: true` and keeps the existing proof, whose
   `session_id`, `started_at` and `process` must match the receipt's `proof` object; the
   launcher checks the owned simulation process is still live before attaching.
4. `cases` include green cube to green square and orange to open box, each with
   `physics.pass: true`, `physics.event_cameras.pass: true` and `props_reset` containing
   `green_cube` or `orange`.
5. Each case's `evidence_dir` holds native turns, `physics/gpu-physical-audit.json`,
   `physics/placed-<camera>.jpg` and `physics/reset-<camera>.jpg`; all three cameras advance.

`PREPARED` means the dependencies are ready. `READY` requires the physical proof.
`STARTED (UNVERIFIED: robot proof skipped)` is not an accepted result. An old receipt or an
English success message is not proof. (Historical, 9cf5402 source only: its OpenClaw tool could
report "Placement unverified" on a correct placement because it checked only the object's
centre. The physical audit decides.) A later 5/5 direct-runtime verdict campaign is in the
[placement-verdict receipt](../benchmark/results/spark_placement_verdict_20261001.json); it is
not a native chat acceptance.

## Ports and coexistence

| Loopback port | Service |
| --- | --- |
| 8080 | This checkout's local Qwen API |
| 8091 | This checkout's read-only camera surface |
| 8092 | This checkout's local chat and camera extension |
| 8093 | Optional authenticated visitor surface |
| 5556 | GraspGen-X CUDA inference server |
| 8611 | Isaac bridge |
| 18790 | This checkout's OpenClaw gateway, profile `cascade-demo` |
| 4043 | Optional dedicated ngrok agent API |

The demo does not use port 8090. The separate personal OpenClaw gateway on 18789 stays
untouched. Spark verifies one GraspGen-X diffusion inference on 5556 before connecting the
robot tools ([GraspGen-X on Spark](GRASPGENX_SPARK.md)). Occupancy on 5557 stays off. OpenClaw
state lives under `runs/.launch/profile-cascade-demo/openclaw/`.

1. Run one Spark stack at a time. Before the first start, `ss -ltnp` shows occupied ports;
   resolve a conflict with another application's service before launching.
2. Never kill a process by its port or name. Reopening this checkout's launcher attaches.
3. Never expose the OpenClaw gateway, Qwen API, Isaac bridge or MCP publicly.
4. Separately owned model service: the low-level `scripts/launch.sh` honors an explicit
   `CASCADE_QWEN_BASE_URL` (e.g. `http://127.0.0.1:18080/v1`), also under the Spark profile.
   The resolver still requires the `Qwen/Qwen3.8-27B` id and Spark context settings; a wrong
   model or explicitly empty URL fails, unset keeps 8080. It does not start, own or verify the
   weights: record that service's source/model and process identity for a physical proof.

## Open OpenClaw

The PAAI page on `http://127.0.0.1:8092` relays chat to the `cascade-demo` agent but is not
OpenClaw's UI. The Spark has no `/guide` or `/openclaw` page (those belong to the Brev booth).
You never copy the gateway token: it changes every launch and the commands read the current one.

Native OpenClaw Control UI on port 18790, from the Spark's desktop session:

```bash
cd "${PAAI_INSTALL_DIR:-$HOME/paai-spark}" && ./run.sh dashboard
```

It opens in the PAAI Chromium profile; click the extension or press **Ctrl+Shift+Y** for
Kitchen, Worktop and Side beside the chat. The one-time sign-in link expires after 10 minutes
and is never logged. Without a desktop session the command prints an SSH tunnel command and a
local address (that browser has no camera extension unless you install it).

From another computer, forward both ports over SSH, then open `http://localhost:8092`
(replace `user@spark-host`):

```bash
ssh -L 8092:127.0.0.1:8092 -L 18790:127.0.0.1:18790 user@spark-host
```

Forward these ports only over SSH. For a shareable URL, use the optional
[Bonus Track](#bonus-track-always-on-demo-on-ngrok-for-testing-purposes).

## If something goes wrong

**Install error:** keep `$HOME/paai-spark-install.log`, fix only the reported package, network
or disk issue, then repeat the same pinned install command once (downloads already retry).

**Failed launch or stalled camera:** read the log path the launcher printed, then stop and
retry once:

```bash
cd "${PAAI_INSTALL_DIR:-$HOME/paai-spark}" && ./run.sh down && python3 scripts/desktop.py launch --repo "$PWD" --headless --no-open
```

**Inspect without restarting** (read-only; the first prints the receipt and its log path):

```bash
python3 -m json.tool "${PAAI_INSTALL_DIR:-$HOME/paai-spark}/runs/.install/desktop-latest.json"
python3 "${PAAI_INSTALL_DIR:-$HOME/paai-spark}/scripts/spark_browser.py" --repo "${PAAI_INSTALL_DIR:-$HOME/paai-spark}" --check
```

1. Logs: `runs/.install/desktop-*/progress.log` and `runs/.launch/profile-cascade-demo/`
   (`qwen.log`, `isaac_bridge.log`, `openclaw-gateway.log`). An early bootstrap error may
   leave only the install log.
2. Exit 2 = rejected arguments, consent or host prerequisites; a read-only check exiting 3
   reports `MISSING` components.
3. Same failure again: stop and report the command, exit code and log path. Keep incomplete
   environments and failed evidence. Do not extend timeouts, skip the robot proof, edit
   receipts or change model or physics settings to obtain `READY`.
4. Missed grasp or `did not settle at home`: inspect the failure and cameras, then request a
   scene reset. A failed reset is not a restored scene. Send one order at a time.

## READY checks and quick recovery

READY needs both startup orders (green cube to the green square, orange into the open box)
placed through OpenClaw, confirmed by the independent physics observer and each followed by a
reset.

**Next visitor:** use **Start over** or **Reset** in the chat, or send **"Reset the scene."**:
the arm returns home, props are restored, object beliefs and task images are cleared and a
fresh observation is required. Isaac, Qwen and OpenClaw keep running; the startup proof is not
repeated and learned grasp history stays.

**After a crash, with Qwen still healthy:** restart the robot stack without the startup proof:

```bash
cd "${PAAI_INSTALL_DIR:-$HOME/paai-spark}" && ./run.sh isaac --engine physx --graspgenx local --no-robot-turn --no-judge
```

It still checks services, the model, learned grasp inference, tools and scene identity, and
reports **STARTED (UNVERIFIED: robot proof skipped)** (`--no-robot-turn` never reaches READY).
It does not start Qwen or write a READY receipt, and the desktop camera/chat attachment still
requires a verified proof. Use the normal desktop launch when Qwen or those surfaces need
recovery; the desktop command does not accept the two skip flags.

## Agent checklist

1. Complete the quick path in order. Require the operator's EULA consent.
2. Require `PREPARED`, exit 0, and matching source, path and consent fields.
3. Require read-only `--check` exit 0, then launch exit 0 and `READY`.
4. Verify current health, proof, both physics cases, resets and all cameras.
5. In a graphical session, open the registered launcher; verify the connected chat, loaded
   extension and cameras; send a natural-English order and reset. Then require a clean stop
   (exit 0, no `would stop`). Preserve evidence and unrelated services; report failed or
   unverified results honestly.

## Bonus Track: Always-on demo on ngrok for testing purposes

This is optional and is not part of the official setup. At events, the demo
runs on the Spark with its own monitor and keyboard.

1. Install the Linux arm64 executable from [ngrok's download page](https://ngrok.com/download/linux)
   and use an ngrok account with a stable domain. Keep its token in a private ngrok
   configuration file (mode `0600`).
2. Create a private JSON file `{"username":"<visitor-name>","password":"<visitor-password>"}`
   with mode `0600` and a password of at least 12 characters. Never put credentials in the
   domain or commit these files.
3. Check `loginctl show-user "$USER" -p Linger` shows `Linger=yes` (the user services must
   survive logout); otherwise ask the administrator to enable linger.
4. Enable (replace every placeholder):

```bash
python3 "${PAAI_INSTALL_DIR:-$HOME/paai-spark}/scripts/spark_public.py" enable --repo "${PAAI_INSTALL_DIR:-$HOME/paai-spark}" --domain '<your-domain>.ngrok.dev' --auth-file '<private-visitor-auth.json>' --ngrok '<path-to-ngrok>' --ngrok-config '<existing-ngrok.yml>'
```

Success: `PUBLIC ENABLED` (services registered; a stopped demo still needs its normal startup
time). Check after it reaches `READY`:

```bash
python3 "${PAAI_INSTALL_DIR:-$HOME/paai-spark}/scripts/spark_public.py" check --repo "${PAAI_INSTALL_DIR:-$HOME/paai-spark}"
```

Success: `PUBLIC READY`. Disable the public endpoint and its demo stack:

```bash
python3 "${PAAI_INSTALL_DIR:-$HOME/paai-spark}/scripts/spark_public.py" disable --repo "${PAAI_INSTALL_DIR:-$HOME/paai-spark}"
```

Success: `PUBLIC STOPPED` (installation files stay).

The `paai-spark-demo`, `paai-spark-visitor` and `paai-spark-ngrok` user services are
supervised separately. The demo service restarts after a crash or failed health check, at
most three starts per hour; a launch that never reaches `READY` is not restarted (it would
rebuild the stack in a loop): read `runs/.install/desktop-latest.json`, fix the cause and run
`enable` again, which clears the failed state. Visitor and ngrok services restart until the
demo is ready. The desktop launcher attaches to this same installation; existing ngrok
sessions stay unchanged.

An authenticated visitor gets three live cameras and chat with the `cascade-demo` agent and
its seven attendee tools. Send one order at a time; "Let's start over." resets. Requests without
credentials receive HTTP 401. The public surface excludes the OpenClaw Control UI,
administration, configuration, tool policy and raw backend APIs; the gateway token never
leaves the Spark.

**Learned notes.** Each launch copies the note library (`~/.cascade/aspire/sim/library`,
or `CASCADE_ASPIRE_LIBRARY`) into its state directory and serves that copy until the next
launch, then writes `LEARNED.md` from it beside the kitchen workspace's `AGENTS.md`; the
agent reads it right after `AGENTS.md`. A note you add or edit counts from the next launch.
1. The launch log says `learned notes: froze <library> into <copy>: <n> note files,
   library_sha=<sha>`, then `learned notes: wrote|removed|unchanged <path> library_sha=<sha>`,
   then `learned notes: matches ...` after the proof.
2. `WARNING: learned notes:` means the demo runs without notes (the library could not be
   copied, or the file could not be written or checked). A hand-written `LEARNED.md` is never
   used or overwritten: move it away.
3. Ask "What have you learned from earlier runs?" to see every served note.
