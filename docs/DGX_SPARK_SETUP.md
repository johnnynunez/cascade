# PAAI on NVIDIA DGX Spark

PAAI means Physical Agentic AI. This installs the Build a Claw kitchen demo:
Isaac Sim, local Qwen Q4, OpenClaw and the PAAI camera extension in Chromium.
No model API key is needed.

The commands below remain pinned to the previously verified revision and install
its earlier kitchen. The original kitchen now included in this repository needs
a fresh Spark installation and physical proof before this guide can be re-pinned.

## Quick path

Use a normal account on the Spark. Start with a new `$HOME/paai-spark` folder.
Run each numbered block once. Wait for its success line and exit code 0
before continuing. For diagnostics, use the read-only commands below. Never
repeat Start just to see more output. If a command fails, stop and read
[recovery](#if-something-goes-wrong).
The installer does not change the driver or operating system.

### 1. Check the Spark

This takes a few seconds. DGX OS provides the NVIDIA driver and CUDA toolkit.
The check also requires Git LFS and a compiler. Chromium, `gio` and
`gnome-terminal` are needed only for the desktop launcher in Step 5: from a
graphical session they are required, and over SSH a missing one is a
`WARNING` that does not block Steps 2–4. If the check reports a missing
system package, use the [administrator command](#system-prerequisites)
before continuing.

```bash
bash -o pipefail -c 'curl -fsSL https://raw.githubusercontent.com/johnnynunez/cascade/fac5332416f688fd5d035e0b0a8088d52ef4d469/scripts/spark_prerequisites.py | python3 -'
```

Success: `PREREQUISITES_OK`.

### 2. Install

Before your first installation, review the [NVIDIA Isaac Sim / Omniverse
licenses](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/common/legal.html).
**`--accept-eula` records your explicit acceptance for this checkout.**
An agent must have the operator's explicit consent. If that consent is
already supplied, proceed to the install command; do not fetch the license
page or ask for consent again. `OMNI_KIT_ACCEPT_EULA=YES` alone does not grant consent.

The source URL and `--ref` select the same tested commit. This downloads the
runtime, model and assets, then builds the local CUDA model server. Allow
space for at least 150 GiB. A fresh installation from public download sites
took about 25 minutes. Plan for 30–90 minutes; slower connections take longer.
The progress log shows each stage.

```bash
bash -o pipefail -c 'curl -fsSL https://raw.githubusercontent.com/johnnynunez/cascade/fac5332416f688fd5d035e0b0a8088d52ef4d469/scripts/bootstrap.sh | bash -s -- --ref fac5332416f688fd5d035e0b0a8088d52ef4d469 --profile spark --accept-eula --prepare-only --dir "$HOME/paai-spark" 2>&1 | tee "$HOME/paai-spark-install.log"'
```

Success: `PREPARED`. No demo services have started yet.

### 3. Check the installation

This is read-only. It starts no services and accepts no license. The check
took about five seconds in the measured installation.

```bash
cd "$HOME/paai-spark" && bash scripts/install.sh --profile spark --dir "$PWD" --check && printf 'CHECKED\n'
```

Success: `CHECKED`. The receipt is `runs/.install/install.json`.

### 4. Start and prove the demo

This starts the dedicated `cascade-demo` OpenClaw agent and moves the
simulated robot through two placement checks and resets. The measured start
and proof took about 14 minutes. Cold shader and collision preparation can
take longer. The physical checks can be quiet for several minutes. Wait for
the command to finish even when no new log lines appear; do not start another copy. The model health timeout
is 30 minutes and the Isaac startup timeout is 20 minutes.

```bash
(
  set -o pipefail
  cd "$HOME/paai-spark" || exit
  python3 scripts/desktop.py launch --repo "$PWD" --headless --no-open 2>&1 |
    tee "$HOME/paai-spark-launch.log" | awk '/^\[desktop\]/ { print; fflush() }'
) && python3 "$HOME/paai-spark/scripts/spark_verify.py" --repo "$HOME/paai-spark" --expected-ref fac5332416f688fd5d035e0b0a8088d52ef4d469
```

Success: one `READY` summary with both placements, cameras and resets passing.
The command checks the [current proof](#verify-the-result), health and live
process ownership. Full output stays in `$HOME/paai-spark-launch.log`; the
terminal shows only progress and the compact result. Do not repeat this
block to obtain more output.

### 5. Open the demo in one click

On the Spark desktop, open the app grid and click **PAAI (Spark)**. The
launcher starts the stack when needed or attaches to the running stack.
It opens a dedicated Chromium profile with the camera extension already
loaded, on the PAAI demo page: a chat box that sends each order to the
OpenClaw `cascade-demo` agent, with three live cameras beside it.
No extension setup, token paste or permission prompt is needed.

That page, `http://127.0.0.1:8092`, is the demo's own web UI, not the
OpenClaw interface. For the native OpenClaw dashboard or access from another
computer, see [Open OpenClaw](#open-openclaw).

The same registered launcher can be opened from a graphical terminal:

```bash
gio launch "$HOME/paai-spark/runs/.install/paai-spark.desktop"
```

Success: the connected chat and advancing Worktop (`cam0`), Side (`side`)
and Kitchen (`proof`) views. Attaching normally takes a few seconds.
The launcher file above is also the target for desktop automation.

Write normal English in the chat. First inspect the table:

> What can you see on the table?

Then send one order and wait for it to finish:

> Could you put the green cube in the green square?

Check that the cube is released inside the square. To reset, send:

> Let's start over.

After reset, try the second order:

> Please put the orange in the open box.

Check that the orange is released inside the box. Reset before the next visitor.

### 6. Stop cleanly

Wait for the last request to finish. This stops only this checkout's stack.
The second command checks that no owned process remains. Allow up to a minute.

```bash
bash -e <<'BASH'
cd "$HOME/paai-spark"
./run.sh down
remaining=$(./run.sh down --dry-run)
printf '%s\n' "$remaining"
case "$remaining" in *'would stop'*) exit 1 ;; esac
printf 'STOPPED\n'
BASH
```

Success: `STOPPED`, with no `would stop` lines. Keep the installed files.
Next time, click **PAAI (Spark)** to start. The remaining sections are
reference material; the Bonus Track is optional.

## System prerequisites

Use Linux `aarch64`, glibc 2.35 or newer, a working NVIDIA driver compatible
with CUDA 13, and a CUDA toolkit with `nvcc`. DGX OS ships the driver and
CUDA toolkit; see the [DGX Spark software versions](https://docs.nvidia.com/dgx/dgx-spark/release-notes.html).
The setup check reports missing host components; it does not install them.

If the quick-path check reports missing packages, an administrator runs this
one command, then repeats Step 1. Skip it when Step 1 already passes.

```bash
sudo apt-get update && sudo apt-get install -y git git-lfs curl python3 ca-certificates build-essential libgomp1 libglib2.0-bin gnome-terminal chromium-browser && printf 'PACKAGES_READY\n'
```

Use Chromium for the automatic extension. Branded Google Chrome 137 and
newer [does not support this loading flag](https://groups.google.com/a/chromium.org/g/chromium-extensions/c/1-g8EFx2BBY/m/S0ET5wPjCAAJ).
The launcher handles the Chromium snap's profile and file access paths.
In GNOME, the app-grid entry works immediately. A desktop-file manager may
require **Allow Launching** for an untrusted icon; use the app-grid entry.

Keep at least 150 GiB free in the home volume, with room for later evidence
and shader caches. Also check another cache volume if you override
`XDG_CACHE_HOME` or `UV_CACHE_DIR`. The model and projector use 18.85 GB;
the compressed kitchen archive adds 0.71 GB before extraction.

Outbound HTTPS must reach GitHub and Git LFS, Hugging Face and its file
hosts, PyPI, `pypi.nvidia.com`, `download.pytorch.org`, `astral.sh`,
`openclaw.ai`, Node.js and the npm registry, including redirects.

Start in a clean shell. Remove `ISAACSIM_PATH`, `ISAACSIM_PYTHON_EXE`,
`CASCADE_QWEN_MODEL`, `CASCADE_QWEN_MMPROJ`, `LLAMA_DIR` and `LLAMA_SERVER`
overrides if you used another installation. The quick path uses private
runtimes in this checkout.

## Installed files

Paths are relative to `$HOME/paai-spark`.

| Component | Location |
| --- | --- |
| App, Python 3.12 and CUDA perception packages | `.venv/` |
| Isaac Sim `isaacsim[all,extscache]==6.1.0.0`, Python 3.12 | `.isaacsim/` |
| OpenClaw 2026.9.3 and private Node.js | `.openclaw-cli/` |
| Pinned Qwen Q4 model and BF16 vision projector | `models/qwen3.8-27b/` |
| Pinned llama.cpp CUDA build and receipt | `.llama.cpp/build/` |
| Robot Git LFS files and perception weights | `assets/`, `models/` |
| Earlier kitchen from this guide's pinned revision | `demo/scene/` |
| Source, package and consent receipt; reusable environment | `runs/.install/install.json`, `runs/.install/env.sh` |

The installer also uses `$HOME/.local/bin/uv`, `$HOME/.cache/uv` and
`$HOME/.cache/cascade/installers`. It obtains Python 3.12 when needed.
Preparation registers the desktop launcher. A rerun reuses verified files.

Consent belongs to the absolute checkout path recorded in `install.json`.
Moving the checkout requires installation with explicit consent again.
Do not edit consent receipts. `--dry-run` and `--check` grant no consent.

## Verify the result

The installation receipt must show the checkout's `repo`, the pinned
`source_commit`, `source_dirty: false`, `profile: "spark"`, `brain: "qwen"`
and `eula_accepted: true`.

Step 4 validates `runs/.install/desktop-latest.json` and
`runs/.launch/profile-cascade-demo/proof.json` without printing their full
contents. It checks these facts while the demo is running:

- Health has `"ok": true`. The desktop receipt has `action: "launch"` and
  `exit_code: 0`.
- The current `proof.json` has `verified: true`, `sim: "isaac"` and model
  `Qwen/Qwen3.8-27B`. On a fresh start, `attached: false` in the desktop
  receipt and proof `started_at` between its `started_at` and `finished_at`
  establish that this launch ran the proof.
- Reopening the launcher records `attached: true`. It keeps the existing
  proof, whose `session_id`, `started_at` and `process` must match the
  desktop receipt's `proof` object. The launcher verifies that the owned
  simulation process is still live before attaching.
- Its `cases` include green cube to green square and orange to open box.
  Each has `physics.pass: true`, `physics.event_cameras.pass: true`, and
  `props_reset` containing `green_cube` or `orange` respectively.
- Each case's `evidence_dir` holds native turns,
  `physics/gpu-physical-audit.json`, `physics/placed-<camera>.jpg` and
  `physics/reset-<camera>.jpg`. All three live cameras must advance.

`PREPARED` means the dependencies are ready. `READY` requires the physical
proof. `STARTED (UNVERIFIED: robot proof skipped)` is not an accepted result.
An old receipt or an English success message is not proof.

OpenClaw may report "Placement unverified" when a tool checks only the
object's center. Inspect the placement and the physical audit before
calling the result successful.

## Ports and coexistence

| Loopback port | Service |
| --- | --- |
| 8080 | This checkout's local Qwen API |
| 8091 | This checkout's read-only camera surface |
| 8092 | This checkout's local chat and camera extension |
| 8093 | Optional authenticated visitor surface |
| 8611 | Isaac bridge |
| 18790 | This checkout's OpenClaw gateway, profile `cascade-demo` |
| 4043 | Optional dedicated ngrok agent API |

The demo does not use port 8090. The separate personal OpenClaw gateway
on 18789 stays untouched. Spark does not start occupancy or GraspGen-X
sidecars on 5557/5556. OpenClaw state stays under
`runs/.launch/profile-cascade-demo/openclaw/`.

Run one Spark stack at a time. Before the first start, `ss -ltnp` shows
occupied ports. A service owned by another application must stay running;
resolve the conflict before launching. Never kill a process by its port
or name. Reopening this checkout's launcher attaches to its running stack.
Never expose the OpenClaw gateway, Qwen API, Isaac bridge or MCP publicly.

## Open OpenClaw

The launcher opens the PAAI demo page at `http://127.0.0.1:8092`. Its chat
box relays each message to the OpenClaw `cascade-demo` agent, but the page is
the demo's own web UI, not OpenClaw's.
There is no `/guide` or `/openclaw` page on the Spark; those pages belong to
the Brev booth deployment.

You never need to copy the gateway token. It changes on every launch, and
both the launcher and the command below read the current one.

To open the native OpenClaw Control UI on port 18790 with the current token,
run this in the Spark's desktop session:

```bash
cd "$HOME/paai-spark" && ./run.sh dashboard
```

On the Spark desktop, this opens OpenClaw in the PAAI Chromium profile, where
the camera extension is loaded. Click the extension or press **Ctrl+Shift+Y**
to show Kitchen, Worktop and Side beside the chat. The one-time sign-in link
expires after 10 minutes and is not written to any receipt or log.

Without a desktop session, the same command prints an SSH tunnel command and
the local address to open. That browser does not load the camera extension
unless you install it there yourself.

To use the demo from another computer, forward both ports over SSH, then open
`http://localhost:8092`. Replace `user@spark-host` with your Spark login:

```bash
ssh -L 8092:127.0.0.1:8092 -L 18790:127.0.0.1:18790 user@spark-host
```

Forward these ports only over SSH. For a shareable URL, use the optional
[Bonus Track](#bonus-track-always-on-demo-on-ngrok-for-testing-purposes).

## If something goes wrong

Keep `$HOME/paai-spark-install.log`. For an install error, correct only the
reported package, network or disk issue. Repeat the same pinned install
command once. The installer already has bounded download retries.

For a failed launch or stalled camera, inspect the log path printed by the
launcher. From a terminal, stop this checkout and retry once:

```bash
cd "$HOME/paai-spark" && ./run.sh down && python3 scripts/desktop.py launch --repo "$PWD" --headless --no-open
```

To inspect the current launch without starting it again, use these read-only
commands. The first prints the short desktop receipt and its full log path:

```bash
python3 -m json.tool "$HOME/paai-spark/runs/.install/desktop-latest.json"
python3 "$HOME/paai-spark/scripts/spark_browser.py" --repo "$HOME/paai-spark" --check
```

Full logs are in `runs/.install/desktop-*/progress.log` and
`runs/.launch/profile-cascade-demo/`: `qwen.log`, `isaac_bridge.log` and
`openclaw-gateway.log`. An early bootstrap error may leave only the install
log. Exit 2 means rejected arguments, consent or host prerequisites. A
read-only check that exits 3 reports `MISSING` components.

If the same failure returns, stop and report the command, exit code and
log path. Keep incomplete environments and failed evidence for diagnosis.
Do not extend timeouts repeatedly, skip the robot proof, edit receipts or
change the model or physics settings to obtain `READY`.

For a missed grasp or `did not settle at home`, reset and inspect before
another order. Send only one order at a time.

## Agent checklist

1. Complete the quick path in order. Require the operator's EULA consent.
2. Require `PREPARED`, exit 0, and matching source, path and consent fields.
3. Require read-only `--check` exit 0, then launch exit 0 and `READY`.
4. Verify current health, proof, both physics cases, resets and all cameras.
5. Open the registered launcher in a graphical session. Verify the connected
   chat, loaded extension and cameras. Send a natural-English order and reset.
6. Require clean stop exit 0 and no `would stop` entries. Preserve evidence
   and unrelated services. Report failed or unverified results honestly.

See [Spark delivery](SPARK_DELIVERY.md) for file identities and acceptance details.

## Bonus Track: Always-on demo on ngrok for testing purposes

This is optional and is not part of the official setup. At events, the demo
runs on the Spark with its own monitor and keyboard.

Install the Linux arm64 executable from [ngrok's download page](https://ngrok.com/download/linux)
and use an ngrok account with a stable domain. Keep its token in a private
ngrok configuration file. Create a private JSON file containing
`{"username":"<visitor-name>","password":"<visitor-password>"}` and set its
mode to `0600`. Use a password of at least 12 characters. The ngrok
configuration file must also have mode `0600`. Replace every placeholder
below. Never put credentials in the domain or commit these files.

The desktop user's systemd user manager must have linger enabled to survive
logout. If `loginctl show-user "$USER" -p Linger` does not show `Linger=yes`,
ask the administrator to enable it before using this optional track.

Enable:

```bash
python3 "$HOME/paai-spark/scripts/spark_public.py" enable --repo "$HOME/paai-spark" --domain '<your-domain>.ngrok.dev' --auth-file '<private-visitor-auth.json>' --ngrok '<path-to-ngrok>' --ngrok-config '<existing-ngrok.yml>'
```

Success: `PUBLIC ENABLED`. This registers the services. A stopped demo
still needs its normal startup time. Check after it reaches `READY`:

```bash
python3 "$HOME/paai-spark/scripts/spark_public.py" check --repo "$HOME/paai-spark"
```

Success: `PUBLIC READY`. Disable the public endpoint and its demo stack:

```bash
python3 "$HOME/paai-spark/scripts/spark_public.py" disable --repo "$HOME/paai-spark"
```

Success: `PUBLIC STOPPED`.

The dedicated `paai-spark-demo`, `paai-spark-visitor` and `paai-spark-ngrok`
user services are supervised separately. The demo service restarts after a
crash or a failed health check, at most three starts per hour. A launch that
never reaches `READY` is not restarted, because it would rebuild the whole
stack in a loop. Read `runs/.install/desktop-latest.json`, fix the cause, then
run `enable` again, which clears the failed state. The visitor and ngrok
services restart until the demo is ready. The desktop launcher attaches to
this same installation. The existing ngrok sessions stay unchanged.

An authenticated visitor gets three live cameras and chat with the
`cascade-demo` agent and its six attendee tools. Send one order at a time;
"Let's start over." resets. Requests without credentials receive HTTP 401.
The public surface excludes the OpenClaw Control UI, administration,
configuration, tool policy and raw backend APIs. The gateway token stays on
the Spark and is never sent to a browser. Disabling stops these dedicated
services and keeps the installation files.
