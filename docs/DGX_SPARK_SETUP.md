# PAAI on DGX Spark

This guide follows the existing [Spark installer](SPARK_DELIVERY.md). It is
for preparing a separate Spark installation of PAAI, Physical Agentic AI.
The live Build a Claw kitchen demo runs on Brev; Spark still needs its own
camera, GPU physics, tool-call and reset acceptance run.

## 1. Check the machine

Use a DGX Spark with Linux `aarch64` and NVIDIA DGX OS 7. NVIDIA's
[Isaac Sim 6.1 requirements](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/installation/requirements.html)
list **580.159.03** as the tested Spark driver. Check the installed driver
against that support guidance. The installer checks that it works; it does
not qualify every driver version or replace drivers.

```bash
uname -m
cat /etc/os-release
getconf GNU_LIBC_VERSION
nvidia-smi
free -h
df -h "$HOME" /tmp
ss -lnt
systemctl --user list-units --type=service --state=running
command -v git curl python3
python3 -c 'import ctypes; ctypes.CDLL("libgomp.so.1")'
```

The host needs Bash, Git, curl, trusted CA certificates and `libgomp1`.
The installer requires glibc 2.35 or newer.
Keep existing GPU jobs and services running; arrange a separate rehearsal
window if their memory use leaves too little room. Spark shares memory
between CPU and GPU, so inspect `free` as well as `nvidia-smi`.

Allow disk space for Isaac, model snapshots, isolated environments and
download/shader caches. The dry run does not estimate peak storage. Prepare
downloads before the event. Outbound HTTPS must reach GitHub,
`raw.githubusercontent.com`, `astral.sh`, PyPI, `pypi.nvidia.com`,
`download.pytorch.org`, Hugging Face, `openclaw.ai`, `nodejs.org`, the npm registry and
NVIDIA's online asset host:
`omniverse-content-production.s3-us-west-2.amazonaws.com`. Allow their
artifact/CDN redirects too.

## 2. Choose the source and inspect the plan

Use a reviewed, tested commit. The remote bootstrap URL and `--ref` must
name the same commit. See [the existing one-command flow](SPARK_DELIVERY.md#one-command-per-machine).
From a checkout of that commit:

```bash
bash scripts/bootstrap.sh --dir "$PWD" --profile spark --dry-run
bash scripts/install.sh --dir "$PWD" --profile spark --check
```

Both commands are read-only and start no services. `--check` exits with
code 3 when an environment or asset is missing; that is expected before a
first installation. A passing check verifies preparation, not a live demo.

## 3. Prepare Isaac and the application

Isaac Sim 6.1 requires Python 3.12. The installer uses separate environments
for CASCADE (`.venv`), managed Isaac (`.isaacsim`) and the local reasoning
service. It installs its own OpenClaw CLI under `.openclaw-cli` and obtains
`uv` if needed. Native Python/CUDA packages are selected for arm64; do not
copy environments or binaries from Brev. The exact package pins remain in
[Spark delivery](SPARK_DELIVERY.md#isolation-and-defaults).

If a complete Isaac **6.1.0** source/standalone release already exists,
select its directory containing `python.sh` using the existing flow:

```bash
export ISAACSIM_PATH="$HOME/Projects/isaac/IsaacSim/_build/linux-aarch64/release"
bash scripts/install_isaac.sh --dir "$PWD" --check
```

Use that path only when the release exists. An explicit broken or older
release fails the check. Reuse leaves its embedded packages alone. Without
a selected/discovered release, the installer prepares
`isaacsim[all,extscache]==6.1.0.0` in `.isaacsim`. NVIDIA also recommends
[a dedicated Isaac environment](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/installation/install_python.html).

After reviewing and accepting the NVIDIA license, prepare without starting
the demo:

```bash
bash scripts/install.sh --dir "$PWD" --profile spark --accept-eula --prepare-only
bash scripts/install.sh --dir "$PWD" --profile spark --check
```

For a new machine, the same installer can prepare and launch in one command
after the tested ref is published:

```bash
CASCADE_REF='<tested-commit>'
curl -fsSL "https://raw.githubusercontent.com/johnnynunez/cascade/$CASCADE_REF/scripts/bootstrap.sh" | bash -s -- --ref "$CASCADE_REF" --accept-eula
```

## 4. Start, check and recover

From the prepared checkout, launch the complete stack:

```bash
python3 scripts/desktop.py launch --repo "$PWD"
```

The desktop controller supervises the local reasoning service, restores the
selected Isaac environment and runs the physical proof. It also registers
CASCADE with the dedicated OpenClaw `cascade-demo` profile and checks its
tools. Use this entry point for full startup; the lower-level `run.sh`
launch expects the reasoning service to be running already.

Spark selects the release's `isaacsim.exp.full.newton.kit` experience and
requires the Newton engine. The bridge requests `cuda:0` by default. During
rehearsal, verify the bridge's GPU attestation in `isaac_bridge.log`: the
actual backend must be `newton`, both device fields must identify CUDA,
and the tensor view must have a CUDA context. A requested device or a
successful package import alone does not establish GPU physics.

Use the existing commands and receipts for status:

```bash
./run.sh check isaac
.openclaw-cli/bin/openclaw --profile cascade-demo gateway status
python3 -m json.tool runs/.install/desktop-latest.json
python3 -m json.tool runs/.launch/profile-cascade-demo/proof.json
```

Cold collision preprocessing and shader warmup can exceed ten minutes;
the launcher allows 1,200 seconds for Isaac by default. Logs are under
`runs/.install/` and `runs/.launch/profile-cascade-demo/`. A fresh successful
proof must show a physics-confirmed pick and reset in the same world.
Preparation alone does not print a verified READY result.

For the smoke test, inspect the scene, check that every configured camera
advances, move a supported object, confirm its release, then reset and
inspect again. Use the [full Spark acceptance checklist](SPARK_DELIVERY.md#cold-start-acceptance-on-a-spark)
before offering the machine to attendees.

For a missed grasp, inspect and reset before retrying. For a stale camera,
check the bridge log and current frames. If this installation needs a
restart, `./run.sh down` stops its owned processes; then repeat the desktop
launch. Leave unrelated services and occupied ports alone. Keep the logs
when a check fails, fix the reported prerequisite, and rerun the same ref.

## Ports and access

| Port | Use | Access |
| --- | --- | --- |
| 18790 | Dedicated OpenClaw demo gateway | Loopback; private administration |
| 18789 | Existing default OpenClaw gateway | Leave existing service alone |
| 8611 | Isaac control bridge | Loopback; contains raw control operations |
| 8082 | Local reasoning API | Loopback |
| 5556 | Optional grasp service | Private; the shipped stub binds loopback |
| 5557 | Optional occupancy service | Binds all interfaces; restrict with the host firewall |

The Spark installer does not publish the Brev visitor or create an ngrok
tunnel. A public visitor needs a separately configured authenticated proxy
with a small route allowlist. Keep gateway bootstrap, raw control and admin
routes private. Do not expose these service ports directly. Do not install
Hermes or Telegram on the demo host.

## What differs from Brev

Brev runs an x86_64 container with its own event configuration, camera relay,
authentication and supervision. Spark needs arm64 dependencies, GB10/shared
memory qualification and its own Newton rehearsal. The Brev deployment
profile is not a Spark installer. Shared source and successful Brev runs do
not certify a Spark cold start.

Return to the [booth guide](BOOTH_GUIDE.md) for the staff script and recovery
steps used at Build a Claw.
