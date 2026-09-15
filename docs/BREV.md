# Brev kitchen deployment

The tested machine was an AWS `g7e.2xlarge` with one NVIDIA RTX PRO 6000
Blackwell Server Edition (97,887 MiB VRAM), eight vCPUs, 62.27 GiB RAM and a
separate 1.7 TB NVMe volume. Isaac Sim 6.1 runs CUDA PhysX. The production
brain is Qwen3.8-27B Q8_0 with its BF16 vision projector; every model layer,
KV cache and supported model operation stays on the GPU. A recorded Qwen/Isaac run used 36,590 MiB of VRAM with the optional video
profile disabled.

## Choose and connect a machine

Use `brev search` or the console to inspect availability before creating a
new instance. The tested instance type can be selected with:

```bash
brev create cascade-kitchen --type g7e.2xlarge --flex-ports
```

When searching, `--gpu-name`, `--min-vram 90`, `--min-disk 1700` and
`--flex-ports` filter the candidates. The disk filter does not resize the
root disk. Confirm the mounted storage after boot. Flex ports are useful
for firewall configuration; this deployment does not open Isaac control
ports publicly.

Brev's managed SSH relay can accept a TCP connection and then close it even
while the VM is running. A closed relay is not proof that the GPU machine
needs a reset. On this deployment, direct public port 22 was also
unreachable. Tailscale between the controller and the VM provided working
access to the VM's own SSH server. Check the node-key expiry of **both**
devices: the controller's key expired during deployment while Brev and its
public camera service remained available.

The controller needs Python 3.10+, `rsync`, the Brev CLI and Tailscale.
Authenticate the controller with `brev login` to obtain its Brev identity at
`~/.brev/brev.pem`, and restrict that file to mode 0600. Join the controller
and VM to the same tailnet, permit access to the VM's port 22, and retain
the `ubuntu` account's passwordless sudo access. Keep the private key on the
controller.

Supply an executable OpenSSH-compatible transport wrapper. It must accept
`-F <private-config> <alias> <remote-command>` and forward arguments and
standard input/output unchanged, including for rsync transfers. Set its
absolute path before running deployment commands:

```bash
export PAAI_TRANSPORT_WRAPPER="/usr/local/bin/cascade-transport"
```

The default remains `~/.local/bin/totally-not-ssh.sh`; this local wrapper is
not included in the repository. Home-relative paths are expanded, and the
access gate checks the executable before writing private configuration or
running a remote command. It creates a private alias for the selected
Tailscale peer and preserves existing host entries. No Mac relay or second
computer is needed.

## Prepare once, then start

Copy [the RTX profile](../deploy/brev/profiles/brev-rtx6000.json) outside the
checkout and set `network.tailscale_hostname` to your selected machine.
Keep models, Docker layers, build inputs, Isaac caches and recordings below
`/opt/dlami/nvme/paai-demo`; the root filesystem is only about 117 GB on the
tested machine. A separate Docker daemon uses
`unix:///run/paai-demo-docker.sock`, leaving the system daemon's data alone.

The deployment requires a separately supplied source and licensed asset
bundle with a relative checksum manifest, plus the pinned OpenClaw and
CLIP archives. This kitchen deployment's Lightwheel assets, perception weights
and pinned runtime archives are supplied separately. Supply the external
files at the relative paths listed in
[`bundle_assets.json`](../deploy/brev/bundle_assets.json), preserving their
license notices.

CASCADE's source license remains MIT. The runtime installs
[Ultralytics](https://www.ultralytics.com/license) and its pinned
[CLIP fork](https://github.com/ultralytics/CLIP/blob/a13192f8cb767260d7dfd98c843b0716593169e7/LICENSE)
under AGPL-3.0 terms; separate vendor licensing may apply. Lightwheel assets
use CC BY-NC 4.0. Preserve each dependency's notices and source obligations
when distributing a runtime. Attribution alone does not establish permission
for commercial use of the assets.

Assemble the distribution once outside the checkout, then import it;
valid receipts let subsequent operations reuse the bytes:

```bash
python3 deploy/brev/build_bundle.py --assets /srv/cascade/assets \
  --output /srv/cascade/distribution --profile /srv/cascade/site.json
./deploy/brev/deploy.sh prepare --profile /srv/cascade/site.json \
  --source /srv/cascade/distribution/source \
  --manifest /srv/cascade/distribution/PORTABLE_BUNDLE.json \
  --openclaw /srv/cascade/openclaw.tar.zst \
  --clip /srv/cascade/clip-source.tar.gz
./deploy/brev/deploy.sh install --profile /srv/cascade/site.json
```

The normal entry point after preparation is:

```bash
./deploy/brev/deploy.sh start --profile /srv/cascade/site.json
```

`status`, `restart` and `stop` use the same entry point and profile. They
address the owned application containers, not the Brev instance. A changed
installed source bundle is rejected rather than silently replacing it.
Restart checks preparation, the runtime image, source compatibility and
preflight before stopping the owned services.
Provision a separate target when validating a new bundle. A repository
`STOP` file prevents new controller operations.

Preflight checks the exact GPU, compute capability, free VRAM, separately
mounted data volume and ownership of existing GPU processes. The RTX
profile requires 70,000 MiB free before startup and 12,288 MiB headroom
while running. Keep one simulator on the GPU. Do not alter the installed
NVIDIA driver or kernel. Compose explicitly accepts the Isaac EULA.

The first boot had no `docker0`; bridge-network containers failed with
`Device does not exist`. Restarting the **system Docker service** restored
bridge networking and DNS:

```bash
sudo systemctl restart docker
```

Do this only when that fault is present. The dedicated demo daemon and the
application container already have restart policies; routine disconnects
do not require restarting the VM.

## Visitor access

The private frontend supplies `/guide/`, `/openclaw/` and `/cameras/` on
port 8092. OpenClaw bootstrap and administration stay on the trusted
Tailscale/loopback path. The public visitor serves its own page, status, three camera videos and JPEG
fallbacks through an allowlist on loopback 8093.

Enable the optional public visitor in your external profile before preparing
the distribution. Put the ngrok token in a mode-0600 file owned by the Brev
runtime user; the profile contains its path, never the token:

```json
"public_visitor": {
  "enabled": true,
  "token_file": "/secure/location/ngrok.token",
  "local_auth": false,
  "video": true
}
```

The same `deploy.sh install`, `start`, `status`, `restart` and `stop` commands
manage the owned visitor and ngrok services alongside the application.
`status` prints the current HTTPS visitor URL after checking authentication,
camera availability and blocked administrative routes. An unchanged install
keeps healthy services running. The host helper `deploy/brev/public.py` exposes
the same visitor-only operations when needed for service maintenance.

The generated basic-auth credentials stay in
`data/private/public-visitor/auth.json` under the deployment root. Share only
those visitor credentials with the intended viewers. If the account rejects
ngrok traffic policies, set `local_auth` to `true` to omit the ngrok traffic policy. The visitor server enforces local basic auth
in both modes, so changing the tunnel policy cannot remove authentication.
ngrok request inspection remains disabled. See
[ngrok's Linux instructions](https://ngrok.com/download/linux).

With `video: true`, the visitor shares each existing camera feed through one
host FFmpeg/NVENC encoder. Chrome receives H.264 fragmented MP4 through the
same authenticated HTTPS origin and MediaSource; no extra public port or
WebRTC control endpoint is needed. The tested profile is 960 × 540 with a
nominal 3 fps timeline, 2 Mbps CBR, one-second keyframes and no B-frames.
The camera source updates approximately 2.5–3 times per second; duplicate
frames and catch-up playback do not represent additional scene updates.
This preserves the active Isaac process and does not turn its camera source
into a 30 fps renderer. The host needs FFmpeg with `h264_nvenc`; the installer
installs the Ubuntu package if it is missing. It never selects a software
video encoder.

The player discards old buffered video, reconnects after a stalled stream and
falls back to fully decoded JPEG snapshots while video is unavailable. On
14 September 2026, all three cameras played through the authenticated public
URL in Chrome; a finite encoded sample confirmed the codec, dimensions,
frame rate, bitrate and keyframes. The separate native Isaac RTSP/WebRTC
profile remains disabled and has no live acceptance. A camera image or
encoded video is not evidence of a successful physical grasp.

The encoder keeps its output at 960 × 540 when the smaller offline card
replaces a live capture. This transition was checked with NVENC on Brev using
an isolated feed that alternated the two image sizes. All three public camera
videos were then checked in Chrome after reloading the visitor service.
The simulator, model and OpenClaw kept running during that viewer update.

## Deployment boundaries

The source layout in this branch separates the controller, runtime, scene,
web UI and extension from development evidence. The deployed original
layout established GPU vision/tools and private/public camera access; the
relocated publication layout requires its own complete deployment check.
Managed services in this revision run until an explicit stop, a termination
signal or a required-component failure; startup and health checks retain
finite deadlines. On 14 September 2026, the original deployment recovered
through its existing Docker restart policy after its older worker timers
expired. The new runtime passed its native authentication, tool-discovery
and CUDA PhysX startup checks, and all three cameras resumed. The review
did not trigger a restart of the application container or protected services.
This recovery used the original layout. Complete relocated deployment and
sustained-runtime acceptance remain separate checks.

Kitchen tool sessions request cameras at most four times per second.
OpenClaw retires idle tool runtimes after one minute, checked periodically.
Active requests keep their runtime. The next request in the same chat creates
a fresh runtime if needed. This limits CPU use between visitors; it does not
reset the kitchen.

The model keeps its 16,384-token context window. Chat replies use a 1,024-token
request limit in `extra_body.max_completion_tokens`: the native image estimate
can reduce the ordinary `maxTokens` setting to one token. OpenClaw retains
2,048 recent tokens when compacting a conversation. Model requests time out
after 180 seconds. Old conversations can still give poor replies; start a new
session for each attendee, after the reset finishes. The
[staff guide](BOOTH_GUIDE.md) covers recovery.

If the browser disconnects during Reset, reconnect and read the original tool
result before trying again. The button does not resend an uncertain order.
Readiness briefly showed unavailable during a live reset while cameras kept
updating, then recovered without a restart. Wait for Ready before the next order.

`annotated_view` sends OpenClaw the rendered image and its numbered-object
key. Badges are tracked estimates, which can be stale or mislabeled. The chat
can also miscopy coordinates from tool results. Check the measured tool output
when a position matters; the image and badges do not establish grasp reachability.

The bounded live object matrix passed for orange and tomato can. The first
lemon trial failed to lift. A later single Brev trial lifted it and reached the
green zone, but failed strict settling limits. Lemon placement remains
unverified. Orange-on-can failed stability and was safely reset. The unpacked
Chrome extension passed on Brev in its camera-companion scope.

Named fruit queries keep their configured visual description when they begin
with “the”, “a” or “an”; other qualifiers stay part of the query. On Brev,
`localize_object` resolved `the lemon` from current yellow pixels and depth.

Kitchen profiles cap each pick at two grasp attempts. For the green square
and open box, a tool's measured distance from the configured center is
reported as unverified for full containment and release. The successful
object trials above used separate camera and physics audits of those outcomes.

The DGX Spark is arm64/GB10. It needs architecture-specific Isaac and model
serving setup; the x86_64 container is not a Spark image. Upstream's
Cosmos3-Edge profile remains separate and has no certification from the
Qwen deployment. See [the Spark plan](DGX_SPARK_BREV.md).

For booth support, contact Johnny Núñez and Asier Arranz on Slack.
