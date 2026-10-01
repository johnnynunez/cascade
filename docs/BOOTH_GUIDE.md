# Build a Claw with PAAI

Check [current source and acceptance](PROJECT_STATUS_20261001.md) before
presenting. A dated successful run does not certify a newly installed revision;
use the intended checkout and require its current READY and camera checks.

PAAI means Physical Agentic AI. At the booth, staff teach an attendee to use
the normal OpenClaw chat with the Chrome camera extension beside it. The
attendee asks about the scene, gives one order, and watches the simulated arm.

## Read the guide

The [Build a Claw page](../deploy/brev/staff.html) is the main guide for staff
and attendees. Open `/staff/` on the running demo. It has two sections:
**How this works** and **The workflow**.
The operator shares the live link and access details privately.

On the Brev booth deployment, use the private demo’s `/guide/` page to open
the connected OpenClaw chat. The private links (`/guide/`, `/openclaw/` and
`/cameras/`) are reachable only over the booth’s Tailscale network. The
public attendee link shows cameras only.

On a DGX Spark there is no `/guide/` page. Click **PAAI (Spark)** in the app
grid: it opens the PAAI demo page at `http://127.0.0.1:8092`, a chat box that
relays each order to OpenClaw, with the three cameras beside it. For the
normal OpenClaw chat, run `./run.sh dashboard`; see
[Open OpenClaw](DGX_SPARK_SETUP.md#open-openclaw).
With several installations, select the intended checkout using the
[exact desktop launcher command](DGX_SPARK_SETUP.md#5-open-the-demo-in-one-click)
instead of relying on identical app-grid names. The
[setup guide](DGX_SPARK_SETUP.md#validated-release-and-scope) separates the
current source pin from historical installation, five-object and restart results.

For the complete installation and presenter steps in one page, download the
[standalone Spark HTML guide](spark-presenter/index.html) and open it in a
browser. It works offline, has copy controls and prints the expanded reference
sections. It is documentation, not a live control panel.

## One task

Ask “what are you seeing?” and compare the reply with Worktop. Then give one
order, such as “move the orange to the box” or “move the green cube to the
green zone”. Watch the lift in Side and the release in Worktop. Read the
result, then click **Reset** above the OpenClaw conversation. The button sends
the same “Reset the scene.” order you can type in chat. Keep this chat open
while reset runs. A long conversation can delay it. Wait for **Scene reset**
and **Live** in all three camera views. Check that the gripper is empty, the
arm is home and the props are back. If reset is not confirmed, read the chat
before trying again. If you reload the page, read the reset result in chat
before sending another order. Reset is disabled while the chat is disconnected.

Use one operator chat at a time. All connected chats control the same kitchen;
another chat can move an object after your reset.

After the reset finishes, click **+** in OpenClaw for the next attendee. Each
attendee starts a new session. Starting a session does not reset the kitchen.

The current scene has green and pink cubes, a green zone and a wooden box.
“Move the red cube to the red zone” is unsupported here. Do not substitute
the pink cube or another destination. Scene descriptions and grasps can
fail. Keep an unverified result unverified.

The chat can also say the arm is home or the gripper is empty without measuring
either. Use the reset result and cameras to check readiness.

## Setup

[Chrome extension](CHROME_EXTENSION.md) · [Brev](BREV.md) ·
[DGX Spark](DGX_SPARK_SETUP.md)

PAAI was built by Johnny Núñez and Asier Arranz. Its Cocina Asier room uses
owner-authorized geometry and verified CC0 wood. The orange and bowl are
original artwork authored in this repository. See the
[scene provenance notice](../demo/scene/NOTICE.md).
