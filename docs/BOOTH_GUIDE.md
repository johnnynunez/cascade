# Build a Claw with PAAI

PAAI means Physical Agentic AI. At the booth, staff teach an attendee to use
the normal OpenClaw chat with the Chrome camera extension beside it. The
attendee asks about the scene, gives one order, and watches the simulated arm.

## Read the guide

The [Build a Claw page](../deploy/brev/staff.html) is the main guide for staff
and attendees. Open `/staff/` on the running demo. It has two sections:
**How this works** and **The workflow**, with a real OpenClaw screenshot.
The operator shares the live link and access details privately.

Use the private demo’s `/guide/` page to open the connected OpenClaw chat.
The public attendee link shows cameras only.

## One task

Ask “what are you seeing?” and compare the reply with Worktop. Then give one
order, such as “move the orange to the box” or “move the green cube to the
green zone”. Watch the lift in Side and the release in Worktop. Read the
result, then click **Reset** above the OpenClaw conversation. The button sends
the same “Reset the scene.” order you can type in chat. Keep this chat open
while reset runs. A long conversation can delay it. Wait for **Scene reset**
and **Live** in all three camera views. Check that the gripper is empty, the
arm is home and the props are back. If reset is not confirmed, read the chat
before trying again.
Reset is disabled while the chat is disconnected.

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

PAAI was built by Johnny Núñez and Asier Arranz. Kitchen and orange assets
by [Lightwheel](https://github.com/LightwheelAI/Lightwheel_Kitchen),
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/), adapted for
this demo.
