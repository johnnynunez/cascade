# PAAI, Physical Agentic AI

## Staff guide · Build a Claw

PAAI lets an attendee ask about a simulated kitchen and give its robot arm a
task in plain English. OpenClaw uses CASCADE tools to inspect the scene and
move objects in NVIDIA Isaac Sim 6.1. This is the initial event version.

At the booth, attendees see the chat, the tool being used and three camera
views. **Worktop** shows objects and the green delivery square. **Side** shows
the lift and release. **Kitchen** shows the whole room. The robot is simulated.

## Introduce it in 20 seconds

> This is PAAI, Physical Agentic AI. You can ask about the kitchen or give the
> robot a task in plain English. We'll watch it through three cameras, then
> check whether it moved the right object. Start by asking what it can see.

## Before a group arrives

1. Open the attendee camera page and check Kitchen, Worktop and Side for
   current images. Wait for **Live**; if a view looks frozen, follow the
   camera recovery steps below.
2. Use the connected OpenClaw chat on the booth computer. Keep Worktop beside
   it. This staff page and the attendee link do not send robot commands.
3. Wait for any active task to finish. Reset once, then check that the arm
   has returned home and the loose props are back in their starting places.

## Run one task

Ask the attendee to inspect the scene first. Compare the reply with Worktop:
the green cube and the green delivery square are different things. Help them
choose one visible object and one clear destination.

Send one movement request and wait for the full reply. A pick can take about
one to two minutes. Show the lift in Side and the final position in Worktop.
Read the result together, then reset before the next task.

### Prompts to try

| Task | Prompt |
| --- | --- |
| Inspect the scene | “What can you see on the worktop? Do not move anything.” |
| Locate the zone | “Where is the green delivery zone? Do not move anything.” |
| Move the orange | “Pick up the orange and put it in the green square. Then stop.” |
| Try the lemon | “Pick up the lemon and put it in the green square. Then stop.” |
| Move the can | “Pick up the tomato can and put it in the green square. Then stop.” |
| Reset | “Reset the scene.” |

The green square is the delivery zone. Use the single loose orange near the
arm; the oranges on the plate are outside the grasp area. The lemon can be
hard to identify and grasp. Keep the destination empty and reset between
movement requests.

If the description is unclear, use this guided observation prompt:

> Call get_observation once for a fresh view. Describe only the objects,
> colors and shapes visible in the returned image. Say when something is
> unclear; do not treat configured destinations as proof that an area is
> visible or empty. Use at most three short sentences. Do not move anything
> or call another tool.

## What a successful run looks like

The requested object lifts clear of the counter, stays in the gripper during
the move, and settles fully inside the green border after release. Check
Side for the lift and open jaws, then Worktop for the final position.

After the movement, ask: “Check the object's final position using physics.
What does the evidence confirm, and what is still unverified?”

Keep an **unverified** result as unverified. A measured position near the
zone's center does not by itself prove that the whole object is inside or
that the gripper released it. If the views and readback disagree, pause and
have the demo operator review the result.

Successful orange and tomato-can pick/place cases have been checked with
camera images and simulator physics. Guided scene inspection, locating the
delivery zone, authenticated camera streaming and reset/recovery have also
been tested. Each new run still needs its own check.

## If something goes wrong

**Missed grasp or dropped object.** Let the current tool finish. Say, “The
gripper missed; this attempt did not complete the task.” Read the result,
then reset once. Do not queue retries while the arm is moving. If it keeps
missing, return to scene inspection and ask the demo operator for help.

**Old or frozen camera.** Pause new movement requests. Keep the camera page
open while it reconnects, then refresh it if needed. Check all three views;
a Live label alone is not enough if the image stays frozen during motion.
If a view remains stale, let the active task finish and have the operator
check the feed. Request a fresh observation before the next movement.

**Reset did not happen.** A new chat does not reset the scene. After the
current tool finishes, use the prompt below once and wait for its result:

> Call reset_scene now for a fresh reset. Call it exactly once and wait for
> the tool result. Reply briefly in English, then stop.

Check for an empty gripper at home, the loose props back at their starting
positions and current images from all three cameras. Leave a failed reset
result in the chat and ask the operator to recover the demo before another
turn.

## Known limits

- The demo uses a small set of prepared objects and robot skills. It does
  not demonstrate a general kitchen robot or transfer to physical hardware.
- Object descriptions can be wrong, especially with the lemon or partially
  hidden objects. Compare them with the current image before moving.
- Grasps and unusual arrangements can fail. Do not promise that every
  request will work.
- Camera previews update only a few times per second. A smooth video does
  not mean the scene is being captured at that rate.

## Live Brev access

The event demo runs on Brev **paai-demo-rtx6000**. The operator shares the
current HTTPS link and Basic Auth credentials privately. These fields are
placeholders; keep credentials out of this file and screenshots.

| Item | Staff handoff |
| --- | --- |
| Attendee page | `<public-https-url>/` |
| Separate staff page | `<public-https-url>/staff/` |
| Username | `<staff-username>` |
| Password | `<staff-password>` |

Both public pages require authentication. Use the booth computer's connected
chat for commands. OpenClaw administration and raw control endpoints stay
private.

## Setup and further help

- [DGX Spark setup](DGX_SPARK_SETUP.md): prerequisites, installation and a
  smoke test on Spark.
- [Brev deployment](BREV.md): operator setup and recovery.
- [Private booth walkthrough](../web/guide/index.html): detailed instructions
  for the connected booth computer.

PAAI was built together by Asier Arranz and Johnny Núñez. Kitchen and orange
assets by [Lightwheel](https://github.com/LightwheelAI/Lightwheel_Kitchen),
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/), adapted for
this demo.
