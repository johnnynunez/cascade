"""deploy/openshell/cosmos-plugin: what the Cosmos3-Edge agent profile changes
in the model's input, executed in node on the real plugin module.

The fixture user turn has the exact shape OpenClaw 2026.9.3 sent in a captured
request (`[timestamp] Conversation info: ⟦openclaw:ctx⟧` + a ```json block, then
the message); the plugin must hand the model only the message, keep images and
other roles untouched, and replace the system prompt."""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

PLUGIN = Path(__file__).resolve().parents[1] / "deploy" / "openshell" / "cosmos-plugin" / "index.mjs"
CAPTURED_USER = ("[Mon 2026-10-19 09:49 UTC] Conversation info: \u27e6openclaw:ctx\u27e7\n```json\n"
                 "{\"sender\":{\"id\":\"gateway-owner\"}}\n```\n\nPoint at the lemon.")


def _node(script: str):
    node = shutil.which("node")
    if not node:
        pytest.skip("node not on PATH")
    code = f"import * as plugin from {json.dumps(PLUGIN.as_uri())};\n{script}"
    r = subprocess.run([node, "--input-type=module", "-e", code], capture_output=True, text=True, timeout=30)
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout)


def test_the_captured_user_turn_reaches_the_model_as_the_message_alone():
    out = _node(f"console.log(JSON.stringify(plugin.stripInboundEnvelope({json.dumps(CAPTURED_USER)})))")
    assert out == "Point at the lemon."


def test_only_the_current_request_and_its_own_tool_turns_reach_the_model():
    msgs = [{"role": "user", "content": "Reply OK"}, {"role": "assistant", "content": "OK"},
            {"role": "user", "content": "Put the cube in the box."},
            {"role": "assistant", "content": [{"type": "toolCall", "name": "cascade__pick_and_place"}]},
            {"role": "toolResult", "content": "ok"}]
    out = _node(f"console.log(JSON.stringify([plugin.currentTurn({json.dumps(msgs)}), plugin.currentTurn([]),"
                f" plugin.currentTurn({json.dumps(msgs[:1])})]))")
    assert out == [msgs[2:], [], msgs[:1]]


def test_history_blocks_and_plain_text_are_handled():
    text = ("Chat history since last reply: \u27e6openclaw:ctx\u27e7\n[09:40] gateway-owner: hi\n\n"
            "Conversation info: \u27e6openclaw:ctx\u27e7\n```json\n{\"a\": 1}\n```\n\nOpen the gripper.")
    out = _node(f"console.log(JSON.stringify([plugin.stripInboundEnvelope({json.dumps(text)}),"
                " plugin.stripInboundEnvelope('Go home.'), plugin.stripInboundEnvelope('a ```json\\n{}\\n``` b')]))")
    assert out == ["Open the gripper.", "Go home.", "a ```json\n{}\n``` b"]


def test_the_provider_wrapper_rewrites_only_the_cosmos_models_input():
    script = f"""
const seen = [];
const inner = (model, context) => {{ seen.push(context); return 'stream'; }};
let provider, hook;
plugin.default.register({{
  pluginConfig: {{modelProviderId: 'cascade', modelId: 'cosmos3-edge'}},
  on: (name, fn) => {{ if (name === 'before_prompt_build') hook = fn; }},
  registerProvider: p => {{ provider = p; }},
}});
const ctx = {{provider: 'cascade', modelId: 'cosmos3-edge', streamFn: inner}};
const fn = provider.wrapStreamFn(ctx);
const image = {{type: 'image', data: 'x', mimeType: 'image/jpeg'}};
const messages = [
  {{role: 'user', content: {json.dumps(CAPTURED_USER)}}},
  {{role: 'assistant', content: [{{type: 'text', text: 'Conversation info: \\u27e6openclaw:ctx\\u27e7'}}]}},
  {{role: 'user', content: [{{type: 'text', text: {json.dumps(CAPTURED_USER)}}}, image]}},
];
fn({{provider: 'cascade', id: 'cosmos3-edge'}}, {{systemPrompt: 'long', messages, tools: [1]}}, {{}});
fn({{provider: 'other', id: 'x'}}, {{systemPrompt: 'long', messages, tools: [1]}}, {{}});
console.log(JSON.stringify({{
  rewritten: seen[0], untouched: seen[1],
  hook: hook({{}}, {{modelProviderId: 'cascade', modelId: 'cosmos3-edge'}}),
  otherHook: hook({{}}, {{modelProviderId: 'cascade', modelId: 'nemotron-3.5-bolt'}}) ?? null,
  otherWrap: provider.wrapStreamFn({{provider: 'cascade', modelId: 'qwen3.8-27b', streamFn: inner}}) === inner,
}}));
"""
    out = _node(script)
    rewritten = out["rewritten"]
    assert rewritten["systemPrompt"].startswith("You control a robot arm through the cascade__* tools.")
    assert rewritten["tools"] == [1]
    assert rewritten["messages"] == [{"role": "user", "content": [
        {"type": "text", "text": "Point at the lemon."}, {"type": "image", "data": "x", "mimeType": "image/jpeg"}]}]
    assert len(out["untouched"]["messages"]) == 3, "another model's context is never trimmed"
    assert out["untouched"]["systemPrompt"] == "long"
    assert out["untouched"]["messages"][0]["content"] == CAPTURED_USER
    assert out["hook"] == {"systemPrompt": rewritten["systemPrompt"]}
    assert out["otherHook"] is None and out["otherWrap"] is True


_LOOP_HARNESS = """
const seen = [];
// inner: records each request and answers with the next scripted assistant message
let script = [];
const inner = async (model, context, options) => {
  const payload = {tools: [1], tool_choice: 'auto', parallel_tool_calls: false, max_completion_tokens: 512, messages: []};
  options?.onPayload?.(payload, model);
  seen.push({tools: context.tools, systemPrompt: context.systemPrompt, messages: context.messages, payload});
  const message = script.shift() ?? {role: 'assistant', content: [{type: 'text', text: 'done'}], stopReason: 'stop'};
  const events = [{type: 'start', partial: message}, {type: 'done', reason: message.stopReason, message}];
  return {async* [Symbol.asyncIterator]() { yield* events; }, result: async () => message};
};
let provider;
plugin.default.register({pluginConfig: {modelProviderId: 'cascade', modelId: 'cosmos3-edge', ...CONFIG},
  on: () => {}, registerProvider: p => { provider = p; }});
const fn = provider.wrapStreamFn({provider: 'cascade', modelId: 'cosmos3-edge', streamFn: inner});
const model = {provider: 'cascade', id: 'cosmos3-edge'};
const user = {role: 'user', content: 'Point at the lemon, then go home.'};
const call = (name, args = {}) => ({role: 'assistant', content: [{type: 'toolCall', id: name, name, arguments: args}]});
const ok = {role: 'toolResult', content: [{type: 'text', text: '{"ok": true, "timings": {"move_s": 3.1}}'}]};
const failed = {role: 'toolResult', content: [{type: 'text', text: '{"ok": false, "error": "e-stop latched"}'}]};
const answer = async (messages) => {
  const stream = await fn(model, {systemPrompt: 'long', messages, tools: [1]}, {});
  const events = [];
  for await (const event of stream) events.push(event.type);
  return {events, message: await stream.result()};
};
"""


def _loop(body: str, config: str = "{maxToolRounds: 3}"):
    return _node(f"const CONFIG = {config};\n" + _LOOP_HARNESS + body)


def test_the_first_turn_gets_the_tools_and_the_short_prompt():
    first, = _loop("""
const out = await answer([user]);
console.log(JSON.stringify([seen[0]]));
""")
    assert first["tools"] == [1] and first["systemPrompt"].endswith("Never invent tool results.")
    assert first["payload"] == {"tools": [1], "tool_choice": "auto", "parallel_tool_calls": False,
                                "max_completion_tokens": 512, "messages": []}


def test_after_a_successful_result_a_new_tool_call_from_the_model_runs():
    out = _loop("""
const next = {role: 'assistant', content: [{type: 'toolCall', id: 'h', name: 'cascade__move_home', arguments: {}}],
  stopReason: 'toolUse'};
script = [next];
const res = await answer([user, call('cascade__point_at', {label: 'lemon'}), ok]);
console.log(JSON.stringify({seen, res}));
""")
    seen, res = out["seen"], out["res"]
    assert len(seen) == 1, "a new call needs no report turn"
    out_prompts = _node("console.log(JSON.stringify(plugin.CONTINUE_PROMPT))")
    assert seen[0]["systemPrompt"] == out_prompts and seen[0]["tools"] == [1]
    assert seen[0]["messages"][-1]["content"] == [{"type": "text", "text": '{"ok": true}'}], "results compacted"
    assert res["events"] == ["start", "done"], "the model's stream reaches OpenClaw unchanged"
    assert res["message"]["content"][0]["name"] == "cascade__move_home"


def test_when_the_model_stops_or_repeats_a_call_the_turn_ends_with_the_report():
    out = _loop("""
const stop = {role: 'assistant', content: [{type: 'text', text: '{"ok": true}'}], stopReason: 'stop'};
const again = {role: 'assistant', content: [{type: 'toolCall', id: 'x', name: 'cascade__point_at',
  arguments: {label: ' Lemon'}}], stopReason: 'toolUse'};
const report = {role: 'assistant', content: [{type: 'text', text: 'I pointed at the lemon.'}], stopReason: 'stop'};
const rows = [];
for (const decision of [stop, again]) {
  seen.length = 0;
  script = [decision, report];
  const res = await answer([user, call('cascade__point_at', {label: 'lemon'}), ok]);
  rows.push({kinds: seen.map(s => [s.systemPrompt.slice(0, 40), s.tools.length]), last: seen.at(-1), res});
}
console.log(JSON.stringify({rows, report: plugin.REPORT_PROMPT}));
""")
    for row in out["rows"]:
        assert [k[1] for k in row["kinds"]] == [1, 0], "continuation with tools, then the report without"
        last = row["last"]
        assert last["systemPrompt"] == out["report"]
        assert last["payload"] == {"max_completion_tokens": 200, "messages": []}
        assert last["messages"] == [{"role": "user", "content": (
            "Request: Point at the lemon, then go home.\n\nTool results:\n"
            '1. cascade__point_at({"label":"lemon"}) returned {"ok": true}\n\nReport these results to the user.')}]
        assert row["res"]["message"]["content"][0]["text"] == "I pointed at the lemon."


def test_a_failed_result_or_three_calls_go_straight_to_the_report():
    out = _loop("""
const rows = [];
for (const messages of [
  [user, call('cascade__point_at', {label: 'lemon'}), failed],
  [user, call('cascade__open_gripper'), ok, call('cascade__point_at', {label: 'lemon'}), ok,
   call('cascade__move_home'), ok],
]) {
  seen.length = 0;
  await answer(messages);
  rows.push(seen.map(s => ({tools: s.tools, systemPrompt: s.systemPrompt, messages: s.messages})));
}
console.log(JSON.stringify({rows, max: plugin.MAX_TOOL_ROUNDS, report: plugin.REPORT_PROMPT}));
""")
    assert out["max"] == 3
    fail, capped = out["rows"]
    assert len(fail) == 1 and fail[0]["tools"] == [] and fail[0]["systemPrompt"] == out["report"]
    assert '1. cascade__point_at({"label":"lemon"}) returned {"ok": false, "error": "e-stop latched"}' \
        in fail[0]["messages"][0]["content"]
    assert len(capped) == 1 and capped[0]["tools"] == []
    lines = capped[0]["messages"][0]["content"].split("\n")
    assert [line.split("(")[0] for line in lines if line[:1].isdigit()] == [
        "1. cascade__open_gripper", "2. cascade__point_at", "3. cascade__move_home"]


def test_by_default_one_tool_per_request_then_the_report():
    """The shipped policy (no maxToolRounds): after a successful result the
    model is not asked for another tool; it reports, without tools."""
    out = _loop("""
const report = {role: 'assistant', content: [{type: 'text', text: 'I pointed at the lemon.'}], stopReason: 'stop'};
script = [report];
const res = await answer([user, call('cascade__point_at', {label: 'lemon'}), ok]);
console.log(JSON.stringify({seen, res, report: plugin.REPORT_PROMPT, rounds: plugin.DEFAULT_TOOL_ROUNDS}));
""", config="{}")
    assert out["rounds"] == 1
    assert len(out["seen"]) == 1 and out["seen"][0]["tools"] == []
    assert out["seen"][0]["systemPrompt"] == out["report"]
    assert out["res"]["message"]["content"][0]["text"] == "I pointed at the lemon."


def test_max_tool_rounds_is_validated():
    out = _node("""
const make = cfg => { try { plugin.default.register({pluginConfig: {modelProviderId: 'cascade',
  modelId: 'cosmos3-edge', ...cfg}, on: () => {}, registerProvider: () => {}}); return 'ok'; }
  catch (e) { return e.message; } };
console.log(JSON.stringify([make({}), make({maxToolRounds: 3}), make({maxToolRounds: 0}), make({maxToolRounds: 4}),
  make({maxToolRounds: '2'})]));
""")
    assert out[:2] == ["ok", "ok"]
    assert all("maxToolRounds must be an integer from 1 to 3" in e for e in out[2:])


def test_calls_compare_by_tool_and_arguments_and_failures_read_from_the_result():
    out = _node("""
const c = (name, args) => ({name, arguments: args});
console.log(JSON.stringify([
  plugin.sameCall(c('a', {object: 'Pink cube ', destination: 'open box'}), c('a', {destination: 'open box', object: 'pink cube'})),
  plugin.sameCall(c('a', {object: 'lemon'}), c('a', {object: 'pink cube'})),
  plugin.sameCall(c('a', {}), c('b', {})),
  plugin.resultFailed({role: 'toolResult', isError: true, content: []}),
  plugin.resultFailed({role: 'toolResult', content: [{type: 'text', text: '{"ok": false}'}]}),
  plugin.resultFailed({role: 'toolResult', content: [{type: 'text', text: '{"ok": true}'}]}),
  plugin.resultFailed({role: 'toolResult', content: [{type: 'text', text: 'not json'}]}),
]));
""")
    assert out == [True, False, False, True, True, False, False]


# A pick_and_place failure as the robot server returned it (live Isaac kitchen
# run, 2026-10-19), trimmed: the model reads the outcome, error, postcondition
# verdict and next_action, never the timings, raw measurements or file paths.
FAILED_PICK = {
    "destination": "green square", "target": [0.14, -0.27], "ok": False, "stage": "carry_clearance",
    "error": "_PreCarryLiftError: pre-carry lift is unreachable; keeping the grasp", "holding": "pink cube",
    "outcome": "failed", "postcondition": {"skill": "pick_and_place", "status": "refuted",
                                           "evidence": "pink cube is still within 4.0 cm of where it started",
                                           "measured": {"moved_m": 0.0402, "final": [0.1369, -0.2659, 0.0699]}},
    "verified": False, "next_action": "This attempt FAILED. Tell the user it failed.",
    "perception_timing": {"query": "pink cube", "semantic_s": 0.516}, "evidence_path": "/runs/x.json",
    "camera_statistics_note": "x" * 400, "connection_error": None, "events": [], "tools_hidden_by_operator": [],
}


def test_the_reply_turn_reads_the_tool_result_without_diagnostics_in_the_servers_json_style():
    script = f"""
const msg = {{role: 'toolResult', toolName: 'cascade__pick_and_place', isError: true,
  content: [{{type: 'text', text: {json.dumps(json.dumps(FAILED_PICK))}}}, {{type: 'text', text: 'not json'}}]}};
console.log(JSON.stringify([plugin.compactToolResult(msg), plugin.compactToolResult({{role: 'user', content: 'x'}}),
  plugin.pruneResult({{long: 'y'.repeat(400), deep: {{a: {{b: {{c: 1}}}}}}}})]));
"""
    compacted, user, pruned = _node(script)
    text, plain = compacted["content"]
    assert plain == {"type": "text", "text": "not json"}
    assert compacted["toolName"] == "cascade__pick_and_place" and compacted["isError"] is True
    kept = json.loads(text["text"])
    assert kept == {"destination": "green square", "target": [0.14, -0.27], "ok": False, "stage": "carry_clearance",
                    "error": FAILED_PICK["error"], "holding": "pink cube", "outcome": "failed",
                    "postcondition": {"skill": "pick_and_place", "status": "refuted",
                                      "evidence": FAILED_PICK["postcondition"]["evidence"]},
                    "verified": False, "next_action": FAILED_PICK["next_action"]}
    assert text["text"] == json.dumps(kept), "Python json.dumps separators, like the robot server's"
    assert user == {"role": "user", "content": "x"}
    assert pruned == {"long": "y" * 300 + "..."}, "long strings cut; nesting below depth 3 and empties dropped"


def test_a_result_with_an_image_reaches_the_model_as_its_outcome_and_the_image():
    observation = {"ok": True, "observation_frame": {"camera": "isaac", "frame_id": 1869},
                   "configured_zones": [{"name": "green square"}], "robot": {"tcp_xyz": [0.453, 0, 0.436]},
                   "observation_note": "Identify visible objects from the attached image."}
    script = f"""
const msg = {{role: 'toolResult', toolName: 'cascade__describe_scene', isError: false,
  content: [{{type: 'text', text: {json.dumps(json.dumps(observation))}}}, {{type: 'image', data: 'x', mimeType: 'image/jpeg'}}]}};
console.log(JSON.stringify(plugin.compactToolResult(msg).content));
"""
    assert _node(script) == [{"type": "text", "text": '{"ok": true, "image": "the camera image below"}'},
                             {"type": "image", "data": "x", "mimeType": "image/jpeg"}]


def test_the_reply_prompt_asks_about_the_image_when_the_result_has_one():
    script = """
const image = {role: 'toolResult', content: [{type: 'text', text: '{"ok": true}'}, {type: 'image', data: 'x'}]};
const text = {role: 'toolResult', content: [{type: 'text', text: '{"ok": false, "error": "e-stop latched"}'}]};
console.log(JSON.stringify([plugin.replyPrompt([{role: 'user', content: 'What do you see?'}, image]),
  plugin.replyPrompt([text]), plugin.IMAGE_REPLY_PROMPT, plugin.REPORT_PROMPT,
  plugin.compactToolResult(text).content[0].text,
  plugin.compactToolResult({role: 'toolResult', content: [{type: 'text', text: '{"error": "x", "ok": false}'}]}).content[0].text]));
"""
    on_image, on_text, image_prompt, reply_prompt, kept, reordered = _node(script)
    assert on_image == image_prompt and "camera image" in image_prompt
    assert on_text == reply_prompt and "Report only what these results show." in reply_prompt
    assert kept == '{"ok": false, "error": "e-stop latched"}'
    assert reordered == '{"ok": false, "error": "x"}', "ok first: read last, a failure was reported as a success"
