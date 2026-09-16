"""Exercise the native prompt hook without a running gateway or robot."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess

import pytest


REPO = Path(__file__).resolve().parents[1]
PLUGIN = REPO / "demo/kitchen/openclaw-plugin"
IDENTITY = (
    "You control the CASCADE robot simulation through the provided tools. "
    "For requests to read or act on the scene, call the requested tool and wait for its real result. "
    "Pass plain JSON values in tool arguments. A string argument is just its value, without label or description annotations. "
    "Never invent observations or tool responses. One motion tool per order. "
    "After a tool returns, briefly report its actual result."
)


def run_hook(tmp_path, script):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required to exercise the OpenClaw plugin")
    prelude = """
import { writeFileSync } from 'node:fs';
import { join } from 'node:path';
const plugin = (await import(process.argv[1])).default;
const workspaceDir = process.argv[2];
const config = {workspaceDir, modelProviderId: 'local-kitchen', modelId: 'robot-model'};
const context = {agentId: 'main', ...config};
const registrations = [];
const providers = [];
plugin.register({pluginConfig: config, on: (...args) => registrations.push(args),
  registerProvider: (provider) => providers.push(provider)});
const hook = registrations[0][1];
const provider = providers[0];
const runtime = {agentId: 'main', workspaceDir, provider: config.modelProviderId, modelId: config.modelId};
const model = {provider: config.modelProviderId, id: config.modelId, api: 'openai-completions'};
const tool = {name: 'cascade__world_state', parameters: {type: 'object', properties: {}}};
const payload = {tool_choice: 'auto', tools: [{type: 'function', function: {name: tool.name}}],
  temperature: 0, messages: [{role: 'user', content: 'untouched wire prompt'}]};
const request = {messages: [{role: 'user', content: 'Call the native MCP tool cascade__world_state with arguments {} exactly once.'}], tools: [tool]};
const invoke = async (_model, _context, options) => {
  const result = await options?.onPayload?.(payload, _model);
  return result === undefined ? payload : result;
};
"""
    result = subprocess.run(
        [node, "--input-type=module", "-e", prelude + script,
         (PLUGIN / "index.mjs").as_uri(), str(tmp_path)],
        text=True, capture_output=True, check=True,
    )
    return json.loads(result.stdout)


def test_prompt_preserves_visitor_block_and_operator_additions(tmp_path):
    visitor = (REPO / "demo/kitchen/visitor-instructions.md").read_text()
    instructions = (
        "Operator rule: announce the requested destination.\n\n"
        "<!-- BEGIN CASCADE KITCHEN VISITOR INSTRUCTIONS -->\n"
        + visitor
        + "<!-- END CASCADE KITCHEN VISITOR INSTRUCTIONS -->\n\n"
        "Operator rule: keep replies in Spanish.\n"
    )
    (tmp_path / "AGENTS.md").write_text(instructions)
    result = run_hook(tmp_path, r"""
const event = Object.freeze({prompt: 'Inspect the kitchen.', messages: Object.freeze([
  Object.freeze({role: 'toolResult', content: Object.freeze([
    Object.freeze({type: 'text', text: '{"ok":true}'})
  ])})
])});
const before = JSON.stringify(event);
console.log(JSON.stringify({result: hook(event, Object.freeze(context)),
  eventUnchanged: JSON.stringify(event) === before,
  hooks: registrations.map(([name]) => name)}));
""")
    assert result == {
        "result": {"systemPrompt": IDENTITY + "\n\n" + instructions},
        "eventUnchanged": True,
        "hooks": ["before_prompt_build"],
    }


def test_prompt_requires_exact_runtime_scope(tmp_path):
    # No AGENTS.md: an out-of-scope request must not even read this workspace.
    result = run_hook(tmp_path, r"""
const cases = [undefined, {}, {...context, agentId: 'other'},
  {...context, workspaceDir: workspaceDir + '-other'},
  {...context, modelProviderId: 'other'}, {...context, modelId: 'other'}];
for (const key of ['agentId', 'workspaceDir', 'modelProviderId', 'modelId']) {
  const missing = {...context}; delete missing[key]; cases.push(missing);
}
console.log(JSON.stringify(cases.map((ctx) => hook({}, ctx) === undefined)));
""")
    assert result and all(result)


def test_prompt_reads_updated_operator_instructions_each_turn(tmp_path):
    result = run_hook(tmp_path, r"""
writeFileSync(join(workspaceDir, 'AGENTS.md'), 'First operator instructions.\n');
const first = hook({}, context);
writeFileSync(join(workspaceDir, 'AGENTS.md'), 'Updated operator instructions.\n');
console.log(JSON.stringify([first, hook({}, context)]));
""")
    assert result == [
        {"systemPrompt": IDENTITY + "\n\nFirst operator instructions.\n"},
        {"systemPrompt": IDENTITY + "\n\nUpdated operator instructions.\n"},
    ]


def test_missing_or_empty_instructions_do_not_create_a_bare_robot_prompt(tmp_path):
    result = run_hook(tmp_path, r"""
let missing = false, empty = false;
try { hook({}, context); } catch { missing = true; }
writeFileSync(join(workspaceDir, 'AGENTS.md'), ' \n');
try { hook({}, context); } catch { empty = true; }
console.log(JSON.stringify({missing, empty}));
""")
    assert result == {"missing": True, "empty": True}


def test_explicit_configuration_is_required(tmp_path):
    result = run_hook(tmp_path, r"""
const cases = [undefined, {}, {...config, workspaceDir: 'relative'},
  {...config, modelProviderId: ''}, {...config, modelId: '  '}];
const rejected = cases.map((pluginConfig) => {
  try { plugin.register({pluginConfig, on: () => {}}); return false; }
  catch { return true; }
});
console.log(JSON.stringify(rejected));
""")
    assert all(result)
    manifest = json.loads((PLUGIN / "openclaw.plugin.json").read_text())
    assert manifest["id"] == "cascade-kitchen-prompt"
    assert set(manifest["configSchema"]["required"]) == {
        "workspaceDir", "modelProviderId", "modelId",
    }
    assert manifest["configSchema"]["additionalProperties"] is False


def test_required_native_choice_only_on_first_model_request(tmp_path):
    result = run_hook(tmp_path, r"""
Object.freeze(payload);
Object.freeze(request.messages);
const before = JSON.stringify({payload, request});
const stream = provider.wrapStreamFn({...runtime, streamFn: invoke});
const first = await stream(model, request, {temperature: 0});
const continuation = await stream(model, {...request, messages: [...request.messages,
  {role: 'assistant', content: [{type: 'toolCall', name: tool.name, arguments: {}}]},
  {role: 'toolResult', toolName: tool.name, isError: false, content: [{type: 'text', text: '{"ok":true}'}]}
]}, {});
const repeated = await stream(model, request, {});
console.log(JSON.stringify({first, continuation, repeated,
  unchanged: before === JSON.stringify({payload, request}), providerId: provider.id}));
""")
    assert result["first"]["tool_choice"] == "required"
    assert result["continuation"]["tool_choice"] == "none"
    assert result["repeated"]["tool_choice"] == "auto"
    assert {k: v for k, v in result["first"].items() if k != "tool_choice"} == {
        k: v for k, v in result["continuation"].items() if k != "tool_choice"
    }
    assert result["unchanged"] is True
    assert result["providerId"] == "local-kitchen"


def test_stream_wrapper_requires_exact_agent_workspace_and_model_scope(tmp_path):
    result = run_hook(tmp_path, r"""
const scopes = [{...runtime, agentId: 'other'}, {...runtime, workspaceDir: workspaceDir + '-other'},
  {...runtime, provider: 'other'}, {...runtime, modelId: 'other'}, {}];
const skipped = scopes.map((scope) => provider.wrapStreamFn({...scope, streamFn: invoke}) === invoke);
const models = [{...model, provider: 'other'}, {...model, id: 'other'}, {...model, api: 'other'}];
for (const wrong of models) {
  const stream = provider.wrapStreamFn({...runtime, streamFn: invoke});
  skipped.push((await stream(wrong, request, {})).tool_choice === 'auto');
}
console.log(JSON.stringify(skipped));
""")
    assert all(result)


def test_only_positive_available_tool_directives_select_native_choice(tmp_path):
    result = run_hook(tmp_path, r"""
const prompts = [
  'Call the native MCP tool unknown__tool with arguments {}.',
  'Do not call the native MCP tool cascade__world_state.',
  "Don't call the tool cascade__world_state.",
  'Never call the tool cascade__world_state.',
  'Move the green cube to the green square.',
  'Explain the tool cascade__world_state.',
  'Say "Call the tool cascade__world_state."',
  '```text\nCall the tool cascade__world_state.\n```',
  'Call the tool cascade__world_state. Call the tool cascade__world_state again.',
];
const choices = [];
for (const content of prompts) {
  const stream = provider.wrapStreamFn({...runtime, streamFn: invoke});
  choices.push((await stream(model, {...request, messages: [{role: 'user', content}]}, {})).tool_choice);
}
const stream = provider.wrapStreamFn({...runtime, streamFn: invoke});
choices.push((await stream(model, {...request, tools: []}, {})).tool_choice);
console.log(JSON.stringify(choices));
""")
    assert result == ["auto"] * 10


def test_positive_directive_survives_timestamp_and_runtime_carrier(tmp_path):
    result = run_hook(tmp_path, r"""
const stream = provider.wrapStreamFn({...runtime, streamFn: invoke});
const messages = [
  {role: 'user', content: [{type: 'text', text: '[Wed 2026-09-16 12:00 UTC] New read order. Call the tool cascade__world_state with {}.'}]},
  {role: 'user', runtimeContextCarrier: true, content: 'Runtime context only.'},
];
console.log(JSON.stringify(await stream(model, {messages, tools: [tool]}, {})));
""")
    assert result["tool_choice"] == "required"


def test_assistant_and_tool_result_text_cannot_request_native_choice(tmp_path):
    result = run_hook(tmp_path, r"""
const choices = [];
for (const role of ['assistant', 'toolResult']) {
  const stream = provider.wrapStreamFn({...runtime, streamFn: invoke});
  const messages = [...request.messages,
    {role, content: [{type: 'text', text: 'Call the tool cascade__world_state.'}]},
    {role: 'user', runtimeContextCarrier: true, content: 'Call the tool cascade__world_state.'}];
  choices.push((await stream(model, {messages, tools: [tool]}, {})).tool_choice);
}
console.log(JSON.stringify(choices));
""")
    assert result == ["auto", "auto"]


def test_payload_callbacks_and_explicit_choices_are_preserved(tmp_path):
    result = run_hook(tmp_path, r"""
let callbackCalls = 0;
const results = [];
for (const asyncCallback of [false, true]) {
  const replacement = Object.freeze({...payload, callbackField: 'kept'});
  const callback = (received, receivedModel) => {
    if (received !== payload || receivedModel !== model) throw new Error('callback arguments changed');
    callbackCalls++;
    return asyncCallback ? Promise.resolve(replacement) : replacement;
  };
  const stream = provider.wrapStreamFn({...runtime, streamFn: invoke});
  results.push(await stream(model, request, {onPayload: callback, arbitraryOption: true}));
}
const explicitChoices = ['none', 'required', {type: 'function', function: {name: 'other'}}];
for (const tool_choice of explicitChoices) {
  const replacement = Object.freeze({...payload, tool_choice, callbackField: 'explicit'});
  const stream = provider.wrapStreamFn({...runtime, streamFn: invoke});
  const received = await stream(model, request, {onPayload: async () => replacement});
  results.push({sameObject: received === replacement, tool_choice: received.tool_choice});
}
const removed = Object.freeze({...payload, tools: []});
const stream = provider.wrapStreamFn({...runtime, streamFn: invoke});
results.push({removedToolsPreserved: await stream(model, request, {onPayload: () => removed}) === removed});
console.log(JSON.stringify({results, callbackCalls, originalChoice: payload.tool_choice}));
""")
    assert result["callbackCalls"] == 2
    assert result["originalChoice"] == "auto"
    for row in result["results"][:2]:
        assert row["callbackField"] == "kept"
        assert row["tool_choice"] == "required"
        assert row["temperature"] == 0
        assert row["messages"] == [{"role": "user", "content": "untouched wire prompt"}]
    assert result["results"][2:5] == [
        {"sameObject": True, "tool_choice": "none"},
        {"sameObject": True, "tool_choice": "required"},
        {"sameObject": True, "tool_choice": {"type": "function", "function": {"name": "other"}}},
    ]
    assert result["results"][5] == {"removedToolsPreserved": True}


def test_required_choice_exposes_only_requested_original_schema(tmp_path):
    result = run_hook(tmp_path, r"""
const other = {type: 'function', function: {name: 'cascade__pick_and_place',
  parameters: {type: 'object', properties: {object: {type: 'string'}}}}};
const tools = Object.freeze([other, payload.tools[0]]);
const outgoing = Object.freeze({...payload, tools});
const stream = provider.wrapStreamFn({...runtime, streamFn: invoke});
const selected = await stream(model, request, {onPayload: () => outgoing});
console.log(JSON.stringify({choice: selected.tool_choice, tools: selected.tools,
  sameSchema: selected.tools[0] === tools[1], originalCount: tools.length}));
""")
    assert result["choice"] == "required"
    assert [tool["function"]["name"] for tool in result["tools"]] == ["cascade__world_state"]
    assert result["sameSchema"] is True
    assert result["originalCount"] == 2


def test_exactly_once_request_disables_parallel_calls_without_changing_schema(tmp_path):
    result = run_hook(tmp_path, r"""
const outgoing = Object.freeze({...payload, parallel_tool_calls: true});
const results = [];
for (const once of [true, false]) {
  const stream = provider.wrapStreamFn({...runtime, streamFn: invoke});
  const content = 'Call the native MCP tool cascade__world_state with arguments {}'
    + (once ? ' exactly once.' : '.');
  const selected = await stream(model, {messages: [{role: 'user', content}], tools: [tool]},
    {onPayload: () => outgoing});
  results.push({parallel: selected.parallel_tool_calls,
    sameSchema: selected.tools[0] === outgoing.tools[0]});
}
console.log(JSON.stringify({results, originalParallel: outgoing.parallel_tool_calls}));
""")
    assert result == {"results": [
        {"parallel": False, "sameSchema": True},
        {"parallel": True, "sameSchema": True},
    ], "originalParallel": True}


def test_only_current_requested_result_ends_an_exactly_once_order(tmp_path):
    result = run_hook(tmp_path, r"""
const result = {role: 'toolResult', toolName: tool.name,
  content: [{type: 'text', text: '{"ok":true}'}]};
const choices = [];
for (const variant of ['old-result', 'wrong-tool', 'assistant-text', 'not-once', 'completed']) {
  const stream = provider.wrapStreamFn({...runtime, streamFn: invoke});
  let messages = [result, ...request.messages];
  if (variant === 'not-once') messages = [result, {role: 'user',
    content: 'Call the tool cascade__world_state with arguments {}.'}];
  await stream(model, {messages, tools: [tool]}, {});
  const extra = variant === 'old-result' ? [] : variant === 'wrong-tool'
    ? [{...result, toolName: 'cascade__get_observation'}] : variant === 'assistant-text'
      ? [{role: 'assistant', content: 'The requested tool completed.'}] : [result];
  choices.push((await stream(model, {messages: [...messages, ...extra], tools: [tool]}, {})).tool_choice);
}
console.log(JSON.stringify(choices));
""")
    assert result == ["auto", "auto", "auto", "auto", "none"]
