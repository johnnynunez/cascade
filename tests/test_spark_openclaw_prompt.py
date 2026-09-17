"""Exercise the native prompt hook without a running gateway or robot."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess

import pytest


REPO = Path(__file__).resolve().parents[1]
PLUGIN = REPO / "demo/kitchen/openclaw-plugin"



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
        "Operator rule: keep replies short.\n"
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
    assert result["result"]["systemPrompt"].startswith(instructions + "\n\n")
    assert "Reply in English" in result["result"]["systemPrompt"]
    assert result["eventUnchanged"] is True
    assert result["hooks"] == ["before_prompt_build"]


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
    assert result[0]["systemPrompt"].startswith("First operator instructions.\n\n\n")
    assert result[1]["systemPrompt"].startswith("Updated operator instructions.\n\n\n")


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



def test_payload_callback_fields_survive_auto_selection(tmp_path):
    result = run_hook(tmp_path, r"""
let callbackCalls = 0;
const replacement = Object.freeze({...payload, callbackField: 'kept', tool_choice: 'required',
  chat_template_kwargs: {custom: true, enable_thinking: true}});
const stream = provider.wrapStreamFn({...runtime, streamFn: invoke});
const received = await stream(model, request, {onPayload: async (value, receivedModel) => {
  if (value !== payload || receivedModel !== model) throw new Error('callback arguments changed');
  callbackCalls++;
  return replacement;
}});
console.log(JSON.stringify({received, callbackCalls, original: replacement.tool_choice}));
""")
    assert result["callbackCalls"] == 1
    assert result["original"] == "required"
    assert result["received"]["callbackField"] == "kept"
    assert result["received"]["messages"] == [{"role": "user", "content": "untouched wire prompt"}]
    assert result["received"]["tool_choice"] == "auto"
    assert result["received"]["chat_template_kwargs"] == {"custom": True, "enable_thinking": False}


def test_other_mcp_servers_and_advanced_schemas_stay_outside_attendee_context(tmp_path):
    result = run_hook(tmp_path, r"""
const original = {...payload, tools: [payload.tools[0],
  {type: 'function', function: {name: 'foreign__pick_and_place'}},
  {type: 'function', function: {name: 'cascade__place_at'}}]};
const stream = provider.wrapStreamFn({...runtime, streamFn: invoke});
const received = await stream(model, request, {onPayload: () => original});
console.log(JSON.stringify({names: received.tools.map(t => t.function.name), originalCount: original.tools.length}));
""")
    assert result == {"names": ["cascade__world_state"], "originalCount": 3}
