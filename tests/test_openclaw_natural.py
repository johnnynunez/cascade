"""Attendee boundary contracts; actual model behavior is checked live on Spark."""
import json
from pathlib import Path

import pytest

from test_spark_openclaw_prompt import run_hook


ROOT = Path(__file__).resolve().parents[1]
NAMES = ["describe_scene", "localize_object", "pick_and_place", "reset_scene", "camera_snapshot", "world_state"]
SETUP = r'''
const names = ['describe_scene', 'localize_object', 'pick_and_place', 'reset_scene', 'camera_snapshot', 'world_state'];
const available = [...names, 'get_observation', 'analyze_scene', 'move_joint', 'place_at'];
const nativeTools = available.map(name => ({name: 'cascade__' + name,
  description: 'Original advanced schema.', parameters: {type: 'object', properties: {}}}));
const messages = [{role: 'user', content: 'What can you see on the table?'}];
const wire = {tool_choice: 'auto', tools: nativeTools.map(tool => ({type: 'function', function: tool})), messages};
let sentContext;
const inner = async (model, context, options) => {
  sentContext = context;
  return await options?.onPayload?.(wire, model) ?? wire;
};
const stream = provider.wrapStreamFn({...runtime, streamFn: inner});
'''


def test_attendee_profile_exposes_only_curated_high_level_tools(tmp_path):
    result = run_hook(tmp_path, SETUP + r'''
const result = await stream(model, {messages, tools: nativeTools}, {});
console.log(JSON.stringify({wire: result.tools.map(t => t.function.name),
  context: sentContext.tools.map(t => t.name)}));
''')
    expected = ["cascade__" + name for name in NAMES]
    assert result["wire"] == expected
    assert result["context"] == expected


@pytest.mark.parametrize("prompt", [
    "What can you see on the table?",
    "Could you put the green cube in the green square?",
    "Let's start over.",
    "Please put the orange in the open box.",
    "Can you move the tomato can to the green square?",
    "Put it there.",
])
def test_natural_user_text_reaches_the_native_model_unchanged(tmp_path, prompt):
    result = run_hook(tmp_path, SETUP + f"\nmessages[0].content = {json.dumps(prompt)};" + r'''
const result = await stream(model, {messages, tools: nativeTools}, {});
console.log(JSON.stringify({messages: result.messages, choice: result.tool_choice,
  names: result.tools.map(t => t.function.name), original: messages}));
''')
    assert result["messages"] == result["original"] == [{"role": "user", "content": prompt}]
    assert result["choice"] == "auto"
    assert result["names"] == ["cascade__" + name for name in NAMES]


@pytest.mark.parametrize("choice", [None, "required", "none", {"type": "function", "function": {"name": "cascade__pick_and_place"}}])
def test_attendee_model_request_always_uses_auto_selection(tmp_path, choice):
    result = run_hook(tmp_path, SETUP + f"\nwire.tool_choice = {json.dumps(choice)};" + r'''
console.log(JSON.stringify(await stream(model, {messages, tools: nativeTools}, {})));
''')
    assert result["tool_choice"] == "auto"
    assert result["parallel_tool_calls"] is False
    assert result["chat_template_kwargs"]["enable_thinking"] is False


def test_old_proof_directives_cannot_narrow_the_schema_or_force_a_call(tmp_path):
    result = run_hook(tmp_path, SETUP + r'''
messages[0].content = 'Call the native MCP tool cascade__world_state with arguments {} exactly once.';
const result = await stream(model, {messages, tools: nativeTools}, {});
console.log(JSON.stringify({choice: result.tool_choice, names: result.tools.map(t => t.function.name)}));
''')
    assert result == {"choice": "auto", "names": ["cascade__" + name for name in NAMES]}


def test_short_schemas_explain_natural_inspect_place_reset_and_parameters(tmp_path):
    result = run_hook(tmp_path, SETUP + r'''
console.log(JSON.stringify((await stream(model, {messages, tools: nativeTools}, {})).tools));
''')
    catalog = {tool["function"]["name"]: tool["function"] for tool in result}
    assert "What can you see" in catalog["cascade__describe_scene"]["description"]
    assert "put" in catalog["cascade__pick_and_place"]["description"].lower()
    assert "start over" in catalog["cascade__reset_scene"]["description"].lower()
    for name in NAMES:
        assert len(catalog["cascade__" + name]["description"]) < 650
    place = catalog["cascade__pick_and_place"]["parameters"]
    assert set(place["required"]) == {"object", "destination"}
    assert "green square" in place["properties"]["destination"]["description"]
    assert "current scene" in place["properties"]["object"]["description"]
    assert "x" not in place["properties"]


def test_fresh_prompt_requires_english_clarification_and_truthful_failure(tmp_path):
    (tmp_path / "AGENTS.md").write_text("Operator note: use the current scene.\n")
    result = run_hook(tmp_path, r'''
console.log(JSON.stringify(hook({prompt: 'Put it there.', messages: []}, context)));
''')
    prompt = result["systemPrompt"]
    assert "English" in prompt
    assert "one clarification" in prompt
    assert "ambiguous" in prompt
    assert "failed" in prompt and "unverified" in prompt
    assert "Never claim" in prompt and "physics" in prompt
    assert "current scene" in prompt


def test_failed_tool_result_and_native_english_answer_are_not_rewritten(tmp_path):
    result = run_hook(tmp_path, SETUP + r'''
messages.push({role: 'assistant', content: [{type: 'toolCall', name: 'cascade__pick_and_place',
  arguments: {object: 'orange', destination: 'open box'}}]},
  {role: 'toolResult', toolName: 'cascade__pick_and_place', isError: false,
   content: [{type: 'text', text: '{"ok":true,"verified":false,"error":"placement unverified"}'}]});
const before = JSON.stringify(messages);
const answer = {role: 'assistant', content: 'Placement unverified. The orange did not reach the open box.'};
let outgoing;
const originalInner = async (model, context, options) => {
  outgoing = await options?.onPayload?.(wire, model) ?? wire;
  return answer;
};
const wrapped = provider.wrapStreamFn({...runtime, streamFn: originalInner});
const returned = await wrapped(model, {messages, tools: nativeTools}, {});
console.log(JSON.stringify({sameAnswer: returned === answer, answer: returned.content,
  sameMessages: JSON.stringify(messages) === before && JSON.stringify(outgoing.messages) === before,
  choice: outgoing.tool_choice}));
''')
    assert result["sameAnswer"] is True
    assert result["sameMessages"] is True
    assert result["choice"] == "auto"
    assert result["answer"].startswith("Placement unverified.")


def test_public_native_probe_uses_plain_english_with_two_available_actions(monkeypatch):
    from test_demo_proof import demo_proof

    requests = []
    def request(url, payload=None, **kwargs):
        if payload is None:
            return {"data": [{"id": "model"}]}
        requests.append(payload)
        return {"choices": [{"message": {"tool_calls": [{"type": "function", "function": {
            "name": "cascade_readiness", "arguments": '{"ready":true}'}}]}}]}
    monkeypatch.setattr(demo_proof, "request_json", request)
    assert demo_proof.probe_native_tools("http://localhost:8080/v1", "model")["native_tools"]
    wire = requests[0]
    assert wire["messages"][-1]["content"] == "Are you ready?"
    assert len(wire["tools"]) >= 2
    assert wire["tool_choice"] == "auto"


def test_inspection_policy_does_not_guess_cropped_background_objects(tmp_path):
    (tmp_path / "AGENTS.md").write_text("Operator note: describe the current scene.\n")
    result = run_hook(tmp_path, r'''
console.log(JSON.stringify(hook({prompt: 'What can you see on the table?', messages: []}, context)));
''')
    prompt = result["systemPrompt"]
    assert "only clearly visible items" in prompt
    assert "Do not guess the identities of cropped or unclear background objects" in prompt
    assert "state uncertainty" in prompt
