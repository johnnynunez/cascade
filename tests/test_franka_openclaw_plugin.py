"""Exercise the actual native prompt hook without a gateway or robot."""
import json
from pathlib import Path
import shutil
import subprocess

import pytest


PLUGIN = Path(__file__).resolve().parents[1] / "deploy/openshell/franka-plugin/index.mjs"


def test_franka_prompt_is_scoped_and_preserves_inspection_schemas(tmp_path):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node required for the OpenClaw prompt hook")
    (tmp_path / "AGENTS.md").write_text("Operator's measured FR3 instructions.\n")
    script = r"""
const plugin = (await import(process.argv[1])).default;
const workspaceDir = process.argv[2];
const config = {workspaceDir, modelProviderId:'local-cosmos', modelId:'cosmos3-edge'};
const hooks = [], providers = [];
plugin.register({pluginConfig:config, on:(...v)=>hooks.push(v), registerProvider:v=>providers.push(v)});
const scope = {agentId:'main', ...config};
const hook = hooks.find(([name])=>name==='before_prompt_build')[1];
const prompt = hook({}, scope);
const other = hook({}, {...scope, workspaceDir:workspaceDir+'-different'});
const tools = [{type:'function',function:{name:'cascade__place_at', parameters:{type:'object',properties:{x:{type:'number'}}}}},
               {type:'function',function:{name:'cascade__world_state',parameters:{type:'object',properties:{}}}}];
const messages=[{role:'user',content:'Look at the current workspace.'}];
const payload={messages, tools, tool_choice:'auto', untouched:42, chat_template_kwargs:{custom:true}};
let receivedContext;
const inner=async(model,context,options)=>{receivedContext=context;return options.onPayload({...payload,tools:context.tools.map(t=>({type:'function',function:t}))},model);};
const runtime = {agentId:'main',workspaceDir,provider:'local-cosmos',modelId:'cosmos3-edge',streamFn:inner};
const context={tools:tools.map(t=>t.function),messages};
const stream=providers[0].wrapStreamFn(runtime);
const output=await stream({api:'openai-completions',provider:'local-cosmos',id:'cosmos3-edge'},context,
                         {onPayload:async p=>({...p,previousCallback:'kept'})});
const outOfScope=providers[0].wrapStreamFn({...runtime,agentId:'other'})===inner;
console.log(JSON.stringify({prompt,other:other===undefined,output,inspectionOnly:receivedContext.tools.length===1,
                           sameSchema:receivedContext.tools[0]===context.tools[1],sameMessages:output.messages===messages,outOfScope}));
"""
    result = subprocess.run([node, "--input-type=module", "-e", script, PLUGIN.as_uri(), str(tmp_path)],
                            check=True, capture_output=True, text=True)
    data = json.loads(result.stdout)
    assert data["prompt"]["systemPrompt"].startswith("Operator's measured FR3 instructions.")
    assert "native tools" in data["prompt"]["systemPrompt"]
    assert data["other"] and data["outOfScope"]
    assert data["inspectionOnly"] and data["sameSchema"] and data["sameMessages"]
    assert data["output"]["previousCallback"] == "kept"
    assert data["output"]["untouched"] == 42
    assert data["output"]["tool_choice"] == "auto"
    assert data["output"]["parallel_tool_calls"] is False
    assert data["output"]["chat_template_kwargs"] == {"custom": True, "enable_thinking": False}


def test_franka_plugin_refuses_missing_instructions(tmp_path):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node required for the OpenClaw prompt hook")
    script = r"""
const plugin=(await import(process.argv[1])).default;
const cfg={workspaceDir:process.argv[2],modelProviderId:'local-cosmos',modelId:'cosmos3-edge'};
let hook;
plugin.register({pluginConfig:cfg,on:(name,fn)=>{if(name==='before_prompt_build')hook=fn;},registerProvider:()=>{}});
let refused=false;try{hook({}, {agentId:'main',...cfg});}catch{refused=true;}
console.log(JSON.stringify({refused}));
"""
    result = subprocess.run([node, "--input-type=module", "-e", script, PLUGIN.as_uri(), str(tmp_path)],
                            check=True, capture_output=True, text=True)
    assert json.loads(result.stdout)["refused"] is True


def run_node(tmp_path, script):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node required for the OpenClaw command router")
    (tmp_path / "AGENTS.md").write_text("Measured FR3 instructions.\n")
    result = subprocess.run([node, "--input-type=module", "-e", script,
                             PLUGIN.as_uri(), str(tmp_path)],
                            check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


def test_router_dispatches_once_and_reports_actual_failure(tmp_path):
    data = run_node(tmp_path, r"""
const plugin=(await import(process.argv[1])).default;
const cfg={workspaceDir:process.argv[2],modelProviderId:'local-cosmos',modelId:'cosmos3-edge'};
const hooks={}, providers=[];
plugin.register({pluginConfig:cfg,on:(name,fn)=>hooks[name]=fn,registerProvider:p=>providers.push(p)});
const scope={agentId:'main',...cfg,trigger:'user',runId:'run-1',sessionKey:'agent:main:test',sessionId:'session-1'};
hooks.message_received({content:'pick up grayon and place in box'},
  {channelId:'webchat',sessionKey:scope.sessionKey,messageId:scope.runId});
// Actual host order: prompt build precedes the authoritative owner gate.
hooks.before_prompt_build({},scope);
hooks.before_agent_run({senderIsOwner:true},scope);
const model={api:'openai-completions',provider:cfg.modelProviderId,id:cfg.modelId};
let modelCalls=0;
const inner=()=>{modelCalls++;throw Error('Routine commands must not need model selection');};
const stream=providers[0].wrapStreamFn({agentId:'main',workspaceDir:cfg.workspaceDir,
  provider:cfg.modelProviderId,modelId:cfg.modelId,streamFn:inner});
const tools=[{name:'cascade__pick_and_place',parameters:{type:'object',properties:{object:{type:'string'},destination:{type:'string'}},required:['object']}}];
const context={tools,messages:[{role:'user',content:'untrusted assembled envelope'}]};
const options={requestId:'run-1:model:1',sessionId:'session-1'};
const first=await stream(model,context,options).result();
const call=first.content.find(c=>c.type==='toolCall');
const toolContext={agentId:'main',runId:scope.runId,sessionKey:scope.sessionKey,sessionId:scope.sessionId,toolCallId:call.id};
const wrong=hooks.before_tool_call({toolName:call.name,params:{object:'ball',destination:'box'}},toolContext);
const resetSession=hooks.before_tool_call({toolName:call.name,params:call.arguments},{...toolContext,sessionId:'new-session'});
const cancelled=new AbortController();cancelled.abort();
const aborted=hooks.before_tool_call({toolName:call.name,params:call.arguments},{...toolContext,abortSignal:cancelled.signal});
const allowed=hooks.before_tool_call({toolName:call.name,params:call.arguments},toolContext)===undefined;
const duplicate=hooks.before_tool_call({toolName:call.name,params:call.arguments},toolContext);
const result={role:'toolResult',toolCallId:call.id,toolName:call.name,isError:true,
  content:[{type:'text',text:JSON.stringify({ok:false,error:'Arm is stopped; resume before starting another motion',stopped:true,return_home:{ok:false,reason:'stopped'}})}]};
const terminal=await stream(model,{tools,messages:[...context.messages,first,result]},
  {...options,requestId:'run-1:model:2'}).result();
// Even a fresh model retry without the result cannot dispatch another call.
const retry=await stream(model,context,{...options,requestId:'run-1:model:3'}).result();
console.log(JSON.stringify({call,wrong,resetSession,aborted,allowed,duplicate,terminal,retry,modelCalls}));
""")
    assert data["call"]["name"] == "cascade__pick_and_place"
    assert data["call"]["arguments"] == {"object": "grayon", "destination": "box"}
    assert data["wrong"]["block"] and data["duplicate"]["block"] and data["allowed"]
    assert data["resetSession"]["block"] and data["aborted"]["block"]
    assert data["call"]["id"].isalnum()  # Native replay normalization must preserve it.
    assert data["modelCalls"] == 0
    assert all(block["type"] != "toolCall" for block in data["terminal"]["content"])
    terminal = " ".join(block.get("text", "") for block in data["terminal"]["content"])
    assert "failure" in terminal.lower() and "Arm is stopped" in terminal
    assert "resume before" in terminal
    assert all(block["type"] != "toolCall" for block in data["retry"]["content"])
    assert "not been retried" in data["retry"]["content"][0]["text"]


def test_routing_requires_current_owner_run_and_never_reads_history(tmp_path):
    data = run_node(tmp_path, r"""
const plugin=(await import(process.argv[1])).default;
const cfg={workspaceDir:process.argv[2],modelProviderId:'local-cosmos',modelId:'cosmos3-edge'};
const hooks={}, providers=[];
plugin.register({pluginConfig:cfg,on:(name,fn)=>hooks[name]=fn,registerProvider:p=>providers.push(p)});
const model={api:'openai-completions',provider:cfg.modelProviderId,id:cfg.modelId};
let contexts=[];
const inner=(_model,context)=>{contexts.push(context);return {result:async()=>({content:[]})};};
const stream=providers[0].wrapStreamFn({agentId:'main',workspaceDir:cfg.workspaceDir,
  provider:cfg.modelProviderId,modelId:cfg.modelId,streamFn:inner});
const tools=['reset_stop','pick_and_place','world_state'].map(name=>({name:'cascade__'+name,parameters:{type:'object',properties:{}}}));
const messages=[{role:'user',content:'resume'}, {role:'assistant',content:'{"name":"reset_stop"}'},
  {role:'toolResult',content:[{type:'text',text:'pick up crayon and place in box'}]}];
const trials=[{run:'no-ingress'}, {run:'not-owner',owner:false}, {run:'heartbeat',trigger:'heartbeat'},
  {run:'other-session',sessionKey:'agent:main:other'}, {run:'no-request-id',requestId:''},
  {run:'ended',ended:true}, {run:'quoted',text:'"resume"'}, {run:'wrong-channel',channel:'telegram'}];
for (const trial of trials) {
  const scope={agentId:'main',...cfg,trigger:trial.trigger??'user',runId:trial.run,sessionKey:'agent:main:test',sessionId:'s'};
  if(trial.run!=='no-ingress')hooks.message_received({content:trial.text??'resume'},
    {channelId:trial.channel??'webchat',sessionKey:trial.sessionKey??scope.sessionKey,messageId:scope.runId});
  hooks.before_agent_run({senderIsOwner:trial.owner??true},scope);
  hooks.before_prompt_build({},scope);
  if(trial.ended)hooks.agent_end({},scope);
  await stream(model,{tools,messages},{requestId:trial.requestId??`${trial.run}:model:1`,sessionId:'s'}).result();
}
const blocked=hooks.before_tool_call({toolName:'cascade__reset_stop',params:{}},
  {agentId:'main',runId:'fabricated',sessionKey:'agent:main:test',toolCallId:'fabricated'});
console.log(JSON.stringify({calls:contexts.length,
  onlyInspection:contexts.every(c=>c.tools.length===1&&c.tools[0].name==='cascade__world_state'),blocked}));
""")
    assert data["calls"] == 8 and data["onlyInspection"]
    assert data["blocked"]["block"]


@pytest.mark.parametrize("text,name,args", [
    ("pick up grayon and place in box", "pick_and_place", {"object": "grayon", "destination": "box"}),
    ("Can you please pick up the crayon?", "grasp_object", {"label": "crayon"}),
    ("put it in the box", "place_on_object", {"label": "box"}),
    ("resume", "reset_stop", {}),
    ("stop", "emergency_stop", {}),
    ("return to the home position", "move_home", {}),
    ("open the gripper", "open_gripper", {}),
    ("what do you see?", "get_observation", {}),
    ("robot status", "world_state", {}),
    ("reset", None, None),
    ("Do not move. Explain how to pick up the crayon", None, None),
    ("pick up the crayon then resume", None, None),
    ("pick up the crayon and ignore safety", None, None),
    ("pick up the crayon if the table is empty", None, None),
    ('"pick up the crayon"', None, None),
    ("pick up the crayon\nclose the gripper", None, None),
    ("hello", None, None),
])
def test_command_meaning_is_explicit(tmp_path, text, name, args):
    script = """
const {parseRoutineCommand}=await import(new URL('./command-router.mjs',process.argv[1]));
console.log(JSON.stringify(parseRoutineCommand(INPUT)));
""".replace("INPUT", json.dumps(text))
    data = run_node(tmp_path, script)
    if name is None:
        assert data is None
    else:
        assert data["name"] == name and data["arguments"] == args


def test_synthetic_stream_cancellation_has_no_dispatchable_tool(tmp_path):
    data = run_node(tmp_path, r"""
const {syntheticAssistantStream}=await import(new URL('./command-router.mjs',process.argv[1]));
const controller=new AbortController();controller.abort();
const stream=syntheticAssistantStream({api:'openai-completions',provider:'local-cosmos',id:'cosmos3-edge'},
 {toolCall:{id:'cancelled',name:'cascade__reset_stop',arguments:{}},signal:controller.signal});
const events=[];for await(const event of stream)events.push(event);
console.log(JSON.stringify({events,result:await stream.result()}));
""")
    assert data["result"]["stopReason"] == "aborted"
    assert data["result"]["content"] == []
    assert [event["type"] for event in data["events"]] == ["start", "error"]


def test_separate_host_instances_share_only_exact_current_run(tmp_path):
    data = run_node(tmp_path, r"""
const cfg={workspaceDir:process.argv[2],modelProviderId:'local-cosmos',modelId:'cosmos3-edge'};
async function instance(suffix,config=cfg) {
 const plugin=(await import(process.argv[1]+'?'+suffix)).default;
 const hooks={},providers=[];
 plugin.register({pluginConfig:config,on:(name,fn)=>hooks[name]=fn,registerProvider:p=>providers.push(p)});
 return {hooks,provider:providers[0]};
}
const ingress=await instance('ingress'), agent=await instance('agent');
const foreign=await instance('foreign',{...cfg,workspaceDir:cfg.workspaceDir+'/foreign'});
const scopes=['a','b'].map(id=>({agentId:'main',...cfg,trigger:'user',runId:id,sessionKey:'agent:main:'+id,sessionId:id}));
for(const [i,scope] of scopes.entries()) {
 ingress.hooks.message_received({content:i===0?'pick up crayon':'pick up ball'},
   {channelId:'webchat',messageId:scope.runId,sessionKey:scope.sessionKey});
 agent.hooks.before_prompt_build({},scope);
 agent.hooks.before_agent_run({senderIsOwner:true},scope);
}
let fallback=0;
const inner=()=>{fallback++;return {result:async()=>({content:[]})};};
const runtime={agentId:'main',workspaceDir:cfg.workspaceDir,provider:cfg.modelProviderId,modelId:cfg.modelId,streamFn:inner};
const stream=agent.provider.wrapStreamFn(runtime);
const context={messages:[],tools:[{name:'cascade__grasp_object',parameters:{type:'object',properties:{label:{type:'string'}},required:['label']}}]};
const model={api:'openai-completions',provider:cfg.modelProviderId,id:cfg.modelId};
const b=await stream(model,context,{requestId:'b:model:1',sessionId:'b'}).result();
const a=await stream(model,context,{requestId:'a:model:1',sessionId:'a'}).result();
await foreign.provider.wrapStreamFn({...runtime,workspaceDir:cfg.workspaceDir+'/foreign'})(model,context,
 {requestId:'a:model:1',sessionId:'a'}).result();
console.log(JSON.stringify({a:a.content.find(b=>b.type==='toolCall'),b:b.content.find(b=>b.type==='toolCall'),fallback}));
""")
    assert data["a"]["arguments"] == {"label": "crayon"}
    assert data["b"]["arguments"] == {"label": "ball"}
    assert data["a"]["id"] != data["b"]["id"] and data["fallback"] == 1


def test_stop_resume_cannot_revive_delayed_commands_in_another_chat(tmp_path):
    data = run_node(tmp_path, r"""
const plugin=(await import(process.argv[1])).default;
const cfg={workspaceDir:process.argv[2],modelProviderId:'local-cosmos',modelId:'cosmos3-edge'};
const hooks={},providers=[];
plugin.register({pluginConfig:cfg,on:(name,fn)=>hooks[name]=fn,registerProvider:p=>providers.push(p)});
function inbound(id,text) {
 const scope={agentId:'main',...cfg,trigger:'user',runId:id,sessionKey:'agent:main:'+id,sessionId:id};
 hooks.message_received({content:text},{channelId:'webchat',messageId:id,sessionKey:scope.sessionKey});
 hooks.before_prompt_build({},scope);hooks.before_agent_run({senderIsOwner:true},scope);
 return scope;
}
const stream=providers[0].wrapStreamFn({agentId:'main',workspaceDir:cfg.workspaceDir,
 provider:cfg.modelProviderId,modelId:cfg.modelId,streamFn:()=>{throw Error('unexpected model call');}});
const model={api:'openai-completions',provider:cfg.modelProviderId,id:cfg.modelId};
const context={messages:[],tools:['grasp_object','reset_stop','emergency_stop'].map(n=>({name:'cascade__'+n,
 parameters:{type:'object',properties:{label:{type:'string'}}}}))};
const old=inbound('old','pick up crayon');
inbound('queued','pick up ball');
const dispatched=await stream(model,context,{requestId:'old:model:1',sessionId:'old'}).result();
const call=dispatched.content.find(b=>b.type==='toolCall');
inbound('stop','stop');inbound('resume','resume');
const blocked=hooks.before_tool_call({toolName:call.name,params:call.arguments},
 {agentId:'main',runId:old.runId,sessionKey:old.sessionKey,sessionId:old.sessionId,toolCallId:call.id});
const queued=await stream(model,context,{requestId:'queued:model:1',sessionId:'queued'}).result();
const resume=await stream(model,context,{requestId:'resume:model:1',sessionId:'resume'}).result();
console.log(JSON.stringify({blocked,queued,resume}));
""")
    assert data["blocked"]["block"]
    assert not any(b["type"] == "toolCall" for b in data["queued"]["content"])
    assert "superseded" in data["queued"]["content"][0]["text"]
    assert next(b for b in data["resume"]["content"] if b["type"] == "toolCall")["name"] == "cascade__reset_stop"


def test_prior_dispatched_call_is_consumed_after_gateway_restart(tmp_path):
    data = run_node(tmp_path, r"""
const plugin=(await import(process.argv[1])).default;
const cfg={workspaceDir:process.argv[2],modelProviderId:'local-cosmos',modelId:'cosmos3-edge'};
const scope={agentId:'main',...cfg,trigger:'user',runId:'same-request',sessionKey:'agent:main:test',sessionId:'session'};
const model={api:'openai-completions',provider:cfg.modelProviderId,id:cfg.modelId};
const tools=[{name:'cascade__grasp_object',parameters:{type:'object',properties:{label:{type:'string'}},required:['label']}}];
const options={requestId:'same-request:model:1',sessionId:'session'};
function boot() {
 const hooks={},providers=[];
 plugin.register({pluginConfig:cfg,on:(name,fn)=>hooks[name]=fn,registerProvider:p=>providers.push(p)});
 hooks.message_received({content:'pick up crayon'},{channelId:'webchat',sessionKey:scope.sessionKey,messageId:scope.runId});
 hooks.before_prompt_build({},scope);hooks.before_agent_run({senderIsOwner:true},scope);
 return {hooks,stream:providers[0].wrapStreamFn({agentId:'main',workspaceDir:cfg.workspaceDir,
  provider:cfg.modelProviderId,modelId:cfg.modelId,streamFn:()=>{throw Error('unexpected model');}})};
}
const first=boot();
const message=await first.stream(model,{tools,messages:[]},options).result();
const call=message.content.find(b=>b.type==='toolCall');
globalThis[Symbol.for('cascade.franka.command-routing.v1')].clear(); // new gateway, retained transcript
const restarted=boot();
const replay=await restarted.stream(model,{tools,messages:[message]},options).result();
const blocked=restarted.hooks.before_tool_call({toolName:call.name,params:call.arguments},
 {agentId:'main',runId:scope.runId,sessionKey:scope.sessionKey,sessionId:scope.sessionId,toolCallId:call.id});
console.log(JSON.stringify({replay,blocked}));
""")
    assert not any(b["type"] == "toolCall" for b in data["replay"]["content"])
    assert "already dispatched" in data["replay"]["content"][0]["text"]
    assert data["blocked"]["block"]



def test_resume_without_reset_stop_points_to_the_operator_page(tmp_path):
    """The OpenShell route never offers reset_stop to the agent (L7 refuses it
    and the FR3 catalog omits it): an explicit "resume" is answered with the
    operator's STOP / Resume & Home page, and nothing is dispatched."""
    data = run_node(tmp_path, r"""
const plugin=(await import(process.argv[1])).default;
const cfg={workspaceDir:process.argv[2],modelProviderId:'cascade',modelId:'nemotron-3.5-bolt'};
const hooks={},providers=[];
plugin.register({pluginConfig:cfg,on:(name,fn)=>hooks[name]=fn,registerProvider:p=>providers.push(p)});
const scope={agentId:'main',...cfg,trigger:'user',runId:'r1',sessionKey:'agent:main:s',sessionId:'s'};
hooks.message_received({content:'resume'},{channelId:'webchat',sessionKey:scope.sessionKey,messageId:scope.runId});
hooks.before_prompt_build({},scope);hooks.before_agent_run({senderIsOwner:true},scope);
const stream=providers[0].wrapStreamFn({agentId:'main',workspaceDir:cfg.workspaceDir,provider:cfg.modelProviderId,
  modelId:cfg.modelId,streamFn:()=>{throw Error('unexpected model call');}});
const model={api:'openai-completions',provider:cfg.modelProviderId,id:cfg.modelId};
const tools=['world_state','emergency_stop','pick_and_place'].map(n=>({name:'cascade__'+n,parameters:{type:'object',properties:{}}}));
const reply=await stream(model,{tools,messages:[]},{requestId:'r1:model:1',sessionId:'s'}).result();
console.log(JSON.stringify(reply));
""")
    assert not any(b["type"] == "toolCall" for b in data["content"])
    text = data["content"][0]["text"]
    assert "Resume & Home" in text and "STOP page" in text and "Nothing was dispatched" in text


def test_the_sandbox_config_and_the_plugin_agree(tmp_path):
    """The franka profile rendered by deploy/openshell/render-config.mjs
    (exactly what the sandbox's OpenClaw loads) drives this plugin: its
    pluginConfig registers, its AGENTS.md is the system prompt, and a routine
    command compiles to a tool the rendered catalog allows."""
    node = shutil.which("node")
    if not node:
        pytest.skip("Node required for the OpenClaw plugin")
    deploy = Path(__file__).resolve().parents[1] / "deploy/openshell"
    home = tmp_path / "home"
    home.mkdir()
    r = subprocess.run([node, str(deploy / "render-config.mjs")], capture_output=True, text=True, timeout=30,
                       env={"HOME": str(home), "PATH": "", "CASCADE_AGENT_PROFILE": "franka",
                            "CASCADE_OPENCLAW_TEMPLATE": str(deploy / "openclaw.json")})
    assert r.returncode == 0, r.stderr
    cfg = json.loads((home / ".openclaw/openclaw.json").read_text())
    entry = cfg["plugins"]["entries"]["cascade-franka-prompt"]
    plugin = Path(cfg["plugins"]["load"]["paths"][0]) / "index.mjs"
    # The sandbox's workspace is /sandbox/.openclaw/workspace (HOME=/sandbox).
    config = {**entry["config"], "workspaceDir": str(home / ".openclaw/workspace")}
    script = r"""
const plugin=(await import(process.argv[1])).default;
const cfg=JSON.parse(process.argv[2]), allow=JSON.parse(process.argv[3]);
const hooks={},providers=[];
plugin.register({pluginConfig:cfg,on:(name,fn)=>hooks[name]=fn,registerProvider:p=>providers.push(p)});
const scope={agentId:'main',...cfg,trigger:'user',runId:'r1',sessionKey:'agent:main:s',sessionId:'s'};
hooks.message_received({content:'pick up the crayon and place it in the box'},
  {channelId:'webchat',sessionKey:scope.sessionKey,messageId:scope.runId});
const prompt=hooks.before_prompt_build({},scope);
hooks.before_agent_run({senderIsOwner:true},scope);
const stream=providers[0].wrapStreamFn({agentId:'main',workspaceDir:cfg.workspaceDir,provider:cfg.modelProviderId,
  modelId:cfg.modelId,streamFn:()=>{throw Error('unexpected model call');}});
const model={api:'openai-completions',provider:cfg.modelProviderId,id:cfg.modelId};
const tools=allow.map(n=>({name:n,parameters:{type:'object',properties:{object:{type:'string'},
  destination:{type:'string'},label:{type:'string'}}}}));
const reply=await stream(model,{tools,messages:[]},{requestId:'r1:model:1',sessionId:'s'}).result();
console.log(JSON.stringify({prompt,reply,provider:providers[0].id}));
"""
    result = subprocess.run([node, "--input-type=module", "-e", script, plugin.as_uri(), json.dumps(config),
                             json.dumps(cfg["tools"]["allow"])], check=True, capture_output=True, text=True)
    data = json.loads(result.stdout)
    assert data["provider"] == cfg["agents"]["defaults"]["model"]["primary"].split("/")[0]
    assert data["prompt"]["systemPrompt"].startswith("You operate a real Franka Research 3")
    call = next(b for b in data["reply"]["content"] if b["type"] == "toolCall")
    assert call["name"] == "cascade__pick_and_place" and call["name"] in cfg["tools"]["allow"]
    assert call["arguments"] == {"object": "crayon", "destination": "box"}
