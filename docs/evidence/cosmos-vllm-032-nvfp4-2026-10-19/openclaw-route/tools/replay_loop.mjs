// Multi-step replay of the Cosmos3-Edge agent profile through the REAL plugin
// (wrapStreamFn), against vLLM. The first call of each request is pinned to the
// correct one (first-turn accuracy is measured separately); every later call is
// the model's. Tool results are real robot-server results (live kitchen trace,
// mock-rig dumps, captured world_state), relabelled for the call's arguments.
//
// usage: node replay_loop.mjs <port> <plugin index.mjs> <out.jsonl> <reps> <variant,...>
// variants: one (one tool per request: the shipped WIP policy), loop (plugin as
// is), loop_nonext (continuation reads ok results without next_action),
// loop_done (CONTINUE_PROMPT + a "Done so far" list), naive (continuation with the
// first-turn SYSTEM_PROMPT: the earlier runaway setup)
import fs from 'node:fs';
import path from 'node:path';
import {pathToFileURL} from 'node:url';

const [port, pluginPath, outPath, repsArg, variantsArg] = process.argv.slice(2);
const plugin = await import(pathToFileURL(path.resolve(pluginPath)).href);
const D = '/home/johnny/.hermes/cache/scratch/cosmos-demo';
const WT = '/home/johnny/Projects/cascade-wt/cosmos032';
const load = f => JSON.parse(fs.readFileSync(f, 'utf8'));
const TOOLS_REQ = load(`${D}/requests-profile/0151.json`).request;
if (process.env.TOOLSET === '10') {
  const keep = new Set(['describe_scene', 'point_at', 'pick_and_place', 'open_gripper', 'close_gripper', 'move_home',
    'reset_scene', 'verify_last_action', 'world_state', 'emergency_stop'].map(n => `cascade__${n}`));
  TOOLS_REQ.tools = TOOLS_REQ.tools.filter(t => keep.has(t.function.name));
}
const IMAGE_REQ = load(`${D}/requests-profile/0150.json`).request;
const WS_REQ = load(`${D}/requests-profile/0136.json`).request;
const trace = fs.readFileSync(`${WT}/runs/mcp_3877787_032b2e4a5f644cab8ef8a31614aca847/trace.jsonl`, 'utf8')
  .split('\n').filter(Boolean).map(l => JSON.parse(l));
const mock = load(`${D}/mock_results.json`);
const imageUrl = IMAGE_REQ.messages.at(-1).content.find(p => p.type === 'image_url').image_url.url;
const imageData = imageUrl.split(',')[1];

const deepReplace = (v, pairs) => {
  if (typeof v === 'string') return pairs.reduce((s, [a, b]) => s.split(a).join(b), v);
  if (Array.isArray(v)) return v.map(x => deepReplace(x, pairs));
  if (v && typeof v === 'object') return Object.fromEntries(Object.entries(v).map(([k, x]) => [k, deepReplace(x, pairs)]));
  return v;
};
const pickToSquare = trace[3].result;   // green cube -> green square, ok, physics-confirmed
const pickToBox = trace[4].result;      // orange -> open box, ok, physics-confirmed
const resetOk = trace[1].result;
const describeOk = trace[2].result;
const wsText = WS_REQ.messages.at(-1).content;

function result(name, args) {
  const short = name.replace(/^cascade__/, '');
  let r;
  let image = false;
  if (short === 'pick_and_place') {
    const obj = String(args.object ?? 'object');
    const dest = String(args.destination ?? 'green square');
    r = dest.includes('box') ? deepReplace(pickToBox, [['orange fruit', obj], ['orange', obj], ['open box', dest]])
      : deepReplace(pickToSquare, [['green_cube', obj.replace(/ /g, '_')], ['green cube', obj], ['green square', dest]]);
  } else if (short === 'point_at') {
    r = deepReplace(mock.point_at, [['red cube', String(args.label ?? args.object ?? 'object')]]);
  } else if (short === 'move_home') r = mock.move_home;
  else if (short === 'open_gripper') r = mock.open_gripper;
  else if (short === 'close_gripper') r = {gripper: 'closed', ok: true, outcome: 'ok'};
  else if (short === 'reset_scene') r = resetOk;
  else if (short === 'describe_scene') { r = describeOk; image = true; }
  else if (short === 'world_state') return {text: wsText, image: false};
  else if (short === 'localize_object') r = {ok: true, label: args.label ?? args.object, position: [0.32, 0.005, 0.03]};
  else if (short === 'verify_last_action') r = {ok: true, verified: true};
  else if (short === 'emergency_stop') r = {ok: true, stopped: true};
  else r = {ok: true, outcome: 'ok'};
  return {text: JSON.stringify(r), image};
}

const CASES = [
  // [id, prompt, expected calls [name, args]]
  ['S_green', 'Could you put the green cube in the green square?', [['pick_and_place', {object: 'green cube', destination: 'green square'}]]],
  ['S_orange', 'Please put the orange in the open box.', [['pick_and_place', {object: 'orange', destination: 'open box'}]]],
  ['S_point', 'Point at the lemon.', [['point_at', {label: 'lemon'}]]],
  ['S_reset', 'Let\'s start over.', [['reset_scene', {}]]],
  ['S_describe', 'What can you see on the table?', [['describe_scene', {}]]],
  ['S_ws', 'What is the current world state?', [['world_state', {}]]],
  ['S_home', 'Go home.', [['move_home', {}]]],
  ['S_open', 'Open the gripper.', [['open_gripper', {}]]],
  ['M_box_then_sq', 'Please put the pink cube in the open box, then put the lemon in the green square.',
    [['pick_and_place', {object: 'pink cube', destination: 'open box'}], ['pick_and_place', {object: 'lemon', destination: 'green square'}]]],
  ['M_sq_then_box', 'Please put the pink cube in the green square, then put the lemon in the open box.',
    [['pick_and_place', {object: 'pink cube', destination: 'green square'}], ['pick_and_place', {object: 'lemon', destination: 'open box'}]]],
  ['M_green_orange', 'Put the green cube in the green square and then put the orange in the open box.',
    [['pick_and_place', {object: 'green cube', destination: 'green square'}], ['pick_and_place', {object: 'orange', destination: 'open box'}]]],
  ['M_reset_ws', 'Let\'s start over, then show the session status with the world_state tool.',
    [['reset_scene', {}], ['world_state', {}]]],
  ['M_point_home', 'Point at the lemon, then go home.', [['point_at', {label: 'lemon'}], ['move_home', {}]]],
  ['M_open_home', 'Open the gripper, then go home.', [['open_gripper', {}], ['move_home', {}]]],
  ['M3_open_point_home', 'Open the gripper, point at the lemon, then go home.',
    [['open_gripper', {}], ['point_at', {label: 'lemon'}], ['move_home', {}]]],
  ['S_dropzone', 'Please put the pink cube in the drop zone.',
    [['pick_and_place', {object: 'pink cube', destination: 'green square', material: 'fragile'}]]],
  ['S_dropzone_live', 'Please put the pink cube in the drop zone.',
    [['pick_and_place', {object: 'pink cube', destination: 'green square', material: 'fragile'}]],
    load(`${D}/loop/live_pick_ok_fragile.json`)],
  ['F_pick_stuck', 'Please put the pink cube in the green square.',
    [['pick_and_place', {object: 'pink cube', destination: 'green square'}]], trace[6].result],
  ['F_point_cuda', 'Point at the lemon.', [['point_at', {label: 'lemon'}]], trace[5].result],
  ['F_M_first_fails', 'Please put the pink cube in the green square, then put the lemon in the open box.',
    [['pick_and_place', {object: 'pink cube', destination: 'green square'}]], trace[6].result],
];

function toOpenAI(systemPrompt, messages) {
  const out = [{role: 'system', content: systemPrompt}];
  let n = 0;
  for (const m of messages) {
    if (m.role === 'user') out.push({role: 'user', content: m.content});
    else if (m.role === 'assistant') {
      const calls = m.content.filter(p => p.type === 'toolCall');
      const text = m.content.filter(p => p.type === 'text').map(p => p.text).join('');
      out.push({role: 'assistant', content: calls.length ? null : text,
        ...(calls.length ? {tool_calls: calls.map(c => ({id: c.id, type: 'function',
          function: {name: c.name, arguments: JSON.stringify(c.arguments)}}))} : {})});
    } else if (m.role === 'toolResult') {
      n += 1;
      const text = m.content.filter(p => p.type === 'text').map(p => p.text).join('\n');
      out.push({role: 'tool', content: text, tool_call_id: m.toolCallId});
      const images = m.content.filter(p => p.type === 'image');
      if (images.length) {
        out.push({role: 'user', content: [{type: 'text', text: `Image(s) from tool result #${n} (${m.toolName}):`},
          ...images.map(p => ({type: 'image_url', image_url: {url: `data:${p.mimeType};base64,${p.data}`}}))]});
      }
    }
  }
  return out;
}

function dropNextAction(messages) {
  return messages.map((m) => {
    if (m.role !== 'toolResult') return m;
    return {...m, content: m.content.map((p) => {
      if (p.type !== 'text') return p;
      try {
        const v = JSON.parse(p.text);
        if (v && typeof v === 'object' && v.ok === true && 'next_action' in v) {
          const {next_action: _drop, ...rest} = v;
          return {...p, text: plugin.spacedJson(rest)};
        }
      } catch { /* not json */ }
      return p;
    })};
  });
}

function doneList(messages) {
  const calls = plugin.toolCalls(messages);
  const results = messages.filter(m => m.role === 'toolResult');
  return calls.map((c, i) => `${c.name}(${JSON.stringify(c.arguments)}): ${plugin.resultFailed(results[i]) ? 'failed' : 'ok'}`)
    .join('; ');
}


// The request and this request's calls/results as ONE user message (text form).
function oneMessage(messages, header, style = 'list') {
  const user = messages.find(m => m.role === 'user');
  const text = typeof user.content === 'string' ? user.content : user.content.filter(p => p.type === 'text').map(p => p.text).join('\n');
  const calls = plugin.toolCalls(messages);
  const results = messages.filter(m => m.role === 'toolResult').map(plugin.compactToolResult);
  const lines = calls.map((c, i) => `${i + 1}. ${c.name}(${JSON.stringify(c.arguments)}) returned `
    + `${(results[i]?.content ?? []).filter(p => p.type === 'text').map(p => p.text).join(' ')}`);
  const images = results.flatMap(r => (r.content ?? []).filter(p => p.type === 'image'));
  let body = `Request: ${text}\n\n${header}\n${lines.join('\n')}`;
  if (style === 'tail') body += '\n\nReport these results to the user.';
  if (style === 'prose') {
    body = `The user asked: ${text}\n\n` + calls.map((c, i) => `The robot ran ${c.name} with ${JSON.stringify(c.arguments)}; it returned `
      + `${(results[i]?.content ?? []).filter(p => p.type === 'text').map(p => p.text).join(' ')}.`).join('\n') + '\n\nReport this to the user.';
  }
  return [{role: 'user', content: images.length ? [{type: 'text', text: body}, ...images] : body}];
}

let provider;
plugin.default.register({pluginConfig: {modelProviderId: 'cascade', modelId: 'cosmos3-edge', ...(process.env.ROUNDS ? {maxToolRounds: Number(process.env.ROUNDS)} : {})},
  on: () => {}, registerProvider: (p) => { provider = p; }});
const MODEL = {provider: 'cascade', id: 'cosmos3-edge'};
const toolNames = TOOLS_REQ.tools.map(t => t.function.name);
let VARIANT = 'dtr';
let CASE_ID = '';
let LOG = [];

function messageFrom(out) {
  const content = out.calls.length
    ? out.calls.slice(0, 1).map((c, i) => {
      let args = {};
      try { args = JSON.parse(c.arguments || '{}'); } catch { args = {_raw: c.arguments}; }
      return {type: 'toolCall', id: `gen${Date.now()}${i}`, name: c.name, arguments: args};
    })
    : [{type: 'text', text: out.text}];
  return {role: 'assistant', content, stopReason: out.calls.length ? 'toolUse' : 'stop'};
}

const fakeStream = message => ({
  async* [Symbol.asyncIterator]() { yield {type: 'start', partial: message}; yield {type: 'done', reason: message.stopReason, message}; },
  result: async () => message,
});

async function inner(model, context, options) {
  let ctx = context;
  const isContinue = context.systemPrompt === plugin.CONTINUE_PROMPT;
  const kind = isContinue ? 'continue' : (context.tools?.length ? 'tools' : 'reply');
  if (kind === 'reply' && context.systemPrompt === plugin.REPORT_PROMPT && (VARIANT.startsWith('text') || VARIANT.endsWith('textrep'))) {
    ctx = {...ctx, messages: oneMessage(context.messages, 'Tool results:')};
  }
  if (kind === 'reply' && (context.systemPrompt === plugin.REPORT_PROMPT || context.systemPrompt === plugin.REPLY_PROMPT) && VARIANT.endsWith('textall')) {
    ctx = {...ctx, systemPrompt: plugin.REPORT_PROMPT, messages: oneMessage(context.messages, 'Tool results:', VARIANT.includes('tail') ? 'tail' : VARIANT.includes('prose') ? 'prose' : 'list')};
  }
  if (isContinue && VARIANT.startsWith('gate')) {
    const G = {
      gdone: ["You check a robot arm's work. The user's request and the tool results follow. Did the tool results already do everything the request asks for? Answer only YES or NO.", /^\W*yes/i],
      gleft: ["You check a robot arm's work. The user's request and the tool results follow. Does the request ask for something that the tool results do not show as done? Answer only YES or NO.", /^\W*no/i],
    };
    const key = VARIANT.split('_').find(t => G[t]);
    const [sys, doneRe] = G[key];
    const gp = {...TOOLS_REQ, messages: toOpenAI(sys, oneMessage(context.messages, 'Tool results:', 'list'))};
    delete gp.tools; delete gp.tool_choice; delete gp.parallel_tool_calls;
    gp.max_completion_tokens = 4;
    const g = await send(gp);
    LOG.push({kind: 'gate', text: g.text, finish: g.finish, calls: []});
    if (doneRe.test(g.text.trim())) {
      return fakeStream({role: 'assistant', content: [{type: 'text', text: ''}], stopReason: 'stop'});
    }
    if (VARIANT.includes('textcont')) ctx = {...ctx, messages: oneMessage(context.messages, 'Tool calls made so far:')};
  }
  if (isContinue) {
    if (VARIANT === 'one') {
      LOG.push({kind, forced: 'no-call'});
      return fakeStream({role: 'assistant', content: [{type: 'text', text: ''}], stopReason: 'stop'});
    }
    const CP = {
      pa: "You control a robot arm through the cascade__* tools. The user's request and the tool calls made for it so far follow. A request can name several steps, such as 'do A, then do B'. If the request names a step that no call has done yet, call the one tool for that step. If every step is done, do not call a tool.",
      pb: "You control a robot arm through the cascade__* tools. The user's request and the tool calls made for it so far follow. A request can name several steps, such as 'do A, then do B'. If the request names a step that no call has done yet, call the one tool for that step. If every step is done, do not call a tool. Never repeat a call that already succeeded.",
    };
    const cp = VARIANT.split('_').find(t => CP[t]);
    if (cp) ctx = {...ctx, systemPrompt: CP[cp]};
    if (VARIANT === 'dtr_nonext' || VARIANT === 'dtr_done_nonext') ctx = {...ctx, messages: dropNextAction(ctx.messages)};
    if (VARIANT.startsWith('text')) {
      const src = VARIANT.includes('nonext') ? dropNextAction(context.messages) : context.messages;
      ctx = {...ctx, messages: oneMessage(src, 'Tool calls made so far:')};
    }
    if (VARIANT === 'dtr_done' || VARIANT === 'dtr_done_nonext') {
      ctx = {...ctx, systemPrompt: `${ctx.systemPrompt} Done so far: ${doneList(context.messages)}.`};
    }
  }
  const p = {...TOOLS_REQ, messages: toOpenAI(ctx.systemPrompt, ctx.messages)};
  if (!ctx.tools || ctx.tools.length === 0) delete p.tools;
  options?.onPayload?.(p, model);
  if (!p.tools) { delete p.tool_choice; delete p.parallel_tool_calls; }
  if (process.env.DUMP_DIR) fs.writeFileSync(`${process.env.DUMP_DIR}/${CASE_ID}-${LOG.length}-${kind}.json`, JSON.stringify(p));
  const out = await send(p);
  const message = messageFrom(out);
  LOG.push({kind, finish: out.finish, calls: out.calls, text: out.text.slice(0, 200), s: out.s});
  return fakeStream(message);
}
const wrapped = provider.wrapStreamFn({provider: 'cascade', modelId: 'cosmos3-edge', streamFn: inner});

async function send(payload) {
  const t0 = Date.now();
  const res = await fetch(`http://127.0.0.1:${port}/v1/chat/completions`, {method: 'POST',
    headers: {'content-type': 'application/json'}, body: JSON.stringify(payload)});
  const wire = await res.text();
  let text = ''; let finish = null; const calls = {};
  for (const line of wire.split('\n')) {
    if (!line.startsWith('data: ') || line === 'data: [DONE]') continue;
    const ev = JSON.parse(line.slice(6));
    for (const ch of ev.choices ?? []) {
      finish = ch.finish_reason ?? finish;
      text += ch.delta?.content ?? '';
      for (const c of ch.delta?.tool_calls ?? []) {
        const cur = calls[c.index] ??= {name: '', arguments: ''};
        cur.name += c.function?.name ?? '';
        cur.arguments += c.function?.arguments ?? '';
      }
    }
  }
  return {text, finish, calls: Object.values(calls), s: (Date.now() - t0) / 1000};
}

const norm = v => String(v ?? '').trim().toLowerCase();
const same = (call, [name, args]) => call.name === `cascade__${name}`
  && Object.entries(args).every(([k, v]) => norm(call.arguments[k]) === norm(v));

async function runCase([id, prompt, expected, firstResult], variant, rep) {
  VARIANT = variant;
  CASE_ID = id;
  LOG = [];
  const messages = [{role: 'user', content: prompt}];
  const made = [];
  let final = null; let ended = null;
  for (let round = 0; round < 8; round++) {
    let call;
    if (round === 0) {
      const [name, args] = expected[0];
      call = {type: 'toolCall', id: 'call0', name: `cascade__${name}`, arguments: args};
    } else {
      const stream = await wrapped(MODEL, {systemPrompt: 'OPENCLAW LONG PROMPT', messages, tools: toolNames}, {});
      for await (const _ of stream) { /* drain */ }
      const msg = await stream.result();
      call = msg.content.find(p => p.type === 'toolCall');
      if (!call) { final = msg.content.map(p => p.text ?? '').join('').trim(); ended = LOG.at(-1)?.kind; break; }
    }
    made.push(call);
    messages.push({role: 'assistant', content: [call]});
    const r = round === 0 && firstResult ? {text: JSON.stringify(firstResult), image: false} : result(call.name, call.arguments);
    const content = [{type: 'text', text: r.text}];
    if (r.image) content.push({type: 'image', data: imageData, mimeType: 'image/jpeg'});
    messages.push({role: 'toolResult', toolCallId: call.id, toolName: call.name, isError: false, content});
  }
  const exact = made.length === expected.length && expected.every((e, i) => same(made[i], e));
  const cont = LOG.filter(l => l.kind === 'continue' && !l.forced);
  const repeatDropped = cont.filter(l => l.calls.length && made.filter(m => m.name === l.calls[0].name).length
    && !made.slice(1).some(m => m.name === l.calls[0].name && JSON.stringify(m.arguments) === l.calls[0].arguments)).length;
  const row = {id, variant, rep, steps: expected.length,
    calls: made.map(c => `${c.name.replace('cascade__', '')}(${JSON.stringify(c.arguments)})`),
    n_calls: made.length, exact, extra: Math.max(0, made.length - expected.length), repeats: 0,
    step2_ok: expected.length > 1 && made.length > 1 && same(made[1], expected[1]),
    continue_calls: cont.map(l => l.calls.length ? `${l.calls[0].name.replace('cascade__', '')}(${l.calls[0].arguments})` : `TEXT:${JSON.stringify(l.text.slice(0, 60))}`),
    missing: expected.length > made.length, ended, final: final?.slice(0, 300), turns: LOG};
  fs.appendFileSync(outPath, `${JSON.stringify(row)}\n`);
  return row;
}

const reps = Number(repsArg);
for (const variant of variantsArg.split(',')) {
  for (const c of CASES.filter(c => !process.env.ONLY || process.env.ONLY.split(',').includes(c[0]))) {
    for (let rep = 0; rep < reps; rep++) {
      const row = await runCase(c, variant, rep);
      console.log(`${variant} ${row.id} r${rep} exact=${row.exact} n=${row.n_calls} | ${row.calls.join(' -> ')} | cont: ${row.continue_calls.join(' ; ')} | ${JSON.stringify(row.final)}`);
    }
  }
}
