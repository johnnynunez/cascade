// CASCADE's agent profile for NVIDIA Cosmos3-Edge (a 2.4B reasoner) on the
// OpenShell/OpenClaw route. Measured on OpenClaw's own captured requests
// (docs/evidence/cosmos-vllm-032-nvfp4-2026-10-19/openclaw-route): with
// OpenClaw's JSON metadata envelopes (the user turn's "Conversation info:
// ⟦openclaw:ctx⟧ ```json …```" block, the system prompt's inbound_meta ```json```
// block), its ~10k-character system prompt and all 49 robot tools, the model
// imitates the JSON blocks and answers with a ```json {"action": …}``` code
// block instead of a native tool call. This plugin changes only what the model
// reads: only the current request and this run's tool calls/results reach it,
// the system prompt becomes the short robot prompt below (also the
// prompt hook's, so OpenClaw's own report shows it), and every user turn reaches
// the model without OpenClaw's inbound context envelope (wrapStreamFn: the
// provider request is built from this context, after OpenClaw appended its
// "### Message Context" block to the system prompt). One tool per request by
// default (an opt-in bounded loop: maxToolRounds); then the model reports the
// results in a short turn without tools, reading them without the server's
// diagnostics. The model still chooses every tool call and writes every reply
// itself; nothing here answers for it.
// The tool allowlist and sampling params live in render-config.mjs.

export const SYSTEM_PROMPT = 'You control a robot arm through the cascade__* tools. '
  + 'When the user asks for an action or about the scene, call the one matching tool, '
  + 'wait for its result, then answer briefly in English. Never invent tool results.';

const MARKER = '\u27e6openclaw:ctx\u27e7';
const TIMESTAMP = /^\[[A-Za-z]{3} \d{4}-\d{2}-\d{2} \d{2}:\d{2}[^\]]*\] */;

// OpenClaw's inbound context: an optional leading envelope timestamp, then
// blocks whose header line ends with the provenance marker, each followed by a
// ```json fenced body (formatContextJsonBlock) or by plain lines up to a blank
// line (chat history). The text after them is the user's own message.
export function stripInboundEnvelope(text) {
  if (typeof text !== 'string' || (!text.includes(MARKER) && !TIMESTAMP.test(text))) return text;
  const lines = text.replace(TIMESTAMP, '').split('\n');
  const kept = [];
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i].trim();
    if (line.length > MARKER.length && line.endsWith(MARKER)) {
      let j = i + 1;
      if (lines[j]?.trim() === '```json') {
        j += 1;
        while (j < lines.length && lines[j].trim() !== '```') j += 1;
        j += 1;
      } else {
        while (j < lines.length && lines[j].trim() !== '') j += 1;
      }
      i = j - 1;
      continue;
    }
    kept.push(lines[i]);
  }
  return kept.join('\n').trim();
}

export function stripUserMessage(message) {
  if (message?.role !== 'user') return message;
  if (typeof message.content === 'string') return {...message, content: stripInboundEnvelope(message.content)};
  if (!Array.isArray(message.content)) return message;
  return {...message, content: message.content.map(part => (part?.type === 'text'
    ? {...part, text: stripInboundEnvelope(part.text)} : part))};
}

// The current request and this run's own tool calls and results: the model
// copies earlier turns (after a readiness "OK", "What is the robot's status?"
// was answered "OK"; a session-status question after it came back as JSON
// prose). Earlier requests stay in OpenClaw's session; the robot server keeps
// its own task memory (task_memory, verify_last_action).
export function currentTurn(messages) {
  const list = messages ?? [];
  const last = list.findLastIndex(m => m?.role === 'user');
  return last > 0 ? list.slice(last) : list;
}

export function afterToolResult(messages) {
  const list = messages ?? [];
  return list.length > 0 && list[list.length - 1]?.role === 'toolResult';
}

// This request's tool calls, in order.
export function toolCalls(messages) {
  return (messages ?? []).flatMap(m => (m?.role === 'assistant' && Array.isArray(m.content)
    ? m.content.filter(part => part?.type === 'toolCall') : []));
}

// A tool result that failed: an MCP error, or the robot server's "ok": false.
export function resultFailed(message) {
  if (message?.isError === true) return true;
  const parts = Array.isArray(message?.content) ? message.content : [];
  return parts.some((part) => {
    if (part?.type !== 'text' || typeof part.text !== 'string') return false;
    try {
      const value = JSON.parse(part.text);
      return Boolean(value) && typeof value === 'object' && value.ok === false;
    } catch {
      return false;
    }
  });
}

// Tool rounds per request: ONE by default (pluginConfig.maxToolRounds, 1-3).
// Asked again with tools and the tool prompt after a result, the 2.4B model
// never ended its turn (it re-placed a cube a verified pick_and_place had
// already put in the green square, then retried a failure eight times). The
// bounded loop below (maxToolRounds 2-3, decide-then-report) was measured on
// replays and on the live captures (docs/evidence/cosmos-vllm-032-nvfp4-2026-10-19/
// openclaw-route): it finished some two-step requests ("point at the lemon,
// then go home"), but after a completed pick_and_place it also called
// place_on_object on the captured live request (8 of 8 replays; it failed the
// launcher's readiness proof live), so one tool per request stays the default.
// With the loop on, after a successful result the model is asked, with tools
// and CONTINUE_PROMPT, whether the request still needs a step: a NEW tool call
// runs; no call, or a call repeating one this request already made (same tool,
// same arguments), ends the loop, and so does a failed result or the round
// limit. Either way the turn ends with the short report turn below (no
// tools). The model decides every call; the plugin only bounds the loop and
// never writes a call or a reply itself.
export const DEFAULT_TOOL_ROUNDS = 1;
export const MAX_TOOL_ROUNDS = 3;

export function toolRounds(value) {
  if (value === undefined || value === null) return DEFAULT_TOOL_ROUNDS;
  if (!Number.isInteger(value) || value < 1 || value > MAX_TOOL_ROUNDS) {
    throw new Error(`maxToolRounds must be an integer from 1 to ${MAX_TOOL_ROUNDS}`);
  }
  return value;
}

export function nextTurn(messages, rounds = DEFAULT_TOOL_ROUNDS) {
  if (!afterToolResult(messages)) return 'tools';
  if (resultFailed(messages.at(-1)) || toolCalls(messages).length >= rounds) return 'reply';
  return 'continue';
}

const canonical = (value) => {
  if (typeof value === 'string') return value.trim().toLowerCase();
  if (Array.isArray(value)) return value.map(canonical);
  if (value && typeof value === 'object') {
    return Object.fromEntries(Object.keys(value).sort().map(key => [key, canonical(value[key])]));
  }
  return value;
};

// The same tool with the same arguments (strings compared trimmed, any case).
export function sameCall(a, b) {
  return a?.name === b?.name
    && JSON.stringify(canonical(a?.arguments ?? {})) === JSON.stringify(canonical(b?.arguments ?? {}));
}

export function repeatsCall(call, messages) {
  return toolCalls(messages).some(done => sameCall(call, done));
}

// The report turn (no tools). The request and this request's calls and their
// results reach the model as ONE user message, read in the robot server's
// JSON style (compactToolResult). Measured on replays: as chat tool turns,
// two or more results made the model answer with junk ("座", "ザ"), one
// result was often copied back as JSON (point_at), and when the model had
// stopped after step 1 of "put X there, then put Y there" the reply claimed
// both steps; in this form each case was reported, and an unfinished step
// was named as not done. The earlier single-result prompt: a reset was
// answered "ok" and a failed pick "I'm sorry, but I can't help with that."
export const REPORT_PROMPT = 'You control a robot arm. Tools ran for the user\'s request; their results follow, '
  + 'in order. In one or two short English sentences, tell the user what happened: for each result with ok true, '
  + 'what was done; for a result with ok false, that it failed and why. Report only what these results show.';

const textOf = content => (typeof content === 'string' ? content
  : (Array.isArray(content) ? content.filter(part => part?.type === 'text').map(part => part.text).join('\n') : ''));

export function reportMessages(messages) {
  const list = messages ?? [];
  const request = textOf(list.find(m => m?.role === 'user')?.content);
  const results = list.filter(m => m?.role === 'toolResult').map(compactToolResult);
  const lines = toolCalls(list).map((call, i) => `${i + 1}. ${call.name}(${JSON.stringify(call.arguments ?? {})}) `
    + `returned ${results[i] ? textOf(results[i].content).replace(/\s*\n\s*/g, ' ') : 'nothing'}`);
  return [{role: 'user', content: `Request: ${request}\n\nTool results:\n${lines.join('\n')}\n\n`
    + 'Report these results to the user.'}];
}

// A result with a camera image (describe_scene, get_observation): asked what
// was done, the model answered "The camera image is below."; asked about the
// image, it described the scene. Such a report keeps the chat form.
export const IMAGE_REPLY_PROMPT = 'You control a robot arm. A camera image follows. Answer the user\'s question '
  + 'about the scene from that image in one or two short English sentences.';

export function endsWithImage(messages) {
  const last = (messages ?? []).at(-1);
  return Array.isArray(last?.content) && last.content.some(part => part?.type === 'image');
}

export function replyPrompt(messages) {
  return endsWithImage(messages) ? IMAGE_REPLY_PROMPT : REPORT_PROMPT;
}

export function replyMessages(messages) {
  return endsWithImage(messages) ? (messages ?? []).map(compactToolResult) : reportMessages(messages);
}

// The continuation turn's system prompt (nextTurn 'continue'). Measured
// against more explicit wordings ("a request can name several steps ...",
// "never repeat a call that already succeeded"): those started more second
// steps but also invented place_at coordinates and reset the scene unasked.
export const CONTINUE_PROMPT = 'You control a robot arm through the cascade__* tools. The user\'s request and '
  + 'the tool calls made for it so far follow. If the request asks for a step that is not done yet, call the one '
  + 'tool for the next step. If every step is done, do not call a tool: tell the user in one short English '
  + 'sentence what was done.';

export const DIAGNOSTIC_KEYS = new Set([
  'perception_timing', 'timings', 'evidence_path', 'evidence_sha256', 'measured', 'capabilities', 'backends',
  'session_state', 'grasp_memory', 'events', 'grip_profile', 'post_place_retreat', 'duration_s', 'cameras',
  'camera_statistics_note', 'tools_withheld', 'tools_hidden_by_operator', 'age_s', 'conf', 'last_path',
]);

const empty = v => v === null || v === undefined || (Array.isArray(v) && v.length === 0)
  || (typeof v === 'object' && !Array.isArray(v) && Object.keys(v).length === 0);

export function pruneResult(value, depth = 0) {
  if (Array.isArray(value)) {
    if (depth >= 3) return undefined;
    return value.slice(0, 12).map(v => pruneResult(v, depth + 1)).filter(v => v !== undefined);
  }
  if (value && typeof value === 'object') {
    if (depth >= 3) return undefined;
    const out = {};
    for (const [key, v] of Object.entries(value)) {
      if (DIAGNOSTIC_KEYS.has(key)) continue;
      const kept = pruneResult(v, depth + 1);
      if (!empty(kept)) out[key] = kept;
    }
    return out;
  }
  if (typeof value === 'string' && value.length > 300) return `${value.slice(0, 300)}...`;
  return value;
}

// The robot server's own JSON style (Python json.dumps: ", " and ": "). The
// same pruned failure, re-serialized compactly, was reported as "The pink cube
// is not reachable for placement."; spaced, as "Failed: <the error>".
export function spacedJson(value) {
  if (Array.isArray(value)) return `[${value.map(spacedJson).join(', ')}]`;
  if (value && typeof value === 'object') {
    return `{${Object.entries(value).map(([k, v]) => `${JSON.stringify(k)}: ${spacedJson(v)}`).join(', ')}}`;
  }
  return JSON.stringify(value);
}

// Only what the model reads changes: OpenClaw's session, its tool summary and
// the operator's chat keep the robot server's full result. A result with an
// image (describe_scene, get_observation) is the image: given the JSON next to
// it, the model copied the JSON (live, "What can you see on the table?");
// given only the outcome and a pointer to the image, it described the scene
// ("I see a wooden box, a green square mat, ... a pink cube, a green cube").
export const IMAGE_POINTER = 'the camera image below';

export function compactToolResult(message) {
  if (message?.role !== 'toolResult' || !Array.isArray(message.content)) return message;
  const image = message.content.some(part => part?.type === 'image');
  return {...message, content: message.content.map((part) => {
    if (part?.type !== 'text' || typeof part.text !== 'string') return part;
    let parsed;
    try {
      parsed = JSON.parse(part.text);
    } catch {
      return part;
    }
    if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) return part;
    if (image) {
      const outcome = Object.fromEntries(['ok', 'error'].filter(k => k in parsed).map(k => [k, parsed[k]]));
      return {...part, text: spacedJson({...outcome, image: IMAGE_POINTER})};
    }
    const pruned = pruneResult(parsed);
    // "ok" first: read last, a failed point_at (e-stop latched) was reported
    // as "The tool ran successfully."; read first, as a failure.
    const okFirst = 'ok' in pruned ? {ok: pruned.ok, ...pruned} : pruned;
    return {...part, text: spacedJson(okFirst)};
  })};
}

export const REPLY_MAX_TOKENS = 200;

function replyOnly(options) {
  const original = options?.onPayload;
  return {...options, onPayload: (payload, model) => {
    if (payload && typeof payload === 'object') {
      delete payload.tools;
      delete payload.tool_choice;
      delete payload.parallel_tool_calls;
      for (const key of ['max_completion_tokens', 'max_tokens']) {
        if (typeof payload[key] === 'number') payload[key] = Math.min(payload[key], REPLY_MAX_TOKENS);
      }
    }
    return original?.(payload, model);
  }};
}

// A stream that replays the events of one already finished stream.
export function replayedStream(events, message) {
  return {
    async* [Symbol.asyncIterator]() { yield* events; },
    result: async () => message,
  };
}

// The continuation turn: a NEW tool call goes to OpenClaw as the model streamed
// it; no call, or a repeated one, is dropped and the report turn answers.
export async function continueOrReport(continued, report, messages) {
  const stream = await continued();
  const events = [];
  for await (const event of stream) events.push(event);
  const message = await stream.result();
  if (message?.stopReason === 'error' || message?.stopReason === 'aborted') return replayedStream(events, message);
  const call = (Array.isArray(message?.content) ? message.content : []).find(part => part?.type === 'toolCall');
  if (call && !repeatsCall(call, messages)) return replayedStream(events, message);
  return report();
}

export default {
  id: 'cascade-cosmos-prompt',
  name: 'CASCADE Cosmos3-Edge prompt',
  register(api) {
    const {modelProviderId, modelId, maxToolRounds} = api.pluginConfig ?? {};
    if (![modelProviderId, modelId].every(v => typeof v === 'string' && v.trim() === v && v.length)) {
      throw new Error('Explicit Cosmos model scope required');
    }
    const rounds = toolRounds(maxToolRounds);
    const inScope = ctx => ctx?.modelProviderId === modelProviderId && ctx.modelId === modelId;
    api.on('before_prompt_build', (_event, ctx) => (inScope(ctx) ? {systemPrompt: SYSTEM_PROMPT} : undefined));
    api.registerProvider({
      id: modelProviderId, label: 'CASCADE local brain', auth: [],
      wrapStreamFn(ctx) {
        const inner = ctx.streamFn;
        if (!inner || ctx.provider !== modelProviderId || ctx.modelId !== modelId) return inner;
        return (model, context, options) => {
          if (model?.provider !== modelProviderId || model?.id !== modelId) return inner(model, context, options);
          const messages = currentTurn(context.messages).map(stripUserMessage);
          const turn = nextTurn(messages, rounds);
          const report = () => inner(model, {...context, systemPrompt: replyPrompt(messages),
            messages: replyMessages(messages), tools: []}, replyOnly(options));
          if (turn === 'reply') return report();
          if (turn === 'continue') {
            const continued = () => inner(model, {...context, systemPrompt: CONTINUE_PROMPT,
              messages: messages.map(compactToolResult)}, options);
            return continueOrReport(continued, report, messages);
          }
          return inner(model, {...context, systemPrompt: SYSTEM_PROMPT, messages}, options);
        };
      },
    });
  },
};
