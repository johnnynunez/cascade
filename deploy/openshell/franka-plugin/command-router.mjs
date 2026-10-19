// Explicit operator commands can use CASCADE's deterministic reflex tier.
// Callers must supply a trusted current inbound message, bind it to one run,
// filter the returned tool against that run's catalog, and prevent replay.
// Never pass model prose, tool output, images, or assembled prompts here.

const blockedWords = /\b(?:if|unless|until|when|whenever|while|before|after|then|otherwise|except|but|not|no|never|neither|without|instead|maybe|perhaps|suppose|assuming|provided|imagine|example|quote|explain|ignore|avoid|cancel|abort)\b/i;
const labelControlWords = /\b(?:and|or|nor|also|only|please|pick|grab|take|put|place|drop|move|close|stop|resume|reset|restart|return|execute|run|can|could|would|should|must|might|may|do|as|once|because|depending)\b/i;
const pronoun = /^(?:it|this|that|these|those|something|anything|everything|whatever|here|there)$/i;

function label(value) {
  const name = value.replace(/^(?:the|a|an)\s+/i, '').trim();
  if (!name || name.length > 80 || name.split(' ').length > 8
      || !/^[\p{L}\p{N}][\p{L}\p{N} -]*$/u.test(name)
      || (labelControlWords.test(name) && name !== 'drop zone') || pronoun.test(name)
      // "open box" describes a destination; "open the gripper" is a command.
      || /\bopen\b/.test(name) && !/^open (?:[\p{L}\p{N}-]+ ){0,2}(?:box|container|bowl|tray)$/u.test(name)) return null;
  return name;
}

function command(name, args, intent) {
  return {name, arguments: args, intent};
}

/** Compile one explicit English operator command; uncertainty returns null. */
export function parseRoutineCommand(text) {
  if (typeof text !== 'string' || text.length > 240 || /[\x00-\x1f\x7f\u2028\u2029]/u.test(text)
      || /^ {4}/.test(text)
      || /["'`“”‘’<>\[\]{};:\\/]/u.test(text) || blockedWords.test(text)) return null;
  let input = text.trim().toLowerCase().replace(/[.!?]$/, '').trim().replace(/\s+/g, ' ');
  for (let i = 0; i < 3; i += 1) {
    const next = input.replace(/^(?:please|hey|can you|could you|would you)\s+/, '');
    if (next === input) break;
    input = next;
  }
  if (!input) return null;

  // Reset is deliberately absent: a scene reset is not permission to resume.
  if (/^(?:stop|emergency stop|stop (?:the )?(?:arm|robot)(?: now)?)$/.test(input))
    return command('emergency_stop', {}, 'stop');
  if (/^(?:resume|resume & home|resume and (?:return|go)(?: to)? home|resume and return to the home position)$/.test(input))
    return command('reset_stop', {}, 'resume');
  if (/^(?:home|go home|move home|return home|return to (?:the )?home(?: position)?|back to home)$/.test(input))
    return command('move_home', {}, 'home');
  if (/^(?:open|close) (?:the )?gripper$/.test(input)) {
    const action = input.startsWith('open') ? 'open' : 'close';
    return command(`${action}_gripper`, {}, `${action}_gripper`);
  }
  if (/^(?:look|look around|observe|scan|what do you see|what is on the table|describe (?:the )?(?:scene|table))$/.test(input))
    return command('get_observation', {}, 'observe');
  if (/^(?:status|robot status|arm status|world state|show (?:me )?(?:the )?(?:robot )?status)$/.test(input))
    return command('world_state', {}, 'status');
  if (/^(?:show (?:me )?(?:the )?(?:d435i )?camera)$/.test(input))
    return command('camera_snapshot', input.includes('d435i') ? {camera: 'd435i'} : {}, 'camera');

  let match = /^(?:place|put)(?: (?:it|this|that|(?:the )?held object))? (?:in|into|inside|on|onto) (.+)$/.exec(input);
  if (match) {
    const destination = label(match[1]);
    return destination ? command('place_on_object', {label: destination}, 'place_held') : null;
  }
  match = /^(?:pick(?: up)?|grab|take) (.+?) and (?:place|put)(?: it)? (?:in|into|inside|on|onto) (.+)$/.exec(input)
    ?? /^pick and place (.+?) (?:in|into|inside|on|onto) (.+)$/.exec(input)
    ?? /^(?:place|put) (.+?) (?:in|into|inside|on|onto) (.+)$/.exec(input);
  if (match) {
    const object = label(match[1]), destination = label(match[2]);
    return object && destination
      ? command('pick_and_place', {object, destination}, 'pick_and_place') : null;
  }
  match = /^(?:pick(?: up)?|grab|take) (.+)$/.exec(input);
  if (match) {
    // Do not turn a destination-bearing or compound request into pick-only.
    if (/\b(?:in|into|inside|on|onto|to|at)\b/.test(match[1])) return null;
    const object = label(match[1]);
    return object ? command('grasp_object', {label: object}, 'pick') : null;
  }
  return null;
}

function emptyUsage() {
  return {input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0,
    cost: {input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0}};
}

/** A local reflex response using OpenClaw's documented native event protocol.
 * This never calls a provider or executes a tool. Host policy/MCP own execution.
 * Tool idempotency and run authorization belong to the caller, not this stream.
 */
export function syntheticAssistantStream(model, {text = '', toolCall = null, signal} = {}) {
  if (!model || typeof model.api !== 'string' || typeof model.provider !== 'string'
      || typeof model.id !== 'string' || typeof text !== 'string')
    throw new TypeError('Explicit model identity and string narration required');
  if (toolCall && (typeof toolCall.id !== 'string' || !toolCall.id
      || typeof toolCall.name !== 'string' || !toolCall.name
      || !toolCall.arguments || typeof toolCall.arguments !== 'object'
      || Array.isArray(toolCall.arguments))) throw new TypeError('Invalid native tool call');
  const call = toolCall ? {type: 'toolCall', id: toolCall.id, name: toolCall.name,
    arguments: structuredClone(toolCall.arguments)} : null;
  const base = {role: 'assistant', content: [], api: model.api, provider: model.provider,
    model: model.id, usage: emptyUsage(), stopReason: call ? 'toolUse' : 'stop', timestamp: Date.now(),
    diagnostics: [{type: 'cascade_reflex', timestamp: Date.now(),
      details: {source: 'operator_command', modelInvoked: false}}]};
  const snapshot = content => ({...base, content: structuredClone(content)});
  const content = [];
  const events = [{type: 'start', partial: snapshot(content)}];
  if (text) {
    content.push({type: 'text', text: ''});
    events.push({type: 'text_start', contentIndex: 0, partial: snapshot(content)});
    content[0].text = text;
    events.push({type: 'text_delta', contentIndex: 0, delta: text, partial: snapshot(content)},
      {type: 'text_end', contentIndex: 0, content: text, partial: snapshot(content)});
  }
  if (call) {
    const index = content.length;
    content.push({...call, arguments: {}});
    events.push({type: 'toolcall_start', contentIndex: index, partial: snapshot(content)});
    content[index] = call;
    events.push({type: 'toolcall_delta', contentIndex: index, delta: JSON.stringify(call.arguments),
      partial: snapshot(content)},
    {type: 'toolcall_end', contentIndex: index, toolCall: structuredClone(call), partial: snapshot(content)});
  }
  const terminal = snapshot(content);
  events.push({type: 'done', reason: terminal.stopReason, message: terminal});
  const queue = [];
  let wake, complete = false, resolveResult;
  const result = new Promise(resolve => { resolveResult = resolve; });
  function push(event) { queue.push(event); wake?.(); wake = undefined; }
  queueMicrotask(async () => {
    let final = terminal;
    try {
      for (const event of events) {
        if (event.type !== 'start' && signal?.aborted) {
          final = {...snapshot([]), stopReason: 'aborted', errorMessage: 'Operator command cancelled'};
          push({type: 'error', reason: 'aborted', error: final});
          return;
        }
        push(event);
        await Promise.resolve();
      }
    } finally {
      complete = true;
      resolveResult(final);
      wake?.();
      wake = undefined;
    }
  });
  return {
    async *[Symbol.asyncIterator]() {
      while (!complete || queue.length) {
        if (queue.length) yield queue.shift();
        else await new Promise(resolve => { wake = resolve; });
      }
    },
    result: () => result,
  };
}

function jsonObject(text) {
  try {
    const parsed = JSON.parse(text);
    return parsed && typeof parsed === 'object' && !Array.isArray(parsed) ? parsed : null;
  } catch { return null; }
}

function clipped(value, limit = 1500) {
  const text = typeof value === 'string' ? value : JSON.stringify(value);
  if (typeof text !== 'string') return '';
  return text.length <= limit ? text : `${text.slice(0, limit - 1)}…`;
}

// Inspect only named execution records. Detector labels and tool metadata are
// data, never nested verdicts or instructions to execute another command.
function executionRecords(records) {
  const result = [];
  function visit(row, path = '', depth = 0) {
    if (!row || typeof row !== 'object' || Array.isArray(row) || result.length >= 24) return;
    result.push({row, path});
    if (depth >= 3) return;
    for (const key of ['grasp', 'pick', 'place', 'placed', 'return_home', 'home', 'post_place_retreat', 'result', 'execution']) {
      visit(row[key], path ? `${path}.${key}` : key, depth + 1);
    }
  }
  for (const row of records) visit(row);
  return result;
}

const failureStatus = /^(?:failed|failure|error|refuted|stuck|aborted|cancelled|canceled)$/i;
function hasFailure(row) {
  return row.ok === false || row.execution_ok === false || row.isError === true
    || row.cancelled === true || row.canceled === true
    || ['status', 'outcome', 'verdict'].some(key => typeof row[key] === 'string' && failureStatus.test(row[key]))
    || row.postcondition?.status === 'refuted'
    || (typeof row.error === 'string' && row.error.trim() !== '');
}

function isUnverified(row) {
  return row.verified === false || row.grip_verified === false
    || row.verdict === 'unverified' || row.postcondition?.status === 'unverified';
}

/** Return a terminal report only for the actual result of our exact native call.
 * JSON is data: retain the tool's verdict and diagnostics, never infer success
 * from a normal transport return, a holding label, or an assistant message.
 */
export function formatRoutineResult(toolResult, expectedCall) {
  if (!expectedCall?.id || !expectedCall?.name || toolResult?.role !== 'toolResult'
      || toolResult.toolCallId !== expectedCall.id || toolResult.toolName !== expectedCall.name) return null;
  const texts = (Array.isArray(toolResult.content) ? toolResult.content : [])
    .filter(block => block?.type === 'text' && typeof block.text === 'string').map(block => block.text);
  const records = texts.map(jsonObject).filter(Boolean);
  const execution = executionRecords(records);
  const states = execution.flatMap(({row}) => [row, row.robot, row.arm_motion,
    row.motion_control, row.robot?.motion_control]).filter(row => row && typeof row === 'object');
  const failed = toolResult.isError === true || execution.some(({row}) => hasFailure(row));
  const stopped = states.some(row => row.stopped === true || row.estop_latched === true
    || row.motion_blocked === true || row.latched === true || row.motion_ready === false);
  const unverified = execution.some(({row}) => isUnverified(row));
  const verified = !failed && !unverified && !stopped && records.some(row =>
    row.verified === true || row.postcondition?.status === 'confirmed');
  const readOnly = /(?:^|_)(?:get_observation|world_state|camera_snapshot)$/.test(expectedCall.name);
  let verdict = 'CASCADE returned a result without an execution verdict.';
  if (failed) verdict = 'CASCADE reported a failure.';
  else if (stopped) verdict = 'CASCADE reports motion is stopped or blocked.';
  else if (verified) verdict = 'CASCADE verified the requested action.';
  else if (records.some(row => row.resumed === true)) verdict = 'CASCADE reports the arm resumed.';
  else if (readOnly) verdict = 'CASCADE observation:';
  else if (records.some(row => row.ok === true || row.execution_ok === true))
    verdict = 'CASCADE completed the tool call; the requested physical outcome is not verified.';
  const notes = [verdict];
  if (stopped && failed) notes.push('Motion remains stopped or blocked.');
  if (unverified) notes.push('The physical outcome is unverified.');
  const diagnostics = new Set();
  for (const {row, path} of execution) {
    if (path && hasFailure(row)) notes.push(`${path}: failed.`);
    for (const key of ['error', 'reason', 'motion_block_reason', 'verification_note']) {
      if (row[key] !== undefined && row[key] !== null && row[key] !== '')
        diagnostics.add(`${path ? `${path}.` : ''}${key}: ${clipped(row[key])}`);
    }
    if (row.postcondition?.evidence) diagnostics.add(`Verification: ${clipped(row.postcondition.evidence)}`);
  }
  for (const row of states) {
    if (row.motion_error) diagnostics.add(`Motion error: ${clipped(row.motion_error)}`);
  }
  notes.push(...[...diagnostics].slice(0, 6));

  for (const row of records) {
    const home = row.return_home ?? row.home?.return_home ?? row.home;
    if (home && typeof home === 'object') {
      if (hasFailure(home)) notes.push('Return home: failed.');
      else if (home.attempted === false || home.status === 'skipped') notes.push('Return home: not performed.');
      else if (home.ok === true) notes.push('Return home: controller reports completion.');
      else notes.push('Return home: completion not established.');
    }
    if (row.grip_verified === true) notes.push('Grasp contact: verified by the tool.');
    else if (row.grip_verified === false) notes.push('Grasp contact: unverified.');
    if (row.destination !== undefined) notes.push(`Requested destination: ${clipped(row.destination, 160)}.`);
  }
  const tracked = states.find(row => row.tracked_holding?.label !== undefined
    || Object.hasOwn(row, 'holding') || Object.hasOwn(row, 'held'));
  if (tracked) {
    const held = tracked.tracked_holding?.label ?? tracked.holding ?? tracked.held;
    notes.push(held ? `Tracked holding: ${clipped(held, 160)} (session state).`
      : 'Tracked holding: none recorded; current gripper contents are not established.');
  }
  if (states.some(row => row.live_arm_feedback === false)) notes.push('This status does not contain fresh arm feedback.');
  else {
    const measured = states.find(row => Array.isArray(row.tcp_xyz)
      && row.tcp_xyz.length === 3 && row.tcp_xyz.every(Number.isFinite));
    if (measured) notes.push(`Reported TCP (m): [${measured.tcp_xyz.join(', ')}].`);
  }
  const grip = states.find(row => Number.isFinite(row.gripper_open_frac));
  if (grip) notes.push(`Reported gripper opening fraction: ${grip.gripper_open_frac}.`);

  const seen = records.find(row => Array.isArray(row.objects_visible));
  const remembered = records.find(row => Array.isArray(row.objects_tracked) || Array.isArray(row.objects));
  const objects = seen?.objects_visible ?? remembered?.objects_tracked ?? remembered?.objects;
  if (objects) {
    const labels = objects.slice(0, 12).map(item => {
      if (typeof item === 'string') return clipped(item, 100);
      if (!item || typeof item.label !== 'string') return null;
      const name = [typeof item.color === 'string' ? item.color : '', item.label].filter(Boolean).join(' ');
      const state = typeof item.state === 'string' ? ` [${clipped(item.state, 32)}]` : '';
      return `${clipped(name, 100)}${state}`;
    }).filter(Boolean);
    if (labels.length) notes.push(`${seen ? 'Detector estimates (unconfirmed)' : 'Tracked labels (not a fresh scene inventory)'}: ${labels.join('; ')}${objects.length > 12 ? '; …' : ''}.`);
    else notes.push(seen ? 'No detections were returned; this does not establish an empty scene.'
      : 'No tracked labels were returned; this does not establish an empty scene.');
  }
  if (Array.isArray(toolResult.content) && toolResult.content.some(block => block?.type === 'image'))
    notes.push('Camera image is in the tool result; it does not verify home position or gripper contents.');
  if (!records.length) {
    notes.push('No structured execution or verification verdict was reported.');
    if (texts.length) notes.push(`Tool response: ${clipped(texts.join('\n'))}`);
  }
  notes.push('Full details are in the tool result.');
  return [...new Set(notes)].join('\n');
}
