const $ = id => document.getElementById(id);
let token = location.hash.slice(1); history.replaceState(null, '', location.pathname);
let socket, session, context, stream, capture, captureSource, playback;
let sequence = 0, generation = 0, playbackRevision = 0, playbackAllowed = false;
let captureRevision = 0, capturePending = false;
let connectionRevision = 0, pendingOperations = 0;
const transcripts = new Map();
function log(message) { $('log').textContent = ($('log').textContent + message + '\n').slice(-24000); }
function status(message) { $('status').textContent = message; }
async function api(path, method = 'POST', body) {
  const response = await fetch('/api/' + path, {method, headers: {'Authorization': 'Bearer ' + token,
    'Content-Type': 'application/json'}, body: body === undefined ? undefined : JSON.stringify(body)});
  if (!response.ok) throw new Error(await response.text());
  return response.json();
}
function flush() { playbackRevision++; if (playback) playback.flush(generation); }
function mute() { captureRevision++; capturePending = false;
  if (captureSource) { captureSource.disconnect(); captureSource = undefined; }
  if (capture) { capture.disconnect(); capture = undefined; }
  if (stream) { stream.getTracks().forEach(track => track.stop()); stream = undefined; }
  $('mic').textContent = 'Start microphone'; }
function connectionButton() { $('connect').disabled = pendingOperations > 0 || socket !== undefined; }
function beginOperation() { pendingOperations++; connectionButton(); }
function endOperation() { pendingOperations--; connectionButton(); }
function dropConnection() {
  const revision = ++connectionRevision, previous = socket;
  socket = undefined; session = undefined; playbackAllowed = false; mute(); flush();
  if (previous) previous.close();
  $('mic').disabled = $('send').disabled = $('disconnect').disabled = true;
  connectionButton(); return revision;
}
async function audioContext() {
  if (!context) context = new AudioContext({sampleRate: 24000});
  if (context.sampleRate !== 24000) throw new Error('This browser did not admit 24 kHz audio. Use text or another browser.');
  await context.resume(); return context;
}
function send(event) {
  if (!socket || socket.readyState !== WebSocket.OPEN || socket.bufferedAmount > 96000) {
    mute(); throw new Error('Audio transport unavailable or backlogged');
  }
  socket.send(JSON.stringify({...event, session_id: session.session_id}));
}
async function play(event) {
  if (!playbackAllowed || event.generation !== generation) return;
  const revision = playbackRevision;
  const ctx = await audioContext();
  if (!playbackAllowed || revision !== playbackRevision || event.generation !== generation) return;
  if (!playback) playback = new PcmPlaybackQueue(ctx, fail);
  playback.generation = generation;
  const bytes = Uint8Array.from(atob(event.audio), c => c.charCodeAt(0));
  playback.enqueue(bytes, event.generation);
}
async function fail(error) {
  beginOperation(); dropConnection(); status(error.message);
  try { await api('stop'); await api('session', 'DELETE'); } catch {}
  finally { endOperation(); }
}
$('connect').onclick = async () => {
  if (pendingOperations || socket) return;
  beginOperation(); const revision = ++connectionRevision;
  let attemptedSession = false;
  try {
    const info = await api('status', 'GET');
    if (revision !== connectionRevision) return;
    attemptedSession = true;
    const binding = await api('session', 'POST', {robot_id: info.robot_id});
    if (revision !== connectionRevision) { await api('session', 'DELETE'); return; }
    session = binding;
    sequence = 0; generation = 0; flush(); playbackAllowed = true;
    const transport = new WebSocket(location.origin.replace('http', 'ws') + '/api/media?ticket=' + encodeURIComponent(binding.ticket));
    socket = transport;
    const current = () => revision === connectionRevision && socket === transport && session === binding;
    transport.onopen = () => { if (!current()) { transport.close(); return; }
      status('Connected · ' + info.robot_id); $('mic').disabled = $('send').disabled = $('disconnect').disabled = false; };
    transport.onmessage = async ({data}) => {
      if (!current()) return;
      try {
        const event = JSON.parse(data);
        if (event.session_id !== binding.session_id) throw new Error('Session mismatch');
        if (event.type === 'flush') { generation = event.generation; flush(); }
        else if (event.type === 'audio') await play(event);
        else if (event.type === 'transcript') {
          const speaker = event.event.startsWith('response.') ? 'Robot' : 'You';
          const key = speaker + ':' + (event.item_id || event.response_id || 'current');
          const final = event.event.endsWith('.done') || event.event.endsWith('.completed');
          const text = final ? event.text : (transcripts.get(key) || '') + event.text;
          transcripts.set(key, text);
          if (transcripts.size > 128) transcripts.delete(transcripts.keys().next().value);
          if (final) log(speaker + ': ' + text);
        }
        else if (event.type === 'authority_revoked') {
          mute(); playbackAllowed = false; flush();
          $('mic').disabled = $('send').disabled = true;
          status('Conversation interrupted. Disconnect and reconnect; reset a latched stop explicitly.');
        }
        else if (event.type === 'tool_result') log(event.tool + ': ' + JSON.stringify(event.result));
      } catch (error) { if (current()) await fail(error); }
    };
    transport.onclose = () => { if (!current()) return; dropConnection();
      status('Disconnected. Robot stop requested; reset requires your explicit action.'); };
    transport.onerror = () => { if (current()) return fail(new Error('Media socket failed')); };
  } catch (error) {
    if (revision === connectionRevision) { dropConnection(); status(error.message); }
    // No newer connect is admitted while this operation owns an unfinished POST.
    if (attemptedSession) { try { await api('session', 'DELETE'); } catch {} }
  } finally { endOperation(); }
};
$('mic').onclick = async () => {
  if (stream || capturePending) { mute(); return; }
  const revision = ++captureRevision, owner = session, transport = socket;
  const current = () => revision === captureRevision && session === owner && socket === transport &&
    transport?.readyState === WebSocket.OPEN && playbackAllowed;
  capturePending = true;
  $('mic').textContent = 'Cancel microphone request';
  try {
    const ctx = await audioContext();
    if (!current()) return;
    await ctx.audioWorklet.addModule('/capture.js');
    if (!current()) return;
    const acquired = await navigator.mediaDevices.getUserMedia({audio: {channelCount: 1, echoCancellation: true, noiseSuppression: true}});
    if (!current()) { acquired.getTracks().forEach(track => track.stop()); return; }
    stream = acquired;
    capture = new AudioWorkletNode(ctx, 'pcm-capture');
    capture.port.onmessage = ({data}) => { try {
      if (!current() || stream !== acquired) return;
      const bytes = new Uint8Array(data); let raw = ''; for (const byte of bytes) raw += String.fromCharCode(byte);
      send({type: 'audio', sequence: sequence++, audio: btoa(raw)});
    } catch (error) { fail(error); } };
    captureSource = ctx.createMediaStreamSource(stream);
    captureSource.connect(capture); capture.connect(ctx.destination);
    $('mic').textContent = 'Mute microphone';
  } catch (error) { if (current()) await fail(error); }
  finally { if (revision === captureRevision) { capturePending = false;
    if (!stream) $('mic').textContent = 'Start microphone'; } }
};
$('text-form').onsubmit = event => { event.preventDefault(); try { send({type: 'text', text: $('text').value}); $('text').value = ''; } catch (error) { fail(error); } };
async function operatorAction(action) {
  beginOperation(); const revision = dropConnection();
  try {
    if (action === 'stop') log(JSON.stringify(await api('stop')));
    await api('session', 'DELETE');
    if (revision !== connectionRevision) return;
    if (action === 'reset') {
      const observed = await api('status', 'GET');
      if (revision !== connectionRevision) return;
      if (!Number.isSafeInteger(observed.generation) || observed.generation < 0)
        throw new Error('Invalid robot generation');
      const result = await api('reset', 'POST', {generation: observed.generation}); log(JSON.stringify(result));
      if (revision === connectionRevision) status(result.ok ? 'Stop reset. Connect a new conversation to continue.' : 'Reset refused; inspect stop receipt.');
    } else status(action === 'stop' ? 'Stopped. Explicitly reset before reconnecting.' : 'Disconnected. Robot stop requested.');
  } catch (error) { if (revision === connectionRevision) status(error.message); }
  finally { endOperation(); }
}
$('stop').onclick = () => operatorAction('stop');
$('reset').onclick = () => operatorAction('reset');
$('disconnect').onclick = () => operatorAction('disconnect');
api('status', 'GET').then(info => { if (!connectionRevision) status('Ready · ' + info.robot_id + ' · ' + info.tools.length + ' allowed tools'); }).catch(error => { if (!connectionRevision) status(error.message); });
