const $ = id => document.getElementById(id);
let token = location.hash.slice(1); history.replaceState(null, '', location.pathname);
let socket, session, context, stream, capture, sequence = 0, generation = 0, playAt = 0;
const sources = new Set();
const transcripts = new Map();
function log(message) { $('log').textContent = ($('log').textContent + message + '\n').slice(-24000); }
function status(message) { $('status').textContent = message; }
async function api(path, method = 'POST', body) {
  const response = await fetch('/api/' + path, {method, headers: {'Authorization': 'Bearer ' + token,
    'Content-Type': 'application/json'}, body: body === undefined ? undefined : JSON.stringify(body)});
  if (!response.ok) throw new Error(await response.text());
  return response.json();
}
function flush() { for (const source of sources) { try { source.stop(); } catch {} } sources.clear(); playAt = 0; }
function mute() { if (capture) { capture.disconnect(); capture = undefined; }
  if (stream) { stream.getTracks().forEach(track => track.stop()); stream = undefined; }
  $('mic').textContent = 'Start microphone'; }
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
  if (event.generation !== generation) return;
  const ctx = await audioContext();
  if (event.generation !== generation) return;
  const bytes = Uint8Array.from(atob(event.audio), c => c.charCodeAt(0));
  if (!bytes.length || bytes.length % 2 || bytes.length > 96000) throw new Error('Invalid playback audio');
  const buffer = ctx.createBuffer(1, bytes.length / 2, 24000);
  const samples = buffer.getChannelData(0), view = new DataView(bytes.buffer);
  for (let i=0;i<samples.length;i++) samples[i] = view.getInt16(2*i, true)/32768;
  playAt = Math.max(ctx.currentTime + .02, playAt);
  if (playAt + buffer.duration - ctx.currentTime > 2) throw new Error('Playback exceeds two seconds');
  const source = ctx.createBufferSource(); source.buffer = buffer; source.connect(ctx.destination);
  sources.add(source); source.onended = () => sources.delete(source);
  source.start(playAt); playAt += buffer.duration;
}
async function fail(error) {
  status(error.message); mute(); flush();
  if (socket) socket.close();
  try { await api('stop'); } catch {}
}
$('connect').onclick = async () => {
  $('connect').disabled = true;
  try {
    const info = await api('status', 'GET');
    session = await api('session', 'POST', {robot_id: info.robot_id});
    sequence = 0; generation = 0; flush();
    socket = new WebSocket(location.origin.replace('http', 'ws') + '/api/media?ticket=' + encodeURIComponent(session.ticket));
    socket.onopen = () => { status('Connected · ' + info.robot_id); $('mic').disabled = $('send').disabled = $('disconnect').disabled = false; };
    socket.onmessage = async ({data}) => {
      try {
        const event = JSON.parse(data);
        if (event.session_id !== session.session_id) throw new Error('Session mismatch');
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
        else if (event.type === 'tool_result') log(event.tool + ': ' + JSON.stringify(event.result));
      } catch (error) { await fail(error); }
    };
    socket.onclose = () => { mute(); flush(); status('Disconnected. Robot stop requested; reset requires your explicit action.');
      $('connect').disabled = false; $('mic').disabled = $('send').disabled = $('disconnect').disabled = true; };
    socket.onerror = () => fail(new Error('Media socket failed'));
  } catch (error) { status(error.message); $('connect').disabled = false; }
};
$('mic').onclick = async () => {
  if (stream) { mute(); return; }
  try {
    const ctx = await audioContext();
    await ctx.audioWorklet.addModule('/capture.js');
    stream = await navigator.mediaDevices.getUserMedia({audio: {channelCount: 1, echoCancellation: true, noiseSuppression: true}});
    capture = new AudioWorkletNode(ctx, 'pcm-capture');
    capture.port.onmessage = ({data}) => { try {
      if (!stream) return;
      const bytes = new Uint8Array(data); let raw = ''; for (const byte of bytes) raw += String.fromCharCode(byte);
      send({type: 'audio', sequence: sequence++, audio: btoa(raw)});
    } catch (error) { fail(error); } };
    ctx.createMediaStreamSource(stream).connect(capture); capture.connect(ctx.destination);
    $('mic').textContent = 'Mute microphone';
  } catch (error) { await fail(error); }
};
$('text-form').onsubmit = event => { event.preventDefault(); try { send({type: 'text', text: $('text').value}); $('text').value = ''; } catch (error) { fail(error); } };
$('stop').onclick = async () => { mute(); flush(); try { log(JSON.stringify(await api('stop'))); await api('session', 'DELETE'); status('Stopped. Explicitly reset before reconnecting.'); } catch (error) { status(error.message); } };
$('reset').onclick = async () => { mute(); flush(); try { await api('session', 'DELETE'); const result = await api('reset'); log(JSON.stringify(result)); status(result.ok ? 'Stop reset. Connect a new conversation to continue.' : 'Reset refused; inspect stop receipt.'); } catch (error) { status(error.message); } };
$('disconnect').onclick = async () => { mute(); flush(); try { await api('session', 'DELETE'); } catch (error) { status(error.message); } };
api('status', 'GET').then(info => status('Ready · ' + info.robot_id + ' · ' + info.tools.length + ' allowed tools')).catch(error => status(error.message));
