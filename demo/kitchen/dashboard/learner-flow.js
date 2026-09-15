/* The learner monitor follows live checks and an authenticated chat handshake. */
(function (root) {
  'use strict';
  const document = root.document;
  if (!document) return;
  const panel = document.getElementById('launch-monitor');
  const opener = document.querySelector('[data-learner-openclaw]');
  if (!panel || !opener) return;
  const state = document.getElementById('launch-monitor-state');
  const note = document.getElementById('launch-monitor-note');
  const retry = document.getElementById('launch-retry');
  const rows = new Map([...panel.querySelectorAll('[data-step]')].map(row => [row.dataset.step, row]));
  const labels = {waiting:'Waiting', active:'Active', done:'Done', failed:'Needs attention'};
  const icons = {waiting:'○', active:'●', done:'✓', failed:'!'};
  const waitingDetails = {demo:'Make sure the kitchen is ready to respond.',cameras:'Connect Kitchen, Worktop, and Side.',chat:'Prepare your OpenClaw conversation.',ready:'Confirm the chat is connected.'};
  let popup = null, sequence = 0, running = false, chatReady = false, chatError = '', timer, controller;

  function step(name, value, detail) {
    const row = rows.get(name);
    if (!row) return;
    row.dataset.state = value;
    row.querySelector('.monitor-icon').textContent = icons[value];
    row.querySelector('.monitor-phase').textContent = labels[value];
    if (detail) row.querySelector('.monitor-detail').textContent = detail;
  }
  function phase(value, message) {
    panel.dataset.state = value;
    panel.setAttribute('aria-busy', String(value === 'active'));
    state.textContent = value === 'done' ? 'Ready' : labels[value];
    note.textContent = message;
    retry.hidden = value !== 'failed';
    opener.setAttribute('aria-busy', String(value === 'active'));
  }
  function fail(name, message) {
    clearTimeout(timer);
    controller?.abort();
    running = false;
    step(name, 'failed', message);
    phase('failed', message);
  }
  function finish() {
    if (!running || !chatReady) return;
    step('chat', 'done', 'Your OpenClaw conversation is open.');
    step('ready', 'done', 'The chat confirmed it is connected and ready to receive a message.');
    running = false;
    clearTimeout(timer);
    phase('done', 'Ready. Send the observation order first, and keep Worktop in view.');
    document.dispatchEvent(new CustomEvent('paai-learner-ready'));
  }
  async function status() {
    controller = new AbortController();
    const deadline = setTimeout(() => controller.abort(), 8000);
    try {
      const response = await fetch('/api/status', {cache:'no-store', credentials:'omit', redirect:'error', signal:controller.signal});
      if (!response.ok) throw Error('status unavailable');
      const body = await response.text();
      if (body.length > 65536) throw Error('invalid status');
      const value = JSON.parse(body), timestamp = Date.parse(value.generated_at);
      if (value.version !== 1 || !Number.isFinite(timestamp) || Date.now() - timestamp > 120000 || timestamp > Date.now() + 30000) throw Error('stale status');
      return value;
    } finally { clearTimeout(deadline); }
  }
  async function run(event) {
    event?.preventDefault();
    if (running) { try { popup?.focus(); } catch {} return; }
    const current = ++sequence;
    clearTimeout(timer);
    controller?.abort();
    running = true; chatReady = false; chatError = '';
    for (const name of rows.keys()) step(name, 'waiting', waitingDetails[name]);
    phase('active', 'Checking the kitchen and preparing your OpenClaw conversation.');
    step('demo', 'active', 'Reading the demo’s current readiness.');
    // A direct user click reserves the tab before asynchronous checks begin.
    popup = root.open('/openclaw', 'paai-openclaw-chat');
    if (!popup) { fail('demo', 'The chat tab was blocked. Allow this page to open it, then click Retry connection.'); return; }
    try {
      const data = await status();
      if (current !== sequence || !running) return;
      if (!data.interactive_ready) { fail('demo', 'The kitchen is not ready yet. Wait for the booth operator to restore it, then retry.'); return; }
      step('demo', 'done', 'The kitchen and robot are ready to respond.');
      step('cameras', 'active', 'Checking for current Kitchen, Worktop, and Side views.');
      const discovery = data.camera_discovery;
      if (discovery?.version !== 1 || discovery.live !== true || !Array.isArray(discovery.cameras) || !['kitchen','worktop','side'].every(name => discovery.cameras.some(camera => camera.name === name))) {
        fail('cameras', 'The live cameras are reconnecting. Wait for a Live label, then click Retry connection.'); return;
      }
      step('cameras', 'done', 'Kitchen, Worktop, and Side are reporting live images.');
      step('chat', 'done', 'Your OpenClaw tab has opened.');
      step('ready', 'active', 'Waiting for the chat to confirm it is connected.');
      phase('active', 'OpenClaw is connecting. Keep this guide open; the final check updates automatically.');
      if (chatError) { fail('ready', chatError); return; }
      if (chatReady) { finish(); return; }
      timer = setTimeout(() => {
        if (current === sequence && running) fail('ready', 'The chat has not confirmed a connection yet. Return here and click Retry connection.');
      }, 90000);
    } catch {
      if (current === sequence && running) fail('demo', 'The demo did not answer this check. Refresh the guide, then click Retry connection.');
    }
  }
  root.addEventListener('message', event => {
    if (event.origin !== root.location.origin || event.source !== popup || !event.data || typeof event.data !== 'object') return;
    if (event.data.type === 'paai-openclaw-ready') {
      chatReady = true;
      if (rows.get('ready').dataset.state === 'active') finish();
    }
    if (event.data.type === 'paai-openclaw-error') {
      chatError = 'The chat could not finish connecting. Click Retry connection to try again.';
      if (running && rows.get('ready').dataset.state === 'active') fail('ready', chatError);
    }
  });
  opener.addEventListener('click', run);
  retry.addEventListener('click', run);
  root.addEventListener('pagehide', () => { sequence++; clearTimeout(timer); controller?.abort(); running = false; });
})(globalThis);
