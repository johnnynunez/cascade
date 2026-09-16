/* Reset the kitchen through the current OpenClaw conversation. */
(() => {
  'use strict';
  const ORDER = 'Reset the scene.';
  const COMPLETE = 'Scene reset. Check the cameras before the next order.';
  const UNKNOWN = 'Reset not confirmed. Check the chat before trying again.';

  function messageRun(message) {
    const meta = message.__openclaw;
    return meta?.runId || (meta?.idempotencyKey?.endsWith(':user')
      ? meta.idempotencyKey.slice(0, -5) : null);
  }

  function messageText(message) {
    return typeof message.content === 'string' ? message.content
      : (message.content || []).filter(part => part.type === 'text').map(part => part.text).join('\n');
  }

  function blocker(state) {
    if (!state?.connected || !state.client || typeof state.handleSendChat !== 'function') {
      return 'Connect OpenClaw to reset the scene.';
    }
    const pending = typeof state.hasPendingInitialTurn === 'function'
      ? state.hasPendingInitialTurn(state.sessionKey) : state.hasPendingInitialTurn;
    if (state.chatLoading || state.chatSending || state.chatRunId || state.chatQueue?.length
        || pending || state.pendingAbort) {
      return 'Wait for the current order to finish.';
    }
    if (state.chatReplyTarget || state.selectedChatSessionArchived) {
      return 'Open an active chat without a reply selected.';
    }
    return '';
  }

  function resetConfirmed(messages, runId) {
    const tools = messages.filter(message => message.__openclaw?.runId === runId
      && message.role === 'toolResult' && message.toolName === 'cascade__reset_scene');
    if (tools.length !== 1 || tools[0].isError === true) return false;
    return (Array.isArray(tools[0].content) ? tools[0].content : []).some(part => {
      if (part.type !== 'text') return false;
      try {
        const result = JSON.parse(part.text);
        return result.ok === true && result.observation_refreshed === true
          && Array.isArray(result.props_reset) && result.props_reset.length > 0;
      } catch { return false; }
    });
  }

  async function runReset(state, report, {
    now = Date.now, sleep = ms => new Promise(resolve => setTimeout(resolve, ms)),
    timeoutMs = 360000,
  } = {}) {
    const reason = blocker(state);
    if (reason) return reason;
    const client = state.client, key = state.sessionKey, agentId = state.assistantAgentId;
    const until = now() + timeoutMs;
    const request = async (method, params) => {
      let timer;
      try {
        return await Promise.race([
          client.request(method, params),
          new Promise((_, reject) => { timer = setTimeout(() => reject(Error('timeout')), 15000); }),
        ]);
      } finally { clearTimeout(timer); }
    };
    const transcript = () => request('sessions.get', {key, agentId, limit: 100});
    try {
      report('Sending reset…');
      const before = await transcript();
      const known = new Set(before.messages.map(messageRun).filter(Boolean));
      // Recheck after the read: a normal send may have started in the meantime.
      if (state.client !== client || state.sessionKey !== key || blocker(state)) return UNKNOWN;
      // This is the same handler as the send arrow. A text override preserves
      // the attendee's draft and does not attach their queued files.
      if (await state.handleSendChat(ORDER) !== true) return UNKNOWN;
      let runId = known.has(state.chatRunId) ? null : state.chatRunId;
      report('Resetting scene…');
      while (now() < until) {
        await sleep(1500);
        const current = await transcript();
        const runs = [...new Set(current.messages.filter(message => message.role === 'user'
          && !known.has(messageRun(message)) && messageText(message) === ORDER)
          .map(messageRun).filter(Boolean))];
        if (runs.length > 1) return UNKNOWN;
        runId ||= runs[0];
        if (!runId) continue;
        const terminal = await request('agent.wait', {runId, timeoutMs: 1000});
        if (!['ok', 'error'].includes(terminal.status)) continue;
        const final = await transcript();
        if (!resetConfirmed(final.messages, runId)) return UNKNOWN;
        return terminal.status === 'ok' ? COMPLETE
          : 'Scene reset. The chat reply failed; check the cameras.';
      }
    } catch { /* A transport failure does not prove whether the robot moved. */ }
    return UNKNOWN;
  }

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = {ORDER, COMPLETE, UNKNOWN, blocker, resetConfirmed, runReset};
    return;
  }

  let control, button, status, running = false, result = '', timer, lastSession, lastRun;
  const pane = () => document.querySelector('openclaw-chat-pane.chat-pane-cache__pane--active');
  function update() {
    const active = pane();
    const container = active?.querySelector('.chat-main__conversation-column');
    if (!container) { control?.remove(); return; }
    const state = active.state;
    if (!running && (state.sessionKey !== lastSession || state.chatSending
        || (state.chatRunId && state.chatRunId !== lastRun))) result = '';
    lastSession = state.sessionKey; lastRun = state.chatRunId;
    if (!control) {
      control = document.createElement('div'); control.className = 'paai-reset';
      button = document.createElement('button'); button.type = 'button';
      button.id = 'paai-reset'; button.className = 'paai-reset__button'; button.textContent = 'Reset';
      button.title = 'Reset the kitchen through OpenClaw';
      button.setAttribute('aria-describedby', 'paai-reset-status');
      status = document.createElement('span'); status.id = 'paai-reset-status';
      status.setAttribute('role', 'status'); status.setAttribute('aria-live', 'polite');
      control.append(button, status);
      button.addEventListener('click', async () => {
        if (running || blocker(pane()?.state)) return;
        running = true; result = ''; update();
        result = await runReset(pane().state, message => { result = message; update(); });
        running = false; update();
      });
    }
    if (control.parentElement !== container) container.prepend(control);
    const reason = blocker(state);
    button.disabled = running || Boolean(reason);
    button.textContent = running ? 'Resetting…' : 'Reset';
    const message = result || reason || 'Return the arm and objects to their starting positions.';
    if (status.textContent !== message) status.textContent = message;
  }
  function start() { update(); timer = setInterval(update, 500); }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', start, {once: true});
  else start();
  addEventListener('pagehide', () => clearInterval(timer));
  addEventListener('pageshow', event => { if (event.persisted) start(); });
})();
