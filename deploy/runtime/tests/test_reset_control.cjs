const assert = require('node:assert/strict');
const test = require('node:test');
const {ORDER, COMPLETE, UNKNOWN, blocker, resetConfirmed, runReset} = require('../../../web/openclaw-ui/paai-reset.js');

function tool(runId = 'reset-run', result = {}, isError = false) {
  return {role: 'toolResult', toolName: 'cascade__reset_scene', isError,
    __openclaw: {runId}, content: [{type: 'text', text: JSON.stringify({ok: true,
      observation_refreshed: true, props_reset: ['green_cube', 'orange'], ...result})}]};
}

function fixture({messages = [tool()], terminal = 'ok', afterRead} = {}) {
  const calls = [], reports = [], sent = [];
  let tick = 0, submitted = false;
  const user = {role: 'user', content: ORDER, __openclaw: {idempotencyKey: 'reset-run:user'}};
  const state = {connected: true, sessionKey: 'agent:main:fixture', assistantAgentId: 'main',
    hasPendingInitialTurn: () => false,
    chatMessage: 'move the orange to the box', chatAttachments: ['draft-photo'], chatRunId: null,
    client: {async request(method, params) {
      calls.push({method, params});
      if (method === 'sessions.get') {
        if (!submitted) { afterRead?.(state); return {messages: [tool('old-run')]}; }
        return {messages: [tool('old-run'), user, ...messages]};
      }
      if (method === 'agent.wait') return {status: terminal};
      throw Error('Unexpected RPC');
    }},
    async handleSendChat(text) { submitted = true; sent.push(text); return true; },
  };
  return {state, calls, reports, sent,
    run: () => runReset(state, value => reports.push(value), {
      now: () => tick, sleep: async ms => { tick += ms; }, timeoutMs: 5000,
    })};
}

test('one click sends the established chat order and checks its actual reset tool result', async () => {
  const h = fixture();
  assert.equal(await h.run(), COMPLETE);
  assert.deepEqual(h.sent, [ORDER]);
  assert.deepEqual(h.reports, ['Sending reset…', 'Resetting scene…']);
  assert.equal(h.state.chatMessage, 'move the orange to the box');
  assert.deepEqual(h.state.chatAttachments, ['draft-photo']);
  assert.deepEqual(h.calls.filter(c => c.method === 'agent.wait').map(c => c.params.runId), ['reset-run']);
  assert.ok(h.calls.every(c => ['sessions.get', 'agent.wait'].includes(c.method)));
});

test('disconnected, loading, running, queued and archived chats cannot submit a reset', async () => {
  for (const patch of [{connected: false}, {chatLoading: true}, {chatSending: true},
    {chatRunId: 'active'}, {chatQueue: ['pending']}, {pendingAbort: true},
    {hasPendingInitialTurn: () => true}, {selectedChatSessionArchived: true}, {chatReplyTarget: {}}]) {
    const h = fixture(); Object.assign(h.state, patch);
    assert.ok(blocker(h.state));
    assert.notEqual(await h.run(), COMPLETE);
    assert.deepEqual(h.sent, []);
    assert.deepEqual(h.calls, []);
  }
});

test('an order starting during the preflight read prevents reset submission', async () => {
  const h = fixture({afterRead: state => { state.chatRunId = 'another-order'; }});
  assert.equal(await h.run(), UNKNOWN);
  assert.deepEqual(h.sent, []);
});

test('a changed session during the preflight read cannot receive the reset', async () => {
  const h = fixture({afterRead: state => { state.sessionKey = 'another-session'; }});
  assert.equal(await h.run(), UNKNOWN);
  assert.deepEqual(h.sent, []);
});

test('a successful assistant reply alone never certifies a reset', async () => {
  const h = fixture({messages: [{role: 'assistant', content: [{type: 'text', text: 'Scene reset.'}],
    __openclaw: {runId: 'reset-run'}}]});
  assert.equal(await h.run(), UNKNOWN);
  assert.equal(h.sent.length, 1);
});

test('failed, partial, duplicate, malformed and old reset results remain unconfirmed', () => {
  for (const messages of [[tool('old-run')], [tool('reset-run', {ok: false})],
    [tool('reset-run', {observation_refreshed: false})], [tool('reset-run', {props_reset: []})],
    [tool('reset-run', {}, true)], [tool(), tool()],
    [{...tool(), content: [{type: 'text', text: 'Scene reset.'}]}]]) {
    assert.equal(resetConfirmed(messages, 'reset-run'), false);
  }
});

test('a completed reset with a failed model reply is reported distinctly', async () => {
  assert.equal(await fixture({terminal: 'error'}).run(),
    'Scene reset. The chat reply failed; check the cameras.');
  assert.equal(await fixture({terminal: 'error', messages: []}).run(), UNKNOWN);
});

test('pending results time out without resubmitting the order', async () => {
  const h = fixture({terminal: 'timeout'});
  assert.equal(await h.run(), UNKNOWN);
  assert.deepEqual(h.sent, [ORDER]);
});

test('losing the result connection never retries a potentially completed reset', async () => {
  const h = fixture();
  h.state.handleSendChat = async text => {
    h.sent.push(text);
    h.state.client.request = async () => { throw Error('disconnected'); };
    return true;
  };
  assert.equal(await h.run(), UNKNOWN);
  assert.deepEqual(h.sent, [ORDER]);
});

test('an unaccepted native send is reported without polling or retrying', async () => {
  const h = fixture();
  h.state.handleSendChat = async text => { h.sent.push(text); return false; };
  assert.equal(await h.run(), UNKNOWN);
  assert.deepEqual(h.sent, [ORDER]);
  assert.equal(h.calls.length, 1);
});
