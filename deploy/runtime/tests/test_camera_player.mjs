import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import vm from 'node:vm';
import { CameraPlayer, mountCameraViewer } from '../web/camera-player/viewer.mjs';

const source = readFileSync(new URL('../web/camera-player/mediamtx-reader.js', import.meta.url), 'utf8');
const origin = 'http://100.64.0.20:8092';
const endpoint = origin + '/camera-video/worktop/whep';
const offer = 'v=0\r\na=ice-ufrag:test\r\na=ice-pwd:fixture\r\nm=video 9 UDP/TLS/RTP/SAVPF 96\r\n';
const flush = async () => { for (let i = 0; i < 35; i++) await Promise.resolve(); };
const deferred = () => { let resolve, reject; const promise = new Promise((a, b) => { resolve = a; reject = b; }); return { promise, resolve, reject }; };

class Clock {
  time = 0;
  next = 1;
  jobs = new Map();
  setTimeout = (fn, delay) => { const id = this.next++; this.jobs.set(id, { fn, at: this.time + delay }); return id; };
  clearTimeout = (id) => this.jobs.delete(id);
  setInterval = (fn, delay) => { const id = this.next++; this.jobs.set(id, { fn, at: this.time + delay, delay }); return id; };
  clearInterval = this.clearTimeout;
  async advance(ms) {
    const end = this.time + ms;
    for (;;) {
      const next = [...this.jobs].filter(([, job]) => job.at <= end).sort((a, b) => a[1].at - b[1].at)[0];
      if (!next) break;
      const [id, job] = next;
      this.time = job.at;
      this.jobs.delete(id);
      if (job.delay) this.jobs.set(id, { ...job, at: this.time + job.delay });
      job.fn();
      await flush();
    }
    this.time = end;
    await flush();
  }
}

function response(status = 204, location) {
  return new Response(status === 201 ? 'v=0\r\n' : null, {
    status, headers: location ? { Location: location } : {},
  });
}

function readerEnvironment(handler, { connected = true } = {}) {
  const clock = new Clock(), peers = [], calls = [];
  class Peer {
    transceivers = [];
    connectionState = 'new';
    answers = [];
    constructor() { peers.push(this); }
    addTransceiver(kind) { this.transceivers.push(kind); }
    createDataChannel() { throw new Error('Video viewer must not allocate a data channel'); }
    createOffer() { return Promise.resolve({ type: 'offer', sdp: offer }); }
    setLocalDescription() { return Promise.resolve(); }
    setRemoteDescription(answer) {
      this.answers.push(answer);
      if (connected) { this.connectionState = 'connected'; this.onconnectionstatechange?.(); }
      return Promise.resolve();
    }
    close() { this.connectionState = 'closed'; this.onconnectionstatechange?.(); }
    fail() { this.connectionState = 'failed'; this.onconnectionstatechange?.(); }
  }
  const sandbox = { URL, AbortController, RTCPeerConnection: Peer,
    RTCSessionDescription: class { constructor(value) { Object.assign(this, value); } },
    setTimeout: clock.setTimeout, clearTimeout: clock.clearTimeout,
    fetch: async (url, options) => {
      const call = { url, ...options }; calls.push(call);
      return handler ? handler(call, calls) : response(options.method === 'POST' ? 201 : 204,
        options.method === 'POST' ? '/camera-video/worktop/whep/session-1' : undefined);
    },
  };
  sandbox.window = sandbox;
  vm.runInNewContext(source, sandbox, { filename: 'mediamtx-reader.js' });
  return { Reader: sandbox.MediaMTXWebRTCReader, clock, peers, calls };
}

test('video-only reader allocates one receiving video peer and close deletes its session once', async () => {
  const env = readerEnvironment();
  const reader = new env.Reader({ url: endpoint, videoOnly: true });
  await flush();
  assert.equal(env.peers.length, 1);
  assert.deepEqual(env.peers[0].transceivers, ['video']);
  await reader.close(); await reader.close();
  assert.equal(env.peers[0].connectionState, 'closed');
  assert.equal(env.clock.jobs.size, 0);
  const deletes = env.calls.filter((call) => call.method === 'DELETE');
  assert.equal(deletes.length, 1);
  assert.equal(deletes[0].url, endpoint + '/session-1');
  assert.equal(deletes[0].keepalive, true);
  for (const call of env.calls) {
    assert.equal(call.credentials, 'same-origin');
    assert.equal(call.mode, 'same-origin');
    assert.equal(call.redirect, 'error');
  }
});

test('closing while an offer is in flight deletes a late session and never installs its answer', async () => {
  const post = deferred();
  const env = readerEnvironment((call) => call.method === 'POST' ? post.promise : response());
  const reader = new env.Reader({ url: endpoint, videoOnly: true });
  await flush();
  assert.equal(env.calls.at(-1).method, 'POST');
  await reader.close();
  post.resolve(response(201, '/camera-video/worktop/whep/late-session'));
  await flush();
  assert.deepEqual(env.calls.map((call) => call.method), ['OPTIONS', 'POST', 'DELETE']);
  assert.equal(env.peers[0].answers.length, 0);
  assert.equal(env.clock.jobs.size, 0);
});

test('in-flight OPTIONS is aborted on close and cannot create a peer later', async () => {
  const options = deferred();
  const env = readerEnvironment(() => options.promise);
  const reader = new env.Reader({ url: endpoint, videoOnly: true });
  await reader.close();
  assert.equal(env.calls[0].signal.aborted, true);
  options.resolve(response()); await flush();
  assert.equal(env.peers.length, 0);
  assert.equal(env.clock.jobs.size, 0);
});

test('a lost response times out, retries once after the finite pause, then closes cleanly', async () => {
  let failures = 0, retries = 0;
  const env = readerEnvironment((call) => new Promise((resolve, reject) => {
    call.signal.addEventListener('abort', () => reject(new Error('aborted')), { once: true });
  }));
  const reader = new env.Reader({ url: endpoint, videoOnly: true,
    onError: () => failures++, onRetry: () => retries++ });
  await env.clock.advance(14999);
  assert.equal(failures, 0);
  await env.clock.advance(1);
  assert.equal(failures, 1);
  await env.clock.advance(2000);
  assert.equal(retries, 1);
  await reader.close(); await flush();
  assert.equal(env.clock.jobs.size, 0);
});

test('a peer that never connects is closed and its session deleted after the setup deadline', async () => {
  const env = readerEnvironment(null, { connected: false });
  const reader = new env.Reader({ url: endpoint, videoOnly: true });
  await flush();
  await env.clock.advance(20000);
  assert.equal(env.peers[0].connectionState, 'closed');
  assert.equal(env.calls.filter((call) => call.method === 'DELETE').length, 1);
  await reader.close();
});

test('a failed old PATCH cannot restart or close the next peer', async () => {
  const patch = deferred(); let session = 0;
  const env = readerEnvironment((call) => call.method === 'PATCH' ? patch.promise
    : response(call.method === 'POST' ? 201 : 204,
      call.method === 'POST' ? `/camera-video/worktop/whep/id-${++session}` : undefined));
  let errors = 0, retries = 0;
  const reader = new env.Reader({ url: endpoint, videoOnly: true,
    onError: () => errors++, onRetry: () => retries++ });
  await flush();
  env.peers[0].onicecandidate({ candidate: { sdpMLineIndex: 0, candidate: 'candidate:fixture' } });
  await flush();
  const firstPatch = env.calls.find((call) => call.method === 'PATCH');
  assert.equal(firstPatch.headers['If-Match'], '*');
  env.peers[0].fail(); await flush();
  await env.clock.advance(2000);
  patch.resolve(response(500)); await flush();
  assert.equal(errors, 1);
  assert.equal(retries, 1);
  assert.equal(env.peers.length, 2);
  assert.equal(env.peers[1].connectionState, 'connected');
  await reader.close();
});

for (const location of ['https://foreign.invalid/camera-video/worktop/whep/id-1',
  '/camera-video/side/whep/id-1', '/camera-video/worktop/whep/id-1?x=1',
  '/camera-video/worktop/whep/id-1#fragment', '/camera-video/worktop/whep/%41', null]) {
  test(`unsafe or absent session location cannot be followed: ${location}`, async () => {
    const env = readerEnvironment((call) => response(call.method === 'POST' ? 201 : 204,
      call.method === 'POST' ? location : undefined));
    const reader = new env.Reader({ url: endpoint, videoOnly: true });
    await flush(); await reader.close();
    assert.equal(env.calls.filter((call) => call.method === 'DELETE').length, 0);
    assert.equal(env.peers[0].answers.length, 0);
    assert.equal(env.clock.jobs.size, 0);
  });
}

test('page-exit DELETE failures are contained and abort after three seconds', async () => {
  const env = readerEnvironment((call) => call.method !== 'DELETE'
    ? response(call.method === 'POST' ? 201 : 204, '/camera-video/worktop/whep/id-1')
    : new Promise((resolve, reject) => call.signal.addEventListener('abort',
      () => reject(new Error('offline')), { once: true })));
  const reader = new env.Reader({ url: endpoint, videoOnly: true });
  await flush();
  const cleanup = reader.close();
  await env.clock.advance(3000);
  assert.equal(await cleanup, false);
  assert.equal(env.clock.jobs.size, 0);
});

function videoFixture() {
  let id = 0;
  return { srcObject: null, callbacks: new Map(), dropped: 0, pauses: 0,
    play: () => Promise.resolve(), pause() { this.pauses++; },
    requestVideoFrameCallback(fn) { const key = ++id; this.callbacks.set(key, fn); return key; },
    cancelVideoFrameCallback(key) { this.callbacks.delete(key); },
    getVideoPlaybackQuality() { return { droppedVideoFrames: this.dropped }; },
    present(time, count) {
      const callbacks = [...this.callbacks.values()]; this.callbacks.clear();
      callbacks.forEach((fn) => fn(time, { presentationTime: time - 2, presentedFrames: count }));
    },
  };
}

function playerEnvironment() {
  const clock = new Clock(), video = videoFixture(), readers = [], changes = [];
  let visible = true;
  class Reader {
    closed = false;
    constructor(config) {
      assert.ok(readers.every((item) => item.closed), 'Only one reader may own a peer');
      this.config = config; readers.push(this);
    }
    close() { this.closed = true; return Promise.resolve(true); }
    track() {
      const track = { kind: 'video', stopped: false, stop() { this.stopped = true; } };
      this.config.onTrack({ track, streams: [{ getTracks: () => [track] }] });
      return track;
    }
  }
  const player = new CameraPlayer(video, { Reader, origin, timers: clock, now: () => clock.time,
    isVisible: () => visible, onChange: (value) => changes.push(value) });
  return { player, video, readers, clock, changes, setVisible: (value) => { visible = value; } };
}

test('camera changes close the prior reader, stop its track, and ignore late events', async () => {
  const env = playerEnvironment();
  env.player.select('worktop');
  const track = env.readers[0].track(); await flush();
  env.player.select('side');
  assert.equal(track.stopped, true);
  assert.equal(env.readers[0].closed, true);
  assert.equal(env.video.callbacks.size, 0);
  env.readers[0].track(); env.readers[0].config.onError(); env.readers[0].config.onRetry();
  assert.equal(env.player.phase, 'connecting');
  assert.equal(env.player.retries, 0);
  assert.equal(env.video.srcObject, null);
  assert.equal(env.readers[1].config.url, origin + '/camera-video/side/whep');
  env.player.select('side');
  assert.equal(env.readers.length, 2);
  assert.throws(() => env.player.select('unknown'), /Unknown camera/);
  env.player.close();
  assert.equal(env.clock.jobs.size, 0);
});

test('quality uses presented frame counters, real presentation age, and browser drops', async () => {
  const env = playerEnvironment();
  env.player.select('worktop'); env.readers[0].track(); await flush();
  assert.equal(env.player.snapshot().fps, null);
  assert.equal(env.player.snapshot().ageMs, null);
  env.clock.time = 100; env.video.present(100, 1);
  env.clock.time = 1100; env.video.present(1100, 25);
  env.video.dropped = 3; env.player.tick();
  assert.deepEqual(env.player.snapshot(), { camera: 'worktop', phase: 'live',
    fps: 24, ageMs: 2, retries: 0, droppedFrames: 3 });
  env.clock.time = 2100; env.player.tick();
  assert.equal(env.player.snapshot().fps, 0);
  assert.equal(env.player.snapshot().ageMs, 1002);
  env.player.close();
});

test('frame stalls become visible and restart one peer after twelve seconds', async () => {
  const env = playerEnvironment();
  env.player.select('worktop'); env.readers[0].track(); await flush();
  env.video.present(0, 1);
  await env.clock.advance(4000);
  assert.equal(env.player.phase, 'stalled');
  await env.clock.advance(9000);
  assert.equal(env.readers.length, 2);
  assert.equal(env.player.retries, 1);
  assert.equal(env.readers[0].closed, true);
  assert.equal(env.player.snapshot().ageMs, null);
  env.player.close();
});

test('hidden-tab frame callback throttling does not trigger stream restarts', async () => {
  const env = playerEnvironment();
  env.player.select('worktop'); env.readers[0].track(); await flush();
  env.video.present(0, 1); env.setVisible(false);
  await env.clock.advance(14000);
  assert.equal(env.readers.length, 1);
  env.player.close();
});

test('a connected stream with no first frame gets a finite fresh attempt', async () => {
  const env = playerEnvironment();
  env.player.select('worktop'); env.readers[0].track(); await flush();
  await env.clock.advance(25000);
  assert.equal(env.readers.length, 1);
  await env.clock.advance(1000);
  assert.equal(env.readers.length, 2);
  assert.equal(env.player.retries, 1);
  assert.equal(env.readers[0].closed, true);
  env.player.close();
});

test('autoplay refusal waits for the user instead of cycling camera sessions', async () => {
  const env = playerEnvironment();
  env.video.play = () => Promise.reject(new Error('autoplay refused'));
  env.player.select('worktop'); env.readers[0].track(); await flush();
  assert.equal(env.player.phase, 'blocked');
  await env.clock.advance(30000);
  assert.equal(env.readers.length, 1);
  env.video.play = () => Promise.resolve();
  env.player.play(); await flush();
  assert.equal(env.player.phase, 'connecting');
  await env.clock.advance(1000);
  assert.equal(env.readers.length, 1);
  env.player.close();
});

test('an old frame callback cannot overwrite the refreshed stream after a reader retry', async () => {
  const env = playerEnvironment();
  env.player.select('worktop'); env.readers[0].track(); await flush();
  const callback = [...env.video.callbacks.values()][0];
  env.readers[0].config.onError(); env.readers[0].config.onRetry();
  env.readers[0].track(); await flush();
  callback(1000, { presentationTime: 1000, presentedFrames: 100 });
  assert.equal(env.player.snapshot().ageMs, null);
  assert.equal(env.video.callbacks.size, 1);
  assert.equal(env.player.retries, 1);
  env.player.close();
});

test('an old autoplay rejection cannot replace the next camera state', async () => {
  const env = playerEnvironment(), play = deferred();
  env.video.play = () => play.promise;
  env.player.select('worktop'); env.readers[0].track();
  env.player.select('kitchen');
  play.reject(new Error('autoplay refused')); await flush();
  assert.equal(env.player.phase, 'connecting');
  env.player.close();
});

test('missing frame-timing APIs leave FPS and age unavailable rather than claiming a configured rate', async () => {
  const env = playerEnvironment();
  delete env.video.requestVideoFrameCallback; delete env.video.getVideoPlaybackQuality;
  env.player.select('worktop'); env.readers[0].track(); await flush();
  assert.equal(env.player.phase, 'playing');
  assert.equal(env.player.snapshot().fps, null);
  assert.equal(env.player.snapshot().ageMs, null);
  assert.equal(env.player.snapshot().droppedFrames, null);
  env.player.close();
});

function pageEnvironment() {
  const elements = new Map(), handlers = new Map(), readers = [];
  for (const key of ['#camera', '#video-message', '#connection', '#quality', '#reconnect',
    'kitchen', 'worktop', 'side']) {
    elements.set(key, { ...(key === '#camera' ? videoFixture() : {}), dataset: { camera: key },
      handlers: new Map(), setAttribute() {}, addEventListener(type, fn) { this.handlers.set(type, fn); } });
  }
  const document = { visibilityState: 'visible', querySelector: (key) => elements.get(key),
    querySelectorAll: () => ['kitchen', 'worktop', 'side'].map((key) => elements.get(key)) };
  const clock = new Clock();
  const window = { RTCPeerConnection: class {}, MediaMTXWebRTCReader: class {
    constructor(config) { this.config = config; this.closed = false; readers.push(this); }
    close() { this.closed = true; }
  }, location: { origin, href: origin + '/cameras/?camera=side' }, history: { replaceState() {} },
  performance: { now: () => clock.time }, setInterval: clock.setInterval, clearInterval: clock.clearInterval,
  addEventListener: (type, fn) => handlers.set(type, fn) };
  return { document, window, elements, handlers, readers, clock };
}

test('pagehide releases the reader and presentation callback; restored pages start the selected camera', () => {
  const env = pageEnvironment();
  const player = mountCameraViewer(env.document, env.window);
  assert.equal(player.camera, 'side');
  env.handlers.get('pagehide')();
  assert.equal(env.readers[0].closed, true);
  assert.equal(env.clock.jobs.size, 0);
  env.handlers.get('pageshow')({ persisted: true });
  assert.equal(env.readers.length, 2);
  assert.equal(player.camera, 'side');
  player.close();
});

test('unsupported browser offers image fallback without creating a reader', () => {
  const env = pageEnvironment(); delete env.window.RTCPeerConnection;
  assert.equal(mountCameraViewer(env.document, env.window), null);
  assert.equal(env.readers.length, 0);
  assert.equal(env.elements.get('#reconnect').disabled, true);
  assert.match(env.elements.get('#video-message').textContent, /image view/);
});
