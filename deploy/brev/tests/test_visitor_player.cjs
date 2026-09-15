const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

function fixture() {
  let now = 100000;
  let nextID = 0;
  let failures = 0;
  const timers = new Map();
  const callbacks = new Map();
  const requests = [];
  const revoked = [];
  const buffers = [];
  const image = {hidden: false};
  const video = {
    hidden: true, currentTime: 0, paused: true,
    pause() { this.paused = true; }, load() {}, removeAttribute() {},
    async play() { this.paused = false; },
    requestVideoFrameCallback(fn) { const id = ++nextID; callbacks.set(id, fn); return id; },
    cancelVideoFrameCallback(id) { callbacks.delete(id); },
  };
  class Buffer extends EventTarget {
    appends = 0;
    buffered = {length: 1, start: () => 0, end: () => 1};
    appendBuffer() { this.appends += 1; queueMicrotask(() => this.dispatchEvent(new Event('updateend'))); }
    remove() { this.appendBuffer(); }
  }
  class MediaSource extends EventTarget {
    static isTypeSupported() { return true; }
    addSourceBuffer() { const buffer = new Buffer(); buffers.push(buffer); return buffer; }
  }
  const window = {MediaSource};
  vm.runInNewContext(fs.readFileSync(path.join(__dirname, '../visitor-player.js'), 'utf8'), {
    window, MediaSource, AbortController,
    Date: {now: () => now},
    setTimeout(fn, ms) { const id = ++nextID; timers.set(id, {fn, at: now + ms}); return id; },
    clearTimeout(id) { timers.delete(id); },
    URL: {
      createObjectURL(media) {
        queueMicrotask(() => media.dispatchEvent(new Event('sourceopen')));
        return `blob:${++nextID}`;
      },
      revokeObjectURL(url) { revoked.push(url); },
    },
    async fetch(url, {signal}) {
      let stream;
      const body = new ReadableStream({start(controller) { stream = controller; }});
      signal.addEventListener('abort', () => stream.error(new Error('Aborted')), {once: true});
      requests.push({url, signal, push: () => stream.enqueue(new Uint8Array([1, 2, 3]))});
      return {ok: true, body};
    },
  });
  const player = new window.VisitorVideo(video, image, () => { failures += 1; });
  const flush = async () => { for (let i = 0; i < 12; i++) await Promise.resolve(); };
  return {
    player, video, image, requests, revoked, callbacks, timers, buffers,
    get failures() { return failures; },
    async start(camera = 'worktop') { player.start(camera); await flush(); },
    async bytes() { requests.at(-1).push(); await flush(); },
    async frame() {
      const entries = [...callbacks];
      callbacks.clear();
      for (const [, fn] of entries) fn();
      await flush();
    },
    async advance(ms) {
      const end = now + ms;
      while (true) {
        const next = [...timers].filter(([, timer]) => timer.at <= end)
          .sort((a, b) => a[1].at - b[1].at)[0];
        if (!next) break;
        now = next[1].at;
        timers.delete(next[0]);
        next[1].fn();
        await flush();
      }
      now = end;
      await flush();
    },
  };
}

test('continuous bytes without a presented frame fall back after startup', async () => {
  const f = fixture();
  await f.start();
  for (let second = 0; second < 14; second++) {
    await f.bytes();
    await f.advance(1000);
  }
  assert.equal(f.failures, 0);
  await f.bytes();
  await f.advance(1000);
  assert.equal(f.requests[0].signal.aborted, true);
  assert.equal(f.player.camera, undefined);
  assert.equal(f.image.hidden, false);
  assert.equal(f.failures, 1);
  assert.equal(f.callbacks.size, 0);
  assert.equal(f.timers.size, 0);
});

test('pending playback does not block the next video fragment', async () => {
  const f = fixture();
  let resolvePlay;
  f.video.play = () => {
    f.video.paused = false;
    return new Promise(resolve => { resolvePlay = resolve; });
  };
  await f.start();
  await f.bytes();
  assert.equal(typeof resolvePlay, 'function');
  await f.bytes();
  assert.equal(f.buffers[0].appends, 2);
  resolvePlay();
  await f.frame();
  assert.equal(f.player.playing, true);
  assert.equal(f.failures, 0);
  f.player.stop();
});

test('late playback rejection cannot stop a newer camera', async () => {
  const f = fixture();
  let rejectPlay;
  f.video.play = () => {
    f.video.paused = false;
    return new Promise((resolve, reject) => { rejectPlay = reject; });
  };
  await f.start('worktop');
  await f.bytes();
  const rejectPrevious = rejectPlay;
  await f.start('side');
  await f.bytes();
  rejectPrevious(new Error('Old playback closed'));
  await f.frame();
  assert.equal(f.player.camera, 'side');
  assert.equal(f.player.playing, true);
  assert.equal(f.failures, 0);
  rejectPlay(new Error('Current playback failed'));
  await f.advance(1);
  assert.equal(f.failures, 1);
  assert.equal(f.image.hidden, false);
  assert.equal(f.requests.at(-1).signal.aborted, true);
});

test('a frozen rendered video cannot hide fresh snapshots while bytes arrive', async () => {
  const f = fixture();
  await f.start();
  await f.bytes();
  await f.frame();
  assert.equal(f.image.hidden, true);
  for (let second = 0; second < 5; second++) {
    await f.bytes();
    await f.advance(1000);
  }
  assert.equal(f.video.hidden, true);
  assert.equal(f.image.hidden, false);
  assert.equal(f.failures, 1);
  assert.equal(f.revoked.length, 1);
});

test('advancing rendered frames keep a healthy video active', async () => {
  const f = fixture();
  await f.start();
  for (let second = 0; second < 20; second++) {
    await f.bytes();
    await f.frame();
    await f.advance(1000);
  }
  assert.equal(f.failures, 0);
  assert.equal(f.player.playing, true);
  assert.equal(f.requests[0].signal.aborted, false);
  f.player.stop();
  await f.advance(20000);
  assert.equal(f.failures, 0);
  assert.equal(f.callbacks.size, 0);
  assert.equal(f.timers.size, 0);
});

test('switching cameras cancels the previous presentation watchdog and callback', async () => {
  const f = fixture();
  await f.start('worktop');
  await f.bytes();
  const staleCallback = [...f.callbacks.values()][0];
  await f.frame();
  await f.advance(4000);
  await f.start('side');
  assert.equal(f.requests[0].signal.aborted, true);
  staleCallback();
  assert.equal(f.image.hidden, false);
  await f.advance(2000);
  assert.equal(f.player.camera, 'side');
  assert.equal(f.failures, 0);
  await f.bytes();
  await f.frame();
  assert.equal(f.video.hidden, false);
  f.player.stop();
  await f.advance(20000);
  assert.equal(f.failures, 0);
  assert.equal(f.timers.size, 0);
});
