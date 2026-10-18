import test from 'node:test';
import assert from 'node:assert/strict';

test('pause, resume and reconnect keep the camera action labels in sync', async () => {
  const elements = new Map();
  const original = Object.fromEntries(['document', 'window', 'location', 'chrome']
    .map(key => [key, Object.getOwnPropertyDescriptor(globalThis, key)]));
  const element = id => {
    if (!elements.has(id)) elements.set(id, {
      textContent: '', dataset: {}, events: {}, attributes: {},
      get title() { return this.attributes.title; },
      set title(value) { this.attributes.title = value; },
      addEventListener(event, fn) { this.events[event] = fn; },
      setAttribute(name, value) { this.attributes[name] = value; },
      querySelector() { return null; },
    });
    return elements.get(id);
  };
  Object.assign(globalThis, {
    document: {getElementById: element, querySelectorAll: () => [],
      querySelector: () => ({classList: {toggle() {}}}),
      body: {classList: {add() {}}}, addEventListener() {}},
    window: {addEventListener() {}},
    location: {search: ''},
    chrome: {
      runtime: {sendMessage: async () => ({ok: false})},
      storage: {onChanged: {addListener() {}}},
      permissions: {onRemoved: {addListener() {}}, onAdded: {addListener() {}}},
    },
  });
  const pause = element('pause');
  pause.setAttribute('title', 'Pause video');
  pause.setAttribute('aria-label', 'Pause video');
  const check = label => {
    assert.equal(pause.attributes['aria-label'], label);
    assert.equal(pause.attributes.title, label);
  };
  try {
    await import('../panel.mjs?test=action-labels');
    pause.events.click();
    check('Resume video');
    pause.events.click();
    check('Pause video');
    pause.events.click();
    check('Resume video');
    element('retry').events.click();
    check('Pause video');
  } finally {
    for (const [key, descriptor] of Object.entries(original)) {
      if (descriptor) Object.defineProperty(globalThis, key, descriptor);
      else delete globalThis[key];
    }
  }
});

// A fake Spark camera viewer (scripts/spark_cameras.py): /api/status with its
// discovery contract, /state with an advancing frame_id, and an MJPEG stream
// whose parts the test writes with the server's own header layout.
async function sparkPanel(key) {
  const elements = new Map(), listeners = {};
  const names = ['document', 'window', 'location', 'chrome', 'fetch', 'Image'];
  const original = Object.fromEntries(names.map(name => [name, Object.getOwnPropertyDescriptor(globalThis, name)]));
  const element = id => {
    if (!elements.has(id)) elements.set(id, {
      textContent: '', dataset: {}, events: {}, attributes: {}, hidden: false, options: [],
      addEventListener(event, fn) { this.events[event] = fn; },
      setAttribute(name, value) { this.attributes[name] = value; },
      removeAttribute(name) { delete this.attributes[name]; },
      replaceChildren(...children) { this.options = children; },
      querySelector() { return null; }, focus() {},
    });
    return elements.get(id);
  };
  let frameId = 0, stream = null, stateReads = 0;
  const fake = {error: null};
  const cameras = () => Object.fromEntries(['kitchen', 'worktop', 'side'].map(name =>
    [name, {frame_id: ++frameId, online: true, age_s: 0.1, error: fake.error, fps: 2}]));
  const json = body => new Response(JSON.stringify(body), {headers: {'content-type': 'application/json'}});
  const restore = () => {
    for (const [name, descriptor] of Object.entries(original)) {
      if (descriptor) Object.defineProperty(globalThis, name, descriptor);
      else delete globalThis[name];
    }
  };
  Object.assign(globalThis, {
    document: {getElementById: element, querySelectorAll: () => [], createElement: () => ({}),
      querySelector: () => ({classList: {toggle() {}}}), body: {classList: {add() {}}},
      addEventListener() {}, hidden: false},
    window: {addEventListener(name, fn) { listeners[name] = fn; }},
    location: {search: ''},
    Image: class { decode() { return Promise.resolve(); } },
    chrome: {
      runtime: {sendMessage: async () => ({ok: true})},
      storage: {local: {get: async () => ({discoveryHint: 'http://127.0.0.1:8091/api/status'}), set: async () => {}},
        onChanged: {addListener() {}}},
      permissions: {contains: async () => true, onRemoved: {addListener() {}}, onAdded: {addListener() {}}},
    },
    fetch: async (url, {signal} = {}) => {
      const path = new URL(url).pathname;
      if (path === '/api/status') return json({cameras: cameras(), camera_discovery: {version: 1, base_url: '/',
        state_url: '/state', default_camera: 'worktop',
        cameras: ['kitchen', 'worktop', 'side'].map(name => ({name, stream_url: '/stream/' + name}))}});
      if (path === '/state') { stateReads++; return json({cameras: cameras()}); }
      if (path === '/stream/worktop') {
        const body = new ReadableStream({start(controller) { stream = controller; }});
        signal?.addEventListener('abort', () => stream.error(new DOMException('Aborted', 'AbortError')));
        return new Response(body, {headers: {'content-type': 'multipart/x-mixed-replace; boundary=frame'}});
      }
      throw new Error('unexpected fetch ' + url);
    },
  });
  try {
    await import(`../panel.mjs?test=${key}`);
  } catch (error) { restore(); throw error; }
  const until = async (ok, what) => {
    for (const deadline = Date.now() + 3000; !ok() && Date.now() < deadline;) await new Promise(r => setTimeout(r, 2));
    assert.ok(ok(), `${what}: status "${element('status').textContent}", detail "${element('detail').textContent}"`);
  };
  await until(() => stream !== null, 'the stream opens');
  return {
    element, until, fake, stateReads: () => stateReads,
    // b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: N" + f"\r\nX-Frame-Age: {age:.2f}" + b"\r\n\r\n" + jpeg + b"\r\n"
    // Writes one part and returns once the panel has rendered it (a new image URL).
    async frame(age) {
      const header = '--frame\r\nContent-Type: image/jpeg\r\nContent-Length: 6'
        + (age === undefined ? '' : `\r\nX-Frame-Age: ${age.toFixed(2)}`) + '\r\n\r\n';
      const shown = element('camera-image').src;
      stream.enqueue(new Uint8Array([...new TextEncoder().encode(header), 255, 216, 1, 2, 255, 217, 13, 10]));
      await until(() => element('camera-image').src !== shown, `the frame (age ${age}) is rendered`);
    },
    close() { listeners.pagehide?.(); restore(); },
  };
}

test('a held Spark frame shows its age instead of Live, and Live returns with a fresh frame', async () => {
  const panel = await sparkPanel('frame-age');
  const text = id => panel.element(id).textContent, state = () => panel.element('status').dataset.state;
  const live = () => {
    assert.equal(text('status'), 'Live');
    assert.equal(state(), 'live');
    assert.equal(text('detail'), 'The image is updating live.');
    assert.equal(panel.element('frame-note').hidden, true);
  };
  const held = label => {
    assert.equal(text('status'), label);
    assert.equal(state(), 'held');
    assert.equal(text('detail'), 'No new image from the camera yet. This is the last one it sent.');
    assert.equal(panel.element('frame-note').hidden, false);
    assert.equal(text('frame-note'), label);
    assert.equal(panel.element('retry').hidden, true, 'a held frame is no outage: no Reconnect');
  };
  try {
    await panel.frame(0);
    await panel.until(() => text('status') === 'Live', 'the first fresh frame goes live');
    live();
    await panel.frame(0.8);
    live();  // under 1 s old
    await panel.frame(1.0);
    live();  // the label starts once the age passes 1 s
    await panel.frame(2.4);
    held('Last frame · 2 s old');
    await panel.frame(3.9);
    held('Last frame · 3 s old');
    await panel.frame(0.1);
    live();  // a fresh frame brings Live back
    await panel.frame(2.4);
    held('Last frame · 2 s old');
    await panel.frame();
    live();  // a part without X-Frame-Age (any other server) is live as before
    // the panel's own stale rules still win over the age: a camera error in /state
    panel.fake.error = 'Camera stopped advancing';
    const reads = panel.stateReads();
    await panel.until(() => panel.stateReads() > reads + 1, 'the panel polls /state');
    await panel.frame(2.4);
    assert.equal(text('status'), 'Not updating');
    assert.equal(state(), 'stale');
  } finally {
    panel.close();
  }
});
