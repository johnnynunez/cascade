import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';

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
// whose parts the test writes with the server's own header layout. The panel's
// clock (performance.now) stands still until the test advances it, so a part
// is rendered at an exact age; its timers (the 1 s refresh tick, the /state
// poll) run in real time. Every element counts the writes to its text.
async function sparkPanel(key) {
  const elements = new Map(), listeners = {}, clock = {now: 1000};
  const names = ['document', 'window', 'location', 'chrome', 'fetch', 'Image'];
  const original = Object.fromEntries(names.map(name => [name, Object.getOwnPropertyDescriptor(globalThis, name)]));
  const realNow = Object.getOwnPropertyDescriptor(performance, 'now');
  const element = id => {
    if (!elements.has(id)) elements.set(id, {
      text: '', writes: 0,
      get textContent() { return this.text; },
      set textContent(value) { this.text = String(value); this.writes++; },
      dataset: {}, events: {}, attributes: {}, hidden: false, options: [],
      addEventListener(event, fn) { this.events[event] = fn; },
      setAttribute(name, value) { this.attributes[name] = value; },
      removeAttribute(name) { delete this.attributes[name]; },
      replaceChildren(...children) { this.options = children; },
      querySelector() { return null; }, focus() {},
    });
    return elements.get(id);
  };
  let frameId = 0, stream = null, stateReads = 0;
  const fake = {error: null, learning: undefined};
  const extra = () => (fake.learning === undefined ? {} : {learning: fake.learning});
  const cameras = () => Object.fromEntries(['kitchen', 'worktop', 'side'].map(name =>
    [name, {frame_id: ++frameId, online: true, age_s: 0.1, error: fake.error, fps: 2}]));
  const json = body => new Response(JSON.stringify(body), {headers: {'content-type': 'application/json'}});
  const restore = () => {
    for (const [name, descriptor] of Object.entries(original)) {
      if (descriptor) Object.defineProperty(globalThis, name, descriptor);
      else delete globalThis[name];
    }
    if (realNow) Object.defineProperty(performance, 'now', realNow);
    else delete performance.now;
  };
  performance.now = () => clock.now;
  Object.assign(globalThis, {
    document: {getElementById: element, querySelectorAll: () => [], createElement: tag => ({tag}),
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
      if (path === '/state') { stateReads++; return json({cameras: cameras(), ...extra()}); }
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
    // No part arrives for `ms` more milliseconds of the panel's clock.
    advance(ms) { clock.now += ms; },
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

test('a Spark frame whose stream stops keeps ageing on screen, then the 5 s rule takes over', async () => {
  const panel = await sparkPanel('frame-age-ticks');
  const text = id => panel.element(id).textContent, state = () => panel.element('status').dataset.state;
  try {
    await panel.frame(0);
    await panel.until(() => text('status') === 'Live', 'the first fresh frame goes live');
    panel.advance(1500);  // no further part: the frame on screen is 1.5 s old
    await panel.until(() => text('status') === 'Last frame · 1 s old', 'the refresh tick counts the time since the part');
    assert.equal(state(), 'held');
    assert.equal(text('frame-note'), 'Last frame · 1 s old');
    assert.equal(panel.element('retry').hidden, true, 'a held frame is no outage: no Reconnect');
    panel.advance(1000);
    await panel.until(() => text('status') === 'Last frame · 2 s old', 'the count goes on between parts');
    panel.advance(2600);  // 5.1 s without a part
    await panel.until(() => text('status') === 'Not updating', 'the 5 s stale rule still wins');
    assert.equal(state(), 'stale');
    await panel.frame(0);
    await panel.until(() => text('status') === 'Live', 'a fresh part brings Live back');
  } finally {
    panel.close();
  }
});

test('a live but slowly changing Spark image stays Live until its next part, and a stall still counts', async () => {
  const panel = await sparkPanel('slow-image');
  const text = id => panel.element(id).textContent, said = panel.element('status-live');
  try {
    await panel.frame(0);
    await panel.until(() => text('status') === 'Live', 'the first fresh frame goes live');
    await panel.frame(0.9);  // the image last changed 0.9 s ago; the server writes a part every 0.5 s
    assert.equal(text('status'), 'Live');
    const writes = said.writes;
    panel.advance(600);  // the next part is due, but the refresh tick comes first
    await new Promise(resolve => setTimeout(resolve, 1100));  // at least one refresh tick
    assert.equal(text('status'), 'Live', 'no Last frame flash just before the next part');
    assert.equal(said.writes, writes, 'nothing announced');
    await panel.frame(0.9);
    panel.advance(1200);  // no part for 1.2 s: the stream stalled, so the time on screen counts too
    await panel.until(() => text('status') === 'Last frame · 2 s old', 'the stall counts from the part age');
    assert.equal(said.textContent, 'Last frame');
  } finally {
    panel.close();
  }
});

test('a server that sends no X-Frame-Age stays Live between parts until the 5 s rule', async () => {
  const panel = await sparkPanel('no-age-ticks');
  const text = id => panel.element(id).textContent;
  try {
    await panel.frame();
    await panel.until(() => text('status') === 'Live', 'the first frame goes live');
    panel.advance(3000);  // under the 4 s /state rule: no poll is needed to stay Live
    await new Promise(resolve => setTimeout(resolve, 1100));  // at least one refresh tick
    assert.equal(text('status'), 'Live');
    panel.advance(2100);
    await panel.until(() => text('status') === 'Not updating', 'the 5 s stale rule');
  } finally {
    panel.close();
  }
});

test('screen readers hear the state changes, never the held count ticking', async () => {
  const panel = await sparkPanel('announce');
  const text = id => panel.element(id).textContent, said = panel.element('status-live');
  const detail = panel.element('detail');
  try {
    await panel.frame(0);
    await panel.until(() => text('status') === 'Live', 'the first fresh frame goes live');
    assert.equal(said.textContent, 'Live');
    const writes = said.writes;
    panel.advance(1500);
    await panel.until(() => text('status') === 'Last frame · 1 s old', 'held');
    assert.equal(said.textContent, 'Last frame');
    assert.equal(said.writes, writes + 1);
    const details = detail.writes;
    panel.advance(1000);
    await panel.until(() => text('status') === 'Last frame · 2 s old', 'two seconds');
    panel.advance(1000);
    await panel.until(() => text('status') === 'Last frame · 3 s old', 'three seconds');
    assert.equal(said.writes, writes + 1, 'the count never reaches the live region');
    assert.equal(detail.writes, details, 'nor is the unchanged detail rewritten');
    panel.advance(2100);
    await panel.until(() => text('status') === 'Not updating', 'stale');
    assert.equal(said.textContent, 'Not updating');
    assert.equal(said.writes, writes + 2);
  } finally {
    panel.close();
  }
});

test('the camera badge is no live region: a hidden status region announces the state', async () => {
  const html = await readFile(new URL('../panel.html', import.meta.url), 'utf8');
  const tag = id => html.match(new RegExp(`<[a-z]+ id="${id}"[^>]*>`))?.[0] ?? '';
  assert.match(tag('status'), /class="badge"/);
  assert.doesNotMatch(tag('status'), /aria-live|role=/);
  for (const attribute of ['class="sr-only"', 'role="status"', 'aria-live="polite"']) {
    assert.ok(tag('status-live').includes(attribute), `${attribute} in ${tag('status-live') || 'no #status-live'}`);
  }
});

test('the learning panel shows /state learning as text nodes, and stays hidden without it', async () => {
  const panel = await sparkPanel('learning');
  const section = panel.element('learning');
  const shown = () => section.options.map(node => [node.tag, node.textContent]);
  const tag = 'learned in sim | verifier 4/5 | lib 3fa9c2d1';
  try {
    await panel.until(() => panel.stateReads() > 0 && section.hidden === true, 'no learning key: hidden');
    assert.deepEqual(section.options, []);
    panel.fake.learning = {schema: 1, library_sha: '3fa9c2d1' + '0'.repeat(56),
      served: [{id: 'red-cube', title: '<img src=x onerror=alert(1)>', tag}],
      last_event: {kind: 'note', line: 'From earlier runs: <b>Cube</b> (x).', outcome: 'placed'}};
    await panel.until(() => section.hidden === false, 'a schema-1 learning key: shown');
    assert.deepEqual(shown(), [['p', 'Learned notes · lib 3fa9c2d1'], ['p', `<img src=x onerror=alert(1)> — ${tag}`],
      ['p', 'From earlier runs: <b>Cube</b> (x).']]);
    assert.ok(section.options.every(node => !('innerHTML' in node)), 'text only');
    const nodes = section.options, reads = panel.stateReads();
    await panel.until(() => panel.stateReads() > reads + 1, 'the panel polls /state again');
    assert.equal(section.options, nodes, 'an unchanged state is not rewritten');
    panel.fake.learning = {...panel.fake.learning, schema: 2};
    await panel.until(() => section.hidden === true, 'another schema: hidden again');
    assert.deepEqual(section.options, []);
  } finally {
    panel.close();
  }
});
