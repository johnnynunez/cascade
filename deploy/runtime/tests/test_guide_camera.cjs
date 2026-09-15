const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

async function guide() {
  const elements = new Map(), images = [], timers = new Map(), intervals = new Map();
  const documentEvents = {}, windowEvents = {};
  let timerId = 0;
  class Element {
    constructor(id = '') {
      this.id = id; this.attributes = {}; this.dataset = {}; this.children = [];
      this.events = {}; this.hidden = false; this.naturalWidth = 0;
    }
    setAttribute(name, value) { this.attributes[name] = value; }
    getAttribute(name) { return this.attributes[name]; }
    removeAttribute(name) { delete this.attributes[name]; }
    addEventListener(name, callback) { this.events[name] = callback; }
    replaceChildren(...children) { this.children = children; }
    replaceWith(element) { elements.set(this.id, element); }
    set src(value) { this.setAttribute('src', value); }
    get src() { return this.getAttribute('src'); }
  }
  const document = {
    hidden: false,
    getElementById(id) {
      if (!elements.has(id)) elements.set(id, new Element(id));
      return elements.get(id);
    },
    createElement() { return new Element(); },
    querySelectorAll() { return []; },
    addEventListener(name, callback) { documentEvents[name] = callback; },
  };
  const base = 'http://camera.example/cameras';
  const context = {
    document, location: new URL('http://camera.example/guide/'), URL, Date, AbortController,
    localStorage: {getItem() { return null; }, setItem() {}},
    Image: class extends Element { constructor() { super(); images.push(this); } },
    window: {addEventListener(name, callback) { windowEvents[name] = callback; }},
    setTimeout(callback, delay) { timers.set(++timerId, {callback, delay}); return timerId; },
    clearTimeout(id) { timers.delete(id); },
    setInterval(callback, delay) { intervals.set(++timerId, {callback, delay}); return timerId; },
    clearInterval(id) { intervals.delete(id); },
    fetch: async () => ({ok: true, text: async () => JSON.stringify({
      version: 1, generated_at: new Date().toISOString(), interactive_ready: true,
      camera_discovery: {version: 1, base_url: base, state_url: base + '/state',
        default_camera: 'worktop', live: true,
        cameras: ['worktop', 'kitchen', 'side'].map(name => ({name, label: name,
          stream_url: base + '/stream/' + name}))},
    })}),
  };
  vm.runInNewContext(fs.readFileSync(path.join(__dirname, '../../../web/guide/app.js'), 'utf8'), context);
  await new Promise(setImmediate);
  return {
    elements, images, document, documentEvents, windowEvents, timers,
    complete(frame = images.at(-1)) { frame.naturalWidth = 960; frame.onload(); },
    timeout(delay) {
      const entry = [...timers].find(([, value]) => value.delay === delay);
      assert.ok(entry, `Expected a ${delay} ms deadline`);
      timers.delete(entry[0]); entry[1].callback();
    },
    async poll() {
      for (const {callback} of intervals.values()) callback();
      await new Promise(setImmediate);
    },
  };
}

test('healthy backend status cannot keep a stalled guide image live', async () => {
  const h = await guide();
  assert.match(h.images[0].src, /\/snapshot\/worktop\.jpg\?t=/);
  assert.equal(h.elements.get('scene-state').textContent, 'Connecting');
  h.complete();
  assert.equal(h.elements.get('scene-image'), h.images[0]);
  assert.equal(h.elements.get('scene-state').textContent, 'Live');
  h.timeout(350);
  const stale = h.images[1], lateLoad = stale.onload;
  await h.poll();
  assert.equal(h.images.length, 2, 'Status polling must not duplicate image requests');
  h.timeout(6000);
  assert.equal(h.elements.get('scene-state').dataset.live, 'false');
  assert.equal(h.elements.get('scene-image').hidden, true);
  stale.naturalWidth = 960; lateLoad();
  assert.equal(h.elements.get('scene-state').dataset.live, 'false');
  await h.poll(); h.complete();
  assert.equal(h.elements.get('scene-state').textContent, 'Live');
  assert.equal(h.elements.get('scene-image').hidden, false);
});

test('camera changes and hidden pages cancel pending image generations', async () => {
  const h = await guide(), stale = h.images[0], lateLoad = stale.onload;
  h.elements.get('view-buttons').children.find(button => button.dataset.camera === 'side').events.click();
  assert.match(h.images[1].src, /\/snapshot\/side\.jpg\?t=/);
  stale.naturalWidth = 960; lateLoad();
  assert.equal(h.elements.get('scene-state').textContent, 'Connecting');
  h.complete();
  assert.equal(h.elements.get('scene-name').textContent, 'side');
  h.document.hidden = true; h.documentEvents.visibilitychange();
  assert.equal(h.elements.get('scene-image').hidden, true);
  assert.equal(h.timers.size, 0);
  h.document.hidden = false; h.documentEvents.visibilitychange();
  await new Promise(setImmediate);
  h.complete();
  assert.equal(h.elements.get('scene-state').textContent, 'Live');
  h.windowEvents.pagehide();
  assert.equal(h.timers.size, 0);
});
