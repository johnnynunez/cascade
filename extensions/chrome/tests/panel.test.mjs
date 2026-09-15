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
    });
    return elements.get(id);
  };
  Object.assign(globalThis, {
    document: {getElementById: element, body: {classList: {add() {}}}, addEventListener() {}},
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
