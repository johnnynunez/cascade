import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {frameAgeNote, frameAgeNow} from '../core.mjs';

// Part arrivals captured from the real Spark camera server (see "about" in the file).
const captured = JSON.parse(await readFile(new URL('./spark-part-timings.json', import.meta.url), 'utf8'));

// The badge text at every refresh: one right after each part renders, and the panel's
// 1 s refresh tick, here at tick phase `phaseMs`. `rule(age, shownMs)` -> label ('' = Live).
function badge(parts, phaseMs, rule = (age, shownMs) => frameAgeNote(frameAgeNow(age, shownMs))) {
  const events = parts.map(([t, age]) => ({t, age}));
  for (let t = phaseMs; t <= parts.at(-1)[0] + 1000; t += 1000) events.push({t, tick: true});
  events.sort((a, b) => a.t - b.t || (a.tick ? 1 : -1));
  const out = [];
  let shownAt = null, age = null;
  for (const event of events) {
    if (!event.tick) [shownAt, age] = [event.t, event.age];
    if (shownAt !== null) out.push({t: event.t, age, shownMs: event.t - shownAt, label: rule(age, event.t - shownAt)});
  }
  return out;
}
// Live -> "Last frame" -> Live episodes shorter than 300 ms.
const flashes = labels => {
  let count = 0, since = null;
  for (const {t, label} of labels) {
    if (label && since === null) since = t;
    if (!label && since !== null) { count += t - since < 300; since = null; }
  }
  return count;
};
const phases = Array.from({length: 100}, (_, k) => k * 10);

test('a live but slowly changing Spark image never flashes Last frame, at any refresh-tick phase', () => {
  for (const name of ['newton', 'slow']) {
    const parts = captured[name];
    assert.ok(parts.every(([t], i) => !i || t - parts[i - 1][0] < 1000), `${name}: a part every < 1 s`);
    let shownFlashes = 0;
    for (const phase of phases) {
      const labels = badge(parts, phase);
      // While parts keep coming, the badge reads exactly what each part's own age says.
      for (const {t, age, label} of labels) assert.equal(label, frameAgeNote(age), `${name} phase ${phase} at ${t} ms`);
      assert.equal(flashes(labels), flashes(badge(parts, phase, age => frameAgeNote(age))));
      // The control: counting the time on screen from the first tick flashes on this capture.
      shownFlashes += flashes(badge(parts, phase, (age, shownMs) => frameAgeNote(age + shownMs / 1000)));
    }
    assert.ok(shownFlashes > 0, `${name}: the capture can show a flash`);
  }
});

test('a stalled Spark stream still counts up on screen, at any refresh-tick phase', () => {
  // The captured hung bridge reads, and a stream-only stall: newton with 10-17 s and 22-25 s dropped.
  const dropped = captured.newton.filter(([t]) => !(t >= 10000 && t < 17000) && !(t >= 22000 && t < 25000));
  for (const [name, parts] of [['stall', captured.stall], ['stream-only stall', dropped]]) {
    for (const phase of phases) {
      // Past 5 s without a part the panel's own stale rule reads Not updating (panel.test.mjs).
      const inGap = badge(parts, phase).filter(({shownMs}) => shownMs > 1000 && shownMs <= 5000);
      assert.ok(inGap.length >= 4, `${name} phase ${phase}: ticks inside the gaps`);
      for (const {t, age, shownMs, label} of inGap) {
        assert.equal(label, `Last frame · ${Math.floor(age + shownMs / 1000)} s old`, `${name} phase ${phase} at ${t} ms`);
      }
      const counts = new Set(inGap.map(({label}) => label));
      assert.ok(counts.has('Last frame · 2 s old') && counts.has('Last frame · 3 s old'), `${name}: ${[...counts]}`);
    }
  }
});
