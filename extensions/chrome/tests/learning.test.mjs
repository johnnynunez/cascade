// ASPIRE A10 (spec §4.8): the learning panel's lines, from the camera server's /state `learning`
// (learning.json, schema 1). Pure: no DOM, no network.
import test from 'node:test';
import assert from 'node:assert/strict';
import {learningLines, MAX_LINES, MAX_LINE} from '../learning.mjs';

const SHA = '3fa9c2d1' + '0'.repeat(56);
const TAG = 'learned in sim | not verifier-tested | lib 3fa9c2d1';

test('learning lines from a schema-1 state', () => {
  assert.deepEqual(learningLines({schema: 1, library_sha: SHA,
    served: [{id: 'a', title: 'Retry from home', tag: TAG}],
    last_event: {kind: 'retry', line: 'Learning loop: ...', outcome: 'placed'}}),
  ['Learned notes · lib 3fa9c2d1', `Retry from home — ${TAG}`, 'Learning loop: ...']);
});

test('anything else renders nothing', () => {
  const full = {library_sha: SHA, served: [{title: 'Retry from home', tag: TAG}],
    last_event: {kind: 'note', line: 'From earlier runs: x.'}};
  for (const v of [undefined, null, {}, {schema: 2}, {schema: '1'}, {schema: true}, [], 'x', 1,
    {...full, schema: 2}, {...full, schema: '1'}, {...full, schema: true}, {...full, schema: 1.5}, {...full},
    {schema: 1, library_sha: 3}, {schema: 1, library_sha: 'zz'}, {schema: 1, library_sha: '<b>3fa9c2d1</b>'}])
    assert.deepEqual(learningLines(v), [], JSON.stringify(v));
});

test('an empty library shows the heading only; a retry with no line shows nothing more', () => {
  assert.deepEqual(learningLines({schema: 1, library_sha: SHA, served: [], last_event: null}),
    ['Learned notes · lib 3fa9c2d1']);
  assert.deepEqual(learningLines({schema: 1, library_sha: SHA, served: [],
    last_event: {kind: 'retry', line: null, outcome: 'failed'}}), ['Learned notes · lib 3fa9c2d1']);
});

test('malformed entries are skipped, never shown raw', () => {
  const lines = learningLines({schema: 1, library_sha: SHA, served: [
    null, 'x', {title: 7, tag: TAG}, {title: 'No tag'}, {title: 'Ok', tag: TAG}, {id: 'only-an-id', tag: TAG},
  ], last_event: {kind: 'note', line: {toString() { return 'never'; }}}});
  assert.deepEqual(lines, ['Learned notes · lib 3fa9c2d1', 'No tag', `Ok — ${TAG}`]);
  assert.ok(!lines.join('\n').includes('only-an-id'), 'a note id is never shown');
});

test('the panel stays small: bounded lines, bounded length, control characters dropped', () => {
  const served = Array.from({length: 40}, (_, i) => ({title: `Note ${i} ` + 'x'.repeat(500), tag: TAG}));
  const lines = learningLines({schema: 1, library_sha: SHA, served,
    last_event: {kind: 'note', line: 'From earlier runs:\u0000\u001b[31m red\u2028 cube\n(tag).'}});
  assert.ok(lines.length <= MAX_LINES + 2, lines.length);
  for (const line of lines) assert.ok(line.length <= MAX_LINE, line.length);
  assert.equal(lines.at(-1), 'From earlier runs: [31m red cube (tag).');
  assert.ok(lines.every(line => !/[\u0000-\u001f\u007f-\u009f\u2028\u2029]/.test(line)));
});

test('the module renders through text only', async () => {
  const {readFile} = await import('node:fs/promises');
  const source = await readFile(new URL('../learning.mjs', import.meta.url), 'utf8');
  const panel = await readFile(new URL('../panel.mjs', import.meta.url), 'utf8');
  for (const text of [source, panel]) {
    assert.ok(!/innerHTML|outerHTML|insertAdjacentHTML|document\.write|DOMParser|createContextualFragment/.test(text));
  }
  assert.match(panel, /learningLines\(data\.learning\)/);
});

test('the module ships in both packages and the version says so', async () => {
  const {readFile} = await import('node:fs/promises');
  const read = path => readFile(new URL(path, import.meta.url), 'utf8');
  const chrome = JSON.parse(await read('../manifest.json'));
  const firefox = JSON.parse(await read('../../firefox/manifest.json'));
  assert.equal(chrome.version, '1.6.0');
  assert.equal(firefox.version, chrome.version);
  assert.match(await read('../package.py'), /'learning\.mjs'/);
  assert.match(await read('../panel.html'), /<section id="learning"[^>]* hidden[^>]*><\/section>/);
});
