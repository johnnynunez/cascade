const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const {launcherURL} = require('../../../web/guide/app.js');
const root = path.resolve(__dirname, '../../..');

test('the guide returns to the same Tailscale learner origin', () => {
  assert.equal(launcherURL(new URL('http://100.101.102.103:8092/guide/')),
    'http://100.101.102.103:8092/');
});

test('authenticated TLS deployments keep their own origin', () => {
  assert.equal(launcherURL(new URL('https://kitchen.example/guide/index.html')),
    'https://kitchen.example/');
});

test('an offline guide stays local instead of choosing a remote computer', () => {
  const current = new URL('file:///tmp/booth-guide/index.html');
  assert.equal(new URL(launcherURL(current), current).href, current.href);
});

test('the perception table distinguishes MCP images from measured localization', () => {
  const readme = fs.readFileSync(path.join(root, 'README.md'), 'utf8');
  const row = name => readme.split('\n').find(line => line.startsWith(`| \`${name}\` |`));
  assert.match(row('get_observation'), /MCP.*image.*localize_object/);
  assert.match(row('describe_scene'), /Fresh.*MCP.*image/);
  assert.doesNotMatch(row('describe_scene'), /instant|no camera wait/i);
  assert.match(row('localize_object'), /camera depth.*configuration provenance/);
});

test('the public visitor remains a light camera page without a chat form', () => {
  const html = fs.readFileSync(path.join(root, 'deploy/brev/visitor.html'), 'utf8');
  const css = fs.readFileSync(path.join(root, 'deploy/brev/visitor.css'), 'utf8');
  assert.match(html, /<meta name="color-scheme" content="light">/);
  assert.match(css, /:root\{color-scheme:light;/);
  assert.doesNotMatch(html, /<(?:form|input|textarea)\b/i);
  assert.doesNotMatch(html, /(?:href|src)="[^"\s]*(?:openclaw|bootstrap)/i);
  for (const name of ['camera', 'video', 'status', 'caption']) {
    assert.equal([...html.matchAll(new RegExp(`\\bid="${name}"`, 'g'))].length, 1);
  }
});
