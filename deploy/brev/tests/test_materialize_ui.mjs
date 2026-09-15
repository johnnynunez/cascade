import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {test} from 'node:test';
import {brotliCompressSync, brotliDecompressSync, gzipSync, gunzipSync} from 'node:zlib';
import {materializeUi} from '../container/materialize_ui.mjs';

function fixture(t) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'paai-native-ui-'));
  t.after(() => fs.rmSync(root, {recursive: true, force: true}));
  const options = {upstream: path.join(root, 'native'), patches: path.join(root, 'patches'),
                   ui: path.join(root, 'materialized'), version: '2026.9.3'};
  for (const directory of [options.upstream, options.patches]) {
    fs.mkdirSync(path.join(directory, 'assets'), {recursive: true});
  }
  const before = Buffer.from(':root{color-scheme:dark}');
  const after = Buffer.from(':root{color-scheme:light}');
  fs.writeFileSync(path.join(options.upstream, 'index.html'), '<html>native</html>');
  fs.writeFileSync(path.join(options.upstream, 'assets/style.css'), before);
  fs.writeFileSync(path.join(options.upstream, 'assets/style.css.gz'), gzipSync(before));
  fs.writeFileSync(path.join(options.upstream, 'assets/style.css.br'), brotliCompressSync(before));
  fs.writeFileSync(path.join(options.patches, 'assets/style.css'), after);
  return {options, before, after};
}

test('the overlay repairs both encodings and creates the native checksum framing', t => {
  const {options, before, after} = fixture(t);
  const result = materializeUi(options);
  assert.equal(result.compressed_updated, 2);
  const css = path.join(options.ui, 'assets/style.css');
  assert.deepEqual(gunzipSync(fs.readFileSync(css + '.gz')), after);
  assert.deepEqual(brotliDecompressSync(fs.readFileSync(css + '.br')), after);
  assert.deepEqual(fs.readFileSync(path.join(options.upstream, 'assets/style.css')), before);
  const manifest = JSON.parse(fs.readFileSync(path.join(options.ui, 'asset-manifest.json')));
  const hash = createHash('sha256');
  for (const entry of manifest.assets) {
    const data = fs.readFileSync(path.join(options.ui, entry.path));
    assert.equal(entry.size, data.length);
    assert.equal(entry.sha256, createHash('sha256').update(data).digest('hex'));
    hash.update(entry.path).update('\0').update(String(entry.size)).update('\0').update(entry.sha256).update('\n');
  }
  assert.equal(manifest.generation, hash.digest('hex'));
});

test('an unchanged restart preserves artifact bytes and modification times', t => {
  const {options} = fixture(t);
  materializeUi(options);
  const names = ['assets/style.css', 'assets/style.css.br', 'assets/style.css.gz', 'asset-manifest.json', '.paai-materialization.json'];
  const times = names.map(name => fs.statSync(path.join(options.ui, name), {bigint: true}).mtimeNs);
  const result = materializeUi(options);
  assert.equal(result.compressed_updated, 0);
  assert.deepEqual(names.map(name => fs.statSync(path.join(options.ui, name), {bigint: true}).mtimeNs), times);
});

test('a stale compressed file is repaired even when the overlay was already copied', t => {
  const {options, before, after} = fixture(t);
  materializeUi(options);
  const compressed = path.join(options.ui, 'assets/style.css.br');
  fs.writeFileSync(compressed, brotliCompressSync(before));
  assert.equal(materializeUi(options).compressed_updated, 1);
  assert.deepEqual(brotliDecompressSync(fs.readFileSync(compressed)), after);
});

test('an output symlink cannot redirect the overlay outside its directory', t => {
  const {options, before} = fixture(t);
  fs.mkdirSync(options.ui);
  fs.symlinkSync(path.join(options.upstream, 'assets'), path.join(options.ui, 'assets'));
  assert.throws(() => materializeUi(options), /symbolic links/);
  assert.deepEqual(fs.readFileSync(path.join(options.upstream, 'assets/style.css')), before);
});

test('derived files cannot masquerade as reviewed overlay inputs', t => {
  const {options} = fixture(t);
  fs.writeFileSync(path.join(options.patches, 'assets/style.css.br'), 'stale fixture');
  assert.throws(() => materializeUi(options), /not derived/);
});
