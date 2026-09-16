// Assemble native OpenClaw assets and refresh derived data after the light overlay.
import {createHash} from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {brotliCompressSync, brotliDecompressSync, gzipSync, gunzipSync} from 'node:zlib';

const digest = data => createHash('sha256').update(data).digest('hex');

function files(root, prefix = '') {
  return fs.readdirSync(path.join(root, prefix), {withFileTypes: true}).flatMap(entry => {
    const relative = path.posix.join(prefix, entry.name);
    if (entry.isSymbolicLink()) throw Error('Native UI inputs must not contain symbolic links');
    if (entry.isDirectory()) return files(root, relative);
    if (!entry.isFile()) throw Error('Native UI inputs must be regular files');
    return [relative];
  });
}

function update(filename, contents) {
  if (fs.existsSync(filename)) {
    if (!fs.lstatSync(filename).isFile()) throw Error('Native UI output must be a regular file');
    if (fs.readFileSync(filename).equals(contents)) return false;
  }
  fs.mkdirSync(path.dirname(filename), {recursive: true, mode: 0o700});
  const temporary = filename + `.${process.pid}.tmp`;
  let created = false;
  try {
    fs.writeFileSync(temporary, contents, {mode: 0o600, flag: 'wx'});
    created = true;
    fs.renameSync(temporary, filename);
  } finally {
    if (created) fs.rmSync(temporary, {force: true});
  }
  return true;
}

export function materializeUi({upstream, ui, patches, version}) {
  for (const directory of [upstream, patches]) {
    if (!fs.lstatSync(directory).isDirectory()) throw Error('Native UI input directory is missing');
  }
  if (!/^\d+\.\d+\.\d+(?:[-.a-zA-Z0-9]*)$/.test(version)) throw Error('Native UI version is invalid');
  fs.mkdirSync(ui, {recursive: true, mode: 0o700});
  if (!fs.lstatSync(ui).isDirectory()) throw Error('Native UI output must be a directory');
  files(ui);
  const marker = path.join(ui, '.paai-native-version');
  if (!fs.existsSync(marker) || fs.readFileSync(marker, 'utf8').trim() !== version) {
    for (const relative of files(upstream)) {
      update(path.join(ui, relative), fs.readFileSync(path.join(upstream, relative)));
    }
    update(marker, Buffer.from(version + '\n'));
  }
  const overlay = [];
  let compressedUpdated = 0;
  for (const relative of files(patches)) {
    if (relative.endsWith('.gz') || relative.endsWith('.br') || relative === 'asset-manifest.json') {
      throw Error('The overlay must contain source assets, not derived compression or manifests');
    }
    const source = fs.readFileSync(path.join(patches, relative));
    if (source.length > 16 * 1024 * 1024) throw Error('Native UI overlay exceeds its file budget');
    const destination = path.join(ui, relative);
    update(destination, source);
    for (const [suffix, decode, encode] of [['.gz', gunzipSync, gzipSync], ['.br', brotliDecompressSync, brotliCompressSync]]) {
      const compressed = destination + suffix;
      if (!fs.existsSync(compressed)) continue;
      if (!fs.lstatSync(compressed).isFile()) throw Error('Compressed UI output must be a regular file');
      let matches = false;
      try { matches = decode(fs.readFileSync(compressed), {maxOutputLength: source.length + 1}).equals(source); } catch {}
      if (!matches && update(compressed, encode(source))) compressedUpdated++;
    }
    overlay.push({path: relative, sha256: digest(source), size: source.length});
  }
  const assets = files(path.join(ui, 'assets')).map(relative => {
    const data = fs.readFileSync(path.join(ui, 'assets', relative));
    if (data.length > 67108864) throw Error('Native UI asset exceeds the manifest file limit');
    return {path: `assets/${relative}`, sha256: digest(data), size: data.length};
  }).sort((left, right) => left.path.localeCompare(right.path));
  if (!assets.length || assets.length > 8192 || assets.reduce((total, row) => total + row.size, 0) > 536870912) {
    throw Error('Native UI assets exceed the native manifest limits');
  }
  // This is the pinned native manifest's byte framing, including its locale ordering.
  const generation = createHash('sha256');
  for (const entry of assets) generation.update(`${entry.path}\0${entry.size}\0${entry.sha256}\n`);
  const manifest = {version: 1, generation: generation.digest('hex'), assets};
  update(path.join(ui, 'asset-manifest.json'), Buffer.from(JSON.stringify(manifest, null, 2) + '\n'));
  const receipt = {version, generation: manifest.generation, overlay,
                   asset_count: assets.length, status: 'MATERIALIZED'};
  update(path.join(ui, '.paai-materialization.json'), Buffer.from(JSON.stringify(receipt, null, 2) + '\n'));
  return {...receipt, compressed_updated: compressedUpdated};
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const [upstream, ui, patches, version] = process.argv.slice(2);
  try {
    console.log(JSON.stringify(materializeUi({upstream, ui, patches, version})));
  } catch (error) {
    console.error(JSON.stringify({status: 'FAILED', error: error.name}));
    process.exitCode = 1;
  }
}
