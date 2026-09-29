import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import {execFileSync} from 'node:child_process';

const read = async url => JSON.parse(await fs.readFile(new URL(url, import.meta.url), 'utf8'));
const firefox = await read('../manifest.json');
const chrome = await read('../../chrome/manifest.json');

test('Firefox manifest keeps the Chrome name, version, permissions and pages',()=>{
  for (const key of ['manifest_version','name','version','description','optional_host_permissions','host_permissions',
    'content_scripts','content_security_policy','web_accessible_resources','commands'])
    assert.deepEqual(firefox[key],chrome[key],key);
  assert.deepEqual(firefox.permissions,chrome.permissions.filter(name=>name!=='sidePanel'));
  assert.equal(firefox.action.default_popup,chrome.action.default_popup);
  assert.equal(firefox.action.default_title,chrome.action.default_title);
  assert.equal(firefox.sidebar_action.default_panel,chrome.side_panel.default_path);
  assert.deepEqual(firefox.background,{scripts:[chrome.background.service_worker],type:'module'});
});

test('Firefox manifest uses only Firefox keys and declares its AMO metadata',()=>{
  for (const key of ['side_panel','minimum_chrome_version']) assert.equal(key in firefox,false,key);
  assert.equal('service_worker' in firefox.background,false);
  const gecko=firefox.browser_specific_settings.gecko;
  assert.match(gecko.id,/^[\w.-]+@[\w.-]+$/);
  assert.ok(Number.parseFloat(gecko.strict_min_version)>=140,'data_collection_permissions needs Firefox 140');
  assert.deepEqual(gecko.data_collection_permissions,{required:['none']});
  assert.equal(firefox.action.default_area,'navbar');
  assert.equal(firefox.sidebar_action.open_at_install,false);
});

test('package assembles shared sources with the Firefox manifest, deterministically',async()=>{
  const out=await fs.mkdtemp(path.join(os.tmpdir(),'paai-firefox-'));
  try {
    const script=new URL('../package.py',import.meta.url).pathname;
    execFileSync('python3',[script,'--out',out],{stdio:'pipe'});
    const xpi=path.join(out,'Physical-Agentic-AI-OpenClaw-Demo.xpi');
    const first=await fs.readFile(xpi);
    execFileSync('python3',[script,'--out',out],{stdio:'pipe'});
    assert.deepEqual(await fs.readFile(xpi),first,'rebuilding must give identical bytes');
    const build=path.join(out,'build');
    const names=(await fs.readdir(build)).sort();
    const shared=(await fs.readdir(new URL('../../chrome/',import.meta.url))).filter(name=>/\.(js|mjs|html|css)$/.test(name));
    for (const name of shared) {
      assert.ok(names.includes(name),name);
      assert.deepEqual(await fs.readFile(path.join(build,name)),await fs.readFile(new URL('../../chrome/'+name,import.meta.url)),name);
    }
    assert.deepEqual(JSON.parse(await fs.readFile(path.join(build,'manifest.json'),'utf8')),firefox);
    assert.equal(names.includes('package.py'),false);
    assert.equal(names.some(name=>name.endsWith('.test.mjs')),false);
    const referenced=[firefox.background.scripts[0],firefox.action.default_popup,firefox.sidebar_action.default_panel.split('?')[0],
      ...firefox.content_scripts.flatMap(script=>script.js),...firefox.web_accessible_resources.flatMap(entry=>entry.resources)];
    for (const name of referenced) assert.ok(names.includes(name),name);
  } finally { await fs.rm(out,{recursive:true,force:true}); }
});
