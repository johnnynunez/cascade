import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import vm from 'node:vm';
const source=await fs.readFile(new URL('../local.js',import.meta.url),'utf8');
const manifest=JSON.parse(await fs.readFile(new URL('../manifest.json',import.meta.url),'utf8'));

function page(origin,marker=true){
  const added=[], hidden=[];
  const document={documentElement:{dataset:{}},getElementById:()=>null,
    querySelector:selector=>selector==='body[data-paai-chat]'?(marker?{}:null):{prepend:frame=>added.push(frame)},
    createElement:()=>({style:{}}),querySelectorAll:()=>hidden};
  vm.runInNewContext(source,{location:{origin,pathname:'/'},document,
    chrome:{runtime:{id:'local-fixture',getURL:path=>'chrome-extension://local-fixture/'+path}}});
  return {added,document};
}

test('local companion has automatic loopback permission and its own three-camera page',()=>{
  assert.deepEqual(manifest.host_permissions,['http://127.0.0.1/*']);
  assert.deepEqual(manifest.content_scripts[0].js,['local.js']);
  const fixture=page('http://127.0.0.1:8092');
  assert.equal(fixture.added.length,1);
  assert.equal(fixture.added[0].src,'chrome-extension://local-fixture/local.html');
  assert.equal(fixture.document.documentElement.dataset.paaiCameraExtension,'local-fixture');
});

test('extension never injects into personal loopback services or unrelated pages',()=>{
  for(const origin of ['http://127.0.0.1:8090','http://127.0.0.1:18789','http://127.0.0.1:8093','https://example.com']){
    assert.equal(page(origin).added.length,0);
  }
  assert.equal(page('http://127.0.0.1:8092',false).added.length,0);
});

test('local cameras are never labelled live before a producer counter advances',async()=>{
  const code=await fs.readFile(new URL('../local.mjs',import.meta.url),'utf8');
  let id=1;
  const timers=[],status={};
  const figures=['kitchen','worktop','side'].map(camera=>{
    const label={},image={decode:async()=>{}};
    return {dataset:{camera},querySelector:tag=>tag==='span'?label:image};
  });
  const context={document:{querySelectorAll:()=>figures,getElementById:()=>status},
    URL:{createObjectURL:()=> 'blob:fixture',revokeObjectURL(){}},
    AbortSignal:{timeout:()=>({})},window:{addEventListener(){}},setTimeout:fn=>timers.push(fn),
    fetch:async url=>({ok:true,json:async()=>({cameras:Object.fromEntries(figures.map(f=>[f.dataset.camera,{online:true,frame_id:id}]))}),blob:async()=>({})})};
  vm.runInNewContext(code,context);
  await new Promise(setImmediate);
  assert.ok(figures.every(f=>f.dataset.live==='false'));
  await timers.shift()();
  assert.ok(figures.every(f=>f.dataset.live==='false'),'a repeated frozen counter is not live');
  id=2;await timers.shift()();
  assert.ok(figures.every(f=>f.dataset.live==='true'));
});
