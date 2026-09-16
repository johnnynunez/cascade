import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import vm from 'node:vm';
import {isAccessLink} from '../history.mjs';

const source=(await fs.readFile(new URL('../worker.js',import.meta.url),'utf8')).replace(/^import .*?;\n/,'');

function boot() {
  const events={},sent=[],scripts=[],saved=new Map([['allowed:7','https://demo.example']]);
  const event=name=>({addListener(fn){events[name]=fn;}});
  const chrome={
    history:{onVisited:event('visited')},
    runtime:{id:'fixture',getURL:path=>'chrome-extension://fixture/'+path,
      onInstalled:event('installed'),onMessage:event('message')},
    storage:{session:{get:async key=>({[key]:saved.get(key)}),remove:async key=>saved.delete(key)},
      local:{setAccessLevel(){}}},
    sidePanel:{open:async()=>{}},
    tabs:{onRemoved:event('removed'),onUpdated:event('updated'),sendMessage:async(id,message)=>sent.push({id,message})},
    scripting:{executeScript:async args=>scripts.push(args)},
  };
  vm.runInNewContext(source,{chrome,isAccessLink,URL},{filename:'worker.js'});
  const sender={id:'fixture',url:'chrome-extension://fixture/panel.html',tab:{id:7,url:'https://demo.example/openclaw/'}};
  const navigate=async()=>{events.updated(7,{status:'complete'},{url:'https://demo.example/openclaw/new'});await new Promise(setImmediate);};
  return {events,sent,scripts,saved,sender,navigate};
}

test('switching from an embedded camera to the side panel prevents reinjection on navigation',async()=>{
  const f=boot();
  const reply=await new Promise(resolve=>f.events.message({type:'open-sidepanel'},f.sender,resolve));
  assert.equal(reply.ok,true);
  assert.equal(f.saved.has('allowed:7'),false);
  assert.equal(f.sent[0].message.type,'close');
  await f.navigate();
  assert.equal(f.scripts.length,0,'a second panel must not cover the chat after navigation');
});

test('an explicitly closed camera stays closed after navigation',async()=>{
  const f=boot();f.events.message({type:'close'},f.sender,()=>{});
  await f.navigate();assert.equal(f.scripts.length,0);
});

test('collapsing keeps the allowed embedded view restorable',async()=>{
  const f=boot();f.events.message({type:'collapse'},f.sender,()=>{});
  await f.navigate();assert.equal(f.scripts.length,1);
  assert.equal(f.scripts[0].target.tabId,7);
});

test('camera resize is sent only to the originating extension tab',()=>{
  const f=boot();
  f.events.message({type:'enlarge-camera',enlarged:true},{...f.sender,id:'another-extension'},()=>{});
  assert.equal(f.sent.length,0);
  f.events.message({type:'enlarge-camera',enlarged:true},f.sender,()=>{});
  assert.equal(f.sent.length,1);assert.equal(f.sent[0].id,7);
  assert.equal(f.sent[0].message.enlarged,true);
});
