import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import vm from 'node:vm';
import {isAccessLink} from '../../chrome/history.mjs';

// Firefox exposes chrome.sidebarAction instead of chrome.sidePanel and has no
// storage.local.setAccessLevel. These fakes follow extensions/chrome/tests.
const event=registry=>name=>({addListener(fn){registry[name]=fn;}});

test('background worker starts in Firefox without storage access levels',async()=>{
  const source=(await fs.readFile(new URL('../../chrome/worker.js',import.meta.url),'utf8')).replace(/^import .*?;\n/,'');
  const events={},on=event(events);
  const chrome={history:{onVisited:on('visited')},
    runtime:{id:'fixture',getURL:path=>'moz-extension://fixture/'+path,onInstalled:on('installed'),onMessage:on('message')},
    storage:{session:{get:async()=>({}),remove:async()=>{}},local:{}},
    tabs:{onRemoved:on('removed'),onUpdated:on('updated'),sendMessage:async()=>{}},scripting:{executeScript:async()=>{}}};
  vm.runInNewContext(source,{chrome,isAccessLink,URL},{filename:'worker.js'});
  assert.doesNotThrow(()=>events.installed());
});

async function popup(key){
  const elements=new Map(),opened=[];let closed=false;
  const original={document:globalThis.document,chrome:globalThis.chrome,window:globalThis.window};
  globalThis.document={getElementById(id){
    if(!elements.has(id))elements.set(id,{value:'',textContent:'',dataset:{},events:{},addEventListener(event,fn){this.events[event]=fn;}});
    return elements.get(id);
  }};
  globalThis.window={close(){closed=true;}};
  const session=new Map();
  globalThis.chrome={
    storage:{local:{get:async()=>({}),set:async()=>{}},session:{set:async values=>Object.entries(values).forEach(([k,v])=>session.set(k,v)),remove:async key=>session.delete(key)}},
    permissions:{contains:async()=>false},
    tabs:{sendMessage:async()=>{},query:async()=>[{id:17,windowId:9,url:'http://127.0.0.1:18790/chat'}]},
    runtime:{getContexts:async()=>[]},
    sidebarAction:{open:async()=>{opened.push('sidebar');}},
    scripting:{executeScript:async args=>[{result:args.func?.toString().includes('link[rel=')?null:args.files?{ok:true}:true}]},
  };
  try {
    await import(`../../chrome/popup.mjs?firefox=${key}`);
    return {elements,opened,session,get closed(){return closed;},restore(){Object.assign(globalThis,original);}};
  } catch(error) {Object.assign(globalThis,original);throw error;}
}

test('Side panel opens the Firefox sidebar within the click and removes the page overlay',async()=>{
  const env=await popup('sidebar');
  try {
    assert.equal(env.session.get('allowed:17'),'http://127.0.0.1:18790');
    env.elements.get('side').events.click();
    await new Promise(setImmediate);
    assert.deepEqual(env.opened,['sidebar'],'sidebarAction.open must run synchronously in the click handler');
    assert.equal(env.session.has('allowed:17'),false);
    assert.equal(env.closed,true);
  } finally {env.restore();}
});

async function embeddedPanel(key,permissions){
  const elements=new Map(),messages=[],fetches=[],listeners={};
  const original=Object.fromEntries(['document','window','location','chrome','fetch']
    .map(name=>[name,Object.getOwnPropertyDescriptor(globalThis,name)]));
  const element=id=>{
    if(!elements.has(id))elements.set(id,{textContent:'',dataset:{},events:{},attributes:{},hidden:false,
      addEventListener(event,fn){this.events[event]=fn;},setAttribute(name,value){this.attributes[name]=value;},
      removeAttribute(){},querySelector(){return null;}});
    return elements.get(id);
  };
  Object.assign(globalThis,{
    document:{getElementById:element,querySelectorAll:()=>[],querySelector:()=>({classList:{toggle(){}}}),
      body:{classList:{add(){}}},addEventListener(){},hidden:false},
    window:{addEventListener(name,fn){listeners[name]=fn;}},
    location:{search:'?surface=embedded'},
    chrome:{runtime:{sendMessage:async message=>{messages.push(message);return {ok:true};}},
      storage:{local:{get:async()=>({discoveryHint:'http://127.0.0.1:8091/api/status'}),set:async()=>{}},onChanged:{addListener(){}}},
      ...(permissions?{permissions:{contains:async()=>true,onRemoved:{addListener(){}},onAdded:{addListener(){}}}}:{})},
    fetch:async url=>{fetches.push(url);throw new Error('offline fixture');},
  });
  try {
    await import(`../../chrome/panel.mjs?embedded=${key}`);
    await new Promise(setImmediate);
    return {side:element('side').hidden,messages,fetches};
  } finally {
    listeners.pagehide?.();  // cancels the panel's reconnect timers
    for (const [name,descriptor] of Object.entries(original)) {
      if (descriptor) Object.defineProperty(globalThis,name,descriptor);
      else delete globalThis[name];
    }
  }
}

test('embedded Firefox panel hides the unavailable side panel control and still connects',async()=>{
  // Firefox content-script API set: runtime and storage.local only.
  const firefox=await embeddedPanel('firefox',false);
  assert.equal(firefox.side,true);
  assert.deepEqual(firefox.messages[0],{type:'authorize-panel'});
  assert.equal(firefox.fetches[0],'http://127.0.0.1:8091/api/status','discovery runs without chrome.permissions');
  // Chrome's embedded panel has the permissions API and keeps its side panel button.
  const chrome=await embeddedPanel('chrome',true);
  assert.equal(chrome.side,false);
  assert.equal(chrome.fetches[0],'http://127.0.0.1:8091/api/status');
});
