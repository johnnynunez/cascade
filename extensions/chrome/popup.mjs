import {bootstrap,discover,LABELS} from './discovery.mjs';
const $=id=>document.getElementById(id);let tab,hint;
async function attach(expand=false){
 try{
  if(!tab?.id||!/^https?:/.test(tab.url||''))throw Error('Open OpenClaw to see the cameras beside your chat.');
  const [{result:detected}]=await chrome.scripting.executeScript({target:{tabId:tab.id},func:()=>{const app=document.querySelector('openclaw-app');return /\bOpenClaw\b/i.test(document.title)&&!!(app?.shadowRoot||app)?.querySelector('.shell,.login-gate');}});
  if(!detected)throw Error('Open OpenClaw or use the side panel.');
  const remembered=await chrome.storage.local.get('openclawOrigins');
  await chrome.storage.local.set({openclawOrigins:[...new Set([...(remembered.openclawOrigins||[]),new URL(tab.url).origin])].slice(-8)});
  await chrome.storage.session.set({['allowed:'+tab.id]:new URL(tab.url).origin});
  const [{result}]=await chrome.scripting.executeScript({target:{tabId:tab.id},files:['inject.js']});
  if(!result?.ok)throw Error('Could not show the camera. Use the side panel.');
  if(expand)await chrome.tabs.sendMessage(tab.id,{type:'expand'}).catch(()=>{});
  $('integration').textContent='Panel added to OpenClaw. You can keep chatting.';
 }catch(e){$('integration').textContent=e.message;}
}
function state(kind,title,detail){$('status').dataset.state=kind;$('status').textContent=title;$('connection').textContent=detail;}
async function connect(){
 $('connect').disabled=true;
 try{
  if(!await chrome.permissions.contains({origins:[hint.permission]})){state('idle','Cameras ready to connect','Allow this demo to show its cameras.');$('connect').hidden=false;$('permission-note').hidden=false;return;}
  state('connecting','Finding cameras','Connecting to the demo…');
  const {config}=await discover(hint);
  $('views').replaceChildren(...config.names.map(name=>{const e=document.createElement('span');e.textContent=LABELS[name]||name;e.dataset.camera=name;return e;}));
  state('live','Cameras available','Choose a view in the camera panel.');$('connect').hidden=true;$('permission-note').hidden=true;
 }catch{state('offline','Offline','The demo is not responding. Reconnect when it is available.');$('connect').textContent='Reconnect';$('connect').hidden=false;$('permission-note').hidden=true;}
 finally{$('connect').disabled=false;}
}
$('connect').addEventListener('click',async()=>{
 // Permission request stays directly in this real user gesture.
 try{const granted=await chrome.permissions.request({origins:[hint.permission]});if(!granted){state('idle','Permission needed','Click Connect cameras when you are ready to allow video.');return;}await connect();}
 catch{state('offline','Could not connect','Reopen the extension and try again.');}
});
$('attach').addEventListener('click',()=>attach(true));
$('side').addEventListener('click',()=>{if(!tab)return;chrome.sidePanel.open({windowId:tab.windowId}).then(async()=>{await chrome.tabs.sendMessage(tab.id,{type:'close'}).catch(()=>{});window.close();}).catch(()=>{$('integration').textContent='Click Side panel again to open the camera.';});});
$('revoke').addEventListener('click',async()=>{if(hint){await chrome.permissions.remove({origins:[hint.permission]});$('views').replaceChildren();await connect();}});
[tab]=await chrome.tabs.query({active:true,currentWindow:true});
let advertised;
if(tab?.id&&/^https?:/.test(tab.url||'')){try{const [{result}]=await chrome.scripting.executeScript({target:{tabId:tab.id},func:()=>document.querySelector('link[rel="openclaw-demo"]')?.href||document.querySelector('meta[name="openclaw-demo"],meta[name="openclaw-demo-discovery"]')?.content||null});advertised=result;}catch{}}
hint=await bootstrap({pageURL:tab?.url&&/^https?:/.test(tab.url)?tab.url:undefined,advertised});
await chrome.storage.local.set({discoveryHint:hint.statusURL});
const panels=tab?await chrome.runtime.getContexts({contextTypes:['SIDE_PANEL']}):[];
if(panels.some(c=>c.windowId===tab.windowId||c.windowId===-1))$('integration').textContent='The camera is already open in the side panel.';
else await attach();
await connect();
