import {parseCameraURL,cameraNames} from './core.mjs';
export const LABELS={kitchen:'Kitchen',worktop:'Worktop',side:'Side'};
export function cleanURL(value,relativeTo){
 const u=new URL(value,relativeTo);
 if(!['http:','https:'].includes(u.protocol)||u.username||u.password||u.search||u.hash||/%|\\/.test(u.pathname))throw Error('Invalid discovery endpoint');
 return u.href.replace(/\/$/,'');
}
export function hostPermission(url){const u=new URL(cleanURL(url));return u.protocol+'//'+u.hostname+'/*';}
export function statusHint({pageURL,advertised}={}){
 if(advertised)return cleanURL(advertised,pageURL);
 if(pageURL){const u=new URL(pageURL);if(['http:','https:'].includes(u.protocol))return u.origin+'/api/status';}
 throw Error('Open the demo page before connecting its cameras.');
}
export function contract(raw,statusURL){
 const d=raw?.camera_discovery;let base,stateURL,defaultCamera,streams={},labels={};
 if(d?.version===1){
  base=cleanURL(d.base_url,statusURL);stateURL=cleanURL(d.state_url,base+'/');defaultCamera=d.default_camera;
  if(!Array.isArray(d.cameras)||d.cameras.length>32)throw Error('Invalid camera catalogue');
  for(const c of d.cameras){if(!/^[A-Za-z0-9_.-]+$/.test(c.name)||c.name==='..')throw Error('Invalid camera name');streams[c.name]=cleanURL(c.stream_url,base+'/');labels[c.name]=LABELS[c.name]||c.name;}
 }else if(raw?.urls?.cameras){
  base=parseCameraURL(Object.values(raw.urls.cameras)[0]).base;stateURL=base+'/state';defaultCamera='worktop';
  for(const [name,url] of Object.entries(raw.urls.cameras)){if(/^[A-Za-z0-9_.-]+$/.test(name)&&name!=='..')streams[name]=cleanURL(url);}
 }else if(cameraNames(raw).length){
  base=cleanURL(statusURL).replace(/\/state$/,'');stateURL=base+'/state';defaultCamera='worktop';
  for(const name of cameraNames(raw))streams[name]=base+'/stream/'+name;
 }else throw Error('Demo discovery unavailable');
 // One narrow host grant covers discovery and cameras, including proxy ports.
 const permission=hostPermission(statusURL);
 if([base,stateURL,...Object.values(streams)].some(url=>hostPermission(url)!==permission))throw Error('Camera discovery changed host');
 return {version:1,statusURL:cleanURL(statusURL),base,stateURL,streams,labels,defaultCamera,permission};
}
export async function boundedJSON(url,signal){
 const control=new AbortController(),abort=()=>control.abort();signal?.addEventListener('abort',abort,{once:true});if(signal?.aborted)control.abort();
 const timer=setTimeout(abort,6000);
 try{
  const r=await fetch(url,{signal:control.signal,cache:'no-store',credentials:'omit',redirect:'error',referrerPolicy:'no-referrer'});
  if(!r.ok||!r.headers.get('content-type')?.includes('application/json'))throw Error('Demo unavailable');
  const reader=r.body.getReader(),chunks=[];let size=0;
  for(;;){const item=await reader.read();if(item.done)break;size+=item.value.length;if(size>1024*1024){control.abort();throw Error('Discovery too large');}chunks.push(item.value);}
  const bytes=new Uint8Array(size);let offset=0;for(const c of chunks){bytes.set(c,offset);offset+=c.length;}return JSON.parse(new TextDecoder().decode(bytes));
 }finally{clearTimeout(timer);signal?.removeEventListener('abort',abort);}
}
export async function bootstrap(info){
 const saved=await chrome.storage.local.get(['demoDiscovery','discoveryHint','cameraURL']);let statusURL;
 if(info?.advertised||info?.pageURL)statusURL=statusHint(info);
 else statusURL=saved.discoveryHint||saved.demoDiscovery?.statusURL;
 if(!statusURL&&saved.cameraURL){try{statusURL=parseCameraURL(saved.cameraURL).base+'/state';}catch{}}
 statusURL=cleanURL(statusURL||statusHint(info));
 return {statusURL,permission:hostPermission(statusURL)};
}
export async function discover(hint,signal){
 const config=contract(await boundedJSON(hint.statusURL,signal),hint.statusURL),live=await boundedJSON(config.stateURL,signal);
 const names=cameraNames(live).filter(name=>config.streams[name]);if(!names.length)throw Error('No cameras announced');
 config.names=names;config.defaultCamera=names.includes(config.defaultCamera)?config.defaultCamera:names.includes('worktop')?'worktop':names[0];
 // Only public endpoint metadata enters storage. No token or gateway storage.
 await chrome.storage.local.set({demoDiscovery:config,discoveryHint:config.statusURL});
 return {config,state:live};
}
