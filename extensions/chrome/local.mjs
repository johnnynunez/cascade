const base='http://127.0.0.1:8091';
const figures=[...document.querySelectorAll('[data-camera]')];
const seen=new Map(), urls=new Map();
let stopped=false;
async function poll(){
  if(stopped)return;
  if(document.hidden){setTimeout(poll,1500);return;}
  let live=0;
  try{
    const response=await fetch(base+'/state',{cache:'no-store',credentials:'omit',signal:AbortSignal.timeout(6000)});
    if(!response.ok)throw Error('offline');
    const state=await response.json();
    await Promise.all(figures.map(async figure=>{
      const name=figure.dataset.camera, row=state.cameras?.[name];
      const previous=seen.get(name), advanced=previous?.id!==row?.frame_id;
      const when=advanced?Date.now():previous?.when||0;
      const confirmed=!!previous&&(previous.confirmed||advanced);
      const fresh=!!row?.online&&confirmed&&Date.now()-when<8000;
      seen.set(name,{id:row?.frame_id,when,confirmed});
      figure.dataset.frameId=String(row?.frame_id||0);
      figure.dataset.live=String(fresh);
      figure.querySelector('span').textContent=fresh?'Live':row?.online?'Checking':'Reconnecting';
      if(fresh)live++;
      if(!row?.online)return;
      const image=await fetch(base+'/snapshot/'+name+'.jpg',{cache:'no-store',credentials:'omit',signal:AbortSignal.timeout(6000)});
      if(!image.ok)throw Error('offline');
      const url=URL.createObjectURL(await image.blob());
      const element=figure.querySelector('img');
      element.src=url;
      await element.decode();
      const old=urls.get(name);urls.set(name,url);if(old)URL.revokeObjectURL(old);
    }));
    document.getElementById('status').textContent=live===3?'Three cameras live':'Checking camera freshness…';
  }catch{
    document.getElementById('status').textContent='Cameras reconnecting…';
    figures.forEach(figure=>{figure.dataset.live='false';figure.querySelector('span').textContent='Reconnecting';});
  }
  if(!stopped)setTimeout(poll,600);
}
window.addEventListener('pagehide',()=>{stopped=true;for(const url of urls.values())URL.revokeObjectURL(url);});
poll();
