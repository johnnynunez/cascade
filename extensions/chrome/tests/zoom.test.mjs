import test from 'node:test';
import assert from 'node:assert/strict';
import {cameraZoom} from '../zoom.mjs';

// Small DOM stand-in: moving a node changes its owner but keeps its identity.
function makeDocument() {
  const document = {baseURI:'chrome-extension://fixture/panel.html?surface=side'};
  document.createElement = tag => {
    const classes = new Set();
    return {
      tag, ownerDocument:document, children:[], attributes:{}, events:{},
      classList:{toggle(name,value){if(value)classes.add(name);else classes.delete(name);},contains:name=>classes.has(name)},
      append(...nodes){for(const node of nodes){
        if(node.parentNode)node.parentNode.children.splice(node.parentNode.children.indexOf(node),1);
        node.parentNode=this;node.ownerDocument=this.ownerDocument;this.children.push(node);
      }},
      replaceWith(node){const parent=this.parentNode,index=parent.children.indexOf(this);
        if(node.parentNode)node.parentNode.children.splice(node.parentNode.children.indexOf(node),1);
        parent.children[index]=node;node.parentNode=parent;node.ownerDocument=parent.ownerDocument;this.parentNode=null;
      },
      setAttribute(name,value){this.attributes[name]=value;},
      querySelector(){return null;},
      addEventListener(name,fn){this.events[name]=fn;},
      focus(){this.ownerDocument.activeElement=this;},
    };
  };
  document.documentElement=document.createElement('html');
  document.head=document.createElement('head');document.body=document.createElement('body');
  return document;
}

function fixture({embedded=false,blocked=false}={}) {
  const document=makeDocument(),companion=document.createElement('section'),viewer=document.createElement('button');
  const image={src:'blob:current-camera-frame',reader:{}};
  viewer.camera=image;companion.append(viewer);document.body.append(companion);
  const events={},resize=[],opened=[];
  const popup={document:makeDocument(),closed:false,
    addEventListener(name,fn){events[name]=fn;},focus(){},
    close(){this.closed=true;events.pagehide?.();}};
  const window={screen:{availWidth:1440,availHeight:900},focus(){},
    open(...args){opened.push(args);return blocked?null:popup;}};
  const zoom=cameraZoom({viewer,companion,embedded,resizeEmbedded:value=>resize.push(value),window,document});
  return {document,companion,viewer,image,events,resize,opened,popup,zoom};
}

test('embedded enlargement and repeated return retain the camera element and reader',()=>{
  const f=fixture({embedded:true}),reader=f.image.reader;
  for(let i=0;i<10;i++){
    f.zoom.toggle();assert.equal(f.zoom.active,true);
    assert.equal(f.viewer.attributes['aria-expanded'],'true');
    f.zoom.toggle();assert.equal(f.zoom.active,false);
  }
  assert.deepEqual(f.resize,Array.from({length:20},(_,i)=>i%2===0));
  assert.equal(f.opened.length,0);
  assert.equal(f.document.body.children[0],f.companion);
  assert.equal(f.companion.children[0],f.viewer);
  assert.equal(f.viewer.camera.reader,reader);
  assert.equal(f.viewer.camera.src,'blob:current-camera-frame');
  assert.equal(f.document.activeElement,f.viewer);
});

test('large side-panel view moves the current UI back on a second click',()=>{
  const f=fixture(),reader=f.image.reader;
  f.zoom.toggle();
  assert.equal(f.popup.document.body.children[0],f.companion);
  assert.equal(f.viewer.attributes['aria-label'],'Return camera to panel');
  assert.equal(f.popup.document.head.children[0].href,'chrome-extension://fixture/ui.css');
  assert.equal(f.opened[0][0],'','a camera URL must never enter the window URL');
  f.zoom.toggle();
  assert.equal(f.popup.closed,true);
  assert.equal(f.document.body.children[0],f.companion);
  assert.equal(f.companion.children[0],f.viewer);
  assert.equal(f.viewer.camera.reader,reader);
  assert.equal(f.viewer.attributes['aria-expanded'],'false');
});

test('Escape and the window close button both return the same view',()=>{
  for(const action of ['escape','close']){
    const f=fixture();f.zoom.toggle();
    if(action==='escape')f.events.keydown({key:'Escape',preventDefault(){}});
    else f.popup.close();
    assert.equal(f.zoom.active,false);
    assert.equal(f.document.body.children[0],f.companion);
    assert.equal(f.viewer.attributes['aria-label'],'Enlarge camera');
  }
});

test('a blocked window leaves the original camera available',()=>{
  const f=fixture({blocked:true});
  assert.throws(()=>f.zoom.toggle(),/could not open/);
  assert.equal(f.zoom.active,false);
  assert.equal(f.document.body.children[0],f.companion);
  assert.equal(f.viewer.camera.src,'blob:current-camera-frame');
  assert.equal(f.viewer.attributes['aria-expanded'],'false');
});
