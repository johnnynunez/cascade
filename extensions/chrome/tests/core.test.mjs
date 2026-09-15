import test from 'node:test';
import assert from 'node:assert/strict';
import {parseCameraURL, cameraNames, MJPEGParser, freshness} from '../core.mjs';
test('base URL and actual stream route keep explicit port and proxy prefix',()=>{
 assert.deepEqual(parseCameraURL(' http://matrix:8765/ '),{url:'http://matrix:8765',base:'http://matrix:8765',camera:'',permission:'http://matrix/*'});
 assert.equal(parseCameraURL('https://camera.example/cascade/stream/overhead').base,'https://camera.example/cascade');
 assert.equal(parseCameraURL('http://matrix:8765/stream/front').camera,'front');
});
test('reject credentials, tokens, dangerous schemes and unsupported endpoints',()=>{
 for(const raw of ['javascript:alert(1)','file:///foo','http://user:secret@matrix','http://matrix?token=x','http://matrix/#token','http://matrix/depth/front','http://matrix/snapshot/front.jpg','http://matrix/stream/a%20b']) assert.throws(()=>parseCameraURL(raw));
});
test('camera discovery accepts real stats and filters path/control injection',()=>{
 assert.deepEqual(cameraNames({cameras:{overhead:{frame_id:1},front:{},'../x':{},'..':{},'<img>':{}}}),['overhead','front']);
 assert.deepEqual(cameraNames({cameras:null}),[]);
 assert.deepEqual(cameraNames({cameras:[]}),[]);
});
const jpeg=new Uint8Array([255,216,1,2,255,217]);
const part=new Uint8Array([...new TextEncoder().encode('--wrcframe\r\nContent-Type: image/jpeg\r\nContent-Length: 6\r\n\r\n'),...jpeg,13,10]);
test('MJPEG parser tolerates every byte boundary and multiple frames',()=>{
 const parser=new MJPEGParser(),frames=[];for(const b of [...part,...part])frames.push(...parser.push(new Uint8Array([b])));
 assert.equal(frames.length,2);assert.deepEqual(frames[0],jpeg);
 assert.equal(new MJPEGParser().push(new Uint8Array([...part,...part])).length,2);
});
test('MJPEG parser rejects missing lengths, oversized frames and non-JPEG bytes',()=>{
 for(const header of ['Content-Type: image/jpeg\r\n\r\n','Content-Type: image/jpeg\r\nContent-Length: 9000000\r\n\r\n','Content-Type: text/html\r\nContent-Length: 6\r\n\r\n'])assert.throws(()=>new MJPEGParser().push(new TextEncoder().encode(header)));
 const bad=part.slice();bad[bad.length-8]=0;assert.throws(()=>new MJPEGParser().push(bad));
 assert.throws(()=>new MJPEGParser().push(new Uint8Array(8193)));
});
const recent={now:10000,lastFrame:9900,lastState:9800,lastAdvance:9000,hasAdvanced:true,error:false};
test('freshness requires decoded image and advancing server camera counter',()=>{
 assert.equal(freshness({...recent,lastFrame:0}),'connecting');
 assert.equal(freshness({...recent,hasAdvanced:false}),'unverified');
 assert.equal(freshness(recent),'live');
 assert.equal(freshness({...recent,lastAdvance:1,hasAdvanced:false}),'stale');
 for(const delta of [{lastFrame:1},{lastState:1},{lastAdvance:1},{error:true}])assert.equal(freshness({...recent,...delta}),'stale');
});
