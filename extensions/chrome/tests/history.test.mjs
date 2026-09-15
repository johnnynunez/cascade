import test from 'node:test';
import assert from 'node:assert/strict';
import {isAccessLink} from '../history.mjs';

test('only actual OpenClaw fragment credentials are eligible for history removal',()=>{
 assert.equal(isAccessLink('http://127.0.0.1:18790/#token=test-placeholder'),true);
 assert.equal(isAccessLink('https://remote.example/chat/main#token=test-placeholder',['https://remote.example']),true);
 assert.equal(isAccessLink('https://demo.example/openclaw#token=test-placeholder',['https://demo.example']),true);
 for(const url of ['http://127.0.0.1:18790/chat/main','http://127.0.0.1:18790/#token=',
   'http://127.0.0.1:8092/#token=test-placeholder','https://unrelated.example/#token=test-placeholder',
   'http://127.0.0.1:18790.evil.example/#token=test-placeholder','file:///tmp/access#token=test-placeholder',
   'http://user:password@127.0.0.1:18790/#token=test-placeholder']) assert.equal(isAccessLink(url),false,url);
});
