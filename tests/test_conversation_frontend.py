"""Execute the browser capture worklet without microphone hardware."""
import shutil
import subprocess
from pathlib import Path

import pytest

NODE = shutil.which("node")
STATIC = Path(__file__).resolve().parents[1] / "src/cascade/conversation/static"


@pytest.mark.skipif(NODE is None, reason="Node is optional for browser worklet validation")
def test_worklet_pcm_endianness_saturation_and_fixed_size_without_mic():
    script = r"""
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const chunks = []; let Worklet;
const sandbox = {AudioWorkletProcessor: class {constructor(){this.port={postMessage: b => chunks.push(b)}}},
  registerProcessor: (name, implementation) => {assert.equal(name, 'pcm-capture'); Worklet=implementation;},
  Int16Array, DataView, ArrayBuffer, Math};
vm.runInNewContext(fs.readFileSync(process.argv[1], 'utf8'), sandbox);
const processor = new Worklet();
const input = new Float32Array(960); input[0]=-2;input[1]=2;input[2]=.5;input[3]=-.5;
assert.equal(processor.process([[input]]),true);
assert.equal(chunks.length,1);assert.equal(chunks[0].byteLength,1920);
const bytes = new DataView(chunks[0]);
assert.equal(bytes.getInt16(0,true),-32768);assert.equal(bytes.getInt16(2,true),32767);
assert.equal(bytes.getInt16(4,true),16384);assert.equal(bytes.getInt16(6,true),-16384);
processor.process([[new Float32Array(959)]]);assert.equal(chunks.length,1);
processor.process([[new Float32Array(1)]]);assert.equal(chunks.length,2);
"""
    subprocess.run([NODE, "-e", script, str(STATIC / "capture.js")], check=True, timeout=10)
    subprocess.run([NODE, "--check", str(STATIC / "app.js")], check=True, timeout=10)
