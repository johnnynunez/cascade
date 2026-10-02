// The browser AudioContext must admit exactly 24 kHz; no hidden rate guessing.
class PcmCapture extends AudioWorkletProcessor {
  constructor() { super(); this.chunk = new Int16Array(960); this.offset = 0; }
  process(inputs) {
    const channel = inputs[0]?.[0];
    if (channel) for (const sample of channel) {
      const value = Math.max(-1, Math.min(1, sample));
      this.chunk[this.offset++] = Math.round(value * (value < 0 ? 32768 : 32767));
      if (this.offset === this.chunk.length) {
        // Explicit little endian, independent of the JS platform's endianness.
        const buffer = new ArrayBuffer(1920); const view = new DataView(buffer);
        for (let i = 0; i < this.chunk.length; i++) view.setInt16(2*i, this.chunk[i], true);
        this.port.postMessage(buffer, [buffer]); this.offset = 0;
      }
    }
    return true;
  }
}
registerProcessor('pcm-capture', PcmCapture);
