// PCM bursts may arrive faster than speech is played. Keep bounded local audio,
// schedule only a short horizon, and discard every queued/scheduled frame on flush.
class PcmPlaybackQueue {
  constructor(context, onError) {
    this.context = context;
    this.onError = onError;
    this.generation = 0;
    this.queue = [];
    this.queuedSamples = 0;
    this.sources = new Set();
    this.playAt = 0;
    this.timer = null;
  }
  get pendingSeconds() {
    return this.queuedSamples / 24000 + Math.max(0, this.playAt - this.context.currentTime);
  }
  enqueue(bytes, generation) {
    if (generation !== this.generation) return false;
    if (!(bytes instanceof Uint8Array) || !bytes.length || bytes.length % 2 || bytes.length > 96000)
      throw new Error('Invalid playback audio');
    const samples = bytes.length / 2;
    // 15 s PCM is 720 kB on the wire. Also bound object count for tiny chunks.
    if (this.pendingSeconds + samples / 24000 + Math.max(0, this.context.currentTime + .02 - Math.max(this.context.currentTime, this.playAt)) > 15 ||
        this.queue.length + this.sources.size + Math.ceil(samples / 2400) > 512)
      throw new Error('Playback queue exceeds its audio budget');
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    for (let offset = 0; offset < samples; offset += 2400) {
      const length = Math.min(2400, samples - offset);
      const buffer = this.context.createBuffer(1, length, 24000);
      const target = buffer.getChannelData(0);
      for (let i = 0; i < length; i++) target[i] = view.getInt16(2 * (offset + i), true) / 32768;
      this.queue.push(buffer);
      this.queuedSamples += length;
    }
    this.pump();
    return true;
  }
  pump() {
    const now = this.context.currentTime;
    while (this.queue.length) {
      const buffer = this.queue[0];
      const at = Math.max(now + .02, this.playAt);
      if (at + buffer.duration > now + 2) break;
      this.queue.shift();
      this.queuedSamples -= buffer.length;
      const source = this.context.createBufferSource();
      source.buffer = buffer;
      source.connect(this.context.destination);
      this.sources.add(source);
      source.onended = () => this.sources.delete(source);
      source.start(at);
      this.playAt = at + buffer.duration;
    }
    if (this.queue.length && this.timer === null) {
      this.timer = setTimeout(() => {
        this.timer = null;
        try { this.pump(); }
        catch (error) { this.flush(); this.onError(error); }
      }, 50);
    }
  }
  flush(generation = this.generation) {
    this.generation = generation;
    if (this.timer !== null) clearTimeout(this.timer);
    this.timer = null;
    this.queue.length = 0;
    this.queuedSamples = 0;
    for (const source of this.sources) { try { source.stop(); } catch {} }
    this.sources.clear();
    this.playAt = 0;
  }
}
globalThis.PcmPlaybackQueue = PcmPlaybackQueue;
