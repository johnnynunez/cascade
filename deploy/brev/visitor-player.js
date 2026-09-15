class VisitorVideo {
  constructor(video, image, onFailure) {
    this.video = video;
    this.image = image;
    this.onFailure = onFailure;
    this.lastFrameAt = 0;
    this.camera = undefined;
    this.version = 0;
  }

  get playing() {
    return this.lastFrameAt > Date.now() - 5000 && !this.video.hidden;
  }

  stop() {
    this.version += 1;
    clearTimeout(this.frameTimer);
    this.frameTimer = undefined;
    if (this.frameCallback !== undefined) this.video.cancelVideoFrameCallback(this.frameCallback);
    this.frameCallback = undefined;
    this.controller?.abort();
    this.video.pause();
    this.video.removeAttribute('src');
    this.video.load();
    if (this.url) URL.revokeObjectURL(this.url);
    this.url = undefined;
    this.video.hidden = true;
    this.image.hidden = false;
    this.lastFrameAt = 0;
    this.camera = undefined;
  }

  async start(camera) {
    this.stop();
    const type = 'video/mp4; codecs="avc1.42C01F"';
    if (!window.MediaSource?.isTypeSupported(type) || !this.video.requestVideoFrameCallback) return;
    const version = this.version;
    const controller = new AbortController();
    this.controller = controller;
    this.camera = camera;
    const fail = () => {
      if (version !== this.version) return;
      this.stop();
      this.onFailure();
    };
    this.frameTimer = setTimeout(fail, 15000);
    let timer;
    try {
      const media = new MediaSource();
      const opened = new Promise((resolve, reject) => {
        media.addEventListener('sourceopen', resolve, {once: true});
        controller.signal.addEventListener('abort', () => reject(new Error('Video closed')), {once: true});
      });
      this.url = URL.createObjectURL(media);
      this.video.src = this.url;
      timer = setTimeout(() => controller.abort(), 15000);
      await opened;
      const buffer = media.addSourceBuffer(type);
      const change = action => new Promise((resolve, reject) => {
        const cleanup = () => {
          buffer.removeEventListener('updateend', done);
          buffer.removeEventListener('error', fail);
          controller.signal.removeEventListener('abort', fail);
        };
        const done = () => { cleanup(); resolve(); };
        const fail = () => { cleanup(); reject(new Error('Video buffer closed')); };
        buffer.addEventListener('updateend', done, {once: true});
        buffer.addEventListener('error', fail, {once: true});
        controller.signal.addEventListener('abort', fail, {once: true});
        try { action(); } catch (error) { cleanup(); reject(error); }
      });
      const response = await fetch(`/video/${camera}.mp4`, {cache: 'no-store', signal: controller.signal});
      if (!response.ok || !response.body) throw new Error('Video unavailable');
      const reader = response.body.getReader();
      const presented = () => {
        if (version !== this.version) return;
        this.lastFrameAt = Date.now();
        this.video.hidden = false;
        this.image.hidden = true;
        clearTimeout(this.frameTimer);
        this.frameTimer = setTimeout(fail, 5000);
        this.frameCallback = this.video.requestVideoFrameCallback(presented);
      };
      this.frameCallback = this.video.requestVideoFrameCallback(presented);
      while (version === this.version) {
        const {done, value} = await reader.read();
        if (done) throw new Error('Video ended');
        clearTimeout(timer);
        timer = setTimeout(() => controller.abort(), 7000);
        await change(() => buffer.appendBuffer(value));
        if (buffer.buffered.length) {
          const start = buffer.buffered.start(0);
          const end = buffer.buffered.end(buffer.buffered.length - 1);
          if (this.video.currentTime < start || end - this.video.currentTime > 2.5) {
            this.video.currentTime = Math.max(start, end - 0.6);
          }
          if (start < this.video.currentTime - 10) {
            await change(() => buffer.remove(0, this.video.currentTime - 10));
          }
          // Playback may need the next fragment before its promise resolves.
          if (this.video.paused) this.video.play().catch(fail);
        }
      }
    } catch {
      fail();
    } finally {
      clearTimeout(timer);
      controller.abort();
    }
  }
}
window.VisitorVideo = VisitorVideo;
