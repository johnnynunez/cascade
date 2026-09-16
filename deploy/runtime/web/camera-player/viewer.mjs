export const CAMERAS = Object.freeze(['kitchen', 'worktop', 'side']);

export class CameraPlayer {
  constructor(video, { Reader, origin, now = () => performance.now(),
    timers = globalThis, isVisible = () => true, onChange = () => {} }) {
    Object.assign(this, { video, Reader, origin, now, timers, isVisible, onChange });
    this.reader = null;
    this.generation = 0;
    this.frameCallback = null;
    this.playbackGeneration = 0;
    this.timer = null;
    this.camera = 'worktop';
    this.retries = 0;
    this.phase = 'closed';
    this.clearMeasurements();
  }

  clearMeasurements() {
    this.sample = null;
    this.presentedFrames = null;
    this.fps = null;
    this.lastFrame = null;
    this.dropBaseline = this.video.getVideoPlaybackQuality?.().droppedVideoFrames ?? null;
  }

  detachVideo() {
    this.playbackGeneration++;
    if (this.frameCallback !== null) this.video.cancelVideoFrameCallback?.(this.frameCallback);
    this.frameCallback = null;
    this.video.pause();
    this.video.srcObject?.getTracks().forEach((track) => track.stop());
    this.video.srcObject = null;
    this.clearMeasurements();
  }

  select(camera, retry = false) {
    if (!CAMERAS.includes(camera)) throw new Error('Unknown camera');
    if (this.reader && camera === this.camera && !retry) return;
    const retries = retry ? this.retries + 1 : 0;
    this.close();
    this.camera = camera;
    this.retries = retries;
    this.phase = 'connecting';
    this.startedAt = this.now();
    const generation = ++this.generation;
    this.reader = new this.Reader({
      url: new URL(`/camera-video/${camera}/whep`, this.origin).href,
      videoOnly: true,
      onTrack: (event) => {
        if (generation !== this.generation || event.track.kind !== 'video') return;
        this.detachVideo();
        this.video.srcObject = event.streams[0] ?? new MediaStream([event.track]);
        this.clearMeasurements();
        this.watchFrames(generation);
        this.play(generation);
      },
      onError: () => {
        if (generation !== this.generation) return;
        this.detachVideo();
        this.phase = 'reconnecting';
        this.report();
      },
      onRetry: () => {
        if (generation !== this.generation) return;
        this.retries++;
        this.startedAt = this.now();
        this.phase = 'connecting';
        this.report();
      },
    });
    this.timer = this.timers.setInterval(() => this.tick(), 1000);
    this.report();
  }

  play(generation = this.generation) {
    this.phase = 'connecting';
    this.startedAt = this.now();
    const playbackGeneration = this.playbackGeneration;
    Promise.resolve(this.video.play()).then(() => {
      if (generation !== this.generation || playbackGeneration !== this.playbackGeneration || !this.reader) return;
      if (!this.video.requestVideoFrameCallback) this.phase = 'playing';
      this.report();
    }).catch(() => {
      if (generation !== this.generation || playbackGeneration !== this.playbackGeneration || !this.reader) return;
      this.phase = 'blocked';
      this.report();
    });
  }

  watchFrames(generation) {
    if (!this.video.requestVideoFrameCallback) return;
    const playbackGeneration = this.playbackGeneration;
    this.frameCallback = this.video.requestVideoFrameCallback((now, metadata) => {
      if (generation !== this.generation || playbackGeneration !== this.playbackGeneration || !this.reader) return;
      this.frameCallback = null;
      if (Number.isFinite(metadata.presentationTime) && metadata.presentationTime <= now) {
        this.lastFrame = metadata.presentationTime;
      }
      if (Number.isFinite(metadata.presentedFrames)) {
        this.presentedFrames = metadata.presentedFrames;
        this.sample ??= { time: now, frames: metadata.presentedFrames };
      }
      this.phase = 'live';
      this.watchFrames(generation);
    });
  }

  tick() {
    if (!this.reader) return;
    const now = this.now();
    if (this.sample && now > this.sample.time) {
      this.fps = this.presentedFrames >= this.sample.frames
        ? 1000 * (this.presentedFrames - this.sample.frames) / (now - this.sample.time) : null;
      this.sample = { time: now, frames: this.presentedFrames };
    }
    if (this.lastFrame === null && this.phase === 'connecting' && this.isVisible()
        && now - this.startedAt > 25000) {
      this.select(this.camera, true);
      return;
    }
    if (this.lastFrame !== null && this.isVisible() && this.phase !== 'blocked') {
      if (now - this.lastFrame > 12000) {
        this.select(this.camera, true);
        return;
      }
      if (now - this.lastFrame > 3000) this.phase = 'stalled';
    }
    this.report();
  }

  snapshot() {
    const now = this.now();
    const ageMs = this.lastFrame === null ? null : Math.max(0, now - this.lastFrame);
    const dropped = this.video.getVideoPlaybackQuality?.().droppedVideoFrames;
    if (dropped < this.dropBaseline) this.dropBaseline = 0;
    const droppedFrames = Number.isFinite(dropped) && Number.isFinite(this.dropBaseline)
      ? Math.max(0, dropped - this.dropBaseline) : null;
    return { camera: this.camera, phase: this.phase, retries: this.retries,
      fps: this.fps, ageMs, droppedFrames };
  }

  report() { this.onChange(this.snapshot()); }

  close() {
    this.generation++;
    this.reader?.close();
    this.reader = null;
    if (this.timer !== null) this.timers.clearInterval(this.timer);
    this.timer = null;
    this.detachVideo();
    this.phase = 'closed';
  }
}

export function mountCameraViewer(document, window) {
  const video = document.querySelector('#camera');
  const message = document.querySelector('#video-message');
  const connection = document.querySelector('#connection');
  const quality = document.querySelector('#quality');
  const reconnect = document.querySelector('#reconnect');
  const buttons = [...document.querySelectorAll('[data-camera]')];
  if (!window.RTCPeerConnection || !window.MediaMTXWebRTCReader) {
    message.textContent = 'Live video is unavailable in this browser. Open the image view below.';
    connection.textContent = 'Video unavailable';
    reconnect.disabled = true;
    buttons.forEach((button) => { button.disabled = true; });
    return null;
  }
  const player = new CameraPlayer(video, {
    Reader: window.MediaMTXWebRTCReader, origin: window.location.origin,
    now: () => window.performance.now(), timers: window,
    isVisible: () => document.visibilityState !== 'hidden',
    onChange: ({ camera, phase, fps, ageMs, retries, droppedFrames }) => {
      const name = camera[0].toUpperCase() + camera.slice(1);
      const labels = { closed: 'Closed', connecting: `Connecting to ${name}…`,
        reconnecting: 'Reconnecting…', live: `${name} · Live`, playing: `${name} · Playing`,
        stalled: 'Waiting for a fresh frame…', blocked: 'Select Play video to continue.' };
      connection.textContent = labels[phase];
      message.textContent = labels[phase];
      message.hidden = phase === 'live' || phase === 'playing';
      video.dataset.live = String(message.hidden);
      video.setAttribute('aria-label', `${name} camera`);
      buttons.forEach((button) => button.setAttribute('aria-pressed', String(button.dataset.camera === camera)));
      reconnect.textContent = phase === 'blocked' ? 'Play video' : 'Reconnect';
      const age = ageMs === null ? (phase === 'playing' ? 'frame timing unavailable' : 'no frame yet')
        : `${(ageMs / 1000).toFixed(1)} s since frame`;
      quality.textContent = `${fps === null ? '—' : fps.toFixed(1)} FPS · ${age} · ${retries} retries · ${droppedFrames ?? '—'} dropped`;
    },
  });
  let camera = new URL(window.location.href).searchParams.get('camera');
  if (!CAMERAS.includes(camera)) camera = 'worktop';
  buttons.forEach((button) => button.addEventListener('click', () => {
    player.select(button.dataset.camera);
    const url = new URL(window.location.href);
    url.searchParams.set('camera', player.camera);
    window.history.replaceState(null, '', url);
  }));
  reconnect.addEventListener('click', () => {
    if (player.phase === 'blocked') player.play();
    else player.select(player.camera, true);
  });
  window.addEventListener('pagehide', () => player.close());
  window.addEventListener('pageshow', (event) => {
    if (event.persisted) player.select(player.camera);
  });
  player.select(camera);
  return player;
}

if (typeof window !== 'undefined' && typeof document !== 'undefined') mountCameraViewer(document, window);
