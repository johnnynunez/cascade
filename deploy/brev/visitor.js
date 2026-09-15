const camera = document.querySelector('#camera');
const status = document.querySelector('#status');
const caption = document.querySelector('#caption');
let selected = 'worktop';
let selectedLabel = 'Worktop';
let generation = 0;
let timer;
let pending;
let displayedURL;
let displayedCamera;
let decodedAt = 0;
let frameHealthy = false;
let lastFrame;
let videoEnabled = false;
let nextVideoAttempt = 0;
const player = new window.VisitorVideo(document.querySelector('#video'), camera, () => {
  nextVideoAttempt = Date.now() + 5000;
  refresh();
});

caption.textContent = 'Connecting to Worktop camera…';

function cancelRefresh() {
  clearTimeout(timer);
  pending?.abort();
  pending = undefined;
}

async function refresh() {
  if (document.hidden || pending) return;
  if (player.playing) {
    timer = setTimeout(refresh, 1000);
    return;
  }
  const version = generation;
  const name = selected;
  const label = selectedLabel;
  const controller = new AbortController();
  pending = controller;
  const timeout = setTimeout(() => controller.abort(), 7000);
  let nextURL;
  let delay = 350;
  try {
    const response = await fetch(`/snapshot/${name}.jpg?t=${Date.now()}`, {
      cache: 'no-store', signal: controller.signal,
    });
    if (!response.ok) throw new Error('Camera unavailable');
    const blob = await response.blob();
    if (controller.signal.aborted || version !== generation) return;
    nextURL = URL.createObjectURL(blob);
    const frame = new Image();
    frame.src = nextURL;
    await frame.decode();
    if (controller.signal.aborted || version !== generation) return;

    const previousURL = displayedURL;
    camera.src = nextURL;
    displayedURL = nextURL;
    nextURL = undefined;
    displayedCamera = name;
    decodedAt = Date.now();
    frameHealthy = true;
    camera.alt = `${label} camera`;
    caption.textContent = `${label} · live camera`;
    if (previousURL) URL.revokeObjectURL(previousURL);
  } catch {
    if (version === generation) {
      frameHealthy = false;
      status.textContent = 'Camera reconnecting…';
    }
    delay = 2000;
  } finally {
    clearTimeout(timeout);
    if (nextURL) URL.revokeObjectURL(nextURL);
    if (pending === controller) {
      pending = undefined;
      timer = setTimeout(refresh, delay);
    }
  }
}

document.querySelectorAll('[data-camera]').forEach(button => {
  button.addEventListener('click', () => {
    generation += 1;
    cancelRefresh();
    player.stop();
    selected = button.dataset.camera;
    selectedLabel = button.textContent.trim();
    lastFrame = undefined;
    frameHealthy = false;
    status.textContent = `Loading ${selectedLabel} camera…`;
    document.querySelectorAll('[data-camera]').forEach(item => item.setAttribute('aria-pressed', String(item === button)));
    refresh();
    if (videoEnabled) player.start(selected);
  });
});
document.addEventListener('visibilitychange', refresh);
window.addEventListener('pagehide', () => {
  generation += 1;
  cancelRefresh();
  player.stop();
  if (displayedURL) URL.revokeObjectURL(displayedURL);
  displayedURL = undefined;
});
window.addEventListener('pageshow', refresh);

async function updateStatus() {
  const version = generation;
  try {
    const response = await fetch('/api/status', {cache: 'no-store', signal: AbortSignal.timeout(7000)});
    if (!response.ok) throw new Error('Camera unavailable');
    const state = await response.json();
    if (version !== generation) return;
    videoEnabled = state.video?.enabled === true;
    if (videoEnabled && !player.camera && Date.now() >= nextVideoAttempt) player.start(selected);
    const row = state.cameras.find(item => item.name === selected);
    const ready = player.playing || (frameHealthy && displayedCamera === selected && Date.now() - decodedAt < 7000);
    const advancing = row?.online && row.frame_id != null && row.frame_id !== lastFrame;
    status.textContent = ready && advancing ? (player.playing ? 'Live · H.264 video' : `Live · frame ${row.frame_id}`)
      : displayedCamera !== selected ? `Loading ${selectedLabel} camera…`
      : !frameHealthy ? 'Camera reconnecting…' : 'Waiting for a fresh frame…';
    lastFrame = row?.frame_id;
  } catch {
    if (version === generation) status.textContent = 'Camera reconnecting…';
  } finally {
    setTimeout(updateStatus, 3000);
  }
}
refresh();
updateStatus();

window.addEventListener('offline', () => {
  player.stop();
  nextVideoAttempt = Infinity;
  status.textContent = 'Camera reconnecting…';
});
window.addEventListener('online', () => {
  nextVideoAttempt = 0;
  refresh();
});
