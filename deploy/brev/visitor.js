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
let galleryEnabled = false;
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
  if (document.documentElement.dataset.paaiCameraExtension || galleryEnabled) {
    player.stop();
    return;
  }
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
  if (document.documentElement.dataset.paaiCameraExtension) return;
  const version = generation;
  try {
    const response = await fetch('/api/status', {cache: 'no-store', signal: AbortSignal.timeout(7000)});
    if (!response.ok) throw new Error('Camera unavailable');
    const state = await response.json();
    if (version !== generation) return;
    if (galleryEnabled) {
      const current = state.cameras.map(row => row.frame_id).join(',');
      status.textContent = state.cameras.every(row => row.online) && current !== lastFrame
        ? 'Three live cameras' : 'Cameras reconnecting…';
      lastFrame = current;
      return;
    }
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

// The gateway stays private. This page can submit only attendee messages.
const chatPanel = document.querySelector('#attendee-chat');
const chatStatus = document.querySelector('#chat-status');
const chatHistory = document.querySelector('#chat-history');
const chatMessage = document.querySelector('#chat-message');
const chatSend = document.querySelector('#chat-send');
const chatReset = document.querySelector('#chat-reset');
let pendingOrder;
let lastOrder;
let sending = false;

function cameraGallery() {
  if (galleryEnabled) return;
  galleryEnabled = true;
  cancelRefresh();
  player.stop();
  document.querySelector('main > figure').hidden = true;
  document.querySelector('main > nav').hidden = true;
  const gallery = document.querySelector('#visitor-cameras');
  gallery.hidden = false;
  for (const name of ['kitchen', 'worktop', 'side']) {
    const figure = document.createElement('figure');
    let picture = document.createElement('img');
    picture.alt = `Live ${name} camera`;
    const label = document.createElement('figcaption');
    label.textContent = name[0].toUpperCase() + name.slice(1);
    figure.append(picture, label);
    gallery.append(figure);
    async function frame() {
      if (document.documentElement.dataset.paaiCameraExtension) return;
      if (!document.hidden) {
        try {
          const next = new Image();
          next.alt = `Live ${name} camera`;
          next.src = `/snapshot/${name}.jpg?t=${Date.now()}`;
          await next.decode();
          picture.replaceWith(next);
          picture = next;
          label.textContent = name[0].toUpperCase() + name.slice(1);
        } catch {
          label.textContent = `${name} · reconnecting…`;
        }
      }
      setTimeout(frame, 1000);
    }
    frame();
  }
}

function chatLine(speaker, message) {
  const line = document.createElement('p');
  const name = document.createElement('strong');
  name.textContent = `${speaker}: `;
  line.append(name, document.createTextNode(message));
  chatHistory.append(line);
  while (chatHistory.childElementCount > 30) chatHistory.firstElementChild.remove();
  line.scrollIntoView({block: 'nearest'});
}

async function chatHealth() {
  try {
    const response = await fetch(`/api/chat${pendingOrder ? '?id=' + pendingOrder : ''}`, {
      cache: 'no-store', signal: AbortSignal.timeout(7000),
    });
    if (!response.ok) throw new Error('Chat unavailable');
    const state = await response.json();
    chatPanel.hidden = !state.enabled;
    if (!state.enabled) return;
    if (!state.local_extension) cameraGallery();
    document.querySelector('.note').textContent = 'The demo uses prepared objects in simulation. Grasps can miss. Watch the cameras and send one order at a time.';
    chatSend.disabled = chatReset.disabled = sending || state.busy || !state.ready;
    chatStatus.textContent = state.busy ? 'OpenClaw is working. Watch the cameras…'
      : state.ready ? 'OpenClaw connected · cascade-demo' : 'OpenClaw is reconnecting…';
    if (state.order && state.order.status !== 'running' && state.order.id !== lastOrder) {
      lastOrder = state.order.id;
      pendingOrder = undefined;
      chatLine('Robot', state.order.message);
    }
  } catch {
    chatSend.disabled = chatReset.disabled = true;
    chatStatus.textContent = 'OpenClaw is reconnecting…';
  } finally {
    setTimeout(chatHealth, 1500);
  }
}

async function sendMessage(message) {
  if (sending || !message.trim()) return;
  sending = true;
  chatSend.disabled = chatReset.disabled = true;
  try {
    const response = await fetch('/api/chat', {method: 'POST',
      headers: {'Content-Type': 'application/json'}, body: JSON.stringify({message}),
      signal: AbortSignal.timeout(10000),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || 'The message was not accepted');
    pendingOrder = result.id;
    chatMessage.value = '';
    chatLine('You', message);
    chatStatus.textContent = 'OpenClaw is working. Watch the cameras…';
  } catch (error) {
    chatLine('Chat', error.message);
  } finally {
    sending = false;
  }
}

document.querySelector('#chat-form').addEventListener('submit', event => {
  event.preventDefault();
  sendMessage(chatMessage.value);
});
chatReset.addEventListener('click', () => sendMessage("Let's start over."));
chatHealth();
