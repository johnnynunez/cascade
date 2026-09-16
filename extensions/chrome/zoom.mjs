// Resize the embedded panel, or move the existing side-panel UI into a large
// window. The image element and its stream reader stay alive in both cases.
export function cameraZoom({viewer, companion, embedded, resizeEmbedded, window, document}) {
  let enlarged = false, popup = null, placeholder = null;
  const hint = viewer.querySelector('.zoom-hint');

  function render() {
    companion.classList.toggle('enlarged', enlarged);
    viewer.setAttribute('aria-expanded', String(enlarged));
    viewer.setAttribute('aria-label', enlarged ? 'Return camera to panel' : 'Enlarge camera');
    viewer.title = enlarged ? 'Return camera to panel (Escape)' : 'Enlarge camera';
    if (hint) hint.textContent = enlarged ? 'Click or Esc to return' : 'Click to enlarge';
  }

  function restore({focus = true} = {}) {
    if (!enlarged) return;
    enlarged = false;
    if (embedded) resizeEmbedded(false);
    if (placeholder) {
      placeholder.replaceWith(companion);
      placeholder = null;
    }
    const previous = popup;
    popup = null;
    render();
    if (previous && !previous.closed) previous.close();
    if (focus) {
      window.focus();
      viewer.focus({preventScroll:true});
    }
  }

  function toggle() {
    if (enlarged) { restore(); return; }
    if (embedded) {
      enlarged = true;
      render();
      resizeEmbedded(true);
      viewer.focus({preventScroll:true});
      return;
    }
    // Calling window.open directly from the tile's click keeps Chrome's user
    // gesture. No camera URL, credential, or additional host grant is needed.
    enlarged = true;
    try {
      const width = Math.min(1200, Math.max(640, window.screen.availWidth - 80));
      const height = Math.min(900, Math.max(480, window.screen.availHeight - 80));
      popup = window.open('', '_blank', `popup,width=${width},height=${height}`);
      if (!popup || popup.closed) throw new Error('Camera window was blocked');
      const target = popup.document;
      target.title = 'PAAI · Live camera';
      target.documentElement.lang = 'en';
      const style = target.createElement('link');
      style.rel = 'stylesheet';
      style.href = new URL('ui.css', document.baseURI).href;
      target.head.append(style);
      target.body.className = 'panel side camera-window';
      placeholder = document.createElement('div');
      placeholder.className = 'zoom-placeholder';
      const message = document.createElement('p');
      message.textContent = 'The camera is open in a larger window.';
      const back = document.createElement('button');
      back.textContent = 'Return camera to panel';
      back.addEventListener('click', () => restore());
      placeholder.append(message, back);
      companion.replaceWith(placeholder);
      target.body.append(companion);
      popup.addEventListener('pagehide', () => restore(), {once:true});
      popup.addEventListener('keydown', event => {
        if (event.key === 'Escape') { event.preventDefault(); restore(); }
      });
      render();
      popup.focus();
      viewer.focus({preventScroll:true});
    } catch {
      restore();
      throw new Error('Chrome could not open the camera window. Try Show in chat.');
    }
  }

  render();
  return {toggle, restore, get active() { return enlarged; }};
}
