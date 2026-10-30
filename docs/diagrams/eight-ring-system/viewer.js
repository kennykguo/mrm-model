const image = document.querySelector('#drawing');
const stage = document.querySelector('#stage');
const nav = document.querySelector('nav');
let scale = 1, shiftX = 0, shiftY = 0, active = 'full', drag;

// coordinates refer to the native drawing, before the SVG's export padding.
const views = {
  controller: [2100, 130, 1200, 460],
  drive: [930, 560, 3470, 1160],
  bank: [925, 1260, 3490, 780],
  monitor: [80, 1780, 4300, 830],
  thermal: [80, 2620, 4420, 590],
  receiver: [4460, 990, 1060, 1400],
  detail: [4520, 2530, 1100, 890]
};

function paint() {
  image.style.transform = `translate(${shiftX}px,${shiftY}px) scale(${scale})`;
}

function fit(name) {
  active = name;
  const rect = views[name];
  const [x, y, width, height] = rect ? [rect[0] - 35, rect[1] - 5, rect[2], rect[3]] : [0, 0, image.naturalWidth, image.naturalHeight];
  scale = Math.min((stage.clientWidth - 48) / width, (stage.clientHeight - 48) / height);
  shiftX = (stage.clientWidth - width * scale) / 2 - x * scale;
  shiftY = (stage.clientHeight - height * scale) / 2 - y * scale;
  document.querySelectorAll('[data-view]').forEach(button => button.setAttribute('aria-pressed', String(button.dataset.view === name)));
  paint();
}

function zoom(ratio, x = stage.clientWidth / 2, y = stage.clientHeight / 2) {
  const next = Math.min(4, Math.max(0.08, scale * ratio));
  const factor = next / scale;
  shiftX = x - (x - shiftX) * factor;
  shiftY = y - (y - shiftY) * factor;
  scale = next;
  paint();
}

function layout() {
  stage.style.top = `${nav.getBoundingClientRect().bottom}px`;
  if (image.naturalWidth) fit(active);
}

document.querySelectorAll('[data-view]').forEach(button => button.addEventListener('click', () => fit(button.dataset.view)));
document.querySelector('#minus').addEventListener('click', () => zoom(1 / 1.35));
document.querySelector('#plus').addEventListener('click', () => zoom(1.35));
stage.addEventListener('wheel', event => {
  event.preventDefault();
  const rect = stage.getBoundingClientRect();
  zoom(Math.exp(-event.deltaY * 0.0018), event.clientX - rect.left, event.clientY - rect.top);
}, { passive: false });
stage.addEventListener('pointerdown', event => {
  drag = { x: event.clientX, y: event.clientY, left: shiftX, top: shiftY };
  stage.setPointerCapture(event.pointerId);
  stage.classList.add('dragging');
});
stage.addEventListener('pointermove', event => {
  if (!drag) return;
  shiftX = drag.left + event.clientX - drag.x;
  shiftY = drag.top + event.clientY - drag.y;
  paint();
});
function endDrag() { drag = undefined; stage.classList.remove('dragging'); }
stage.addEventListener('pointerup', endDrag);
stage.addEventListener('pointercancel', endDrag);
window.addEventListener('keydown', event => {
  if (event.ctrlKey || event.metaKey || event.altKey) return;
  if (event.key.toLowerCase() === 'f') fit('full');
  if (event.key === '+' || event.key === '=') zoom(1.35);
  if (event.key === '-') zoom(1 / 1.35);
});
window.addEventListener('resize', layout);
image.addEventListener('load', layout);
if (image.complete && image.naturalWidth) layout();
