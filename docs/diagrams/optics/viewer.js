const diagrams = window.OPTICS_DIAGRAMS;
const topics = document.querySelector('#topics');
let selected = 0;

diagrams.forEach((diagram, index) => {
  const link = document.createElement('a');
  link.href = `#${diagram.slug}`;
  link.textContent = `${String(index).padStart(2, '0')} - ${diagram.title}`;
  topics.append(link);
});

function showDrawing() {
  const requested = decodeURIComponent(location.hash.slice(1));
  selected = Math.max(0, diagrams.findIndex(diagram => diagram.slug === requested));
  const diagram = diagrams[selected];
  document.title = `${diagram.title} - visual atlas`;
  document.querySelector('#count').textContent = `${selected + 1} of ${diagrams.length}`;
  const preview = document.querySelector('#preview');
  preview.src = `previews/${diagram.slug}.svg`;
  preview.alt = `${diagram.title}. ${diagram.subtitle}`;
  document.querySelector('#drawing').href = `drawings/${diagram.slug}.excalidraw`;
  document.querySelector('#svg').href = `previews/${diagram.slug}.svg`;
  document.querySelector('#code').textContent = `code connection - ${diagram.code.replaceAll('; ', ' - ')}`;
  document.querySelector('#previous').disabled = selected === 0;
  document.querySelector('#next').disabled = selected === diagrams.length - 1;
  [...topics.children].forEach((link, index) => {
    if (index === selected) link.setAttribute('aria-current', 'page');
    else link.removeAttribute('aria-current');
  });
  document.querySelector('#stage').scrollTo(0, 0);
}

function move(direction) {
  const index = Math.max(0, Math.min(diagrams.length - 1, selected + direction));
  location.hash = diagrams[index].slug;
}

document.querySelector('#previous').addEventListener('click', () => move(-1));
document.querySelector('#next').addEventListener('click', () => move(1));
document.querySelector('#zoom').addEventListener('click', event => {
  const enlarged = document.querySelector('#stage').classList.toggle('enlarged');
  event.currentTarget.setAttribute('aria-pressed', String(enlarged));
  event.currentTarget.textContent = enlarged ? 'fit diagram' : 'enlarge diagram';
});
window.addEventListener('hashchange', showDrawing);
window.addEventListener('keydown', event => {
  if (event.altKey || event.ctrlKey || event.metaKey || event.shiftKey) return;
  if (event.key === 'ArrowRight') { event.preventDefault(); move(1); }
  if (event.key === 'ArrowLeft') { event.preventDefault(); move(-1); }
});
showDrawing();
