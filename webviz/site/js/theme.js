// Theme: light / dark tokens live in style.css; this module reads them for Plotly and three.js,
// and runs the theme toggle (auto -> light -> dark).

const listeners = new Set();
const media = window.matchMedia('(prefers-color-scheme: dark)');

export function token(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

export function isDark() {
  const forced = document.documentElement.dataset.theme;
  if (forced === 'dark') return true;
  if (forced === 'light') return false;
  return media.matches;
}

export function onThemeChange(fn) { listeners.add(fn); }

function emit() { for (const fn of listeners) fn(); }

media.addEventListener('change', () => { if (!document.documentElement.dataset.theme) emit(); });

const MODES = ['auto', 'light', 'dark'];

export function initThemeToggle(button) {
  let mode = 'auto';
  try { mode = localStorage.getItem('webviz-theme') || 'auto'; } catch (e) { /* storage blocked */ }
  if (!MODES.includes(mode)) mode = 'auto';
  const apply = () => {
    if (mode === 'auto') delete document.documentElement.dataset.theme;
    else document.documentElement.dataset.theme = mode;
    button.textContent = `Theme: ${mode}`;
    button.setAttribute('aria-label', `Colour theme: ${mode}. Click to change.`);
  };
  apply();
  button.addEventListener('click', () => {
    mode = MODES[(MODES.indexOf(mode) + 1) % MODES.length];
    try { localStorage.setItem('webviz-theme', mode); } catch (e) { /* storage blocked */ }
    apply();
    emit();
  });
}

// Common Plotly layout pieces in the current theme.
export function plotBase() {
  const ink2 = token('--text-secondary');
  const grid = token('--grid');
  const axis = token('--axis');
  const ax = {
    gridcolor: grid, zerolinecolor: axis, linecolor: axis, tickcolor: axis,
    tickfont: { color: ink2, size: 11 }, title: { font: { color: ink2, size: 12 } },
    showline: true, ticks: 'outside', ticklen: 4,
  };
  return {
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: token('--surface-1'),
    font: { family: 'system-ui, -apple-system, "Segoe UI", sans-serif', size: 12, color: ink2 },
    hoverlabel: {
      bgcolor: token('--surface-2'), bordercolor: token('--axis'),
      font: { color: token('--text-primary'), size: 12, family: 'system-ui, -apple-system, "Segoe UI", sans-serif' },
    },
    legend: { font: { color: ink2, size: 11 }, bgcolor: 'rgba(0,0,0,0)' },
    ax,
  };
}

export const PLOT_CONFIG = {
  displaylogo: false,
  responsive: true,
  modeBarButtonsToRemove: ['select2d', 'lasso2d', 'autoScale2d', 'toggleSpikelines', 'hoverClosestCartesian',
    'hoverCompareCartesian'],
};

// Plotly axis object from the theme's axis base plus overrides.
export function axis(base, extra) {
  return { ...base.ax, ...extra, title: { ...base.ax.title, ...(extra && extra.title) } };
}
