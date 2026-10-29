// Theme: light / dark tokens live in style.css; this module reads them for Plotly and three.js,
// and runs the theme toggle (auto -> light -> dark).

import { renderPlotMath, watchPlotMath } from './math.js';

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
  const family = token('--font-ui') || '"SF Mono", monospace';
  const grid = token('--grid');
  const axis = token('--axis');
  const ax = {
    gridcolor: grid, zerolinecolor: axis, linecolor: axis, tickcolor: axis,
    tickfont: { color: ink2, size: 12 }, title: { font: { color: ink2, size: 13 } },
    showline: true, ticks: 'outside', ticklen: 4,
  };
  return {
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: token('--surface-2'),
    font: { family, size: 13, color: ink2 },
    hoverlabel: {
      bgcolor: token('--surface-2'), bordercolor: token('--axis'),
      font: { color: token('--text-primary'), size: 13, family },
    },
    legend: { font: { color: ink2, size: 12 }, bgcolor: 'rgba(0,0,0,0)' },
    ax,
  };
}

export const PLOT_CONFIG = {
  displaylogo: false,
  responsive: false,
  modeBarButtonsToRemove: ['select2d', 'lasso2d', 'autoScale2d', 'toggleSpikelines', 'hoverClosestCartesian',
    'hoverCompareCartesian'],
};

// Plotly axis object from the theme's axis base plus overrides.
export function axis(base, extra) {
  return { ...base.ax, ...extra, title: { ...base.ax.title, ...(extra && extra.title) } };
}

// Plotly rotates native y-axis titles. Keep the same quantity and units in a
// horizontal heading above its plotting area instead. Axis-domain coordinates
// keep the heading attached to the correct subplot while panning or zooming.
export function horizontalYTitles(layout) {
  const out = { ...layout, margin: { ...layout.margin }, annotations: [...(layout.annotations || [])] };
  let hasHeading = false;
  for (const [key, value] of Object.entries(layout)) {
    if (!/^yaxis\d*$/.test(key)) continue;
    const title = typeof value.title === 'string' ? { text: value.title } : (value.title || {});
    out[key] = { ...value, tickangle: 0, title: { ...title, text: '' } };
    if (!title.text || value.visible === false) continue;
    const yref = `y${key.slice(5)} domain`;
    const xref = /^x\d*$/.test(value.anchor || '') ? `${value.anchor} domain` : 'x domain';
    out.annotations.push({ name: `horizontal-${key}`, xref, x: 0, yref, y: 1,
      xanchor: 'left', yanchor: 'bottom', yshift: 9, showarrow: false, textangle: 0,
      text: title.text, font: { ...layout.font, size: 12, ...title.font } });
    hasHeading = true;
  }
  if (hasHeading) out.margin.t = Math.max(out.margin.t || 0, 36);
  return out;
}

// Serialize updates on each chart. Hidden charts keep their last valid size;
// Plotly's global resize handler can otherwise race a redraw at zero width.
const plotQueue = new WeakMap();
const plotSizes = new WeakMap();
function queuePlot(gd, update) {
  const previous = plotQueue.get(gd);
  const next = previous ? previous.catch(() => {}).then(update) : Promise.resolve(update());
  plotQueue.set(gd, next);
  return next;
}

function watchPlotSize(gd) {
  let state = plotSizes.get(gd);
  if (state) { state.width = gd.clientWidth; return; }
  state = { width: gd.clientWidth, frame: 0 };
  const observer = new ResizeObserver(() => {
    if (!gd.isConnected) { observer.disconnect(); cancelAnimationFrame(state.frame); return; }
    const width = gd.clientWidth;
    if (!gd.offsetParent || !width || Math.abs(width - state.width) < 2) return;
    state.width = width;
    cancelAnimationFrame(state.frame);
    state.frame = requestAnimationFrame(() => {
      relayoutPlot(gd, { width: gd.clientWidth }).catch((error) => console.error('Chart resize failed', error));
    });
  });
  observer.observe(gd);
  plotSizes.set(gd, state);
}

export function relayoutPlot(gd, update) {
  return queuePlot(gd, () => {
    if (!gd.isConnected || !gd.offsetParent || !gd.clientWidth || !gd.layout) return;
    // Read width when the queued operation runs, not when resize was reported.
    const changes = 'width' in update ? { ...update, width: gd.clientWidth } : update;
    return Plotly.relayout(gd, changes);
  });
}

export function restylePlot(gd, update, indices) {
  return queuePlot(gd, () => {
    if (!gd.isConnected || !gd.offsetParent || !gd.clientWidth || !gd.data) return;
    return Plotly.restyle(gd, update, indices);
  });
}

export function renderPlot(gd, data, layout, config = PLOT_CONFIG) {
  return queuePlot(gd, () => {
    if (!gd.isConnected || !gd.offsetParent || !gd.clientWidth) return;
    const bar = (colorbar) => ({ ...colorbar, title: {
      ...(typeof colorbar.title === 'string' ? { text: colorbar.title } : colorbar.title), side: 'top',
    } });
    const traces = data.map((trace) => ({ ...trace,
      ...(trace.colorbar ? { colorbar: bar(trace.colorbar) } : {}),
      ...(trace.marker?.colorbar ? { marker: { ...trace.marker, colorbar: bar(trace.marker.colorbar) } } : {}),
    }));
    return Plotly.react(gd, traces, horizontalYTitles({ ...layout, width: gd.clientWidth }), { ...config, responsive: false }).then((result) => {
      watchPlotMath(gd);
      renderPlotMath(gd);
      watchPlotSize(gd);
      return result;
    });
  });
}
