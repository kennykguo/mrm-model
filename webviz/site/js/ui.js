// Shared pieces of the device panels, the bank table and the metrics section: number formats, stat tiles, a
// stacked time plot (one y-axis per panel, shared time axis, a cursor at the scrubber) and the categorical colours.
// Nothing here adds physics: every number shown comes from the data files (model output).

import { plotBase, axis, PLOT_CONFIG, token, renderPlot } from './theme.js';
import { hexA, fmtNum, shortEvent } from './traces.js';

export { fmtNum, hexA };
export const K0 = 273.15;

// categorical slots 1..8 (dataviz reference palette; light / dark values in style.css)
export const cat = (i) => token(`--cat-${(i % 8) + 1}`);

export function fmt(v, unit = '', d = null) {
  if (v == null || (typeof v === 'number' && !Number.isFinite(v))) return '–';
  const s = d == null ? fmtNum(v) : v.toFixed(d);
  return unit ? `${s} ${unit}` : s;
}

// a duration in seconds as microseconds with 3 significant digits ('–' for none)
export const formatMicroseconds = (s) => (s == null ? '–' : `${+(s * 1e6).toPrecision(3)} µs`);

export function sgn(v, d = 2) {
  if (v == null || !Number.isFinite(v)) return '–';
  return `${v > 0 ? '+' : v < 0 ? '−' : ''}${Math.abs(v).toFixed(d)}`;
}

// scientific format for quantities that span many decades (BER)
export function sci(v) {
  if (v == null || !Number.isFinite(v)) return '–';
  if (v === 0) return '0 (below the float range)';
  const e = Math.floor(Math.log10(Math.abs(v)));
  const m = v / 10 ** e;
  return `${m.toFixed(2)}×10^${e}`;
}

export const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

// provenance tag (assumed / inferred / ...) next to a value
export function prov(p) {
  if (!p) return '';
  return ` <span class="prov prov-${esc(p)}" title="provenance in the preset: ${esc(p)}">${esc(p)}</span>`;
}

// ------------------------------------------------------------------ stat tiles
// tiles: [{label, note, get(S, r, f) -> {v: string, flag?: string}}]
export class Tiles {
  constructor(host) { this.host = host; this.items = []; }

  build(defs, { rings = 0, selected = 0 } = {}) {
    this.defs = defs;
    this.rings = rings;
    this.host.classList.toggle('comparison', rings > 0);
    if (rings) {
      this.host.innerHTML = '<div class="tablewrap"><table class="summary snapshot"><thead><tr><th scope="col">At the scrubber frame</th>' +
        Array.from({ length: rings }, (_, r) => `<th scope="col" class="${r === selected ? 'selc' : ''}"><span class="sw" style="background:${cat(r)}"></span>R${r}</th>`).join('') +
        '</tr></thead><tbody>' + defs.map((d) => `<tr><th scope="row"><div class="mn">${esc(d.label)}</div><div class="md">${esc(d.note || '')}</div></th>` +
          Array.from({ length: rings }, (_, r) => `<td class="${r === selected ? 'selc' : ''}"><div class="tv"></div><div class="tf"></div></td>`).join('') + '</tr>').join('') + '</tbody></table></div>';
      this.items = [...this.host.querySelectorAll('tbody tr')].map((tr) => [...tr.querySelectorAll('td')].map((td) => ({ v: td.querySelector('.tv'), f: td.querySelector('.tf') })));
      return;
    }
    this.host.innerHTML = defs.map((d, i) => `<div class="tile" data-i="${i}"><div class="tl">${esc(d.label)}</div>` +
      '<div class="tv">–</div><div class="tf"></div>' + `<div class="tn">${d.note || ''}</div></div>`).join('');
    this.items = [...this.host.querySelectorAll('.tile')].map((el) => ({ v: el.querySelector('.tv'), f: el.querySelector('.tf') }));
  }

  update(S, r, f) {
    this.defs.forEach((d, i) => {
      const set = (it, ring) => {
        let out;
        try { out = d.get(S, ring, f); } catch (e) { out = { v: '–' }; }
        if (typeof out === 'string') out = { v: out };
        // LaTeX adds child nodes; cache the source value to avoid rebuilding unchanged cells.
        if (it.value !== out.v) { it.value = out.v; it.v.textContent = out.v; }
        const fl = out.flag || '';
        if (it.flag !== fl) { it.flag = fl; it.f.textContent = fl; }
      };
      if (this.rings) this.items[i].forEach((it, ring) => set(it, ring));
      else set(this.items[i], r);
    });
  }
}

// ------------------------------------------------------------------ stacked time plot
// panels: [{title, traces: [{y, x?, name, color?, dash?, width?, shape?, band?: [lo, hi], mode?, marker?}],
//           question?, description?, log?, hlines?: [{y, dash?, color?, label?}], range?}]
export class TimeStack {
  constructor(wrap, { onSeek, height = 120 } = {}) {
    this.wrap = wrap;
    this.captionLayer = document.createElement('div');
    this.captionLayer.className = 'plot-captions';
    wrap.appendChild(this.captionLayer);
    this.gd = document.createElement('div');
    this.gd.className = 'plot';
    wrap.appendChild(this.gd);
    this.cursor = document.createElement('div');
    this.cursor.className = 'cursor';
    wrap.appendChild(this.cursor);
    this.onSeek = onSeek;
    this.rowH = Math.max(height, parseFloat(token('--plot-row-height')) || 200);
    this.t = 0;
    this.bound = false;
    this.width = wrap.clientWidth;
    this.observer = new ResizeObserver(() => {
      const width = wrap.clientWidth;
      if (width > 0 && Math.abs(width - this.width) > 1 && this.lastRender) {
        this.width = width;
        const { S, panels, options } = this.lastRender;
        this.render(S, panels, { ...options, timeRange: this.gd.layout?.xaxis?.range });
      }
    });
    this.observer.observe(wrap);
  }

  // opts: legend, ring (event markers of that ring only), events (default true), shapes (extra Plotly shapes)
  render(S, panels, { legend = false, ring = null, events = true, shapes = [], timeRange = null } = {}) {
    this.lastRender = { S, panels, options: { legend, ring, events, shapes, timeRange } };
    const b = plotBase();
    const muted = token('--text-muted');
    const n = panels.length;
    this.captionLayer.innerHTML = panels.map((p) => `<section class="plot-caption"><div class="plot-caption-heading">` +
      (p.question ? `<h3>${esc(p.question)}</h3>` : '') + `<span>${esc(p.title)}</span></div>` +
      (p.description ? `<p>${esc(p.description)}</p>` : '') + '</section>').join('');
    this.captions = [...this.captionLayer.children];
    window.MathText?.refresh(this.captionLayer);
    this.captionHeights = this.captions.map((el) => Math.ceil(el.getBoundingClientRect().height) || 24);
    const rowH = this.rowH;
    // Allocate actual pixels for each caption rather than squeezing text into a
    // fixed subplot gap. Wrapped captions on a phone get the space they need.
    const gaps = this.captionHeights.map((height) => height + 72);
    const plotH = n * rowH + gaps.slice(1).reduce((a, v) => a + v, 0);
    let fromTop = 0;
    this.domains = panels.map((_, i) => {
      const domain = [1 - (fromTop + rowH) / plotH, 1 - fromTop / plotH];
      fromTop += rowH + (gaps[i + 1] || 0);
      return domain;
    });
    const dom = (i) => this.domains[i];
    // Keep event labels in separate lanes when nearby events would collide.
    const plotWidth = Math.max(160, this.wrap.clientWidth - 92);
    const timeBounds = timeRange || [0, S.m.duration_s * 1e3];
    const lanes = [];
    const eventLabels = new Map();
    if (events) {
      for (const ev of S.scn.events) {
        const tl = ev.t_s * 1e3;
        if (tl <= 0 || (ev.label || '').startsWith('metrics start') || (ev.ring != null && ev.ring !== ring)) continue;
        if (tl < timeBounds[0] || tl > timeBounds[1]) continue;
        const text = shortEvent(ev, ring);
        if (!text) continue;
        const px = (tl - timeBounds[0]) / (timeBounds[1] - timeBounds[0]) * plotWidth;
        const width = Math.min(plotWidth, text.length * 7.2 + 16);
        const left = Math.max(0, Math.min(plotWidth - width, px - width / 2));
        let lane = lanes.findIndex((right) => right + 16 < left);
        if (lane < 0) { lane = lanes.length; lanes.push(-Infinity); }
        lanes[lane] = left + width;
        eventLabels.set(ev, { text, lane, anchor: px < width / 2 ? 'left' : px > plotWidth - width / 2 ? 'right' : 'center' });
      }
    }
    const top = this.captionHeights[0] + 32 + lanes.length * 26;
    const bottom = legend ? 128 : 56;
    const data = [];
    const t = S.tMs;
    const layout = {
      paper_bgcolor: b.paper_bgcolor, plot_bgcolor: b.plot_bgcolor, font: b.font, hoverlabel: b.hoverlabel,
      margin: { l: 72, r: 20, t: top, b: bottom }, height: plotH + top + bottom,
      hovermode: 'x', dragmode: 'zoom', showlegend: legend,
      legend: { ...b.legend, orientation: 'h', x: 0, y: -72 / plotH, yanchor: 'top' },
      xaxis: axis(b, { anchor: `y${n > 1 ? n : ''}`, range: timeRange || [0, S.m.duration_s * 1e3], title: { text: 'time (ms)', standoff: 14 },
        showspikes: true, spikemode: 'across', spikethickness: 1, spikecolor: muted, spikedash: 'dot' }),
      shapes: [], annotations: [],
    };
    panels.forEach((p, i) => {
      const ya = i === 0 ? 'y' : `y${i + 1}`;
      const key = i === 0 ? 'yaxis' : `yaxis${i + 1}`;
      layout[key] = axis(b, { domain: dom(i), anchor: 'x', title: { text: '' }, tickangle: 0, automargin: false,
        nticks: 4, type: p.log ? 'log' : 'linear', ...(p.range ? { range: p.range } : {}), exponentformat: 'power' });
      for (const tr of p.traces) {
        const c = tr.color || token('--series-1');
        if (tr.band) {
          data.push({ x: tr.x || t, y: tr.band[0], yaxis: ya, mode: 'lines', line: { width: 0, color: c }, hoverinfo: 'skip', showlegend: false });
          data.push({ x: tr.x || t, y: tr.band[1], yaxis: ya, mode: 'lines', line: { width: 0, color: c }, fill: 'tonexty', fillcolor: hexA(c, 0.2), hoverinfo: 'skip', showlegend: false });
        }
        data.push({
          x: tr.x || t, y: tr.y, yaxis: ya, mode: tr.mode || 'lines', name: tr.name, showlegend: legend && tr.legend !== false,
          legendgroup: tr.group || tr.name,
          ...(tr.stack ? { stackgroup: tr.stack, fillcolor: hexA(c, 0.55) } : {}),
          line: { width: tr.width || 1.6, color: c, dash: tr.dash || 'solid', shape: tr.shape || 'linear' },
          marker: tr.marker || { size: 6, color: c },
          hovertemplate: `%{y:${tr.fmt || '.4g'}} ${tr.unit || ''}<extra>${esc(tr.name || '')}</extra>`,
        });
      }
      for (const hl of p.hlines || []) {
        layout.shapes.push({ type: 'line', xref: 'paper', x0: 0, x1: 1, yref: ya, y0: hl.y, y1: hl.y, line: { width: 1, dash: hl.dash || 'dash', color: hl.color || muted } });
        if (hl.label) layout.annotations.push({ xref: 'paper', x: 1, xanchor: 'right', yref: ya, y: hl.y, yanchor: 'bottom', showarrow: false, text: hl.label, font: { size: 11, color: token('--text-secondary') } });
      }
    });
    const score = S.scn.summary.score_from_s;
    if (score) this.domains.forEach(([y0, y1]) => layout.shapes.push({ type: 'rect', xref: 'x', x0: 0, x1: score * 1e3, yref: 'paper', y0, y1, fillcolor: hexA(muted, 0.12), line: { width: 0 }, layer: 'below' }));
    if (events) {
      for (const ev of S.scn.events) {
        const tl = ev.t_s * 1e3;
        if (tl <= 0 || (ev.label || '').startsWith('metrics start')) continue;
        if (ev.ring != null && ev.ring !== ring) continue;
        this.domains.forEach(([y0, y1]) => layout.shapes.push({ type: 'line', xref: 'x', x0: tl, x1: tl, yref: 'paper', y0, y1, line: { width: 1, dash: 'dot', color: muted } }));
        const lab = eventLabels.get(ev);
        if (lab) layout.annotations.push({ xref: 'x', x: tl, xanchor: lab.anchor, yref: 'paper', y: 1, yanchor: 'bottom', yshift: this.captionHeights[0] + 24 + lab.lane * 26, showarrow: false, text: lab.text, font: { size: 11, color: token('--text-secondary') } });
      }
    }
    layout.shapes.push(...shapes);
    // Legend offsets use pixels converted to paper coordinates, so a tall stack
    // does not push its legend hundreds of pixels below the time axis.
    if (legend) {
      const names = data.filter((tr) => tr.showlegend).map((tr) => tr.name || '');
      let rows = 1, used = 0;
      for (const name of names) {
        const width = Math.min(plotWidth, name.length * 8 + 56);
        if (used && used + width > plotWidth) { rows++; used = 0; }
        used += width;
      }
      layout.margin.b = 104 + rows * 24;
      layout.height = plotH + top + layout.margin.b;
    }
    this.cursor.innerHTML = this.domains.map(() => '<span class="cursor-segment"></span>').join('');
    this.layout = layout;
    renderPlot(this.gd, data, layout, PLOT_CONFIG).then(() => this.placeCursor());
    if (!this.bound && typeof this.gd.on === 'function') {
      this.bound = true;
      this.gd.on('plotly_click', (ev) => { if (ev.points && ev.points.length && this.onSeek) this.onSeek(ev.points[0].x); });
      this.gd.on('plotly_relayout', () => this.placeCursor());
      this.gd.on('plotly_afterplot', () => this.placeCursor());
    }
  }

  placeCaptions() {
    const fl = this.gd._fullLayout;
    if (!fl?._size || !this.captions) return;
    this.captions.forEach((el, i) => {
      el.style.left = `${this.gd.offsetLeft + fl._size.l}px`;
      el.style.width = `${fl._size.w}px`;
      el.style.top = `${this.gd.offsetTop + fl._size.t + (1 - this.domains[i][1]) * fl._size.h - this.captionHeights[i] - 20}px`;
    });
  }

  setTime(tMs) { this.t = tMs; this.placeCursor(); }

  placeCursor() {
    this.placeCaptions();
    const fl = this.gd._fullLayout;
    if (!fl || !fl.xaxis || !fl._size || !this.gd.offsetParent) { this.cursor.style.display = 'none'; return; }
    const xa = fl.xaxis;
    const px = xa.l2p(this.t);
    const inside = px >= -0.5 && px <= xa._length + 0.5;
    this.cursor.style.display = inside ? 'block' : 'none';
    this.cursor.style.left = `${this.gd.offsetLeft + xa._offset + px}px`;
    this.cursor.style.top = `${this.gd.offsetTop + fl._size.t}px`;
    this.cursor.style.height = `${fl._size.h}px`;
    [...this.cursor.children].forEach((el, i) => {
      const [lo, hi] = this.domains[i];
      el.style.top = `${(1 - hi) * fl._size.h}px`;
      el.style.height = `${(hi - lo) * fl._size.h}px`;
    });
  }
}

// one column of a [frame][ring] array, with a scale
export const colOf = (rows, r, s = 1) => rows.map((row) => (row == null || row[r] == null ? null : row[r] * s));
export const mapRows = (rows, fn) => rows.map((row, f) => (row == null ? null : fn(row, f)));

// a small SVG sparkline (no axes): values over the run, with a cursor line moved by setSparkCursor
export function sparkSVG(vals, { w = 120, h = 24, color, lo = null, hi = null, refs = [] } = {}) {
  let mn = Infinity;
  let mx = -Infinity;
  for (const v of vals) if (v != null && Number.isFinite(v)) { if (v < mn) mn = v; if (v > mx) mx = v; }
  if (lo != null) mn = Math.min(mn, lo);
  if (hi != null) mx = Math.max(mx, hi);
  if (!(mx > mn)) { mx = mn + 1; mn -= 1; }
  const n = vals.length;
  const step = Math.max(1, Math.floor(n / 240));
  const X = (i) => (i / Math.max(1, n - 1)) * w;
  const Y = (v) => h - 2 - ((v - mn) / (mx - mn)) * (h - 4);
  let d = '';
  let pen = false;
  for (let i = 0; i < n; i += step) {
    const v = vals[i];
    if (v == null || !Number.isFinite(v)) { pen = false; continue; }
    d += `${pen ? 'L' : 'M'}${X(i).toFixed(1)},${Y(v).toFixed(1)}`;
    pen = true;
  }
  const rl = refs.filter((v) => v >= mn && v <= mx).map((v) => `<line x1="0" x2="${w}" y1="${Y(v).toFixed(1)}" y2="${Y(v).toFixed(1)}" class="spk-ref"/>`).join('');
  return `<svg class="spark" viewBox="0 0 ${w} ${h}" width="${w}" height="${h}" aria-hidden="true">${rl}` +
    `<path d="${d}" fill="none" stroke="${color}" stroke-width="1.5"/><line class="spk-cur" x1="0" x2="0" y1="0" y2="${h}"/></svg>`;
}

export function setSparkCursor(root, frac) {
  for (const l of root.querySelectorAll('.spk-cur')) {
    const w = Number(l.ownerSVGElement.getAttribute('width'));
    const x = (frac * w).toFixed(1);
    l.setAttribute('x1', x);
    l.setAttribute('x2', x);
  }
}

// a segmented button group: host gets one button per [value, label]; fn(value) on click
export function segs(host, items, value, fn) {
  host.innerHTML = items.map(([v, l]) => `<button type="button" data-v="${esc(v)}" aria-pressed="${v === value}">${esc(l)}</button>`).join('');
  host.querySelectorAll('button').forEach((b) => b.addEventListener('click', () => {
    host.querySelectorAll('button').forEach((x) => x.setAttribute('aria-pressed', String(x === b)));
    fn(b.dataset.v);
  }));
}

// wire an existing group of buttons with data-v
export function bindSegs(host, value, fn) {
  const btns = [...host.querySelectorAll('button[data-v]')];
  const mark = (v) => btns.forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.v === v)));
  mark(value);
  btns.forEach((b) => b.addEventListener('click', () => { mark(b.dataset.v); fn(b.dataset.v); }));
  return mark;
}

// a number with its unit, in the format used by the tables (scientific for BER-like values)
export function num(v, unit = '', { sciForm = false, d = null } = {}) {
  if (v == null || (typeof v === 'number' && !Number.isFinite(v))) return '–';
  const s = sciForm ? sci(v) : (d == null ? fmtNum(v) : v.toFixed(d));
  return unit ? `${s} ${unit}` : s;
}
