// "What is the controller doing over time?" The selected ring's traces, stacked on one time axis (one y-axis
// per quantity, no dual axes), with the scenario drive in the bottom panel and a cursor at the current time.

import { plotBase, axis, PLOT_CONFIG, token } from './theme.js';
import { K0 } from './data.js';

export function shortEvent(ev, ring) {
  const l = ev.label || '';
  if (l.startsWith('metrics start')) return null;
  if (l.startsWith('aggressor power at its peak')) return 'aggressor peak';
  if (l.includes('hot cusp')) return null;
  if (l.includes('laser power on every line')) return l.replace(' on every line', '');
  if (ev.ring != null) return ev.ring === ring ? `R${ring} locked` : null;
  if (ev.t_s === 0) return null;
  return l.length > 24 ? `${l.slice(0, 22)}…` : l;
}

export class TracesView {
  constructor(card, { onSeek }) {
    this.card = card;
    this.gd = card.querySelector('.plot');
    this.cursor = document.createElement('div');
    this.cursor.className = 'cursor';
    this.gd.parentElement.appendChild(this.cursor);
    this.t = 0;
    this.onSeek = onSeek;
    this.bound = false;
  }

  render(S, ring) {
    this.S = S;
    this.ring = ring;
    const b = plotBase();
    const c1 = token('--series-1');
    const c2 = token('--series-2');
    const muted = token('--text-muted');
    const t = S.tMs;
    const R = S.scn.rings;
    const tmax = S.m.duration_s * 1e3;
    const n = 5;
    const gap = 0.045;
    const h = (1 - gap * (n - 1)) / n;
    const dom = (i) => [1 - (i + 1) * h - i * gap, 1 - i * h - i * gap];
    const det = S.col('detuning_pm', ring);
    const dmin = S.col('detuning_min_pm', ring);
    const dmax = S.col('detuning_max_pm', ring);
    const tempC = S.col('t_ring_k', ring).map((v) => v - K0);
    const drive = S.drive.series(ring);
    const hov = (unit, fmt) => `%{y:${fmt}} ${unit}<extra></extra>`;
    const data = [
      { x: t, y: dmin, yaxis: 'y', mode: 'lines', line: { width: 0, color: c1 }, hoverinfo: 'skip', showlegend: false },
      { x: t, y: dmax, yaxis: 'y', mode: 'lines', line: { width: 0, color: c1 }, fill: 'tonexty', fillcolor: hexA(c1, 0.22), hoverinfo: 'skip', showlegend: false },
      { x: t, y: det, yaxis: 'y', mode: 'lines', line: { width: 1.6, color: c1 }, name: 'detuning', hovertemplate: hov('pm', '.2f'), showlegend: false },
      { x: t, y: S.col('heater_code', ring), yaxis: 'y2', mode: 'lines', line: { width: 1.6, color: c1, shape: 'hv' }, name: 'heater code', hovertemplate: hov('', '.0f'), showlegend: false },
      { x: t, y: S.col('monitor_code', ring), yaxis: 'y3', mode: 'lines', line: { width: 1.6, color: c1, shape: 'hv' }, name: 'monitor code', hovertemplate: hov('', '.0f'), showlegend: false },
      { x: t, y: tempC, yaxis: 'y4', mode: 'lines', line: { width: 1.6, color: c1 }, name: 'ring temperature', hovertemplate: hov('°C', '.2f'), showlegend: false },
      { x: t, y: drive, yaxis: 'y5', mode: 'lines', line: { width: 1.6, color: c2, shape: S.drive.kind === 'laser' ? 'hv' : 'linear' }, name: S.drive.label, hovertemplate: hov('', '.4g'), showlegend: false },
    ];
    const ytitles = ['detuning (pm)', 'heater (code)', 'monitor (code)', 'ring (°C)', S.drive.label];
    const layout = {
      paper_bgcolor: b.paper_bgcolor, plot_bgcolor: b.plot_bgcolor, font: b.font, hoverlabel: b.hoverlabel,
      margin: { l: 78, r: 16, t: 26, b: 44 },
      height: 560,
      hovermode: 'x',
      dragmode: 'zoom',
      xaxis: axis(b, { anchor: 'y5', range: [0, tmax], title: { text: 'time (ms)' }, showspikes: true, spikemode: 'across', spikethickness: 1, spikecolor: muted, spikedash: 'dot' }),
      shapes: [],
      annotations: [],
    };
    for (let i = 0; i < n; i++) {
      const key = i === 0 ? 'yaxis' : `yaxis${i + 1}`;
      layout[key] = axis(b, { domain: dom(i), anchor: 'x', title: { text: ytitles[i], standoff: 6, font: { size: 11 } }, automargin: false, nticks: 4 });
    }
    // lock band about the lock reference (the operating point, or the line in the cold start)
    const dref = S.dRef[ring];
    layout.shapes.push(
      { type: 'rect', xref: 'paper', x0: 0, x1: 1, yref: 'y', y0: dref - S.thr, y1: dref + S.thr, fillcolor: hexA(token('--series-3'), 0.16), line: { width: 0 }, layer: 'below' },
      { type: 'line', xref: 'paper', x0: 0, x1: 1, yref: 'y', y0: dref, y1: dref, line: { width: 1, dash: 'dot', color: muted } },
    );
    // (the band is named in the card's readout, not on the panel: a label here would sit on the trace)
    // the start-up window that is not scored, and the events
    const score = S.scn.summary.score_from_s;
    if (score) {
      layout.shapes.push({ type: 'rect', xref: 'x', x0: 0, x1: score * 1e3, yref: 'paper', y0: 0, y1: 1, fillcolor: hexA(muted, 0.12), line: { width: 0 }, layer: 'below' });
    }
    for (const ev of S.scn.events) {
      const lab = shortEvent(ev, ring);
      const tl = ev.t_s * 1e3;
      if (ev.label && ev.label.startsWith('metrics start')) continue;
      if (ev.ring != null && ev.ring !== ring) continue;
      if (tl <= 0) continue;
      layout.shapes.push({ type: 'line', xref: 'x', x0: tl, x1: tl, yref: 'paper', y0: 0, y1: 1, line: { width: 1, dash: 'dot', color: muted } });
      if (lab) layout.annotations.push({ xref: 'x', x: tl, yref: 'paper', y: 1, yanchor: 'bottom', showarrow: false, text: lab, font: { size: 10, color: token('--text-secondary') } });
    }
    Plotly.react(this.gd, data, layout, PLOT_CONFIG).then(() => this.placeCursor());
    if (!this.bound) {
      this.bound = true;
      this.gd.on('plotly_click', (ev) => { if (ev.points && ev.points.length) this.onSeek(ev.points[0].x); });
      this.gd.on('plotly_relayout', () => this.placeCursor());
      this.gd.on('plotly_afterplot', () => this.placeCursor());
    }
  }

  setTime(tMs) {
    this.t = tMs;
    this.placeCursor();
  }

  placeCursor() {
    const fl = this.gd._fullLayout;
    if (!fl || !fl.xaxis || !fl._size) return;
    const xa = fl.xaxis;
    const px = xa.l2p(this.t);
    const inside = px >= -0.5 && px <= xa._length + 0.5;
    this.cursor.style.display = inside ? 'block' : 'none';
    this.cursor.style.left = `${this.gd.offsetLeft + xa._offset + px}px`;
    this.cursor.style.top = `${this.gd.offsetTop + fl._size.t}px`;
    this.cursor.style.height = `${fl._size.h}px`;
  }
}

export function hexA(color, a) {
  const c = color.trim();
  if (c.startsWith('#') && c.length === 7) {
    const r = parseInt(c.slice(1, 3), 16);
    const g = parseInt(c.slice(3, 5), 16);
    const b = parseInt(c.slice(5, 7), 16);
    return `rgba(${r},${g},${b},${a})`;
  }
  return c;
}

export function fmtNum(v) {
  if (v == null || !Number.isFinite(v)) return '–';
  const a = Math.abs(v);
  if (a >= 1000) return v.toFixed(0);
  if (a >= 100) return v.toFixed(1);
  if (a >= 10) return v.toFixed(2);
  if (a >= 1) return v.toFixed(2);
  if (a >= 0.01) return v.toFixed(3);
  if (a === 0) return '0';
  return v.toPrecision(2);
}
