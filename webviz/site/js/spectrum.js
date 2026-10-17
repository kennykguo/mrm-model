// "Where is this ring's resonance relative to its laser line?" The selected ring's through port (both data
// levels) and drop port at the nearest key frame, with the laser lines; a strip above places the window
// within the whole comb, with every ring's resonance.

import { plotBase, axis, PLOT_CONFIG, token } from './theme.js';
import { hexA } from './traces.js';

const Y_MIN = -50;

// the exact notch bottom (resonance_level_nm, notch_level_db) inserted into the 12.5 pm grid
function withNotch(lam, y, lr, nd) {
  if (lr == null || nd == null || lr <= lam[0] || lr >= lam[lam.length - 1]) return [lam, y];
  let i = 0;
  while (i < lam.length && lam[i] < lr) i++;
  return [[...lam.slice(0, i), lr, ...lam.slice(i)], [...y.slice(0, i), nd, ...y.slice(i)]];
}

export class SpectrumView {
  constructor(card) {
    this.gd = card.querySelector('.plot');
    this.readout = card.querySelector('.readout');
  }

  setScenario(S) {
    // fixed locator range per scenario: every line and every resonance the key frames visit
    let lo = Infinity;
    let hi = -Infinity;
    for (const row of S.spec.line_nm) for (const v of row) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
    for (const row of S.spec.lam_res_nm) for (const v of row) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
    this.locRange = [lo - 1.4, hi + 1.4];
  }

  update(S, ring, k) {
    const sp = S.spec;
    const b = plotBase();
    const ink = token('--text-primary');
    const ink2 = token('--text-secondary');
    const muted = token('--text-muted');
    const off = sp.grid.offset_nm;
    const cen = sp.center_nm[k][ring];
    const lam = off.map((o) => cen + o);
    const lines = sp.line_nm[k];
    const f = S.keys[k];
    const decl = S.scn.rings.declared_line[f][ring];
    const own = lines[Math.max(0, decl)];
    const lres = sp.lam_res_nm[k][ring];
    const resl = sp.resonance_level_nm[k][ring];
    const notch = sp.notch_level_db[k][ring];
    const lv = sp.levels;
    const [x0, y0] = withNotch(lam, sp.thru0_db[k][ring], resl[0], notch[0]);
    const [x1, y1] = withNotch(lam, sp.thru1_db[k][ring], resl[1], notch[1]);
    const v0 = lv.v_lo_v[k][ring];
    const v1 = lv.v_hi_v[k][ring];
    const data = [
      { x: x0, y: y0, mode: 'lines', name: `through, bit 0 (${v0.toFixed(2)} V)`, line: { width: 2, color: token('--series-1') }, hovertemplate: '%{x:.4f} nm<br>%{y:.2f} dB<extra>bit 0</extra>' },
      { x: x1, y: y1, mode: 'lines', name: `through, bit 1 (${v1.toFixed(2)} V)`, line: { width: 2, color: token('--series-2') }, hovertemplate: '%{x:.4f} nm<br>%{y:.2f} dB<extra>bit 1</extra>' },
      { x: lam, y: sp.drop_db[k][ring], mode: 'lines', name: 'drop to monitor (data average)', line: { width: 2, color: token('--series-3') }, hovertemplate: '%{x:.4f} nm<br>%{y:.2f} dB<extra>drop</extra>' },
    ];
    // locator strip: every laser line and every ring's resonance
    const nl = lines.length;
    data.push({
      x: lines, y: lines.map(() => 0.45), xaxis: 'x2', yaxis: 'y2', mode: 'markers+text', showlegend: false, cliponaxis: false,
      text: lines.map((_, i) => `L${i}`), textposition: 'top center', textfont: { size: 10, color: lines.map((_, i) => (i === decl ? ink : muted)) },
      marker: { symbol: 'line-ns-open', size: 14, line: { width: lines.map((_, i) => (i === decl ? 3 : 1.5)), color: lines.map((_, i) => (i === decl ? ink : muted)) } },
      hovertemplate: lines.map((_, i) => `line L${i}<br>%{x:.3f} nm<extra></extra>`),
    });
    const res = sp.lam_res_nm[k];
    // ring labels: the selected ring first, then each ring whose label clears the labels already placed (two
    // resonances can sit within a few pm of each other, e.g. during the cold start's sweeps); hover names them all
    const stripPx = Math.max(200, (this.gd.clientWidth || 600) - 58 - 16);
    const minSep = 22 * (this.locRange[1] - this.locRange[0]) / stripPx;
    const resText = res.map(() => '');
    const placed = [];
    for (const i of [ring, ...res.map((_, j) => j).filter((j) => j !== ring)]) {
      if (placed.every((x) => Math.abs(res[i] - x) > minSep)) { placed.push(res[i]); resText[i] = `R${i}`; }
    }
    data.push({
      x: res, y: res.map(() => -0.45), xaxis: 'x2', yaxis: 'y2', mode: 'markers+text', showlegend: false, cliponaxis: false,
      text: resText, textposition: 'bottom center', textfont: { size: 10, color: res.map((_, i) => (i === ring ? ink : muted)) },
      marker: { size: res.map((_, i) => (i === ring ? 10 : 7)), color: res.map((_, i) => (i === ring ? ink : muted)) },
      hovertemplate: res.map((_, i) => `R${i} resonance<br>%{x:.3f} nm<extra></extra>`),
    });
    const shapes = [
      { type: 'rect', xref: 'x2', yref: 'y2', x0: lam[0], x1: lam[lam.length - 1], y0: -1, y1: 1, fillcolor: hexA(muted, 0.18), line: { width: 0 }, layer: 'below' },
    ];
    const ann = [];
    const inWin = (x) => x > lam[0] && x < lam[lam.length - 1];
    lines.forEach((x, i) => {
      if (!inWin(x)) return;
      const isOwn = i === decl;
      shapes.push({ type: 'line', xref: 'x', yref: 'y', x0: x, x1: x, y0: Y_MIN, y1: 2, line: { width: isOwn ? 2 : 1, color: isOwn ? ink : muted, dash: isOwn ? 'solid' : 'dash' } });
    });
    const red = lres >= own;
    if (inWin(own)) {
      ann.push({ x: own, y: 2, xref: 'x', yref: 'y', yanchor: 'bottom', xanchor: red ? 'right' : 'left', showarrow: false, text: `L${decl} (own line)`, font: { size: 11, color: ink } });
    } else {
      const right = own > lam[lam.length - 1];
      ann.push({ x: right ? 1 : 0, xref: 'paper', xanchor: right ? 'right' : 'left', y: 2, yref: 'y', yanchor: 'bottom', showarrow: false,
        text: right ? `own line L${decl}: ${(own - cen).toFixed(2)} nm →` : `← own line L${decl}: ${(own - cen).toFixed(2)} nm`, font: { size: 11, color: ink } });
    }
    if (inWin(lres)) {
      shapes.push({ type: 'line', xref: 'x', yref: 'y', x0: lres, x1: lres, y0: Y_MIN, y1: 2, line: { width: 1, color: ink2, dash: 'dot' } });
      ann.push({ x: lres, y: 2, xref: 'x', yref: 'y', yanchor: 'bottom', xanchor: red ? 'left' : 'right', showarrow: false, text: 'resonance', font: { size: 11, color: ink2 } });
    }
    // a notch deeper than the axis: say how deep
    const deep = [0, 1].filter((l) => notch[l] != null && notch[l] < Y_MIN && inWin(resl[l]));
    if (deep.length) {
      ann.push({ x: lres, xref: 'x', y: Y_MIN, yref: 'y', yanchor: 'bottom', xanchor: 'center', showarrow: false, bgcolor: token('--surface-1'),
        text: deep.map((l) => `bit ${l} notch ${notch[l].toFixed(0)} dB ↓`).join(' · '), font: { size: 10, color: ink2 } });
    }
    const layout = {
      paper_bgcolor: b.paper_bgcolor, plot_bgcolor: b.plot_bgcolor, font: b.font, hoverlabel: b.hoverlabel,
      height: 470,
      margin: { l: 58, r: 16, t: 30, b: 44 },
      hovermode: 'closest',
      showlegend: true,
      // legend under the x-axis title (Plotly widens the bottom margin for it), so a wrapped legend on a narrow
      // screen never runs into the comb strip
      legend: { ...b.legend, orientation: 'h', x: 0, xanchor: 'left', y: -0.17, yanchor: 'top' },
      xaxis: axis(b, { domain: [0, 1], range: [lam[0], lam[lam.length - 1]], title: { text: 'wavelength (nm)' }, anchor: 'y', tickformat: '.1f', hoverformat: '.4f' }),
      yaxis: axis(b, { domain: [0, 0.72], range: [Y_MIN, 4], title: { text: 'transmission (dB)' }, anchor: 'x', dtick: 10 }),
      xaxis2: axis(b, { domain: [0, 1], range: this.locRange, anchor: 'y2', side: 'top', showgrid: false, ticks: '', showticklabels: false, showline: false, zeroline: false, fixedrange: true }),
      yaxis2: { domain: [0.84, 1], range: [-1.25, 1.25], visible: false, fixedrange: true },
      shapes,
      annotations: ann.concat([{ xref: 'paper', x: 0, yref: 'paper', y: 1, yanchor: 'bottom', xanchor: 'left', showarrow: false,
        text: 'whole comb (shaded: the window below)', font: { size: 10, color: muted } }]),
    };
    Plotly.react(this.gd, data, layout, PLOT_CONFIG);
    return { decl, detuning: (lres - own) * 1e3 };
  }
}
