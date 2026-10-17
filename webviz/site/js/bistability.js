// "Why is the ring bistable?" Ring 0 swept open loop, heater code up then down: the engine run (two legs)
// over the static S-curve of the model's laws, with the fold points where a stable branch ends.

import { plotBase, axis, PLOT_CONFIG, token } from './theme.js';

const YS = {
  drop: { key: 'p_drop_uw', title: 'drop power to the monitor (µW)', unit: 'µW', fmt: '.2f' },
  det: { key: 'detuning_pm', title: 'detuning (pm)', unit: 'pm', fmt: '.1f' },
};

export function powerLabel(p) {
  const dbm = `${p.p_ring_dbm > 0 ? '+' : ''}${p.p_ring_dbm.toFixed(0)} dBm`;
  return p.lines_on === 'all' ? `${dbm}, all 8 lines` : dbm;
}

export class BistabilityView {
  constructor(card, data) {
    this.card = card;
    this.gd = card.querySelector('.plot');
    this.data = data;
    this.idx = Math.max(0, data.powers.findIndex((p) => p.p_ring_dbm === 4 && p.lines_on !== 'all'));
    this.y = 'drop';
    const pw = card.querySelector('[data-role=powers]');
    data.powers.forEach((p, i) => {
      const bt = document.createElement('button');
      bt.type = 'button';
      bt.className = 'seg';
      bt.textContent = powerLabel(p);
      bt.setAttribute('aria-pressed', String(i === this.idx));
      bt.addEventListener('click', () => {
        this.idx = i;
        pw.querySelectorAll('button').forEach((x, j) => x.setAttribute('aria-pressed', String(j === i)));
        this.render();
      });
      pw.appendChild(bt);
    });
    card.querySelectorAll('[data-role=yaxis] button').forEach((bt) => {
      bt.addEventListener('click', () => {
        this.y = bt.dataset.y;
        card.querySelectorAll('[data-role=yaxis] button').forEach((x) => x.setAttribute('aria-pressed', String(x === bt)));
        this.render();
      });
    });
  }

  render() {
    const p = this.data.powers[this.idx];
    const Y = YS[this.y];
    const b = plotBase();
    const muted = token('--text-muted');
    const ink2 = token('--text-secondary');
    const st = p.static;
    // split the static curve into stable (solid) and unstable (dotted) runs
    // (the unstable run also takes its stable neighbours, so the S has no gap)
    const runs = { true: { x: [], y: [] }, false: { x: [], y: [] } };
    const n = st.code.length;
    for (let i = 0; i < n; i++) {
      const s = st.stable[i];
      const nearUnstable = !s || (i > 0 && !st.stable[i - 1]) || (i < n - 1 && !st.stable[i + 1]);
      runs.true.x.push(s ? st.code[i] : null);
      runs.true.y.push(s ? st[Y.key][i] : null);
      runs.false.x.push(nearUnstable ? st.code[i] : null);
      runs.false.y.push(nearUnstable ? st[Y.key][i] : null);
    }
    const sw = p.sweep;
    const yk = Y.key;
    const data = [
      { x: runs.true.x, y: runs.true.y, mode: 'lines', name: 'static law, stable', line: { width: 2, color: ink2 }, connectgaps: false, hovertemplate: `code %{x:.0f}<br>%{y:${Y.fmt}} ${Y.unit}<extra>static, stable</extra>` },
      { x: runs.false.x, y: runs.false.y, mode: 'lines', name: 'static law, unstable', line: { width: 2, color: muted, dash: 'dot' }, connectgaps: false, hovertemplate: `code %{x:.0f}<br>%{y:${Y.fmt}} ${Y.unit}<extra>static, unstable</extra>` },
      { x: sw.up.code, y: sw.up[yk], mode: 'lines', name: 'engine, code rising', line: { width: 2, color: token('--series-1') }, hovertemplate: `code %{x:.0f}<br>%{y:${Y.fmt}} ${Y.unit}<extra>code rising</extra>` },
      { x: sw.down.code, y: sw.down[yk], mode: 'lines', name: 'engine, code falling', line: { width: 2, color: token('--series-2') }, hovertemplate: `code %{x:.0f}<br>%{y:${Y.fmt}} ${Y.unit}<extra>code falling</extra>` },
      {
        x: st.folds.map((f) => f.code), y: st.folds.map((f) => (this.y === 'drop' ? f.p_drop_uw : f.detuning_pm)), mode: 'markers', name: 'fold (stable branch ends)',
        marker: { symbol: 'diamond', size: 11, color: token('--text-primary'), line: { width: 2, color: token('--surface-1') } },
        hovertemplate: st.folds.map((f) => `${f.kind}<br>code %{x:.0f}<br>%{y:${Y.fmt}} ${Y.unit}<extra></extra>`),
      },
    ];
    const ann = st.folds.map((f) => {
      const up = f.kind.startsWith('cold');
      return {
        x: f.code, y: this.y === 'drop' ? f.p_drop_uw : f.detuning_pm, showarrow: true, arrowhead: 0, arrowcolor: muted, ax: up ? 30 : -30, ay: up ? 34 : -30,
        text: up ? 'fold: jumps up' : 'fold: falls off', font: { size: 11, color: ink2 }, bgcolor: token('--surface-1'),
      };
    });
    const lo = Math.min(...sw.code_knots);
    const hi = Math.max(...sw.code_knots);
    const pad = (hi - lo) * 0.04;
    const layout = {
      paper_bgcolor: b.paper_bgcolor, plot_bgcolor: b.plot_bgcolor, font: b.font, hoverlabel: b.hoverlabel,
      height: 420,
      margin: { l: 62, r: 16, t: 10, b: 96 },
      hovermode: 'closest',
      legend: { ...b.legend, orientation: 'h', x: 0, y: -0.2, yanchor: 'top' },
      xaxis: axis(b, { title: { text: 'heater code (open loop)' }, range: [lo - pad, hi + pad] }),
      yaxis: axis(b, { title: { text: Y.title } }),
      annotations: ann,
    };
    Plotly.react(this.gd, data, layout, PLOT_CONFIG);
    const sub = this.card.querySelector('.readout');
    const w = Math.abs(st.folds[0].code - st.folds[1].code);
    sub.textContent = `ring 0 · ${powerLabel(p)} at the ring · loop width ${w.toFixed(0)} codes · ` +
      `40 ms ramp each way${sw.monitor_code_max >= 65535 ? ' · monitor ADC saturates at the peak' : ''}`;
  }
}
