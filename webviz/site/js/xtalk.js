// "Crosstalk": who heats whom (the DC gains of the model's thermal system between every observer and every heat
// source, K/mW), how fast (the lags of each pair: step responses and 10/63/90 % times), how much of each ring's
// temperature is crosstalk over the run (the engine's lag states grouped by source), and the optical crosstalk
// between lines that the model has (other lines' light in each ring's monitor and heat). All model output.

import { relayoutPlot, plotBase, axis, PLOT_CONFIG, token, renderPlot } from './theme.js';
import { Tiles, TimeStack, esc, fmtNum, cat, bindSegs } from './ui.js';
import { drawXtMatrix, baseLayout } from './panels.js';

const GROUP_LABEL = {
  own_heater: 'own heater', own_light: 'own absorbed light', own_jd: 'own junction + driver',
  other_heaters: 'other heaters', other_light: "other rings' light", other_jd: 'other junctions + drivers', aggressor: 'aggressor sites',
};
const XT_GROUPS = ['other_heaters', 'other_light', 'other_jd', 'aggressor'];

function obsLabel(n) {
  let m = n.match(/^ring_mode_weighted(\d+)$/);
  if (m) return `R${m[1]} ring`;
  m = n.match(/^heater_resistor(\d+)$/);
  if (m) return `R${m[1]} heater`;
  m = n.match(/^monitor_pd(\d+)$/);
  if (m) return `R${m[1]} PD`;
  return n === 'die0' ? 'die sensor' : n;
}

function srcLabel(n) {
  const m = n.match(/^(heater|waveguide_absorption|junction|driver)(\d+)$/);
  if (m) return `${{ heater: 'heater', waveguide_absorption: 'light', junction: 'junction', driver: 'driver' }[m[1]]} ${m[2]}`;
  return n === 'aggressor' ? 'aggressor' : n === 'aggressor_pd_adjacent' ? 'stress site' : n;
}

const OBS_SETS = [['rings', 'Rings'], ['heaters', 'Heater resistors'], ['pds', 'Monitor PDs'], ['all', 'All 25 observers']];
const SRC_SETS = [['all', 'All 34 sources'], ['heater', 'Heaters'], ['light', 'Absorbed light'], ['junction', 'Junctions'], ['driver', 'Drivers'], ['aggr', 'Aggressor sites']];

const us = (s) => (s == null ? '–' : s < 1e-3 ? `${fmtNum(s * 1e6)} µs` : s < 1 ? `${fmtNum(s * 1e3)} ms` : `${fmtNum(s)} s`);

export class CrosstalkView {
  constructor(root, { onSeek }) {
    this.root = root;
    this.dcObs = root.querySelector('#cardDc [data-role=obs]');
    this.dcSrc = root.querySelector('#cardDc [data-role=src]');
    this.dcPlot = root.querySelector('#cardDc .plot');
    this.tauObs = root.querySelector('#cardTau [data-role=obs]');
    this.tauSub = root.querySelector('#cardTau [data-role=sub]');
    this.tauTable = root.querySelector('#cardTau [data-role=table]');
    this.stepPlot = root.querySelector('#cardTau .plot.step');
    this.tiles = new Tiles(root.querySelector('#cardShare .tiles'));
    this.stack = new TimeStack(root.querySelector('#cardShare [data-role=area]'), { onSeek, height: 200 });
    this.bankHost = document.createElement('div');
    this.bankHost.className = 'bank-heat-grid';
    this.stack.wrap.before(this.bankHost);
    this.shareTable = root.querySelector('#cardShare [data-role=table]');
    this.mTitle = root.querySelector('#cardShare [data-role=mtitle]');
    this.xtPlot = root.querySelector('#cardShare .xtmat');
    this.optDef = root.querySelector('#cardOpt [data-role=def]');
    this.optTitle = root.querySelector('#cardOpt [data-role=mtitle]');
    this.oPlot = root.querySelector('#cardOpt .omat');
    this.oShare = root.querySelector('#cardOpt .oshare');
    this.dcObs.innerHTML = OBS_SETS.map(([v, l]) => `<option value="${v}">${esc(l)}</option>`).join('');
    this.dcSrc.innerHTML = SRC_SETS.map(([v, l]) => `<option value="${v}">${esc(l)}</option>`).join('');
    this.dcObs.value = 'rings';
    this.dcSrc.value = 'heater';
    this.dcObs.addEventListener('change', () => this.renderDc());
    this.dcSrc.addEventListener('change', () => this.renderDc());
    this.tauObs.addEventListener('change', () => this.renderTau());
    this.which = 'ring';
    this.win = 'scored';
    this.kind = 'all';
    bindSegs(root.querySelector('#cardShare [data-role=which]'), this.which, (v) => { this.which = v; this.renderShare(); });
    bindSegs(root.querySelector('#cardShare [data-role=window]'), this.win, (v) => { this.win = v; this.renderShareTable(); });
    bindSegs(root.querySelector('#cardShare [data-role=xtkind]'), this.kind, (v) => { this.kind = v; this.slowUpdate(true); });
    this.lastKey = -1;
  }

  setState(S, r, f, all = true) { this.S = S; this.r = r; this.f = f; this.all = all; }

  render() {
    if (!this.S || !this.S.meta.thermal_coupling.full) return;
    this.renderDc();
    this.fillTauObs();
    this.renderTau();
    this.renderShare();
    this.renderOptDef();
    this.lastKey = -1;
    this.slowUpdate(true);
  }

  // ---------------------------------------------------------------- DC coupling matrix
  renderDc() {
    const full = this.S.meta.thermal_coupling.full;
    const ov = this.dcObs.value;
    const sv = this.dcSrc.value;
    const oi = full.obs_names.map((n, i) => [n, i]).filter(([n]) => ov === 'all' || (ov === 'rings' && n.startsWith('ring_mode')) ||
      (ov === 'heaters' && n.startsWith('heater_resistor')) || (ov === 'pds' && n.startsWith('monitor_pd'))).map(([, i]) => i);
    const re = { heater: /^heater\d/, light: /^waveguide_absorption/, junction: /^junction/, driver: /^driver/, aggr: /^aggressor/ }[sv];
    const si = full.src_names.map((n, i) => [n, i]).filter(([n]) => sv === 'all' || re.test(n)).map(([, i]) => i);
    const z = oi.map((o) => si.map((s) => full.g_k_per_mw[o][s]));
    const b = plotBase();
    const lz = z.map((row) => row.map((v) => Math.log10(Math.max(v, 1e-5))));
    const small = si.length <= 12 && oi.length <= 12;
    const h = Math.max(420, 38 * oi.length + 130);
    this.dcPlot.style.minWidth = `${Math.max(480, si.length * 84 + 140)}px`;
    renderPlot(this.dcPlot, [{ type: 'heatmap', z: lz, x: si.map((s) => srcLabel(full.src_names[s])), y: oi.map((o) => obsLabel(full.obs_names[o])), customdata: z,
      colorscale: [[0, token('--surface-2')], [1, token('--series-1')]], zmin: -5, zmax: Math.log10(Math.max(...z.flat())),
      text: small ? z.map((row) => row.map((v) => (v >= 1 ? v.toFixed(2) : v >= 0.01 ? v.toFixed(3) : v.toExponential(1)))) : null, texttemplate: small ? '%{text}' : undefined, textfont: { size: 12 },
      hovertemplate: '%{y} from %{x}: %{customdata:.4g} K/mW<extra></extra>', colorbar: { thickness: 10, title: { text: 'log10 K/mW', side: 'right' } } }], {
      ...baseLayout(h), margin: { l: 96, r: 16, t: 10, b: 90 }, xaxis: axis(b, { tickangle: -45 }), yaxis: axis(b, { autorange: 'reversed' }),
    }, PLOT_CONFIG);
  }

  // ---------------------------------------------------------------- time constants
  fillTauObs() {
    const full = this.S.meta.thermal_coupling.full;
    const r = this.r;
    const want = [`ring_mode_weighted${r}`, `heater_resistor${r}`, `monitor_pd${r}`, 'die0'];
    const prev = this.tauObs.value;
    this.tauObs.innerHTML = want.filter((n) => full.obs_names.includes(n)).map((n) => `<option value="${n}">${esc(obsLabel(n))}</option>`).join('');
    const kind = prev ? prev.replace(/\d+$/, '') : 'ring_mode_weighted';
    const keep = want.find((n) => n.replace(/\d+$/, '') === kind);
    this.tauObs.value = keep || want[0];
  }

  renderTau() {
    const full = this.S.meta.thermal_coupling.full;
    const o = full.obs_names.indexOf(this.tauObs.value);
    if (o < 0) return;
    const rows = full.src_names.map((n, s) => ({ n, s, g: full.g_k_per_mw[o][s], lags: full.lags[o][s] })).filter((x) => x.lags.length && x.g > 0).sort((a, b) => b.g - a.g);
    this.tauSub.textContent = `${obsLabel(full.obs_names[o])}: each curve shows the response to switching on 1 mW at one source and holding it. Faster rises mean quicker heat paths; the plateau is the final coupling. Both axes are logarithmic. The plot shows the ${Math.min(6, rows.length)} strongest sources; the table lists up to 12.`;
    let h = '<colgroup><col style="width:14%"><col style="width:12%"><col style="width:44%"><col style="width:10%"><col style="width:10%"><col style="width:10%"></colgroup><thead><tr><th scope="col">Source</th><th scope="col">DC gain (K/mW)</th><th scope="col">Poles τ (residue K/mW)</th><th scope="col">t10</th><th scope="col">t63</th><th scope="col">t90</th></tr></thead><tbody>';
    for (const x of rows.slice(0, 12)) {
      h += `<tr><th scope="row">${esc(srcLabel(x.n))}</th><td>${fmtNum(x.g)}</td><td class="md">${x.lags.map(([t, rr]) => `<span class="thermal-pole">${us(t)} (${fmtNum(rr)})</span>`).join(' ')}</td>` +
        `<td>${us(full.t10_s[o][x.s])}</td><td>${us(full.t63_s[o][x.s])}</td><td>${us(full.t90_s[o][x.s])}</td></tr>`;
    }
    this.tauTable.innerHTML = `${h}</tbody>`;
    const t = Array.from({ length: 241 }, (_, i) => 10 ** (-7 + (i * 7) / 240));
    const bank = this.root.querySelector('#cardTau [data-role=bank-step]');
    const kind = this.tauObs.value.replace(/\d+$/, '');
    bank.hidden = !this.all || kind === 'die0';
    if (!bank.hidden) {
      const noun = kind === 'monitor_pd' ? 'monitor PD' : kind === 'heater_resistor' ? 'heater resistor' : 'ring';
      bank.querySelector('h3').textContent = `How quickly does each ${noun} respond to its own ring’s heater?`;
      const bb = plotBase();
      const curves = Array.from({ length: this.S.nR }, (_, q) => {
        const oi = full.obs_names.indexOf(kind + q);
        const si = full.src_names.indexOf(`heater${q}`);
        return { x: t, y: t.map((tt) => full.lags[oi][si].reduce((a, [tau, rr]) => a + rr * (1 - Math.exp(-tt / tau)), 0)), mode: 'lines', name: `R${q}`, line: { color: cat(q), width: q === this.r ? 2.6 : 1.2 }, hovertemplate: `R${q}: %{y:.4g} K/mW at %{x:.3g} s<extra></extra>` };
      });
      renderPlot(bank.querySelector('.plot'), curves, { ...baseLayout(420), showlegend: true, legend: { ...bb.legend, orientation: 'h', y: -0.25 }, xaxis: axis(bb, { type: 'log', title: { text: 'time after a 1 mW heater step (s)' } }), yaxis: axis(bb, { type: 'log', title: { text: 'rise (K per mW)' } }) }, PLOT_CONFIG);
    }
    const b = plotBase();
    const data = rows.slice(0, 6).map((x, i) => ({
      x: t, y: t.map((tt) => x.lags.reduce((a, [tau, rr]) => a + rr * (1 - Math.exp(-tt / tau)), 0)), mode: 'lines', name: srcLabel(x.n),
      line: { width: 1.8, color: cat(i) }, hovertemplate: `${esc(srcLabel(x.n))}: %{y:.4g} K/mW at %{x:.3g} s<extra></extra>`,
    }));
    renderPlot(this.stepPlot, data, {
      ...baseLayout(420), showlegend: true, legend: { ...b.legend, orientation: 'h', y: -0.28 },
      xaxis: axis(b, { type: 'log', title: { text: 'time after a 1 mW step (s)' }, exponentformat: 'power' }),
      yaxis: axis(b, { type: 'log', title: { text: 'rise (K per mW)' }, exponentformat: 'power' }),
    }, PLOT_CONFIG);
  }

  // ---------------------------------------------------------------- crosstalk share over the run
  groups() {
    const xt = this.S.dev && this.S.dev.xt;
    if (!xt) return null;
    return { names: xt.groups, rows: this.which === 'pd' ? xt.pd_k : xt.ring_k };
  }

  renderShare() {
    const S = this.S;
    const G = this.groups();
    if (!G) return;
    const r = this.r;
    const who = this.which === 'pd' ? `R${r}'s monitor PD` : `R${r}`;
    const tiles = [{ label: 'Rise above the base', note: 'The sum of every source, averaged per frame.', get: (S_, r_, f) => `${fmtNum(G.rows[f][r_].reduce((a, b) => a + b, 0))} K` }];
    for (const [g, l] of Object.entries(GROUP_LABEL)) {
      const j = G.names.indexOf(g);
      tiles.push({ label: `From the ${l}`, note: '', get: (S_, r_, f) => `${fmtNum(G.rows[f][r_][j])} K` });
    }
    tiles.push({ label: 'Crosstalk share now', note: 'Other heaters, other rings\' light, junctions and drivers, and the aggressor, as a % of the rise.', get: (S_, r_, f) => {
      const row = G.rows[f][r_];
      const tot = row.reduce((a, b) => a + b, 0);
      const x = XT_GROUPS.reduce((a, g) => a + row[G.names.indexOf(g)], 0);
      return `${fmtNum((100 * x) / tot)} %`;
    } });
    this.tiles.build(tiles, { rings: this.all ? S.nR : 0, selected: r });
    this.tiles.update(S, r, this.f);
    const order = ['own_heater', 'own_light', 'own_jd', 'other_heaters', 'other_light', 'other_jd', 'aggressor'];
    const colors = { own_heater: cat(1), own_light: cat(3), own_jd: cat(4), other_heaters: cat(0), other_light: cat(2), other_jd: cat(5), aggressor: cat(7) };
    const area = order.map((g) => ({ y: G.rows.map((row) => row[r][G.names.indexOf(g)]), name: GROUP_LABEL[g], group: g, color: colors[g], stack: 'one', width: 0.5, unit: 'K', legend: true }));
    const share = [];
    for (let q = 0; q < S.nR; q++) {
      share.push({ y: G.rows.map((row) => { const tot = row[q].reduce((a, b) => a + b, 0); return (100 * XT_GROUPS.reduce((a, g) => a + row[q][G.names.indexOf(g)], 0)) / tot; }),
        name: `R${q}`, group: `R${q}`, color: cat(q), width: q === r ? 2.6 : 1.1, unit: '%', legend: true });
    }
    const sharePanel = { title: 'crosstalk share (%)', question: `What fraction of ${this.which === 'pd' ? 'monitor PD' : 'ring'} heating is crosstalk?`, description: 'Heat from other rings and aggressor sites as a percentage of total rise above the base. Each color is a ring; the selected ring is thicker. This is a fraction of heating, not an absolute temperature.', traces: share };
    this.bankHost.hidden = !this.all;
    if (this.all) {
      if (!this.bankStacks || this.bankStacks.length !== S.nR) {
        this.bankHost.innerHTML = '<div class="heat-source-key">' + order.map((g) => `<span><i class="sw" style="background:${colors[g]}"></i>${esc(GROUP_LABEL[g])}</span>`).join('') + '</div><div class="ring-grid">' + Array.from({ length: S.nR }, () => '<div class="ring-panel"><div class="plotwrap"></div></div>').join('') + '</div>';
        this.bankStacks = [...this.bankHost.querySelectorAll('.plotwrap')].map((host) => new TimeStack(host, { onSeek: this.stack.onSeek, height: 150 }));
      }
      let maxRise = 0;
      for (const frame of G.rows) for (const row of frame) maxRise = Math.max(maxRise, row.reduce((a, b) => a + b, 0));
      this.bankStacks.forEach((stack, q) => {
        stack.wrap.parentElement.classList.toggle('selected', q === r);
        const traces = order.map((g) => ({ y: G.rows.map((row) => row[q][G.names.indexOf(g)]), name: GROUP_LABEL[g], group: g, color: colors[g], stack: 'one', width: 0.5, unit: 'K' }));
        stack.render(S, [{ title: 'rise (K)', question: `What is heating R${q}${this.which === 'pd' ? '’s monitor PD' : ''}?`, description: 'Colored layers add to total rise above the base. All panels use the same temperature scale.', traces, range: [0, Math.max(1, maxRise * 1.05)] }], { events: false });
        stack.setTime(S.tMs[this.f]);
      });
      this.stack.render(S, [sharePanel], { legend: true, ring: r });
    } else this.stack.render(S, [{ title: `${who} rise (K)`, question: `What is heating ${who}?`, description: 'Each colored layer is one group of heat sources. Their stacked height is the total temperature rise above the base; a thicker layer means a larger contribution.', traces: area }, sharePanel], { legend: true, ring: r });
    this.stack.setTime(S.tMs[this.f]);
    this.renderShareTable();
  }

  renderShareTable() {
    const S = this.S;
    const G = this.groups();
    if (!G) return;
    const sf = (S.scn.summary.score_from_s || 0) * 1e3;
    const sel = S.tMs.map((t) => this.win === 'run' || t - S.frameMs >= sf - 1e-9);
    const n = sel.filter(Boolean).length;
    const order = Object.keys(GROUP_LABEL);
    let h = '<thead><tr><th scope="col">Ring</th><th scope="col">Rise (K)</th>' + order.map((g) => `<th scope="col">${esc(GROUP_LABEL[g])}</th>`).join('') + '<th scope="col">Crosstalk share</th></tr></thead><tbody>';
    for (let q = 0; q < S.nR; q++) {
      const m = G.names.map((_, j) => G.rows.reduce((a, row, f) => a + (sel[f] ? row[q][j] : 0), 0) / n);
      const tot = m.reduce((a, b) => a + b, 0);
      const x = XT_GROUPS.reduce((a, g) => a + m[G.names.indexOf(g)], 0);
      h += `<tr class="${q === this.r ? 'sel' : ''}"><th scope="row"><span class="sw" style="background:${cat(q)}"></span>R${q}</th><td>${fmtNum(tot)}</td>` +
        order.map((g) => { const v = m[G.names.indexOf(g)]; return `<td>${fmtNum(v)} <span class="md">${fmtNum((100 * v) / tot)} %</span></td>`; }).join('') +
        `<td><b>${fmtNum((100 * x) / tot)} %</b></td></tr>`;
    }
    this.shareTable.innerHTML = `${h}</tbody>`;
  }

  // ---------------------------------------------------------------- optical crosstalk
  renderOptDef() {
    this.optDef.innerHTML = '<p><b>What the model has.</b> Each ring\'s drop port and loss take light at every line wavelength, not only at its own line: '
      + 'the other lines reach it through its other resonance orders and the tails of its resonance. That light goes to the ring\'s monitor PD '
      + '(a background the controller cannot tell from its own line) and heats the ring. Each ring also takes some light of every line from the bus, '
      + 'so the rings later on the bus get less (see Parts › Laser, WDM and bus).</p>'
      + '<p><b>What the model does not have.</b> No data crosstalk between channels: the BER of a line uses its OMA at the bus output with the data of every other ring '
      + 'averaged, so one ring\'s data never shows as noise on another line. No backscatter (g26) and no coherent (interferometric) crosstalk.</p>';
  }

  slowUpdate(force = false) {
    const S = this.S;
    if (!S || !S.dev) return;
    const k = S.nearestKey[this.f];
    if (!force && k === this.lastKey) return;
    this.lastKey = k;
    const tk = S.tMs[S.keys[k]];
    this.mTitle.textContent = `Who heats whom now: rise of each ring by source owner, key frame ${tk.toFixed(2)} ms (K)`;
    drawXtMatrix(this.xtPlot, S, this.r, k, this.kind);
    const bus = S.spec.bus;
    if (!bus.p_drop_uw) { this.optTitle.textContent = 'spectra file has no per-line drop power: regenerate the data.'; return; }
    this.optTitle.textContent = `Light of each line in each ring's monitor PD, key frame ${tk.toFixed(2)} ms (µW, log colour; boxes: the ring's own declared line)`;
    const z = bus.p_drop_uw[k];
    const b = plotBase();
    const decl = S.scn.rings.declared_line[S.keys[k]];
    const shapes = decl.map((l, q) => ({ type: 'rect', xref: 'x', yref: 'y', x0: l - 0.5, x1: l + 0.5, y0: q - 0.5, y1: q + 0.5, line: { width: 1.5, color: token('--text-primary') } }));
    renderPlot(this.oPlot, [{ type: 'heatmap', z: z.map((row) => row.map((v) => Math.log10(Math.max(v, 1e-6)))), customdata: z, x: z[0].map((_, l) => l), y: z.map((_, q) => q),
      colorscale: [[0, token('--surface-2')], [1, token('--series-2')]], zmin: -6,
      hovertemplate: 'R%{y} PD, line L%{x}: %{customdata:.4g} µW<extra></extra>', colorbar: { thickness: 10, title: { text: 'log10 µW', side: 'right' } } }], {
      ...baseLayout(420), xaxis: axis(b, { title: { text: 'laser line' }, tickvals: z[0].map((_, l) => l), ticktext: z[0].map((_, l) => `L${l}`) }),
      yaxis: axis(b, { title: { text: 'ring' }, autorange: 'reversed', tickvals: z.map((_, q) => q), ticktext: z.map((_, q) => `R${q}`) }), shapes,
    }, PLOT_CONFIG);
    if (force || !this.oShareDone) this.drawOptShare();
    else relayoutPlot(this.oShare, { 'shapes[0].x0': tk, 'shapes[0].x1': tk });
  }

  drawOptShare() {
    const S = this.S;
    const bus = S.spec.bus;
    const kt = S.keys.map((f) => S.tMs[f]);
    const b = plotBase();
    const data = [];
    const frac = (arr, k, q) => {
      const row = arr[k][q];
      const own = row[S.scn.rings.declared_line[S.keys[k]][q]];
      const tot = row.reduce((a, v) => a + v, 0);
      return tot > 0 ? (100 * (tot - own)) / tot : null;
    };
    for (let q = 0; q < S.nR; q++) {
      data.push({ x: kt, y: kt.map((_, k) => frac(bus.p_drop_uw, k, q)), mode: 'lines+markers', marker: { size: 3, color: cat(q) }, name: `R${q} monitor`, legendgroup: `R${q}`,
        line: { width: q === this.r ? 2.4 : 1.1, color: cat(q) }, hovertemplate: `R${q} monitor: %{y:.3g} %%<extra></extra>` });
      data.push({ x: kt, y: kt.map((_, k) => frac(bus.p_abs_uw, k, q)), mode: 'lines', name: `R${q} absorbed`, legendgroup: `R${q}`, showlegend: false,
        line: { width: q === this.r ? 2 : 1, color: cat(q), dash: 'dot' }, hovertemplate: `R${q} absorbed: %{y:.3g} %%<extra></extra>` });
    }
    const t = S.tMs[S.keys[S.nearestKey[this.f]]];
    renderPlot(this.oShare, data, {
      ...baseLayout(420), showlegend: false, xaxis: axis(b, { title: { text: 'time (ms)' }, range: [0, S.m.duration_s * 1e3] }),
      yaxis: axis(b, { title: { text: 'from foreign lines (%)' }, type: 'log', exponentformat: 'power' }),
      shapes: [{ type: 'line', xref: 'x', x0: t, x1: t, yref: 'paper', y0: 0, y1: 1, line: { width: 1.5, color: token('--text-primary') } }],
    }, PLOT_CONFIG);
    this.oShareDone = true;
  }

  setFrame(f) {
    this.f = f;
    if (!this.S || !this.S.dev) return;
    this.tiles.update(this.S, this.r, f);
    this.stack.setTime(this.S.tMs[f]);
    if (this.all && this.bankStacks) this.bankStacks.forEach((stack) => stack.setTime(this.S.tMs[f]));
  }

  invalidate() { this.oShareDone = false; }
}

