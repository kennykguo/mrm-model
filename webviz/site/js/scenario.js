// "Scenario": what the run tests. A plain paragraph (written by generate.py from the run's own numbers), the facts
// of the run, every disturbance input over time with the events and (when zoomed in) the controller steps, a
// timing diagram of one controller step, the events and every scenario input signal with its value.

import { relayoutPlot, token } from './theme.js';
import { TimeStack, esc, cat, colOf, fmt, K0 } from './ui.js';
import { fmtNum } from './traces.js';

// plain names of the plant's scenario inputs (mrm.scenarios.generators.SIGNALS)
const SIG = {
  'thermal.t_base_k': ['Die base temperature', 'The package / TEC temperature under the die (ambient or soak).'],
  'thermal.xpu_rise_k': ['Processor heat event', 'A die-wide temperature rise from a processor power change.'],
  'thermal.aggressor_w': ['Aggressor power', 'Power into each on-die aggressor heater.'],
  'laser.dfreq_hz': ['Laser frequency offset', 'Change of each line frequency (drift, sweep).'],
  'laser.power_delta_db': ['Laser power change', 'Change of each line power at the laser.'],
  'laser.line_enable': ['Laser line on/off', 'Power factor of each line, 0 = off, 1 = on.'],
  'optical.input_loss_delta_db': ['Chip input loss change', 'Change of the laser-to-chip loss (a fibre re-mate).'],
  'data.ones_density': ['Ones density', 'Fraction of ones in the data of each ring.'],
  'data.transition_density': ['Transition density', 'Transitions per bit of each ring (junction and driver heat).'],
  'data.swing_scale': ['Data swing factor', 'Factor on the driver swing; 0 stops the modulation.'],
  'data.driver_enable': ['Data traffic (driver on)', 'The driver of each ring sends data (1) or not (0).'],
  'tx.bias_offset_v': ['Bias offset', 'Added junction bias (ripple, droop).'],
  'tx.bias_v': ['Junction bias set point', 'The bias of each ring junction.'],
  'fault.adc_drop': ['Fault: ADC drops samples', 'The monitor ADC loses its samples.'],
  'fault.adc_stuck': ['Fault: ADC stuck', 'The monitor ADC repeats one code.'],
  'fault.adc_saturate': ['Fault: ADC saturates', 'The monitor ADC reads full scale.'],
  'fault.dac_glitch_codes': ['Fault: DAC glitch', 'Codes added to the heater DAC output.'],
  'fault.controller_reset': ['Fault: controller reset', 'The controller restarts from its initial state.'],
};
const LEVEL = { die: 'die', line: 'per line', ring: 'per ring', aggressor: 'per site' };

function series(sig, nF) {
  // [frame][column] from an inputs entry (frame means, or one constant value)
  if (sig.mean) return sig.mean;
  return Array.from({ length: nF }, () => sig.constant);
}

function valueText(sig) {
  const unit = sig.unit === 'frac' ? '' : sig.unit;
  const show = (v) => {
    if (sig.name === 'thermal.t_base_k') return `${(v - K0).toFixed(2)} °C`;
    if (sig.name === 'laser.dfreq_hz') return `${fmtNum(v * 1e-6)} MHz`;
    return `${fmtNum(v)}${unit ? ` ${unit}` : ''}`;
  };
  if (sig.constant) {
    const u = [...new Set(sig.constant.map((v) => v))];
    if (u.length === 1) return `${show(u[0])}${sig.constant.length > 1 ? ' (all)' : ''}`;
    if (sig.columns) return sig.columns.map((c, i) => `${c} ${show(sig.constant[i])}`).join(' · ');
    return sig.constant.map(show).join(', ');
  }
  return `${show(sig.range[0])} to ${show(sig.range[1])}`;
}

export class ScenarioView {
  constructor(root, { onSeek }) {
    this.root = root;
    this.para = root.querySelector('[data-role=para]');
    this.facts = root.querySelector('[data-role=facts]');
    this.zoomNote = root.querySelector('[data-role=zoom]');
    this.stack = new TimeStack(root.querySelector('#cardInputs .plotwrap'), { onSeek, height: 96 });
    this.timing = root.querySelector('[data-role=timing]');
    this.timingNote = root.querySelector('[data-role=timingNote]');
    this.events = root.querySelector('[data-role=events]');
    this.signals = root.querySelector('[data-role=signals]');
    this.onSeek = onSeek;
    this.events.addEventListener('click', (e) => {
      const tr = e.target.closest('tr[data-t]');
      if (tr) onSeek(Number(tr.dataset.t));
    });
    this.zoomBound = false;
  }

  render(S, ring = 0) {
    this.S = S;
    this.ring = ring;
    const sm = S.scn.summary;
    const ins = Object.fromEntries((S.scn.inputs ? S.scn.inputs.signals : []).map((s) => [s.name, s]));
    this.ins = ins;
    this.para.textContent = S.scn.plain_description || S.m.description;
    const lr = S.scn.lock_reference;
    const facts = [
      ['Controller', `${sm.controller}${sm.controller_path ? ` (${sm.controller_path} path)` : ''}`],
      ['Start', sm.start === 'warm' ? 'warm: settled at the lock before t = 0' : sm.start === 'cold' ? 'cold: heaters at code 0, lasers off at t = 0' : (sm.start || '–')],
      ['Length', `${fmt(sm.duration_s * 1e3)} ms, ${S.nF} frames of ${fmt(S.frameMs)} ms, ${S.keys.length} key frames`],
      ['Scored', sm.score_from_s ? `from ${fmt(sm.score_from_s * 1e3)} ms (the start-up before it is not counted)` : 'from the start'],
      ['Lock band', `±${fmt(sm.lock_threshold_pm)} pm (hysteresis ${fmt(sm.hysteresis_pm)} pm) about ${lr.kind === 'line' ? 'the declared line' : "each ring's own operating point"}`],
      ['World', 'g26 preset, nominal world (no sampled uncertainty, no device variation), one simulation tick = 1 µs'],
      ['Driven inputs', Object.values(ins).filter((s) => s.active).map((s) => (SIG[s.name] || [s.name])[0]).join(', ') || 'none'],
    ];
    this.facts.innerHTML = facts.map(([k, v]) => `<div><dt>${esc(k)}</dt><dd>${esc(v)}</dd></div>`).join('');
    this.renderPlot();
    this.renderTiming();
    this.renderEvents();
    this.renderSignals();
  }

  renderPlot() {
    const S = this.S;
    const ins = this.ins;
    const nF = S.nF;
    const panels = [];
    const ag = ins['thermal.aggressor_w'];
    if (ag) {
      const v = series(ag, nF);
      const d = S.meta.geometry.distances.ring_to_site_um;
      panels.push({ title: 'aggressor (W)', question: "When does a nearby heat source disturb the rings?", description: "Each trace is power dissipated at an aggressor site. A rise injects heat nearby; the ring temperature responds later through the thermal paths.", traces: ag.columns.map((c, j) => ({ y: v.map((row) => row[j]), name: `${c === 'aggressor' ? 'aggressor site' : 'stress site'} (${fmtNum(Math.min(...d[c]))} µm from R${d[c].indexOf(Math.min(...d[c]))})`, unit: 'W', color: cat(j === 0 ? 7 : 4), legend: true })) });
    }
    const tb = ins['thermal.t_base_k'];
    const xr = ins['thermal.xpu_rise_k'];
    const tbase = tb ? series(tb, nF).map((row) => row[0] - K0) : S.scn.drives.t_base_k.map((v) => v - K0);
    const tr = [{ y: tbase, name: 'die base temperature', unit: '°C', color: token('--series-2'), legend: true }];
    if (xr && xr.active) tr.push({ y: series(xr, nF).map((row) => row[0]), name: 'processor heat rise (K)', unit: 'K', color: cat(3), legend: true });
    panels.push({ title: 'base (°C)', question: "Does the thermal background change?", description: "The base trace sets the thermal background. If shown, processor heat is an additional temperature rise in K, not an absolute temperature.", traces: tr });
    const L = S.meta.n_lines;
    if (S.dev) {
      let pmax = 0;
      for (const row of S.dev.laser.p_chip_mw) for (const v of row) if (v > pmax) pmax = v;
      panels.push({ title: 'laser into chip (mW)', question: "Does the amount of incoming light change?", description: "One trace per laser line, after chip coupling and before the rings. Power changes affect both the monitor reading and optical self-heating; cold starts also show the turn-on ramp.", range: [0, pmax * 1.15], traces: Array.from({ length: L }, (_, l) => ({ y: colOf(S.dev.laser.p_chip_mw, l), name: `L${l}`, group: `L${l}`, unit: 'mW', color: cat(l), legend: true, width: 1.2 })) });
      panels.push({ title: 'laser f − grid (MHz)', question: "Does the laser frequency move?", description: "Each line’s offset from its nominal grid frequency. Zero means no frequency shift: a power change can still move the ring resonance through heating.", traces: Array.from({ length: L }, (_, l) => ({ y: colOf(S.dev.laser.f_dev_mhz, l), name: `L${l}`, group: `L${l}`, unit: 'MHz', color: cat(l), legend: false, width: 1.2 })) });
    }
    const de = ins['data.driver_enable'];
    const od = ins['data.ones_density'];
    const dt = [];
    if (de) dt.push({ y: series(de, nF).map((row) => row.reduce((a, b) => a + (b >= 0.5 ? 1 : 0), 0)), name: 'rings sending data', unit: 'rings', shape: 'hv', color: cat(2), legend: true, fmt: '.0f' });
    panels.push({ title: 'traffic (rings on)', question: "How many rings are sending data?", description: "The count of enabled data drivers. A step changes which rings carry traffic and dissipate driver heat.", traces: dt, range: [-0.5, S.nR + 0.5] });
    if (od) panels.push({ title: 'ones density', question: "Does the balance of zeros and ones change?", description: "One trace per ring shows its fraction of ones; 0.5 means equal proportions of zeros and ones. The selected ring is thicker; identical inputs overlap.", traces: Array.from({ length: S.nR }, (_, r) => ({ y: series(od, nF).map((row) => row[r]), name: `R${r} ones density`, color: cat(r), group: `R${r}`, width: r === this.ring ? 2.6 : 1.1, legend: true, fmt: '.3f' })), range: [-0.05, 1.05] });
    const faults = Object.values(ins).filter((s) => s.name.startsWith('fault.'));
    panels.push({ title: 'faults (rings)', question: "When are faults applied, and how widespread are they?", description: "One trace per fault type counts the affected rings. Zero means that fault is absent; the height is a ring count, not fault severity.", range: [-0.5, S.nR + 0.5], traces: faults.map((s, i) => ({ y: series(s, nF).map((row) => row.reduce((a, b) => a + (b !== 0 ? 1 : 0), 0)), name: (SIG[s.name] || [s.name])[0].replace('Fault: ', 'fault: '), shape: 'hv', fmt: '.0f', color: cat(i + 3), legend: true, width: 1.2 })) });
    const bv = ins['tx.bias_v'];
    if (bv) {
      const off = ins['tx.bias_offset_v'];
      const ov = off ? series(off, nF) : null;
      panels.push({ title: 'bias set point (V)', question: "Does the requested junction bias change?", description: "Each ring’s requested junction bias plus its scenario offset. Colors identify rings; the selected ring is thicker. This sets the data operating point, separately from heater power and individual bit voltage levels.", traces: Array.from({ length: S.nR }, (_, r) => ({ y: series(bv, nF).map((row, f) => row[r] + (ov ? ov[f][r] : 0)), name: `R${r} bias set point + offset`, group: `R${r}`, unit: 'V', color: cat(r), width: r === this.ring ? 2.6 : 1.1, legend: true, fmt: '.4f' })) });
    }
    this.stack.render(S, panels, { legend: true, ring: null });
    this.baseShapes = this.stack.layout.shapes.map((x) => ({ ...x }));
    this.panelsN = panels.length;
    if (!this.zoomBound) {
      this.zoomBound = true;
      this.stack.gd.on('plotly_relayout', (ev) => this.onZoom(ev));
    }
    this.zoomNote.textContent = '';
  }

  // controller steps as thin lines when the time window is below 1.5 ms
  onZoom(ev) {
    const S = this.S;
    const t = S.meta.devices.timing;
    if (!t || this.inZoom) return;
    const fl = this.stack.gd._fullLayout;
    if (!fl || !fl.xaxis) return;
    const [a, b] = fl.xaxis.range;
    const per = t.controller_period_ticks * t.dt_s * 1e3;
    const ph = t.controller_phase_ticks * t.dt_s * 1e3;
    const shapes = this.baseShapes.map((x) => ({ ...x }));
    if (b - a < 1.5) {
      for (let k = Math.ceil((a - ph) / per); ph + k * per <= b; k++) {
        const x = ph + k * per;
        this.stack.domains.forEach(([y0, y1]) => shapes.push({ type: 'line', xref: 'x', x0: x, x1: x, yref: 'paper', y0, y1, line: { width: 0.6, color: token('--axis') }, layer: 'below' }));
      }
      this.zoomNote.textContent = `Thin lines: controller steps, every ${fmtNum(per * 1e3)} µs (${Math.round((b - a) / per)} in this window).`;
    } else {
      this.zoomNote.textContent = ev && ev['xaxis.autorange'] ? '' : `Window ${fmtNum(b - a)} ms: zoom below 1.5 ms to see the controller steps.`;
    }
    this.inZoom = true;
    relayoutPlot(this.stack.gd, { shapes }).finally(() => { this.inZoom = false; });
  }

  renderTiming() {
    const S = this.S;
    const t = S.meta.devices.timing;
    if (!t) { this.timing.innerHTML = '<p class="sub">meta.json has no timing (regenerate the data).</p>'; return; }
    const us = t.dt_s * 1e6;
    const P = t.controller_period_ticks;
    const span = 3 * P;
    const W = 640;
    const H = 178;
    const x0 = 132;
    const X = (k) => x0 + ((W - x0 - 12) * k) / span;
    const rows = ['ADC integrates', 'ADC code ready', 'Controller step', 'Heater DAC output'];
    const y = (i) => 26 + i * 34;
    const ink = token('--text-primary');
    const ink2 = token('--text-secondary');
    const grid = token('--grid');
    const c1 = token('--series-1');
    const c2 = token('--series-2');
    let svg = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Timing of three controller steps">`;
    for (let k = 0; k <= span; k += 2) svg += `<line x1="${X(k)}" x2="${X(k)}" y1="12" y2="${H - 22}" stroke="${grid}" stroke-width="1"/>`;
    for (let k = 0; k <= span; k += 10) svg += `<text x="${X(k)}" y="${H - 6}" font-size="11" fill="${ink2}" text-anchor="middle">${k * us} µs</text>`;
    rows.forEach((r, i) => { svg += `<text x="4" y="${y(i) + 4}" font-size="12" fill="${ink}">${r}</text>`; });
    const ph = t.adc_phase_ticks[0] || 0;
    for (let s = ph; s <= span; s += t.adc_period_ticks) {
      if (s - t.adc_window_ticks >= 0) svg += `<rect x="${X(s - t.adc_window_ticks)}" y="${y(0) - 7}" width="${X(s) - X(s - t.adc_window_ticks)}" height="14" rx="2" fill="${c1}" opacity="0.55"/>`;
      const ready = s + t.adc_latency_ticks;
      if (ready <= span && s > 0) svg += `<circle cx="${X(ready)}" cy="${y(1)}" r="4" fill="${c1}"/>`;
    }
    for (let k = t.controller_phase_ticks; k <= span; k += P) {
      svg += `<path d="M${X(k)},${y(2) - 9} L${X(k)},${y(2) + 9}" stroke="${ink}" stroke-width="2.5"/>`;
      const d = k + (t.controller_latency_ticks || 0) + t.dac_latency_ticks;
      if (d <= span) svg += `<path d="M${X(d)},${y(3) - 8} l5,8 l-5,8 l-5,-8 z" fill="${c2}"/>`;
    }
    svg += '</svg>';
    this.timing.innerHTML = svg;
    const per = (n) => `${n * us} µs`;
    this.timingNote.textContent = `Ticks of ${us} µs. The ADC integrates the PD current for ${per(t.adc_window_ticks)} (bars), one sample every ${per(t.adc_period_ticks)}; each code is ready ${per(t.adc_latency_ticks)} after its sample (dots). ` +
      `The controller steps every ${per(P)} (${(P / t.adc_period_ticks).toFixed(1)} ADC samples per step) with ${per(t.controller_latency_ticks || 0)} latency; the heater DAC output changes ${per(t.dac_latency_ticks)} after the step (diamonds). All 8 rings sample at the same times.`;
  }

  renderEvents() {
    const S = this.S;
    const evs = [...S.scn.events];
    let h = '<thead><tr><th scope="col">Time (ms)</th><th scope="col">Event</th></tr></thead><tbody>';
    for (const e of evs) {
      const t = e.t_s * 1e3;
      h += `<tr data-t="${t}" tabindex="0"><td>${t.toFixed(3)}</td><td>${esc(e.label)}${e.value != null ? ` (${fmtNum(e.value)} ${esc(e.unit || '')})` : ''}</td></tr>`;
    }
    this.events.innerHTML = `${h}</tbody>`;
  }

  renderSignals() {
    const sigs = this.S.scn.inputs ? this.S.scn.inputs.signals : [];
    let h = '<thead><tr><th scope="col">Input</th><th scope="col">What it is</th><th scope="col">Applies</th><th scope="col">In this run</th><th scope="col">Value</th></tr></thead><tbody>';
    for (const s of sigs) {
      const [name, what] = SIG[s.name] || [s.name, ''];
      const varies = !s.constant;
      h += `<tr title="${esc(s.name)}: ${esc(s.doc)}"><th scope="row">${esc(name)}</th><td>${esc(what)}</td><td>${esc(LEVEL[s.level] || s.level)}</td>` +
        `<td>${s.active ? (varies ? '<b>driven, changes</b>' : '<b>driven, constant</b>') : 'not driven'}</td><td>${esc(valueText(s))}</td></tr>`;
    }
    this.signals.innerHTML = `${h}</tbody>`;
  }

  setTime(t) { this.stack.setTime(t); }
}
