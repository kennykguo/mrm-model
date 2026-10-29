// "Hardware limits": the limits the model assumes (meta.hardware_limits: value, unit, provenance from the g26
// preset) and, for the open scenario, how near the run comes to each one (stats_<id>.json limits: closest
// approach, time at the limit, first time reached), with a plot of the margin over the run from the frame data.

import { TimeStack, esc, fmtNum, prov, cat, bindSegs } from './ui.js';
import { token } from './theme.js';

const fmtVal = (v, unit) => {
  if (v == null) return '–';
  if (Array.isArray(v)) return v.length === 2 && unit !== '×' ? `${fmtNum(v[0])} to ${fmtNum(v[1])}` : v.map(fmtNum).join(', ');
  if (typeof v === 'number' && v !== 0 && Math.abs(v) < 1e-3) return v.toExponential(0);
  return typeof v === 'number' ? fmtNum(v) : String(v);
};

// margin over the run from the frame arrays (the same quantities as the 10 us limits, at frame resolution)
function marginSeries(S, key) {
  const R = S.scn.rings;
  const D = S.dev;
  const h = S.meta.devices.heater;
  const mo = S.meta.devices.monitor;
  const N = S.nR;
  const per = (fn) => Array.from({ length: N }, (_, q) => S.tMs.map((_, f) => fn(f, q)));
  const one = (fn) => [S.tMs.map((_, f) => fn(f))];
  const lim = Object.fromEntries((S.st ? S.st.limits : []).map((l) => [l.key, l]));
  switch (key) {
    case 'heater_lo': return per((f, q) => R.heater_code_min[f][q] - h.code_lo);
    case 'heater_hi': return per((f, q) => h.code_hi - R.heater_code_max[f][q]);
    case 'heater_p': return per((f, q) => D.heater.p_full_code_mw[f][q] - R.p_heater_mw[f][q]);
    case 'compliance': return per((f, q) => h.compliance_v - D.heater.v_v[f][q]);
    case 'bank_p': return one((f) => h.supply_power_max_w * 1e3 - D.heater.bank_p_mw_max[f]);
    case 'bank_i': return one((f) => h.supply_current_max_a * 1e3 - D.heater.bank_i_ma_max[f]);
    case 'adc_fs': return per((f, q) => mo.code_max - R.monitor_code_max[f][q]);
    case 'adc_zero': return per((f, q) => R.monitor_code_min[f][q]);
    case 'lock_band': return per((f, q) => S.thr - Math.max(Math.abs(R.detuning_min_pm[f][q] - S.dRef[q]), Math.abs(R.detuning_max_pm[f][q] - S.dRef[q])));
    case 'ber': return per((f, q) => Math.log10(S.scn.summary.ber_limit) - Math.log10(Math.max(R.ber[f][q], 1e-300)));
    case 'bias_lo': return per((f, q) => R.bias_v[f][q] - lim.bias_lo.value);
    case 'bias_hi': return per((f, q) => lim.bias_hi.value - R.bias_v[f][q]);
    default: return null;
  }
}

const LIMIT_PLOTS = {
  "heater_lo": [
    "Can the heater reduce its code any further?",
    "Distance from each frame’s lowest heater code to the low rail."
  ],
  "heater_hi": [
    "Can the heater increase its code any further?",
    "Distance from each frame’s highest heater code to the high rail."
  ],
  "heater_p": [
    "How much additional heater power is available?",
    "Full-code power at the heater’s current temperature minus its frame-mean power."
  ],
  "compliance": [
    "How close is the heater voltage to its compliance limit?",
    "Compliance voltage minus the applied heater voltage."
  ],
  "bank_p": [
    "Do all heaters together exceed the supply power limit?",
    "Supply power limit minus the highest total heater power in each frame. This is one shared bank margin."
  ],
  "bank_i": [
    "Do all heaters together exceed the supply current limit?",
    "Supply current limit minus the highest total heater current in each frame. This is one shared bank margin."
  ],
  "adc_fs": [
    "How close does the monitor come to full scale?",
    "Full-scale code minus the highest released ADC code in each frame. Zero means no upward code range remains."
  ],
  "adc_zero": [
    "How close does the monitor come to zero?",
    "The lowest released ADC code in each frame: its distance above zero."
  ],
  "lock_band": [
    "Does the ring stay inside the lock band?",
    "Allowed error magnitude minus the largest absolute lock error in each frame. Negative values mean the band was crossed."
  ],
  "ber": [
    "How far below the BER limit is the link?",
    "The logarithmic gap between the BER limit and frame-mean BER. A margin of 1 means ten times below the limit; −1 means ten times above it."
  ],
  "bias_lo": [
    "How close is the junction bias to its lower limit?",
    "Frame-mean junction bias minus its lower limit."
  ],
  "bias_hi": [
    "How close is the junction bias to its upper limit?",
    "Upper bias limit minus frame-mean junction bias."
  ],
  "selfheat": [
    "How close is optical self-heating to its stability boundary?",
    "One minus the self-heating loop gain, evaluated only at saved key frames. Zero marks the local static stability boundary; negative values indicate an unstable equilibrium."
  ]
};

export class LimitsView {
  constructor(root, { onSeek }) {
    this.root = root;
    this.reach = root.querySelector('#cardReach [data-role=table]');
    this.title = root.querySelector('#cardReach [data-role=title]');
    this.stack = new TimeStack(root.querySelector('#cardReach .plotwrap'), { onSeek, height: 230 });
    this.hw = root.querySelector('#cardHw [data-role=table]');
    this.hwFoot = root.querySelector('#cardHw [data-role=foot]');
    this.win = 'scored';
    this.sel = 'heater_lo';
    bindSegs(root.querySelector('#cardReach [data-role=window]'), this.win, (v) => { this.win = v; this.render(); });
    this.reach.addEventListener('click', (e) => {
      const tr = e.target.closest('tr[data-key]');
      if (tr) { this.sel = tr.dataset.key; this.render(); }
    });
    this.reach.addEventListener('keydown', (e) => {
      const tr = e.target.closest('tr[data-key]');
      if (tr && (e.key === 'Enter' || e.key === ' ')) { e.preventDefault(); this.sel = tr.dataset.key; this.render(); }
    });
  }

  renderHardware(meta) {
    const hl = meta.hardware_limits;
    if (!hl) { this.hw.innerHTML = '<tbody><tr><td>meta.json has no hardware limits: regenerate the data (npm run regen).</td></tr></tbody>'; return; }
    let h = '<thead><tr><th scope="col">Part</th><th scope="col">Limit</th><th scope="col">Value</th><th scope="col">Unit</th><th scope="col">Note</th></tr></thead><tbody>';
    let last = '';
    for (const r of hl.rows) {
      h += `<tr class="${r.group !== last ? 'grp' : ''}"><th scope="row">${r.group !== last ? esc(r.group) : ''}</th><td>${esc(r.label)}${r.key ? `<div class="md">${esc(r.key)}</div>` : ''}</td>` +
        `<td><b>${esc(fmtVal(r.value, r.unit))}</b>${prov(r.provenance)}</td><td>${esc(r.unit)}</td><td class="md">${esc(r.note || '')}</td></tr>`;
      last = r.group;
    }
    this.hw.innerHTML = `${h}</tbody>`;
    this.hwFoot.textContent = `Provenance: the preset's own tag (printed, inferred, literature, assumed), or derived (model output from the ${hl.junction_from || 'steady'} run), model default (a default of the model code), preset / model setting (a setting without a tag).`;
  }

  setState(S, r) { this.S = S; this.r = r; }

  render() {
    const S = this.S;
    if (!S) return;
    if (!S.st) { this.reach.innerHTML = '<tbody><tr><td>stats file missing: regenerate the data (npm run regen).</td></tr></tbody>'; return; }
    const w = this.win;
    let h = '<thead><tr><th scope="col">Limit</th><th scope="col">Limit value</th><th scope="col">Closest approach</th><th scope="col">Reached by</th><th scope="col">Time at the limit</th><th scope="col">First reached</th></tr></thead><tbody>';
    for (const l of S.st.limits) {
      const d = l[w];
      const u = l.unit ? ` ${l.unit}` : '';
      if (!d) { h += `<tr data-key="${l.key}"><th scope="row">${esc(l.label)}</th><td colspan="5" class="muted">no samples in this window</td></tr>`; continue; }
      const mm = d.margin_min;
      let wi = 0;
      mm.forEach((v, i) => { if (v != null && (mm[wi] == null || v < mm[wi])) wi = i; });
      const who = l.per === 'bank' ? 'bank' : `R${wi}`;
      const hit = d.reached.map((x, i) => (x ? (l.per === 'bank' ? 'bank' : `R${i}`) : null)).filter(Boolean);
      const tAt = d.time_at_ms ? Math.max(...d.time_at_ms.map((v) => v || 0)) : null;
      const firsts = d.first_ms.filter((v) => v != null);
      const reached = hit.length > 0;
      h += `<tr data-key="${l.key}" tabindex="0" class="${l.key === this.sel ? 'sel' : ''}"><th scope="row"><div class="mn">${esc(l.label)}${reached ? ' <span class="tag">REACHED</span>' : ''}</div><div class="md">${esc(l.note)}</div></th>` +
        `<td>${l.value == null ? 'per heater temperature' : esc(fmtVal(l.value, l.unit) + (l.unit === 'decades' ? '' : u))}</td>` +
        `<td>${fmtNum(mm[wi])}${esc(u)} <span class="md">(${who} at ${fmtNum(d.t_margin_min_ms[wi])} ms)</span></td>` +
        `<td>${reached ? esc(hit.length > 4 ? `${hit.length} rings` : hit.join(', ')) : 'none'}</td>` +
        `<td>${tAt == null ? (l.basis === 'key frames' ? 'key frames only' : '–') : `${fmtNum(tAt)} ms${reached && l.per !== 'bank' ? ` <span class="md">(longest: R${d.time_at_ms.indexOf(tAt)})</span>` : ''}`}</td>` +
        `<td>${firsts.length ? `${fmtNum(Math.min(...firsts))} ms` : '–'}</td></tr>`;
    }
    this.reach.innerHTML = `${h}</tbody>`;
    this.renderPlot();
  }

  renderPlot() {
    const S = this.S;
    const l = S.st.limits.find((x) => x.key === this.sel) || S.st.limits[0];
    this.sel = l.key;
    const u = l.unit ? ` (${l.unit})` : '';
    let traces;
    if (l.key === 'selfheat') {
      const kt = S.keys.map((f) => S.tMs[f]);
      traces = Array.from({ length: S.nR }, (_, q) => ({ x: kt, y: S.dev.key.selfheat_loop_gain.map((row) => 1 - row[q]), name: `R${q}`, group: `R${q}`, color: cat(q), width: q === this.r ? 2.6 : 1.1, mode: 'lines+markers', marker: { size: 4, color: cat(q) } }));
    } else {
      const ser = marginSeries(S, l.key);
      if (!ser) { this.title.textContent = ''; return; }
      traces = ser.map((y, q) => ({ y, name: ser.length === 1 ? 'bank' : `R${q}`, group: ser.length === 1 ? 'bank' : `R${q}`, color: ser.length === 1 ? token('--series-1') : cat(q), width: ser.length === 1 || q === this.r ? 2.6 : 1.1, unit: l.unit }));
    }
    this.title.textContent = `${l.label}: margin over the run${u}, frame values. 0 = at the limit; below 0 = past it.`;
    const [question, explanation] = LIMIT_PLOTS[l.key];
    const description = explanation + ' Zero is the limit; positive means room remains, negative means past it.' + (l.per === 'bank' ? '' : ' Each color is a ring; the selected ring is thicker.');
    this.stack.render(S, [{ title: `margin${u}`, question, description, traces, hlines: [{ y: 0, label: 'limit', color: token('--critical') }] }], { legend: true, ring: this.r });
  }

  setTime(t) { this.stack.setTime(t); }
}

