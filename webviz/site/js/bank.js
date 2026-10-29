// "Bank overview": one row per ring at the scrubber frame (line, lock state, detuning, lock error, temperatures,
// heater and monitor codes, relOMA, BER, self-heating loop gain) and a sparkline of one chosen quantity over the
// whole run, with the scrubber as a vertical line. Values are read from the data files.

import { fmt, sgn, sci, esc, cat, sparkSVG, setSparkCursor, K0 } from './ui.js';
import { lineStatus, lockState } from './panels.js';

const SPARK = {
  err: { label: 'lock error (pm)', get: (S, r) => S.err.map((row) => row[r]), refs: (S) => [S.thr, -S.thr, 0] },
  det: { label: 'detuning (pm)', get: (S, r) => S.scn.rings.detuning_pm.map((row) => row[r]), refs: () => [0] },
  temp: { label: 'ring temperature (°C)', get: (S, r) => S.scn.rings.t_ring_k.map((row) => row[r] - K0), refs: () => [] },
  code: { label: 'heater code', get: (S, r) => S.scn.rings.heater_code.map((row) => row[r]), refs: (S) => [S.meta.devices.heater.code_lo, S.meta.devices.heater.code_hi] },
  ph: { label: 'heater power (mW)', get: (S, r) => S.scn.rings.p_heater_mw.map((row) => row[r]), refs: () => [] },
  mon: { label: 'monitor code', get: (S, r) => S.scn.rings.monitor_code.map((row) => row[r]), refs: (S) => [S.meta.devices.monitor.code_max] },
  rel: { label: 'relOMA (dB)', get: (S, r) => S.scn.rings.reloma_db.map((row) => row[r]), refs: () => [] },
};

const SPARK_COPY = {
  "err": [
    "Does each ring stay close to its lock reference?",
    "Lock error over time; horizontal references mark zero and the lock-band edges when visible."
  ],
  "det": [
    "How does resonance move relative to the laser?",
    "Resonance-minus-laser detuning over time. Zero means alignment; positive means the resonance is at a longer wavelength."
  ],
  "temp": [
    "Which rings warm or cool during the run?",
    "Each ring’s temperature over time. Rising curves mean warming; compare when the changes occur across rings."
  ],
  "code": [
    "How does the controller use each heater’s range?",
    "Heater register code over time. Reference lines mark the low and high rails when visible."
  ],
  "ph": [
    "How much heating does each ring need?",
    "Electrical heater power over time. A larger value means more electrical heat supplied to that ring."
  ],
  "mon": [
    "What monitor reading does each controller receive?",
    "Released photodiode ADC codes over time. The full-scale reference appears when in range; high readings can include optical power changes and background current."
  ],
  "rel": [
    "Does each ring maintain useful data contrast?",
    "Normalized optical modulation amplitude over time, in dB. Higher means more separation between optical data levels per unit input light."
  ]
};

export class BankView {
  constructor(card, { onPick }) {
    this.card = card;
    this.table = card.querySelector('table');
    this.sel = card.querySelector('select');
    this.onPick = onPick;
    this.kind = 'err';
    this.sel.innerHTML = Object.entries(SPARK).map(([k, v]) => `<option value="${k}">${esc(v.label)}</option>`).join('');
    this.sel.addEventListener('change', () => { this.kind = this.sel.value; this.render(this.S, this.ring); this.setFrame(this.f); });
    this.table.addEventListener('click', (e) => { const tr = e.target.closest('tr[data-ring]'); if (tr) this.onPick(Number(tr.dataset.ring)); });
    this.table.addEventListener('keydown', (e) => {
      const tr = e.target.closest('tr[data-ring]');
      if (tr && (e.key === 'Enter' || e.key === ' ')) { e.preventDefault(); this.onPick(Number(tr.dataset.ring)); }
    });
  }

  render(S, ring) {
    this.S = S;
    this.ring = ring;
    const sp = SPARK[this.kind];
    const [question, description] = SPARK_COPY[this.kind];
    this.card.querySelector('[data-role=spark-question]').textContent = question;
    this.card.querySelector('[data-role=spark-description]').textContent = description + ' Read the last column left to right across the whole run; the vertical line marks the scrubber. Each row has its own vertical scale, so compare numbers in the table to judge magnitudes.';
    let h = '<thead><tr><th scope="col">Ring</th><th scope="col">Line (declared → on)</th><th scope="col">Lock state</th>' +
      '<th scope="col">Detuning (pm)</th><th scope="col">Lock error (pm)</th><th scope="col">Ring (°C)</th><th scope="col">PD (°C)</th>' +
      '<th scope="col">Heater code</th><th scope="col">Heater (mW)</th><th scope="col">Monitor code</th><th scope="col">relOMA (dB)</th>' +
      '<th scope="col">BER</th><th scope="col">Loop gain</th>' +
      `<th scope="col">${esc(sp.label)} over the run</th></tr></thead><tbody>`;
    for (let r = 0; r < S.nR; r++) {
      h += `<tr data-ring="${r}" tabindex="0" class="${r === ring ? 'sel' : ''}"><th scope="row"><span class="sw" style="background:${cat(r)}"></span>R${r}</th>` +
        '<td class="c-line"></td><td class="c-lock"></td><td class="c-det"></td><td class="c-err"></td><td class="c-tr"></td><td class="c-tp"></td>' +
        '<td class="c-code"></td><td class="c-ph"></td><td class="c-mon"></td><td class="c-rel"></td><td class="c-ber"></td><td class="c-lg"></td>' +
        `<td class="c-spark">${sparkSVG(sp.get(S, r), { w: 144, h: 36, color: cat(r), refs: sp.refs(S) })}</td></tr>`;
    }
    this.table.innerHTML = `${h}</tbody>`;
    this.rows = [...this.table.querySelectorAll('tbody tr')].map((tr) => {
      const q = (c) => tr.querySelector(c);
      return { tr, line: q('.c-line'), lock: q('.c-lock'), det: q('.c-det'), err: q('.c-err'), tr_: q('.c-tr'), tp: q('.c-tp'), code: q('.c-code'), ph: q('.c-ph'), mon: q('.c-mon'), rel: q('.c-rel'), ber: q('.c-ber'), lg: q('.c-lg') };
    });
  }

  setRing(r) {
    this.ring = r;
    if (this.rows) this.rows.forEach((x, i) => x.tr.classList.toggle('sel', i === r));
  }

  setFrame(f) {
    this.f = f;
    const S = this.S;
    if (!S || !this.rows) return;
    const R = S.scn.rings;
    const h = S.meta.devices.heater;
    const mo = S.meta.devices.monitor;
    const k = S.nearestKey[f];
    const set = (el, html) => { if (el._v !== html) { el._v = html; el.innerHTML = html; } };
    this.rows.forEach((x, r) => {
      const st = lineStatus(S, r, f);
      set(x.line, `L${R.declared_line[f][r]} → ${R.line_on[f][r] < 0 ? 'none' : `L${R.line_on[f][r]}`}${st.flag ? ` <span class="tag">${esc(st.flag.toLowerCase())}</span>` : ''}`);
      const lk = S.dev ? lockState(S, r, f) : { v: '–' };
      set(x.lock, lk.flag ? `<span class="tag">${lk.v}</span>` : lk.v);
      set(x.det, sgn(R.detuning_pm[f][r], 2));
      const e = S.err[f][r];
      set(x.err, Math.abs(e) > S.thr ? `<b>${sgn(e, 2)}</b> <span class="tag">out</span>` : sgn(e, 2));
      set(x.tr_, fmt(R.t_ring_k[f][r] - K0, '', 2));
      set(x.tp, fmt(R.t_pd_k[f][r] - K0, '', 2));
      const c = R.heater_code[f][r];
      set(x.code, `${c}${c <= h.code_lo || c >= h.code_hi ? ' <span class="tag">rail</span>' : ''}`);
      set(x.ph, fmt(R.p_heater_mw[f][r]));
      const m = R.monitor_code[f][r];
      set(x.mon, `${m}${R.monitor_code_max[f][r] >= mo.code_max ? ' <span class="tag">clipped</span>' : ''}`);
      set(x.rel, fmt(R.reloma_db[f][r]));
      set(x.ber, sci(R.ber[f][r]));
      set(x.lg, S.dev ? fmt(S.dev.key.selfheat_loop_gain[k][r]) : '–');
    });
    setSparkCursor(this.table, S.nF > 1 ? f / (S.nF - 1) : 0);
  }
}
