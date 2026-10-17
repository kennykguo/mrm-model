// "How well is it locked overall?" The engine's own run metrics (every 1 us tick after scoring starts), one
// row per ring. A value outside the lock band is marked with text, not colour alone.

import { fmtNum } from './traces.js';

export class SummaryView {
  constructor(card, { onPick }) {
    this.card = card;
    this.head = card.querySelector('.readout');
    this.table = card.querySelector('table');
    this.onPick = onPick;
    this.table.addEventListener('click', (e) => {
      const tr = e.target.closest('tr[data-ring]');
      if (tr) this.onPick(Number(tr.dataset.ring));
    });
    this.table.addEventListener('keydown', (e) => {
      const tr = e.target.closest('tr[data-ring]');
      if (tr && (e.key === 'Enter' || e.key === ' ')) { e.preventDefault(); this.onPick(Number(tr.dataset.ring)); }
    });
  }

  render(S, ring) {
    const sm = S.scn.summary;
    const pr = sm.per_ring;
    const thr = sm.lock_threshold_pm;
    const lastF = S.nF - 1;
    const ref = S.scn.lock_reference.kind === 'line' ? 'the declared line' : "each ring's own operating point";
    const score = sm.score_from_s ? `from ${fmtNum(sm.score_from_s * 1e3)} ms` : 'from the start';
    this.head.textContent = `${sm.controller} · lock band ±${fmtNum(thr)} pm about ${ref} (hysteresis ${fmtNum(sm.hysteresis_pm)} pm) · ` +
      `scored ${score} to ${fmtNum(sm.duration_s * 1e3)} ms · unlocked at the end: ${sm.unlocked_rings} · ` +
      `on a wrong line: ${sm.wrong_line_rings} · excursions: ${sm.excursions_total}`;
    const ms = (v) => (v > 0 ? fmtNum(v * 1e3) : '0');
    let html = '<thead><tr><th scope="col">Ring</th><th scope="col">Line</th>' +
      '<th scope="col">Max |lock error| (pm)</th><th scope="col">RMS lock error (pm)</th>' +
      '<th scope="col">OMA penalty (dB)</th><th scope="col">Unlocked after first lock (ms)</th>' +
      '<th scope="col">Heater at code 0 (ms)</th><th scope="col">Locked at end</th></tr></thead><tbody>';
    for (let r = 0; r < S.nR; r++) {
      const maxe = pr.max_abs_lock_error_pm[r];
      const out = maxe > thr;
      const line = S.scn.rings.declared_line[lastF][r];
      html += `<tr data-ring="${r}" tabindex="0" class="${r === ring ? 'sel' : ''}" aria-selected="${r === ring}">` +
        `<th scope="row">R${r}</th><td>${line >= 0 ? `L${line}` : '–'}</td>` +
        `<td class="${out ? 'flag' : ''}">${fmtNum(maxe)}${out ? ' <span class="tag">▲ out of band</span>' : ''}</td>` +
        `<td>${fmtNum(pr.rms_lock_error_pm[r])}</td><td>${fmtNum(pr.oma_penalty_db[r])}</td>` +
        `<td>${ms(pr.time_unlocked_s[r])}</td><td>${ms(pr.heater_low_rail_s[r])}</td>` +
        `<td>${pr.locked_final[r] ? 'yes' : '<span class="tag">no</span>'}</td></tr>`;
    }
    html += '</tbody>';
    this.table.innerHTML = html;
    const warn = (sm.warnings || []).join('; ');   // textContent: no escaping needed
    this.card.querySelector('.note').textContent = warn ? `Engine warnings: ${warn}` : '';
  }

  setRing(r) {
    this.table.querySelectorAll('tr[data-ring]').forEach((tr) => {
      const on = Number(tr.dataset.ring) === r;
      tr.classList.toggle('sel', on);
      tr.setAttribute('aria-selected', String(on));
    });
  }
}
