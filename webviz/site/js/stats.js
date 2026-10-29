// "Statistics": numbers of the whole run (or the scored time) for every part: min, percentiles, mean, max, rms
// and spread, time in each state and counts, per ring and for the bank. Every number is from stats_<id>.json,
// which generate.py computes from the engine's 10 us series (README, "stats_<id>.json").

import { esc, fmtNum, sci, bindSegs } from './ui.js';

const PARTS = [['ring', 'Ring'], ['heater', 'Heater'], ['pd', 'Photodiode and ADC'], ['laser', 'Laser lines'], ['ctrl', 'Controller I/O'], ['data', 'Data path']];
const STAT_LABEL = { min: 'min', p5: '5 %', p50: 'median', mean: 'mean', p95: '95 %', max: 'max', rms: 'rms', std: 'spread (std)' };
const STAT_NOTE = 'min / max: the extremes; 5 % / median / 95 %: the percentiles of the 10 µs values; rms: root mean square; spread: the standard deviation about the mean.';

function val(v, q) {
  if (v == null) return '–';
  if (q.key === 'ber') return sci(v);
  return fmtNum(v);
}

export class StatsView {
  constructor(root, { onPart }) {
    this.root = root;
    this.card = root.querySelector('#cardStats');
    this.table = this.card.querySelector('[data-role=table]');
    this.title = this.card.querySelector('[data-role=title]');
    this.foot = this.card.querySelector('[data-role=foot]');
    this.basis = this.card.querySelector('[data-role=basis]');
    this.pick = this.card.querySelector('[data-role=statpick]');
    this.sel = this.pick.querySelector('select');
    this.states = root.querySelector('[data-role=states]');
    this.counts = root.querySelector('[data-role=counts]');
    this.stSub = root.querySelector('#cardStates [data-role=sub]');
    this.win = 'scored';
    this.part = 'ring';
    this.stat = 'mean';
    this.onPart = onPart;
    this.sel.innerHTML = Object.entries(STAT_LABEL).map(([k, v]) => `<option value="${k}">${esc(v)}</option>`).join('');
    this.sel.value = this.stat;
    this.sel.addEventListener('change', () => { this.stat = this.sel.value; this.render(); });
    this.markWin = bindSegs(this.card.querySelector('[data-role=window]'), this.win, (v) => { this.win = v; this.render(); });
    const tabs = this.card.querySelector('[data-role=parts]');
    tabs.innerHTML = PARTS.map(([id, l]) => `<button type="button" role="tab" data-id="${id}" aria-pressed="${id === this.part}">${esc(l)}</button>`).join('');
    tabs.addEventListener('click', (e) => {
      const b = e.target.closest('button[data-id]');
      if (!b) return;
      this.setPart(b.dataset.id);
      this.onPart(b.dataset.id);
    });
    this.tabs = tabs;
  }

  setPart(id) {
    const ok = PARTS.some(([p]) => p === id);
    this.part = ok ? id : 'ring';
    this.tabs.querySelectorAll('button').forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.id === this.part)));
    if (this.S) this.render();
  }

  setState(S, r, all) { this.S = S; this.r = r; this.all = all; }

  render() {
    const S = this.S;
    if (!S) return;
    const st = S.st;
    if (!st) { this.table.innerHTML = '<tbody><tr><td>stats file missing: regenerate the data (npm run regen).</td></tr></tbody>'; return; }
    const w = this.win;
    const span = w === 'run' ? `the whole run (0 to ${fmtNum(st.windows.run[1] * 1e3)} ms)` : `the scored time (${fmtNum(st.windows.scored[0] * 1e3)} to ${fmtNum(st.windows.scored[1] * 1e3)} ms)`;
    this.basis.textContent = `Over ${span}: ${st.samples} samples of 10 µs (one per controller step). Each sample is the mean over its ten 1 µs ticks (or the kernel state at the controller step, for the heater, dark current, laser and command values). ${STAT_NOTE}`;
    this.tabs.querySelectorAll('button').forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.id === this.part)));
    const qs = st.quantities.filter((q) => q.part === this.part);
    const lineMode = this.part === 'laser';
    const cols = this.all || lineMode;
    this.pick.hidden = !cols;
    if (!cols) this.renderRing(qs, w);
    else this.renderCols(qs, w, lineMode);
    this.renderStates(st, w);
  }

  renderRing(qs, w) {
    const r = this.r;
    this.title.textContent = `Ring R${this.r}: statistics of each quantity (bank values are the sum of the 8 heaters)`;
    let h = '<thead><tr><th scope="col">Quantity</th><th scope="col">Unit</th>';
    for (const s of this.S.st.stats) h += `<th scope="col">${esc(STAT_LABEL[s])}</th>`;
    h += '</tr></thead><tbody>';
    for (const q of qs) {
      const d = q[w];
      if (!d) continue;
      const bank = q.per === 'bank';
      h += `<tr><th scope="row"><div class="mn">${esc(q.label)}${bank ? ' (bank)' : ''}</div><div class="md">${esc(q.note)}</div></th><td>${esc(q.unit)}</td>`;
      for (const s of this.S.st.stats) h += `<td>${val(d[s][bank ? 0 : r], q)}</td>`;
      h += '</tr>';
    }
    this.table.innerHTML = `${h}</tbody>`;
    this.foot.textContent = 'Select "All" in the ring buttons to compare the rings for one statistic. –: not defined for this quantity (rms and spread of a dB or BER value).';
  }

  renderCols(qs, w, lineMode) {
    const s = this.stat;
    const n = lineMode ? this.S.meta.n_lines : this.S.nR;
    const tag = lineMode ? 'L' : 'R';
    this.title.textContent = `${STAT_LABEL[s][0].toUpperCase()}${STAT_LABEL[s].slice(1)} of each quantity, every ${lineMode ? 'laser line' : 'ring'}`;
    let h = '<thead><tr><th scope="col">Quantity</th><th scope="col">Unit</th>';
    for (let i = 0; i < n; i++) h += `<th scope="col" class="${!lineMode && i === this.r ? 'selc' : ''}">${tag}${i}</th>`;
    h += `<th scope="col">${lineMode ? 'All lines' : 'All rings'}</th></tr></thead><tbody>`;
    for (const q of qs) {
      const d = q[w];
      if (!d) continue;
      h += `<tr><th scope="row"><div class="mn">${esc(q.label)}${q.per === 'bank' ? ' (bank)' : ''}</div><div class="md">${esc(q.note)}</div></th><td>${esc(q.unit)}</td>`;
      if (q.per === 'bank') {
        h += `<td colspan="${n}" class="muted">one value for the bank →</td><td>${val(d[s][0], q)}</td>`;
      } else {
        for (let i = 0; i < n; i++) h += `<td class="${!lineMode && i === this.r ? 'selc' : ''}">${val(d[s][i], q)}</td>`;
        const pl = q[`${w}_pooled`];
        h += `<td>${pl ? val(pl[s], q) : '–'}</td>`;
      }
      h += '</tr>';
    }
    this.table.innerHTML = `${h}</tbody>`;
    this.foot.textContent = `"All ${lineMode ? 'lines' : 'rings'}": the statistic of every ${lineMode ? 'line' : 'ring'}'s samples taken together. Change the statistic at the top right.`;
  }

  renderStates(st, w) {
    const N = this.S.nR;
    this.stSub.textContent = `Time in ms within ${w === 'run' ? 'the whole run' : 'the scored time'}, per ring. For the bank: the sum over the rings (or the one bank value).`;
    let h = '<thead><tr><th scope="col">State</th>';
    for (let i = 0; i < N; i++) h += `<th scope="col" class="${i === this.r ? 'selc' : ''}">R${i}</th>`;
    h += '<th scope="col">Bank</th></tr></thead><tbody>';
    const row = (label, note, vals, entries) => {
      let x = `<tr><th scope="row"><div class="mn">${esc(label)}</div><div class="md">${esc(note)}</div></th>`;
      if (!vals) return `${x}<td colspan="${N + 1}" class="muted">scored time only (the engine counts from the scoring start)</td></tr>`;
      if (vals.length === 1) return `${x}<td colspan="${N}" class="muted">one value for the bank →</td><td>${fmtNum(vals[0])}</td></tr>`;
      vals.forEach((v, i) => {
        const e = entries ? entries[i] : null;
        x += `<td class="${i === this.r ? 'selc' : ''}">${v ? `<b>${fmtNum(v)}</b>` : fmtNum(v)}${e ? `<div class="md">${e} start${e === 1 ? '' : 's'}</div>` : ''}</td>`;
      });
      return `${x}<td>${fmtNum(vals.reduce((a, b) => a + (b || 0), 0))}</td></tr>`;
    };
    for (const s of st.states) {
      h += row(s.label, s.note, s[w], s[`${w}_entries`]);
      if (s.engine_scored && w === 'scored') h += row(`${s.label} (engine)`, 'The engine\'s own metric, every 1 µs tick.', s.engine_scored, null);
    }
    this.states.innerHTML = `${h}</tbody>`;
    let c = '<thead><tr><th scope="col">Count</th><th scope="col">Window</th>';
    for (let i = 0; i < N; i++) c += `<th scope="col" class="${i === this.r ? 'selc' : ''}">R${i}</th>`;
    c += '<th scope="col">Bank</th></tr></thead><tbody>';
    for (const k of st.counts) {
      const useW = k[w] ? w : (k.scored ? 'scored' : 'run');
      const v = k[useW];
      if (!v) continue;
      c += `<tr><th scope="row"><div class="mn">${esc(k.label)}</div><div class="md">${esc(k.note)}</div></th><td>${useW === 'run' ? 'whole run' : 'scored'}</td>`;
      v.forEach((x, i) => { c += `<td class="${i === this.r ? 'selc' : ''}">${x ? `<b>${fmtNum(x)}</b>` : fmtNum(x)}</td>`; });
      const isMax = k.key === 'excursion_max_s';
      c += `<td>${fmtNum(isMax ? Math.max(...v) : v.reduce((a, b) => a + (b || 0), 0))}${isMax ? ' (max)' : ''}</td></tr>`;
    }
    this.counts.innerHTML = `${c}</tbody>`;
  }
}
