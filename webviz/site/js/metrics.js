// "Metrics": the main optimisation metrics of a lock controller, per ring and for the worst ring, with the
// direction of each (minimise / maximise / must be 0), a one-sentence definition and the unit. Every value is the
// engine's own metric (mrm/sim/metrics.py: streaming accumulators over every 1 us tick after scoring starts);
// "over the run" plots the same accumulators copied at every frame end (devices_<id>.json running.*).

import { TimeStack, fmt, sci, esc, cat } from './ui.js';

const ms = (v) => (v == null ? null : v * 1e3);

// goal: 'min' | 'max' | 'zero' | 'ref' (a reference value, no direction)
function defs(S) {
  const sm = S.scn.summary;
  const thr = sm.lock_threshold_pm;
  const hy = sm.hysteresis_pm;
  const h = S.meta.devices.heater;
  const pr = sm.per_ring;
  const run = S.dev ? S.dev.running : null;
  const scoredS = sm.duration_s - (sm.score_from_s || 0);
  const runCol = (key, s = 1) => (run ? run[key].map((row) => (row == null ? null : row.map((v) => (v == null ? null : v * s)))) : null);
  const headroom = (lo, hi) => lo.map((v, i) => (v == null || hi[i] == null ? null : Math.min(v - h.code_lo, h.code_hi - hi[i])));
  return [
    { key: 'oma', name: 'OMA penalty vs the oracle', goal: 'min', unit: 'dB', vals: pr.oma_penalty_db,
      def: 'How much less modulation the ring gave than the best stable operating point at the same instants: 10 log10(time-sum of the oracle relOMA / time-sum of the ring\'s relOMA).', run: runCol('oma_penalty_db') },
    { key: 'wrongf', name: 'Rings on a foreign line at the end', goal: 'zero', unit: 'ring', vals: pr.wrong_line_final, bool: true,
      def: 'At the end the ring sits within 1 FWHM of a laser line that is not its declared line (it would carry another channel\'s data).' },
    { key: 'wrongs', name: 'Time on a foreign line', goal: 'zero', unit: 'ms', vals: pr.wrong_line_s.map(ms),
      def: 'Total time the ring sat within 1 FWHM of a line that is not its declared line.', run: runCol('wrong_line_s', 1e3) },
    { key: 'wronge', name: 'Foreign-line entries', goal: 'zero', unit: 'count', vals: pr.wrong_line_events,
      def: 'Number of times the ring moved onto a foreign line.', run: runCol('wrong_line_events') },
    { key: 'exc', name: 'Lock excursions', goal: 'min', unit: 'count', vals: pr.excursion_count,
      def: `Number of times the lock error went outside ±${thr} pm after the first lock (relock at ±${thr - hy} pm: ${hy} pm hysteresis).`, run: runCol('excursion_count') },
    { key: 'excmax', name: 'Longest excursion', goal: 'min', unit: 'ms', vals: pr.excursion_max_s.map(ms),
      def: 'Duration of the longest unlocked run after the first lock (an unlocked run still open at the end counts).', run: runCol('excursion_max_s', 1e3) },
    { key: 'unl', name: 'Time unlocked', goal: 'min', unit: 'ms', vals: pr.time_unlocked_s.map(ms),
      def: 'Total time unlocked after the ring\'s first lock (the time before the first lock is not counted).', run: runCol('time_unlocked_s', 1e3) },
    { key: 'unlf', name: 'Unlocked rings at the end', goal: 'zero', unit: 'ring', vals: pr.locked_final.map((v) => !v), bool: true,
      def: 'The ring is not in the locked state (with hysteresis) at the end of the run. Over the run: the number of rings unlocked at each frame end.',
      runSingle: run ? run.unlocked_rings : null },
    { key: 'maxe', name: 'Max |lock error|', goal: 'min', unit: 'pm', vals: pr.max_abs_lock_error_pm, band: thr,
      def: `Largest |detuning − lock reference| over every scored tick (lock band ±${thr} pm).`, run: runCol('max_abs_lock_error_pm') },
    { key: 'rmse', name: 'RMS lock error', goal: 'min', unit: 'pm', vals: pr.rms_lock_error_pm,
      def: 'Root mean square of the lock error over every scored tick.', run: runCol('rms_lock_error_pm') },
    { key: 'ttl', name: 'Time to lock', goal: 'min', unit: 'ms', vals: pr.time_to_lock_s.map(ms),
      def: 'Start of the locked run that is still on at the end, from the scoring start; – when the ring ends unlocked.', run: runCol('time_to_lock_s', 1e3) },
    { key: 'rel', name: 'relOMA (mean)', goal: 'max', unit: 'dB', vals: pr.mean_reloma_db,
      def: 'OMA divided by the light into the ring, at its through port, averaged over time (relOMA = OMA / P_in, the acceptance definition: OMA per 1 mW of laser).', run: runCol('mean_reloma_db') },
    { key: 'orc', name: 'Oracle relOMA (reference)', goal: 'ref', unit: 'dB', vals: pr.oracle_reloma_db,
      def: 'The best relOMA among open-loop-stable operating points at the same instants, averaged: the target of the OMA penalty.', run: runCol('oracle_reloma_db') },
    { key: 'energy', name: 'Heater energy', goal: 'min', unit: 'mJ', vals: pr.heater_energy_j.map((v) => (v == null ? null : v * 1e3)),
      def: 'Electrical energy into the heater over the scored time (sum of P_heater × tick).', run: runCol('heater_energy_j', 1e3) },
    { key: 'pmean', name: 'Mean heater power', goal: 'min', unit: 'mW', vals: pr.heater_energy_j.map((v) => (v == null ? null : (v / scoredS) * 1e3)),
      def: `Heater energy divided by the scored time (${fmt(scoredS * 1e3)} ms).` },
    { key: 'head', name: 'Heater headroom', goal: 'max', unit: 'code', vals: headroom(pr.code_min_used, pr.code_max_used),
      def: `The smaller of (lowest code used − ${h.code_lo}) and (${h.code_hi} − highest code used): codes left before a rail.`,
      run: run ? run.code_min_used.map((row, f) => (row == null ? null : headroom(row, run.code_max_used[f]))) : null },
    { key: 'rail0', name: 'Time at code 0', goal: 'min', unit: 'ms', vals: pr.heater_low_rail_s.map(ms),
      def: 'Time with the heater register at the low rail: the heater cannot cool the ring further.', run: runCol('heater_low_rail_s', 1e3) },
    { key: 'railfs', name: 'Time at full-scale code', goal: 'min', unit: 'ms', vals: pr.heater_sat_s.map(ms),
      def: 'Time with the heater register at the high rail: the heater cannot heat the ring further.', run: runCol('heater_sat_s', 1e3) },
    { key: 'sat', name: 'Clipped heater commands', goal: 'min', unit: 'count', vals: sm.saturation_counts ? sm.saturation_counts.tx_heater : pr.code_min_used.map(() => null),
      def: 'Controller steps whose heater command was outside the code range and was clipped (the engine\'s clip counter, whole run including start-up).',
      run: S.dev ? S.dev.heater.sat_count : null },
    { key: 'slew', name: 'Peak demanded slew', goal: 'min', unit: 'code/ms', vals: pr.peak_slew_codes_per_s.map((v) => (v == null ? null : v * 1e-3)),
      def: 'Largest |register change| / time between two controller latches (what the controller asked for; the DAC has no slew limit here).', run: runCol('peak_slew_codes_per_s', 1e-3) },
    { key: 'ber', name: 'BER (mean)', goal: 'min', unit: '', vals: pr.ber_mean, sci: true, log: true,
      def: `Expected errors divided by bits sent, every tick; BER from the bus-output OMA (Gaussian Q, mrm.plant.datapath). The ${sm.ber_sample_period_s} s window metrics read 0 on these short runs (see the BER section below).`, run: runCol('ber_mean') },
    { key: 'errs', name: 'Expected errors', goal: 'min', unit: 'bit', vals: pr.expected_errors, sci: true,
      def: 'Sum of BER × bits over the scored time.', run: runCol('expected_errors') },
    { key: 'fsr', name: 'FSR hop events', goal: 'zero', unit: 'count', vals: pr.fsr_hop_events,
      def: 'The ring relocked on another resonance order than the one of its previous lock.', run: runCol('fsr_hop_events') },
    { key: 'ord', name: 'Resonance order changes', goal: 'min', unit: 'count', vals: pr.order_changes,
      def: 'Steps at whose end the order nearest the declared line changed (lock-independent; a sweep over an FSR changes it on purpose).', run: runCol('order_changes') },
    { key: 'pabs', name: 'Mean absorbed light', goal: 'ref', unit: 'mW', vals: pr.mean_p_abs_w.map((v) => (v == null ? null : v * 1e3)),
      def: 'Mean optical power absorbed in the ring (it heats the ring).' },
  ];
}

// Teaching copy follows the selected metric; the curves keep their engine definitions.
const METRIC_PLOTS = {
  "oma": [
    "How much modulation is lost relative to the best stable point?",
    "The penalty compares accumulated modulation with the oracle’s best stable result at the same instants. Zero is a match; a larger positive value means worse modulation."
  ],
  "wrongs": [
    "How much time has each ring spent near the wrong laser line?",
    "Accumulated time within one linewidth of a foreign line. An upward slope means time is being added; a flat trace means none is being added."
  ],
  "wronge": [
    "How often has each ring moved onto a foreign line?",
    "A running count of entries near a laser line other than the assigned line. Each upward step is another entry, not another millisecond spent there."
  ],
  "exc": [
    "How often has a ring lost lock after acquiring it?",
    "A running count of lock-band exits after the first lock. Each step marks a new excursion; hysteresis prevents tiny boundary crossings from repeatedly counting."
  ],
  "excmax": [
    "How long is the longest loss of lock so far?",
    "The longest unlocked interval after first lock, including an interval still in progress. This record can grow or stay flat; it does not reset when lock returns."
  ],
  "unl": [
    "How much time has a ring spent unlocked after first lock?",
    "A running total of unlocked time. It grows during later losses of lock; the initial acquisition period is excluded."
  ],
  "maxe": [
    "What is the largest lock error seen so far?",
    "The running maximum absolute error. It stays high after a disturbance passes; the reference line marks the allowed lock-error magnitude."
  ],
  "rmse": [
    "How large has the typical lock error been?",
    "Root mean square error over all scored ticks so far. Large deviations weigh more heavily; this is an accumulated statistic, not the current error."
  ],
  "ttl": [
    "When did the currently surviving lock begin?",
    "At each frame, the start time of the locked interval still active then, measured from scoring start. Gaps mean the ring is unlocked; reacquisition can move the value later."
  ],
  "rel": [
    "How much modulation does the ring produce per unit input light?",
    "The running mean of normalized optical modulation amplitude, in dB. Higher values mean more useful contrast between data levels for the incoming light."
  ],
  "orc": [
    "What modulation could the best stable operating point provide?",
    "The oracle’s running mean normalized modulation, evaluated at the same instants. It is a reference for judging the controller, not a measurement supplied to it."
  ],
  "energy": [
    "How much heater energy has been used?",
    "Accumulated electrical heater energy since scoring began. A steeper rise means greater average heater power during that interval."
  ],
  "head": [
    "How much heater range remains after the commands used so far?",
    "The smaller distance from the lowest or highest code used to its rail. This running headroom can shrink; zero means a rail has been reached."
  ],
  "rail0": [
    "How much time has the heater spent at its minimum code?",
    "Accumulated time at the low rail. While it rises, the controller cannot lower the heater register any further."
  ],
  "railfs": [
    "How much time has the heater spent at its maximum code?",
    "Accumulated time at the high rail. While it rises, the controller cannot raise the heater register any further."
  ],
  "sat": [
    "How often has a requested heater command been clipped?",
    "A running count of controller commands outside the allowed code range, including start-up. Steps count requests that were clipped, not time spent at a rail."
  ],
  "slew": [
    "What is the fastest heater-code change requested so far?",
    "The running maximum register change divided by time between latches. A jump sets a new record; a flat trace does not mean the heater has stopped changing."
  ],
  "ber": [
    "What error fraction has the link accumulated?",
    "Expected errors divided by bits sent since scoring began, shown on a logarithmic axis. Lower is better; this running mean can hide a brief burst of errors."
  ],
  "errs": [
    "How many bit errors are expected so far?",
    "Predicted BER multiplied by bits sent and accumulated since scoring began. This is an expected count, so fractional values are meaningful."
  ],
  "fsr": [
    "Has the ring relocked on a different resonance order?",
    "A running count of relocks onto another order of the ring’s repeating spectrum. Each step records an order hop, not merely a small resonance shift."
  ],
  "ord": [
    "How often has the resonance order nearest the assigned line changed?",
    "A running count of nearest-order changes, whether locked or not. Acquisition sweeps can intentionally increase it, so it is not a count of lock failures."
  ],
  "wrongf": [
    "Does any ring finish near the wrong laser line?",
    "The table reports whether each ring ends within one linewidth of a foreign line. Only the final result is exported for this metric, so there is no time curve."
  ],
  "pmean": [
    "How much heater power was needed on average?",
    "The table reports heater energy divided by scored duration for each ring. Only the whole-run result is exported here, so there is no time curve."
  ],
  "pabs": [
    "How much optical power was absorbed on average?",
    "The table reports each ring’s mean absorbed light, which supplies optical heat. Only the whole-run result is exported here, so there is no time curve."
  ],
  "unlf": [
    "How many rings are unlocked right now?",
    "The number unlocked at each frame end, including rings still acquiring lock. A fall means more rings are locked; this is a current count, not a running total."
  ]
};

const GOAL = { min: 'minimise', max: 'maximise', zero: 'must be 0', ref: 'reference' };

function worst(d) {
  const v = d.vals;
  let best = -1;
  for (let i = 0; i < v.length; i++) {
    const x = d.bool ? (v[i] ? 1 : 0) : v[i];
    if (x == null || !Number.isFinite(x)) { if (d.key === 'ttl' && best < 0) best = i; continue; }
    const b = best < 0 ? null : (d.bool ? (v[best] ? 1 : 0) : v[best]);
    if (best < 0 || b == null || (d.goal === 'max' ? x < b : x > b)) best = i;
  }
  return best;
}

function cell(d, v) {
  if (d.bool) return v ? '<span class="tag">yes</span>' : 'no';
  if (v == null || (typeof v === 'number' && !Number.isFinite(v))) return '–';
  const s = d.sci ? sci(v) : fmt(v);
  const bad = (d.goal === 'zero' && v !== 0) || (d.band != null && v > d.band);
  return bad ? `<b>${s}</b> <span class="tag">${d.goal === 'zero' ? '≠ 0' : 'out of band'}</span>` : s;
}

export class MetricsView {
  constructor(card, { onPick, onSeek }) {
    this.card = card;
    this.head = card.querySelector('.readout');
    this.table = card.querySelector('table');
    this.chartTitle = card.querySelector('.mtitle');
    this.definition = document.createElement('p');
    this.definition.className = 'sub metric-definition';
    this.chartTitle.after(this.definition);
    this.stack = new TimeStack(card.querySelector('.plotwrap'), { onSeek, height: 220 });
    this.onPick = onPick;
    this.sel = 'oma';
    this.table.addEventListener('click', (e) => {
      const th = e.target.closest('[data-ring]');
      if (th) { this.onPick(Number(th.dataset.ring)); return; }
      const tr = e.target.closest('tr[data-key]');
      if (tr) { this.sel = tr.dataset.key; this.mark(); this.renderChart(); }
    });
    this.table.addEventListener('keydown', (e) => {
      const tr = e.target.closest('tr[data-key]');
      if (tr && (e.key === 'Enter' || e.key === ' ')) { e.preventDefault(); this.sel = tr.dataset.key; this.mark(); this.renderChart(); }
    });
  }

  render(S, ring) {
    this.S = S;
    this.ring = ring;
    const sm = S.scn.summary;
    const ref = S.scn.lock_reference.kind === 'line' ? 'the declared line' : "each ring's own operating point";
    this.head.textContent = `${sm.controller} · scored ${sm.score_from_s ? `from ${fmt(sm.score_from_s * 1e3)} ms` : 'from the start'} to ${fmt(sm.duration_s * 1e3)} ms · ` +
      `lock band ±${fmt(sm.lock_threshold_pm)} pm about ${ref} (hysteresis ${fmt(sm.hysteresis_pm)} pm) · every 1 µs tick, the engine's own metric definitions (mrm/sim/metrics.py)`;
    this.defs = defs(S);
    let h = '<thead><tr><th scope="col">Metric</th><th scope="col">Goal</th><th scope="col">Unit</th>';
    for (let r = 0; r < S.nR; r++) h += `<th scope="col" class="rc" data-ring="${r}" title="select ring ${r}">R${r}</th>`;
    h += '</tr></thead><tbody>';
    for (const d of this.defs) {
      const w = worst(d);
      const differs = new Set(d.vals.filter((v) => v != null && (typeof v === 'boolean' || Number.isFinite(v)))).size > 1;
      h += `<tr data-key="${d.key}" tabindex="0" title="show definition and time history"><th scope="row"><div class="mn">${esc(d.name)}</div></th>` +
        `<td><span class="goal goal-${d.goal}">${GOAL[d.goal]}</span></td><td>${esc(d.unit)}</td>`;
      d.vals.forEach((v, r) => {
        const isWorst = differs && d.goal !== 'ref' && r === w;
        h += `<td class="rc${r === ring ? ' selc' : ''}${isWorst ? ' worst' : ''}"${isWorst ? ' title="Worst ring for this metric"' : ''}>${cell(d, v)}</td>`;
      });
      h += '</tr>';
    }
    this.table.innerHTML = `${h}</tbody>`;
    if (!this.defs.some((d) => d.key === this.sel)) this.sel = 'oma';
    this.mark();
    this.renderChart();
  }

  mark() {
    this.table.querySelectorAll('tr[data-key]').forEach((tr) => tr.classList.toggle('sel', tr.dataset.key === this.sel));
  }

  setRing(r) {
    this.ring = r;
    this.table.querySelectorAll('tbody tr').forEach((tr) => {
      tr.querySelectorAll('td.rc').forEach((td, i) => td.classList.toggle('selc', i === r));
    });
    this.renderChart();
  }

  renderChart() {
    const S = this.S;
    const d = this.defs.find((x) => x.key === this.sel);
    if (!S || !d) return;
    this.definition.textContent = d.def;
    const unit = d.unit ? ` (${d.unit})` : '';
    const [question, explanation] = METRIC_PLOTS[d.key];
    const description = explanation + (d.run ? ' Each color is a ring; the selected ring is thicker.' : '');
    if (d.runSingle) {
      this.chartTitle.textContent = `${d.name} over the run: rings unlocked at each frame end`;
      this.stack.render(S, [{ title: 'unlocked rings', question, description, traces: [{ y: d.runSingle, name: 'unlocked rings', shape: 'hv', fmt: '.0f' }] }]);
      return;
    }
    if (!d.run) {
      this.chartTitle.textContent = `${d.name}: a value of the whole run only (no curve over the run)`;
      this.stack.render(S, [{ title: d.name + unit, question, description, traces: [] }]);
      return;
    }
    this.chartTitle.textContent = `${d.name} over the run: the metric's value from the scoring start to each frame end (all rings; the selected ring thick)`;
    const traces = [];
    for (let r = 0; r < S.nR; r++) {
      traces.push({ y: d.run.map((row) => (row == null ? null : row[r])), name: `R${r}`, color: cat(r), width: r === this.ring ? 2.6 : 1.1, fmt: d.sci ? '.3g' : '.4g', unit: d.unit });
    }
    this.stack.render(S, [{ title: d.name + unit, question, description, traces, log: !!d.log, hlines: d.band != null ? [{ y: d.band, label: `lock band ${d.band} pm` }] : [] }], { legend: true });
  }

  setTime(t) { this.stack.setTime(t); }
}

// ------------------------------------------------------------------ BER: the error metric
// The engine's BER metrics (mrm/sim/metrics.py) with their definition (mrm/plant/datapath.py), why the 2 s window
// metrics read 0 on these short runs, and the per-tick BER (frame means) that is meaningful here.
export class BerView {
  constructor(card, { onSeek, onPick }) {
    this.card = card;
    this.def = card.querySelector('[data-role=def]');
    this.table = card.querySelector('[data-role=table]');
    this.stack = new TimeStack(card.querySelector('.plotwrap'), { onSeek, height: 260 });
    this.table.addEventListener('click', (e) => { const th = e.target.closest('[data-ring]'); if (th) onPick(Number(th.dataset.ring)); });
  }

  render(S, ring) {
    this.S = S;
    this.ring = ring;
    const sm = S.scn.summary;
    const pr = sm.per_ring;
    const dp = S.meta.devices.datapath;
    const win = sm.ber_sample_period_s;
    const dur = sm.duration_s;
    const bitsTick = dp.bitrate_gbps * 1e9 * S.meta.devices.dt_s;
    this.def.innerHTML = `<div class="ber-method"><section class="explanation-panel"><h3>How does optical contrast become an error rate?</h3>` +
      `<p>The receiver decides each bit in Gaussian noise. The model predicts BER from the declared line’s optical modulation amplitude (OMA) at the bus output.</p>` +
      `<div class="equation-stack"><div class="equation-step"><span>1. Account for link attenuation</span><div class="equation-line">\\(\\displaystyle \\mathrm{OMA}_{\\mathrm{rx}}=${dp.link_transmission}\\,\\mathrm{OMA}_{\\mathrm{bus}}\\)</div></div>` +
      `<div class="equation-step"><span>2. Convert received contrast into signal quality</span><div class="equation-line">\\(\\displaystyle Q=Q_{\\mathrm{ref}}\\frac{\\mathrm{OMA}_{\\mathrm{rx}}}{\\mathrm{OMA}_{\\mathrm{sens}}}\\)</div></div>` +
      `<div class="equation-step"><span>3. Predict the fraction of incorrect bits</span><div class="equation-line">\\(\\displaystyle \\mathrm{BER}=\\frac12\\operatorname{erfc}(Q/\\sqrt2)\\)</div></div></div>` +
      `<p>The receiver sensitivity is ${fmt(dp.oma_sensitivity_uw)} µW at BER 1e-9, with reference quality \\(Q_{\\mathrm{ref}}=${dp.q_ref}\\).</p></section>` +
      `<section class="explanation-panel"><h3>Why do the window counters read zero?</h3>` +
      `<p>This run lasts ${fmt(dur * 1e3)} ms. The G26 error counter uses ${win} s windows, so none completes. The complete-window count, windows above the ${sm.ber_limit} limit and largest window BER therefore remain zero.</p>` +
      `<p>Zero here means “no window measured.” Use mean BER, expected errors and the time plot to judge this short run.</p>` +
      `<h3>What is still counted on every tick?</h3><p>Every 1 µs, the engine adds BER × bits to the expected error count: ${fmt(bitsTick)} bits per tick at ${dp.bitrate_gbps} Gb/s.</p>` +
      `<div class="equation-line">\\(\\displaystyle \\texttt{mean BER}=\\frac{\\texttt{expected errors}}{\\texttt{bits sent}}\\)</div>` +
      `<p>The mean uses the scored time.</p></section></div>`;
    const st = S.st;
    const q = st ? st.quantities.find((x) => x.key === 'ber') : null;
    const over = st ? st.states.find((x) => x.key === 'ber_over') : null;
    const rows = [
      ['Mean BER (engine)', 'Expected errors / bits, scored time. Meaningful on a short run.', pr.ber_mean, true],
      ['Expected errors (engine)', 'Sum of BER × bits over the scored time.', pr.expected_errors, true],
      ['Bits sent (engine)', 'Bits over the scored time.', pr.bits, true],
      ['Median BER of the 10 µs samples', 'Half the time the BER is below this value (scored time).', q ? q.scored.p50 : null, true],
      ['Highest BER of the 10 µs samples', 'The worst 10 µs of the scored time.', q ? q.scored.max : null, true],
      ['Time with BER above the limit (ms)', `10 µs samples with BER > ${sm.ber_limit}, scored time.`, over ? over.scored : null, false],
      [`Complete ${win} s BER windows (engine)`, 'Reads 0 when the run is shorter than one window.', pr.ber_windows, false],
      ['Windows above the limit (engine)', 'Reads 0 when no window completes.', pr.ber_windows_over_limit, false],
      ['Largest window BER (engine)', 'Reads 0 when no window completes: no measurement, not a perfect link.', pr.max_window_ber, false],
    ];
    let h = '<thead><tr><th scope="col">Quantity</th>';
    for (let r = 0; r < S.nR; r++) h += `<th scope="col" class="rc" data-ring="${r}" title="select ring ${r}">R${r}</th>`;
    h += '</tr></thead><tbody>';
    for (const [name, def, vals, sciForm] of rows) {
      h += `<tr><th scope="row"><div class="mn">${esc(name)}</div><div class="md">${esc(def)}</div></th>`;
      for (let r = 0; r < S.nR; r++) {
        const v = vals ? vals[r] : null;
        h += `<td class="rc${r === ring ? ' selc' : ''}">${v == null ? '–' : sciForm ? sci(v) : fmt(v)}</td>`;
      }
      h += '</tr>';
    }
    this.table.innerHTML = `${h}</tbody>`;
    const traces = [];
    for (let r = 0; r < S.nR; r++) {
      traces.push({ y: S.scn.rings.ber.map((row) => (row[r] == null ? null : Math.max(row[r], 1e-300))), name: `R${r}`, group: `R${r}`, color: cat(r), width: r === ring ? 2.6 : 1.1, fmt: '.3g', legend: true });
    }
    this.stack.render(S, [{ title: 'BER', question: 'When does the predicted bit-error rate exceed the limit?', description: 'Frame-mean BER for each ring on a logarithmic axis; the selected ring is thicker. Lower is better, and the reference line marks the allowed rate. These are model predictions, not counted errors.', log: true, traces, hlines: [{ y: sm.ber_limit, label: `limit ${sm.ber_limit}` }] }], { legend: true, ring });
  }

  setTime(t) { this.stack.setTime(t); }
}
