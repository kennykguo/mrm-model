// The fixed device settings (meta.devices, read from the realized plant) and the distances between the sites
// (meta.geometry.distances, from the preset's positions). Each value carries its provenance from the preset.

import { fmt, esc, prov, formatMicroseconds } from './ui.js';

export function renderConstants(card, meta) {
  const d = meta.devices;
  if (!d) { card.querySelector('table').innerHTML = '<tbody><tr><td>meta.json has no device settings: regenerate the data (npm run regen).</td></tr></tbody>'; return; }
  const h = d.heater;
  const m = d.monitor;
  const c = d.controller;
  const l = d.laser;
  const p = d.datapath;
  const hp = h.provenance || {};
  const mp = m.provenance || {};
  const lp = l.provenance || {};
  const rows = [
    ['Heater', 'Driver law', `${h.law} drive`, hp['ring.heater.driver_mode']],
    ['Heater', 'DAC', `${h.bits} bits, codes ${h.code_lo} to ${h.code_hi}`, hp['ring.heater.dac_bits']],
    ['Heater', 'Full-scale voltage', fmt(h.v_full_scale_v, 'V'), hp['ring.heater.full_scale_v']],
    ['Heater', 'Driver compliance', fmt(h.compliance_v, 'V'), hp['ring.heater.driver_compliance_v']],
    ['Heater', 'Resistance at 25 °C', fmt(h.r_cold_ohm[0], 'Ω'), hp['ring.heater.r_cold_ohm']],
    ['Heater', 'TCR', `${h.tcr_per_k} /K`, hp['ring.heater.tcr_per_k']],
    ['Heater', 'Bank supply limits', `${fmt(h.supply_power_max_w, 'W')}, ${fmt(h.supply_current_max_a * 1e3, 'mA')}`, hp['array.heater_total_power_max_w']],
    ['Heater', 'DAC latency', formatMicroseconds(h.dac_latency_s), hp['ring.heater.latency_s']],
    ['Heater', 'Slew limit', h.slew_codes_per_s == null ? 'none' : `${h.slew_codes_per_s} codes/s`, null],
    ['Monitor PD', 'Responsivity', `${m.responsivity_a_per_w} A/W`, mp['ring.monitor.responsivity_a_per_w']],
    ['Monitor PD', 'Dark current', `${m.dark_i0_na} nA at ${(m.dark_t0_k - 273.15).toFixed(0)} °C, activation ${m.dark_ea_ev} eV`, mp['ring.monitor.dark_i0_na']],
    ['Monitor PD', 'Input noise', `${m.input_noise_pa_per_rt_hz} pA/√Hz${m.shot_noise ? ' + shot noise' : ''}`, mp['ring.monitor.input_noise_pa_per_rt_hz']],
    ['Monitor ADC', 'Converter', `${m.afe.replace('_', ' and ')}, ${m.adc_bits} bits, full scale ${fmt(m.full_scale_ua_gain1, 'µA')} at gain 1, LSB ${m.lsb_na} nA`, mp['ring.monitor.adc_bits']],
    ['Monitor ADC', 'PGA gains', m.pga_gains.join(', '), null],
    ['Monitor ADC', 'Timing', `${formatMicroseconds(m.integration_window_s)} window, one sample every ${formatMicroseconds(m.sample_period_s)}, ${formatMicroseconds(m.latency_s)} latency`, mp['ring.monitor.adc_rate_hz']],
    ['Controller', 'Step', `every ${formatMicroseconds(c.period_s)} (${c.schedule}); ${c.adc_samples_per_step_nominal} ADC samples per step on average (room for ${c.samples_per_step_capacity})`, null],
    ['Laser', 'Lines', `${l.f0_thz.length} lines, ${l.p0_mw[0]} mW each at the module`, lp['laser.power_per_line_dbm']],
    ['Laser', 'Chip input', `× ${l.input_transmission} (${(10 * Math.log10(l.input_transmission)).toFixed(1)} dB)`, lp['optical_path.chip_input_loss_db']],
    ['Laser', 'RIN', l.rin_db_per_hz == null ? 'off' : `${l.rin_db_per_hz} dB/Hz`, lp['laser.rin_db_per_hz']],
    ['Laser', 'Drift in this world', `${l.f_rate_hz_per_s[0]} Hz/s, random walk ${l.f_rw_hz_per_rt_s[0]} Hz/√s (nominal world: no drift)`, null],
    ['Data path', 'Bit rate', fmt(p.bitrate_gbps, 'Gb/s'), null],
    ['Data path', 'Receiver', `link × ${p.link_transmission}; ${p.oma_sensitivity_uw} µW OMA gives Q = ${p.q_ref}`, null],
  ];
  let html = '<thead><tr><th scope="col">Part</th><th scope="col">Setting</th><th scope="col">Value</th></tr></thead><tbody>';
  for (const [part, k, v, pv] of rows) html += `<tr><th scope="row">${esc(part)}</th><td>${esc(k)}</td><td>${esc(v)}${prov(pv)}</td></tr>`;
  card.querySelector('table').innerHTML = `${html}</tbody>`;
  card.querySelector('.note').textContent = d.note || '';
}

export function renderDistances(host, meta, ring) {
  const g = meta.geometry;
  const d = g.distances;
  if (!d) { host.innerHTML = ''; return; }
  const pv = g.provenance || {};
  const r = ring;
  const site = d.ring_to_site_um.aggressor ? 'aggressor' : Object.keys(d.ring_to_site_um)[0];
  const row = (k, v, p) => `<tr><th scope="row">${esc(k)}</th><td>${v}${prov(p)}</td></tr>`;
  const die = Object.keys(d.ring_to_temp_sensor_um)[0];
  host.innerHTML = '<b>Distances</b> (centre to centre, from the preset; the G26 layout is not published)' +
    `<table class="mini dist"><tbody>${
      row('Ring pitch', fmt(d.ring_pitch_um[0], 'µm'), pv['layout.ring_pitch_um'])
    }${row('Ring radius (r_ref)', fmt(d.ring_radius_um, 'µm', 3), pv['ring.geometry.radius_um'])
    }${row(`R${r} → its PD (drop side)`, `${fmt(d.ring_to_own_pd_um[r], 'µm')} (${fmt(d.ring_edge_to_own_pd_um[r], 'µm')} from the rib edge)`, pv['ring.thermal.pd_offset_um'])
    }${row(`R${r} → its driver`, fmt(d.ring_to_own_driver_um ? d.ring_to_own_driver_um[r] : null, 'µm'), pv['ring.thermal.driver_offset_um'])
    }${row(`R${r} → aggressor site`, fmt(d.ring_to_site_um[site][r], 'µm'), pv['layout.sites'])
    }${row(`R${r}'s PD → aggressor site`, fmt(d.pd_to_site_um[site][r], 'µm'), pv['layout.sites'])
    }${d.ring_to_site_um.aggressor_pd_adjacent ? row(`R${r}'s PD → stress site`, fmt(d.pd_to_site_um.aggressor_pd_adjacent[r], 'µm'), pv['layout.sites']) : ''
    }${die ? row(`R${r} → die sensor ${die} (off the map)`, fmt(d.ring_to_temp_sensor_um[die][r], 'µm'), pv['layout.sites']) : ''
    }</tbody></table>`;
}
