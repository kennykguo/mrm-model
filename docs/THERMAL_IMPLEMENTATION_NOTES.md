# thermal implementation details

the code keeps short purpose comments and same-line type contracts.
these notes retain the detailed assumptions, equations, provenance, and artifact conventions from the implementation comments.
the equations and numerical constants are unchanged.

## calculate_linear_hold_lag_weights

exact hold coefficients

integrate one thermal pole with linearly varying input power.

for tau*dx/dt = -x + r*u(t), x(h) = a*x(0) + r*(b0*u0+b1*u1).

x (K) is the lag state, r (K/W) is its residue, and u0 and u1 (W) are endpoint powers.

y = h/tau, a = exp(-y), b1 = 1-(1-a)/y, and b0 = (1-a)/y-a.

held input has u0=u1 and b0+b1=1-a.

use five Taylor terms for y<1e-3 to avoid cancellation.

h<=0 leaves the state unchanged.

## calculate_linear_hold_coefficients

compute exact first-order-hold coefficients for a thermal network.

x_next = a*x + c0*u0 + c1*u1, with x (K) and u0 and u1 (W).

c0=r*b0 and c1=r*b1 - see foh_scalar.

## calculate_constant_hold_coefficients

compute exact zero-order-hold coefficients for constant input power.

x_next = a*x + c*u, with c=(1-a)*r, x (K), and u (W).

## advance_thermal_state_for_world

per-step kernels

advance one world's thermal lag states in place.

x_j <- a_j*x_j + s_src*(c0_j*u0_src+c1_j*u1_src).

src_scale is normally one - temperature scaling or a fault can change it.

## advance_thermal_state_for_world_with_step_size

like thermal_step_b for a variable step h (s) — the exact FOH coefficients are computed inline per lag.

for adaptive multi-tick steps (`sim.substep_policy — auto`).

taus (M,) s, res (M,) K/W.

## write_observer_temperatures_for_world

sum lag-state temperature rises at each observer in one world.

## validate_thermal_step_array_shapes

shape guard of the batched kernels (numba does not bounds-check) — every per-world input has 1 or B rows, a/c0/c1 have the same rows and M columns, lag_src has M entries.

raises ValueError.

## advance_thermal_states

batched thermal_step_b.

x (B, M).

a, c0, c1 (B|1, M).

u0, u1 (B|1, S) W.

src_scale (B|1, S).

## write_observer_temperatures

batched thermal_outputs_b.

x (B, M).

t_base (B|1,) K.

out (B, O) K.

## advance_thermal_state_and_temperatures_for_world

thermal_step_b and thermal_outputs_b fused in one pass over the lags (one world) — x advanced in place, out (O,) the end-of-step observer temperatures t_base + sum x (K).

same equation as thermal_step_b.

## advance_thermal_states_and_temperatures

batched thermal_advance_b — x (B, M).

a, c0, c1 (B|1, M).

u0, u1, src_scale (B|1, S).

t_base (B|1,) - out (B, O).

## separate_known_heat_and_source_gain_for_observer

separate one observer's temperature rise into known heat and an unknown source's gain.

return (pre, g), where T_end-T_base = pre + g*u_self.

pre uses K.

g uses K/W.

u_self is held over the step.

pre includes stored heat and all other sources driven by u0/u1.

the heater TCR solve uses T_heater_pre=T_base+pre and g_self=g because resistance depends on the heater's own resulting temperature.

see electrical.heater_power_row.

unknown powers from other sources must match those later used for the state update.

using their previous values delays that coupling by one step.

## save_rom

write a `thermal_rom/1` artefact and its sidecar `<stem>.sidecar.json` with the sha256 of the .npz (the same sidecar naming as mrm/extraction/common.write_artefact).

the .npz holds taus_s, sources, observers, self_res, cross_r_um, cross_res, site_offset_names, site_offsets_um.

the sidecar gets schema and sha256 added to `sidecar` (family, film_nm, inputs, tool_versions, mesh, validation{step_max_rel_err, freq_max_db_err, freq_max_deg_err, superposition_rel_err}, created_utc).

## load_rom

load a `thermal_rom/1` .npz and verify it against its sidecar `<stem>.sidecar.json` (or `<stem>.json`).

verify=True — the sidecar must exist, carry schema 'thermal_rom/1' and the sha256 of the .npz bytes.

otherwise

RomIntegrityError.

warn=True — RuntimeWarning when the sidecar's validation errors exceed the extraction targets

(step < 2 % of steady state, |dH| < 0.3 dB and < 3 deg to 1 MHz, superposition < 10 %) or are missing, or when the pole set lacks a sub-us pole (waveguide self-heating) or a millisecond pole (crosstalk).

both writers are read — save_rom (flat validation keys) and mrm/extraction/thermal_fem.write_artefact (`<stem>.json`, nested validation report, offsets also in `inputs.config.site_offsets_um`).

see rom_validation_numbers.

both write the site names as `site_offset_names`.

`site_names` is also accepted.

## read_rom_validation_metrics

the four acceptance numbers of a ROM sidecar — {step_max_rel_err, freq_max_db_err, freq_max_deg_err, superposition_rel_err} (None when absent).

read from flat keys `validation.<key>` (save_rom), or from the nested report that mrm/extraction/thermal_fem.verify writes (`validation.{self, nearest_neighbour}.{max_time_err_frac_of_dc, max_db, max_deg}`. the targets apply to the self and the nearest-neighbour terms, so the worse of the two is used).

the superposition error is also looked up in `stages.<stage>.superposition_rel_err` (thermal_fem.update_sidecar).

## LayoutSites.from_sites

parse `layout.sites` {name — {x_um, y_um, kind, [ring]}} and the `temp_sensors` table.

kind ring — name 'ring<i>' or field ring=i (missing rings — ring<i> on a line at ring_pitch_um, schema doc).

kind monitor_pd / driver — field ring=i or a trailing index in the name.

a driver site with no ring index is shared by every ring (one driver block).

kind aggressor, and kind heater without an explicit `ring` field (a trailing digit in the name does not attach it. the schema's site fields have no `ring`) — aggressor heat sources.

a heater site with `ring` is that ring's own heater, which the model places at the ring centre.

kind temp_sensor — a free sensor site named after the site, unless a `temp_sensors` entry references it through `site` (then that entry, with its own name, is the one sensor at that site. no duplicate).

temp_sensors entries {name, site | attach} — attach 'ring_<i>' puts the sensor at ring i (offset added by the model), 'bank' at the ring centroid, 'die' 1000 um from the centroid (assumed placement for a die-level sensor).

## InterimRom.textbook_corner

the pessimistic textbook corner — 10 us + 300 us (25 %) self path, no tail, crosstalk 10 % nearest / 3 % next / 0 beyond with the self-path dynamics (textbook REF tauThUs, tauSlowUs, crosstalk*).

these textbook numbers are unverified.

the default interim model uses the imec numbers instead.

## calculate_crosstalk_time_constant_s

first-order crosstalk time constant [s] at heat-source-to-victim distance d [um] (thermal notes, 2.4-2.5).

d is the distance from the heat source to the victim as [CO22-TCPMT] defines it (thermal notes, section 2.5 - 60 um for the first neighbour at their 100 um pitch).

this is the convention of dist_ref_um, tau_ref_s and the X2 table (XT_TABLE_D_UM), not the centre-to-centre distance — the interim assembly passes d_cc - r_src - r_obs (InterimRom.ring_radius_um. at imec's geometry, R ~ 20 um at a 100 um pitch, it reproduces Table II).

Centre distances would give 4.75 ms instead of the printed 4.14 ms at imec's own geometry, and 2.85 ms instead of 1.52 ms for the g26 nearest neighbour.

near field (derived d^2 diffusion law, [CO22-TCPMT] Eq. 4) — tau_near = max(tau_min, tau_ref (d/d_ref)^2).

law='literal' (the interim model as first specified) — tau_near for d <= far_um, tau_far beyond (tau_far — a number, or None = the printed X2 table interpolated in d).

note — at d = 100 um the d^2 law gives 11.4 ms, above every X2 value (4.1-9.0 ms), so the literal law is not monotone in d.

law='capped' (default) — tau = min(tau_near, tau_far(d)) with tau_far(d) the X2 table (or the given number) and the result floored at tau_min.

Monotone and continuous.

equal to the d^2 law inside about 60 um.

Arrays broadcast.

returns s.

## calculate_site_coupling_fraction

interim static coupling of a site path, as a fraction of Z_heater->ring(0) (assumed law, the shape of rows X1-X5 of the thermal notes) -

c(d) = clip(nearest_frac * ref_um / d, floor_frac, max_frac), d the centre (site) distance in um.

at ref_um it equals the nearest-neighbour crosstalk.

it never falls below floor_frac, the long-range floor (InterimRom site_floor_frac, default xt_floor_frac — a package term, so it holds at any distance on the die. imec measured it to 300 um with natural convection, the FEM's TEC-held package gives 0.09 % at 1 mm).

## build_thermal_sources_and_observers

the source and observer instances of the bank, and the per-ring index maps (heater_src, ring_obs, ...).

Ring sources and ring/heater observers sit at the ring centre.

a PD without a layout site sits at the ring plus pd_offset [um].

an attached temperature sensor at its ring plus ts_offset [um].

a driver at its layout site or at the ring plus driver_offset [um] (None — no driver).

## append_extracted_heat_path_lags

FEM ROM residues — self_res for same-ring pairs, distance-interpolated cross_res otherwise.

self_res was solved with the site elements (driver source, monitor_pd and temp_sensor observers) at the ROM's default offsets (site_offsets_um).

a same-ring pair whose site element sits more than tol_um away from that offset (for example one shared driver block far from the rings) uses cross_res at its actual source-to-observer distance instead, with a warning.

## replace_package_tail_preserving_dc_gain

replace or add the package tail on every tailable pair, preserving each pair's DC.

Tailable pairs — the self pairs (and, with crosstalk_dynamics='self', the ring-to-ring pairs of the textbook corner, which use the self shape).

the tail tau is absolute (fixed_tau — not rescaled later).

## calculate_shared_pair_time_scales

find mixed-sign pairs that need one time-scale factor across both layers.

FEM fits can use large positive and negative residues that cancel.

separate fast and slow factors can destroy that cancellation and distort the step response.

metal_above_160 showed a 0.22*DC

overshoot/dip at factors 2 and 0.5.

same-sign pairs retain monotone steps and |H(jw)|<=|H(0)| under

separate layer factors.

this rule does not make a non-monotone original fit monotone.

for a mixed-sign pair spanning both layers, use k=s_fast^w*s_slow^(1-w).

w is the step-response

fraction reached at tau_split, clipped to [0, 1].

the whole pair then follows y_scaled(t)=y(t/k).

equal layer factors give exactly k=s_fast.

exclude fixed-tau package tails from detection and scaling.

tau and res have shape (B, M) before scaling.

obs, src, layer, and fixed have length M.

s_fast and s_slow have length B.

return (whole, s_pair), with shapes (M,) and (B, M).

s_pair is None when no pair needs whole-pair scaling.

## assemble

assemble the full source-to-observer thermal system for n_rings and B worlds.

rom is a ThermalRom or InterimRom.

layout is LayoutSites or a layout.sites mapping.

None is valid only for one ring at the origin.

mappings also use ring_pitch_um and temp_sensors.

scales is ThermalScales or its field mapping.

None uses unit factors and the resistance hypothesis.

B is the common broadcast length of all per-world inputs, including scaling_hypothesis.

sources and observers select kind names, default all.

drivers need layout sites or an interim driver offset.

for a FEM ROM, explicitly selecting 'driver' permits its default offset.

aggressors and temperature sensors need sites.

every included source couples to every observer.

same-ring paths use the self table.

cross-ring paths use centre-distance residues.

free or moved sites use their actual distances.

see _rom_lags for the self-table offset check.

apply DC factors by source/observer kind and crosstalk factors on cross-ring residues.

scale time by layer, keeping mixed-sign pairs together.

the resistance hypothesis also scales tau with the DC factor, never the crosstalk factor.

an added package tail preserves DC and has fixed tau.

prune lags with |residue|<=prune_abs_k_per_w in every world.

zero drops exact zeros only.

sort by observer, source, and time constant.

classify optical lags for the self-heating solve.

return ThermalSystem with (B, M) time constants and residues.

cost is about 4–10 ns per lag per substep.

a K-pole ROM can contain K*sources*observers lags.

select only needed kinds for shorter runs.

## assemble_interim_thermal_system

build the interim thermal ThermalSystem from preset numbers — assemble(InterimRom(**preset), ...).

preset — InterimRom fields (schema names `ring.thermal.*` in SI), e.g. z_heater_ring_k_per_w, tau_fast_s, share_fast_frac, tau_slow_s, tail_tau_s, tail_share_frac, selfheat_tau_s, selfheat_share_frac, z_ratio_opt_heater, xt_nearest_frac, xt_floor_frac, xt_tau_ref_s, ...

see InterimRom for the equations.

## simulate

Python-level drivers and analysis

return open-loop observer temperatures at sample times 0, h, ..., n*h.

output shape is (n+1, B, O), in K.

u_samples_w has shape (n+1, B|1, S), in W.

'zoh' holds u[k] over [k*h, (k+1)*h).

'foh' interpolates linearly between samples.

the update is exact for these input holds.

source power is supplied, without nonlinear optical feedback.

## state_space

continuous state space of world b for the selected observers/sources (for linearize and validation).

returns (A, Bm, C, D) — A = diag(-1/tau_m) (1/s), Bm[m, s] = r_m/tau_m (K/(W s)), C[o, m] = 1, D = 0, with one

state per selected lag m (tau_m dx_m/dt = -x_m + r_m u_s).

outputs are rises above T_base in K.

## reciprocity_error

largest relative reciprocity violation over like pairs (a verification test — Z_ij = Z_ji), on |Z(j 2 pi f)|.

Pairs checked for every ring i, j (including i = j where it is not trivial) — heater_j -> heater_resistor_i vs

heater_i -> heater_resistor_j.

waveguide_j -> ring_i vs waveguide_i -> ring_j.

heater_j -> ring_i vs waveguide_i -> heater_resistor_j.

returns max |Z_ab - Z_ba| / max|Z| over the pairs that exist.

## cross_optical_lag_gain

per-substep gain of the lagged cross-ring optical heating, (B, N) [-].

mrm.plant.selfheat drives the cross-ring optical lags with the previous substep's P_abs.

That lag is harmless while this gain is small (realize() asserts g < 0.05).

g_i = sum over LAG_OPT_CROSS_RING lags m into ring i of (1 - a_m) r_m * max|dP_abs,j/dT_j| (j = source ring), where dpabs_dt_max_w_per_k (B|1, N) W/K bounds |dP_abs/dT| of each ring (for one Lorentzian line - (3 sqrt(3)/8) P_abs,peak * (dlambda/dT)/HWHM, see selfheat.lorentzian_bounds).

## calculate_conductivity_residue_scale

quasi-static k(T) scaling of the thermal residues, R(T)/R(T_ref) [-] (temperature-dependent conductivity).

R = R_ref [w_film k_film(T_ref)/k_film(T) + w_ox k_ox(T_ref)/k_ox(T) + w_sub k_Si(T_ref)/k_Si(T)], with the in-plane film fit (Matthiessen), oxide 1.38 (T/300)^0.33 and bulk Si 148 (T/300)^-1.3 W/(m K) (Palankovski 2000), taken from mrm.materials (the materials notes, docs/research/02_materials.md, section 5).

Temperatures in K.

weights = shares of the DC resistance in the film, the oxide and the substrate — assumed (no FEM sensitivity exists yet. the BOX path usually dominates).

apply as `src_scale` of the step kernels (per source, evaluated at a slowly varying temperature).
