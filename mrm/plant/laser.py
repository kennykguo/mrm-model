"""Laser lines: the line table, static frequency error, drift (random walk + linear), RIN, power drift, line enable.

Scope (docs/ARCHITECTURE.md section 2; 07 = docs/research/07_design_critique.md): 07 P1-12 (the line table
{freq, direction, source_laser, pol} covering G26 and L20; laser_line_enable; failover steps), P1-13 (counter RNG,
streams laser_rin / laser_drift), P0-10 (time compression of slow exogenous signals), P2-8 (``laser.tune``).

Units: Hz (frequency), W (power), s. Frequencies are absolute optical frequencies (c / lambda).

Published numbers (printed unless stated; sources in docs/research/01 section 2 and 04 section 3.2):
- G26 channel plan: "2 x 8 lambda, 400 GHz grid centred at 1310 nm, offset 200 GHz" (LM26-P section 2). Reading
  used here (inferred): the 16 lines form one 200 GHz comb centred at 1310 nm, f_i = c/1310 nm + (i - 7.5) 200 GHz,
  i = 0..15, and the two directions take alternate lines (each direction a 400 GHz grid, the other offset by
  200 GHz). This puts the comb at 1301.47-1318.64 nm. The comb read-off is 1301.5-1318.9 nm (+-0.1 nm, HC slide 17,
  01 REF table): the blue end agrees, the red end is 0.26 nm (about 45 GHz) longer than this grid, more than the
  read-off error (unresolved; a 200 GHz grid anchored at 1301.47 nm cannot reach 1318.9 nm). The PAT-4 example grid
  (8 lines, 1301.47-1309.43 nm, 200 GHz apart) equals this comb's 8 bluest lines, which supports the 200 GHz comb
  and its anchor but says nothing about how the two directions split it. Which direction takes the lower slot is not
  published (``tx_slot``; assumed even).
- L20: 1311 and 1331 nm, one per direction, every port programmable to transmit on either (BIDI, printed).
- Absolute accuracy: +-20 GHz required (LM26-P section 4), +-15 GHz Guide VLSP claim (YOLE), 3 GHz roadmap (HOTI26).
- Drift after burn-in: +-0.02 nm (about 3.5 GHz) and +-2.5 % of power over 24 h (Guide VLSP; textbook
  REF.laserDriftNm24h / laserPowerDrift24hPct [YOLE]). These are bounds, not rates; how they split into a linear part
  and a random walk is not published (see :func:`drift_from_24h_bound`, assumed).
- RIN: not published for the Guide laser (no default; ``laser.rin_db_per_hz``).

Model of line l (module m = source_laser[l]) at scenario time t_s = (k + 1/2) dt c, c = scenario.time_compression:
    f_l = f0_l + r_f,l t_s + D_f,l sqrt(G) [sqrt(rho_f) W_m^f(t_s/G) + sqrt(1 - rho_f) W_l^f(t_s/G)]
          + f_offset_l + tune_l
    P_l = p0_l max(0, 1 + r_p,l t_s + D_p,l sqrt(G) [sqrt(rho_p) W_m^p + sqrt(1 - rho_p) W_l^p])
          * p_scale_l * max(0, 1 + sqrt(RIN/(2 dt)) z_{l,k}) * enable_l
- f0 = grid + static error (realize puts laser.line_offset_ghz there; :func:`draw_static_error_hz` draws one).
- W are standard Brownian motions on a grid of period G (scenario seconds), built by the Levy midpoint construction
  from counter-RNG draws keyed by the dyadic node, and linearly interpolated between grid points. The value at any
  tick is a pure function of (seed, line, tick): it does not depend on how the run was stepped (P0-10 multi-tick
  stepping, quasi-static jumps), on the batch or on the thread count. rho is the variance fraction shared by the
  lines of one module (07 P1-12 / 04 3.2 "correlated common-mode offset (one laser module)"; an option).
- RIN: z_{l,k} is the tick average of white relative intensity noise with one-sided PSD RIN (1/Hz):
  Var = RIN/(2 dt), the exact average of white noise over one tick, so window averages are dt-invariant (P1-13).
  RIN is independent between lines (assumed; mode-partition correlations are not modelled).
- Time compression (07 P0-10) applies to the drift only (it is a slow exogenous signal), never to RIN.

Randomness: ``Stream.LASER_DRIFT`` (ring = line index, or L + module index for the common part; tick = Levy node id;
slot 0 frequency, 1 power) and ``Stream.LASER_RIN`` (ring = line pair l // 2, the two normals of one Box-Muller call
going to lines 2i and 2i + 1; tick = plant tick), keyed by the scenario/noise
seed; static errors and per-line drift rates use ``Stream.VARIATION`` with the DEVICE seed.
"""
from __future__ import annotations

import math
import zlib
from dataclasses import dataclass

import numba as nb
import numpy as np
from numba import njit

from mrm.core import rng as _rng
from mrm.units import C0

__all__ = [
    "DIR_TX", "DIR_RX", "POL_TE", "POL_TM", "DIRECTIONS", "POLS",
    "G26_CENTER_M", "G26_SPACING_HZ", "G26_OFFSET_HZ", "G26_LINES_PER_DIRECTION", "L20_LAMBDAS_M",
    "LASER_ACCURACY_SPEC_HZ", "LASER_ACCURACY_DEMONSTRATED_HZ", "LASER_ACCURACY_ROADMAP_HZ",
    "LASER_DRIFT_24H_M", "LASER_POWER_DRIFT_24H_FRAC", "SECONDS_PER_24H", "DRIFT_GRID_S_DEFAULT", "LEVY_DEPTH",
    "LineTable", "g26_line_table", "l20_line_table", "grid_line_table",
    "levy_bm", "laser_tick_row", "laser_tick", "laser_tick_par",
    "draw_static_error_hz", "drift_from_24h_bound", "LaserHw", "LaserBank",
]

DIR_TX = 0
DIR_RX = 1
DIRECTIONS = {"tx": DIR_TX, "rx": DIR_RX}
POL_TE = 0
POL_TM = 1
POLS = {"te": POL_TE, "tm": POL_TM}

G26_CENTER_M = 1310e-9            # printed, LM26-P section 2: "centered at 1310 nm"
G26_SPACING_HZ = 400e9            # printed, LM26-P section 2: 400 GHz grid per direction
G26_OFFSET_HZ = 200e9             # printed, LM26-P section 2: the two sets are offset by 200 GHz
G26_LINES_PER_DIRECTION = 8       # printed, LM26-P section 2: 2 x 8 lambda
L20_LAMBDAS_M = (1311e-9, 1331e-9)  # printed, BIDI tech brief: L20 at 1311/1331 nm
LASER_ACCURACY_SPEC_HZ = 20e9     # printed, LM26-P section 4: +-20 GHz required (and achieved)
LASER_ACCURACY_DEMONSTRATED_HZ = 15e9  # printed, YOLE: Guide VLSP "+/- 15 GHz"
LASER_ACCURACY_ROADMAP_HZ = 3e9   # printed, HOTI26: VLSP 3 GHz frequency accuracy
LASER_DRIFT_24H_M = 0.02e-9       # printed (via textbook REF.laserDriftNm24h, YOLE): +-0.02 nm over 24 h
LASER_POWER_DRIFT_24H_FRAC = 0.025  # printed (via textbook REF.laserPowerDrift24hPct, YOLE): +-2.5 % over 24 h
SECONDS_PER_24H = 86400.0
DRIFT_GRID_S_DEFAULT = 1e-3       # design choice: drift interpolation grid (1 ms of scenario time)
LEVY_DEPTH = 44                   # design choice: 2^44 grid steps (557 years at 1 ms); node ids < 2^45

_SLOT_FREQ = 0
_SLOT_POW = 1
_SLOT_RIN = 0
_TAG_STATIC = zlib.crc32(b"laser.static_error") & 0x7FFFFFFF   # VARIATION tick tag of the static error draws
_TAG_RATE_F = zlib.crc32(b"laser.drift_rate_freq") & 0x7FFFFFFF
_TAG_RATE_P = zlib.crc32(b"laser.drift_rate_power") & 0x7FFFFFFF


# =============================================================================================== line table
@dataclass(frozen=True)
class LineTable:
    """The laser line table of 07 P1-12: one row per line {freq, direction, source_laser, pol}.

    freq_hz (L,) nominal grid frequency; direction (L,) DIR_TX | DIR_RX; source_laser (L,) index into
    ``laser_names``; pol (L,) POL_TE | POL_TM. Line order is the table order (realize: increasing frequency for a
    centre/spacing grid).
    """
    freq_hz: np.ndarray
    direction: np.ndarray
    source_laser: np.ndarray
    pol: np.ndarray
    laser_names: tuple = ("laser0",)

    def __post_init__(self):
        f = np.asarray(self.freq_hz, dtype=np.float64).reshape(-1)
        L = f.size
        for name in ("direction", "source_laser", "pol"):
            a = np.asarray(getattr(self, name), dtype=np.int64).reshape(-1)
            if a.size != L:
                raise ValueError(f"LineTable.{name} has {a.size} entries for {L} lines")
            object.__setattr__(self, name, a)
        object.__setattr__(self, "freq_hz", f)
        if L < 1 or not np.all(np.isfinite(f) & (f > 0)):
            raise ValueError("LineTable needs at least one positive finite frequency")
        if np.any((self.direction < 0) | (self.direction > 1)) or np.any((self.pol < 0) | (self.pol > 1)):
            raise ValueError("direction and pol codes must be 0 or 1")
        if np.any(self.source_laser < 0) or np.any(self.source_laser >= len(self.laser_names)):
            raise ValueError("source_laser indexes outside laser_names")
        object.__setattr__(self, "laser_names", tuple(str(x) for x in self.laser_names))

    @property
    def n_lines(self) -> int:
        return int(self.freq_hz.size)

    @property
    def n_lasers(self) -> int:
        return len(self.laser_names)

    @property
    def wavelength_m(self) -> np.ndarray:
        return C0 / self.freq_hz

    def select(self, direction: str) -> "LineTable":
        """The sub-table of one direction ('tx' or 'rx'), lasers renumbered."""
        m = self.direction == DIRECTIONS[direction]
        used = sorted(set(self.source_laser[m].tolist()))
        remap = {old: i for i, old in enumerate(used)}
        return LineTable(self.freq_hz[m], self.direction[m], np.array([remap[s] for s in self.source_laser[m]]),
                         self.pol[m], tuple(self.laser_names[i] for i in used))

    def records(self) -> list[dict]:
        """Rows as the schema's channel_plan.lines records {freq_thz, direction, source_laser, pol}."""
        inv_d = {v: k for k, v in DIRECTIONS.items()}
        inv_p = {v: k for k, v in POLS.items()}
        return [dict(freq_thz=float(f) / 1e12, direction=inv_d[int(d)], source_laser=self.laser_names[int(s)],
                     pol=inv_p[int(p)]) for f, d, s, p in zip(self.freq_hz, self.direction, self.source_laser,
                                                               self.pol)]

    @classmethod
    def from_records(cls, rows) -> "LineTable":
        """From channel_plan.lines records: {freq_thz | freq_hz, direction ('tx'), source_laser (name), pol ('te')}."""
        names: list[str] = []
        f, d, s, p = [], [], [], []
        for r in rows:
            r = {k: (v.get("value") if isinstance(v, dict) and "value" in v else v) for k, v in dict(r).items()}
            f.append(float(r["freq_hz"]) if "freq_hz" in r else float(r["freq_thz"]) * 1e12)
            d.append(DIRECTIONS[str(r.get("direction", "tx"))])
            src = str(r.get("source_laser") or "laser0")
            if src not in names:
                names.append(src)
            s.append(names.index(src))
            p.append(POLS[str(r.get("pol", "te"))])
        return cls(np.array(f), np.array(d), np.array(s), np.array(p), tuple(names) or ("laser0",))

    @classmethod
    def from_plant_params(cls, pp, source_laser: dict | None = None) -> "LineTable":
        """From PlantParams.laser (line_freq_grid_hz, direction, source_laser, pol). A line whose source_laser is ''
        (the channel_plan.lines table does not say) is given laser 'local' (tx) or 'remote' (rx) unless
        ``source_laser`` ({direction: name}) says otherwise; pol defaults to TE."""
        las = pp.laser
        src = {"tx": "local", "rx": "remote"}
        src.update(source_laser or {})
        freqs = np.asarray(las.line_freq_grid_hz).reshape(-1)
        given = tuple(getattr(las, "source_laser", ()) or ())
        pols = tuple(getattr(las, "pol", ()) or ())
        rows = [dict(freq_hz=float(f), direction=str(d),
                     source_laser=(given[i] if i < len(given) and given[i] else src[str(d)]),
                     pol=(pols[i] if i < len(pols) and pols[i] else "te"))
                for i, (f, d) in enumerate(zip(freqs, las.direction))]
        return cls.from_records(rows)


def g26_line_table(*, center_m: float = G26_CENTER_M, spacing_hz: float = G26_SPACING_HZ,
                   offset_hz: float = G26_OFFSET_HZ, n_per_direction: int = G26_LINES_PER_DIRECTION,
                   tx_slot: int = 0, tx_laser: str = "local", rx_laser: str = "remote") -> LineTable:
    """The G26 16-line bidirectional plan (module docstring): two sets of n lines on a ``spacing_hz`` grid, the
    second offset by ``offset_hz`` (printed 400 and 200 GHz, so the 2n lines form one 200 GHz comb), the whole comb
    centred at ``center_m``. Lines are ordered by increasing frequency; direction tx takes the set whose lowest line
    is the lowest of the comb when tx_slot = 0 (not published; assumed), rx the other. TX lines come from the local
    laser module (index 0), RX lines from the far end's module (index 1)."""
    n = int(n_per_direction)
    if n < 1 or tx_slot not in (0, 1):
        raise ValueError("need n_per_direction >= 1 and tx_slot in {0, 1}")
    fc = C0 / center_m
    # set A at fc + (j - (n-1)/2) spacing - offset/2, set B = set A + offset: the 2n lines are centred at fc
    base = fc + (np.arange(n) - (n - 1) / 2.0) * spacing_hz - offset_hz / 2.0
    f = np.concatenate([base, base + offset_hz])
    slot = np.concatenate([np.zeros(n, np.int64), np.ones(n, np.int64)])
    order = np.argsort(f)
    f, slot = f[order], slot[order]
    direction = np.where(slot == tx_slot, DIR_TX, DIR_RX)
    source = np.where(direction == DIR_TX, 0, 1)
    return LineTable(f, direction, source, np.zeros(2 * n, np.int64), (tx_laser, rx_laser))


def l20_line_table(*, tx_m: float = L20_LAMBDAS_M[0], rx_m: float = L20_LAMBDAS_M[1], tx_laser: str = "local",
                   rx_laser: str = "remote") -> LineTable:
    """The L20 BiDi pair: transmit on 1311 nm and receive on 1331 nm by default (printed; every port may swap them,
    pass tx_m=1331e-9, rx_m=1311e-9). Ordered by increasing frequency."""
    f = np.array([C0 / tx_m, C0 / rx_m])
    d = np.array([DIR_TX, DIR_RX])
    order = np.argsort(f)
    return LineTable(f[order], d[order], d[order], np.zeros(2, np.int64), (tx_laser, rx_laser))


def grid_line_table(center_hz: float, spacing_hz: float, n_lines: int, *, direction: str = "tx",
                    laser: str = "laser0") -> LineTable:
    """A single-direction grid f_k = center + (k - (L-1)/2) spacing (the realize rule for channel_plan.center_thz /
    spacing_ghz / n_lines)."""
    f = center_hz + (np.arange(int(n_lines)) - (int(n_lines) - 1) / 2.0) * spacing_hz
    return LineTable(f, np.full(f.size, DIRECTIONS[direction]), np.zeros(f.size, np.int64),
                     np.zeros(f.size, np.int64), (laser,))


# =============================================================================================== kernels
@njit(cache=True)
def levy_bm(key, ring, slot, j, depth):
    """Standard Brownian motion W(j) (unit variance per unit time) at integer j in [0, 2^depth] (Levy midpoint
    construction): W(2^depth) = sqrt(2^depth) z_0; the midpoint m of a dyadic interval [a, b] (heap id h, root 1) is
    (W(a) + W(b))/2 + sqrt((b - a)/4) z_h. z are counter-RNG normals keyed by (ring, tick = node id, slot), so W(j)
    is a pure function of j: the same path whatever order or step it is evaluated in. Cost: depth normals."""
    top = np.int64(1) << depth
    if j <= 0:
        return 0.0
    if j > top:
        j = top
    a = np.int64(0)
    b = top
    wa = 0.0
    wb = math.sqrt(float(top)) * _rng.normal(key, ring, 0, slot)
    h = np.int64(1)
    while b - a > 1:
        m = (a + b) >> 1
        wm = 0.5 * (wa + wb) + math.sqrt((b - a) / 4.0) * _rng.normal(key, ring, h, slot)
        if j == m:
            return wm
        if j < m:
            b = m
            wb = wm
            h = 2 * h
        else:
            a = m
            wa = wm
            h = 2 * h + 1
    return wa if j == a else wb


@njit(cache=True)
def _fill_w(key, j, depth, n_lines, n_mod, out):
    """out[s, i] = W at grid j for slot s (0 freq, 1 power) and index i (lines 0..L-1, then modules L..L+M-1)."""
    for s in range(2):
        for i in range(n_lines + n_mod):
            out[s, i] = levy_bm(key, i, s, j, depth)


@njit(cache=True)
def laser_tick_row(k, dt_s, tcomp, grid_s, depth, f0, p0, f_rate, p_rate, f_rw, p_rw, rho_f, rho_p, module, n_mod,
                   rin_per_hz, enable, f_offset, p_scale, tune_hz, key_drift, key_rin, cache_j, w_lo, w_hi,
                   out_f, out_p):
    """Frequency (Hz) and power (W) of the L lines of one world over plant interval [k, k+1) (module docstring).

    f0, p0 (L,): static frequency (grid + static error) and nominal power at the module output. f_rate (Hz/s),
    p_rate (1/s): linear drift; f_rw (Hz/sqrt(s)), p_rw (1/sqrt(s)): random-walk coefficients; rho_f, rho_p: shared
    variance fraction per module; module (L,) int64 source laser, n_mod modules. rin_per_hz: one-sided RIN (1/Hz;
    0 = off). enable (L,) bool (laser_line_enable), f_offset (L,) Hz and p_scale (L,) scenario steps (failover),
    tune_hz (L,) laser.tune offset. Cache: cache_j (1,) int64 (grid index held in w_lo, -1 = empty), w_lo / w_hi
    (2, L + n_mod) the Brownian values at grid j and j + 1 (pure functions of j, so the cache never changes results).
    """
    n_lines = f0.shape[0]
    t = (k + 0.5) * dt_s * tcomp
    u = t / grid_s
    j = np.int64(math.floor(u))
    phi = u - j
    if cache_j[0] != j:
        if cache_j[0] >= 0 and cache_j[0] + 1 == j:
            for s in range(2):
                for i in range(n_lines + n_mod):
                    w_lo[s, i] = w_hi[s, i]
        else:
            _fill_w(key_drift, j, depth, n_lines, n_mod, w_lo)
        _fill_w(key_drift, j + 1, depth, n_lines, n_mod, w_hi)
        cache_j[0] = j
    sg = math.sqrt(grid_s)
    cf_c = math.sqrt(rho_f)
    cf_l = math.sqrt(1.0 - rho_f)
    cp_c = math.sqrt(rho_p)
    cp_l = math.sqrt(1.0 - rho_p)
    sd_rin = math.sqrt(0.5 * rin_per_hz / dt_s) if rin_per_hz > 0.0 else 0.0
    z_next = 0.0
    for l in range(n_lines):
        m = n_lines + module[l]
        wl_f = w_lo[0, l] + phi * (w_hi[0, l] - w_lo[0, l])
        wm_f = w_lo[0, m] + phi * (w_hi[0, m] - w_lo[0, m])
        wl_p = w_lo[1, l] + phi * (w_hi[1, l] - w_lo[1, l])
        wm_p = w_lo[1, m] + phi * (w_hi[1, m] - w_lo[1, m])
        out_f[l] = (f0[l] + f_rate[l] * t + f_rw[l] * sg * (cf_c * wm_f + cf_l * wl_f) + f_offset[l] + tune_hz[l])
        g = 1.0 + p_rate[l] * t + p_rw[l] * sg * (cp_c * wm_p + cp_l * wl_p)
        if g < 0.0:
            g = 0.0
        r = 1.0
        if sd_rin > 0.0:
            if l % 2 == 0:                      # one Box-Muller pair per two lines (ring word = line pair l // 2)
                z, z_next = _rng.normal2(key_rin, l >> 1, k, _SLOT_RIN)
            else:
                z = z_next
            r = 1.0 + sd_rin * z
            if r < 0.0:
                r = 0.0
        ps = p_scale[l] if p_scale[l] > 0.0 else 0.0     # optical power cannot go negative
        out_p[l] = p0[l] * g * ps * r if enable[l] else 0.0


@njit(cache=True)
def laser_tick(k, dt_s, tcomp, grid_s, depth, f0, p0, f_rate, p_rate, f_rw, p_rw, rho_f, rho_p, module, n_mod,
               rin_per_hz, enable, f_offset, p_scale, tune_hz, key_drift, key_rin, cache_j, w_lo, w_hi, out_f, out_p):
    """Batched :func:`laser_tick_row`: (B, L) arrays; rho_f, rho_p, rin_per_hz, keys (B,); cache_j (B, 1),
    w_lo / w_hi (B, 2, L + n_mod)."""
    for b in range(f0.shape[0]):
        laser_tick_row(k, dt_s, tcomp, grid_s, depth, f0[b], p0[b], f_rate[b], p_rate[b], f_rw[b], p_rw[b],
                       rho_f[b], rho_p[b], module, n_mod, rin_per_hz[b], enable[b], f_offset[b], p_scale[b],
                       tune_hz[b], key_drift[b], key_rin[b], cache_j[b], w_lo[b], w_hi[b], out_f[b], out_p[b])


@njit(cache=True, parallel=True)
def laser_tick_par(k, dt_s, tcomp, grid_s, depth, f0, p0, f_rate, p_rate, f_rw, p_rw, rho_f, rho_p, module, n_mod,
                   rin_per_hz, enable, f_offset, p_scale, tune_hz, key_drift, key_rin, cache_j, w_lo, w_hi, out_f,
                   out_p):
    """:func:`laser_tick` with prange over b (thread-count invariant)."""
    for b in nb.prange(f0.shape[0]):
        laser_tick_row(k, dt_s, tcomp, grid_s, depth, f0[b], p0[b], f_rate[b], p_rate[b], f_rw[b], p_rw[b],
                       rho_f[b], rho_p[b], module, n_mod, rin_per_hz[b], enable[b], f_offset[b], p_scale[b],
                       tune_hz[b], key_drift[b], key_rin[b], cache_j[b], w_lo[b], w_hi[b], out_f[b], out_p[b])


# ================================================================================================ Python API
def _arr_b(x, B, name, dtype=np.float64):
    a = np.asarray(x, dtype=dtype).reshape(-1)
    if a.size == 1:
        return np.full(B, a[0], dtype=dtype)
    if a.size != B:
        raise ValueError(f"{name}: shape {np.shape(x)} is not scalar or (B={B},)")
    return a.copy()


def _arr_bl(x, B, L, name, dtype=np.float64):
    """Broadcast to (B, L); a 1-D array of length L is per line, of length B (!= L) per world."""
    a = np.asarray(x, dtype=dtype)
    if a.ndim == 1:
        a = a.reshape(1, -1) if a.shape[0] == L else a.reshape(-1, 1)
    try:
        return np.ascontiguousarray(np.broadcast_to(a, (B, L))).copy()
    except ValueError:
        raise ValueError(f"{name}: shape {np.shape(x)} does not broadcast to (B={B}, L={L})") from None


def _keys(seeds, B, stream: int) -> np.ndarray:
    seeds = _arr_b(seeds, B, "seeds", np.int64)
    return np.array([_rng.stream_key(int(s), int(stream)) for s in seeds], dtype=np.uint64)


def _uniform_lm(device_seeds, lines: LineTable, tag: int, slot: int) -> tuple[np.ndarray, np.ndarray]:
    """Uniforms in (0, 1) per (b, line) and per (b, module) from Stream.VARIATION (device seed), tick = tag."""
    seeds = np.atleast_1d(np.asarray(device_seeds, dtype=np.int64))
    B, L, M = seeds.size, lines.n_lines, lines.n_lasers
    keys = _keys(seeds, B, _rng.Stream.VARIATION)
    ul = np.array([[_rng.uniform(keys[b], l, tag, slot) for l in range(L)] for b in range(B)])
    um = np.array([[_rng.uniform(keys[b], L + m, tag, slot) for m in range(M)] for b in range(B)])
    return ul, um


def draw_static_error_hz(device_seeds, lines: LineTable, bound_hz: float = LASER_ACCURACY_DEMONSTRATED_HZ, *,
                         common_weight: float = 0.0, dist: str = "uniform") -> np.ndarray:
    """Static absolute frequency error (B, L), Hz, per device (Stream.VARIATION, device seed).

    e_l = bound [w U_m + (1 - w) U_l] with U uniform in [-1, 1] (dist 'uniform'; |e| <= bound always) or standard
    normals scaled so that the bound is 3 sigma (dist 'normal3sigma'); m = the line's source laser, w =
    common_weight (0: independent lines; 1: one offset per module). Defaults: +-15 GHz demonstrated (YOLE); the spec
    corner is LASER_ACCURACY_SPEC_HZ = 20 GHz. Do not add it on top of a realized laser.line_offset_ghz (realize
    already puts that leaf's variation in PlantParams.laser.line_freq_hz).
    """
    if not 0.0 <= common_weight <= 1.0:
        raise ValueError("common_weight must be in [0, 1]")
    ul, um = _uniform_lm(device_seeds, lines, _TAG_STATIC, 0)
    if dist == "uniform":
        xl, xm = 2.0 * ul - 1.0, 2.0 * um - 1.0
        return bound_hz * (common_weight * xm[:, lines.source_laser] + (1.0 - common_weight) * xl)
    if dist == "normal3sigma":
        from scipy.special import ndtri
        xl, xm = ndtri(ul), ndtri(um)
        w = common_weight
        return bound_hz / 3.0 * (math.sqrt(w) * xm[:, lines.source_laser] + math.sqrt(1.0 - w) * xl)
    raise ValueError("dist must be 'uniform' or 'normal3sigma'")


def drift_from_24h_bound(bound: float, mode: str, *, device_seeds=None, lines: LineTable | None = None,
                         what: str = "freq") -> tuple[np.ndarray | float, float]:
    """(linear rate, random-walk coefficient) consistent with a 24 h drift bound (ASSUMED split; module docstring).

    bound: Hz (frequency, e.g. 3.5e9 for 0.02 nm at 1310 nm) or a fraction (power, 0.025).
    mode 'linear': per-line rates uniform in [-bound, +bound]/24 h (device stream; needs device_seeds and lines),
                   no random walk; 'worst_linear': every line at +bound/24 h; 'random_walk': no linear part, a random
                   walk whose 24 h rms is bound/2 (95 % of lines inside the bound); 'none': zeros.
    Returns (rate: scalar or (B, L), rw per sqrt(s)).
    """
    if mode == "none":
        return 0.0, 0.0
    if mode == "worst_linear":
        return bound / SECONDS_PER_24H, 0.0
    if mode == "random_walk":
        return 0.0, 0.5 * bound / math.sqrt(SECONDS_PER_24H)
    if mode == "linear":
        if device_seeds is None or lines is None:
            raise ValueError("mode 'linear' draws per-line rates: give device_seeds and lines")
        ul, _um = _uniform_lm(device_seeds, lines, _TAG_RATE_F if what == "freq" else _TAG_RATE_P, 0)
        return bound * (2.0 * ul - 1.0) / SECONDS_PER_24H, 0.0
    raise ValueError("mode must be one of 'none', 'linear', 'worst_linear', 'random_walk'")


@dataclass
class LaserHw:
    """Realized laser lines for B worlds and L lines, SI. Build with ``build`` or ``from_plant_params``."""

    B: int
    L: int
    dt_s: float
    lines: LineTable
    f0_hz: np.ndarray            # (B, L) grid + static error
    p0_w: np.ndarray             # (B, L) power at the module output
    f_rate_hz_per_s: np.ndarray  # (B, L)
    p_rate_per_s: np.ndarray     # (B, L)
    f_rw_hz_per_rt_s: np.ndarray  # (B, L)
    p_rw_per_rt_s: np.ndarray    # (B, L)
    rho_f: np.ndarray            # (B,)
    rho_p: np.ndarray            # (B,)
    rin_per_hz: np.ndarray       # (B,)
    grid_s: float = DRIFT_GRID_S_DEFAULT
    depth: int = LEVY_DEPTH
    tune_lsb_hz: float = 0.0

    @classmethod
    def build(cls, B: int, dt_s: float, lines: LineTable, *, power_w, static_error_hz=0.0, f_rate_hz_per_s=0.0,
              p_rate_per_s=0.0, f_rw_hz_per_rt_s=0.0, p_rw_per_rt_s=0.0, rho_f=0.0, rho_p=0.0, rin_per_hz=0.0,
              grid_s: float = DRIFT_GRID_S_DEFAULT, depth: int = LEVY_DEPTH, tune_lsb_hz: float = 0.0
              ) -> "LaserHw":
        """power_w, static_error_hz and the drift coefficients broadcast to (B, L) (1-D of length L: per line;
        of length B: per world); rho_f, rho_p (shared variance fraction per module, 0 = independent lines, 1 = the
        module drifts as one) and rin_per_hz (one-sided, 1/Hz; 10^(dB/10)) per world."""
        L = lines.n_lines
        rf = _arr_b(rho_f, B, "rho_f")
        rp = _arr_b(rho_p, B, "rho_p")
        if np.any((rf < 0) | (rf > 1)) or np.any((rp < 0) | (rp > 1)):
            raise ValueError("rho_f and rho_p must be in [0, 1]")
        rin = _arr_b(rin_per_hz, B, "rin_per_hz")
        if np.any(rin < 0):
            raise ValueError("rin_per_hz must be >= 0 (linear 1/Hz)")
        if not (grid_s > 0 and 1 <= int(depth) <= 60):
            raise ValueError("need grid_s > 0 and 1 <= depth <= 60")
        p0 = _arr_bl(power_w, B, L, "power_w")
        if np.any(p0 < 0):
            raise ValueError("power_w must be >= 0")
        return cls(B=int(B), L=L, dt_s=float(dt_s), lines=lines,
                   f0_hz=lines.freq_hz[None, :] + _arr_bl(static_error_hz, B, L, "static_error_hz"), p0_w=p0,
                   f_rate_hz_per_s=_arr_bl(f_rate_hz_per_s, B, L, "f_rate_hz_per_s"),
                   p_rate_per_s=_arr_bl(p_rate_per_s, B, L, "p_rate_per_s"),
                   f_rw_hz_per_rt_s=_arr_bl(f_rw_hz_per_rt_s, B, L, "f_rw_hz_per_rt_s"),
                   p_rw_per_rt_s=_arr_bl(p_rw_per_rt_s, B, L, "p_rw_per_rt_s"), rho_f=rf, rho_p=rp, rin_per_hz=rin,
                   grid_s=float(grid_s), depth=int(depth), tune_lsb_hz=float(tune_lsb_hz))

    @classmethod
    def from_plant_params(cls, pp, lines: LineTable | None = None, **kw) -> "LaserHw":
        """From PlantParams.laser: f0 = line_freq_hz (grid + the realized laser.line_offset_ghz), power_w,
        rin_per_hz; laser.tune.lsb_hz when present. Drift parameters have no schema keys yet: keyword arguments
        (docs/handoff/plant_sensing.md). ``lines`` defaults to :meth:`LineTable.from_plant_params`."""
        las = pp.laser
        lines = LineTable.from_plant_params(pp) if lines is None else lines
        B = int(pp.batch)
        f = np.asarray(las.line_freq_hz, dtype=float)
        err = f - lines.freq_hz[None, :]
        tune = pp.values.get("laser.tune.lsb_hz") if hasattr(pp, "values") else None
        args = dict(power_w=np.asarray(las.power_w, dtype=float), static_error_hz=err,
                    rin_per_hz=np.asarray(las.rin_per_hz, dtype=float).reshape(-1),
                    tune_lsb_hz=float(np.asarray(tune).reshape(-1)[0]) if tune is not None else 0.0)
        args.update(kw)
        return cls.build(B, float(pp.timing.dt_s), lines, **args)


class LaserBank:
    """Per-tick line frequencies and powers for B worlds (07 P1-12, P1-13, P0-10).

    Usage::
        hw = LaserHw.build(B, 1e-6, g26_line_table().select('tx'), power_w=10e-3, rin_per_hz=10**(-14.5))
        bank = LaserBank(hw, scenario_seeds=rng.batch_seeds(3, B), time_compression=1.0)
        f_hz, p_w = bank.tick(k, enable=..., f_offset_hz=..., p_scale=..., tune_codes=...)
    """

    def __init__(self, hw: LaserHw, scenario_seeds, noise_seeds=None, time_compression: float = 1.0):
        self.hw = hw
        B, L = hw.B, hw.L
        if not (time_compression >= 1.0 and math.isfinite(time_compression)):
            raise ValueError("time_compression must be finite and >= 1 (07 P0-10)")
        self.tcomp = float(time_compression)
        M = hw.lines.n_lasers
        self.cache_j = np.full((B, 1), -1, np.int64)
        self.w_lo = np.zeros((B, 2, L + M))
        self.w_hi = np.zeros((B, 2, L + M))
        self.f = np.zeros((B, L))
        self.p = np.zeros((B, L))
        self._en = np.ones((B, L), np.bool_)
        self._zero = np.zeros((B, L))
        self._one = np.ones((B, L))
        self.reset(scenario_seeds, noise_seeds)

    def reset(self, scenario_seeds=None, noise_seeds=None) -> None:
        """Re-key the drift (scenario seed) and the RIN (noise seed; default the scenario seed)."""
        B = self.hw.B
        if scenario_seeds is not None:
            self.key_drift = _keys(scenario_seeds, B, _rng.Stream.LASER_DRIFT)
            if noise_seeds is None:
                noise_seeds = scenario_seeds
        if noise_seeds is not None:
            self.key_rin = _keys(noise_seeds, B, _rng.Stream.LASER_RIN)
        self.cache_j[:] = -1

    def tick(self, k: int, enable=None, f_offset_hz=None, p_scale=None, tune_codes=None, parallel: bool = False
             ) -> tuple[np.ndarray, np.ndarray]:
        """(freq_hz, power_w) (B, L) of interval [k, k+1); views of internal buffers (copy to keep them).

        enable: bool, or a power factor in [0, 1] as the scenario signal ``laser.line_enable`` defines it (0 = off,
        a ramp is a turn-on; values are clipped to [0, 1] and multiply the power). f_offset_hz: Hz added to the line
        frequency (scenario ``laser.dfreq_hz``, failover steps). p_scale: linear power factor (>= 0; for the scenario
        ``laser.power_delta_db`` pass 10**(dB/10)). tune_codes: laser.tune codes (x hw.tune_lsb_hz)."""
        hw = self.hw
        B, L = hw.B, hw.L
        fo = self._zero if f_offset_hz is None else _arr_bl(f_offset_hz, B, L, "f_offset_hz")
        ps = self._one if p_scale is None else _arr_bl(p_scale, B, L, "p_scale")
        if enable is None:
            en = self._en
        else:
            ef = np.clip(_arr_bl(enable, B, L, "enable"), 0.0, 1.0)   # bool -> 0/1; a fraction scales the power
            en = ef > 0.0
            if np.any((ef > 0.0) & (ef < 1.0)):
                ps = ps * ef
        tu = self._zero if tune_codes is None else _arr_bl(tune_codes, B, L, "tune_codes") * hw.tune_lsb_hz
        fn = laser_tick_par if parallel else laser_tick
        fn(np.int64(k), hw.dt_s, self.tcomp, hw.grid_s, hw.depth, hw.f0_hz, hw.p0_w, hw.f_rate_hz_per_s,
           hw.p_rate_per_s, hw.f_rw_hz_per_rt_s, hw.p_rw_per_rt_s, hw.rho_f, hw.rho_p, hw.lines.source_laser,
           hw.lines.n_lasers, hw.rin_per_hz, en, fo, ps, tu, self.key_drift, self.key_rin, self.cache_j, self.w_lo,
           self.w_hi, self.f, self.p)
        return self.f, self.p
