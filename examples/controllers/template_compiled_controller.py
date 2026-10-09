"""Template: a compiled controller (07 P0-9 flat int64 ABI) with an optional fast_kernel (07 P0-7).

``init(spec_i64, spec_f64) -> (p_i64, p_f64, s_i64, s_f64)`` and ``step(p_i64, p_f64, s_i64, s_f64, obs_i64,
act_i64)`` are numba functions; ``CompiledController`` packages them and also runs them on the Python path, so the
same controller works in the engine's compiled loop and in a Python harness. Channels are found by id, never by a
hard-coded offset: ``obs_offset(spec_i64, ObsId.X)``, ``act_offset(spec_i64, ActId.X)``. Your own parameters come
after the spec header (``user_i64``/``user_f64``). Everything must also run under NUMBA_DISABLE_JIT=1.

The example is the same integral lock as template_python_controller.py. The full compiled baselines are
``mrm/control/baselines/pi_lock.py`` (pi_init / pi_step), ``dither_lockin.py`` (dither_init / dither_step) and, with a
fast kernel, ``bias_dither.py`` (bias_init / bias_step / bias_fast).

fast_kernel signature (optional): ``fast(p_i64, p_f64, s_i64, s_f64, fobs, fact)`` with
``fobs = [tick, N, new[N], code[N], sample_tick[N]]`` (one released monitor sample per ring) and
``fact = [heater_offset[N], bias_offset[N]]`` (codes added to the latched commands before the DAC FIFO). It shares the
state arrays with ``step``.
"""
from __future__ import annotations

import numba as nb
import numpy as np

from mrm.control import api
from mrm.control.api import CompiledController
from mrm.core.channels import ActId, ObsId

_LATEST, _APPLIED, _HEATER = int(ObsId.TX_MONITOR_LATEST), int(ObsId.TX_HEATER_APPLIED), int(ActId.TX_HEATER)


@nb.njit(cache=True)
def my_init(spec_i64, spec_f64):
    n = spec_i64[api.SPEC_I_N_RINGS]
    p = np.zeros(6, dtype=np.int64)
    p[0] = n
    p[1] = api.obs_offset(spec_i64, _LATEST)          # monitor code (held), one per ring
    p[2] = api.obs_offset(spec_i64, _APPLIED)         # latched heater register (warm start)
    p[3] = api.act_offset(spec_i64, _HEATER)
    p[4] = spec_i64[api.SPEC_I_HEATER_MIN]
    p[5] = spec_i64[api.SPEC_I_HEATER_MAX]
    pf = np.zeros(3)
    pf[0] = api.user_f64(spec_f64, 0)                 # peak code
    pf[1] = api.user_f64(spec_f64, 1)                 # fraction
    pf[2] = api.user_f64(spec_f64, 2)                 # k_i (codes per step per unit normalized error)
    s_i = np.ones(1, dtype=np.int64)                  # [first]
    s_f = np.zeros(n)                                 # heater code per ring (float)
    return p, pf, s_i, s_f


@nb.njit(cache=True)
def my_step(p, pf, s, sf, obs, act):
    n = p[0]
    for i in range(n):
        if s[0] == 1:
            # warm start from the latched register; obs_offset is -1 when the channel is not exposed
            sf[i] = float(obs[p[2] + i]) if p[2] >= 0 else float(p[4])
        e = (float(obs[p[1] + i]) - pf[1] * pf[0]) / pf[0]
        u = sf[i] + pf[2] * e
        sf[i] = min(max(u, float(p[4])), float(p[5]))
        act[p[3] + i] = np.int64(np.rint(sf[i]))
    s[0] = 0


def make_my_compiled_lock(peak_codes: float = 58000.0, fraction: float = 0.7, k_i_codes: float = 400.0):
    """A CompiledController ready for mrm.sim.runner.run(..., controller=...)."""
    return CompiledController("my_compiled_lock", my_init, my_step, params_f64=[peak_codes, fraction, k_i_codes])
