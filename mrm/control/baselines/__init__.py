"""Built-in baseline controllers (07 P1-17 registry names).

Importing this package runs every baseline module's ``@register`` decorator, so ``mrm.control.registry.create(name)``
finds them without an explicit import (the registry imports this package once on its first miss). Registered names:

- controllers A (feedback locks): ``pi_lock``, ``pi_lock_c``, ``dither_lockin``, ``dither_lockin_c``,
  ``bias_dither``, ``es_po``, ``es_sine``, ``feedforward_pi``, ``midranging`` (docs/handoff/controllers_a.md);
- controllers B (acquisition, WDM, bit statistics): ``ramp_acquire``, ``acquire_then_lock``, ``bitstat_lock``,
  ``wdm_supervisor`` (docs/handoff/controllers_b.md);
- controllers C (model based): ``kalman_lqg``, ``kalman_lqg_c``, ``mpc``, ``mpc_c`` (docs/handoff/controllers_c.md).

The Truth-fed oracle (mrm.validation.oracle) is not a baseline and is never registered (07 P1-1).
"""
from mrm.control.baselines import (bias_dither, bitstat_lock, dither_lockin, extremum_seeking,  # noqa: F401
                                   feedforward, kalman_lqg, midranging, mpc, pi_lock, ramp_acquire,
                                   wdm_supervisor)

__all__ = ["bias_dither", "bitstat_lock", "dither_lockin", "extremum_seeking", "feedforward", "kalman_lqg",
           "midranging", "mpc", "pi_lock", "ramp_acquire", "wdm_supervisor"]
