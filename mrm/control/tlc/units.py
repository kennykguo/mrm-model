"""Unit laws between the TLC core (GHz, heater power fraction u in [0, 1]) and mrm (pm, DAC and ADC codes).

The G26 heater is driven by a voltage DAC, so heater power goes as code squared (docs/EXPLAINED.md, heater
section): u = (code / code_max)^2 and code = round(code_max sqrt(u)). The core asks for a power fraction, as it
did on the ECE496 plant, and this law makes that true on G26.

Detuning sign: the source's delta (GHz) grows with heating (``thermal_control/plant.py``: delta = unheated +
kappa T - laser), as mrm's detuning (resonance minus line, pm) does, so pm = +pm_per_ghz * GHz.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

C_M_PER_S = 299_792_458.0


@dataclass(frozen=True)
class HeaterLaw:
    """Power fraction <-> heater DAC code for a voltage-mode DAC (power proportional to code squared)."""
    code_min: int
    code_max: int

    def __post_init__(self):
        if not 0 <= self.code_min < self.code_max:
            raise ValueError(f"HeaterLaw: need 0 <= code_min < code_max, got {self.code_min}, {self.code_max}")

    @property
    def u_min(self) -> float:
        """Smallest reachable power fraction (at code_min)."""
        return (self.code_min / self.code_max) ** 2

    def u_from_code(self, code: np.ndarray) -> np.ndarray:
        return (np.asarray(code, dtype=np.float64) / self.code_max) ** 2

    def code_from_u(self, u: np.ndarray) -> np.ndarray:
        """Nearest DAC code, clamped to [code_min, code_max]; NaN goes to code_min (heater off)."""
        u = np.nan_to_num(np.asarray(u, dtype=np.float64), nan=0.0)
        code = np.rint(self.code_max * np.sqrt(np.clip(u, 0.0, 1.0)))
        return np.clip(code, self.code_min, self.code_max).astype(np.int64)

    def saturated(self, u: np.ndarray) -> np.ndarray:
        """True where the command is outside the reachable power range [u_min, 1]."""
        u = np.asarray(u, dtype=np.float64)
        return (u < self.u_min) | (u > 1.0)


def level_from_codes(codes: np.ndarray, full_scale_codes: int) -> np.ndarray:
    """Monitor ADC codes as a fraction of full scale (the core's normalised PD level)."""
    return np.asarray(codes, dtype=np.float64) / float(full_scale_codes)


def pm_per_ghz(line_hz: np.ndarray) -> np.ndarray:
    """Wavelength step (pm) of a 1 GHz frequency step at each line: lambda^2 / c (5.72 pm at 1310 nm)."""
    lam_m = C_M_PER_S / np.asarray(line_hz, dtype=np.float64)
    return lam_m ** 2 / C_M_PER_S * 1e9 * 1e12
