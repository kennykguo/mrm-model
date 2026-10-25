"""Thermal Locking Controller (TLC) package for Lightmatter G26 MRM Plant.

Provides modular signal processing, state estimation, and setpoint blocks:
- B1 Dither: Orthogonal sinusoidal dither generation with adaptive amplitude scheduling
- B2 Conditioner: Linear-phase FIR lowpass filtering and squelch variance detection
- B3 Lock-In: Sliding 4 ms Hann-windowed dual-harmonic I/Q demodulator
- B4 Reduced Thermal Model: Discrete-time state-space (Ad, Bd, Cd) thermal crosstalk model
- B4 UKF: Dual-rate 24-state Unscented Kalman Filter fusing fast DC and slow harmonic ratios
- B6 Setpoints: Red-flank inflection point setpoint generator for thermo-optic stability
"""

from mrm.control.tlc.b1_dither import B1Dither, DitherConfig, DitherGenerator
from mrm.control.tlc.b2_conditioner import (
    B2Conditioner,
    ConditionerConfig,
    SignalConditioner,
)
from mrm.control.tlc.b3_lockin import B3LockIn, LockInConfig, LockInDemodulator
from mrm.control.tlc.b4_thermal_model import ReducedThermalModel
from mrm.control.tlc.b4_ukf import B4UKF, DualRateUKF, UKFConfig
from mrm.control.tlc.b6_setpoints import (
    B6Setpoints,
    SetpointConfig,
    SetpointGenerator,
)

__all__ = [
    "DitherConfig",
    "DitherGenerator",
    "B1Dither",
    "ConditionerConfig",
    "SignalConditioner",
    "B2Conditioner",
    "LockInConfig",
    "LockInDemodulator",
    "B3LockIn",
    "ReducedThermalModel",
    "DualRateUKF",
    "B4UKF",
    "UKFConfig",
    "SetpointGenerator",
    "B6Setpoints",
    "SetpointConfig",
]
