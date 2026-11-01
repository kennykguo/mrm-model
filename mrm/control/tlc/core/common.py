# Vendored from msilve160/ECE496_project@f98ce84, thermal_control/tlc/common.py (Apache-2.0, see LICENSE).
# Changes from the source are listed in mrm/control/tlc/core/README.md.
"""Shared definitions: ring count and the per-ring supervisor modes (B5)."""
from enum import IntEnum

from .params import N_RINGS

N = N_RINGS


class Mode(IntEnum):
    INIT = 0
    ACQUIRE = 1
    TRACK = 2
    HOLD = 3
    REACQUIRE = 4
    FAULT = 5
