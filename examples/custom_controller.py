"""Example interface, deliberately simple: hold a configured heater code.

Use controller.name: examples.custom_controller:MyController from the repo root.
Only observations and the public digital interface are provided to this class.
"""
from numbers import Integral

import numpy as np


class MyController:
    def __init__(self, code=1000):
        if isinstance(code, (bool, np.bool_)) or not isinstance(code, Integral):
            raise ValueError("code must be an integer DAC code")
        self.code = int(code)

    def reset(self, interface):
        if not 0 <= self.code <= interface.heater_max_code:
            raise ValueError("code exceeds this heater DAC range")
        self.codes = np.full(interface.rings, self.code, dtype=np.int64)

    def step(self, observation):
        # observation.fresh distinguishes a new ADC sample from a held one.
        # A real algorithm would update internal state only on useful data.
        return self.codes.copy()
