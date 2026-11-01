"""The ECE496 thermal locking controller (TLC) on the mrm controller API (docs/TLC_PORT_SPEC_AND_PLAN.md).

Layout: ``controller`` is the adapter (registry name ``tlc``), the only mrm-specific code. It owns the units
(``units``), the monitor sample buffer (``sample_buffer``), the multirate schedule and the telemetry
(``telemetry_log``). The core runs in the source's own units (GHz, heater power fraction); in Phase 1 it is
``stub_core``, which holds the warm command and adds the source's dither tones.

Like every controller, nothing here imports mrm.plant, mrm.sim or mrm.validation (tests/tlc/test_boundaries.py).
"""
