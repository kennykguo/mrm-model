"""Waveform mode: short time-domain runs of one ring and the normalized data-modulation LUT (07 P2-6).

Modules (docs/handoff/waveform.md is the user guide):

- :mod:`mrm.waveform.tcmt_time`: CW/CCW TCMT with backscatter driven by the instantaneous junction voltage
  (exponential integrator), its small-signal linearization and the Sacher & Poon closed form (03 section 5.6);
  :class:`RingTCMT` builds the junction-voltage tables from PlantParams through the plant's phase model.
- :mod:`mrm.waveform.driver`: PRBS / Markov patterns, the Gaussian-edged NRZ driver with output resistance and the
  junction network with C_j(V), R_s(V) (:class:`DriverJunction`).
- :mod:`mrm.waveform.eye`: end-to-end runs (:func:`simulate`), eye metrics (height, width, OMA, ER, levels), the
  small-signal EO response and its 3 dB bandwidth and peaking (:func:`eo_bandwidth`, CAL-10 / VAL-3 replica).
- :mod:`mrm.waveform.lut`: the normalized-coordinate LUT (artefact ``waveform_lut_<name>.npz``), its numba lookup
  :func:`lut_eval_nb` for the engine and the builder.
"""
from mrm.waveform.driver import DriverJunction, JunctionRun, markov_bits, pattern_stats, prbs
from mrm.waveform.eye import EyeMetrics, WaveformResult, bandwidth_3db, eo_bandwidth, eo_response, eye_metrics, simulate
from mrm.waveform.tcmt_time import RingTCMT, h_opt_closed_form, optical_response_rates, port_powers

__all__ = [
    "RingTCMT", "h_opt_closed_form", "optical_response_rates", "port_powers",
    "DriverJunction", "JunctionRun", "prbs", "markov_bits", "pattern_stats",
    "WaveformResult", "EyeMetrics", "simulate", "eye_metrics", "eo_response", "eo_bandwidth", "bandwidth_3db",
    "WaveformLUT", "LutDomainError", "build_lut", "load_lut", "lut_eval_nb", "lut_eval_strict_nb", "lut_query_nb",
    "normalized_node", "normalized_correction",
]

_LUT_NAMES = ("LutDomainError", "WaveformLUT", "build_lut", "load_lut", "lut_eval_nb", "lut_eval_strict_nb",
              "lut_query_nb", "normalized_correction", "normalized_node")


def __getattr__(name):
    # lazy: ``python -m mrm.waveform.lut`` must not import the module twice (PEP 562)
    if name in _LUT_NAMES:
        from mrm.waveform import lut
        return getattr(lut, name)
    raise AttributeError(f"module 'mrm.waveform' has no attribute {name!r}")
