"""Combine recorded generic FEM references into the same experiment interface.

Only coefficients actually extracted are replaced. Couplers, loss, heater and
sensor electronics are still assumed. This is NOT a Lightmatter reconstruction.
"""

import json
from dataclasses import replace
from pathlib import Path

from mrm_model.cli import write_json
from mrm_model.config import default_config
from mrm_model.simulation import _steady_state
from mrm_model.thermal import ThermalParams


def main():
    root = Path(__file__).resolve().parents[1]
    heat = json.loads((root / "examples/thermal_fem.json").read_text())["fine"]
    mode = json.loads((root / "examples/waveguide_modes.json").read_text())["fine"]
    config = default_config()
    config = replace(
        config,
        thermal=ThermalParams(**heat["thermal_params"]),
        optical=(replace(config.optical[0], group_index=mode["ng"]),),
    )
    # Choose a reproducible *synthetic* reference at the available initial bias.
    # Keep iterating until absorptive self-heating and the aligned spectrum agree.
    for _ in range(12):
        thermal, _ = _steady_state(config)
        p = config.optical[0]
        reference = (
            config.run.laser_frequency_hz[0]
            - 25e9
            - p.df_dtemperature_hz_per_k * (thermal.ring_temperatures_k[0] - p.reference_temperature_k)
        )
        config = replace(config, optical=(replace(p, reference_frequency_hz=float(reference)),))
    provenance = dict(
        config.provenance,
        profile="generic FEM-informed SOI ring hypothesis; NOT Lightmatter",
        extracted={"thermal": "examples/thermal_fem.json", "group_index": "examples/waveguide_modes.json"},
        assumptions="Ring loss/couplers, curved-mode correction, signed thermal sensitivity, heater "
        "electronics and sensor remain assumed/public constraints. Resonance is synthetically aligned.",
        calibrated=False,
    )
    config = replace(config, provenance=provenance)
    destination = root / "examples/fem_ring.json"
    write_json(destination, config.to_dict())
    print(destination)


if __name__ == "__main__":
    main()
