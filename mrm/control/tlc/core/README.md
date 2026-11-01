# TLC core (vendored)

The ECE496 thermal locking controller, blocks B1-B10, copied from
[msilve160/ECE496_project](https://github.com/msilve160/ECE496_project) at commit `f98ce84`
(`thermal_control/tlc/*.py`, `thermal_control/params.py`, `thermal_control/optics.py`). Apache License 2.0, see
[`LICENSE`](LICENSE). Each file starts with a provenance line.

The core is kept in the source's units (GHz, heater power fraction) and stays diffable against the source. The
adapter in `mrm/control/tlc/controller.py` is the only mrm-specific code.

## Changes from the source
Only the deviations allowed by docs/TLC_PORT_SPEC_AND_PLAN.md 2.3. Every change is listed here, and
`tests/tlc/test_core_equivalence.py` checks that with the source defaults the vendored core is bit-identical to the
source core on the source plant.

| # | Change | Files | Spec 2.3 |
|---|---|---|---|
| 1 | Imports: `from params` / `from optics` become relative (`from .params`, `from .optics`) | all | packaging |
