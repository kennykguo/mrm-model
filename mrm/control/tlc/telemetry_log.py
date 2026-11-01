"""In-memory log of controller telemetry, one row per core step.

mrm's runner records only Truth fields, never ``controller.telemetry()``, so the adapter keeps its own log;
``tools/tlc_run.py`` saves it into the run directory as ``tlc_telemetry.npz``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


class TelemetryLog:
    def __init__(self):
        self._t: list[float] = []
        self._rows: list[dict[str, np.ndarray]] = []

    def __len__(self) -> int:
        return len(self._t)

    def record(self, t_s: float, telemetry: dict[str, np.ndarray]) -> None:
        self._t.append(float(t_s))
        self._rows.append({k: np.array(v) for k, v in telemetry.items()})

    def to_arrays(self) -> dict[str, np.ndarray]:
        """{'t_s': (S,), key: (S, B, N)}; keys are those of the first row."""
        out = {"t_s": np.asarray(self._t, dtype=np.float64)}
        if self._rows:
            out.update({k: np.stack([r[k] for r in self._rows]) for k in self._rows[0]})
        else:
            out["state"] = np.zeros((0, 0, 0))
        return out

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        np.savez_compressed(path, **self.to_arrays())
        return path
