# check that the active controller and CLI paths do not load optional learning code.
from __future__ import annotations

import importlib.abc
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


# resolve every built-in controller while learning imports are forbidden.
def _verify_import_boundary() -> None:
    sys.path.insert(0, str(ROOT))
    assert importlib.util.find_spec("mrm.sim.gym_env") is None
    assert importlib.util.find_spec("mrm.control.fxp_mlp") is None
    blocked = ("gymnasium", "mrm.sim.gym_env", "mrm.control.fxp_mlp", "archive.rl")
    attempts = []

    class BlockLearningImports(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if any(fullname == name or fullname.startswith(name + ".") for name in blocked):
                attempts.append(fullname)
                raise ImportError("learning imports are disabled for this check - " + fullname)

    sys.meta_path.insert(0, BlockLearningImports())
    import mrm.cli
    import mrm.control.cli
    import mrm.sim.cli
    from mrm.control import fxp, registry
    from mrm.plant.plant import Plant
    from mrm.sim import kernel, runner

    names = registry.available()
    for name in names:
        registry.resolve(name)
    assert "fxp_pi" in names
    assert fxp.FxpPi
    assert not attempts, attempts
    print(json.dumps({"controllers": len(names), "learning_imports": attempts}))


# a fresh process prevents earlier imports from hiding a learning dependency.
def test_core_controller_registry_without_learning() -> None:
    result = subprocess.run(
        [sys.executable, __file__, "--verify"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout)["learning_imports"] == []


if __name__ == "__main__":
    _verify_import_boundary()
