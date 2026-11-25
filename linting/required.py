# run both required checks and return a failing status when either one fails.
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent


def main():
    failed = False
    for name, command in (
        ("ruff", [sys.executable, "-m", "ruff", "check", "."]),
        ("function documentation", [sys.executable, str(HERE / "stage_2_structure" / "check.py")]),
    ):
        print(f"checking {name}", flush=True)
        result = subprocess.run(command, cwd=HERE.parent)
        failed |= result.returncode != 0
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
