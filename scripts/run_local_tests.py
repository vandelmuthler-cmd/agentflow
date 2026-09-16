from __future__ import annotations

import importlib
import inspect
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    count = 0
    for path in sorted(Path("tests").glob("test_*.py")):
        module = importlib.import_module(f"tests.{path.stem}")
        for name, function in inspect.getmembers(module, inspect.isfunction):
            if name.startswith("test_") and not inspect.signature(function).parameters:
                function()
                count += 1
    print(f"ALL_NO_FIXTURE_TESTS_OK={count}")


if __name__ == "__main__":
    main()
