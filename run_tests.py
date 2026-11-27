"""Minimal zero-dependency test runner.

Runs plain ``test_*`` functions from ``tests/test_*.py``. Works when pytest is
not installed; with pytest available, ``pytest tests`` also works since the
tests are standard pytest-style functions.

Usage: python run_tests.py
"""

from __future__ import annotations

import importlib.util
import os
import sys
import traceback

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)


def load_module(path):
    name = "t_" + os.path.splitext(os.path.basename(path))[0]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    test_dir = os.path.join(ROOT, "tests")
    passed = failed = 0
    failures = []
    for filename in sorted(os.listdir(test_dir)):
        if not filename.startswith("test_") or not filename.endswith(".py"):
            continue
        module = load_module(os.path.join(test_dir, filename))
        for attr in sorted(dir(module)):
            if not attr.startswith("test_"):
                continue
            fn = getattr(module, attr)
            if not callable(fn):
                continue
            try:
                fn()
                passed += 1
                print(f"PASS  {filename}::{attr}")
            except Exception:
                failed += 1
                failures.append((filename, attr, traceback.format_exc()))
                print(f"FAIL  {filename}::{attr}")
    print(f"\n{passed} passed, {failed} failed")
    for filename, attr, tb in failures:
        print(f"\n--- {filename}::{attr} ---\n{tb}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
