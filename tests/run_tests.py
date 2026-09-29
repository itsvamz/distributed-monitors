"""Plain-assert test runner (works without pytest):  python -m tests.run_tests"""
import importlib
import sys
import traceback

MODULES = ["tests.test_monitors", "tests.test_fusion", "tests.test_game", "tests.test_pipeline"]


def main():
    total = failed = 0
    for m in MODULES:
        mod = importlib.import_module(m)
        for name in sorted(dir(mod)):
            if name.startswith("test_"):
                total += 1
                try:
                    getattr(mod, name)()
                    print("PASS", m.split(".")[-1], name)
                except Exception:
                    failed += 1
                    print("FAIL", m, name)
                    traceback.print_exc()
    print(f"\n{total - failed}/{total} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
