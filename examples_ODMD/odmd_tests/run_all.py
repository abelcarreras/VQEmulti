"""
Run every validation script in this folder.

    python run_all.py           # all tests
    python run_all.py --plot    # also show plots
"""
import importlib
import pathlib
import sys
import odmd
from helpers import run_tests

# Guard against importing a stale copy of odmd.py from somewhere else.
print(f"Using ODMD from: {pathlib.Path(odmd.__file__).resolve()}\n")

here = pathlib.Path(__file__).parent
modules = sorted(p.stem for p in here.glob("test_*.py"))
plot = "--plot" in sys.argv

failed = []
for name in modules:
    print("=" * 70 + f"\n{name}\n" + "=" * 70)
    mod = importlib.import_module(name)
    if not run_tests(vars(mod), plot=plot):
        failed.append(name)
    print()

print("ALL PASSED" if not failed else f"FAILED: {failed}")
sys.exit(1 if failed else 0)
