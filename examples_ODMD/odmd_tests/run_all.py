"""
Discover and run every test_*.py in this folder with unittest.

    python run_all.py                    # all tests
    ODMD_PLOT=1 python run_all.py        # also show plots
    ODMD_VERBOSE=0 python run_all.py     # only the pass/fail lines
"""
import pathlib
import sys
import unittest

try:                         # imported as part of the package
    from . import _imports   # noqa: F401
except ImportError:          # run as a script / top-level module
    import _imports          # noqa: F401

import odmd

# Guard against importing a stale copy of odmd.py from somewhere else.
print(f"Using ODMD from: {pathlib.Path(odmd.__file__).resolve()}\n")

here = pathlib.Path(__file__).resolve().parent
suite = unittest.defaultTestLoader.discover(str(here), pattern="test_*.py",
                                            top_level_dir=str(here))
result = unittest.TextTestRunner(verbosity=2).run(suite)
sys.exit(0 if result.wasSuccessful() else 1)
