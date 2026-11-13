"""
Path setup shared by every test file and run_all.py.

Adds two folders to Python's module search path:
  * examples_ODMD/  -> so `from odmd import ODMD` works
  * this folder     -> so `import signals` / `from helpers import ...` work
                       even when the tests are run as a package
                       (e.g. `python -m unittest tests_odmd.test_01_exact`
                       from the project root, or from PyCharm).

Change EXAMPLES_DIR if the package layout changes.
"""
import sys
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TESTS_DIR.parent
EXAMPLES_DIR = PROJECT_ROOT / "examples_ODMD"

for folder in (EXAMPLES_DIR, TESTS_DIR):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))
