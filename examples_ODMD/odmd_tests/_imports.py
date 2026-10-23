import sys
from pathlib import Path

# This file adds examples_ODMD/ to Python's module search path.
# Change accordingly to package organization.

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES_DIR = PROJECT_ROOT / "examples_ODMD"

if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))
