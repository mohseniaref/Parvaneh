"""Make the ``src`` layout importable when the package is not installed.

The import package lives in ``src/parvaneh``, so a bare ``git clone`` does not
put it on ``sys.path``.  CI installs the distribution first and does not need
this file; a local ``pytest -q`` does.
"""

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"

if (SRC / "parvaneh").is_dir() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
