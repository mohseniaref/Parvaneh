#!/usr/bin/env python
"""Run the unwrapping command straight from a source checkout.

``pip install -e .`` gives you the installed ``parvaneh`` command.  This script
is for the other common case, a fresh ``git clone`` where nothing has been
installed yet::

    python unwrap.py wrapped.tif -o unwrapped.tif --method goldstein
    python unwrap.py --method list
    python unwrap.py --help

It only puts ``src/`` on the import path and then hands over to the very same
code the installed command runs, so the two never drift apart.
"""

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from parvaneh.cli import main

if __name__ == "__main__":
    sys.exit(main())
