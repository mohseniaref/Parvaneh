"""Support ``python -m parvaneh.cli``."""

import sys

from .command import main

if __name__ == "__main__":
    sys.exit(main())
