"""Shared plumbing for the Parvaneh command line.

Every subcommand reports failures the same way: an ``error: ...`` line through
:class:`SystemExit` rather than a traceback.  Progress chatter goes to stderr, so
that machine-readable output on stdout stays clean.
"""

import argparse
import sys


class CommandParser(argparse.ArgumentParser):
    """Parser that reports errors through :class:`SystemExit`."""

    def error(self, message):
        self.print_usage(sys.stderr)
        raise SystemExit("error: %s" % message)


def log(message):
    """Write a progress or diagnostic message to stderr."""
    print(message, file=sys.stderr)
