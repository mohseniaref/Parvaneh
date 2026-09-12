"""The top-level ``parvaneh`` command and its subcommand dispatch."""

import argparse
import sys

from .. import __version__
from . import unwrap
from .base import CommandParser

#: Subcommands understood by ``parvaneh``.  Unwrapping is currently the only one;
#: the dispatch in :func:`main` keeps adding another one a two-line change.
COMMANDS = ("unwrap",)


def build_parser():
    """Build the top-level ``parvaneh`` parser."""
    parser = CommandParser(
        prog="parvaneh",
        description="Parvaneh: accelerated and reproducible phase unwrapping.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  parvaneh unwrap wrapped.npy --method goldstein -o unwrapped.npy
  parvaneh wrapped.npy --method goldstein -o unwrapped.npy     # same thing
  parvaneh unwrap --method list                               # list algorithms

run "parvaneh unwrap --help" for the full set of unwrapping options
""")
    parser.add_argument("-V", "--version", action="store_true",
                        help="print the package version and exit")
    commands = parser.add_subparsers(dest="command", metavar="COMMAND")
    commands.add_parser("unwrap", parents=[unwrap.build_parser(add_help=False)],
                        help="unwrap a wrapped phase image (this is the default)")
    return parser


def main(argv=None):
    """Entry point for the ``parvaneh`` command and ``python -m parvaneh``."""
    tokens = list(sys.argv[1:] if argv is None else argv)

    # `parvaneh --help` and `parvaneh --version` describe the command as a whole.
    if not tokens or tokens[0] in ("-h", "--help", "-V", "--version"):
        top = build_parser()
        args = top.parse_args(tokens)          # -h prints help and exits here
        if args.version:
            print(__version__)
            return 0
        top.print_help()
        return 0

    # Everything else is the `unwrap` subcommand.  Its name may be left out,
    # because unwrapping is what this command exists for.
    if tokens[0] in COMMANDS:
        tokens = tokens[1:]

    return unwrap.run(unwrap.build_parser().parse_args(tokens))
