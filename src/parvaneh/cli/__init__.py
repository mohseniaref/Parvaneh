"""Command-line interface for :mod:`parvaneh`.

``parvaneh`` is the project's single entry point.  Its main subcommand unwraps a
wrapped phase image with a chosen algorithm::

    parvaneh unwrap wrapped.raw --shape 1024 1024 --method goldstein -o out.npy

Because unwrapping is what the command is for, the subcommand name may be left
out::

    parvaneh wrapped.raw --shape 1024 1024 --method goldstein -o out.npy

Three spellings reach the same :func:`main` function::

    parvaneh                       # the console script
    python -m parvaneh             # the package
    python -m parvaneh.cli         # this command package

``base`` holds the shared parser and logging helpers, ``unwrap`` the unwrapping
command itself, and ``command`` the dispatch between subcommands.
"""

from .command import main

__all__ = ["main"]
