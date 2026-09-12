#!/usr/bin/env python3
"""Build a Zenodo-safe source archive from an explicit allowlist."""

import argparse
import hashlib
import json
import zipfile
from pathlib import Path


ROOT_FILES = ("README.md", "LICENSE", "pyproject.toml", "setup.py",
              "conftest.py", "CITATION.cff", ".zenodo.json")
# ``src`` carries the import package, so the archive installs like the checkout.
TREES = ("src", "benchmarks", "tests", "tools")
DOCS = ("algorithms.md", "cli.md", "notebooks.md", "validation.md",
        "repository_scope.md", "binary_formats.md", "porting_status.md",
        "release_checklist.md", "performance.md",
        "phase_unwrapping_history_and_theory.md")
NOTEBOOKS = ("independent_synthetic_examples.ipynb",
             "chapter_01_introduction.ipynb",
             "chapter_02_line_integrals_residues.ipynb",
             "synthetic_insar_generator.ipynb")
EXCLUDED_SUFFIXES = (".so", ".pyc", ".c", ".pdf")


def allowed_files(root):
    for name in ROOT_FILES:
        path = root / name
        if path.is_file():
            yield path
    for tree in TREES:
        for path in sorted((root / tree).rglob("*")):
            if (path.is_file() and "__pycache__" not in path.parts
                    and path.suffix not in EXCLUDED_SUFFIXES):
                yield path
    for name in DOCS:
        path = root / "docs" / name
        if path.is_file():
            yield path
    for name in NOTEBOOKS:
        path = root / "notebooks" / name
        if path.is_file():
            yield path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", default="0.1.0a1")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output or root / "dist" / ("parvaneh-{}.zip".format(args.version))
    output.parent.mkdir(parents=True, exist_ok=True)
    prefix = "parvaneh-{}".format(args.version)
    manifest = []
    files = list(dict.fromkeys(allowed_files(root)))
    with zipfile.ZipFile(str(output), "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            relative = path.relative_to(root)
            data = path.read_bytes()
            archive.writestr(str(Path(prefix) / relative), data)
            manifest.append({"path": str(relative), "sha256": hashlib.sha256(data).hexdigest(),
                             "bytes": len(data)})
        payload = json.dumps({"schema_version": 1, "files": manifest}, indent=2).encode() + b"\n"
        archive.writestr(str(Path(prefix) / "ARCHIVE_MANIFEST.json"), payload)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    checksum = output.with_suffix(output.suffix + ".sha256")
    checksum.write_text("{}  {}\n".format(digest, output.name))
    print("wrote {} ({} files)".format(output, len(files) + 1))
    print("SHA-256 {}".format(digest))


if __name__ == "__main__":
    main()
