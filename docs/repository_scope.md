# Repository scope and copyright boundaries

This page states exactly what the public Parvaneh repository does and does not
contain, and what the license covers. It exists so that a reader, a reviewer, or
a release manager can check the boundary without reading the whole project —
and because a phase-unwrapping package sits close to a body of work that is
**not** freely redistributable.

## What the repository contains

Everything committed here is written for this project:

- the Python source of the `parvaneh` package, including its command line;
- documentation in [`docs/`](.);
- the notebooks in [`notebooks/`](../notebooks), which run on synthetic data;
- the test suite in [`tests/`](../tests) and the scripts in
  [`benchmarks/`](../benchmarks);
- release tooling in [`tools/`](../tools); and
- packaging and metadata files (`pyproject.toml`, `CITATION.cff`,
  `.zenodo.json`, `LICENSE`).

What "independent" means here, concretely: no line of the source is copied or
translated from another program, every docstring and document is original text,
and every figure is produced by the code in this repository from a seeded
synthetic input. Mathematical provenance — which paper or textbook describes an
algorithm — is cited in
[`porting_status.md`](porting_status.md),
[`algorithms.md`](algorithms.md),
[`mathematics.md`](mathematics.md) and
[`phase_unwrapping_history_and_theory.md`](phase_unwrapping_history_and_theory.md),
and collected with DOIs and ISBNs in [`references.md`](references.md),
because citing an algorithm is not the same as redistributing an implementation
of it.

## What the repository does not contain

The following are deliberately excluded, and are also listed in
[`.gitignore`](../.gitignore):

- the *Two-Dimensional Phase Unwrapping: Theory, Algorithms, and Software*
  textbook PDF;
- the book's original C and MATLAB programs, in any form, including partial
  excerpts;
- the supplied phase, surface, correlation, and mask rasters that accompany
  that material;
- third-party ZIP archives and document files;
- locally compiled binaries and generated C extension sources, PDF exports,
  benchmark and profile output under `reports/`, and build logs;
- historical-data notebooks, which live only in a local working copy; and
- internal planning documents, roadmap and decision records under
  `docs/_local/`.

Because the development machine may also hold copies of reference material
outside version control, the check that matters is what a fresh clone contains:
`git ls-files` on a clean checkout is the authoritative list.

## Why the boundary is drawn here

Phase unwrapping has a canonical reference implementation that is distributed
with a commercial textbook. That implementation is not open source, and its data
files are not redistributable. Rather than depend on it, Parvaneh:

1. implements each algorithm from its published description;
2. validates each implementation against a **mathematical property** or a
   synthetic scene with a known answer, never against a stored third-party
   output — see [`validation.md`](validation.md); and
3. cites the literature for the mathematics while shipping only original code.

This is also why the notebooks do not embed third-party figures or require an
external executable, and why the benchmark inputs are generated in memory.

## What the license covers

The [BSD 3-Clause License](../LICENSE) applies to the files distributed as part
of Parvaneh — everything tracked in this repository. It does **not** grant any
right to separately obtained third-party material, and it does not relicense
the algorithms' published descriptions or the cited works.

Practically:

| You may | Condition |
|---|---|
| use, modify, and redistribute the code, including commercially | keep the copyright notice and the license text |
| cite the software in academic work | cite the archived Zenodo release, see [`CITATION.cff`](../CITATION.cff) |
| reuse the documentation | same license, same attribution |

| You may not | Because |
|---|---|
| assume this repository contains the textbook or its programs | it does not |
| treat the citations as a redistribution license | they are references, not permissions |

## Verifying the boundary

```bash
git ls-files                    # every file in a clean checkout
git ls-files '*.pdf' '*.zip'    # should be empty
git ls-files docs               # docs must contain only original pages
```

The release archive is assembled from an explicit allowlist rather than from
"whatever is in the working tree", so a local, untracked file cannot leak into a
published release:

```bash
python tools/build_zenodo_archive.py
```

That script names the trees, documentation pages, notebooks, and root files that
are permitted in the archive. Anything else in the working tree is skipped
rather than packaged, and `.pdf`, `.zip`-adjacent document material, compiled
objects, and C sources are excluded by suffix even inside an allowed tree. The
archive carries an `ARCHIVE_MANIFEST.json` with a SHA-256 and a byte count per
file, and a `.sha256` file beside the ZIP.

If you believe a file in this repository is in the wrong place, please open an
issue — a provenance problem is treated as a release blocker, see
[`release_checklist.md`](release_checklist.md).
