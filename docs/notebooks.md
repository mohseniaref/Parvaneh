# Notebooks

Four Jupyter notebooks ship with Parvaneh. They are the teaching and
reproducibility layer of the project: each one is fully self-contained, needs no
external data, and produces every figure it shows from code in the notebook
itself.

| Notebook | Level | Purpose |
|---|---|---|
| [`chapter_01_introduction.ipynb`](../notebooks/chapter_01_introduction.ipynb) | introductory | what wrapping is, why unwrapping is needed, and how the one-dimensional rule fails |
| [`chapter_02_line_integrals_residues.ipynb`](../notebooks/chapter_02_line_integrals_residues.ipynb) | intermediate | path independence, discrete curl, residues, and the irrotational/rotational decomposition |
| [`independent_synthetic_examples.ipynb`](../notebooks/independent_synthetic_examples.ipynb) | practical | every public algorithm family on one deterministic synthetic scene |
| [`synthetic_insar_generator.ipynb`](../notebooks/synthetic_insar_generator.ipynb) | advanced | building a complete synthetic InSAR scene with exact ground truth |

The two `chapter_*` notebooks follow the exposition of Ghiglia and Pritt
(1998); all code in them is original. The other two build the project's own
synthetic data, which is what makes quantitative statements about accuracy
possible at all.

## Running them

```bash
python -m pip install -e '.[numba]' jupyterlab matplotlib
jupyter lab notebooks/
```

The notebooks need `numpy`, `scipy`, `matplotlib` and `jupyterlab`; Numba and
Cython are optional and their absence only changes the speed comparisons.
Chapter 1 also compares against `skimage.restoration.unwrap_phase` when
scikit-image is installed and simply skips that comparison when it is not. All
random values come from seeded NumPy generators, so a fresh run reproduces the
same figures.

Each notebook starts with a short bootstrap cell that puts the repository's
`src/` directory on `sys.path`. That is what lets the notebooks import
`parvaneh` from a clone without installing anything. The form used by the
synthetic-example notebook is the shortest:

```python
from pathlib import Path
import sys

ROOT = Path.cwd().resolve().parent if Path.cwd().name == 'notebooks' else Path.cwd().resolve()
SRC = ROOT / 'src'
if (SRC / 'parvaneh').is_dir() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
```

It works when the notebook is run from the repository root and from the
`notebooks/` directory, and it is a harmless no-op after
`python -m pip install -e .`, because then `parvaneh` is already importable.
The other three notebooks use an equivalent spelling of the same two lines.

## Contents

### Chapter 1 — Introduction to phase unwrapping

Starts from complex-valued imaging: translation moves phase, not magnitude, and
a measured phase is only known modulo $2\pi$. It then shows quantized phase
measurements, interference fringes, and how the integer wrap count can be
recovered in the simplest case. The second half is about failure: aliasing, the
probability of a noise-induced cycle slip, and a parameter sweep that shows
where the naive one-dimensional rule breaks. It closes with a short look ahead
at two dimensions and a measured comparison of the NumPy, BLAS, Numba and
Cython paths on the machine that runs it.

### Chapter 2 — Line integrals, residues, and 2-D unwrapping

Treats the wrapped gradient as a differential form. Path independence (Itoh's
condition), the discrete curl, and the definition of a residue are developed in
that order, followed by the irrotational/rotational decomposition that explains
why residues are the only obstruction and how least squares handles them. The
closing sections are small experiments that let you *see* path dependence
rather than take it on faith.

### Independent synthetic examples

One deterministic experiment — a smooth ramp with two Gaussian bumps and
additive noise — run through every public algorithm family, followed by
reproducibility checks. This is the notebook to read if you want to know how
the families differ in practice, and the one to copy from if you want a
template for your own comparison.

### Synthetic InSAR generator

The most advanced notebook. It builds a synthetic interferogram from first
principles with an exact answer key: a ground grid, two analytic deformation
sources, projection onto the satellite line of sight, displacement-to-phase
conversion, coherence, a physically motivated phase-noise model, wrapping, the
integer ambiguity, residues, masks and no-data, and finally a self-consistency
validation section. Use it when you need a scene whose truth is known, for
example to test a new unwrapping method or to calibrate an error budget.

## House style

The notebooks are committed **without stored outputs** and with execution counts
cleared. A reader runs them, or does not; either way the diff of a change stays
readable and no stale figure or stale number is ever published. If you edit a
notebook, run it, then clear the outputs before committing:

```bash
jupyter nbconvert --clear-output --inplace notebooks/*.ipynb
```

Please keep the dependency set as it is. A notebook that only runs on the
author's machine is not reproducible.

## Related documentation

- [`algorithms.md`](algorithms.md) — the algorithms the synthetic notebooks exercise.
- [`cli.md`](cli.md) — the same algorithms from the command line.
- [`validation.md`](validation.md) — how the notebooks are checked automatically.
