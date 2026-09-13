# Notebooks

Twenty-two Jupyter notebooks ship with Parvaneh. They are the teaching and
reproducibility layer of the project: each one is fully self-contained, needs no
external data, and produces every figure it shows from code in the notebook
itself.

| Notebook | Level | Purpose |
|---|---|---|
| [`chapter_01_introduction.ipynb`](../notebooks/chapter_01_introduction.ipynb) | introductory | what wrapping is, why unwrapping is needed, and how the one-dimensional rule fails |
| [`chapter_02_line_integrals_residues.ipynb`](../notebooks/chapter_02_line_integrals_residues.ipynb) | intermediate | path independence, discrete curl, residues, and the irrotational/rotational decomposition |
| [`independent_synthetic_examples.ipynb`](../notebooks/independent_synthetic_examples.ipynb) | practical | every public algorithm family on one deterministic synthetic scene |
| [`three_dimensional_unwrapping.ipynb`](../notebooks/three_dimensional_unwrapping.ipynb) | advanced | stacks and volumes: does the third axis actually help, measured against per-slice unwrapping and an independent baseline |
| [`synthetic_insar_generator.ipynb`](../notebooks/synthetic_insar_generator.ipynb) | advanced | building a complete synthetic InSAR scene with exact ground truth |
| [`phase_unwrapping_fundamentals.ipynb`](../notebooks/phase_unwrapping_fundamentals.ipynb) | introductory | the $2\pi$ ambiguity, wrapping, local differences, and a first reconstruction |
| [`least_squares_and_weights.ipynb`](../notebooks/least_squares_and_weights.ipynb) | intermediate | the Poisson least-squares objective, weighting, and a noisy synthetic surface |
| [`reliability_and_flow_notes.ipynb`](../notebooks/reliability_and_flow_notes.ipynb) | intermediate | a four-pixel residue, tree decisions, integer flow corrections, dual-network diagrams, and measured reconstruction errors |
| [`competitor_methods.ipynb`](../notebooks/competitor_methods.ipynb) | advanced | additional algorithm comparisons |

The two `chapter_*` notebooks follow the exposition of Ghiglia and Pritt
(1998); all code in them is original. The other notebooks build the project's own
synthetic data, which is what makes quantitative statements about accuracy
possible at all.

## One worked notebook per method

Each lesson starts with a small calculation, draws the algorithm's decisions,
runs the public API, and changes one condition to measure its effect. Outputs
are saved from actual Jupyter execution. All explanations are in English.

| Method | Notebook | What you can see and calculate |
|---|---|---|
| Least squares, weighted and unweighted | [least_squares_and_weights](../notebooks/least_squares_and_weights.ipynb) | explicit four-pixel matrix, normal equations, squared endpoint weights |
| Quality-guided paths | [quality_guided_unwrapping](../notebooks/quality_guided_unwrapping.ipynb) | parent proposal, numbered traversal, priority comparison |
| Reliability sorting | [reliability_sorting](../notebooks/reliability_sorting.ipynb) | reliability arithmetic, edge sorting, four successive tree states |
| Goldstein branch cuts | [goldstein_branch_cuts](../notebooks/goldstein_branch_cuts.ipynb) | computed positive/negative residue pair, conceptual connection, actual cut-pixel mask |
| Quality-guided mask cuts | [quality_mask_cuts](../notebooks/quality_mask_cuts.ipynb) | mask growth and thinning, comparison on a clean ramp |
| Flynn minimum discontinuity | [flynn_minimum_discontinuity](../notebooks/flynn_minimum_discontinuity.ipynb) | region shift, integer jumps, quality and sweep counts |
| Minimum-cost flow | [minimum_cost_flow](../notebooks/minimum_cost_flow.ipynb) | four correction choices, dual graph, verified optimal single-cell cost |
| Robust minimum Lp | [robust_lp_unwrapping](../notebooks/robust_lp_unwrapping.ipynb) | penalty curves, IRLS weights, exponent sweep |
| Multigrid | [multigrid_unwrapping](../notebooks/multigrid_unwrapping.ipynb) | error scales, coarse grid, measured convergence versus one grid |
| Graph cuts | [graph_cut_unwrapping](../notebooks/graph_cut_unwrapping.ipynb) | four binary move costs, integer labels, a volume check |
| N-D cycle flow | [nd_cycle_flow](../notebooks/nd_cycle_flow.ipynb) | cube geometry, face counts, joint versus per-slice offsets |
| Statistical costs | [statistical_cost_unwrapping](../notebooks/statistical_cost_unwrapping.ipynb) | uncertainty curves, coherence, convex versus capped-model reports |
| Space-time priors | [space_time_priors](../notebooks/space_time_priors.ipynb) | temporal weights, circular mean, reference-pixel time series |
| Extended minimum-cost flow | [extended_minimum_cost_flow](../notebooks/extended_minimum_cost_flow.ipynb) | acquisition triangle, both input APIs, final closure measurement |

The combined [reliability and flow walkthrough](../notebooks/reliability_and_flow_notes.ipynb)
adds a shared noisy-scene comparison with correction maps and cycle-error counts.
Backend choices implement the same objectives; NumPy, BLAS, Numba, Cython and
CuPy are not counted as different unwrapping methods.

The source for the 14 method lessons is
[`tools/build_method_notebooks.py`](../tools/build_method_notebooks.py).
Regenerating them clears outputs, so execute them again before committing.

## Running them

```bash
python -m pip install -e '.[numba]' jupyterlab matplotlib
jupyter lab notebooks/
```

The notebooks need `numpy`, `scipy`, `matplotlib` and `jupyterlab`; Numba and
Cython are optional and their absence only changes the speed comparisons.
Chapter 1 and the 3-D notebook also compare against
`skimage.restoration.unwrap_phase` when scikit-image is installed and simply skip
that comparison when it is not.

Every random value the notebooks draw themselves comes from a seeded NumPy
generator, so a fresh run reproduces the same figures. One caveat applies to the
scikit-image comparison only: its higher-dimensional solver keeps a random
stream inside the compiled library that advances with each call, and the `seed`
argument does not redirect it. Repeated calls therefore agree to a few
hundredths of a radian rather than exactly, and the notebooks quote that
baseline to two decimals. Results from `parvaneh` are unaffected.

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
The other notebooks use an equivalent repository-path bootstrap.

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
additive noise — run through every public algorithm family, followed by a look
inside the flow solver's own report and a set of reproducibility checks. This is
the notebook to read if you want to know how the families differ in practice,
and the one to copy from if you want a template for your own comparison. The
flow section is also the shortest honest demonstration of how the solver's
bookkeeping works: it prints the residue charge, the network size, the number of
augmentations, and the imbalance the ground node had to absorb, and the reader
can check those four numbers against each other.

### Three-dimensional unwrapping

Answers one question with measurements instead of opinion: when a wrapped volume
is available, does solving it jointly beat unwrapping the slices one at a time?
It first defines the error measures it needs — a per-slice offset is a
legitimate freedom of the problem, so a plain RMSE would punish the honest
answer — then reproduces the classic stack failure in which every slice looks
perfect on its own and the volume does not, and finally sweeps the noise level
for least squares, reliability sorting and the scikit-image baseline. It also
measures what the joint solve costs, and it states plainly where a comparison is
only good to limited precision: the scikit-image baseline is not
bit-reproducible.

### Synthetic InSAR generator

The most advanced notebook. It builds a synthetic interferogram from first
principles with an exact answer key: a ground grid, two analytic deformation
sources, projection onto the satellite line of sight, displacement-to-phase
conversion, coherence, a physically motivated phase-noise model, wrapping, the
integer ambiguity, residues, masks and no-data, and finally a self-consistency
validation section. Use it when you need a scene whose truth is known, for
example to test a new unwrapping method or to calibrate an error budget.

## House style

The notebooks are committed **with their outputs and their execution counts**.
GitHub then renders every figure straight in the browser and a reader can check
a number without installing anything, which is most of the point of a teaching
notebook. Every stored output is therefore the result of a real end-to-end run;
never type into an output cell by hand.

When you change a notebook, run it end to end and commit the fresh outputs:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb
```

Commit only a run that finished without an error — a stored traceback is worse
than no output at all. The execution counts should read `1`, `2`, `3`, ... in
cell order, because a run that was restarted half way through leaves counts
like `10`, `11`, `12` that tell a reader nothing. Outputs are part of the
repository history, so keep figures and printed tables to what the surrounding
text actually discusses.

Please keep the dependency set as it is. A notebook that only runs on the
author's machine is not reproducible.

## Related documentation

- [`algorithms.md`](algorithms.md) — the algorithms the synthetic notebooks exercise.
- [`cli.md`](cli.md) — the same algorithms from the command line.
- [`performance.md`](performance.md) — what the notebooks' timing cells measure.
- [`validation.md`](validation.md) — how the notebooks are checked automatically.
