# Algorithms

Parvaneh implements the classical phase-unwrapping families described by Ghiglia
and Pritt (1998), plus reliability sorting in the spirit of Herráez and
co-workers (2002). Every algorithm is written from the published mathematics; no
historical C or MATLAB program is executed or loaded. Two of the solvers are
dimension-independent: least squares (`unwrap`) and reliability sorting
(`reliability_unwrap`) accept an array of any rank whose axes all have at least
two samples, so the same call unwraps a single image or a stack of them (see
[One image, a stack, or a cube](#one-image-a-stack-or-a-cube)). Exact
implementation and validation status is recorded in
[`porting_status.md`](porting_status.md).

## The problem in one paragraph

A wrapped phase image stores every value modulo $2\pi$:

$$\psi_{i,j} = \phi_{i,j} + 2\pi k_{i,j}, \qquad k_{i,j} \in \mathbb{Z},$$

where $\phi$ is the continuous surface we want and $\psi \in (-\pi, \pi]$ is what
we measured. Unwrapping means choosing the integers $k_{i,j}$ so that the result
is continuous. Locally this is easy — follow the phase and add $2\pi$ whenever
the jump between neighbours is too large — but "follow the phase" requires a
*path*, and in a noisy or masked image different paths can disagree. The
disagreement is concentrated at **residues**: $2\times2$ cells whose four wrapped
differences add up to a non-zero multiple of $2\pi$. Residues are the reason
unwrapping is a global problem and the reason so many algorithms exist.

## The two families

Unwrapping algorithms split into two broad families, and Parvaneh contains both.

**Global (minimum-norm) methods** treat the whole image as one optimisation
problem. They never integrate a path, so residues cannot derail them. Instead
they look for the surface whose *gradients* best match the measured wrapped
gradients, typically by solving a sparse linear system (least squares) or a
robust non-quadratic objective (minimum $L^p$). They are stable and smooth, and
they are the right default when you want an answer on every pixel without
thinking about topology.

**Path-following methods** build a spanning structure over the image — a
tree of reliable edges, a set of branch cuts, or a region-growing schedule —
and integrate the phase along it. They respect the integer nature of the
problem exactly and preserve sharp features better than a smoother, but they
need decisions about which pixels or edges to trust, and the decisions can go
wrong in low-coherence areas.

## One image, a stack, or a cube

The `phase` argument is just an array, and for the two dimension-independent
solvers it may have any number of axes. A 2-D array is one interferogram; a 3-D
array is the usual InSAR case, a stack of interferograms or a time series of the
same scene, with the third axis indexing the acquisitions.

The rule those solvers follow is *local and uniform*: every voxel is coupled to
its neighbours along **every** axis, exactly as the 2-D code couples a pixel to
its left/right and up/down neighbours. Nothing else changes — the wrapped
difference to a neighbour is still the measurement, and the objective is still to
make those differences consistent.

That small change fixes the single worst ambiguity of the 2-D problem. A constant
added to one slice is invisible *inside that slice*, so independent per-slice
unwrapping can return a stack whose slices are each correct and whose relative
levels are wrong by whole multiples of $2\pi$. Once the third axis is in play,
the difference between slice $k$ and slice $k+1$ is a measured quantity like any
other, and those offsets are pinned by the data. The illustration is the third
notebook, [`three_dimensional_unwrapping.ipynb`](../notebooks/three_dimensional_unwrapping.ipynb).

The details worth knowing:

- **Rank is a parameter, not a separate API.** `unwrap` and
  `reliability_unwrap` take whatever shape you give them — there is no
  `unwrap_3d` to learn.
- **The path-following and cycle-based families are 2-D.** `quality_guided_unwrap`,
  `goldstein_unwrap`, `mask_cut_unwrap`, `flynn_unwrap` and `unwrap_lp` require a
  2-D array and reject anything else rather than guessing. To use one of them as
  a stack baseline, call it once per slice in a loop; the third notebook does
  exactly that, and that loop is what "slice by slice" means in its tables.
- **Only one constant stays unobservable.** The global mean of a solution is
  arbitrary, exactly as in 2-D; the per-slice means are now determined. See
  [The two unobservable quantities](#the-two-unobservable-quantities).
- **`phase_residues` is 2-D.** Residues are defined on $2\times2$ cells of a
  slice and are still computed slice by slice.
- **Compiled kernels are 2-D.** The Numba stencil and the CuPy path apply to a
  2-D array. The reliability merge kernel is rank-independent, so it serves both.
  For any other rank the least-squares solvers fall back to the NumPy or BLAS
  path, which is still a single direct solve.
- **The CLI stays 2-D.** `parvaneh unwrap` reads rasters and arrays through a
  2-D input layer and refuses anything else, so a stack is unwrapped from Python
  (`unwrap(volume)`, `reliability_unwrap(volume)`). Raster formats would need a
  band or cube convention before a command-line stack makes sense.
- **Cost is linear in the number of neighbours.** Going from 2-D to 3-D adds one
  more difference per voxel, not a new problem: the measured cost of the joint
  solve is within a small factor of solving the slices separately.

## Implemented algorithms

| Family | Public API | CLI `--method` | Rank | Status |
|---|---|---|---|---|
| Unweighted least squares | `unwrap(phase)` | `ls` | any | Implemented and reference-tested |
| Weighted least squares | `unwrap(phase, weight)` | `ls` + `--weight` | any | Implemented and backend-tested |
| Quality-guided path following | `quality_guided_unwrap` | `quality-guided` | 2-D | Implemented and reference-tested |
| Reliability sorting (Herráez) | `reliability_unwrap` | `reliability` | any | Implemented and reference-tested |
| Residue detection | `phase_residues` | — (diagnostic) | 2-D | Implemented and unit-tested |
| Goldstein branch cuts | `goldstein_unwrap` | `goldstein` | 2-D | Implemented and reference-tested |
| Quality-guided mask cuts | `mask_cut_unwrap` | `mask-cut` | 2-D | Implemented and reference-tested |
| Flynn minimum discontinuity | `flynn_unwrap` | `flynn` | 2-D | Implemented and reference-tested |
| Minimum-$L^p$ norm | `unwrap_lp` | `lp` | 2-D | Implemented and synthetically tested |
| Multigrid families | — | — | — | Not yet ported |

"Rank: any" means any number of axes, each of length at least 2 (see
[One image, a stack, or a cube](#one-image-a-stack-or-a-cube)); "2-D" means the
function validates its input and raises for any other shape.

### Least squares — `unwrap`, CLI `ls`

Solves the Poisson equation implied by the wrapped gradients with a
matrix-free preconditioned conjugate-gradient iteration. The preconditioner is
a discrete cosine transform, which is why the solver needs almost no memory
beyond the image itself. A **weight raster** turns it into the weighted
least-squares (Ghiglia–Romero) formulation: the weight of an edge is the
minimum of the weights of its two endpoint pixels, so unreliable regions
influence their neighbours less. Weights are squared internally, and pixels
with zero weight are simply ignored. This is the default method and the safest
first choice for noisy data. It is also the method to reach for when the input
is a stack: `unwrap` treats the array's neighbours along every axis uniformly,
so passing a 3-D volume unwraps it in one solve instead of slicing it first —
see [One image, a stack, or a cube](#one-image-a-stack-or-a-cube).

### Quality-guided path following — `quality_guided_unwrap`, CLI `quality-guided`

Computes a per-pixel *quality* map, starts from the most reliable pixel, and
grows a region outward along the highest-quality available edge. Each new pixel
receives its neighbour's value plus the wrapped difference, so the integer
decision is made at a single trusted step and then frozen. It has no iteration
count and it is very fast on smooth data; it is less robust than least squares
when the quality map itself is misleading.

Four quality measures are available:

| `--quality` | Meaning |
|---|---|
| `min_gradient` | lower of the two absolute wrapped gradients at the pixel (edge priority) |
| `max_gradient` | maximum absolute wrapped gradient in a window |
| `pseudocorrelation` | coherence-like measure over a window |
| `derivative_variance` | variance of the wrapped derivative in a window |

`min_gradient` is the classic Goldstein–Zebker edge priority and is the
default. The other three take a `--window` (pixels) and default to a window of
3 when none is given. Larger quality values mean more reliable.

### Reliability sorting — `reliability_unwrap`, CLI `reliability`

The spanning-tree cousin of quality-guided path following, following Herráez,
Burton, Lalor and Gdeisat (2002). Where quality-guided grows a single region and
must commit to an order as it goes, reliability sorting makes one global sorted
list of *all* candidate edges between neighbours, walks down that list from the
most trustworthy edge to the least, and accepts an edge whenever it joins two
regions that are not yet connected. Accepted edges therefore form a
**maximum-reliability spanning forest**: one tree per connected region, exactly
`pixels - 1` accepted edges per region, no matter how many residues the data
contains. The loop is a union–find.

The reliability of a pixel is one number built from its immediate neighbours,

$$R_p = \frac{1}{1 + \sum_{q} \min(c_p, c_q)\,\lvert \Delta_{pq} \rvert},$$

where $\Delta_{pq}$ is the wrapped phase step to neighbour $q$ and $c$ is the
confidence (1 where nothing is supplied, the weight elsewhere, 0 inside a mask).
A pixel in smooth phase is surrounded by small steps and scores close to 1; a
pixel crossed by a steep fringe, or sitting in noise, accumulates a large sum
and scores low. Because the sum runs over neighbours **along every axis**, the
rating is defined in any number of dimensions, and because it uses only wrapped
steps it does not change if the input is wrapped again.

An edge is prioritised by $\min(R_a, R_b)$ — a chain is only as strong as its
weakest link — so masks and low-confidence samples are avoided automatically
instead of needing a special case. When a candidate edge has zero shared
confidence it is dropped; otherwise it is either accepted (joining two regions)
or reported as discarded, which is what happens to the edges that would have
closed a loop.

The bookkeeping is exact integer arithmetic, not floating point: each step
records how many whole turns were removed, and merging two regions transfers that
turn count along the tree. This is why the Python reference loop and the compiled
Numba kernel return bit-identical results.

**What the third axis buys.** A tree built only inside one slice has no way to
relate that slice to the next, so each slice is correct only up to its own
constant. A tree built through the volume includes edges between neighbouring
slices, and the wrapped inter-slice step is then an ordinary measurement that
fixes the relative levels. The third notebook measures exactly this effect.

`reliability_unwrap(phase, mask=None, weight=None, *, backend="python",
return_info=False)` accepts an array of any rank whose axes each have at least
two samples, and returns `NaN` where the input was excluded. `return_info=True`
also returns a `ReliabilityInfo` dataclass with `backend`, `pixels`, `edges`
(usable candidate edges), `merges` (accepted), `discarded` (rejected) and
`components`; a single connected region gives `components == 1` and
`merges == pixels - 1`. `backend="numba"` is the fast path and is what the CLI
uses when Numba is installed; `backend="python"` is the readable reference
implementation and is only practical on small arrays.

### Residues — `phase_residues`

Not an unwrapper: it is the diagnostic that explains why the others differ. It
returns the integer charge of every $2\times2$ cell (normally $-1$, $0$ or
$+1$), computed as the discrete circulation of the wrapped gradients. A cell
with a non-zero charge is a topological obstruction; every valid unwrapping
path must either avoid it or cut through it. `np.count_nonzero(phase_residues(phase))`
is a quick noise indicator for a scene. Cells touching a masked-out pixel are
assigned zero charge, because a residue that depends on an unknown observation
is not meaningful.

### Goldstein branch cuts — `goldstein_unwrap`, CLI `goldstein`

Detects residues, connects nearby opposite charges with branch cuts, and
extends cuts to the image border so they do not form closed loops; the phase is
then integrated over the image with the cuts acting as barriers. The
`--max-cut-length` option caps how far the expanding search for a balancing
charge is allowed to grow, which bounds the runtime. If the residue density is
high, cuts can become long and the unwrapped surface develops visible seams.

### Quality-guided mask cuts — `mask_cut_unwrap`, CLI `mask-cut`

Instead of balancing residues individually, this method builds the cut set from
the quality map: it repeatedly places cuts along the least reliable regions so
that no residue loop survives, then integrates. It usually produces smoother
results than Goldstein cuts on InSAR-like data at the cost of a more expensive
search.

### Flynn minimum discontinuity — `flynn_unwrap`, CLI `flynn`

Looks for the unwrapped surface with the fewest discontinuities, i.e. it
minimises the count of $2\pi$ jumps instead of penalising their magnitude
quadratically. Regions are grown and then merged while a tree of "jump" edges
is improved until no change reduces the number of discontinuities. This is the
best choice for images containing genuine $2\pi$ cliffs (faults, steep
topography) because it does not smear them the way a quadratic cost does. It is
also the slowest of the six methods here, so pass `--quality min_gradient` and
expect seconds rather than milliseconds on large rasters.

### Minimum $L^p$ norm — `unwrap_lp`, CLI `lp`

Iteratively reweighted least squares on the same Poisson structure, with a
robust $L^p$ norm on the gradient misfit: each outer iteration re-estimates
edge weights from the current residual, and each inner iteration runs the
least-squares solver. `p = 2` reproduces ordinary least squares; smaller values
(bounded below by 1) behave more like the minimum-discontinuity ideal while
remaining a convex, smoothly solvable problem. The default `p = 1.2` is a good
compromise; `--epsilon` controls the softening that keeps the reweighting
finite near zero residual.

## Choosing a method

| Situation | Suggested method |
|---|---|
| First attempt, unknown data | `ls` |
| A stack or time series (3-D) | `ls` or `reliability` on the whole volume, never slice by slice |
| Known reliability map (coherence, amplitude) | `ls --weight coherence.npy` |
| Very large raster, smooth phase, speed matters | `quality-guided` |
| Moderate noise, visible seams are unacceptable | `lp` |
| Real $2\pi$ discontinuities that must not be smoothed | `flynn` |
| Wrapped phase with many residues, one tree wanted | `reliability` |
| Dense residues, low coherence | `ls` with a mask, then compare with `mask-cut` |
| Want to know whether the scene is even unwrappable | `phase_residues` count |

No single method is best on every scene, which is why `--method list` and the
notebooks exist: run two or three families on a reduced crop and compare.

## Masks, weights, and what they mean

A **mask** says which pixels hold a valid measurement (non-zero means valid).
Masked pixels are excluded from the answer and, for the cycle-based methods,
from the cut search. A **weight** is the smoother, continuous version of the
same information and is only meaningful for least-squares-style methods
(`ls`, and internally `lp`). Use a mask when pixels are definitely invalid
(shadow, water, layover) and a weight when they are merely unreliable.

Neither can create information that is not in the data. A mask that is too
aggressive disconnects the image into islands; residues then have nowhere to
go and the cuts get longer.

## Backends

`available_backends()` reports which accelerators were actually found on the
machine (the CLI prints the same table in `--method list`):

| Backend | Requirement | Notes |
|---|---|---|
| `numpy` | NumPy + SciPy | always available, reference behaviour |
| `blas` | SciPy's BLAS wrapper | dot products through an explicit BLAS `ddot` |
| `numba` | `numba` installed | JIT-compiled kernels; usually the fastest CPU choice |
| `cython` | extension built locally | optional compiled kernel |
| `cupy` | CuPy **and** a working CUDA device | reported available only if a device is present |

All CPU backends solve the same equations and are tested to agree; a backend
change must not change the mathematics, only the runtime. `--backend auto`
picks the fastest available backend that supports the chosen method, and
methods without a compiled kernel ignore the request (with a note on stderr).

Backend coverage is not uniform:

- The least-squares and $L^p$ families accept every backend in the table. The
  compiled `numba` and `cython` kernels and the `cupy` path are 2-D
  implementations; a higher-rank array runs on the remaining CPU backend.
- `quality-guided` and `reliability` choose between `numba` and `python`, their
  own kernels rather than a general linear-algebra backend. Reliability's Numba
  kernel is rank-independent and is bit-identical to its Python reference; for a
  large volume the compiled kernel is roughly an order of magnitude faster,
  which is why `auto` selects it when available.
- `goldstein`, `mask-cut` and `flynn` are fixed implementations and accept no
  backend choice at all.

## The two unobservable quantities

Two things can never be recovered from a wrapped image, and both are handled
explicitly:

1. **A constant offset.** Adding any real constant to a solution gives another
   equally valid solution. `unwrap` therefore returns a zero-mean surface,
   while the cycle-based methods return `input + π` modulo $2\pi$ because their
   integer bookkeeping starts at `((phase + π) / 2π) % 1`. Never compare two
   unwrapped images pixel-wise without removing this offset; use
   `rmse_aligned`, or the CLI's `--center circular` (the default), which pins
   the result back onto the wrapped input so different methods are directly
   comparable.
2. **The integer field itself.** Unwrapping recovers $\phi$ up to a constant,
   not the individual $k_{i,j}$, and no algorithm can tell a real $2\pi$ cliff
   from a wrapping event. That ambiguity is what the algorithm families
   disagree about, and it is why validation uses synthetic scenes with a known
   truth.

## References

- D. C. Ghiglia and M. D. Pritt, *Two-Dimensional Phase Unwrapping: Theory,
  Algorithms, and Software*. Wiley, 1998. — the mathematical source of every
  family implemented here except reliability sorting.
- M. A. Herráez, D. R. Burton, M. J. Lalor and M. A. Gdeisat, "Fast
  two-dimensional phase-unwrapping algorithm based on sorting by reliability
  following a noncontinuous path", *Applied Optics* **41**(35), 7437–7444, 2002.
  — pixel reliability and the sorted spanning tree behind `reliability_unwrap`.
- Historical and theoretical background:
  [`phase_unwrapping_history_and_theory.md`](phase_unwrapping_history_and_theory.md).
