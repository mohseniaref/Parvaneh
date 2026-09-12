# Algorithms

Parvaneh implements the classical two-dimensional phase-unwrapping families
described by Ghiglia and Pritt (1998). Every algorithm is written from the
published mathematics; no historical C or MATLAB program is executed or loaded.
Exact implementation and validation status is recorded in
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

## Implemented algorithms

| Family | Public API | CLI `--method` | Status |
|---|---|---|---|
| Unweighted least squares | `unwrap(phase)` | `ls` | Implemented and reference-tested |
| Weighted least squares | `unwrap(phase, weight)` | `ls` + `--weight` | Implemented and backend-tested |
| Quality-guided path following | `quality_guided_unwrap` | `quality-guided` | Implemented and reference-tested |
| Residue detection | `phase_residues` | — (diagnostic) | Implemented and unit-tested |
| Goldstein branch cuts | `goldstein_unwrap` | `goldstein` | Implemented and reference-tested |
| Quality-guided mask cuts | `mask_cut_unwrap` | `mask-cut` | Implemented and reference-tested |
| Flynn minimum discontinuity | `flynn_unwrap` | `flynn` | Implemented and reference-tested |
| Minimum-$L^p$ norm | `unwrap_lp` | `lp` | Implemented and synthetically tested |
| Multigrid families | — | — | Not yet ported |

### Least squares — `unwrap`, CLI `ls`

Solves the Poisson equation implied by the wrapped gradients with a
matrix-free preconditioned conjugate-gradient iteration. The preconditioner is
a discrete cosine transform, which is why the solver needs almost no memory
beyond the image itself. A **weight raster** turns it into the weighted
least-squares (Ghiglia–Romero) formulation: the weight of an edge is the
minimum of the weights of its two endpoint pixels, so unreliable regions
influence their neighbours less. Weights are squared internally, and pixels
with zero weight are simply ignored. This is the default method and the safest
first choice for noisy data.

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
| Known reliability map (coherence, amplitude) | `ls --weight coherence.npy` |
| Very large raster, smooth phase, speed matters | `quality-guided` |
| Moderate noise, visible seams are unacceptable | `lp` |
| Real $2\pi$ discontinuities that must not be smoothed | `flynn` |
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
  family implemented here.
- Historical and theoretical background:
  [`phase_unwrapping_history_and_theory.md`](phase_unwrapping_history_and_theory.md).
