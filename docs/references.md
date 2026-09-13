# References

Every formula and every algorithm in Parvaneh comes from a published source.
This page is the single list of those sources: books with their ISBNs, papers
with their DOIs, and the open-source projects that are compared against or used
as baselines.

The papers are grouped by what they are used for. Entries 5 to 19 are the
sources of the mathematics that is implemented here. Entries 20 to 29 are the
methods that are *not* implemented: they are listed because they are the
alternative answers to the same problem, and because
[`algorithms.md`](algorithms.md) and
[`phase_unwrapping_history_and_theory.md`](phase_unwrapping_history_and_theory.md)
compare against them. Where a claim could not be checked against the published
text, the entry says so rather than asserting it.

The purpose is reproducibility of *claims*. When
[`mathematics.md`](mathematics.md) derives an equation, the last line of the
section points here; when [`algorithms.md`](algorithms.md) describes a method,
it names the same source. You should be able to check any sentence in the
documentation against the original publication.

**Nothing from these sources is redistributed in this repository.** The code is
independently written, the worked examples were computed here, and the
explanations were written for this project. See
[`repository_scope.md`](repository_scope.md) for the full boundary.

---

## Books

1. **D. C. Ghiglia and M. D. Pritt**, *Two-Dimensional Phase Unwrapping: Theory,
   Algorithms, and Software*. Wiley-Interscience, New York, 1998.
   ISBN 978-0-471-24935-1.

   The standard monograph. Chapter 1 is the measurement model; chapters 2 and 3
   are least squares and the fast-transform solver; chapter 4 is quality maps,
   residues and reliability; chapters 5 and 6 are branch cuts, mask cuts and
   path following. It is the source for the classification of methods used
   throughout this documentation. *The book itself, and the C and MATLAB
   programs that accompany it, are not part of this repository and are not
   required to run anything here.*

2. **P. J. Huber and E. M. Ronchetti**, *Robust Statistics*, 2nd edition.
   Wiley, Hoboken, 2009. ISBN 978-0-470-12990-6.

   The general theory of M-estimators: why a squared loss is not robust, and
   how reweighting recovers robustness. This is the framework that the
   minimum-$L^p$ solver uses.

3. **G. B. Arfken, H. J. Weber and F. E. Harris**, *Mathematical Methods for
   Physicists*, 7th edition. Academic Press, Oxford, 2013.
   ISBN 978-0-12-384654-9.

   The Helmholtz decomposition, used to explain what the least-squares normal
   equations do: a vector field splits into a gradient part and a divergence-free
   part, and the solver keeps only the first.

4. **G. Strang**, *Computational Science and Engineering*. Wellesley-Cambridge
   Press, Wellesley, 2007. ISBN 978-0-9614088-1-7.

   The discrete cosine transform as the diagonaliser of the Neumann Laplacian,
   which is the reason the least-squares solver needs so few iterations. Any
   numerical-analysis text with a chapter on fast Poisson solvers covers the same
   result; this one is written at the right level for a first read.

---

## Papers

### The problem, and what can be recovered from it

5. **R. M. Goldstein, H. A. Zebker and C. L. Werner**, "Satellite radar
   interferometry: Two-dimensional phase unwrapping." *Radio Science* **23**
   (1988) 713–720. <https://doi.org/10.1029/RS023i004p00713>

   The paper that made unwrapping an established problem in radar
   interferometry, and that introduced residues and branch cuts.

6. **K. Itoh**, "Analysis of the phase unwrapping algorithm." *Applied Optics*
   **21** (1982) 2470. <https://doi.org/10.1364/AO.21.002470>

   The local consistency condition: if neighbouring phase differences are all
   smaller than $\pi$ in magnitude, summing them along a path recovers the phase
   regardless of which path you choose.

7. **B. R. Hunt**, "Matrix formulation of the reconstruction of phase values
   from phase differences." *Journal of the Optical Society of America* **69**
   (1979) 393–399. <https://doi.org/10.1364/JOSA.69.000393>

   Unwrapping written as a linear system in the phase differences, together with
   the observation that the additive constant lives in the null space.

### Least squares, fast transforms and robust norms

8. **D. C. Ghiglia and L. A. Romero**, "Robust two-dimensional weighted and
   unweighted phase unwrapping that uses fast transforms and iterative methods."
   *Journal of the Optical Society of America A* **11** (1994) 107–117.
   <https://doi.org/10.1364/JOSAA.11.000107>

   Weighted least squares, the Poisson equation with reflecting boundary
   conditions, and the DCT-preconditioned conjugate-gradient solver that
   `unwrap()` implements.

9. **D. C. Ghiglia and L. A. Romero**, "Minimum $L^p$-norm two-dimensional phase
   unwrapping." *Journal of the Optical Society of America A* **13** (1996)
   1999–2013. <https://doi.org/10.1364/JOSAA.13.001999>

   The minimum-$L^p$ objective and its solution by iteratively reweighted least
   squares, which is `unwrap_lp()`.

### Quality-guided and reliability sorting

10. **M. A. Herráez, D. R. Burton, M. J. Lalor and M. A. Gdeisat**, "Fast
    two-dimensional phase-unwrapping algorithm based on sorting by reliability
    following a noncontinuous path." *Applied Optics* **41**(35) (2002)
    7437–7444. <https://doi.org/10.1364/AO.41.007437>

    The *strategy* behind `reliability_unwrap()`: rate every pixel, sort the
    edges by that rating, and merge in that order so the result does not depend
    on any traversal. One honest difference is recorded here because it is easy
    to get wrong: Herráez and co-workers rate a pixel from the **second**
    differences in its $3\times3$ neighbourhood (four terms, one per direction
    and diagonal, summed as squares) and give an edge the **sum** of its two
    endpoints' values, whereas Parvaneh's rating uses **first** differences
    around the pixel, weighted by the confidence of each step, and normalises
    with $1/(1+S)$ so that larger is better. The recipe in
    [`mathematics.md`](mathematics.md) is therefore Parvaneh's own rating, built
    on their sorting scheme, not their printed formula.

11. **H. S. Abdul-Rahman, M. A. Gdeisat, D. R. Burton and M. J. Lalor**, "Fast
    three-dimensional phase-unwrapping algorithm based on sorting by reliability
    following a noncontinuous path." *Proceedings of SPIE* **5856** (2005)
    32–40. <https://doi.org/10.1117/12.611415>

    The three-dimensional best-path construction. This is the paper
    scikit-image's `unwrap_phase` cites for its three-dimensional case.

12. **H. S. Abdul-Rahman, M. A. Gdeisat, D. R. Burton, M. J. Lalor, F. Lilley
    and C. J. Moore**, "Fast and robust three-dimensional best path phase
    unwrapping algorithm." *Applied Optics* **46**(26) (2007) 6623–6635.
    <https://doi.org/10.1364/AO.46.006623>

    The journal version of the same idea, and the one the volume branch of
    `reliability_unwrap()` is benchmarked against.

13. **H. S. Abdul-Rahman, M. Arevalillo-Herráez, M. A. Gdeisat, D. R. Burton,
    M. J. Lalor, F. Lilley, C. J. Moore, D. B. Sheltraw and M. Qudeisat**,
    "Robust three-dimensional best-path phase-unwrapping algorithm that avoids
    singularity loops." *Applied Optics* **48**(23) (2009) 4582–4596.
    <https://doi.org/10.1364/AO.48.004582>

    The follow-up that handles the three-dimensional residue loops a
    best-path order can otherwise meet. Recorded because it is the paper that
    answers the most common objection to reliability sorting in 3-D; the
    remedy is not implemented here.

### Minimum discontinuity and discrete optimisation

14. **T. J. Flynn**, "Two-dimensional phase unwrapping with minimum weighted
    discontinuity." *Journal of the Optical Society of America A* **14**(10)
    (1997) 2692–2701. <https://doi.org/10.1364/JOSAA.14.002692>

    The minimum-discontinuity formulation, the incremental cost table and the
    sweep strategy used by `flynn_unwrap()`.

### Numerics and algorithms

15. **M. R. Hestenes and E. Stiefel**, "Methods of conjugate gradients for
    solving linear systems." *Journal of Research of the National Bureau of
    Standards* **49**(6) (1952) 409–436. <https://doi.org/10.6028/jres.049.044>

    The original conjugate-gradient method, used for the weighted systems where
    the DCT is no longer an exact inverse.

16. **G. Strang**, "The discrete cosine transform." *SIAM Review* **41**(1)
    (1999) 135–147. <https://doi.org/10.1137/S0036144598336745>

    The shortest citable statement of the fact the fast solver rests on: *each
    DCT basis contains the eigenvectors of a symmetric second-difference
    matrix*, and the boundary condition (reflecting, as here, or fixed) selects
    which of the four cosine transforms applies. Section 9 of
    [`mathematics.md`](mathematics.md) is this result.

17. **J. B. Kruskal**, "On the shortest spanning subtree of a graph and the
    traveling salesman problem." *Proceedings of the American Mathematical
    Society* **7**(1) (1956) 48–50.
    <https://doi.org/10.1090/S0002-9939-1956-0078686-7>

    The greedy maximum-spanning-forest algorithm that reliability sorting is an
    instance of.

18. **R. E. Tarjan**, "Efficiency of a good but not linear set union algorithm."
    *Journal of the ACM* **22**(2) (1975) 215–225.
    <https://doi.org/10.1145/321879.321884>

    Union–find with path compression and union by size, which is what makes the
    merge loop of the reliability sorter run essentially in linear time.

19. **W. H. Pritt and J. S. Shipman**, "Least-squares two-dimensional phase
    unwrapping using FFT's." *IEEE Transactions on Geoscience and Remote
    Sensing* **32**(3) (1994) 706–708. <https://doi.org/10.1109/36.297989>

    The same unweighted least-squares solution reached through an FFT rather
    than a DCT. It is the reason the solver here is described as
    "fast-transform" rather than "DCT-only": the two differ only in how the
    reflecting boundary condition is imposed.

### The methods compared against

20. **M. Costantini**, "A novel phase unwrapping method based on network
    programming." *IEEE Transactions on Geoscience and Remote Sensing* **36**(3)
    (1998) 813–821. <https://doi.org/10.1109/36.673674>

    Minimum-cost flow: the integer-optimisation formulation that SNAPHU and the
    3-D flow codes are instances of. Recorded as the reference point for the
    minimum-cost-flow work planned in the roadmap; not implemented here.

21. **M. Costantini and P. A. Rosen**, "A generalized phase unwrapping approach
    for sparse data." *IEEE International Geoscience and Remote Sensing
    Symposium (IGARSS)* **1** (1999) 267–269.
    <https://doi.org/10.1109/IGARSS.1999.773467>

    The flow formulation relaxed to sparse and irregular samples, which is what
    makes it relevant to a network of coherent points.

22. **C. W. Chen and H. A. Zebker**, "Network approaches to two-dimensional phase
    unwrapping: intractability and two new algorithms." *Journal of the Optical
    Society of America A* **17**(3) (2000) 401–414.
    <https://doi.org/10.1364/JOSAA.17.000401>

    Why the general minimum-discontinuity problem is NP-hard, and the two
    approximations that make it tractable in practice.

23. **C. W. Chen and H. A. Zebker**, "Two-dimensional phase unwrapping with use
    of statistical models for cost functions in nonlinear optimization."
    *Journal of the Optical Society of America A* **18**(2) (2001) 338–351.
    <https://doi.org/10.1364/JOSAA.18.000338>

    The statistical-cost formulation that the SNAPHU program implements, and the
    source of the cost functions a flow-based method needs.

24. **J. M. Bioucas-Dias and G. Valadão**, "Phase unwrapping via graph cuts."
    *IEEE Transactions on Image Processing* **16**(3) (2007) 698–709.
    <https://doi.org/10.1109/TIP.2006.888351>

    PUMA: the same minimum-$\ell^1$ discontinuity problem solved by maximum
    flow / minimum cut, which is the discrete-optimisation route rather than the
    iterative-reweighting route taken by `unwrap_lp()`.

25. **F. Liu and B. Pan**, "A new 3-D minimum cost flow phase unwrapping
    algorithm based on closure phase." *IEEE Transactions on Geoscience and
    Remote Sensing* **58**(3) (2020) 1857–1867.
    <https://doi.org/10.1109/TGRS.2019.2949926>

    The three-dimensional minimum-cost-flow method implemented by the `3D-MCF`
    project listed under *Software* below. It is the closest published relative
    of the volumetric flow solver on the roadmap.

26. **T. Dubois-Taine, S. Akiki and A. d'Aspremont**, "Iteratively reweighted
    least squares for phase unwrapping." *Optimization Methods and Software*
    **40**(6) (2025) 1368–1408.
    <https://doi.org/10.1080/10556788.2025.2522348>
    Preprint: <https://arxiv.org/abs/2401.09961>

    A modern analysis of the $L^1$ objective solved by IRLS with a
    preconditioned conjugate gradient, including how to reuse a flow method's
    weights. The same reweighting structure as `unwrap_lp()`, with the theory of
    why it converges.

27. **R. Cusack, J. M. Huntley and H. T. Goldrein**, "Improved noise-immune
    phase-unwrapping algorithm." *Applied Optics* **34**(5) (1995) 781–789.
    <https://doi.org/10.1364/AO.34.000781>

    The reliability-guided approach that precedes the sorted form used here, and
    the origin of the "flood the most reliable pixel first" idea.

28. **J. R. Buckland, J. M. Huntley and S. R. E. Turner**, "Unwrapping noisy
    phase maps by use of a minimum-cost-matching algorithm." *Applied Optics*
    **34**(23) (1995) 5100–5108. <https://doi.org/10.1364/AO.34.005100>

    The minimum-cost-matching approach to residue pairing, which is the
    principled alternative to Goldstein's greedy branch cuts.

29. **H. A. Zebker and Y. Lu**, "Phase unwrapping algorithms for radar
    interferometry: residue-cut, least-squares, and synthesis algorithms."
    *Journal of the Optical Society of America A* **15**(3) (1998) 586–598.
    <https://doi.org/10.1364/JOSAA.15.000586>

    The comparative study that separates the two families implemented here and
    shows when each wins; the basis for the "choosing a method" advice in
    [`algorithms.md`](algorithms.md).

---

## Software

These projects are used as comparisons, as baselines, or as structural
references. None of their code is copied into Parvaneh, and none of them is
required to install or run Parvaneh.

- **scikit-image** — `skimage.restoration.unwrap_phase`, the independent
  higher-dimensional baseline used in the notebooks and in the reliability
  tests. Its two-dimensional and three-dimensional reliability sorters follow
  Herráez *et al.* (2002) and Abdul-Rahman *et al.* (2005) respectively.
  Documentation:
  <https://scikit-image.org/docs/stable/api/skimage.restoration.html#skimage.restoration.unwrap_phase>.
  Worked example:
  <https://scikit-image.org/docs/stable/auto_examples/filters/plot_phase_unwrap.html>.
  Project paper: S. van der Walt *et al.*, "scikit-image: image processing in
  Python", *PeerJ* **2** (2014) e453.
  <https://doi.org/10.7717/peerj.453>.
  Parvaneh's higher-dimensional solver keeps a deterministic order, whereas the
  baseline's is randomised, so the two are compared on accuracy and not on
  bit-identity.

- **SNAPHU** — the reference implementation of the statistical-cost flow method
  of Chen and Zebker (2001). The program itself is distributed by Stanford; the
  packaging used for comparisons is
  <https://github.com/isce-framework/snaphu-py>, which contains the original C
  program as a submodule. Neither is copied here.

- **SPURT** — InSAR spatial–temporal unwrapping in ISCE, the reference point for
  the space–time formulation:
  <https://github.com/isce-framework/spurt>.
  The project supplies **software** citation metadata only; no journal article
  describing it was located, so it is cited as software and nothing more.

- **3D-MCF** — the MATLAB implementation of the three-dimensional
  minimum-cost-flow method of Liu and Pan (2020):
  <https://github.com/Fei0906/3D-MCF>.

- **kamui** — a Python implementation of the PUMA family of regularised
  unwrapping solvers, that is, of Bioucas-Dias and Valadão (2007):
  <https://github.com/yoyolicoris/kamui>.

- **IRLS reference implementation** — an independent implementation of
  iteratively reweighted least squares for phase unwrapping, used to check the
  robust path, and the code behind Dubois-Taine, Akiki and d'Aspremont (2025):
  <https://github.com/bpauld/PhaseUnwrapping>.

- **MintPy** — the `src/` + `cli/` package layout, the configuration-file
  style, and the separation of library from command line follow the conventions
  of this project:
  <https://github.com/insarlab/MintPy>.

---

## Which source is behind which function

| Function or option | Source |
| --- | --- |
| `unwrap` (unweighted) | Ghiglia & Romero (1994); Ghiglia & Pritt (1998), ch. 2; Pritt & Shipman (1994) for the FFT variant |
| `unwrap(..., weight=...)` | Ghiglia & Romero (1994) |
| `unwrap_lp` | Ghiglia & Romero (1996); Huber & Ronchetti (2009); Dubois-Taine, Akiki & d'Aspremont (2025) for the modern IRLS analysis |
| `phase_residues` | Goldstein, Zebker & Werner (1988) |
| `goldstein_unwrap` | Goldstein, Zebker & Werner (1988); Buckland, Huntley & Turner (1995) for the matching alternative |
| `mask_cut_unwrap` | Ghiglia & Pritt (1998), ch. 6 |
| `quality_guided_unwrap` | Ghiglia & Pritt (1998), ch. 4; Cusack, Huntley & Goldrein (1995) |
| `max_gradient_quality`, `pseudocorrelation_quality`, `derivative_variance_quality` | Ghiglia & Pritt (1998), ch. 4 |
| `pixel_reliability`, `reliability_unwrap` | sorting scheme from Herráez *et al.* (2002); three-dimensional construction from Abdul-Rahman *et al.* (2005, 2007); the rating itself is Parvaneh's own — see entry 10 |
| `flynn_unwrap` | Flynn (1997) |
| local consistency rule | Itoh (1982) |
| normal equations and null space | Hunt (1979) |
| DCT diagonalisation | Strang (1999); Strang (2007) |
| preconditioner | Ghiglia & Romero (1994) |
| conjugate gradients | Hestenes & Stiefel (1952) |
| maximum spanning forest | Kruskal (1956) |
| union–find | Tarjan (1975) |
| decomposition of the least-squares error | Arfken, Weber & Harris (2013) |
| choice between the method families | Zebker & Lu (1998) |
| not implemented: network-flow methods | Costantini (1998); Costantini & Rosen (1999); Chen & Zebker (2000, 2001) |
| not implemented: graph cuts, 3-D flow, space–time | Bioucas-Dias & Valadão (2007); Liu & Pan (2020); SPURT (software) |
| scikit-image baseline | van der Walt *et al.* (2014) |

---

## If you are writing about this project

Cite the original sources for the methods and this repository for the
implementation. The machine-readable version of the project citation is in
[`CITATION.cff`](../CITATION.cff); the archived release DOIs live in
[`.zenodo.json`](../.zenodo.json).

```text
M. Mohseni Aref, "Parvaneh: Accelerated phase unwrapping", version 0.1.0a1.
https://github.com/mohseniaref/Parvaneh
```
