# History and Theory of Two-Dimensional Phase Unwrapping

This chapter summarizes the historical development and mathematical foundations
of two-dimensional phase unwrapping, with particular emphasis on the framework
presented by Dennis C. Ghiglia and Mark D. Pritt in *Two-Dimensional Phase
Unwrapping: Theory, Algorithms, and Software*. The equations are rederived in a
consistent notation; this document is not a transcription of the book.

## 1. Historical development

Phase unwrapping developed through optical interferometry, Fourier-phase
reconstruction, adaptive optics, radar interferometry, and related imaging
problems.

### 1.1 Matrix and least-squares reconstruction (1979)

Hunt formulated phase reconstruction from measured phase differences as a
matrix inverse problem. This established the relationship between gradient
integration, least-squares estimation, and numerical linear algebra
([Hunt, 1979](https://doi.org/10.1364/JOSA.69.000393)).

### 1.2 Itoh's local sampling condition (1982)

Itoh identified a sufficient condition for exact sequential phase unwrapping.
If every true phase difference satisfies

$$
\left|\phi_{n+1}-\phi_n\right|<\pi,
$$

the true difference can be recovered from the wrapped difference
([Itoh, 1982](https://doi.org/10.1364/AO.21.002470)).

### 1.3 Residues and branch cuts for InSAR (1988)

Goldstein, Zebker, and Werner introduced an influential two-dimensional
branch-cut method for radar interferometry. They demonstrated that local phase
inconsistencies, called residues, can propagate into large regional errors
unless the permitted integration paths are constrained
([Goldstein et al., 1988](https://doi.org/10.1029/RS023i004p00713)).

### 1.4 Fast and weighted least squares (1994)

Ghiglia and Romero developed fast-transform methods for unweighted
least-squares unwrapping and iterative methods for the weighted problem.
Spatial weights allow unreliable or inconsistent measurements to exert less
influence on the reconstructed phase surface
([Ghiglia and Romero, 1994](https://doi.org/10.1364/JOSAA.11.000107)).

### 1.5 Minimum-$L^p$ reconstruction (1996)

Ghiglia and Romero generalized the least-squares objective to minimum-$L^p$
estimation. For $p<2$, isolated large gradient errors have less influence than
they have under a quadratic objective. This work also established a conceptual
connection between minimum-norm and branch-cut approaches
([Ghiglia and Romero, 1996](https://doi.org/10.1364/JOSAA.13.001999)).

### 1.6 Minimum weighted discontinuity (1997)

Flynn formulated unwrapping as the minimization of weighted phase
discontinuities. When discontinuities cannot be eliminated, the weighting
encourages them to occur in low-quality regions
([Flynn, 1997](https://doi.org/10.1364/JOSAA.14.002692)).

### 1.7 The Ghiglia--Pritt synthesis (1998)

Ghiglia and Pritt consolidated these ideas into a common theoretical and
computational framework. Their book presents the theory, eight representative
algorithms, C implementations, datasets, and comparative evaluation. Its
organization progresses from fundamental phase and line-integral theory to
quality maps, path-following methods, minimum-norm methods, and systematic
comparison ([Wiley, 1998](https://www.wiley-vch.de/en/areas-interest/engineering/two-dimensional-phase-unwrapping-978-0-471-24935-1)).

## 2. The wrapped-phase inverse problem

A complex interferometric measurement can be represented as

$$
z(\mathbf{x})=A(\mathbf{x})\exp\!\left[i\phi(\mathbf{x})\right].
$$

The sensor records only the principal argument

$$
\psi(\mathbf{x})=\arg z(\mathbf{x})
=\mathcal W\!\left\{\phi(\mathbf{x})\right\}
\in[-\pi,\pi),
$$

where a convenient definition of the wrapping operator is

$$
\mathcal W(x)
=x-2\pi\left\lfloor\frac{x+\pi}{2\pi}\right\rfloor.
$$

Consequently, the desired continuous phase satisfies

$$
\boxed{\phi_{i,j}=\psi_{i,j}+2\pi k_{i,j}},
\qquad k_{i,j}\in\mathbb Z.
$$

Phase unwrapping is therefore the estimation of an integer cycle field
$k_{i,j}$, rather than simply the removal of visible colour discontinuities.
The data also possess a global gauge ambiguity: $\phi$ and $\phi+2\pi c$, with
$c\in\mathbb Z$, produce identical wrapped observations. A reference pixel,
stable region, or external datum is required to determine the absolute phase
level.

For an interferometric SAR measurement, the unwrapped phase is generally a
combination of several physical contributions:

$$
\psi=\mathcal W\!\left\{
\phi_{\mathrm{flat}}+\phi_{\mathrm{topo}}+\phi_{\mathrm{def}}
+\phi_{\mathrm{atm}}+\phi_{\mathrm{orb}}+\phi_{\mathrm{ion}}
+\phi_{\mathrm{noise}}
\right\}.
$$

Unwrapping estimates the missing integer cycles; it does not by itself
separate deformation, atmosphere, orbit, ionosphere, or residual topography.

## 3. Itoh's condition and wrapped gradients

Define horizontal and vertical wrapped edge measurements by

$$
b^x_{i,j}
=\mathcal W(\psi_{i,j+1}-\psi_{i,j}),
$$

$$
b^y_{i,j}
=\mathcal W(\psi_{i+1,j}-\psi_{i,j}).
$$

Using $D$ for the discrete gradient operator and stacking all measured edges
into $\mathbf b$, the ideal relation is

$$
\mathbf b=D\phi.
$$

This relation is recovered exactly when every true neighbour increment obeys
the Itoh condition

$$
\boxed{\left|(D\phi)_e\right|<\pi\quad\text{for every edge }e.}
$$

Under this condition, path integration is exact up to an additive constant.
In practice, the condition can be violated by

- insufficient spatial sampling;
- decorrelation and phase noise;
- radar layover and shadow;
- masked or missing observations; and
- genuine phase discontinuities.

Once such a violation occurs, a local integer-cycle error can propagate through
all subsequently integrated pixels.

## 4. Residues and discrete topology

For a $2\times2$ pixel cell, calculate the wrapped circulation around its four
directed edges:

$$
\begin{aligned}
s_{i,j}={}&
\mathcal W(\psi_{i,j+1}-\psi_{i,j})
+\mathcal W(\psi_{i+1,j+1}-\psi_{i,j+1})\\
&+\mathcal W(\psi_{i+1,j}-\psi_{i+1,j+1})
+\mathcal W(\psi_{i,j}-\psi_{i+1,j}).
\end{aligned}
$$

The corresponding residue charge is

$$
\boxed{
r_{i,j}=\operatorname{round}\!\left(\frac{s_{i,j}}{2\pi}\right)
}.
$$

For ordinary wrapped data, $r_{i,j}$ normally belongs to
$\{-1,0,+1\}$. In matrix notation,

$$
\mathbf r=\frac{1}{2\pi}C\mathbf b,
$$

where $C$ is the discrete curl operator. Because

$$
CD=0,
$$

the gradient of a valid scalar phase surface has zero discrete curl. A nonzero
residue therefore identifies a local obstruction to path-independent
integration; it is not merely a visually noisy pixel.

For two paths $P_1$ and $P_2$ having the same endpoints,

$$
\sum_{e\in P_1}b_e-\sum_{e\in P_2}b_e
=2\pi\sum_{c\in\Omega(P_1-P_2)}r_c.
$$

Thus, if the closed contour formed by the two paths encloses nonzero total
charge, their integrated phases disagree by one or more complete cycles.

## 5. Quality maps, masks, and filtering

The book treats data reliability as an essential part of the inverse problem.
A quality map $q_{i,j}\in[0,1]$ may be derived from interferometric coherence,
phase-derivative variance, maximum phase gradient, pseudocorrelation, or other
application-specific information.

Pixel reliability can be converted to conservative edge weights through

$$
w^x_{i,j}=\min(q_{i,j},q_{i,j+1}),
\qquad
w^y_{i,j}=\min(q_{i,j},q_{i+1,j}).
$$

An edge receives zero weight when either endpoint is excluded by the validity
mask. In InSAR, low-coherence water, dense vegetation, layover, and radar
shadow should generally not be treated as ordinary high-quality noisy pixels.

Because phase is circular, wrapped phase should not be smoothed with an
ordinary arithmetic mean across the $-\pi/\pi$ boundary. Circular filtering
uses

$$
\widetilde\psi
=\arg\!\left(h*e^{i\psi}\right)
=\operatorname{atan2}\!\left(h*\sin\psi,h*\cos\psi\right),
$$

where $h$ is a normalized spatial kernel. Filtering can reduce phase variance
and residue count, but it can also blur real discontinuities.

## 6. Path-following methods

All local path-following algorithms use the propagation equation

$$
\widehat\phi_q
=\widehat\phi_p+\mathcal W(\psi_q-\psi_p),
$$

where pixel $p$ has already been unwrapped and $q$ is an adjacent pixel. The
algorithms differ in the order in which edges are traversed and in the edges
that are prohibited.

### 6.1 Goldstein branch cuts

Goldstein's method connects positive and negative residues into
charge-balanced clusters, or connects remaining charge to a boundary. The
phase is then flood-filled without crossing the cuts. The cuts ensure that an
allowed integration loop does not enclose unbalanced charge.

Branch cuts are computational constraints rather than automatically detected
physical faults. Regions disconnected by cuts or masks can retain independent
integer-cycle offsets.

### 6.2 Quality-guided integration

Quality-guided methods begin in a reliable region and use a priority rule to
select the next edge. Their purpose is to delay unreliable measurements so
that errors are less likely to contaminate good regions. They are efficient,
but an early incorrect decision can still create a large regional cycle error.

### 6.3 Minimum weighted discontinuity

Minimum-discontinuity methods seek an integer-cycle configuration that places
unavoidable discontinuities along low-cost or low-quality edges. Flynn's
algorithm repeatedly changes a connected region by $2\pi$ when that operation
reduces the weighted discontinuity objective.

## 7. Minimum-norm methods

Minimum-norm methods do not select a single integration path. They estimate a
phase surface globally by minimizing disagreement between its gradient and the
measured wrapped-gradient field:

$$
\boxed{
\widehat\phi
=\underset{\phi}{\operatorname{arg\,min}}
\sum_{e\in E}w_e\left|(D\phi)_e-b_e\right|^p
}.
$$

### 7.1 Weighted least squares

For $p=2$,

$$
\widehat\phi
=\underset{\phi}{\operatorname{arg\,min}}
\left\|W^{1/2}(D\phi-\mathbf b)\right\|_2^2.
$$

Differentiating the objective produces the normal equation

$$
\boxed{
D^{\mathsf T}WD\,\widehat\phi
=D^{\mathsf T}W\mathbf b
}.
$$

For uniform weights, the continuous analogue is a Poisson equation:

$$
\nabla^2\widehat\phi=\nabla\cdot\mathbf b.
$$

Depending on the boundary conditions and weights, the discrete system can be
solved by FFT or DCT methods, multigrid, conjugate gradient, or preconditioned
conjugate gradient. All four appear in this project as solvers or as the thing a
solver is measured against; the multigrid one is derived in section 18 of
[`mathematics.md`](mathematics.md), and the transfer operators it rests on are
entries 6 to 8 of [`references.md`](references.md).

Least squares produces a globally consistent smooth surface, but an isolated
large gradient inconsistency can be distributed across a broad region. This is
the characteristic smoothing behavior of a quadratic penalty.

### 7.2 Robust minimum-$L^p$ reconstruction

When $p<2$, large residuals are penalized less aggressively than in ordinary
least squares. Iteratively reweighted least squares can approximate the
solution using residual-dependent weights:

$$
w_e^{(t)}=
\left(
\left|(D\phi^{(t)})_e-b_e\right|^2+\epsilon^2
\right)^{p/2-1}.
$$

Each iteration solves a weighted least-squares problem. Large inconsistencies
receive smaller weights, allowing their influence to remain spatially
localized. For very small $p$, the result approaches the minimum-discontinuity
idea: reproduce as many observed gradients as possible and concentrate errors
on a relatively small set of edges.

## 8. Fundamental limitations

Phase congruence with the wrapped observation is necessary but not sufficient
for physical correctness. A congruence diagnostic is

$$
\epsilon_{\mathrm{cong}}
=\left[
\frac{1}{N}\sum_p
\mathcal W(\widehat\phi_p-\psi_p)^2
\right]^{1/2}.
$$

A solution may have $\epsilon_{\mathrm{cong}}\approx0$ while one connected
region is displaced by $2\pi$ relative to another. When a reference phase is
available, gradient accuracy can be assessed with

$$
\epsilon_{\nabla}
=\left[
\frac{1}{|E|}\sum_{e\in E}
\left((D\widehat\phi)_e-(D\phi_{\mathrm{ref}})_e\right)^2
\right]^{1/2}.
$$

Reliable validation should therefore consider

- phase congruence;
- phase and gradient errors against synthetic or reference data;
- residue density and closure errors;
- independent connected-component offsets;
- coherence and mask boundaries; and
- agreement with known stable areas or external elevation data.

No two-dimensional algorithm can uniquely reconstruct cycle counts that were
destroyed by severe undersampling or decorrelation. Coherence, spatial
smoothness, masks, temporal networks, external elevation models, and physical
deformation constraints provide the prior information needed to resolve such
ambiguities.

## 9. Application in this project

The theory is represented by independent notebooks that generate every input
in memory:

- [`chapter_01_introduction.ipynb`](../notebooks/chapter_01_introduction.ipynb):
  principal phase, wrapping, integer ambiguity, Itoh's condition, and synthetic
  phase surfaces.
- [`chapter_02_line_integrals_residues.ipynb`](../notebooks/chapter_02_line_integrals_residues.ipynb):
  line integrals, path dependence, discrete curl, and residue charge.
- [`independent_synthetic_examples.ipynb`](../notebooks/independent_synthetic_examples.ipynb):
  quality maps, residues, Goldstein cuts, quality guidance, weighted $L^2$,
  robust minimum-$L^p$, Flynn reconstruction, timing, and error comparison.

The implementations use the theory and algorithmic structure described by
Ghiglia and Pritt, while the Python code, synthetic scenes, numerical
experiments in this project are newly constructed.

## 10. Summary

Phase unwrapping is not merely the removal of visible $2\pi$ jumps. It is the
reconstruction of an integrable phase field from modulo-$2\pi$, noisy, and
potentially topologically inconsistent gradient measurements. Path-following
methods control where information is integrated, while minimum-norm methods
project the inconsistent measured gradients toward a globally integrable
field. The quality map, validity mask, objective function, and physical prior
determine how each method treats ambiguity and error.

## References

1. B. R. Hunt, “Matrix formulation of the reconstruction of phase values from
   phase differences,” *Journal of the Optical Society of America*, 69,
   393–399, 1979. <https://doi.org/10.1364/JOSA.69.000393>
2. K. Itoh, “Analysis of the phase unwrapping algorithm,” *Applied Optics*,
   21, 2470, 1982. <https://doi.org/10.1364/AO.21.002470>
3. R. M. Goldstein, H. A. Zebker, and C. L. Werner, “Satellite radar
   interferometry: Two-dimensional phase unwrapping,” *Radio Science*, 23,
   713–720, 1988. <https://doi.org/10.1029/RS023i004p00713>
4. D. C. Ghiglia and L. A. Romero, “Robust two-dimensional weighted and
   unweighted phase unwrapping that uses fast transforms and iterative
   methods,” *Journal of the Optical Society of America A*, 11, 107–117,
   1994. <https://doi.org/10.1364/JOSAA.11.000107>
5. D. C. Ghiglia and L. A. Romero, “Minimum $L^p$-norm two-dimensional phase
   unwrapping,” *Journal of the Optical Society of America A*, 13, 1999–2013,
   1996. <https://doi.org/10.1364/JOSAA.13.001999>
6. T. J. Flynn, “Two-dimensional phase unwrapping with minimum weighted
   discontinuity,” *Journal of the Optical Society of America A*, 14,
   2692–2701, 1997. <https://doi.org/10.1364/JOSAA.14.002692>
7. D. C. Ghiglia and M. D. Pritt, *Two-Dimensional Phase Unwrapping: Theory,
   Algorithms, and Software*. New York: Wiley, 1998. ISBN 978-0-471-24935-1.

The complete bibliography of the project — including the numerical-methods and
algorithms sources, the software compared against, and a table mapping each
function to the publication it comes from — is in
[`references.md`](references.md). The step-by-step derivations are in
[`mathematics.md`](mathematics.md), and the implementation notes are in
[`algorithms.md`](algorithms.md).
