# The mathematics of phase unwrapping

This page derives every equation that Parvaneh implements. It is written for a
reader who has had one course in linear algebra (vectors, matrices, matrix
multiplication, eigenvalues) and knows what a derivative is; no previous
exposure to interferometry or to unwrapping is assumed. Each result ends with a
line saying **where the mathematics comes from**, so you can check it against
the published source. Full bibliographic details, with DOIs for the papers and
ISBNs for the books, are collected in [`references.md`](references.md).

Nothing in this repository is copied from those sources. The published papers
and books describe the *mathematics*; the Python code, the worked numbers, and
the wording on this page were written for this project.

## Contents

1. [The measurement model](#1-the-measurement-model)
2. [Two things that can never be determined](#2-two-things-that-can-never-be-determined)
3. [The local rule and Itoh's condition](#3-the-local-rule-and-itohs-condition)
4. [Residues: when no consistent answer exists](#4-residues-when-no-consistent-answer-exists)
5. [Least squares: from residuals to the Poisson equation](#5-least-squares-from-residuals-to-the-poisson-equation)
6. [Solving the Poisson equation: DCT and conjugate gradients](#6-solving-the-poisson-equation-dct-and-conjugate-gradients)
7. [Weights, masks, and what they mean](#7-weights-masks-and-what-they-mean)
8. [Robust minimum $L^p$ norm and iteratively reweighted least squares](#8-robust-minimum-lp-norm-and-iteratively-reweighted-least-squares)
9. [Quality maps](#9-quality-maps)
10. [Reliability sorting: a maximum spanning forest](#10-reliability-sorting-a-maximum-spanning-forest)
11. [Path following and branch cuts](#11-path-following-and-branch-cuts)
12. [Minimum discontinuity](#12-minimum-discontinuity)
13. [Minimum-cost flow: unwrapping as a network](#13-minimum-cost-flow-unwrapping-as-a-network)
14. [Any number of dimensions](#14-any-number-of-dimensions)
15. [What each method guarantees](#15-what-each-method-guarantees)
16. [One worked experiment with all ten variants](#16-one-worked-experiment-with-all-ten-variants)
17. [When the noise is strong: a discriminating experiment](#17-when-the-noise-is-strong-a-discriminating-experiment)
18. [Multigrid: one equation on many grids](#18-multigrid-one-equation-on-many-grids)

---

## 1. The measurement model

A phase-measuring instrument — a radar interferometer, a fringe-projection
scanner, an MRI phase map — cannot see the absolute phase. It sees the phase
*wrapped* into one turn of a circle:

$$\psi = W(\phi), \qquad W(x) = ((x + \pi) \bmod 2\pi) - \pi .$$

The true phase $\phi$ is a real number at every pixel and the measured phase
$\psi$ lies in a half-open interval of width $2\pi$: the code in
[`core.py`](../src/parvaneh/core.py) uses $[-\pi,\pi)$, i.e. an exact $\pm\pi$
comes back as $-\pi$. The choice of which endpoint belongs to the interval only
matters when a phase difference is *exactly* half a cycle, which is a
measure-zero case; the reliability sorter in
[`reliability.py`](../src/parvaneh/reliability.py) uses the other convention
$(-\pi,\pi]$ for exactly this reason, and both conventions give the same answer
on all data that is not adversarially constructed.

Equivalently, there is an unknown **integer field** $k_{i,j}\in\mathbb{Z}$ with

$$\psi_{i,j} = \phi_{i,j} + 2\pi k_{i,j}
\quad\Longleftrightarrow\quad
\phi_{i,j} = \psi_{i,j} - 2\pi k_{i,j}.$$

Unwrapping means choosing the integers. Locally that is trivial: walk along the
image and add $2\pi$ whenever the jump between neighbours looks too large. The
whole difficulty is that "walk along the image" needs a *path*, and in a noisy
or masked image different paths disagree. Parvaneh's solvers differ in how they
resolve that disagreement.

Numbers used below are for

$$\psi = \begin{bmatrix} 0 & \pi/2 \\ -\pi/2 & \pi \end{bmatrix},$$

a $2\times 2$ picture of the polar angle around its centre. It is the smallest
array that shows the central difficulty of the subject, and every claim made
about it here can be checked by hand.

**Where this comes from.** The wrapping operator and the "integer field"
viewpoint are standard; see Ghiglia & Pritt (1998), chapter 1, and the survey
introduction in Goldstein, Zebker & Werner (1988).

## 2. Two things that can never be determined

**(a) The additive constant.** Every equation in this page uses only *phase
differences* between neighbours. If $u$ is an answer, then $u + c$ for any
constant $c$ is an equally good answer: it has the same gradients, and
$W(u+c) = W(u)$ only if $c$ is a multiple of $2\pi$, but the residuals below are
completely unchanged by any constant. This is why `unwrap` returns the
zero-mean solution — it subtracts the mean of the result before returning it —
and why the constant part of any unwrapped result is a *convention* of the
solver, not a measurement.

**(b) A whole turn over a disconnected region.** If a mask, a cut, or a residue
separates one part of the image from another, that part may be shifted by
$2\pi$ without changing a single measured difference. Two answers that differ by
$2\pi$ on an island are equally valid. This is the deeper version of (a):
the constant is unobservable *per connected region*, not once for the whole
image.

The practical consequence is a procedure for judging any unwrapping result.
Compare two things, never just one:

1. **Wrap preservation**, $W(u)$ against the measurement $\psi$. This is
   convention-free: it tests whether the answer still carries the measured
   phase (the topology) or has smoothed it away.
2. **Error after removing the unobservable offsets**, either one constant for
   the whole array or one constant per island.

The project ships the helper used by its tests for the simple case:

```python
from parvaneh import rmse_aligned          # from parvaneh.synthetic.legacy
error = rmse_aligned(result, truth)        # removes the mean difference first
```

It subtracts the *mean* difference, which removes a global constant but
deliberately does **not** remove per-island offsets. Chapter 16 shows the two
numbers side by side for all ten variants. A result with a perfect
wrap-preservation number and a large aligned error is not "wrong": it is a
different branch of the same measurement.

**Where this comes from.** The unobservability of the additive constant is the
null space of the least-squares system (Hunt, 1979); the per-region version is
the reason branch-cut methods must connect unbalanced residues to the border
(Goldstein, Zebker & Werner, 1988). Both are discussed in Ghiglia & Pritt
(1998).

## 3. The local rule and Itoh's condition

Everything starts from one quantity: the **wrapped difference** across an edge
between two neighbouring pixels $a$ and $b$,

$$g_{ab} = W(\psi_b - \psi_a).$$

The hope is that $g_{ab}$ is the true difference $\Delta\phi_{ab} = \phi_b -
\phi_a$. Write $\Delta\phi_{ab} = g_{ab} + 2\pi q_{ab}$ with $q_{ab}\in\mathbb{Z}$.
Then $g_{ab} = \Delta\phi_{ab} - 2\pi q_{ab}$ and $g_{ab}\in[-\pi,\pi)$ force
$q_{ab} = 0$ exactly when $|\Delta\phi_{ab}| < \pi$. That is **Itoh's condition**:

$$|\phi_b - \phi_a| < \pi \quad\Longrightarrow\quad g_{ab} = \phi_b - \phi_a .$$

When the condition holds on every edge of a path, summing the wrapped
differences along that path telescopes and returns the true difference between
the two endpoints. Nothing more is needed: with no noise and no residues, a
single pass of cumulative sums solves the problem.

*Example (sampling is good).* With $\phi = (0,\ 2.4,\ 4.8,\ 7.2,\ 9.6)$ the
steps are $2.4 < \pi$, the wrapped signal is

$$\psi = (0,\ 2.4,\ -1.4832,\ 0.9168,\ -2.9664),$$

its wrapped differences are $(2.4, 2.4, 2.4, 2.4)$, and the running sum returns
$\phi$ exactly.

*Example (sampling is violated).* With steps of $3.5 > \pi$,

$$\phi = (0,\ 3.5,\ 7.0) \;\longrightarrow\; \psi = (0,\ -2.7832,\ 0.7168),
\qquad g = (-2.7832,\ -2.7832),$$

because $3.5 - 2\pi = -2.7832$. Here the information is genuinely gone: every
method in this repository returns a field whose steps are $-2.7832$, i.e. the
answer differs from the truth by one turn *per step*. This is not a defect of an
algorithm, and no algorithm can repair it — the measurement never contained the
distinction. It is why the sampling condition is checked, quoted, and reported
alongside results. Note the two views of the same output:

$$\underbrace{(0,\ -2.7832,\ -5.5664)}_{\text{cumulative sum} = \phi + 2\pi k}
\qquad\text{and}\qquad
\underbrace{(2.7832,\ 0,\ -2.7832)}_{\text{the same field, zero mean}} .$$

They differ by a constant $2.7832$; least squares (`unwrap`) returns the second,
a cumulative-sum method returns the first, and both are the same branch of the
same measurement.

**Where this comes from.** Itoh (1982) states the condition and the telescoping
argument; Hunt (1979) gives the matrix view taken up in section 5. The
distinction between "aliased but consistent" and "inconsistent" is treated in
Ghiglia & Pritt (1998).

## 4. Residues: when no consistent answer exists

Itoh's condition can hold on every edge and still give contradictory answers
along different paths. The obstruction is local and can be detected. Take the
four pixels of a $2\times2$ cell and add the four wrapped differences
**counter-clockwise**:

$$c = g_{\text{right}} + g_{\text{up}} - g_{\text{left}} - g_{\text{down}},$$

or, in indices (pixel $(i,j)$ with $i$ down and $j$ right),

$$c_{i,j} = W(\psi_{i,j+1}-\psi_{i,j})
 + W(\psi_{i+1,j+1}-\psi_{i,j+1})
 - W(\psi_{i+1,j+1}-\psi_{i+1,j})
 - W(\psi_{i+1,j}-\psi_{i,j}).$$

Then $c_{i,j}$ is always a multiple of $2\pi$, and the integer

$$r_{i,j} = \operatorname{round}\!\left(\frac{c_{i,j}}{2\pi}\right)
\;\in\;\{-1, 0, +1\}$$

is the **residue charge** of that cell. If $u$ were a function with no
discontinuities, going around the cell would return to where it started, so
$c = 0$. A non-zero charge therefore means: *there is no single-valued field $u$
whose wrapped differences equal the measured ones.* This is the discrete form
of the statement that the measured gradient field is not a gradient — its curl
does not vanish — so integration along paths must depend on the path.

*Example.* For the $2\times2$ array of section 1 the four wrapped differences
are $-1.5708$ and $+1.5708$ along each axis, the circulation is $2\pi$, and the
charge is $+1$:

$$\psi = \begin{bmatrix} 0 & \pi/2 \\ -\pi/2 & \pi \end{bmatrix},
\qquad r = \begin{bmatrix} 1 \end{bmatrix}.$$

Roll the array one column and the charge becomes $-1$. A charge of $+1$ or $-1$
is the generic case in interferometry; larger magnitudes appear when several
residues sit inside one cell, and they are normal in coarsely sampled data.

Three facts are worth remembering, because two of them are commonly misstated.

* **Residues are observed, not hidden.** They can be computed from $\psi$ with
  no other input — see `phase_residues` and
  [`path_following.py`](../src/parvaneh/path_following.py).
* **Total charge is not zero.** It is the circulation of the *outer boundary* of
  the array divided by $2\pi$, and wrapping is not additive along a path, so
  this is generally non-zero (measured totals of $-3$ and $-1$ on two random
  test arrays). Algorithms that pair up charges therefore have to send the
  unpaired ones to the border.
* **A residue is a property of the samples, not of the physics.** A genuine
  discontinuity of the surface and an under-sampled smooth surface produce the
  same charge. Unwrapping cannot distinguish them; a mask or a weight, from
  another data channel (coherence, intensity), is the only way to express which
  one you believe.

**Where this comes from.** Residues and the discrete circulation test were
introduced into practice by Goldstein, Zebker & Werner (1988); the index
formula and the "total charge equals boundary circulation" caveat are in
Ghiglia & Pritt (1998), chapter 4.

## 5. Least squares: from residuals to the Poisson equation

The first family of solvers never integrates along a path. It treats the whole
array as one optimisation problem: find the field $u$ whose *gradients* best
match the measured wrapped gradients.

Label the edges of the grid $e = (a,b)$, and let $\Delta u_e = u_b - u_a$ and
$g_e = g_{ab}$. The objective is

$$J(u) = \sum_e \left(\Delta u_e - g_e\right)^2 ,$$

the sum running over all neighbouring pairs in all axes. Setting
$\partial J/\partial u = 0$ gives the **normal equations**. Writing $D$ for the
matrix that turns a field into its edge differences, the same statement is

$$D^{\mathsf T} D\, u = D^{\mathsf T} g .$$

The left-hand side is the discrete Laplacian and the right-hand side is the
divergence of the measured gradients, so the normal equations are exactly the
discrete **Poisson equation**

$$\Delta u = \rho, \qquad \rho = \operatorname{div} g .$$

Parvaneh builds both sides edge by edge. It stores, for each axis, the field

$$f^{(a)}_p = \left(\text{weighted wrapped difference across the edge leaving
pixel } p \text{ along axis } a\right),$$

and computes the divergence by the backward-difference rule

$$\rho_p = \sum_{a} \left( f^{(a)}_p - f^{(a)}_{p - \hat a} \right),
\qquad f = 0 \text{ outside the array},$$

where $p - \hat a$ is the neighbour of $p$ in the negative direction of axis
$a$. The same operator is applied to a field when the solver needs
$\operatorname{div}\operatorname{grad} u$, so the identity

$$\operatorname{div}\operatorname{grad} = -D^{\mathsf T} D$$

holds exactly, including at the image border, where the rule for $f = 0$
outside is the **reflecting (Neumann)** boundary condition: at index $0$ the
Laplacian reads $u_1 - u_0$ and at the far end $u_{N-2} - u_{N-1}$, which is
what mirrors $u$ about the border would give.

Two consequences matter for the reader.

* **Compatibility.** Summing $\rho$ over every pixel telescopes to only
  boundary terms and gives $\sum_p \rho_p = 0$: the measured divergences always
  cancel. This is the discrete counterpart of $D^{\mathsf T}\rho$ being
  orthogonal to the constant vector, and it is what makes the linear system
  solvable at all. Its null space is that same constant, which is property 2(a)
  of section 2 seen from the solver's side.
* **Rotation is thrown away.** A vector field splits into a gradient part and a
  rotational part (Helmholtz decomposition). The normal equations keep only the
  gradient part: the rotational part of $g$ is annihilated by
  $\operatorname{div}$. A pure vortex therefore leaves no trace in $\rho$. For
  the $2\times2$ example of section 4 the right-hand side is *exactly zero*:

  $$\rho = \begin{bmatrix} 0 & 0 \\ 0 & 0 \end{bmatrix}
  \qquad\Longrightarrow\qquad
  u = \begin{bmatrix} 0 & 0 \\ 0 & 0 \end{bmatrix},$$

  so least squares returns a flat field for an array whose true phase is a
  ramp of more than a radian. That is not a bug: the residue has made the
  concept "the" phase undefined for this measurement, and least squares
  answers the well-posed question "which smooth field has these divergences?"
  rather than the ill-posed one "which field produced these samples?".

Least squares is also *not* wrap-preserving: it is a smoothing estimator, so
$W(u)$ is only approximately $\psi$. Because they fit first differences, these
methods cannot see a residue and cannot leave a $2\pi$ inconsistency behind;
their error is smoothly spread over the image instead. For a real scene this
tends to look good and to be wrong in exactly the places that matter (steep
slopes, layover, cut masks).

**Where this comes from.** Hunt (1979) is the origin of the matrix formulation
of unwrapping as a least-squares problem; the weighted version together with
the fast-transform solver used here is Ghiglia & Romero (1994); the textbook
treatment, including the discrepancy between the least-squares solution and the
"true" phase in the presence of residues, is Ghiglia & Pritt (1998), chapters 2
and 3. The Helmholtz reading of the normal equations is standard vector
calculus (see, e.g., Arfken, Weber & Harris, 2013, "Helmholtz's theorem").

## 6. Solving the Poisson equation: DCT and conjugate gradients

The system $\Delta u = \rho$ is sparse but large, and it must be solved without
forming any matrix. The trick is that the eigenvectors of the discrete
reflecting Laplacian are cosines — a fact you can check by hand.

Work in one dimension on $N$ pixels $i = 0,\dots,N-1$ with the reflecting rule
$u_{-1} = u_0$, $u_N = u_{N-1}$, and try the candidate eigenvector

$$v_i = \cos\!\left(\theta\left(i + \tfrac12\right)\right),
\qquad \theta = \frac{\pi k}{N},\quad k = 0,\dots,N-1 .$$

The reflection conditions are satisfied because
$v_{-1} = \cos(-\theta/2) = \cos(\theta/2) = v_0$ and, since
$\theta N = \pi k$,

$$v_N = \cos\!\left(\theta N + \tfrac{\theta}{2}\right)
      = (-1)^k \cos\!\left(\tfrac{\theta}{2}\right) = v_{N-1}.$$

The trigonometric identity $\cos(\alpha-\beta) + \cos(\alpha+\beta) =
2\cos\alpha\cos\beta$ with $\alpha = \theta(i+\tfrac12)$ and $\beta = \theta$
gives $v_{i-1} + v_{i+1} = 2\cos\theta\, v_i$, hence

$$(\Delta v)_i = v_{i-1} - 2v_i + v_{i+1} = 2\left(\cos\theta - 1\right) v_i .$$

So $\lambda_k = 2(\cos(\pi k/N) - 1)$ is the eigenvalue of mode $k$, the set
$\{v^{(k)}\}$ is a complete orthogonal basis, and in $d$ dimensions the
eigenvalue of mode $k = (k_1,\dots,k_d)$ is the sum of the per-axis values:

$$\lambda_k = 2\sum_{a=1}^{d}\left(\cos\frac{\pi k_a}{N_a} - 1\right).$$

The basis is exactly the type-II discrete cosine transform, computed by
`scipy.fft.dctn` (and `scipy.fft.idctn`). Because a transform and its inverse
share the same normalisation constant, that constant cancels when we use

$$M^{-1} = \text{idct}\left(\frac{1}{\lambda_k}\,\text{dct}(\,\cdot\,)\right)
\qquad\text{with } \lambda_0 := 1$$

as a *preconditioner* for the conjugate-gradient method: the zero eigenvalue of
the constant mode is replaced by $1$ so the operator can be inverted, and the
missing mean of the solution is recovered afterwards by the zero-mean
convention of section 2. This is what `_poisson_scale` in
[`core.py`](../src/parvaneh/core.py) computes.

Conjugate gradients (CG) is an iterative solver for symmetric positive-definite
systems: it builds a sequence of search directions $p^{(n)}$ that are conjugate
with respect to the operator, and the update

$$\alpha^{(n)} = \frac{r^{(n)}\cdot z^{(n)}}{p^{(n)}\cdot (Q p^{(n)})},
\qquad
u^{(n+1)} = u^{(n)} + \alpha^{(n)} p^{(n)},
\qquad
r^{(n+1)} = r^{(n)} - \alpha^{(n)} Q p^{(n)}$$

with $Q = -\operatorname{div}\operatorname{grad}$ and
$z = M^{-1} r$. The iteration stops when the relative residual
$\lVert r\rVert/\lVert r^{(1)}\rVert$ falls below `tol` (default $10^{-8}$) or
after `max_iter` steps (default $100$).

For the **unweighted** problem the preconditioner is not just a preconditioner:
it is the exact inverse of the operator, because the DCT diagonalises it. One
iteration is then enough, and that is what the code does — measured, on random
data of every rank:

| array | weights | CG iterations | relative residual |
| --- | --- | --- | --- |
| $24\times31$ | none | 1 | $5\times10^{-16}$ |
| $24\times31$ | $0.3$ over a $4\times6$ patch | 11 | $1\times10^{-9}$ |
| $12\times14\times6$ | none | 1 | $5\times10^{-16}$ |
| $12\times14\times6$ | $0.3$ over a $4\times6\times6$ block | 12 | $4\times10^{-9}$ |

The weighted case is slower *by construction*: weights make the operator
position-dependent, the cosine modes are no longer its eigenvectors, and the
transform only approximates the inverse. More contrast in the weights means
more iterations. This is the price of using weights at all, and it is why the
solvers report their iteration count and residual.

**Where this comes from.** The cosine diagonalisation of the discrete Neumann
Laplacian is a classical result of numerical analysis. The shortest citation
that states it cleanly is Strang (1999), "The discrete cosine transform", whose
point is that each DCT basis contains the eigenvectors of a symmetric
second-difference matrix; the textbook treatment, boundary conditions included,
is Strang (2007), and it is the basis of the fast Poisson solvers described
there. Its use for unwrapping, with the DCT and the same reflecting boundary
conditions, is Ghiglia & Romero (1994); the same unweighted solution reached
through an FFT instead of a DCT is Pritt & Shipman (1994). Conjugate gradients
is Hestenes & Stiefel (1952). The preconditioned form and
the convergence test follow the standard presentation in any numerical linear
algebra text.

## 7. Weights, masks, and what they mean

Real data come with a reliability measure per pixel: InSAR coherence, intensity,
a mask of layover and shadow, or the output of a water-detection step. Parvaneh
takes a per-pixel `weight` $w_p \ge 0$ and `mask` (true = valid), and turns them
into an **edge weight**

$$c_{ab} = \min\!\left(w_a^2,\ w_b^2\right), \qquad w_p = 0 \text{ where masked,}$$

so that the objective becomes

$$J(u) = \sum_e c_e \left(\Delta u_e - g_e\right)^2 .$$

Three choices deserve a word each.

* **Why the square?** $w$ is meant as an inverse standard deviation, so
  $w^2 = 1/\sigma^2$ is the inverse variance, and $c_e(\Delta u_e - g_e)^2$ has
  the units of a squared z-score. Minimising the sum is then maximum-likelihood
  estimation for Gaussian errors of unequal size, and the very same number is a
  sensible cost function even when the Gaussian story is only approximate.
* **Why the minimum at an edge?** An edge is trustworthy only if *both*
  endpoints are, so the conservative choice is the smaller of the two. (A
  product $w_a w_b$ or the mean are equally defensible; the minimum is what the
  reference formulation uses.)
* **What a zero weight does.** With $c_e = 0$ the edge contributes nothing to
  the objective, so the two sides are decoupled. Masked pixels drop out of the
  solve and come back as `NaN`, and the graph of remaining edges may split into
  several connected components. Each component then has its own unobservable
  constant — property 2(b). Path-following and reliability methods have the
  same property and for the same reason.

Weights appear in both least-squares objectives (`unwrap` and `unwrap_lp`) and
in the reliability rating of section 10, where a low-weight pixel is a cheap
place to put a $2\pi$ step.

**Where this comes from.** Weighted least squares with fast transforms and its
statistical reading are Ghiglia & Romero (1994); the role of coherence and mask
information is described in Ghiglia & Pritt (1998) and in the InSAR literature
surveyed in [`phase_unwrapping_history_and_theory.md`](phase_unwrapping_history_and_theory.md).

## 8. Robust minimum $L^p$ norm and iteratively reweighted least squares

Squaring residuals is what makes least squares easy, and it is also its
weakness: one badly wrong edge pulls the whole solution. A robust alternative
is to minimise a *non-quadratic* norm of the residuals. Ghiglia and Romero's
minimum-$L^p$ formulation, with $1 \le p \le 2$, replaces the objective by

$$J_p(u) = \sum_e \left( r_e^2 + \varepsilon^2 \right)^{p/2},
\qquad r_e = \Delta u_e - g_e ,$$

where $\varepsilon > 0$ (default $10^{-3}$ in Parvaneh) keeps the objective
differentiable at $r_e = 0$. The gradient with respect to the field is

$$\frac{\partial J_p}{\partial u}
 = -\sum_e \underbrace{p\left(r_e^2+\varepsilon^2\right)^{p/2-1}}_{=:\,c_e(r)}
  r_e\,\nabla r_e ,$$

so setting it to zero gives a *weighted* least-squares problem in which the
edge weights depend on the current residual. That observation is the algorithm:
solve a weighted least-squares problem, recompute the weights from the new
residuals, repeat.

$$c_e \leftarrow \left(r_e^2 + \varepsilon^2\right)^{p/2-1}
\qquad\text{then solve } \min  \sum_e c_e r_e^2 .$$

This is **iteratively reweighted least squares** (IRLS). Two limits explain the
behaviour: $p = 2$ gives $c_e = 1$ and returns plain least squares after one
pass, while $p \to 1$ gives $c_e = 1/(r_e^2+\varepsilon^2)^{1/2}$, which is
small for large residuals — outliers are progressively switched off. Parvaneh
runs the outer loop `outer_iter` times (default 12) and stops when the objective
stops changing (`tol`, default $10^{-7}$); each outer step is one of the
DCT-preconditioned solves of section 6, with independent weights per axis.
Smaller $p$ means a sharper penalty on discontinuities, a more "median-like"
solution, and more outer iterations.

**Where this comes from.** The minimum-$L^p$ formulation and its IRLS solution
are Ghiglia & Romero (1996); the general theory of M-estimators and
reweighting is Huber & Ronchetti (2009); and the modern analysis of IRLS for
this exact problem, including why the reweighting converges, is Dubois-Taine,
Akiki & d'Aspremont (2025).

## 9. Quality maps

Path-following methods need a rule for *where to start and where to walk*.
Parvaneh provides three quality maps; all return a float array where **larger
means better**, and all are computed from the wrapped phase alone.

**Maximum phase-gradient quality.** With

$$g_x = W(\psi_{i,j+1}-\psi_{i,j}), \qquad
  g_y = W(\psi_{i+1,j}-\psi_{i,j}),$$

compute the local cost $\text{cost} = |g_x| + |g_y|$, optionally maximised over
a $(2\,\text{window}+1)\times(2\,\text{window}+1)$ neighbourhood, and map it to

$$Q = \frac{\max \text{cost} - \text{cost}}{\max \text{cost} - \min\text{cost}}
\;\in[0,1].$$

Large $|g|$ means a steep local slope, exactly where unwrapping is at risk, so
the map is inverted. This is the default of `quality_guided_unwrap`.

**Pseudocorrelation.** Average the unit phasors over a
$(\text{window}\times\text{window})$ window (default $3\times3$):

$$Q = \sqrt{\langle \cos\psi\rangle_{\text{win}}^2
             + \langle \sin\psi\rangle_{\text{win}}^2 } \in [0,1].$$

This is the length of the local mean unit vector. It is $1$ where the phase is
locally coherent and near $0$ where the phase is random, so it is the natural
quality measure when the phase comes with noise, and it needs no
min–max rescaling.

**Derivative variance.** Take the local variance of $g_x$ and $g_y$ over a
window, invert it, and rescale to $[0,1]$ in the same way as the gradient map.
A small variance of the wrapped derivatives means the surface is locally smooth,
which is the assumption unwrapping makes.

**Where this comes from.** All three maps are the classical ones described in
Ghiglia & Pritt (1998), chapter 4; the pseudocorrelation map is a standard
InSAR construction, reviewed with its statistical motivation in the literature
surveyed in
[`phase_unwrapping_history_and_theory.md`](phase_unwrapping_history_and_theory.md).
The "flood the most reliable pixel first" idea that these maps feed is Cusack,
Huntley & Goldrein (1995), which is the predecessor of the sorted form in
section 10.

## 10. Reliability sorting: a maximum spanning forest

Reliability sorting turns unwrapping into a graph problem. Make a graph with one
**node** per pixel and one **edge** between each pair of neighbours. Every edge
carries a *priority*; the algorithm grows a maximum spanning forest by merging
the best edge first, exactly as Kruskal's algorithm grows a minimum spanning
tree.

**Step 1 — rate the pixels.** Parvaneh's rating is

$$R_p = \frac{1}{1 + S_p}, \qquad\text{larger is better},$$

$$S_p = \sum_{a}\ \sum_{q \in \text{neighbours}(p)} \left|g_{pq}\right|\,
\min\!\left(w_p, w_q\right),$$

with the confidence $w$ combining the user's weight (default $1$) and the mask
(a masked pixel has confidence $0$). $S_p$ is the total weighted wrapped
gradient around a pixel; a pixel sitting in a smooth area has small $S_p$ and
therefore large $R_p$, one sitting on a steep slope or a residue has small
$R_p$. Pixels with zero confidence get `NaN` and are skipped everywhere.

**Step 2 — order the edges.** Each edge is given the priority

$$\text{prio}_{ab} = \min(R_a, R_b),$$

so an edge is only as good as its worse endpoint, and the edges are sorted from
highest to lowest priority (a stable sort, so ties are broken by position).

**Step 3 — merge in order, and count turns.** Merging uses a
**union–find** structure: each pixel starts as its own root, and merging two
components attaches the smaller root under the larger one. When the edge
$(a,b)$ is accepted, the algorithm records how many whole turns were removed
when comparing their phases. With `raw` the raw phase difference and

$$q = \begin{cases} +1 & \text{raw} > \pi\\ -1 & \text{raw} \le -\pi\\ 0 & \text{otherwise}\end{cases}
\qquad \text{(computed without any rounding decision)}$$

the wrapped step is $g = \text{raw} - 2\pi q$. Writing $T(x)$ for the number of
turns accumulated from pixel $x$ to its current root, the accepted edge fixes
the relation

$$T(b) - T(a) = -q \qquad\Longleftrightarrow\qquad
\left(\psi_b + 2\pi T(b)\right) - \left(\psi_a + 2\pi T(a)\right) = g_{ab}.$$

A later edge whose endpoints are *already* in the same component would close a
loop; it is counted as `discarded` and used only for information. At the end
every pixel is assigned $u_p = \psi_p + 2\pi T(p)$, and each discarded edge is
exactly the $2\pi$ inconsistency left by the residues inside that loop.

**What the output guarantees.** By construction, $u_b - u_a = g_{ab}$ for *every
accepted edge*: the answer is wrap-preserving, $W(u) = \psi$ up to
floating-point rounding (measured $4\times10^{-16}$ on a noisy test), with the
only exceptions being the discarded loop-closing edges. Every connected region
of valid pixels is anchored to its own root, so the answer keeps whatever branch
the measurement had there. The bookkeeping is reported:

```
ReliabilityInfo(backend='python', pixels=4, edges=4, merges=3,
                discarded=1, components=1)
```

for the $2\times2$ vortex — four pixels, four possible edges, three merges
(which is what a tree on four nodes needs), one loop-closing edge discarded, one
component. In general `merges = pixels - components`, and the number of
discarded edges is the cycle rank of the edge graph.

Two practical notes. The cost is dominated by sorting $E$ edges,
$\mathcal{O}(E\log E)$; and the priority rule means that the most reliable
pixels decide the $2\pi$ bookkeeping, which is precisely the intended
behaviour: errors are pushed into the least trustworthy corners of the image
instead of being smeared everywhere as in the least-squares family.

**Where this comes from.** The sorting strategy — rate every pixel, sort the
edges that rating induces, merge in that order — and the two-dimensional
sorted/region-growing scheme are Herráez, Burton, Lalor & Gdeisat (2002).
Kruskal's algorithm is Kruskal (1956); union–find with path compression is
Tarjan (1975). The extension of the same recipe to volumes is Abdul-Rahman
*et al.* (2005, 2007), which is what section 14 benchmarks; their 2009 paper
handles the singularity loops a best-path order can otherwise meet, and that
remedy is not implemented here. The implementation in
[`reliability.py`](../src/parvaneh/reliability.py) is a direct transcription of
the equations above, with a Numba kernel for the merge loop that produces
bit-identical output to the Python one.

One honest caveat belongs here, because the two ratings are easy to confuse.
The equations above are **Parvaneh's** rating: first differences to the
immediate neighbours, each weighted by the confidence of that step, normalised
so that larger is better. Herráez and co-workers instead build a pixel's
unreliability from the four **second** differences in its $3\times3$
neighbourhood, squared and summed, and give an edge the **sum** of its two
endpoints' values, sorted in the opposite sense. What is taken from them is the
sorting scheme, not the printed formula; see
[`references.md`](references.md), entry 14.

## 11. Path following and branch cuts

**Quality-guided path following** repeatedly takes the unvisited pixel with the
best quality and integrates it from an already-unwrapped neighbour:

$$u_p = u_q + W(\psi_p - \psi_q).$$

Implemented with a priority queue, this visits the pixels in the order of the
quality map of section 9, starting from the best pixel and restarting in each
disconnected region. Its failure mode is worth stating clearly: the first
decision in a region is a *choice*. If it is taken on a noisy pixel, the error
propagates to everything reachable before a better path can correct it. Quality
ordering makes that unlikely, not impossible. The traversal order is itself a
spanning tree, which is why the guarantee is the same as for reliability
sorting: within one connected region, the result is wrap-preserving; different
regions may differ by multiples of $2\pi$.

**Branch cuts** attack the residues directly instead of avoiding them. The idea
of Goldstein, Zebker & Werner is:

1. compute the charges $r_{i,j}$ of section 4;
2. connect the charges in pairs with **branch cuts** — thin chains of pixels
   whose net charge is zero — so that every closed integration path encloses
   equal numbers of $+1$ and $-1$ charges and is therefore consistent;
3. if a charge cannot be balanced inside the array, cut to the border, because
   the total charge is generally non-zero (section 4);
4. integrate a path that does not cross a cut, and fill the cut pixels
   afterwards from an already-unwrapped neighbour.

The cut costs are what the algorithms differ in. Goldstein's original method
grows boxes of size $3, 5, 7, \dots$ around a charge, sums the charges they
cover, stops as soon as the sum is zero, and places the cut along the cheapest
chain through that box. Parvaneh's `goldstein_unwrap` follows that recipe with
the box grown up to twice `max_cut_length` and a Bresenham-like chain through
the connecting pixels.

**Quality-guided mask cuts** (`mask_cut_unwrap`) replace the box search by a
best-first search from each charge that walks along the smallest wrapped
gradients. It accumulates the charge of the pixels it visits and stops when the
balance is zero or when it reaches the array border or the guard ring around an
invalid mask. The resulting cut network is then thinned — a cut pixel is removed
when it is not adjacent to a charge and its removal does not disconnect the cut
— which keeps the cuts as short as possible.

Both methods report values in *cycles* internally. They build the cycle field
$((\psi+\pi)/2\pi) \bmod 1$, integrate it, and multiply by $2\pi$ at the end;
there is no `- pi` on the way back, so their raw output is offset by exactly
$\pi$ relative to the measurement. That offset is an artefact of the cycle
bookkeeping — a missing $-\pi$ on the way from cycles back to radians — not a
property of the algorithm, and this project records it as a defect to be
removed rather than as a convention to keep.

The experiment of section 16 measures it: the deviation
$\max|W(u)-\psi|$ of Goldstein, mask cuts and Flynn is $3.1416$ rad, that is
$\pi$ to the last printed digit. Adding $\pi$ to their output removes it
except for rounding ($6\times10^{-6}$ rad for Goldstein and mask cuts,
$5\times10^{-7}$ rad for Flynn). Until the bookkeeping is corrected, the CLI's
`--center circular` (the default) removes the offset for you, and the test
suite aligns the median offset, exactly as section 2 recommends. See
[`cli.md`](cli.md) for the offset table and
[`algorithms.md`](algorithms.md) for the per-backend notes.

**Where this comes from.** Branch cuts are Goldstein, Zebker & Werner (1988);
the residue-pairing problem they solve greedily has a principled
minimum-cost-matching formulation in Buckland, Huntley & Turner (1995); the
quality-guided traversal, the guard-ring treatment of masked regions, and
the mask-cut variant are described in Ghiglia & Pritt (1998), chapters 5 and 6.
The cycle bookkeeping and the guard ring around invalid masks follow the
reference algorithm's structure; the code itself is original.

## 12. Minimum discontinuity

Flynn's minimum-discontinuity method considers the whole *unwrapped* field and
counts how often it jumps by a whole turn with respect to the measurement. Write

$$\text{jump}_{ab} = \operatorname{nint}\!\left(
   \frac{u_b - u_a - g_{ab}}{2\pi} \right)$$

for an edge; the estimator tries to find the field for which

$$\sum_e \text{cost}_e \left|\text{jump}_{ab}\right|$$

is as small as possible, subject to the jumps being *consistent* around every
cell in the grid (a node-balance constraint, the same anti-symmetry condition
the reliability sorter enforces with union–find). Two details make this the
sharpest method in the repository and the slowest:

* the jumps of an image that already contains a residue cannot all be zero, so
  the method has to choose a place to spend a $2\pi$ discrepancy, and it spends
  it where it is cheapest;
* the cost of a jump is the local quality,

  $$\text{cost}_p = 1 + \text{BIG}\cdot q_p, \qquad \text{BIG} = 25500,$$

  where $q_p$ is the pixel's loss of quality, so jumps are pushed into noisy,
  low-coherence, or masked pixels (a masked pixel has $q_p = 0$ and a jump there
  is nearly free).

Parvaneh solves the resulting optimisation by sweeping the image (left to right,
then top to bottom) and, whenever a jump network can be improved, applying
increments to the affected edges: interior edges with $\pm\text{cost}$, edges of
the search window with $\pm 1$. The loop removes inconsistent cycles and orphan
nodes as they appear. Because a jump flips a facet's ambiguity, one flip can
require several passes, and the return value reports the number of iterations
(`return_iterations=True`) for exactly that reason. Its output obeys the same
guarantee as the other path-following methods — wrap preservation, per-region
offsets — and shares the cycle-bookkeeping offset described in section 11.

It is worth noticing what the objective above actually is. The quantity
$\text{jump}_{ab}$ is the integer ambiguity $k_{ab}$ of equation (13.1): it
counts the turns the measurement lost. So Flynn minimises the *same* linear
objective as the network-flow method of section 13, over the same feasible set
— the node-balance constraint written above is conservation, equation (13.3).
The difference is how the optimum is found. Flynn improves an existing field
by local increments and makes no claim of global optimality; the flow method
solves the identical linear program exactly, and pays for it with a slower
solve of its own. It is a useful comparison to run both on the same image,
because any gap between them is the price of the heuristic.

**Where this comes from.** Flynn (1997). The thinning, the sweep order, and the
cost table are the ones the published algorithm describes; the implementation
in [`flynn.py`](../src/parvaneh/flynn.py) is original Python.

## 13. Minimum-cost flow: unwrapping as a network

Sections 10 to 12 choose the whole turns step by step: a greedy merge, a
best-first walk, a local sweep. This section writes *all* of the unknowns into a
single optimisation problem on a graph and solves that problem exactly.

### 13.1 The unknowns live on the edges

Start from the difference identity of section 3. Across any two neighbouring
pixels the true phase changes by the measured wrapped step plus some whole
number of turns that the measurement could not see:

$$u_b - u_a = g_{ab} + 2\pi k_{ab}, \qquad k_{ab} \in \mathbb{Z}. \qquad (13.1)$$

The measurement does not say which integer, and every method in this page is a
different way of choosing it. What makes the choice interesting is that the
integers are not independent: added around a closed loop they must cancel the
defect of the measured steps along that loop. For the four pixels of one cell
that statement is the residue of section 4 again,

$$k^{\rightarrow}_{i,j} + k^{\downarrow}_{i,j+1} - k^{\rightarrow}_{i+1,j}
- k^{\downarrow}_{i,j} = -r_{i,j}, \qquad (13.2)$$

where $r_{i,j}$ is the charge and $k^{\rightarrow}_{i,j}$,
$k^{\downarrow}_{i,j}$ are the integers on the right step and on the down step
leaving pixel $(i,j)$.

If every charge is zero, $k \equiv 0$ satisfies (13.2) everywhere and there is
nothing left to decide. If a charge is non-zero, (13.2) cannot hold around that
cell for *any* integers, and something has to give. Which thing gives is what
separates the families:

* least squares (sections 5 to 8) lets the **measurements** give: it bends the
  steps until the loops close;
* the cut methods (sections 11 and 12) lets the **loops** give: it cuts the
  image so that the offending loops no longer exist;
* the flow family keeps every measured step exactly as measured and lets the
  **integers** give; the turns are free to be placed anywhere, but each one is
  paid for.

### 13.2 The dual network

Costantini (1998) noticed that (13.2) has the shape of a flow conservation law.
Read it that way and the nodes of the optimisation problem turn out to be
*cells*, not pixels.

* **Nodes.** One node for each $2\times2$ block of four valid pixels, plus one
  extra **ground** node that stands for "outside the data". A cell touching a
  masked pixel does not exist, and is identified with the ground.
* **Arcs.** One arc per pixel edge, carrying the integer $k$ of (13.1) as its
  flow. An arc is oriented by turning its pixel step a quarter turn, the *same*
  turn for both families:

  | pixel step | arc runs | from | to |
  | --- | --- | --- | --- |
  | right: row $i$, columns $j \rightarrow j+1$ | along a row of cells | cell $(i,j)$ | cell $(i-1,j)$ |
  | down: rows $i \rightarrow i+1$, column $j$ | along a column of cells | cell $(i,j-1)$ | cell $(i,j)$ |

  A pixel edge that has a missing cell on one side becomes an arc between the
  remaining cell and the ground. There is therefore no separate family of
  "border arcs": the arcs that meet the ground are ordinary pixel edges that
  happen to lie on the edge of the valid area.
* **Supplies.** Every existing cell demands a net inflow of $r_{i,j}$, and the
  ground absorbs whatever remains. Conservation at cell $(i,j)$ is then exactly
  (13.2), because the four arcs around a cell are its four pixel steps, and the
  ground node's own balance is minus the sum of the cell balances, so the whole
  system can always be satisfied.

**The objective.** Give every arc a price $c_e \ge 0$ and ask for the cheapest
flow that respects all the conservation laws,

$$\min \sum_e c_e\,|k_e|. \qquad (13.3)$$

This is not a proxy for the quantity we care about. Equation (13.3) *is* the
weighted total discontinuity of the answer: every unit of $|k_e|$ is one whole
turn by which the corrected step departs from the measured step, and $c_e$ says
what one turn costs in that place; section 7 explains why giving up on an edge
should cost something. The only question left is where the prices come from.

**Why the optimum is a whole number.** The constraint matrix of a network is
**totally unimodular**: every square sub-matrix formed from its rows and columns
has determinant $0$, $+1$, or $-1$. The intimidating part of (13.2) is that $k$
was declared to be an integer, and the integrality theorem for such matrices
says that the declaration costs nothing — the linear program in which $k$ is
allowed to be any real number already has an integer optimum. So the problem can
be solved as a linear program, with no rounding, no branching, and no search,
and the answer that comes out is the exact optimum of (13.3). This is the sense
in which this family is *exact* while Goldstein and Flynn are *heuristics*: the
difference is a theorem, not a tuning.

### 13.3 Prices, and how the cheapest flow is found

The prices follow the simple rule used by SNAPHU (Chen & Zebker 2002):

1. Turn the confidence into an integer weight,
   $\operatorname{round}\!\left(1000 \cdot
   \operatorname{clip}(\text{confidence}, 0, 1)\right)$, and never let a
   non-zero confidence be worth less than 1.
2. An arc costs the **smaller** of the weights of its two end pixels: an edge is
   only as trustworthy as its weaker side.
3. Confidence exactly 0 means "do not trust this pixel at all", and removes it
   from the network: the four cells that touch it do not exist, so its edges
   become ground arcs. Crossing costs nothing there, which is how a residue that
   cannot be paired inside the array escapes through a masked strip or across
   the border.

The optimum of (13.3) is found by **successive shortest augmenting paths**, the
classical algorithm for this problem (Ahuja, Magnanti & Orlin 1993,
Algorithm 9.5):

* start from zero flow and keep the *excess* of every node, the amount by which
  its balance is still unsatisfied;
* find the cheapest route from a node with surplus to a node with deficit using
  Dijkstra's algorithm on the residual network, with prices *reduced* by a
  vector of node potentials; push one unit along that route, which settles two
  units of imbalance;
* because every original price is non-negative, the potentials may start at
  zero, and because the prices are integers the potentials stay exact integers.
  There is no floating-point comparison and no tie tolerance, and the algorithm
  stops at the true optimum.

The ground node guarantees that the next unit always has somewhere to go, so the
only possible failures are a programming error or an exhausted iteration cap.
What decides the run time is the number of augmentations, and that number is not
the number of pixels: every unit of flow starts at a residue, so the count is
essentially half the total charge — the measurement in section 13.6 confirms
this, and it is also why a clean scene with no residues needs no augmentation at
all and returns $k \equiv 0$. The word *essentially* is doing real work: the
charges of a closed array need not cancel, because their sum is the net number
of whole turns that the measured gradient accumulates around the outer boundary
of the array, and that leftover is what the ground node absorbs and what the
report calls `ground_imbalance`. A scene with $492$ units of total charge and a
ground imbalance of $2$ therefore needs $(492 + 2)/2 = 247$ augmentations, not
$246$, and it is the `augmentations` field — not half the `residues` field —
that sets the run time when the two disagree.

The alternative objective `cost="quadratic"` replaces $c_e|k_e|$ by
$c_e k_e^2$, which dislikes one large jump more than several small ones. The
same solver is used, with the marginal price of the $n$-th unit on an arc equal
to $c_e (2n-1)$; that marginal-price rule is the standard treatment of convex
arc prices, and convexity is what keeps it correct. What does *not* carry over
is the integrality argument of section 13.2, which is specific to the linear
objective, so the quadratic mode is the weaker of the two guarantees.

### 13.4 From a flow back to a field

The flow is the answer, but users want a field. Once the integers are known,
every measured step is corrected to $g + 2\pi k$, and by construction those
corrected steps are curl free inside each valid region: the loops that used to
be inconsistent now contain the turns that pay for them. A breadth-first walk
from the first valid pixel of a region therefore integrates the steps exactly —
any path inside the region gives the same answer, so one walk per region is
enough — and each region receives its own arbitrary additive constant. That is
ambiguity (b) of section 2, no longer a footnote but a visible property of the
output. Pixels outside the mask are returned as `nan`, and the report counts the
regions.

Three properties of the result are worth remembering, because they are visible
in the examples below.

* The objective value is unique; the **field is not**. Different sets of
  integers can cost exactly the same, and the solver returns the first of them
  it finds.
* The optimum is not asked to be smooth. Where the data really do contain a
  discontinuity, the cheapest flow puts the jump there, often as a compact cut,
  instead of spreading it over the image in the way least squares must.
* Two regions separated by a mask are unwrapped independently: no arc connects
  them, so the flow cannot compare them and their relative offset is arbitrary.
  Section 16 measures how large that arbitrariness can be.

### 13.5 Two worked examples

**A residue that cannot be paired.** The $2\times2$ array of section 4,

$$\psi = \begin{bmatrix} 0 & \pi/2 \\ -\pi/2 & \pi \end{bmatrix},$$

has one cell, whose charge is $+1$, so its single conservation law (13.2) says
that one whole turn must be spent on one of its four sides. The four arcs of
that cell all lead to the ground — it is the only cell, so no cell lies behind
any of its sides — and all four cost the same 1000 units. The four choices are
therefore equally cheap, and the solver takes the first one it examines, the
down step from $(0,0)$ to $(1,0)$:

| quantity | value |
| --- | --- |
| pixels / nodes / arcs | 4 / 2 / 4 |
| residues, augmentations | 1, 1 |
| `ground_imbalance` | 1 |
| `max_jump` (whole turns on one edge) | 1 |
| `total_cost` (internal units) | 1000 |
| field $u$ | $\begin{bmatrix} 0 & \pi/2 \\ 3\pi/2 & \pi \end{bmatrix}$ |
| largest $\left\|\,W(u - \psi)\,\right\|$ | $2.4\times10^{-16}$ rad |

The returned field differs from the measurement by one whole turn on that one
step and agrees with it modulo $2\pi$ everywhere else. No other placement of the
turn can cost less, and three other placements cost exactly the same, which is
the first thing a user of this family has to accept: the flow gives *a* correct
answer, and the price of every answer it could have given is available in the
report.

**A dipole, and the ground left unused.** The $3\times3$ vortex of section 4,

$$\psi = \frac{\pi}{2}
\begin{bmatrix} 0 & 0 & -1 \\ 0 & 1 & 1 \\ -1 & 1 & 1 \end{bmatrix},$$

has two cells of charge, $+1$ and $-1$, a short dipole. Here all four cells of
the grid exist, so no arc meets the ground and the ground balance is zero. The
dipole forms one loop, and the solver needs a single augmentation to satisfy it:

| quantity | value |
| --- | --- |
| pixels / nodes / arcs | 9 / 5 / 12 |
| residues, augmentations | 2, 1 |
| `ground_imbalance` | 0 |
| `max_jump` | 1 |
| `total_cost` | 2000 |
| field $u$ | $\frac{\pi}{2}\begin{bmatrix} 0 & 0 & -1 \\ 0 & -3 & -3 \\ -1 & -3 & -3\end{bmatrix}$ |
| largest $\left\|\,W(u - \psi)\,\right\|$ | $2.4\times10^{-16}$ rad |

Two arcs carry one unit each — the down step $(0,1)\to(1,1)$ and the right step
$(1,0)\to(1,1)$ — and together they form the top-left corner of the $2\times2$
block $\{(1,1),(1,2),(2,1),(2,2)\}$. Correcting those two steps by $-2\pi$ makes
every other loop in the array consistent, so the walk of section 13.4 shifts
that whole block by $-2\pi$ and returns a field whose departure from the
measurement is $0$ everywhere outside it. For a vortex of this size, the exact
optimum is the visual answer: one compact cut around the block that the data say
is rotated.

### 13.6 What it costs

The number that decides the run time is the total absolute charge, not the
number of pixels, because every unit of charge is one augmentation and each
augmentation runs Dijkstra on the dual network. The table below is the output of
`benchmarks/benchmark_mcf.py`: a square array from `make_synthetic` with noise
scale $1$ rad together with that generator's own coherence weight, median of
three runs on one core of an Intel Core i7-8650U.

| grid | pixels | charge | nodes | arcs | augmentations | `max_jump` | linear (s) | quadratic (s) | RMSE (rad) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| $32^2$ | 1024 | 134 | 962 | 1984 | 67 | 2 (1) | 0.040 | 0.045 | 1.243 (1.238) |
| $48^2$ | 2304 | 267 | 2210 | 4512 | 134 | 1 | 0.157 | 0.158 | 1.247 |
| $64^2$ | 4096 | 440 | 3970 | 8064 | 220 | 1 | 0.422 | 0.473 | 1.190 |

Two readings of that table. First, the augmentations are half the charge, as
predicted — these synthetic scenes close on their own boundary, so the ground
node stays unused — while the number of nodes and arcs grows with the area; it
is the *product* of charge and nodes that sets the time — $1.3\times10^5$, then
$5.9\times10^5$, then $1.75\times10^6$, against $0.040$, $0.157$ and $0.422$
seconds. So a scene with three times the width and three times the charge is not
three but roughly ten times slower, and the charge can be reduced by gaining
coherence, not by gaining pixels. Second, both objectives return fields that
differ by less than $10^{-13}$ rad from the measured steps modulo $2\pi$, and
they differ from each other in RMSE by less than $0.01$ rad on these scenes: on
noise the two objectives are nearly the same problem, and the quadratic one
earns its extra care only when genuine discontinuities are expected. Repeating a
cell of this table on the same laptop moves the median by 10 to 20 percent; the
sweep in [`performance.md`](performance.md) carries the same rows out to
$256^2$ and quotes the spread it found there.

How does that sit next to the other families? Timings for the ten variants on
one shared $64\times64$ scene (median of three warmed runs, same machine) are:
least squares $0.001$ s, weighted least squares $0.006$ s, reliability sorting
$0.023$ s, quality-guided $0.107$ s, Goldstein $0.113$ s, minimum $L^p$ norm
$0.258$ s, minimum-cost flow $0.331$ s, weighted minimum-cost flow $0.411$ s,
Flynn $0.414$ s, mask cuts $0.460$ s. The direct solvers and the sorter are
effectively free, the exact flow solve costs a few hundred times the fast
Poisson solve, and the flow family lands in the middle of the robust group:
cheaper than mask cuts, level with Flynn, and — unlike either of them —
carrying a proof that its answer is optimal.

**Where this comes from.** The dual network, the cell conservation law, and the
$\sum c_e|k_e|$ objective are Costantini (1998), whose network this section
follows. The integer prices, the confidence handling, and the practical
behaviour of large runs are described in Chen & Zebker (2002), the paper behind
SNAPHU; SNAPHU itself uses a different network, with one node per pixel, and is
cited here as a cross-check rather than as the construction used. The solver is
Algorithm 9.5 of Ahuja, Magnanti & Orlin (1993), and the treatment of convex arc
prices behind the quadratic mode is the same book. The implementation in
`network_flow.py` is this page's own work: no source code from SNAPHU or from
any other unwrapping package was copied, and the two worked examples above are
reproduced by the test suite, which also re-derives the optimum from the
returned field instead of trusting the reported total.

## 14. Any number of dimensions

Nothing in sections 5–7 is specific to two dimensions. Replace "row and column"
by "axis $a = 1,\dots,d$" and everything holds:

* the objective sums over edges in every axis, and the divergence becomes the
  sum over axes given in section 5;
* the eigenvector of section 6 factorises into a product of one-dimensional
  cosines, so the eigenvalue of mode $k$ is the *sum* of the per-axis
  eigenvalues, and one $d$-dimensional DCT does the whole job;
* the reliability sorter of section 10 walks every axis, which raises the number
  of candidate edges from $d\,N$ (per slice) to roughly $d\,N$ per voxel while
  also joining the slices.

Parvaneh's `unwrap` and `reliability_unwrap` therefore accept an array of any
rank whose axes all have at least two samples: a single image, a stack of
interferograms (spatial–temporal unwrapping), or a cube. That generality is not
a convenience — it is the point of the third dimension, and the measurement
confirms it. On a $48\times48\times8$ cube with a Gaussian-bump surface:

| noise $\sigma$ | per-slice 2-D least squares | joint 3-D least squares |
| --- | --- | --- |
| $0$ | exact | exact |
| $0.45$ rad | 0.565 rad RMSE, per-slice offsets of 3.044 rad | 0.447 rad RMSE, offsets 0.006 rad |
| $1.5$ rad | worst of the three | best of the three |

The third dimension supplies candidate edges that the slices cannot see
(measured 52 224 joint edges against 36 096 per-slice edges, a ratio of 1.45),
so a pixel that is unreliable in one slice can be anchored through a neighbour
in the same position in the next slice. The price is a larger linear system
about 1.4 times slower than the per-slice solve, and a memory footprint that
grows with the cube.

**Where this comes from.** The three-dimensional reliability scheme is
Abdul-Rahman *et al.* (2005), extended in Abdul-Rahman *et al.* (2007); the
joint least-squares/Poisson formulation in
$d$ dimensions follows directly from Hunt (1979) and Ghiglia & Romero (1994) as
generalised here. The benchmark behind the table is
[`benchmark_nd.py`](../benchmarks/benchmark_nd.py) and the notebook
[`three_dimensional_unwrapping.ipynb`](../notebooks/three_dimensional_unwrapping.ipynb).

## 15. What each method guarantees

| Python API | CLI | finds the answer by | wrap-preserving? | what happens to residues |
| --- | --- | --- | --- | --- |
| `unwrap` | `--method ls` | solving $\Delta u = \operatorname{div}g$ | no | invisible: the rotation is annihilated by the divergence, so the error spreads out |
| `unwrap(..., weight=)` | any method plus `--weight` | the same, with edge weights $c_e$ | no | invisible; unreliable areas are pulled toward the weights |
| `unwrap_lp` | `--method lp` | IRLS on a robust objective | no | invisible, with large residuals down-weighted |
| `reliability_unwrap` | `--method reliability` | Kruskal merge of the most reliable edges | yes, up to floating-point rounding | left as $2\pi$ steps on the discarded loop-closing edges |
| `quality_guided_unwrap` | `--method quality-guided` | best-first traversal by quality | yes, within each connected island | avoided by the traversal order |
| `goldstein_unwrap` | `--method goldstein` | expanding boxes around each charge, then integrate away from the cuts | yes, modulo the documented $\pi$ convention | made consistent by the cut network |
| `mask_cut_unwrap` | `--method mask-cut` | cuts grown from each charge along minimum-gradient paths | yes, modulo the $\pi$ convention | made consistent by the cut network |
| `flynn_unwrap` | `--method flynn` | minimising the weighted discontinuity count | yes, modulo the $\pi$ convention | resolved as the cheapest jumps in the network |
| `network_flow_unwrap` | `--method mcf` | the cheapest set of $2\pi$ jumps that leaves the wrapped field consistent | yes, exactly, with no convention to correct | charged to the flow, at the lowest total jump cost |

"Wrap-preserving" means $W(u) = \psi$ for every valid pixel, so the answer
still carries everything the measurement contained and any disagreement is
concentrated in a few jumps. It is the property to check first when you cannot
trust the constant, the offsets, or the noise level. The least-squares family
never has it: it can be closer to the truth in mean-square terms, and it will
still be wrong at a whole-cycle level in the places that matter.

The last column is also the practical difference in *error character*. The
least-squares family distributes its error as a smooth field over the whole
image, which is easy to interpret statistically; the sorter concentrates it at
the pixels it rated worst, which is easier to check against other data
(coherence, a map of known faults); the cut methods concentrate it near the
cuts, which are themselves a diagnostic you can plot; the flow family
concentrates it in the jumps themselves, and the objective it minimises is a
single number you can compare between runs. All four are honest answers to
slightly different questions, and choosing between them is choosing which
failure you can tolerate.

**Where this comes from.** The classification follows Ghiglia & Pritt (1998),
which divides the field into minimum-norm and path-following/cut families; the
comparative study that separates their practical behaviour is Zebker & Lu
(1998); the flow family is Costantini (1998) in its two-dimensional linear
form and Chen & Zebker (2002) for the quadratic cost, both implemented here.
The alternatives that are *not* implemented are the statistical non-convex
costs of Chen & Zebker (2000, 2001), the graph-cut formulation of Bioucas-Dias
& Valadão (2007), the three-dimensional flow method of Liu & Pan (2020), and
the spatial–temporal formulation of SPURT (software only). The verification
numbers are this project's.

## 16. One worked experiment with all ten variants

The table below is reproducible: a $24\times24$ grid, the true phase a broad
Gaussian bump of amplitude 6 rad, and additive noise of $\sigma = 0.6$ rad on
the phase before wrapping,

```python
import numpy as np
from parvaneh.core import _wrap
rng = np.random.default_rng(3)
yy, xx = np.mgrid[0:24, 0:24]
truth = 6.0 * np.exp(-((yy - 12) ** 2 + (xx - 12) ** 2) / 150.0)
psi = _wrap(truth + rng.normal(0.0, 0.6, size=truth.shape))
```

The sampling condition is satisfied — the largest true step is $0.42$ rad —
and the scene contains no residues at all (total charge $0$), so all ten
variants are solving the same, easy problem. "Aligned RMSE" is the root-mean
square of $u - \text{truth}$ after the mean difference has been removed, and
$W$ is the wrapping operator of section 2.

| variant | $\max |W(u-\psi)|$ | spread of $W(u-\psi)$ | aligned RMSE |
| --- | --- | --- | --- |
| least squares | 2.8365 rad | $2.0\times10^{-14}$ rad | 0.5946 rad |
| weighted $L^2$ | 2.8365 rad | $2.0\times10^{-14}$ rad | 0.5946 rad |
| minimum $L^p$, $p = 1.2$ | 2.8365 rad | $3.6\times10^{-15}$ rad | 0.5946 rad |
| reliability sorting | 0 rad | $2.4\times10^{-16}$ rad | 0.5946 rad |
| quality-guided | 0 rad | $2.4\times10^{-16}$ rad | 0.5946 rad |
| Goldstein | $\pi$ rad | 6.283 rad | 0.5946 rad |
| mask cuts | $\pi$ rad | 6.283 rad | 0.5946 rad |
| Flynn | $\pi$ rad | 6.283 rad | 0.5946 rad |
| MCF, linear cost | 0 rad | $3.6\times10^{-15}$ rad | 0.5946 rad |
| MCF, quadratic cost | 0 rad | $3.6\times10^{-15}$ rad | 0.5946 rad |

Read it in three parts.

* **All ten variants return the same field.** The aligned RMSE is identical to
  four decimal places in every row, 0.5946 rad against a noise level of
  0.60 rad, and the difference between any two outputs is a single constant —
  the spread column is at the level of floating-point rounding, not a smooth
  error field. When the sampling condition holds and no residues are present,
  the choice of algorithm cannot change the answer.
* **They differ only in the free constant** that a residue-free scene cannot
  determine. Reliability sorting, quality-guided and both flow modes leave it
  at 0; the three cut methods leave it at exactly $\pi$, which is the
  cycle-counting convention of section 11; the least-squares family leaves it
  at $+2.8365$ rad, which *looks* like an error but is one number for the whole
  array — remove it and the least-squares output agrees with the flow output to
  $10^{-14}$ rad.
* **Removing it is the first thing to do with any unwrapped product**, and the
  cut methods do it for you: `--center circular`, the CLI default, subtracts
  the $\pi$ exactly, and the residual of Goldstein and mask cuts drops to
  $6\times10^{-6}$ rad and of Flynn to $5\times10^{-7}$ rad. What remains is
  single-precision accumulation in the cut search, not an ambiguity.

So on an easy scene the ten variants are not ten answers but one answer in ten
conventions, and the differences you see in the first two columns are
bookkeeping rather than quality. That is a statement about *this* scene: an
easy scene is exactly the case in which a well-designed method cannot lose.

**Splitting the array in two.** Now repeat it with
`mask[:, 11:13] = False`, which cuts the array into a left and a right island
and removes every residue with it (charge $0$ again). A residue-free scene with
two components has *two* free constants, so an absolute comparison is
meaningless; what is testable is the offset between the islands, and the truth
has both islands at $\approx 0$ rad, so that offset should be $\approx 0$.

| variant | relative offset, right $-$ left | $\max |W(u-\psi)|$ on valid pixels | aligned RMSE |
| --- | --- | --- | --- |
| least squares | $-0.0008$ rad | 2.8365 rad | 0.5846 rad |
| weighted $L^2$ | $-0.0008$ rad | 2.8365 rad | 0.5846 rad |
| minimum $L^p$, $p = 1.2$ | $-0.0008$ rad | 2.8365 rad | 0.5846 rad |
| reliability sorting | $-0.0008$ rad | $2.4\times10^{-16}$ rad | 0.5846 rad |
| quality-guided | $-0.0008$ rad | $2.4\times10^{-16}$ rad | 0.5846 rad |
| Goldstein, mask cuts, Flynn | $-0.0008$ rad | $\pi$ rad | 0.5846 rad |
| MCF, linear and quadratic | $-0.0008$ rad | $1.8\times10^{-15}$ rad | 0.5846 rad |

Every variant gets the relative offset right on this seed, so the guarantee of
section 15 survives the mask: the exact methods still satisfy $W(u) = \psi$ on
both islands. But the free constants are now free *per island*, and a variant
that sets one island's constant one whole turn away from the other's is still
"wrap-preserving" while being useless. That is not hypothetical. Calling the
reliability sorter with the mask,

```python
from parvaneh import reliability_unwrap, network_flow_unwrap
u_rel = reliability_unwrap(psi, mask=mask)   # right island 2*pi too high
u_mcf = network_flow_unwrap(psi, mask=mask)  # both islands on one branch
```

gives a relative offset of $-6.2840$ rad — one whole turn — and an aligned RMSE
of 3.1959 rad against 0.5846 rad for the flow solver. The flow solution is
global: its ground node keeps the two islands in one network, so the constant
is decided by the cost of the edges that reach the ground, not by which island
the traversal happened to start in.

Repeating the masked experiment over sixteen seeds (seeds $0$ to $15$) shows
how often the pairing goes wrong. A whole-turn island offset appears in

| variant | seeds with a whole-turn island offset |
| --- | --- |
| reliability sorting | 6 of 16 (seeds 1, 2, 3, 7, 8, 15) |
| quality-guided, Goldstein, mask cuts, MCF | 4 of 16 (seeds 5, 8, 9, 14) |
| Flynn, least squares, weighted $L^2$, minimum $L^p$ | 0 of 16 |

The four methods in the middle row fail on the *same* four seeds, which is worth
reading carefully: whenever a whole-turn offset appears it is one decision being
made the same way, not four independent accidents. All four decide an island's
constant from the quality of the pixels inside that island — the sorter and the
flow solver from the same per-pixel weights, the cut methods from the same
residue positions — and across a two-column gap no per-pixel evidence connects
the two islands, so each method has to guess the offset and guesses alike.

The bottom row is not a better guess, it is a different kind of answer. Flynn
picks its jumps along a spanning tree whose cuts reach the boundary, and the
least-squares family does not place turns at all: give it the mask as a zero
weight in the gap (Python's `unwrap(psi, weight)` with `weight = 0` on the
excluded columns) and it still never lands a whole turn away, because the
minimum-norm answer distributes the mismatch into a shallow seam instead — up to
$0.38$ rad across the gap in these sixteen runs. A $0.38$ rad seam is a smaller
number than $6.28$ rad and it is also a *wrong* field, just wrong in a way that
is harder to notice.

**Where this comes from.** The experiment, the numbers in it, and the code that
produced them are this project's; the methods are the ones cited in sections 5
to 13. The variants are shown with figures in
[`independent_synthetic_examples.ipynb`](../notebooks/independent_synthetic_examples.ipynb),
and the practical instructions for running them from the command line are in
[`cli.md`](cli.md).

## 17. When the noise is strong: a discriminating experiment

The easy scene above cannot separate the families, so here is one that can: the
same $24\times24$ grid and the same shape of surface, but amplitude $8$ rad and
noise $\sigma = 1.2$ rad, which is twice the noise of the first experiment.
The largest true step is $0.56$ rad, so the sampling condition *still* holds,
and the scene now carries residues: 91 of them for seed 8 and 88 for seed 3.
Both are measured over the whole array, with "aligned RMSE" defined as above.

The `weight` column says what was handed to the method that accepts a weight;
`pixel_reliability(psi)` is the reliability rating of section 10, which is a
per-pixel quality, not a variance.

| variant | weight | $\max |W(u-\psi)|$, seed 8 | RMSE seed 8 | RMSE seed 3 |
| --- | --- | --- | --- | --- |
| least squares | — | 3.1401 rad | 1.5779 rad | 1.6666 rad |
| weighted $L^2$ | rating | 3.1303 rad | 1.2900 rad | 1.3053 rad |
| minimum $L^p$, $p = 1.2$ | — | 2.8021 rad | **1.1574 rad** | **1.2237 rad** |
| reliability sorting | — | 0 rad | 1.5191 rad | 1.3201 rad |
| quality-guided | — | 0 rad | 1.8766 rad | 1.5021 rad |
| Goldstein | — | $\pi$ rad | 1.5139 rad | 2.8023 rad |
| mask cuts | — | $\pi$ rad | 3.3041 rad | 3.3055 rad |
| Flynn | rating | $\pi$ rad | 1.4019 rad | 1.6375 rad |
| MCF, linear cost | — | 0 rad | 1.3152 rad | 1.3904 rad |
| MCF, quadratic cost | — | 0 rad | 1.3152 rad | 1.3904 rad |

Now the choice of method changes the answer by a factor of almost three, and
the reading is completely different from section 16.

* **The spread is caused by the residues, not by the noise.** Every variant
  still satisfies the sampling condition, and every variant still recovers the
  surface to the right order of magnitude, but the RMSE now ranges from 1.16 to
  3.30 rad. The extra error is the price of the 91 (or 88) locations where the
  wrapped field is locally inconsistent: a method that resolves them correctly
  spreads a small error everywhere, and a method that resolves them wrongly
  leaves whole regions a cycle away from the truth.
* **Wrap preservation and accuracy are not the same thing.** The four
  exact groups still reproduce $\psi$ to $10^{-15}$ rad, and that is a real
  guarantee — the measurement is not corrupted — yet the *most accurate*
  variant is minimum $L^p$, which is not wrap-preserving at all. That is
  expected rather than contradictory: $L^p$ is asked to minimise a sum of
  residuals and it satisfies that request by smoothing, and mean-square error is
  the criterion under which smoothing wins. Among the wrap-preserving variants
  the flow modes are the most accurate on both seeds (1.3152 and 1.3904 rad),
  ahead of reliability sorting and quality-guided, and ahead of Flynn on seed 3.
* **Weighting is not automatically an improvement.** Passing the reliability
  rating to the flow solver improves it (1.3152 $\to$ 1.2613 rad on seed 8,
  1.3904 $\to$ 1.3660 rad on seed 3) and improves least squares (1.5779
  $\to$ 1.2900 rad), but it makes Flynn *worse* (1.3040 $\to$ 1.4019 rad on
  seed 8). The rating is an ordering of edges for a traversal, and it was never
  designed as a weight for a global objective; when it is used as one, the
  answer is whatever the objective says, which need not be better. Compare like
  with like: given the same weight, the flow solver beats Flynn on both seeds
  (1.2613 against 1.4019 rad, and 1.3660 against 1.6375 rad).
* **The two flow cost modes agree here.** Linear and quadratic costs give
  identical fields on both seeds, which is worth knowing before paying for
  either: on a scene with this density of residues the two objectives happen to
  select the same set of jumps, and it is the benchmark of section 13.6 — not
  this scene — that separates their runtimes.

**Where this comes from.** As in section 16, the experiment is this project's
and the methods are cited in sections 5 to 13. The residual counts, the true
step size and the RMSE values were all recomputed for this page from the script
shown; the two seeds are the ones used to check that a conclusion is not an
artefact of one noise realisation.

## 18. Multigrid: one equation on many grids

Sections 5 and 6 solved the normal equations once and for all: assemble the
weighted Laplacian $L$ of the edges, assemble the divergence $f$, solve
$L u = f$. This section solves the *same* system, and it produces the same
answer. What changes is how the work is done. The cosine transform of section 6
inverts $L$ in one step when the weights are uniform, so a weighted problem is
the only hard one; a hierarchy of grids attacks that problem without a
transform at all, which is also what makes it the only one of the two solvers
that extends to a domain which is not a rectangle. That is the last paragraph
of this section.

### 18.1 One grid is quick on the rough half of the error

Take the simplest iteration on $L u = f$: visit every pixel in turn and set it
to whatever value makes its own equation hold. For the reflecting stencil of
section 5 that update is

$$u_p \leftarrow u_p + \frac{f_p - (L u)_p}{L_{pp}},$$

which is Gauss--Seidel, and it is exactly the smoother a multigrid cycle uses.

Section 6 supplies the tool for seeing what this iteration does to the *error*.
In one dimension with the reflecting rule at both ends, the mode
$v_i = \cos(\theta(i + \tfrac12))$ with $\theta = \pi k/N$ satisfies
$(L v)_i = 2(1 - \cos\theta) v_i$, and the diagonal of $L$ is $2$ away from the
border, so a Jacobi step multiplies that mode by

$$1 - \frac{2(1 - \cos\theta)}{2} = \cos\theta .$$

Two consequences follow, and between them they are the whole argument for
multigrid.

* **Rough modes die.** A mode that alternates from sample to sample has
  $\theta \to \pi$, so $\cos\theta \to -1$ and a sweep very nearly cancels it;
  Gauss--Seidel does better still, and for it the same mode's factor approaches
  $0$ rather than $-1$, because the samples it has just updated are the
  neighbours of the ones it is about to update.
* **Smooth modes survive.** The slowest mode has $\theta = \pi/N$, so its
  factor is $1 - O(N^{-2})$, just short of one. Reducing it by ten orders of
  magnitude therefore needs a number of sweeps proportional to $N^2$ — that is,
  proportional to the number of pixels, not to the side of the image.

Measured on this implementation, with a zero right-hand side so that the sweep
acts on the error alone, one sweep of a $128\times128$ uniform-weight grid does
this to a single mode:

| $k$ | $\theta/\pi$ | decay in one sweep |
| --- | --- | --- |
| 1 | 0.0078 | $0.9998$ |
| 8 | 0.0625 | $0.9854$ |
| 32 | 0.2500 | $0.7883$ |
| 64 | 0.5000 | $0.3702$ |
| 127 | 0.9922 | $-0.0015$ |

The roughest mode is gone after one sweep; the smoothest needs of order $10^5$
of them. The measured factors are not exactly the $\cos\theta$ of the model
problem, because the border samples of a finite array update with a smaller
diagonal than the interior ones, so the mode is not an eigenvector of the sweep
— the *shape* of the table is the point, near $1$ at the smooth end and near
$0$ at the rough end.

This is why a single grid cannot be rescued by patience. The same smoother on
the finest grid with no hierarchy at all is published as a baseline: after 500
sweeps a $128\times128$ uniform scene is at a relative residual of
$1.8\times10^{-3}$ and a $512\times512$ one at $3.9\times10^{-4}$, with a
measured contraction rate of 0.95 to 0.99 per sweep. That the larger grid looks
*better* is a trap: the 500 sweeps are spent removing the rough error, which
dies in the first few of them, and the residual at the end is carried by the
smooth tail the table above describes. On the smallest grid of the sweep,
$16\times16$, the baseline does converge — the mask case takes 226 sweeps —
because $N^2$ sweeps is affordable when $N$ is small.

### 18.2 A bump is rough on a coarse grid

The smoother fails on smooth error for a local reason: information moves one
sample per step, and no small neighbourhood can tell a long-wavelength bump
from a constant. Multigrid's observation is that smoothness is *relative to the
grid*. A bump that covers a quarter of a $128\times128$ image covers half of a
$64\times64$ one and all of a $32\times32$ one. In the notation of section 6, a
mode of wavenumber $\theta$ on a grid of spacing $h$ has wavenumber $2\theta$
on the grid of spacing $2h$, because the same physical wavelength now covers
half as many samples. The slowest mode of the fine grid is therefore the
half-way mode of the next grid down, and that one a sweep *does* damp.

The recipe, and the one this package implements, is the **coarse-grid
correction**:

1. **Smooth** the fine grid a few times. This is cheap, and it removes the
   rough part of the error.
2. **Measure** what is left. The residual $r = f - L u$ is zero exactly where
   $u$ is right, and the error $e = u^{*} - u$ satisfies $L e = r$ — the same
   operator, a different right-hand side.
3. **Restrict** $r$ to the next grid down and solve there for a correction.
   This pays off twice: the residual is smooth, so the mode that stalled the
   fine sweep is a dampable mode one level down; and the coarse grid has a
   quarter of the samples in two dimensions, or an eighth in three, so a sweep
   there costs proportionally less.
4. **Prolong** the coarse correction back to the fine grid and add it.
5. **Smooth** again, because a coarse correction is only approximately a
   correction.

Entering the recursion at step 3 and returning from it once the coarsest grid
is reached gives a **V-cycle**: the order of visits traces a V through the
levels. The recursion bottoms out on a grid small enough that many sweeps there
cost nothing (50 in this implementation).

One detail in step 3 matters. The unknown being solved for is the *correction*,
not the solution, so the coarse level starts from zero and the coarse
right-hand side is the restricted residual — nothing of the fine $u$ is copied
down. And since a constant is invisible to $L$ (section 5), a right-hand side
with a nonzero average contains a component the coarse solve can never reduce,
so that average is removed before the restriction.

### 18.3 The grids have to fit together exactly

Restriction takes a fine array to a coarse one, prolongation the other way, and
both have to be chosen so that the coarse problem is the *same* equation
without the fine detail. The choices this implementation makes are visible in
the two properties the tests pin down: the coarse problem must not deform the
image, and its scale must match the fine one level after level.

**Restriction is the transpose of interpolation**, $R = \tfrac12 P^{\mathsf T}$
along each axis. In the interior of an axis that makes the coarse sample a full
weighting $[\tfrac14, \tfrac12, \tfrac14]$ of its fine neighbours, rather than
plain injection, which would let rough residual components *alias* into smooth
ones on the coarse grid — the correction would then chase an error that is not
there (the aliasing argument is Press *et al.* 2007, section 19.6). At the ends
adjointness leaves the weights asymmetric — $\tfrac34$ of a sample at the first
coarse position and $\tfrac54$ at the last, on an eight-sample axis — because
the interpolation mirrors the last fine sample onto the last coarse one, so
that column of the interpolation matrix is twice as heavy.

**Prolongation is linear interpolation**: an even fine sample copies the coarse
sample it sits on, an odd one averages its two coarse neighbours. With a
reflecting problem the interpolation has to reproduce the reflection too, or
the correction would dent the border; the mirrored end above does exactly that.

**The coarse confidences are averaged with those same weights, then divided by
the weights themselves** — $\tilde w = Rw / R\mathbf{1}$, the row-normalised
restriction. The division is what keeps a uniform confidence of one equal to
one on every level. Without it the two-dimensional restriction of ones would be
$0.5625$ at a corner and $1.5625$ diagonally opposite, every level would dent
its own border, the coarse problem would be a different equation from the fine
one, and the contraction per cycle would grow with the grid size instead of
staying near $0.2$ as measured in section 18.4.

**Coarse edge weights follow the same rule as fine ones**, the smaller of the
two end confidences squared, from section 7. The hierarchy therefore
re-discretises the objective rather than borrowing coefficients: the coarse
grid gets the same kind of problem, with the same edge rule, at a spacing twice
as large. That is also where the scale factor comes from — the code divides its
per-level operator scale by four per level, because doubling the spacing
shrinks the Laplacian by $4$ in any number of dimensions. Without that division
every coarse correction would come back off by $4^{\ell}$ and the cycle would
diverge.

A hierarchy can be built the other way, *variationally*, by forming the coarse
operator from the fine one as $L_{\ell+1} = R L_\ell P$. That is exact for the
transfers, but it produces a wider stencil, which is a real cost on the coarse
sweeps this design depends on. Re-discretising keeps a five-point stencil, and
its price appears where the weights jump between neighbouring pixels: there the
coarse operator and the fine one no longer describe the same problem, and the
correction is not quite the one the fine grid needs. How much of section 18.4's
slow cases that explains is *not* among the measurements in this document, so
that section reports the stalls as a limit of the solver rather than as a
diagnosis. The levels also lose the fine structure of the weights before the
operator question is even reached, because a coarse confidence is an average of
the fine ones; on a cut that is the first thing to check.

### 18.4 What the hierarchy does, measured

The point of the levels is that the number of cycles stops caring how large the
grid is. Measured with two smoothing sweeps a side, noise $\sigma = 0.6$ rad,
seed 11, a relative residual target of $10^{-10}$ and a 200-cycle budget:

| Size | Weight field | MG cycles | MG time | CG iterations | CG time | Ratio |
| --- | --- | --- | --- | --- | --- | --- |
| 16x16 | uniform | 12 | 0.062 s | 1 | under 0.001 s | 129x |
| 16x16 | mask | 16 | 0.083 s | 16 | 0.002 s | 41x |
| 128x128 | uniform | 13 | 0.479 s | 1 | 0.005 s | 96x |
| 128x128 | mask | 79 | 1.924 s | 39 | 0.044 s | 44x |
| 256x256 | uniform | 13 | 0.864 s | 1 | 0.018 s | 47x |
| 256x256 | mask | 155 | 9.580 s | 56 | 0.675 s | 14x |
| 512x512 | uniform | 12 | 1.914 s | 1 | 0.051 s | 37x |
| 512x512 | mask | 200, stalled at 8.0e-09 | 31.933 s | 72 | 3.104 s | 10x |

Three readings:

* **On uniform weights the cycle count is flat**: 12, 13, 13 and 12 across a
  thousand-fold change in the number of pixels, where plain relaxation needs
  sweeps proportional to that number. This is the property being bought, and it
  is the reason the hierarchy is worth having at all. The time per cycle still
  grows with the pixel count, so the totals rise from 0.062 s to 1.914 s, and
  the margin over the transform-preconditioned conjugate gradient narrows from
  $129\times$ at $16\times16$ to $37\times$ at $512\times512$.
* **A cycle is not a sweep, and the conversion is exact.** One V-cycle at two
  smoothing sweeps a side costs
  $2s\left(1 + \tfrac14 + \tfrac1{16} + \cdots\right) = \tfrac{8s}{3} \approx
  5.33$ sweeps of the finest grid; the benchmark stores that as
  `fine_sweep_equivalents` per cell, and it reads 5.34 at $128\times128$ and
  above (6.03 at $16\times16$, where the coarsest level is a larger share of
  the total). The contraction is 0.17 to 0.23 per *cycle* on the uniform field,
  against 0.95 to 0.99 per *sweep* for the hierarchy-free baseline: a cycle is
  not fast, it is worth five sweeps of the finest grid, it costs less than five
  sweeps, and it reduces an error that no such number of sweeps reduces.
* **The mask costs cycles**: 16, 79, 155, and then a stall at
  $8.0\times10^{-9}$ at $512\times512$; the thin-line and blocky fields never
  reach the target at all, and one sweep a side on the blocky field diverges
  outright. That is a documented limit of this solver rather than an accident:
  see the matrix in [`algorithms.md`](algorithms.md) for when to use `ls`
  instead. Which part of the hierarchy is responsible is left open here — the
  coarse operator of section 18.3 is one candidate, the smoother's behaviour on
  a strongly anisotropic weight field is another — and no claim in this document
  rests on the choice.

One further measurement is worth stating here, because it is easy to mistake for
an error. Both solvers minimise the same objective, and on the masked scene they
agree on every cell that carries data to $2.0\times10^{-8}$ rad while the
whole-grid distance between them is $8.7\times10^{-2}$ rad. The difference lives
entirely in the cells the weights have disconnected, where the objective says
nothing about the answer and the two solvers are therefore free to differ. The
benchmark reports both distances, `known_*` and `aligned_*`, for that reason.

**Where this comes from.** The two-grid picture, the V-cycle, and the smoothing
and approximation properties are Briggs, Henson & McCormick (2000), chapters
3–4, and Trottenberg, Oosterlee & Schuller (2001), sections 2.3, 2.4, 5.3 and
7; the latter is also where the transfer operators are set out, where the
variational coarse operator $L_{\ell+1} = R L_\ell P$ is defined, and where the
remark on re-discretisation under large coefficient jumps is made. Full
weighting as the answer to aliasing is Press *et al.* (2007), section 19.6. The
use of multigrid for phase unwrapping itself is Pritt (1996), and the book
treatment is Ghiglia & Pritt (1998), chapter 5. The numbers in section 18.4
come from [`benchmark_multigrid.py`](../benchmarks/benchmark_multigrid.py),
whose `--modes` probe is the source of the decay table in section 18.1; the
equation being solved is the one derived in section 5, and the cosine modes are
those of section 6.

## Where to go next

* [`references.md`](references.md) — the full bibliography: books with ISBNs,
  papers with DOIs, and the software this project was compared against.
* [`algorithms.md`](algorithms.md) — the same families described as
  implemented, with the API and CLI names, defaults, and complexity.
* [`phase_unwrapping_history_and_theory.md`](phase_unwrapping_history_and_theory.md)
  — where the methods came from historically, and the InSAR setting they grew up
  in.
* [`validation.md`](validation.md) — how each claim on this page is tested.
