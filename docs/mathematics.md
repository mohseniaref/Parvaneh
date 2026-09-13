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
13. [Any number of dimensions](#13-any-number-of-dimensions)
14. [What each method guarantees](#14-what-each-method-guarantees)
15. [One worked experiment with all seven methods](#15-one-worked-experiment-with-all-seven-methods)

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
deliberately does **not** remove per-island offsets. Chapter 15 shows the two
numbers side by side for all seven methods. A result with a perfect
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
*et al.* (2005, 2007), which is what section 13 benchmarks; their 2009 paper
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
[`references.md`](references.md), entry 10.

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
$\pi$ relative to the measurement. This is a documented convention, not an
error: the CLI's `--center circular` (the default) removes it, and the test
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
offsets — and shares the $\pi$ convention of section 11.

**Where this comes from.** Flynn (1997). The thinning, the sweep order, and the
cost table are the ones the published algorithm describes; the implementation
in [`flynn.py`](../src/parvaneh/flynn.py) is original Python.

## 13. Any number of dimensions

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

## 14. What each method guarantees

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
cuts, which are themselves a diagnostic you can plot. All three are honest
answers to slightly different questions, and choosing between them is choosing
which failure you can tolerate.

**Where this comes from.** The classification follows Ghiglia & Pritt (1998),
which divides the field into minimum-norm and path-following/cut families; the
comparative study that separates their practical behaviour is Zebker & Lu
(1998). The alternatives that are not implemented here are the network-flow
methods of Costantini (1998), Costantini & Rosen (1999) and Chen & Zebker (2000,
2001), the graph-cut formulation of Bioucas-Dias & Valadão (2007), the
three-dimensional flow method of Liu & Pan (2020), and the spatial–temporal
formulation of SPURT (software only). The verification numbers are this
project's.

## 15. One worked experiment with all seven methods

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

The sampling condition is satisfied (the largest true step is 3.5 rad/24 pixels,
far below $\pi$), so all seven methods return the *same branch* and the only
question is how faithfully each represents the noise.

| method | $\max|W(u)-\psi|$ | aligned RMSE (noise $\sigma = 0.6$) |
| --- | --- | --- |
| least squares | 3.45 rad | 0.59 rad |
| weighted $L^2$ | 3.45 rad | 0.59 rad |
| minimum $L^p$, $p = 1.2$ | 3.45 rad | 0.59 rad |
| reliability sorting | $4.4\times10^{-16}$ rad | 0.59 rad |
| quality-guided | 0 rad | 0.59 rad |
| Goldstein | 3.14 rad, exactly $\pi$ | 0.59 rad |
| mask cuts | 3.14 rad, exactly $\pi$ | 0.59 rad |
| Flynn | 3.14 rad, exactly $\pi$ | 0.59 rad |

Read it in three parts.

* Every method recovers the surface to within the noise, and the aligned errors
  are identical: 0.59 rad against a noise level of 0.60 rad. When the
  sampling condition holds, the choice of algorithm does not change the answer.
* The three rows with an exact $\pi$ deviation are the cycle-counting
  convention of section 11. Adding $\pi$ to their output — or passing
  `--center circular`, which is the CLI default — reduces the deviation to
  $6\times10^{-6}$ rad (Goldstein, mask cuts) and $5\times10^{-7}$ rad (Flynn):
  wrap preservation up to single-precision accumulation.
* The top three rows, by contrast, deviate by 3.45 rad, and aligning them does
  not help: the deviation is not an offset but the smoothing itself, spread over
  the whole array. These are the methods to reach for when you trust the noise
  model and want the smallest mean-square error; they are the wrong choice when
  a whole-cycle error in one corner of the image would ruin the product.

Repeating the experiment with a mask that splits the array in two
(`mask[:, 11:13] = False`) leaves the guarantee intact: reliability and
quality-guided still satisfy $W(u)=\psi$ on both islands
($0$ and $4.4\times10^{-16}$ rad), and the mean difference between the two
islands is $0.0000$ rad — they were anchored to the same branch here, but
nothing forced them to be.

**Where this comes from.** The experiment, the numbers in it, and the code that
produced them are this project's; the methods are the ones cited in sections 5
to 12. The same seven methods are shown with figures in
[`independent_synthetic_examples.ipynb`](../notebooks/independent_synthetic_examples.ipynb),
and the practical instructions for running them from the command line are in
[`cli.md`](cli.md).

## Where to go next

* [`references.md`](references.md) — the full bibliography: books with ISBNs,
  papers with DOIs, and the software this project was compared against.
* [`algorithms.md`](algorithms.md) — the same families described as
  implemented, with the API and CLI names, defaults, and complexity.
* [`phase_unwrapping_history_and_theory.md`](phase_unwrapping_history_and_theory.md)
  — where the methods came from historically, and the InSAR setting they grew up
  in.
* [`validation.md`](validation.md) — how each claim on this page is tested.
