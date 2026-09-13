# Independent implementation status

Every public algorithm is implemented inside the Python package and can be
tested without external executables or datasets. Historical algorithm names
identify the mathematical family, not a runtime dependency.

| Family | Public API | Implementation | Independent validation |
|---|---|---|---|
| Unweighted least squares | `unwrap(phase)` | DCT Poisson solver | Exact recovery on smooth synthetic phase |
| Weighted least squares PCG | `unwrap(phase, weight)` | Matrix-free PCG with DCT preconditioner | Convergence, backend equivalence, finite-output tests |
| Quality-guided path following | `quality_guided_unwrap` | Python and Numba priority traversal | Cross-backend equivalence and phase congruence |
| Reliability sorting | `reliability_unwrap` | Herráez rating plus union–find spanning tree | Cross-backend bit-equality, 1-D/2-D/3-D recovery, spanning-tree and mask accounting |
| Residue detection | `phase_residues` | Discrete wrapped circulation | Isolated synthetic vortex with known charge |
| Goldstein branch cuts | `goldstein_unwrap` | Independent charge-balancing implementation | Smooth recovery, topology shape, congruence tests |
| Quality-guided mask cuts | `mask_cut_unwrap` | Independent mask/cut implementation | Smooth synthetic recovery and cut validation |
| Flynn minimum discontinuity | `flynn_unwrap` | Independent region optimization | Phase congruence and deterministic convergence tests |
| Minimum-$L^p$ norm | `unwrap_lp` | IRLS with matrix-free weighted solves | $p=2$ equivalence, finite objective, iteration tests |
| Minimum-cost flow | `network_flow_unwrap` | Dual-network flow solved by successive shortest augmentations | Optimality cross-check against `scipy.optimize.linprog`, synthetic charge recovery, mask and weight behaviour |
| Difference/jump diagnostics | `surface_difference`, `discontinuity_map` | Native NumPy | Deterministic offset and jump tests |

“Independent” means the installed package executes only code distributed in
this repository plus declared Python dependencies. It does not invoke or load
historical C/MATLAB implementations. Mathematical provenance remains cited in
the documentation.

The least-squares and reliability rows (plus their weighted, masked and volume
variants) are exercised on 1-D, 2-D and 3-D input. A volume built by repeating
one slice must reproduce the 2-D answer slice for slice, the accepted
reliability edges must form a spanning tree with exactly `pixels - 1` merges per
region, and the compiled and reference backends must agree bit for bit. The
remaining families, including the minimum-cost flow solver, are
validated as 2-D implementations. The flow solver is checked against the
optimum that a linear program on the same network finds, in both cost
modes, so its own answer is verified rather than only its plausibility.

