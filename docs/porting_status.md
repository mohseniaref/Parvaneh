# Independent implementation status

Every public algorithm is implemented inside the Python package and can be
tested without external executables or datasets. Historical algorithm names
identify the mathematical family, not a runtime dependency.

| Family | Public API | Implementation | Independent validation |
|---|---|---|---|
| Unweighted least squares | `unwrap(phase)` | DCT Poisson solver | Exact recovery on smooth synthetic phase |
| Weighted least squares PCG | `unwrap(phase, weight)` | Matrix-free PCG with DCT preconditioner | Convergence, backend equivalence, finite-output tests |
| Quality-guided path following | `quality_guided_unwrap` | Python and Numba priority traversal | Cross-backend equivalence and phase congruence |
| Residue detection | `phase_residues` | Discrete wrapped circulation | Isolated synthetic vortex with known charge |
| Goldstein branch cuts | `goldstein_unwrap` | Independent charge-balancing implementation | Smooth recovery, topology shape, congruence tests |
| Quality-guided mask cuts | `mask_cut_unwrap` | Independent mask/cut implementation | Smooth synthetic recovery and cut validation |
| Flynn minimum discontinuity | `flynn_unwrap` | Independent region optimization | Phase congruence and deterministic convergence tests |
| Minimum-$L^p$ norm | `unwrap_lp` | IRLS with matrix-free weighted solves | $p=2$ equivalence, finite objective, iteration tests |
| Difference/jump diagnostics | `surface_difference`, `discontinuity_map` | Native NumPy | Deterministic offset and jump tests |

“Independent” means the installed package executes only code distributed in
this repository plus declared Python dependencies. It does not invoke or load
historical C/MATLAB implementations. Mathematical provenance remains cited in
the documentation.

