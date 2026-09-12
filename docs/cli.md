# Command line

`parvaneh` is the command-line front end of the package. It reads a wrapped
phase image, runs one of the unwrapping algorithms described in
[`algorithms.md`](algorithms.md), and writes the unwrapped phase back out. It is
the fastest way to try the software because it needs no Python code at all.

This page is written for someone who has never unwrapped interferometric phase
before. Every option is explained, every example can be typed as-is, and the
last section is a troubleshooting table.

## Installing the command

```bash
git clone https://github.com/mohseniaref/Parvaneh.git
cd Parvaneh
python -m pip install -e '.[numba]'
```

The installation creates a `parvaneh` command. If you prefer not to install
anything, or the command is not on your `PATH`, run the package as a module
from the repository root instead — the two are equivalent:

```bash
python -m parvaneh --help
```

## Two ways to spell a command

```bash
parvaneh unwrap wrapped.npy --method goldstein -o unwrapped.npy
parvaneh wrapped.npy --method goldstein -o unwrapped.npy        # same thing
```

`unwrap` is currently the only subcommand, so it is optional: if the first word
after `parvaneh` is not a known subcommand, the input file name is assumed.
Both forms behave identically.

## First run

```bash
$ parvaneh unwrap wrapped.npy -o unwrapped.npy
input    wrapped.npy  shape=1024x1024
method   ls (backend numba, workers -1)
elapsed  0.842 s
wrote    unwrapped.npy
```

The three progress lines go to **stderr**, and `backend numba` is the concrete
kernel that `--backend auto` selected on this machine. The file is written to
`-o`. If you do not pass `-o`, the command unwraps and reports the time but
writes nothing, which is handy for timing a method.

Three things worth knowing immediately:

- **The default method is `ls`** (least squares). It is the safest first guess.
- **The output offset is adjusted** so that the result lines up with the input
  phase (see [The offset question](#the-offset-question) below).
- **`-o` decides the format** by its extension: `.npy`, `.npz`, or anything
  else, which is treated as a headerless raw raster.

## Listing the algorithms

```bash
parvaneh unwrap --method list
```

```text
available methods:
  ls              weighted least squares (Ghiglia-Romero); smooth, globally optimal
                    backends: numba, blas, numpy
  quality-guided  best-first traversal from the most reliable edges
                    backends: numba, python
  goldstein       Goldstein expanding-box branch cuts
                    backends: no backend choice
  ...
compiled kernels on this machine:
  numpy    available
  blas     available
  numba    available
  cython   not available
  cupy     not available
```

The listing goes to standard output, so it can be piped or searched. The second
half is machine-specific: it reports which accelerated kernels this installation
can actually use, not which are advertised by the package.

## Reading and writing files

### `.npy` and `.npz` — NumPy's own formats

```bash
parvaneh unwrap wrapped.npy -o unwrapped.npy            # single array
parvaneh unwrap scene.npz --npz-key phase -o out.npz    # array inside an archive
```

`.npy` files carry their own shape and data type, so nothing else is needed.
For `.npz` archives use `--npz-key` to say which array inside the archive holds
the phase (default: `phase`). If the archive does not contain that name, the
error message lists the names it does contain.

### Raw rasters — no header at all

Many InSAR tools write the pixels of an image with no shape and no data type.
You must supply both:

```bash
parvaneh unwrap wrapped.raw --shape 1024 1024 --dtype '<f4' -o unwrapped.raw
```

| Option | Meaning | Default |
|---|---|---|
| `--shape ROWS COLS` | image size; **required** for raw input | — |
| `--dtype DTYPE` | how the raw bytes are encoded | `<f4` (little-endian float32) |
| `--out-dtype` | encoding of a raw **output** | same as `--dtype` |
| `--order {C,F}` | row-major (`C`) or column-major (`F`) storage | `C` |
| `--scale` | multiply raw input by this value before unwrapping | `1.0` |
| `--offset` | add this after scaling | `0.0` |

`--shape`, `--dtype` and `--order` apply to a raw **input**. For a raw
**output**, `--shape` is optional (the result already has a shape) and
`--out-dtype` chooses the encoding; if you leave `--out-dtype` out, the output
uses `--dtype`. `--scale` and `--offset` only affect raw input, because `.npy`
and `.npz` already store numerical values. They exist for data stored as scaled
integers, for example a product whose phase is `value * 0.001` radians:

```bash
parvaneh unwrap wrapped.int16 --shape 1024 1024 --dtype '<i2' \
    --scale 0.001 -o unwrapped.raw --out-dtype '<f4'
```

### Non-finite pixels in a raw output

A raw raster cannot store `NaN`, so if the result contains non-finite pixels
(the usual cause is a masked-out region) and the output is raw, the command
**refuses to write** rather than silently corrupting the file:

```text
error: the unwrapped phase contains non-finite pixels, which a raw raster cannot store.
       Pass --invalid-fill VALUE to substitute a sentinel, or write to .npy/.npz instead.
```

Follow the advice — either write `.npy`, or choose a sentinel:

```bash
parvaneh unwrap wrapped.npy --mask valid.npy -o out.raw --invalid-fill -9999
```

The same guard applies when you pass `--invalid-fill` to a `.npy` output: the
option is ignored there only in the sense that `.npy` can store `NaN`, so
nothing needs to be replaced.

## Masks and weights

A **mask** marks which pixels carry a usable measurement. Non-zero means valid.
Everything else — shadow, water, layover, a coherence threshold — should be
zero in the mask:

```bash
parvaneh unwrap wrapped.npy --mask valid.npy -o out.npy
```

The mask can be raw as well; use `--mask-dtype` for that (default `<f4`). It
must have exactly the same shape as the phase image, and the command says so if
it does not.

A **weight** is the continuous version of the same idea: a per-pixel
reliability, for instance coherence, which is more informative than a hard
yes/no. Weights must be finite and nonnegative, and only the least-squares
method (`--method ls`) accepts them:

```bash
parvaneh unwrap wrapped.npy --weight coherence.npy -o out.npy
```

A weight can be a raw raster too, with `--weight-dtype` (default `<f4`) and, for
raw input, `--shape`.

For `ls` a mask is simply converted into a 0/1 weight, so `--mask` and
`--weight` are the same kind of information; if you pass both, the weight wins.
For the path-following methods (`quality-guided`, `goldstein`, `mask-cut`,
`flynn`) a mask keeps the algorithm away from bad pixels and no weight is
needed, because those methods build their own quality map. Masked pixels stay
invalid in the output, which is exactly the case the `--invalid-fill` guard
above exists for.

Two caveats about masks:

- Masking a scene can split it into **disconnected islands**. Path-following
  methods restart each island independently, so each island gets its own offset
  and the gaps between islands are not meaningful. Values inside each island
  are.
- A mask is not a repair tool. Everything inside the mask is still the
  measured phase; unwrapping cannot invent information that was never
  recorded.

## Quality maps

`--method quality-guided` and `--method flynn` need to know which pixels are
trustworthy. You choose how that is measured:

| `--quality` | What it measures | Window |
|---|---|---|
| `min_gradient` | edge priority: the smaller of the two wrapped gradients | none (default) |
| `max_gradient` | largest wrapped gradient in a window | `--window` (default 3) |
| `pseudocorrelation` | coherence-like measure in a window | `--window` (default 3) |
| `derivative_variance` | variance of the wrapped derivative in a window | `--window` (default 3) |

```bash
parvaneh unwrap wrapped.npy --method quality-guided --quality pseudocorrelation \
    --window 5 -o out.npy
```

Two restrictions are worth memorising:

- The compiled Numba path follower only implements `min_gradient`. Asking for
  another quality map with `--backend numba` is rejected with an explanation;
  drop `--backend` (or set `--backend python`) to use it.
- `--quality min_gradient` is also a special *sentinel*: for `quality-guided`
  it selects the edge-priority traversal. For any other method that wants a
  map, the sentinel is turned into a real `min_gradient`-style map without you
  doing anything.

## Choosing and tuning the algorithm

```bash
parvaneh unwrap wrapped.npy --method flynn -o out.npy
parvaneh unwrap wrapped.npy --method lp --p 1.1 --outer-iter 20 -o out.npy
parvaneh unwrap wrapped.npy --method goldstein --max-cut-length 200 -o out.npy
```

| Option | Method | Meaning | Default |
|---|---|---|---|
| `--method` | all | which algorithm to run | `ls` |
| `--backend` | `ls`, `quality-guided` | `auto`, `numpy`, `blas`, `numba`, `cython`, `cupy` | `auto` |
| `--p` | `lp` | norm exponent, between 1 and 2 | `1.2` |
| `--outer-iter` | `lp` | reweighting iterations | `12` |
| `--inner-iter` | `lp` | least-squares iterations per reweighting | `100` |
| `--epsilon` | `lp` | softening of the reweighting, in radians | `1e-3` |
| `--max-cut-length` | `goldstein` | cap on a branch cut's expansion search | none |

`--backend auto` picks the fastest kernel that is both available on this
machine and supported by the chosen method. Naming a backend that is missing or
unsupported is an error with a message that lists the usable alternatives. The
methods without a compiled kernel (`goldstein`, `mask-cut`, `flynn`, `lp`)
ignore `--backend` and say so on stderr.

## Speed

The least-squares solver is iterative, so it has two knobs:

| Option | Meaning | Default |
|---|---|---|
| `--workers` | threads used by the DCT preconditioner; `-1` uses every core | `-1` |
| `--max-iter` | iteration cap | `100` |
| `--tol` | relative residual target; **larger is faster** | `1e-8` |

```bash
parvaneh unwrap big.raw --shape 4096 4096 --tol 1e-6 --max-iter 40 -o big-u.raw
```

`--tol 1e-8` is converged to the point where further iterations change nothing
you can see. `1e-6` and `1e-5` are common production choices for large scenes
and can cut the runtime substantially. These three options only affect `ls`
(and the inner solves of `lp`).

## The offset question

This is the one conceptual trap of phase unwrapping, and the CLI handles it for
you by default.

An unwrapped image is only defined **up to an additive constant**: if a surface
is a valid unwrapping, so is that surface plus 17.3 radians, because the
constant disappears when the phase is wrapped again. Different algorithms pick
different constants, so two correct results can look completely different when
you subtract one from the other.

`--center circular` (the default) removes that ambiguity by aligning the offset
of the result with the wrapped input:

| `--center` | Behaviour |
|---|---|
| `circular` (default) | output is shifted so that `result − input` is a multiple of $2\pi$; results from different methods are directly comparable |
| `none` | raw algorithm output, offset included |

Use `--center none` only when you have your own reference (a GPS station, a
stable region, a corner reflector) and you want to see the untouched numbers.

The `center` value used is reported in `--info`.

One consequence of `circular` is worth stating explicitly, because it surprises
people comparing methods: the cycle-counting methods (`goldstein`, `mask-cut`,
`flynn`) count from `((phase + π) / 2π)`, so their result is shifted by exactly
$\pi$ relative to the input. `--center circular` therefore reports an offset of
$\pi$ for those methods and `0` for the least-squares family. Both describe the
same physical surface, and both stay internally consistent from run to run.

## Machine-readable results: `--info`

```bash
$ parvaneh unwrap wrapped.npy --method ls --info -o out.npy
```

```json
{
  "backend": "numba",
  "center": "circular",
  "converged": true,
  "input": "wrapped.npy",
  "iterations": 1,
  "method": "ls",
  "output": "out.npy",
  "relative_residual": 1.9e-13,
  "seconds": 0.34,
  "shape": [1024, 1024],
  "weights": "uniform"
}
```

The JSON object goes to **stdout**, so it can be piped into `jq`, a shell
variable, or a batch script:

```bash
parvaneh unwrap wrapped.npy --info -o out.npy | jq -r .seconds
```

Always present: `method`, `backend` (`null` when the method has no backend
choice), `center`, `shape`, `seconds`, `input`, `output`. Then a
method-specific block:

| Method | Extra keys |
|---|---|
| `ls` | `weights` (`uniform`, `mask`, or `custom`), `iterations`, `relative_residual`, `converged` |
| `quality-guided` | `quality` |
| `goldstein`, `mask-cut` | `cut_pixels` |
| `flynn` | `iterations` |
| `lp` | `p`, `outer_iterations`, `inner_iterations`, `objective`, `relative_change` |

Two of these are worth watching in practice: `converged: false` for `ls` means
`--max-iter` stopped the solver early (raise it or relax `--tol`), and a large
`cut_pixels` for `goldstein` or `mask-cut` means the cut set grew large, which
is a warning sign that the residue density is high.

## Output, exit codes, and scripting

| Option | Effect |
|---|---|
| `-q`, `--quiet` | suppress the progress lines on stderr |
| `-V`, `--version` | print the version and exit |
| `-h`, `--help` | usage summary |

| Exit code | Meaning |
|---|---|
| `0` | success (also `--help`, `--version`, `--method list`) |
| `1` | any error; a line beginning `error:` explains it on stderr |

Progress and warnings go to stderr, results go to stdout, so `-q` plus `--info`
gives a clean pipe. The command exits `1` rather than the argparse default `2`
so that a single check (`if parvaneh …; then`) covers all failures.

A complete batch script:

```bash
set -euo pipefail
for scene in scenes/*.npy; do
    name=$(basename "$scene" .npy)
    parvaneh unwrap "$scene" --method ls --tol 1e-6 --max-iter 60 \
        -q --info -o "out/${name}-unwrapped.npy" > "out/${name}-info.json"
done
```

## Comparing methods on your own data

There is no ground truth in real data, so compare methods with each other and
with the residue count:

```bash
# how bad is the scene?
python -c "
import numpy as np
from parvaneh import phase_residues
print(np.count_nonzero(phase_residues(np.load('wrapped.npy'))))
"

for m in ls lp goldstein mask-cut flynn quality-guided; do
    parvaneh unwrap wrapped.npy --method "$m" -q -o "out-$m.npy"
done
```

Because every result is offset-aligned by default, the differences between the
files now mean something. `flynn` and `lp` preserve genuine $2\pi$ cliffs;
`ls` smooths them; a large disagreement concentrated along a line usually means
residues along a real discontinuity.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `error: … is not .npy/.npz, so --shape ROWS COLS is required` | raw input needs its size; add `--shape` |
| `error: input file not found` | path typo, or you are not in the directory you think you are |
| `error: the input phase contains NaN or infinity` | fill or mask the invalid pixels before unwrapping; unwrapping needs a fully defined image |
| `error: mask has shape … but the input has shape …` | the mask was written for a different crop or a different `--order` |
| `error: --backend numba only supports --quality min_gradient` | use `--backend python`, or `--quality min_gradient` |
| `error: backend 'cython' is not available on this machine (usable here: numpy, numba)` | that kernel is not built/installed here; use a listed one or `--backend auto` |
| output is wildly offset from what you expected | you used `--center none`; the default `circular` aligns to the input |
| `parvaneh` prints nothing and does nothing | pass an input file, or `--method list`; with no arguments it prints help |
| huge runtime on a big scene | reduce iterations: `--tol 1e-6 --max-iter 40`; or use `quality-guided` |
| unwrapped image looks striped/folded | too many residues for the algorithm: mask the bad region, then try `ls` with a weight or `mask-cut` |

## Related documentation

- [`algorithms.md`](algorithms.md) — what each method does and when to use it.
- [`binary_formats.md`](binary_formats.md) — details of the raw raster reader
  and writer used by `--shape`, `--dtype` and `--order`.
- [`performance.md`](performance.md) — measured runtimes and backend
  comparisons.
