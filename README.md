# MLE for incomplete PCA

This repository contains a C++ implementation of the star-quiver canonical
decomposition criterion for generic unpenalized iPCA MLE existence, together
with a flip-flop covariance estimator.

## Build

Requires CMake 3.12 or newer and a C++11 compiler:

```text
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

The executable is `ipca` (or `ipca.exe` on Windows).

## Check feasibility from dimensions

Run the check before optimization with the number of samples followed by one
feature count per group:

```text
ipca check 20 8 5 3
```

The output includes the generic feasibility verdict and the canonical
decomposition. The dimension vector is `[samples, group-1-features, ...]` for
the star quiver whose arrows point from each feature-group vertex to the sample
vertex. The criterion implemented here declares generic MLE existence when
the distinct summands of the generic canonical decomposition are pairwise
orthogonal under the quiver Euler form. A single-summand decomposition is
therefore feasible.

This is a *generic, dimension-only* result: `yes` means an MLE exists for
generic feature data of those dimensions, while `no` means generic local
semisimplicity and generic MLE existence fail. It is not a data-specific
semistability test: exceptional observed datasets may differ from the generic
case. The check is algebraic and runs before any numerical optimization.

The command exits with status 0 for `yes`, 1 for `no`, and 2 for invalid input
or an analysis error.

## Estimate covariance matrices

Provide one headerless CSV per feature group. Each file has samples as rows,
features as columns, comma-separated finite numbers, and all files must have
the same number of rows:

```text
ipca estimate 1000 group_a.csv group_b.csv
```

The command reports the dimension-only feasibility analysis first, then runs
the existing flip-flop estimator and prints the sample covariance and each
group covariance. Estimation is attempted even when generic MLE existence
fails for the dimensions, because individual data can have different
algebraic properties. A non-converged run exits with status 2; singular
intermediate covariances are reported as errors.

## Existing example search

Run `ipca demo` to retain the original search over selected five-group
dimensions and random-data flip-flop examples.
