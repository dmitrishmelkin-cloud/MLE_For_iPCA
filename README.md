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

For rank-deficient data, run the explicitly regularized variant:

```text
ipca estimate-regularized 1000 0.001 0.5 group_a.csv group_b.csv
```

The arguments after the iteration limit are the shrinkage fraction and update
damping. Shrinkage must be in `(0, 1)` and damping in `(0, 1]`. Each covariance
update is shrunk toward a scaled identity using its mean diagonal, then blended
with the previous iterate using the damping factor. This can prevent singular
intermediate matrices, but changes the estimator: convergence is not evidence
that the unpenalized MLE exists or has been computed. The original `estimate`
command is unchanged.

## Diagnose numerical rank and flip-flop iterations

Run diagnostics on observed CSV groups:

```text
ipca diagnose 1000 group_a.csv group_b.csv
```

The command reports numerical ranks for each group and the horizontally
concatenated design matrix, then prints the dimension-only generic feasibility
result. Its CSV iteration trace records the sample covariance `Sigma` and
every group covariance `Delta[j]` immediately after each update, including the
minimum and maximum eigenvalues, relative minimum eigenvalue, and condition
number. This makes the first `Sigma` update visible even when it is singular
and prevents the next SPD solve from proceeding. Numerical ranks use a
pivoted, re-orthogonalized QR calculation with relative tolerance `1e-10`
scaled by the matrix dimensions; they are numerical diagnostics, not exact
symbolic ranks.

To compare with an independent continuous random instance of the same
dimension vector, use seeded standard Gaussian groups:

```text
ipca diagnose-gaussian 1000 20 17 8 5 3
```

Arguments are the iteration limit, sample count, random seed, and feature
count of each group. The generator samples each entry independently from
`Gaussian(0, 1)` and prints the seed for reproducibility.

The flip-flop updates solve SPD linear systems using Cholesky factorization,
rather than forming covariance inverses. Before solving, the implementation
checks symmetry, computes the minimum and maximum eigenvalues, reports their
ratio as the condition number, and rejects matrices whose relative minimum
eigenvalue is at most `1e-12` (equivalently, condition number above `1e12`).
These checks are scale-relative and apply to both estimation modes.

The dimension-only feasibility result still uses the repository's existing
Derksen--Weyman canonical-decomposition implementation. The diagnostics do not
claim an independent implementation of that algorithm; independently
cross-checking canonical decompositions remains a separate validation task.

## Existing example search

Run `ipca demo` to retain the original search over selected five-group
dimensions and random-data flip-flop examples.
