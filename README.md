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
command remains unregularized and does not apply shrinkage. The regularized
mode stops when every covariance's relative Frobenius change is below
`1e-10`; this scale-relative criterion is more meaningful when the feature
groups and imputed values have different scales. After each full regularized
iteration, the sample covariance is normalized to unit mean diagonal and
every feature covariance is multiplied by the removed scale. This fixes the
model's covariance-factor scale indeterminacy without changing their
Kronecker products. The original unregularized `estimate` mode retains its
aggregate absolute-change criterion. The regularized mode initializes each
feature covariance to that group's mean squared input magnitude, reducing the
scale transient when feature values are large or groups have different units.

## Score an observed-data likelihood (small problems)

The library exposes `ObservedDataNegativeLogLikelihood` for an exact Gaussian
score with explicit binary observation masks. The zero-mean overload is
convenient, and an overload accepts one fixed feature-mean vector per group;
each feature mean is shared across sample rows. This scores the data
conditional on a fixed mask; it does not model the mask probability. Ignoring
the missingness mechanism requires appropriate assumptions and distinct
parameters for the data and mask models. The observed association between
class and group-14 missingness in the supplied workbook means MCAR should not
be presumed; this score alone does not establish MAR or handle MNAR. Each
group is modeled as an independent matrix-normal observation with mean matrix
`1 * feature_mean[group]^T` and covariance `Delta[group] ⊗ Sigma`; a mask
entry of `1` means observed and `0` means missing. Values at masked-out
positions are not read, so they may contain a sentinel or a non-finite
placeholder. The score is the sum of the negative log-likelihoods of the
observed subvectors, including their log-determinants. Both covariance
factors must be symmetric positive definite.

This is a correctness-first reference implementation, not a production
estimator: the reference function forms a dense covariance over observed cells
and therefore rejects more than 512 observed values in any one group. It has
no optimizer, EM updates, or missingness-mechanism model; supplied means are
not estimated. The ordinary `estimate` and `estimate-regularized` commands
still require complete finite CSV inputs and do not infer missingness from
sentinels.

For the repeated row-mask layout in the current data, the library also
provides `StructuredObservedDataNegativeLogLikelihood`, with the same
zero-mean or supplied-feature-mean options. It uses matrix-normal
factorization for one repeated non-empty mask, or for fully observed rows plus
one repeated partial-column mask, conditioning the partial rows on the complete
rows. Entirely missing rows are ignored. It supports at most two distinct
non-empty row masks per group and rejects other layouts explicitly; this
avoids building a covariance over all observed cells. It remains a likelihood
evaluator, not an optimizer or EM routine.

## Unregularized fit after explicit input transformation

The original unregularized `estimate` command does not impute, scale, or
perturb its inputs. If the observed matrices are rank deficient, changing
initial covariance matrices cannot create the missing directions. For an
explicit sensitivity experiment, `estimate-jittered` first divides each
feature by its RMS, adds seeded independent Gaussian noise, saves the changed
groups to a caller-provided existing directory, and fits the **unregularized**
model:

```text
ipca estimate-jittered 1000 0.01 20261110 transformed group_a.csv group_b.csv
```

The input CSVs must already be finite and imputed; the noise SD is relative to
each feature's RMS, must be in `(0, 1]`, and the seed makes the transformation
reproducible. This is a perturbed-data
sensitivity analysis, not a numerically improved fit to the exact original
data: adding noise changes the observations and can change the MLE and its
existence. Inspect the saved `jittered_group_*.csv` files and report the noise
level with any estimates. A zero-RMS feature is mapped to zero before the
noise is added.

This mode uses the balanced unregularized iteration. It fixes the
covariance-factor scale after each full update (unit mean diagonal for `Sigma`,
with the removed scale transferred to every `Delta`), and stops when the
maximum relative Frobenius change across all factors is below `1e-10`. This
normalization leaves the Kronecker covariance products unchanged. It does not
add shrinkage or a penalty.

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

For per-iteration progress on the regularized estimator, use:

```text
ipca diagnose-regularized 1000 0.25 1 group_a.csv group_b.csv
```

The trace includes the relative Frobenius change for each covariance update,
alongside its eigenvalue and conditioning diagnostics.

The flip-flop updates solve SPD linear systems using Cholesky factorization,
rather than forming covariance inverses. Each fit checks symmetry and
scale-relative Cholesky pivots, rejecting zero or numerically unstable pivots.
This avoids repeating a full eigenvalue decomposition inside every covariance
solve; optional iteration diagnostics still report eigenvalues and condition
numbers. The updated sample covariance is solved against all feature groups
in one multi-right-hand-side operation per iteration.

The dimension-only feasibility result still uses the repository's existing
Derksen--Weyman canonical-decomposition implementation. The diagnostics do not
claim an independent implementation of that algorithm; independently
cross-checking canonical decompositions remains a separate validation task.

## Existing example search

Run `ipca demo` to retain the original search over selected five-group
dimensions and random-data flip-flop examples.
