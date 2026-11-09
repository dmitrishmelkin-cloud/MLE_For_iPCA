#include "ipca.h"
#include "quiver.h"
#include <stdexcept>

FeasibilityResult AnalyzeGenericMLEFeasibility(
    int num_samples,
    const std::vector<int> &group_sizes)
{
    if (num_samples <= 0)
        throw std::invalid_argument("the number of samples must be positive");
    if (group_sizes.empty())
        throw std::invalid_argument("at least one feature group is required");

    std::vector<int> dimensions;
    dimensions.push_back(num_samples);
    for (std::size_t i = 0; i < group_sizes.size(); ++i)
    {
        if (group_sizes[i] <= 0)
            throw std::invalid_argument("feature group sizes must be positive");
        dimensions.push_back(group_sizes[i]);
    }

    cQuiver quiver;
    quiver.InitStar(static_cast<int>(group_sizes.size()));

    sDecom decomposition(static_cast<int>(dimensions.size()));
    decomposition.alpha = dimensions;
    const int count = quiver.GenDecom(&decomposition);
    if (count < 0 || count != decomposition.n_smds)
        throw std::runtime_error("quiver canonical decomposition failed");

    FeasibilityResult result;
    result.generic_mle_feasible = true;
    result.dimension_vector = dimensions;
    result.canonical_summands = decomposition.smnds;
    result.multiplicities = decomposition.mltp;
    result.explanation =
        "The distinct summands in the generic canonical decomposition are pairwise "
        "orthogonal under the star-quiver Euler form.";

    for (int i = 0; i < count; ++i)
    {
        for (int j = i + 1; j < count; ++j)
        {
            const int forward = quiver.EulerForm(
                decomposition.smnds[i], decomposition.smnds[j]);
            const int reverse = quiver.EulerForm(
                decomposition.smnds[j], decomposition.smnds[i]);
            if (forward != 0 || reverse != 0)
            {
                result.generic_mle_feasible = false;
                result.explanation =
                    "At least two distinct summands in the generic canonical "
                    "decomposition are not orthogonal under the star-quiver "
                    "Euler form; generic local semisimplicity and generic "
                    "unpenalized MLE existence fail.";
                return result;
            }
        }
    }

    return result;
}
