#include "drake/planning/iris_region_from_clique_builder.h"

#include "drake/geometry/optimization/hyperellipsoid.h"
namespace drake {
namespace planning {
using geometry::optimization::ConvexSets;
using geometry::optimization::Hyperellipsoid;
using geometry::optimization::IrisOptions;

/**
 * The default configurations for running IRIS when building a convex set from a
 * clique. Currently, it is recommended to only run IRIS for one iteration when
 * building from a clique so as to avoid discarding the information gained from
 * the clique.
 */
DefaultIrisOptionsForIrisRegionFromCliqueBuilder::
    DefaultIrisOptionsForIrisRegionFromCliqueBuilder()
    : IrisOptions() {
  iteration_limit = 1;
}

IrisRegionFromCliqueBuilder::IrisRegionFromCliqueBuilder(
    const ConvexSets& obstacles, const HPolyhedron& domain,
    const IrisOptions& options, const double rank_tol_for_lowner_john_ellipse)
    : ConvexSetFromCliqueBuilderBase(),
      obstacles_{obstacles},
      domain_{domain},
      options_{options},
      rank_tol_for_lowner_john_ellipse_{rank_tol_for_lowner_john_ellipse} {}

std::unique_ptr<ConvexSet> IrisRegionFromCliqueBuilder::DoBuildConvexSet(
    const Eigen::Ref<const Eigen::MatrixXd>& clique_points) {
  const Hyperellipsoid starting_ellipse =
      Hyperellipsoid::MinimumVolumeCircumscribedEllipsoid(
          clique_points, rank_tol_for_lowner_john_ellipse_);
  options_.starting_ellipse = starting_ellipse;
  return std::move(geometry::optimization::Iris(
      obstacles_, starting_ellipse.center(), domain_, options_)).Clone();
}

}  // namespace planning
}  // namespace drake
