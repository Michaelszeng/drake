#include "drake/planning/visibility_graph_builder.h"

#include <utility>

#include "drake/planning/visibility_graph.h"

namespace drake {
namespace planning {

VisibilityGraphBuilder::VisibilityGraphBuilder(
    const CollisionChecker&  checker, const Parallelism parallelize)
    : checker_{checker}, parallelize_{parallelize} {}

Eigen::SparseMatrix<bool> VisibilityGraphBuilder::DoBuildAdjacencyMatrix(
    const Eigen::Ref<const Eigen::MatrixXd>& points) const {
  return VisibilityGraph(checker_, points, parallelize_);
}

}  // namespace planning
}  // namespace drake
