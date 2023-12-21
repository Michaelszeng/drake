#pragma once
#include <memory>

#include "drake/planning/adjacency_matrix_builder_base.h"
#include "drake/planning/collision_checker.h"

namespace drake {
namespace planning {

class VisibilityGraphBuilder final : public AdjacencyMatrixBuilderBase {
 public:
  VisibilityGraphBuilder(std::shared_ptr<CollisionChecker> checker,
                         bool parallelize = true);

 private:
  Eigen::SparseMatrix<bool> DoBuildAdjacencyMatrix(
      const Eigen::Ref<const Eigen::MatrixXd>& points) const override;

  std::shared_ptr<CollisionChecker> checker_;
  bool parallelize_;
};

}  // namespace planning
}  // namespace drake
