#pragma once
#include <memory>

#include "drake/common/parallelism.h"
#include "drake/planning/adjacency_matrix_builder_base.h"
#include "drake/planning/collision_checker.h"

namespace drake {
namespace planning {

class VisibilityGraphBuilder final : public AdjacencyMatrixBuilderBase {
 public:
  VisibilityGraphBuilder(const CollisionChecker& checker,
                         const Parallelism parallelize = Parallelism::Max());

 private:
  Eigen::SparseMatrix<bool> DoBuildAdjacencyMatrix(
      const Eigen::Ref<const Eigen::MatrixXd>& points) const override;

  const CollisionChecker& checker_;
  const Parallelism parallelize_;
};

}  // namespace planning
}  // namespace drake
