#include "drake/planning/visibility_graph.h"

#include <algorithm>
#include <iterator>
#include <vector>

#include <common_robotics_utilities/parallelism.hpp>

using common_robotics_utilities::parallelism::DegreeOfParallelism;
using common_robotics_utilities::parallelism::DynamicParallelForIndexLoop;
using common_robotics_utilities::parallelism::ParallelForBackend;
using common_robotics_utilities::parallelism::StaticParallelForIndexLoop;

namespace drake {
namespace planning {
namespace {

/* A custom iterator that turns undirected edges encoded in the given
std::vector<std::vector<int>> `edges` into a sequence of Eigen::SparseMatrix
triplets which define a symmetric matrix. The edges in `edges` are undirected,
so each such edge creates two triplets: (i, j, value) and (j, i, value).
*/
class EdgesIterator {
 public:
  using T = Eigen::Triplet<bool>;
  using iterator_category = std::input_iterator_tag;
  using value_type = T;
  using difference_type = std::ptrdiff_t;
  using pointer = T*;
  using reference = T&;

  EdgesIterator(const std::vector<std::vector<int>>& edges, int outer_idx = 0,
                int inner_idx = 0, bool upper_triangle = false)
      : edges_(edges),
        outer_idx_(outer_idx),
        inner_idx_(inner_idx),
        upper_triangle_(upper_triangle) {
    skip_empty_outer();
  }

  EdgesIterator begin() {
    return EdgesIterator(edges_, /* outer_idx = */ 0, /* inner_idx = */ 0,
                         /* upper_triangle = */ false);
  }

  // To implement end(), we create the element that will be obtained by
  // incrementing past the final reasonable edge output.
  EdgesIterator end() {
    return EdgesIterator(edges_, /* outer_idx = */ edges_.size(),
                         /* inner_idx = */ 0, /* upper_triangle = */ false);
  }

  // Returns a Triplet* with the address of a static Eigen::Triplet<bool> object
  // representing the current edge. This works in this context because
  // setFromTriplets() just reads from the iterators and doesn't store them for
  // later.
  const Eigen::Triplet<bool>& operator*() const {
    static Eigen::Triplet<bool> e;
    if (upper_triangle_) {
      e = Eigen::Triplet<bool>(edges_[outer_idx_][inner_idx_], outer_idx_,
                               true);
    } else {
      e = Eigen::Triplet<bool>(outer_idx_, edges_[outer_idx_][inner_idx_],
                               true);
    }
    return e;
  }

  const Eigen::Triplet<bool>* operator->() const { return &**this; }

  // Advance the iterator to the next edge.
  EdgesIterator& operator++() {
    /* Each edge in `edges` produces a lower- and upper-triplet. In that order.
    So, we flip the upper/lower bit on each advance. When flipping to lower,
    it's time to advance to the next edge in `edges`. */
    upper_triangle_ = !upper_triangle_;
    if (!upper_triangle_) {
      if (++inner_idx_ >= ssize(edges_[outer_idx_])) {
        ++outer_idx_;
        inner_idx_ = 0;
        skip_empty_outer();
      }
    }
    return *this;
  }

  // Compares two iterators.
  bool operator!=(const EdgesIterator& other) const {
    return outer_idx_ != other.outer_idx_ || inner_idx_ != other.inner_idx_ ||
           upper_triangle_ != other.upper_triangle_;
  }

 private:
  // If a row is empty, skip to the next non-empty row
  void skip_empty_outer() {
    while (outer_idx_ < ssize(edges_) && edges_[outer_idx_].empty()) {
      ++outer_idx_;
    }
  }

  // Iterators pointing to each vector in the list of vectors.
  const std::vector<std::vector<int>>& edges_;
  int outer_idx_{0};
  int inner_idx_{0};
  bool upper_triangle_{false};
};

bool CheckEdgeCollisionFreeWithConfigurationObstacles(
    const CollisionChecker& checker, const Eigen::VectorXd& q1, 
    const Eigen::VectorXd& q2, const ConvexSets* configuration_obstacles) 
    const {
  const double distance = checker.ComputeConfigurationDistance(q1, q2);
  const int num_steps = static_cast<int>(std::max(1.0, std::ceil(distance / 
      checker.edge_step_size())));
  for (int step = 0; step < num_steps; ++step) {
    const double ratio =
        static_cast<double>(step) / static_cast<double>(num_steps);
    const Eigen::VectorXd qinterp =
        checker.InterpolateBetweenConfigurations(q1, q2, ratio);
    if (!checker.CheckConfigCollisionFree(model_context, qinterp) || 
        (configuration_obstacles != nullptr && 
        std::any_of(iris_options.configuration_obstacles.begin(), 
            iris_options.configuration_obstacles.end(), 
            [&](const auto& configuration_obstacle){ 
              return configuration_obstacle->PointInSet(qinterp);
            }))) {
      return false;
    }
  }
  return true;
}

}  // namespace

Eigen::SparseMatrix<bool> VisibilityGraph(
    const CollisionChecker& checker,
    const Eigen::Ref<const Eigen::MatrixXd>& points,
    const Parallelism parallelize,
    const ConvexSets* configuration_obstacles = nullptr) {
  DRAKE_THROW_UNLESS(checker.plant().num_positions() == points.rows());

  const int num_points = points.cols();
  const int num_threads_to_use =
      checker.SupportsParallelChecking()
          ? std::min(parallelize.num_threads(),
                     checker.num_allocated_contexts())
          : 1;
  drake::log()->debug("Generating VisibilityGraph using {} threads",
                      num_threads_to_use);

  // Choose std::vector<uint8_t> as a thread-safe data structure for the
  // parallel evaluations.
  std::vector<uint8_t> points_free(num_points, 0x00);

  const auto point_check_work = [&](const int thread_num, const int64_t i) {
    // First, check collision against context obstacles using `checker`
    bool is_collision_free = checker.CheckConfigCollisionFree(points.col(i), 
                                                              thread_num);
    
    // If point is collision-free, then check against configuration obstacles
    if (is_collision_free && configuration_obstacles != nullptr) {
      for (int obstacle : *configuration_obstacles) {
        if (obstacle.distance_to(points.col(i)) < 
            configuration_space_margin) {
          is_collision_free = false;
          break;
        }
      }
    }

    // Set the value in points_free based on the checks
    points_free[i] = static_cast<uint8_t>(is_collision_free);
  };

  StaticParallelForIndexLoop(DegreeOfParallelism(num_threads_to_use), 0,
                             num_points, point_check_work,
                             ParallelForBackend::BEST_AVAILABLE);

  // Choose std::vector as a thread-safe data structure for the parallel
  // evaluations.
  std::vector<std::vector<int>> edges(num_points);

  const auto edge_check_work = [&](const int thread_num, const int64_t index) {
    const int i = static_cast<int>(index);
    if (points_free[i] > 0) {
      edges[i].push_back(i);
      for (int j = i + 1; j < num_points; ++j) {
        if (points_free[j] > 0 &&
            CheckEdgeCollisionFreeWithConfigurationObstacles(checker, 
                points.col(i), points.col(j), configuration_obstacles)) {
          edges[i].push_back(j);
        }
      }
    }
  };

  DynamicParallelForIndexLoop(DegreeOfParallelism(num_threads_to_use), 0,
                              num_points, edge_check_work,
                              ParallelForBackend::BEST_AVAILABLE);

  // Convert edges into the SparseMatrix format, using a custom iterator to
  // avoid explicitly copying the data into a list of Eigen::Triplet.
  Eigen::SparseMatrix<bool> mat(num_points, num_points);
  EdgesIterator edges_iterator(edges);
  mat.setFromTriplets(edges_iterator.begin(), edges_iterator.end());
  return mat;
}

}  // namespace planning
}  // namespace drake
