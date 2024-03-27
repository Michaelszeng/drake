#pragma once

#include <Eigen/Sparse>

#include "drake/common/parallelism.h"
#include "drake/planning/collision_checker.h"

namespace drake {
namespace planning {

/** Given some number of sampled points in the configuration space of
`checker`'s plant(), computes the "visibility graph" -- two `points` have an
edge between them if the line segment connecting them is collision free. See
CollisionChecker documentation for more information on how edge collision
checks are performed.

Note that this method assumes the collision checker has symmetric behavior
(i.e. checking edge (q1, q2) is the same as checking edge (q2, q1)). This is
true for many collision checkers (e.g. those using
LinearDistanceAndInterpolationProvider, which is the default), but some more
complex spaces with non-linear interpolation (e.g. a Dubin's car) are not
symmetric.

If `parallelize` specifies more than one thread, then the
CollisionCheckerParams::distance_and_interpolation_provider for `checker` must
be implemented in C++, either by providing the C++ implementation directly
directly or by using the default provider.

@returns the adjacency matrix, A(i,j) == true iff points.col(i) is visible from
points.col(j). A is always symmetric.

@pre points.rows() == total number of positions in the collision checker plant.
*/
Eigen::SparseMatrix<bool> VisibilityGraph(
    const CollisionChecker& checker,
    const Eigen::Ref<const Eigen::MatrixXd>& points,
    Parallelism parallelize = Parallelism::Max());

/** A more flexible variant of the function VisibilityGraph() above. This method 
also computes the "visibility graph" given a number of sampled points, but 
accepts addtional lambda function parameters that allow you to define custom 
methods for collision checking (for example, if you want to check collisions 
with user-defined configuration obstacles, as the CollisionChecker is only aware
of context obstacles).

ConfigurableVisibilityGraph will call `point_check_work` once for each point in 
`points` to determine whether that point is in collision. It takes parameters 
`const int thread_num`, `const int64_t i`, and 
`std::vector<uint8_t>* points_free`. `points_free[i]` is passed by pointer and 
should be modified to contain a binary value 1 if the point is collision-free, 
or a 0 otherwise. `thread_num` can be used for parallelization (i.e. as a 
`context_number` for CollisionChecker).

Similarly, ConfigurableVisibilityGraph calls `edge_check_work` once for each
point in `points_free`. `edge_check_work` takes parameters 
`const int thread_num`, `const int64_t i`, 
`const std::vector<uint8_t>& points_free`, `const int num_points`, and 
`std::vector<std::vector<int>>* edges`. It should check collisions along each 
edge from `points_free[i]` to `points_free[j]` for all `j` with
`points_free[j] = 1`, and append the value `j` to `edges[i]` if that edge is
collision-free. `thread_num` can be used for parallelization (i.e. as a 
`context_number` for CollisionChecker).

If `parallelize` specifies more than one thread, then the
CollisionCheckerParams::distance_and_interpolation_provider for `checker` must
be implemented in C++, either by providing the C++ implementation directly
directly or by using the default provider.

@returns the adjacency matrix, A(i,j) == true iff points.col(i) is visible from
points.col(j). A is always symmetric.

@pre points.rows() == total number of positions in the collision checker plant.
*/
Eigen::SparseMatrix<bool> ConfigurableVisibilityGraph(
    std::function<void(const int, const int64_t,  
        std::vector<uint8_t>*)> point_check_work,
    std::function<void(const int, const int64_t,  
        const std::vector<uint8_t>&, const int, 
        std::vector<std::vector<int>>*)> edge_check_work,
    const CollisionChecker& checker,
    const Eigen::Ref<const Eigen::MatrixXd>& points,
    const Parallelism parallelize);

}  // namespace planning
}  // namespace drake
