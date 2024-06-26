#pragma once

#include <limits>
#include <vector>

#include "drake/common/drake_copyable.h"
#include "drake/common/eigen_types.h"
#include "drake/examples/mass_spring_cloth/cloth_spring_model_params.h"
#include "drake/systems/framework/leaf_system.h"

namespace drake {
namespace examples {
namespace mass_spring_cloth {
/** A system that models a rectangular piece of cloth using as a model a
 structured grid of springs and dampers. All quantities are expressed in the
 world frame unless otherwise noted. The system consists of nx-by-ny mass
 particles with neighboring particles separated by a distance of h meters. The
 rectangular grid initially lies in the z=0 plane and is fixed at two corners at
 (0,0,0) and (0, Ly, 0), where Ly is the side length of the cloth in the
 y-direction and is equal to (ny-1)*h. Shearing springs are added to penalize
 shearing modes. The model is described in detail in [Provot 1995].

 The system has three physical parameters:

 - __k__: elastic stiffness of the springs in the unit of N/m, measures the
 elastic force generated by the spring per unit displacement from the rest
 length.
 - __d__: damping coefficient of the springs in the unit of Ns/m, measures the
 damping force generated by the spring per unit relative velocity of the two
 particles connected by the spring.
 - __gravity__: the gravitational constant, a (usually negative) scalar. The
 gravity experienced by the particles is (0,0,gravity).

 The dynamics of the system is described by:

       q̇ = v,
       Mv̇ = fe(q) + fd(q, v),

 where `fe` contains the elastic spring force and `fd` contains the
 dissipation terms.

 When the system is integrated discretely, the elastic force and gravity are
 integrated explicitly while the damping force is integrated implicitly. In
 particular, the discretization reads:

       Mv̂ = Mvⁿ + dt*fe(qⁿ),
       Mvⁿ⁺¹ = Mv̂ + dt*fd(qⁿ, vⁿ⁺¹),
       qⁿ⁺¹ = qⁿ + dt*vⁿ⁺¹.

 which is first order accurate, but similar in spirit to the scheme in [Bridson,
 2005]. One should be careful not to take too large a time step when using the
 discrete system because it is conditionally stable, and large time steps can
 lead to an unstable numerical solution.

 Note that the spring energy is finite and thus the particles can overlap in
 certain scenarios. For both the discrete and the continuous system, when two
 particles overlap, the state is invalid and the system will throw a
 `std::runtime_error` and cause the program to crash.

 The system has a single output port that provides the positions of the
 particles. The 3*i-th, 3*i+1-th, and 3*i+2-th entry describe the position of
 the i-th particle.

 Beware that this class is not thread-safe as it contains mutable data storing
 the state of the discrete solver. This mutable data has no effect on the
 simulation results and therefore it is not part of the system's states.

 @system
 name: ClothSpringModel
 output_ports:
 - particle_positions
 @endsystem

 <h3>References</h3>
 - [Bridson, 2005] Bridson, Robert, Sebastian
 Marino, and Ronald Fedkiw. "Simulation of clothing with folds and wrinkles."
 ACM SIGGRAPH 2005 Courses. 2005.
 - [Provot 1995] Provot, Xavier.
 "Deformation constraints in a mass-spring model to describe rigid cloth
 behaviour." Graphics interface. Canadian Information Processing Society, 1995.
*/
template <typename T>
class ClothSpringModel final : public systems::LeafSystem<T> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(ClothSpringModel)

  /** Creates a rectangular cloth with a grid of nx by ny particles.

   @param[in] nx   The number of particles in the x-direction.
   @param[in] ny   The number of particles in the y-direction.
   @param[in] h    The spacing (in meters) between neighboring particles.
   @param[in] dt   The duration (in seconds) of the discrete solver's time step.
                   If dt <= 0, the continuous solver will be used.
   @pre @p nx > 0.
   @pre @p ny > 0.
   @pre @p h > 0.
  */
  ClothSpringModel(int nx, int ny, T h, double dt);

  /** This returns nx * ny. */
  int num_particles() const { return num_particles_; }

  T h() const { return h_; }

  /** For discrete mode only: set the max number of iterations for the Conjugate
    Gradient solve for the implicit damping force. It has no effect on
    continuous mode. */
  void set_linear_solve_max_iterations(int max_iterations) {
    cg_.setMaxIterations(max_iterations);
  }

  /** For discrete mode only: set the accuracy for the Conjugate Gradient solve
    for the implicit damping force. It has no effect on continuous mode.
    @param[in] accuracy  The unit-less permissible relative error on velocity
    update. */
  void set_linear_solve_accuracy(T accuracy = 0.0001) {
    cg_.setTolerance(accuracy);
  }

 private:
  struct Spring {
    // Indices of the two particles connected by the spring.
    int particle0{};
    int particle1{};
    T rest_length{};
  };

  /* This function extracts the position/velocity/force corresponding to the
   particle indexed with particle_index from the vector of that quantity.
   All per-particle quantities are elements of R³ and are aligned in each of
   the quantity vectors.*/
  static Vector3<T> particle_state(int particle_index,
                                   const Eigen::Ref<const VectorX<T>>& vec) {
    const int p_index = particle_index * 3;
    return Vector3<T>{vec[p_index], vec[p_index + 1], vec[p_index + 2]};
  }

  /* The setter corresponding to particle_state. Set position/velocity/force the
   particle indexed with particle_index to the state parameter.
   All per-particle quantities are elements of R³ and are aligned in each of
   the quantity vectors.*/
  static void set_particle_state(int particle_index, const Vector3<T>& state,
                                 EigenPtr<VectorX<T>> vec) {
    DRAKE_ASSERT(vec != nullptr);
    const int p_index = particle_index * 3;
    (*vec)[p_index] = state(0);
    (*vec)[p_index + 1] = state(1);
    (*vec)[p_index + 2] = state(2);
  }

  /* Similar to set_particle_state, but add state into the corresponding
   position in vec without zeroing out the old value.
   All per-particle quantities are elements of R³ and are aligned in each of
   the quantity vectors.*/
  static void accumulate_particle_state(int particle_index,
                                        const Vector3<T>& state,
                                        EigenPtr<VectorX<T>> vec) {
    DRAKE_ASSERT(vec != nullptr);
    const int p_index = particle_index * 3;
    (*vec)[p_index] += state(0);
    (*vec)[p_index + 1] += state(1);
    (*vec)[p_index + 2] += state(2);
  }

  /* Initialize the positions of the particles to be in a rectangular grid of
    nx-by-ny particles with neighboring particles separated by h. The velocities
    are initialized to zero. */
  systems::BasicVector<T> InitializePositionAndVelocity() const;

  /* TODO(xuchenhan-tri) Expose the use_shearing_springs parameter in the
   constructor to give the user the ability to toggle the configuration. */
  /* Generates the connectivity of the mesh of springs. */
  void BuildConnectingSprings(bool use_shearing_springs);

  void CopyContinuousStateOut(const systems::Context<T>& context,
                              systems::BasicVector<T>* output) const;

  void CopyDiscreteStateOut(const systems::Context<T>& context,
                            systems::BasicVector<T>* output) const;

  void DoCalcTimeDerivatives(
      const systems::Context<T>& context,
      systems::ContinuousState<T>* derivatives) const override;

  void UpdateDiscreteState(const systems::Context<T>& context,
                           systems::DiscreteValues<T>* next_states) const;

  /* Calculates the spring (elastic and damping combined) forces given the
   context. This is only used for computing forces in the continuous
   integration. The values contained in forces should be set to zero outside
   this function if fresh values are required. */
  void AccumulateContinuousSpringForce(const systems::Context<T>& context,
                                       EigenPtr<VectorX<T>> forces) const;

  const ClothSpringModelParams<T>& GetParameters(
      const systems::Context<T>& context) const {
    return this->template GetNumericParameter<ClothSpringModelParams>(
        context, param_index_);
  }

  /* Calculates the elastic force from springs given the positions of the
   particles and add to the output elastic_force. The values contained in
   elastic_force should be set to zero outside this function if fresh values
   are required. */
  void AccumulateElasticForce(const ClothSpringModelParams<T>& param,
                              const Eigen::Ref<const VectorX<T>>& q,
                              EigenPtr<VectorX<T>> elastic_force) const;

  /* Calculates the damping force from springs given the positions and
   velocities of the particles and add to the output damping_force. The values
   contained in damping_force should be set to zero outside this function if
   fresh values are required. */
  void AccumulateDampingForce(const ClothSpringModelParams<T>& param,
                              const Eigen::Ref<const VectorX<T>>& q,
                              const Eigen::Ref<const VectorX<T>>& v,
                              EigenPtr<VectorX<T>> damping_force) const;

  /* Calculates the change in discrete velocity, dv = vⁿ⁺¹ - v̂, induced by the
   implicit damping force, where v̂ is the velocity after the contribution of
   elastic and gravity forces are added. This function overwrites the values
   in dv.

   CalcDiscreteDv solves the equation

        M * dv = f(qⁿ, vⁿ⁺¹) * dt,

   where f is the damping force, which is equivalent to

        M * dv = (f(qⁿ, v̂) + ∂f/∂v(qⁿ, v̂) * dv) * dt,

   because damping force is linear in v. Moving terms we end up with

        (M - ∂f/∂v(qⁿ, v̂)) * dv = f(qⁿ, v̂) * dt

   which we abbreviate as

        H * dv = f * dt.
   @pre q, f, and dv must be of the same size.
   */
  void CalcDiscreteDv(const ClothSpringModelParams<T>& param,
                      const VectorX<T>& q, VectorX<T>* f, VectorX<T>* dv) const;

  /*
  Apply Dirichlet boundary conditions to the two corners of the rectangular
  grid.
  */
  void ApplyDirichletBoundary(EigenPtr<VectorX<T>> state) const {
    set_particle_state(bottom_left_corner_, {0, 0, 0}, state);
    set_particle_state(top_left_corner_, {0, 0, 0}, state);
  }

  /* Customized throw to prevent invalid configuration of springs. */
  void ThrowIfInvalidSpringLength(const T& spring_length,
                                  const T& rest_length) const;

  /* Return the number of degrees of freedoms corresponding to positions. */
  int num_positions() const { return num_particles_ * 3; }

  /* Return the number of degrees of freedoms corresponding to velocities. */
  int num_velocities() const { return num_particles_ * 3; }

  /* Number of particles in the x direction. */
  const int nx_{};
  /* Number of particles in the y direction.*/
  const int ny_{};
  /* Total number of mass particles.*/
  const int num_particles_{};
  /* The distance between neighboring particles.*/
  const T h_{};
  /* The time period between discrete updates.*/
  const T dt_{};
  /* The index of the fixed particle at the bottom-left corner of the grid.*/
  const int bottom_left_corner_{};
  /* The index of the fixed particle at the top-left corner of the grid.*/
  const int top_left_corner_{};
  /* The starting index of the parameters of this system.*/
  int param_index_{};
  /* A list of springs in the system. Indexing does not matter here.*/
  std::vector<Spring> springs_;
  /* Pre-allocated H matrix to prevent reallocations.*/
  mutable Eigen::SparseMatrix<T> H_;
  /* We use a CG solver for the symmetric positive definite matrix in the linear
   solve. We use the Lower|Upper flag for better performance per Eigen Doc:
   https://eigen.tuxfamily.org/dox/classEigen_1_1ConjugateGradient.html
  */
  mutable Eigen::ConjugateGradient<Eigen::SparseMatrix<double>,
                                   Eigen::Lower | Eigen::Upper>
      cg_;
};
}  // namespace mass_spring_cloth
}  // namespace examples
}  // namespace drake
