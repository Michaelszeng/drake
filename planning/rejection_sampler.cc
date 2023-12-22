#include "drake/planning/rejection_sampler.h"

#include <memory>
#include <utility>
#include <iostream>
namespace drake {
namespace planning {

RejectionSampler::RejectionSampler(
    std::shared_ptr<PointSamplerBase> sampler,
    std::function<bool(const Eigen::Ref<const Eigen::VectorXd>&)> rejection_fun)
    : sampler_{std::move(sampler)}, rejection_fun_{std::move(rejection_fun)} {}

Eigen::MatrixXd RejectionSampler::DoSamplePoints(int num_points) {
  Eigen::VectorXd sample = sampler_->SamplePoints(1);
  Eigen::MatrixXd ret(sample.rows(), num_points);
  int points_added{0};
  while (points_added < num_points) {
    if (!rejection_fun_(sample)) {
      ret.col(points_added) = sample;
      ++points_added;
    }
    sample = sampler_->SamplePoints(1);
  }
  return ret;
}

}  // namespace planning
}  // namespace drake
