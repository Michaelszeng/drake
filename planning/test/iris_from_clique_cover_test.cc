#include "drake/planning/iris_from_clique_cover.h"

#include <gtest/gtest.h>

#include "drake/common/find_resource.h"
#include "drake/common/ssize.h"
#include "drake/geometry/optimization/convex_set.h"
#include "drake/geometry/optimization/hpolyhedron.h"
#include "drake/geometry/optimization/vpolytope.h"
#include "drake/multibody/parsing/parser.h"
#include "drake/planning/coverage_checker_via_bernoulli_test.h"
#include "drake/planning/rejection_sampler.h"
#include "drake/planning/robot_diagram_builder.h"
#include "drake/planning/scene_graph_collision_checker.h"
#include "drake/planning/uniform_set_sampler.h"
#include "drake/systems/framework/diagram_builder.h"

namespace drake {
namespace planning {
namespace {
using Eigen::Vector2d;
using geometry::optimization::ConvexSets;
using geometry::optimization::HPolyhedron;
using geometry::optimization::IrisOptions;
using geometry::optimization::VPolytope;

GTEST_TEST(IrisFromCliqueCover, BoxWithCornerObstaclesTest) {
  /* A 1x1 box domain with a 0.45x0.45 obstacle in each corner. The free space
   * is a cross composed to the union of
   * Box1 with vertices (0.45,0), (0.45, 1), (0.55, 0), (0.55, 1)
   * and
   * Box2 with vertices (0, 0.45), (0, 0.55), (1, 0.45), (1, 0.55)
   */
  const HPolyhedron domain =
      HPolyhedron::MakeBox(Vector2d(0, 0), Vector2d(1, 1));
  ConvexSets obstacles;
  obstacles.emplace_back(
      HPolyhedron::MakeBox(Vector2d(0, 0), Vector2d(0.45, 0.45)));
  obstacles.emplace_back(
      HPolyhedron::MakeBox(Vector2d(0.55, 0), Vector2d(1, 0.45)));
  obstacles.emplace_back(
      HPolyhedron::MakeBox(Vector2d(0, 0.55), Vector2d(0.45, 1)));
  obstacles.emplace_back(
      HPolyhedron::MakeBox(Vector2d(0.55, 0.55), Vector2d(1, 1)));

  IrisFromCliqueCoverOptions options;
  options.num_builders = 2;
  options.num_points_per_coverage_check = 100;
  options.num_points_per_visibility_round = 100;
  std::vector<HPolyhedron> sets;
  IrisFromCliqueCover(obstacles, domain, options, &sets);

  EXPECT_EQ(ssize(sets), 2);

  // The sampling distribution for the domain of the configuration space.
  RandomGenerator generator{0};
  std::shared_ptr<PointSamplerBase> sampler =
      options.point_sampler.has_value()
          ? options.point_sampler.value()
          : std::make_shared<UniformSetSampler<HPolyhedron>>(domain, generator);

  const std::function<bool(const Eigen::Ref<const Eigen::VectorXd>&)>&
      reject_in_collision =
          [&obstacles](const Eigen::Ref<const Eigen::VectorXd>& sample) {
            for (const auto& obstacle : obstacles) {
              if (obstacle->PointInSet(sample)) {
                return true;
              }
            }
            return false;
          };

  // The rejection sampler for the C-Free.
  std::unique_ptr<PointSamplerBase> rejection_collision_sampler =
      std::make_unique<RejectionSampler>(sampler, reject_in_collision);

  double coverage_threshold{0.9};
  CoverageCheckerViaBernoulliTest coverage_checker{
      coverage_threshold,
      static_cast<int>(1e3) /* number of sampled points for test*/,
      std::move(rejection_collision_sampler)};

  ConvexSets abstract_sets;
  double computed_volume{0};
  for (const auto& set : sets) {
    abstract_sets.emplace_back(set.Clone());
    computed_volume +=
        set.CalcVolumeViaSampling(&generator,
                                  1e-4 /* Ask for high relative accuracy */)
            .volume;
  }
  EXPECT_TRUE(coverage_checker.CheckCoverage(abstract_sets));
  // The free space is 0.2 units of area. We want to achieve at least 90% of 0.2
  EXPECT_GE(computed_volume, 0.2 * coverage_threshold);
  // In reality we achieve slighlty more than 0.189 units of volume. Leave this
  // test to detect regressions.
  EXPECT_GE(computed_volume, 0.189);
}

const char boxes_in_2d_urdf[] = R"""(
<robot name="boxes">
  <link name="fixed">
    <collision name="right">
      <origin rpy="0 0 0" xyz="2 0 0"/>
      <geometry><box size="1 1 1"/></geometry>
    </collision>
    <collision name="left">
      <origin rpy="0 0 0" xyz="-2 0 0"/>
      <geometry><box size="1 1 1"/></geometry>
    </collision>
  </link>
  <joint name="fixed_link_weld" type="fixed">
    <parent link="world"/>
    <child link="fixed"/>
  </joint>
  <link name="movable">
    <collision name="center">
      <geometry><box size="1 1 1"/></geometry>
    </collision>
  </link>
  <link name="for_joint"/>
  <joint name="x" type="prismatic">
    <axis xyz="1 0 0"/>
    <limit lower="-2" upper="2"/>
    <parent link="world"/>
    <child link="for_joint"/>
  </joint>
  <joint name="y" type="prismatic">
    <axis xyz="0 1 0"/>
    <limit lower="-1" upper="1"/>
    <parent link="for_joint"/>
    <child link="movable"/>
  </joint>
</robot>
)""";


/* A movable sphere with fixed boxes in all corners.
┌─────┬───┬─────┐
│     │   │     │
│     │   │     │
├─────┘   └─────┤
│       o       │
├─────┐   ┌─────┤
│     │   │     │
│     │   │     │
└─────┴───┴─────┘ */
const char boxes_in_corners[] = R"""(
<robot name="boxes">
  <link name="fixed">
    <collision name="top_left">
      <origin rpy="0 0 0" xyz="-1 1 0"/>
      <geometry><box size="1 1 1"/></geometry>
    </collision>
    <collision name="top_right">
      <origin rpy="0 0 0" xyz="1 1 0"/>
      <geometry><box size="1 1 1"/></geometry>
    </collision>
    <collision name="bottom_left">
      <origin rpy="0 0 0" xyz="-1 -1 0"/>
      <geometry><box size="1 1 1"/></geometry>
    </collision>
    <collision name="bottom_right">
      <origin rpy="0 0 0" xyz="1 -1 0"/>
      <geometry><box size="1 1 1"/></geometry>
    </collision>
  </link>
  <joint name="fixed_link_weld" type="fixed">
    <parent link="world"/>
    <child link="fixed"/>
  </joint>
  <link name="movable">
    <collision name="sphere">
      <geometry><sphere radius="0.1"/></geometry>
    </collision>
  </link>
  <link name="for_joint"/>
  <joint name="x" type="prismatic">
    <axis xyz="1 0 0"/>
    <limit lower="-2" upper="2"/>
    <parent link="world"/>
    <child link="for_joint"/>
  </joint>
  <joint name="y" type="prismatic">
    <axis xyz="0 1 0"/>
    <limit lower="-2" upper="2"/>
    <parent link="for_joint"/>
    <child link="movable"/>
  </joint>
</robot>
)""";

GTEST_TEST(IrisInConfigurationSpaceFromCliqueCover,
           BoxWithCornerObstaclesTest) {
  CollisionCheckerParams params;

  RobotDiagramBuilder<double> builder(0.0);
//  builder.parser().package_map().AddPackageXml(FindResourceOrThrow(
//      "drake/multibody/parsing/test/box_package/package.xml"));
//  params.robot_model_instances =
//      builder.parser().AddModelsFromString(boxes_in_2d_urdf, "urdf");
params.robot_model_instances =
      builder.parser().AddModelsFromString(boxes_in_corners, "urdf");
  params.edge_step_size = 0.01;

  params.model = builder.Build();
  auto checker =
      std::make_unique<SceneGraphCollisionChecker>(std::move(params));

  IrisFromCliqueCoverOptions options;

//  ConvexSets obstacles;
//  obstacles.emplace_back(VPolytope::MakeBox(Vector2d(.2, .2), Vector2d(1, 1)));
//  options.iris_options.configuration_obstacles = obstacles;
  options.num_builders = 2;
  options.num_points_per_coverage_check = 100;
  options.num_points_per_visibility_round = 25;
  std::vector<HPolyhedron> sets;

  IrisInConfigurationSpaceFromCliqueCover(*checker, options, &sets);
  EXPECT_GE(ssize(sets), 2);
}
}  // namespace
}  // namespace planning
}  // namespace drake
