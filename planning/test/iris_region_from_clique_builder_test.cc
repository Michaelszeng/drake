#include "drake/planning/iris_region_from_clique_builder.h"

#include <gtest/gtest.h>

#include "drake/common/find_resource.h"
#include "drake/common/ssize.h"
#include "drake/geometry/optimization/convex_set.h"
#include "drake/geometry/optimization/hpolyhedron.h"
#include "drake/geometry/optimization/vpolytope.h"
#include "drake/multibody/parsing/parser.h"
#include "drake/systems/framework/diagram_builder.h"

namespace drake {
namespace planning {
namespace {
using Eigen::Vector2d;
using geometry::optimization::ConvexSets;
using geometry::optimization::HPolyhedron;
using geometry::optimization::IrisOptions;
using geometry::optimization::VPolytope;

GTEST_TEST(IrisRegionFromCliqueBuilder, TestCtorSettersAndGetters) {
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

  IrisOptions options{};
  // Change one of the default fields so we can be sure the options are set
  // properly.
  options.iteration_limit = 2;

  const double rank_tol_for_lowner_john_ellipse = 1e-6;

  IrisRegionFromCliqueBuilder builder{obstacles, domain, options,
                                      rank_tol_for_lowner_john_ellipse};
  EXPECT_EQ(obstacles.size(), builder.get_obstacles().size());
  EXPECT_EQ(domain.A(), builder.get_domain().A());
  EXPECT_EQ(domain.b(), builder.get_domain().b());
  EXPECT_EQ(options.iteration_limit, builder.get_options().iteration_limit);
  EXPECT_EQ(rank_tol_for_lowner_john_ellipse,
            builder.get_rank_tol_for_lowner_john_ellipse());

  const HPolyhedron domain2 =
      HPolyhedron::MakeBox(Vector2d(0, 0), Vector2d(1, 1));
  builder.set_domain(domain2);
  EXPECT_EQ(domain2.A(), builder.get_domain().A());

  ConvexSets obstacles2;
  obstacles.emplace_back(
      HPolyhedron::MakeBox(Vector2d(0, 0), Vector2d(0.45, 0.45)));
  obstacles.emplace_back(
      HPolyhedron::MakeBox(Vector2d(0.55, 0), Vector2d(1, 0.45)));
  builder.set_obstacles(obstacles2);
  EXPECT_EQ(obstacles2.size(), builder.get_obstacles().size());

  IrisOptions options2{};
  options2.iteration_limit = 4;
  builder.set_options(options2);
  EXPECT_EQ(options2.iteration_limit, builder.get_options().iteration_limit);

  const double rank_tol_for_lowner_john_ellipse2 = 1e-10;
  builder.set_rank_tol_for_lowner_john_ellipse(
      rank_tol_for_lowner_john_ellipse2);
  EXPECT_EQ(rank_tol_for_lowner_john_ellipse2,
            builder.get_rank_tol_for_lowner_john_ellipse());
}

GTEST_TEST(IrisRegionFromCliqueBuilder, TestCtorDefaults) {
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
  IrisRegionFromCliqueBuilder builder{obstacles, domain};
  EXPECT_EQ(builder.get_options().iteration_limit, 1);
  EXPECT_EQ(builder.get_rank_tol_for_lowner_john_ellipse(), 1e-6);
}

GTEST_TEST(IrisRegionFromCliqueBuilder, TestBuildConvexSet) {
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

  IrisRegionFromCliqueBuilder builder{obstacles, domain};

  const Eigen::Matrix<double, 2, 4> clique1{{0.45, 0.45, 0.55, 0.55},
                                            {0.0, 1.0, 0.0, 1.0}};
  std::unique_ptr<ConvexSet> set1_base = builder.BuildConvexSet(clique1);
  const HPolyhedron* set1 = dynamic_cast<const HPolyhedron*>(set1_base.get());
  // The convex hull of these points is approximately Box1.
  const Eigen::Matrix<double, 2, 4> test_points1{{0.46, 0.46, 0.54, 0.54},
                                                 {0.01, 0.99, 0.01, 0.99}};
  for (int i = 0; i < test_points1.cols(); ++i) {
    EXPECT_TRUE(set1->PointInSet(test_points1.col(i), 1e-6));
  }

  const Eigen::Matrix<double, 2, 4> clique2{{0.0, 1.0, 0.0, 1.0},
                                            {0.45, 0.45, 0.55, 0.55}};
  std::unique_ptr<ConvexSet> set2_base = builder.BuildConvexSet(clique2);
  const HPolyhedron* set2 = dynamic_cast<const HPolyhedron*>(set2_base.get());
  // The convex hull of these points is approximately Box2.
  const Eigen::Matrix<double, 2, 4> test_points2{{0.01, 0.99, 0.01, 0.99},
                                                 {0.46, 0.46, 0.54, 0.54}};
  for (int i = 0; i < test_points2.cols(); ++i) {
    EXPECT_TRUE(set2->PointInSet(test_points2.col(i), 1e-6));
  }
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

GTEST_TEST(IrisInConfigurationSpaceRegionFromCliqueBuilder,
           TestCtorSettersAndGetters) {
  systems::DiagramBuilder<double> builder;
  multibody::MultibodyPlant<double>& plant =
      multibody::AddMultibodyPlantSceneGraph(&builder, 0.0);
  multibody::Parser parser(&plant);
  parser.package_map().AddPackageXml(FindResourceOrThrow(
      "drake/multibody/parsing/test/box_package/package.xml"));
  parser.AddModelsFromString(boxes_in_2d_urdf, "urdf");
  plant.Finalize();
  auto diagram = builder.Build();

  auto context = diagram->CreateDefaultContext();

  IrisOptions options{};
  IrisInConfigurationSpaceRegionFromCliqueBuilder region_builder(
      plant, *context, options, 1e-6);

  const double rank_tol_for_lowner_john_ellipse2 = 1e-10;
  region_builder.set_rank_tol_for_lowner_john_ellipse(
      rank_tol_for_lowner_john_ellipse2);
  EXPECT_EQ(rank_tol_for_lowner_john_ellipse2,
            region_builder.get_rank_tol_for_lowner_john_ellipse());

  IrisOptions options2{};
  options2.iteration_limit = 4;
  region_builder.set_options(options2);
  EXPECT_EQ(options2.iteration_limit,
            region_builder.get_options().iteration_limit);
}

/* Box obstacles in one corner.
┌───────┬─────┐
│       │     │
│       │     │
│       └─────┤
│      *      │
│             │
│             │
│             │
└─────────────┘
We use only a single configuration obstacle, and verify the the computed
halfspace changes.
*/
GTEST_TEST(IrisInConfigurationSpaceRegionFromCliqueBuilder,
           TestBuildConvexSet) {
  systems::DiagramBuilder<double> builder;
  multibody::MultibodyPlant<double>& plant =
      multibody::AddMultibodyPlantSceneGraph(&builder, 0.0);
  multibody::Parser parser(&plant);
  parser.package_map().AddPackageXml(FindResourceOrThrow(
      "drake/multibody/parsing/test/box_package/package.xml"));
  parser.AddModelsFromString(boxes_in_2d_urdf, "urdf");
  plant.Finalize();
  auto diagram = builder.Build();

  auto context = diagram->CreateDefaultContext();

  IrisOptions options{};
  ConvexSets obstacles;
  obstacles.emplace_back(VPolytope::MakeBox(Vector2d(.2, .2), Vector2d(1, 1)));
  options.configuration_obstacles = obstacles;

  IrisInConfigurationSpaceRegionFromCliqueBuilder region_builder(
      plant, *context, options, 1e-6);

  // A clique spread along the y direction.
  const Eigen::Matrix<double, 2, 4> clique1{{0.75, 0.75, 0.125, 0.125},
                                            {0.15, 0.85, 0.15, 0.85}};
  // The convex hull of these points is approximately the long side of the box.
  const Eigen::Matrix<double, 2, 4> test_points1{{0.05, 0.05, 0.19, 0.19},
                                                 {0.1, 0.9, 0.1, 0.9}};
  // A clique spread along the x direction.
  const Eigen::Matrix<double, 2, 4> clique2{{0.1, 0.1, 0.9, 0.9},
                                            {0.1, 0.2, 0.1, 0.2}};
  const Eigen::Matrix<double, 2, 4> test_points2{{0.05, 0.05, 0.96, 0.96},
                                                 {0.1, 0.3, 0.1, 0.3}};

  std::unique_ptr<ConvexSet> set1_base = region_builder.BuildConvexSet(clique1);
  const HPolyhedron* set1 = dynamic_cast<const HPolyhedron*>(set1_base.get());
  for (int i = 0; i < test_points1.cols(); ++i) {
    EXPECT_TRUE(set1->PointInSet(test_points1.col(i), 1e-6));
  }
  bool test_points2_not_covered_by_set1 = false;
  for (int i = 0; i < test_points2.cols(); ++i) {
    if(!set1->PointInSet(test_points2.col(i), 1e-6)) {
        test_points2_not_covered_by_set1 = true;
    }
  }
  EXPECT_TRUE(test_points2_not_covered_by_set1);


  std::unique_ptr<ConvexSet> set2_base = region_builder.BuildConvexSet(clique2);
  const HPolyhedron* set2 = dynamic_cast<const HPolyhedron*>(set2_base.get());
  for (int i = 0; i < test_points2.cols(); ++i) {
    EXPECT_TRUE(set2->PointInSet(test_points2.col(i), 1e-6));
  }
  bool test_points1_not_covered_by_set2 = false;
  for (int i = 0; i < test_points1.cols(); ++i) {
    if(!set2->PointInSet(test_points1.col(i), 1e-6)) {
        test_points1_not_covered_by_set2 = true;
    }
  }
  EXPECT_TRUE(test_points1_not_covered_by_set2);
}

}  // namespace
}  // namespace planning
}  // namespace drake
