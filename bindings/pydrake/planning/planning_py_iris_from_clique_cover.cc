#include "drake/bindings/pydrake/common/cpp_template_pybind.h"
#include "drake/bindings/pydrake/common/wrap_function.h"
#include "drake/bindings/pydrake/documentation_pybind.h"
#include "drake/bindings/pydrake/geometry/optimization_pybind.h"
#include "drake/bindings/pydrake/pydrake_pybind.h"
#include "drake/geometry/optimization/hyperrectangle.h"
#include "drake/planning/adjacency_matrix_builder_base.h"
#include "drake/planning/approximate_convex_cover_builder_base.h"
#include "drake/planning/convex_set_from_clique_builder_base.h"
#include "drake/planning/coverage_checker_base.h"
#include "drake/planning/coverage_checker_via_bernoulli_test.h"
#include "drake/planning/iris_from_clique_cover.h"
#include "drake/planning/point_sampler_base.h"
#include "drake/planning/rejection_sampler.h"
#include "drake/planning/uniform_set_sampler.h"
#include "drake/planning/visibility_graph_builder.h"

namespace drake {
namespace pydrake {
namespace internal {

void DefinePlanningIrisFromCliqueCover(py::module m) {
  // NOLINTNEXTLINE(build/namespaces): Emulate placement in namespace.
  using namespace drake::planning;
  constexpr auto& doc = pydrake_doc.drake.planning;

  // PointSamplerBase, UniformSetSampler, RejectionSampler
  {
    class PyPointSamplerBase : public py::wrapper<PointSamplerBase> {
     public:
      // Trampoline virtual methods.
      // The private virtual method of DoSamplePoints is made public to enable
      // Python implementations to override it.
      Eigen::MatrixXd DoSamplePoints(int num_points) override {
        /* Acquire GIL before calling Python code */
        py::gil_scoped_acquire acquire;
        PYBIND11_OVERRIDE_PURE(
            Eigen::MatrixXd, PointSamplerBase, DoSamplePoints, num_points);
      }
    };
    const auto& cls_doc = doc.PointSamplerBase;
    // Use non-default holder type due to use of shared pointers of
    // PointSamplerBase being used throughout iris_from_clique_cover.
    py::class_<PointSamplerBase, std::shared_ptr<PointSamplerBase>,
        PyPointSamplerBase>(m, "PointSamplerBase", cls_doc.doc)
        .def(py::init<>(), cls_doc.ctor.doc)
        .def("SamplePoints", &PointSamplerBase::SamplePoints,
            py::arg("num_points"), cls_doc.SamplePoints.doc);
  }
  {
    const auto& cls_doc = doc.UniformSetSampler;
    using geometry::optimization::HPolyhedron;
    using Class = UniformSetSampler<HPolyhedron>;
    // Use non-default holder type due to use of shared pointers of
    // PointSamplerBase being used throughout iris_from_clique_cover.
    py::class_<Class, std::shared_ptr<Class>, PointSamplerBase>(
        m, "UniformHPolyhedronSampler", cls_doc.doc)
        .def(py::init<const HPolyhedron&>(), py::arg("set"),
            cls_doc.ctor.doc_1args)
        .def(py::init<const HPolyhedron&, const RandomGenerator&>(),
            py::arg("set"), py::arg("generator"), cls_doc.ctor.doc_2args)
        .def("Set", &Class::Set, cls_doc.Set.doc);
  }
  {
    const auto& cls_doc = doc.UniformSetSampler;
    using geometry::optimization::Hyperrectangle;
    using Class = UniformSetSampler<Hyperrectangle>;
    // Use non-default holder type due to use of shared pointers of
    // PointSamplerBase being used throughout iris_from_clique_cover.
    py::class_<Class, std::shared_ptr<Class>, PointSamplerBase>(
        m, "UniformHyperrectangleSampler", cls_doc.doc)
        .def(py::init<const Hyperrectangle&>(), py::arg("set"),
            cls_doc.ctor.doc_1args)
        .def(py::init<const Hyperrectangle&, const RandomGenerator&>(),
            py::arg("set"), py::arg("generator"), cls_doc.ctor.doc_2args)
        .def("Set", &Class::Set, cls_doc.Set.doc);
  }
  {
    // Use non-default holder type due to use of shared pointers of
    // PointSamplerBase being used throughout iris_from_clique_cover.
    const auto& cls_doc = doc.RejectionSampler;
    py::class_<RejectionSampler, PointSamplerBase,
        std::shared_ptr<RejectionSampler>>(m, "RejectionSampler", cls_doc.doc)
        .def(py::init<std::shared_ptr<PointSamplerBase>,
                 std::function<bool(
                     const Eigen::Ref<const Eigen::VectorXd>&)>>(),
            py::arg("sampler"), py::arg("rejection_fun"), cls_doc.ctor.doc);
  }

  // CoverageCheckerBase and CoverageCheckerViaBernoulliTest
  {
    class PyCoverageCheckerBase : public py::wrapper<CoverageCheckerBase> {
     public:
      // Trampoline virtual methods.
      // The private virtual method of DoCheckCoverage is made public to enable
      // Python implementations to override it.
      bool DoCheckCoverage(const ConvexSets& current_sets) const override {
        // Since DoCheckCoverage in the derived Python classes cannot assume
        // ownership of the unique pointers, we need to pass Python the raw
        // pointers.
        std::vector<const geometry::optimization::ConvexSet*> sets;
        sets.reserve(ssize(current_sets));
        for (const auto& set : current_sets) {
          sets.push_back(set.get());
        }
        /* Acquire GIL before calling Python code */
        py::gil_scoped_acquire acquire;
        PYBIND11_OVERRIDE_PURE(
            bool, CoverageCheckerBase, DoCheckCoverage, sets);
      }
    };
    const auto& cls_doc = doc.CoverageCheckerBase;
    py::class_<CoverageCheckerBase, PyCoverageCheckerBase>(
        m, "CoverageCheckerBase", cls_doc.doc)
        .def(py::init<>(), cls_doc.ctor.doc)
        .def(
            "CheckCoverage",
            [](CoverageCheckerBase& self,
                const std::vector<geometry::optimization::ConvexSet*>& sets) {
              return self.CheckCoverage(CloneConvexSets(sets));
            },
            py::arg("sets"), cls_doc.CheckCoverage.doc);
  }
  {
    const auto& cls_doc = doc.CoverageCheckerViaBernoulliTest;
    py::class_<CoverageCheckerViaBernoulliTest, CoverageCheckerBase>(
        m, "CoverageCheckerViaBernoulliTest", cls_doc.doc)
        .def(py::init<const double, const int,
                 std::shared_ptr<PointSamplerBase>, int, double>(),
            py::arg("alpha"), py::arg("num_points_per_check"),
            py::arg("sampler"), py::arg("num_threads") = -1,
            py::arg("point_in_set_tol") = 1e-8, cls_doc.ctor.doc)
        .def("get_alpha", &CoverageCheckerViaBernoulliTest::get_alpha,
            cls_doc.get_alpha.doc)
        .def("set_alpha", &CoverageCheckerViaBernoulliTest::set_alpha,
            py::arg("alpha"), cls_doc.set_alpha.doc)
        .def("get_num_points_per_check",
            &CoverageCheckerViaBernoulliTest::get_num_points_per_check,
            cls_doc.get_num_points_per_check.doc)
        .def("set_num_points_per_check",
            &CoverageCheckerViaBernoulliTest::set_num_points_per_check,
            py::arg("num_points_per_check"),
            cls_doc.set_num_points_per_check.doc)
        .def("get_point_in_set_tol",
            &CoverageCheckerViaBernoulliTest::get_point_in_set_tol,
            cls_doc.get_point_in_set_tol.doc)
        .def("set_point_in_set_tol",
            &CoverageCheckerViaBernoulliTest::set_point_in_set_tol,
            py::arg("point_in_set_tol"), cls_doc.set_point_in_set_tol.doc)
        .def("get_num_threads",
            &CoverageCheckerViaBernoulliTest::get_num_threads,
            cls_doc.get_num_threads.doc)
        .def("set_num_threads",
            &CoverageCheckerViaBernoulliTest::set_num_threads,
            py::arg("num_threads"), cls_doc.set_num_threads.doc)
        .def(
            "GetSampledCoverageFraction",
            [](CoverageCheckerViaBernoulliTest& self,
                const std::vector<geometry::optimization::ConvexSet*>& sets) {
              return self.GetSampledCoverageFraction(CloneConvexSets(sets));
            },
            py::arg("sets"), cls_doc.GetSampledCoverageFraction.doc);
  }

  // AdjacencyMatrixBuilderBase
  {
    class PyAdjacencyMatrixBuilderBase
        : public py::wrapper<AdjacencyMatrixBuilderBase> {
     public:
      // Trampoline virtual methods.
      // The private virtual method of DoSamplePoints is made public to enable
      // Python implementations to override it.
      Eigen::SparseMatrix<bool> DoBuildAdjacencyMatrix(
          const Eigen::Ref<const Eigen::MatrixXd>& points) const override {
        /* Acquire GIL before calling Python code */
        py::gil_scoped_acquire acquire;
        PYBIND11_OVERRIDE_PURE(Eigen::SparseMatrix<bool>,
            AdjacencyMatrixBuilderBase, DoBuildAdjacencyMatrix, points);
      }
    };
    const auto& cls_doc = doc.AdjacencyMatrixBuilderBase;

    py::class_<AdjacencyMatrixBuilderBase, PyAdjacencyMatrixBuilderBase>(
        m, "AdjacencyMatrixBuilderBase", cls_doc.doc)
        .def(py::init<>(), cls_doc.ctor.doc)
        .def("BuildAdjacencyMatrix",
            &AdjacencyMatrixBuilderBase::BuildAdjacencyMatrix,
            py::arg("points"), cls_doc.BuildAdjacencyMatrix.doc);
  }
  {
    const auto& cls_doc = doc.VisibilityGraphBuilder;
    py::class_<VisibilityGraphBuilder, AdjacencyMatrixBuilderBase>(
        m, "VisibilityGraphBuilder", cls_doc.doc)
        .def(py::init<std::shared_ptr<CollisionChecker>, bool>(),
            py::arg("checker"), py::arg("parallelize") = true,
            cls_doc.ctor.doc);
  }

  // ConvexSetFromCliqueBuilderBase and IrisRegionFromCliqueBuilder
  {
    class PyConvexSetFromCliqueBuilderBase
        : public py::wrapper<ConvexSetFromCliqueBuilderBase> {
     public:
      // Trampoline virtual methods.
      // The private virtual method of DoBuildConvexSet is made public to enable
      // Python implementations to override it.
      std::unique_ptr<ConvexSet> DoBuildConvexSet(
          const Eigen::Ref<const Eigen::MatrixXd>& clique_points) override {
        /* Release the GIL */
        py::gil_scoped_release release;
        {
          /* Acquire GIL before calling Python code */
          py::gil_scoped_acquire acquire;
          PYBIND11_OVERRIDE_PURE(std::unique_ptr<ConvexSet>,
              ConvexSetFromCliqueBuilderBase, DoBuildConvexSet, clique_points);
        }
      }
    };
    const auto& cls_doc = doc.ConvexSetFromCliqueBuilderBase;
    py::class_<ConvexSetFromCliqueBuilderBase,
        PyConvexSetFromCliqueBuilderBase>(
        m, "ConvexSetFromCliqueBuilderBase", cls_doc.doc)
        .def(py::init<>(), cls_doc.ctor.doc)
        .def("BuildConvexSet", &ConvexSetFromCliqueBuilderBase::BuildConvexSet,
            cls_doc.BuildConvexSet.doc);
  }
  {
    py::class_<DefaultIrisOptionsForIrisRegionFromCliqueBuilder>(m,
        "DefaultIrisOptionsForIrisRegionFromCliqueBuilder",
        doc.DefaultIrisOptionsForIrisRegionFromCliqueBuilder.doc);
    const auto& cls_doc = doc.IrisRegionFromCliqueBuilder;
    using Class = IrisRegionFromCliqueBuilder;
    py::class_<Class, ConvexSetFromCliqueBuilderBase>(
        m, "IrisRegionFromCliqueBuilder", cls_doc.doc)
        .def(py::init([](const std::vector<ConvexSet*>& obstacles,
                          const HPolyhedron& domain, const IrisOptions& options,
                          const double rank_tol_for_lowner_john_ellipse) {
          return Class(CloneConvexSets(obstacles), domain, options,
              rank_tol_for_lowner_john_ellipse);
        }),
            py::arg("obstacles"), py::arg("domain"),
            py::arg("options") =
                DefaultIrisOptionsForIrisRegionFromCliqueBuilder(),
            py::arg("rank_tol_for_lowner_john_ellipse") = 1e-6,
            cls_doc.ctor.doc)
        .def(
            "get_obstacles",
            [](const Class& self) {
              std::vector<std::unique_ptr<geometry::optimization::ConvexSet>>
                  ret;
              ret.reserve(self.get_obstacles().size());
              for (const auto& set : self.get_obstacles()) {
                ret.emplace_back(set->Clone());
              }
              return ret;
            },
            cls_doc.get_obstacles.doc)
        .def(
            "set_obstacles",
            [](Class& self,
                const std::vector<geometry::optimization::ConvexSet*>& sets) {
              return self.set_obstacles(CloneConvexSets(sets));
            },
            py::arg("obstacles"), cls_doc.set_obstacles.doc)
        .def("get_domain", &Class::get_domain, cls_doc.get_domain.doc)
        .def("set_domain", &Class::set_domain, py::arg("domain"),
            cls_doc.set_domain.doc)
        .def("get_options", &Class::get_options, cls_doc.get_options.doc)
        .def("set_options", &Class::set_options, py::arg("options"),
            cls_doc.set_options.doc)
        .def("get_rank_tol_for_lowner_john_ellipse",
            &Class::get_rank_tol_for_lowner_john_ellipse,
            cls_doc.get_rank_tol_for_lowner_john_ellipse.doc)
        .def("set_rank_tol_for_lowner_john_ellipse",
            &Class::set_rank_tol_for_lowner_john_ellipse,
            py::arg("rank_tol_for_lowner_john_ellipse"),
            cls_doc.set_rank_tol_for_lowner_john_ellipse.doc);
  }
  // ApproximateConvexCoverFromCliqueCover
  {
    auto cls_doc = doc.ApproximateConvexCoverFromCliqueCoverOptions;
    py::class_<ApproximateConvexCoverFromCliqueCoverOptions>(
        m, "ApproximateConvexCoverFromCliqueCoverOptions", cls_doc.doc)
        .def(py::init<>())
        .def_readwrite("num_sampled_points",
            &ApproximateConvexCoverFromCliqueCoverOptions::num_sampled_points,
            cls_doc.num_sampled_points.doc)
        .def_readwrite("minimum_clique_size",
            &ApproximateConvexCoverFromCliqueCoverOptions::minimum_clique_size,
            cls_doc.minimum_clique_size.doc);
    m.def(
        "ApproximateConvexCoverFromCliqueCover",
        [](CoverageCheckerBase* coverage_checker,
            PointSamplerBase* point_sampler,
            AdjacencyMatrixBuilderBase* adjacency_matrix_builder,
            MaxCliqueSolverBase* max_clique_solver,
            const std::vector<std::unique_ptr<ConvexSetFromCliqueBuilderBase>>&
                set_builders,
            const ApproximateConvexCoverFromCliqueCoverOptions& options) {
          ConvexSets convex_sets;
          ApproximateConvexCoverFromCliqueCover(coverage_checker, point_sampler,
              adjacency_matrix_builder, max_clique_solver, set_builders,
              options, &convex_sets);
          std::vector<std::unique_ptr<ConvexSet>> returned_sets;
          returned_sets.reserve(ssize(convex_sets));
          for (auto& set : convex_sets) {
            returned_sets.emplace_back(std::move(set));
          }
          return returned_sets;
        },
        py::arg("coverage_checker"), py::arg("point_sampler"),
        py::arg("adjacency_matrix_builder"), py::arg("max_clique_solver"),
        py::arg("set_builders"), py::arg("options"),
        doc.ApproximateConvexCoverFromCliqueCover.doc);
  }
  {
    auto cls_doc = doc.IrisFromCliqueCoverOptions;
    py::class_<IrisFromCliqueCoverOptions>(
        m, "IrisFromCliqueCoverOptions", cls_doc.doc)
        .def(py::init<>())
        // A Python specific constructor to allow the clique cover solver to be set.
        .def(py::init([](std::unique_ptr<MaxCliqueSolverBase> solver){
          IrisFromCliqueCoverOptions ret{};
          ret.max_clique_solver = std::move(solver);
          return ret;
        }), py::arg("solver"))
        .def_readwrite("iris_options",
            &IrisFromCliqueCoverOptions::iris_options, cls_doc.iris_options.doc)
        .def_readwrite("coverage_termination_threshold",
            &IrisFromCliqueCoverOptions::coverage_termination_threshold,
            cls_doc.coverage_termination_threshold.doc)
        .def_readwrite("num_points_per_coverage_check",
            &IrisFromCliqueCoverOptions::num_points_per_coverage_check,
            cls_doc.num_points_per_coverage_check.doc)
        .def_readwrite("minimum_clique_size",
            &IrisFromCliqueCoverOptions::minimum_clique_size,
            cls_doc.minimum_clique_size.doc)
        .def_readwrite("num_points_per_visibility_round",
            &IrisFromCliqueCoverOptions::num_points_per_visibility_round,
            cls_doc.num_points_per_visibility_round.doc)
        .def_readwrite("point_sampler",
            &IrisFromCliqueCoverOptions::point_sampler,
            cls_doc.point_sampler.doc)
        .def_readwrite("num_builders",
            &IrisFromCliqueCoverOptions::num_builders, cls_doc.num_builders.doc)
        .def_readwrite("generator", &IrisFromCliqueCoverOptions::generator,
            cls_doc.generator.doc)
        .def_property_readonly(
            "max_clique_solver",
            [](const IrisFromCliqueCoverOptions& self) {
              return self.max_clique_solver.get();
            },
            py_rvp::reference_internal);
    m.def(
        "IrisFromCliqueCover",
        [](const std::vector<geometry::optimization::ConvexSet*>& obstacles,
            const HPolyhedron& domain,
            const IrisFromCliqueCoverOptions& options,
            std::vector<HPolyhedron> sets) {
          IrisFromCliqueCover(
              CloneConvexSets(obstacles), domain, options, &sets);
          return sets;
          ;
        },
        py::arg("obstacles"), py::arg("domain"), py::arg("options"),
        py::arg("sets"), doc.IrisFromCliqueCover.doc);
  }

}  // DefinePlanningIrisFromCliqueCover
}  // namespace internal
}  // namespace pydrake
}  // namespace drake
