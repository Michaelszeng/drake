load("//tools/workspace:github.bzl", "github_archive")

def mypy_internal_repository(
        name,
        mirrors = None):
    github_archive(
        name = name,
        # This dependency is part of a "cohort" defined in
        # drake/tools/workspace/new_release.py.  When practical, all members
        # of this cohort should be updated at the same time.
        repository = "python/mypy",
        commit = "v1.10.0",
        sha256 = "5550f427e9492de27e734ed182f9418f41bc632863b47470c6aab56420a0e661",  # noqa
        build_file = ":package.BUILD.bazel",
        patches = [
            ":patches/no_retry.patch",
            ":patches/reject_double_colon.patch",
            ":patches/timeout.patch",
        ],
        mirrors = mirrors,
    )
