# Publishing wave-lang packages to PyPI

1. Pin the `iree-compiler` and `iree-runtime` version in `pyproject.toml` to the current stable relase (for example: `==3.10.0`).

2. Create a git tag for the release, the git tag will be used to derive the version field in `pyproject.toml`

3. Consider triggering the "Build and release packages" workflow, which is
necessary if the latest wheels don't have the right version. This workflow is
also automatically triggered every night.

> [!NOTE]
> `wave-lang` is not tested after it is built (nor before publishing), so make
> sure the commit it's built from passed all tests.

4. Identify the run ID of the specific "Build and release packages" workflow
whose Python wheels you want to publish. The run ID can be found in the numeric
slug of a run's URL (e.g. 16763019481 in
https://github.com/iree-org/wave/actions/runs/16763019481).

5. Manually trigger the "Publish to PyPI" workflow, which uses
[Trusted Publishing](https://docs.pypi.org/trusted-publishers/). This step uses
the run ID identified previously.

6. After a release, for the next development cycle, pin the `iree-compiler` and `iree-runtime` version in `pyproject.toml` to a compatible build with the upcoming release candidate (for example: `~=3.11.0rc`).
