# Publishing wave-lang packages to PyPI

1. Update version.json as needed and commit to `main` (make sure it doesn't
conflict with existing versions on PyPI).

2. Consider triggering the "Build and release packages" workflow, which is
necessary if the latest wheels don't have the right version. This workflow is
also automatically triggered every night.

> [!NOTE]
> `wave-lang` is not tested after it is built (nor before publishing), so make
> sure the commit it's built from passed all tests.

3. Identify the run ID of the specific "Build and release packages" workflow
whose Python wheels you want to publish. The run ID can be found in the numeric
slug of a run's URL (e.g. 16763019481 in
https://github.com/iree-org/wave/actions/runs/16763019481).

4. Manually trigger the "Publish to PyPI" workflow, which uses
[Trusted Publishing](https://docs.pypi.org/trusted-publishers/). This step uses
the run ID identified previously.
