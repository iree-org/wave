# SPDX-FileCopyrightText: 2026 ixsimpl contributors
# SPDX-License-Identifier: Apache-2.0
"""Shared pytest configuration for the ixsimpl test suite."""

from __future__ import annotations

from hypothesis import HealthCheck, settings


def pytest_addoption(parser):  # type: ignore[no-untyped-def]
    parser.addoption(
        "--quick",
        action="store_true",
        default=False,
        help="Run Hypothesis tests with fewer examples (for CI).",
    )
    parser.addoption(
        "--torture",
        action="store_true",
        default=False,
        help="Run Hypothesis tests with many more examples and no shrinking limit.",
    )


def pytest_configure(config):  # type: ignore[no-untyped-def]
    settings.register_profile(
        "quick",
        max_examples=200,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    settings.register_profile(
        "full",
        max_examples=2000,
        deadline=None,
    )
    settings.register_profile(
        "torture",
        max_examples=50_000,
        deadline=None,
        suppress_health_check=[h for h in HealthCheck if h != HealthCheck.too_slow],
    )
    if config.getoption("--torture", default=False):
        settings.load_profile("torture")
    elif config.getoption("--quick", default=False):
        settings.load_profile("quick")
    else:
        settings.load_profile("full")
