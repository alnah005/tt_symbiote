# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures for per-model capability tests.

The ``device`` fixture used by all capability tests comes from the
ttnn pytest plugin (or the model-specific test's own conftest). This
file provides additional shared utilities.
"""

import pytest


@pytest.fixture
def pcc_threshold():
    """Default PCC threshold for model accuracy tests."""
    return 0.99
