# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Shared conftest for the shared capability tests under tests/shared/.

The ``device`` / ``mesh_device`` fixtures used by these capability tests come
from the ttnn pytest plugin (installed with tt-metal). The ``pcc_threshold``
fixture and the non-blocking ``tt_metal_commit_check`` fixture are defined ONCE
in the ROOT ``tests/conftest.py`` (the common ancestor of all test trees) so
they reach the sibling ``tests/models/`` and ``tests/experimental/`` per-model
trees as well. This file intentionally defines no fixtures of its own.
"""
