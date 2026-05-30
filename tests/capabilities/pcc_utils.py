# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Shared PCC (Pearson Correlation Coefficient) assertion utilities."""


def assert_pcc(actual, expected, threshold=0.99, msg=""):
    """Assert PCC between TTNN output and PyTorch reference meets threshold."""
    raise NotImplementedError("Implement during Skill 2 execution")
