# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Processor Auto class.

In v0.1 ``tt_symbiote.AutoProcessor`` is a verbatim re-export of
``transformers.AutoProcessor``.
"""

from transformers import AutoProcessor as AutoProcessor  # noqa: F401  (re-export)

__all__ = ["AutoProcessor"]
