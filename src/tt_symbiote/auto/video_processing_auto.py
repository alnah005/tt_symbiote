# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Video processor Auto class.

In v0.1 ``tt_symbiote.AutoVideoProcessor`` is a verbatim re-export of
``transformers.AutoVideoProcessor``.
"""

from transformers import AutoVideoProcessor as AutoVideoProcessor  # noqa: F401  (re-export)

__all__ = ["AutoVideoProcessor"]
