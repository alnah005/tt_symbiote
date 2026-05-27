# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Image processor Auto class.

In v0.1 ``tt_symbiote.AutoImageProcessor`` is a verbatim re-export of
``transformers.AutoImageProcessor``.
"""

from transformers import AutoImageProcessor as AutoImageProcessor  # noqa: F401  (re-export)

__all__ = ["AutoImageProcessor"]
