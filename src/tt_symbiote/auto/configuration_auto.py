# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Configuration Auto class.

In v0.1 ``tt_symbiote.AutoConfig`` is a verbatim re-export of
``transformers.AutoConfig``. Phase 5+ may introduce subclasses that override
configuration loading for specific models; the indirection here keeps the
public import path stable when that happens.
"""

from transformers import AutoConfig as AutoConfig  # noqa: F401  (re-export)

__all__ = ["AutoConfig"]
