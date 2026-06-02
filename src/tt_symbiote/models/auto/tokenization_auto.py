# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Tokenizer Auto class.

In v0.1 ``tt_symbiote.AutoTokenizer`` is a verbatim re-export of
``transformers.AutoTokenizer``.
"""

from transformers import AutoTokenizer as AutoTokenizer  # noqa: F401  (re-export)

__all__ = ["AutoTokenizer"]
