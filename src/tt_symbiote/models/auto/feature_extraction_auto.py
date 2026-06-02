# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Feature-extractor Auto class.

In v0.1 ``tt_symbiote.AutoFeatureExtractor`` is a verbatim re-export of
``transformers.AutoFeatureExtractor``.
"""

from transformers import AutoFeatureExtractor as AutoFeatureExtractor  # noqa: F401  (re-export)

__all__ = ["AutoFeatureExtractor"]
