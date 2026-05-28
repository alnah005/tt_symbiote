# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Small math helpers used across integrations and model packages."""

from __future__ import annotations

__all__ = ["next_power_of_2"]


def next_power_of_2(n: int, minimum: int = 256) -> int:
    """Return the smallest power of 2 >= ``n`` (with a floor of ``minimum``).

    Used by sequence-length padding paths in the rotary / bailing-padded
    decoder layer: padding to a power-of-2 sequence length reduces the
    number of unique trace cache keys during prefill, since many distinct
    prompt lengths map to the same padded length.

    Previously this lived as a private ``_next_power_of_2`` inside
    :mod:`tt_symbiote.models.bailing_moe_v2.modeling_bailing_moe_v2` as a
    consequence of the Phase 2 mechanical migration; moved here in
    Phase 5 to break a circular import between the bailing model and the
    generic ``ttnn_embedding`` integration that also needs it.
    """
    if n <= 1:
        return 1
    if n <= minimum:
        return minimum
    result = 1 << ((n - 1).bit_length() + 1)
    # If n is already a power of 2 we want n itself, not 2n. The original
    # implementation detected this with ``result == n * 4`` after the
    # one-extra-bit shift; preserved verbatim to keep numerical equivalence.
    if result == n * 4:
        result = n
    return result
