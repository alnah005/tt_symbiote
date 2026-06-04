# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""TTNN port of the Physical Intelligence pi0.5 (PI0.5) vision-language-action model.

pi0.5 is a lerobot/openpi checkpoint (``lerobot/pi05_base``), *not* a
``transformers`` model, so this package uses the **manual integration path**:
TTNN ``TTNNModule`` wrappers whose ``from_torch`` consumes the plain-object
PyTorch reference (``reference/torch_*.py`` in the tt-metal ``pi0_5`` tree),
which is the PCC golden. There is no ``@register_recipe`` / Auto API entry.

Top-level model: :class:`~tt_symbiote.models.pi05.modeling_pi05.TTNNPi05Model`.
"""

from tt_symbiote.models.pi05.configuration_pi05 import (
    DenoiseConfig,
    GemmaConfig,
    PaliGemmaConfig,
    Pi0_5ModelConfig,
    PrefixConfig,
    SigLIPConfig,
    SuffixConfig,
)
from tt_symbiote.models.pi05.modeling_pi05 import (
    TTNNPi05Model,
    from_checkpoint,
    load_reference_pi05_model,
)

__all__ = [
    "DenoiseConfig",
    "GemmaConfig",
    "PaliGemmaConfig",
    "Pi0_5ModelConfig",
    "PrefixConfig",
    "SigLIPConfig",
    "SuffixConfig",
    "TTNNPi05Model",
    "from_checkpoint",
    "load_reference_pi05_model",
]
