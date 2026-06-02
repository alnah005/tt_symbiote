# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Smoke test for Ling-mini-2.0 with the TTNN backend.

This is the **Phase 5 reference test**. It exercises the full public
``tt_symbiote`` API path:

1. :class:`tt_symbiote.AutoModelForCausalLM` loads the HF model and applies
   the registered :class:`BailingMoEV2Recipe` (single-dict, single-pass
   module replacement).
2. :func:`tt_symbiote.set_device` binds every TTNN module to ``mesh_device``,
   subsumes the per-module ``preprocess_weights`` / ``move_weights_to_device``
   loop, and allocates the paged-attention KV cache via the recipe's
   ``make_kv_cache`` hook — attached as ``model._tt_kv_cache``.
3. :meth:`model.generate` runs end-to-end with the recipe-built paged cache
   passed back in as ``past_key_values``.

When this test passes on hardware, the repo is tagged ``v0.0.0``.
"""

import os

import pytest
import torch
from transformers import AutoTokenizer

import ttnn
from tt_symbiote import AutoModelForCausalLM, set_device
from tt_symbiote.core.run_config import DispatchManager, TracedRun


@pytest.mark.parametrize(
    "device_params",
    [{"trace_region_size": 200000000, "num_command_queues": 1, "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}],
    indirect=True,
)
@pytest.mark.parametrize(
    "mesh_device",
    [
        {
            "N150": (1, 1),
            "N300": (1, 2),
            "N150x4": (1, 4),
            "T3K": (1, 8),
            "TG": (8, 4),
            "P150": (1, 1),
            "P300": (1, 2),
            "P150x4": (1, 4),
            "P150x8": (1, 8),
            "BHGLX": (8, 4),
        }.get(os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids()))
    ],
    indirect=True,
)
def test_ling_mini_2_0(mesh_device):
    """Ling-mini-2.0 end-to-end through the new public API."""

    tokenizer = AutoTokenizer.from_pretrained("inclusionAI/Ling-mini-2.0", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained("inclusionAI/Ling-mini-2.0", trust_remote_code=True)

    messages = [
        {
            "role": "user",
            "content": (
                "What is your favorite condiment? There are so many condiments to choose from, "
                "each bringing its unique flavor and texture to enhance different dishes. Do you "
                "prefer the classic taste of ketchup, the creamy richness of mayonnaise, the "
                "spicy kick of mustard, or perhaps something more exotic like sriracha or hoisin "
                "sauce? Maybe you enjoy the tangy zest of salsa or the smooth and savory taste "
                "of aioli. Share what your favorite condiment is and why you love it. Does it "
                "remind you of a specific dish or meal?"
            ),
        },
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    inputs.pop("token_type_ids", None)

    set_device(model, mesh_device)
    assert hasattr(model, "_tt_kv_cache"), (
        "set_device should have invoked BailingMoEV2Recipe.make_kv_cache " "and attached model._tt_kv_cache"
    )

    model.eval()
    torch.set_grad_enabled(False)

    paged_cache = model._tt_kv_cache

    # Warmup run without trace, then a short trace-warmup run.
    for max_new in (2, 4):
        model.generate(
            **inputs,
            max_new_tokens=max_new,
            use_cache=True,
            past_key_values=paged_cache,
        )
        paged_cache.reset()

    DispatchManager.clear_timings()
    outputs = model.generate(
        **inputs,
        max_new_tokens=128,
        use_cache=True,
        past_key_values=paged_cache,
    )

    decoded = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1] :])
    print(f"Ling-mini-2.0 PAGED ATTENTION OUTPUT: {decoded}")

    assert len(decoded.strip()) > 0, "Generated output should not be empty"

    DispatchManager.save_stats_to_file("ling_mini_2_0_paged_attention_timing_stats.csv")
    TracedRun.release_all()
