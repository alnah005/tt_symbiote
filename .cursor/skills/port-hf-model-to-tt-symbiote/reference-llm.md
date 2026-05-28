# LLM (causal LM) port specifics

Reference material for `port-hf-model-to-tt-symbiote` when the target HF
model maps to `AutoModelForCausalLM`. Read this in addition to
[`SKILL.md`](SKILL.md).

The canonical LLM port is
[`inclusionAI/Ling-mini-2.0`](../../src/tt_symbiote/models/bailing_moe_v2/)
(Phase 5). It is a **full TTNN** port — significantly more involved than
the CPU-first VLM pattern. The Phase 7 skill ships CPU-first by default;
follow this reference only if the user explicitly asks for a full TTNN
LLM port.

## CPU-first variant

A CPU-first LLM port produces:

```
src/tt_symbiote/models/<NAME>/
├── __init__.py
├── configuration_<NAME>.py     # re-export HF config + per-variant tuning
└── modeling_<NAME>.py          # @register_recipe(hf_class_name="<NAME>ForCausalLM")
                                #   build_module_dict -> {}
                                #   post_register -> patches device, runtime_config
                                #   tt_implemented / cpu_fallback / out_of_scope lists
```

`<HF_CLASS>` ends in `ForCausalLM` (not `ForConditionalGeneration`).
`<AUTO_CLASS>` is `AutoModelForCausalLM`.

## Tokenizer + generate surface

LLM demos use `AutoTokenizer.apply_chat_template` directly (no
`AutoProcessor`):

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
inputs = tokenizer.apply_chat_template(
    [{"role": "user", "content": "Explain X."}],
    add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt",
).to("cpu")
inputs.pop("token_type_ids", None)   # some tokenizers add this; HF rejects it
```

Generation:

```python
out = model.generate(
    **inputs,
    max_new_tokens=512,
    do_sample=False,
    use_cache=True,
)
```

Decode skips the prompt:

```python
print(tokenizer.decode(out[0][inputs["input_ids"].shape[-1]:]))
```

LLM semantic check is a **manual eyeball** of the output (the demo
prints; no `assert` on content). The skill's acceptance gate for LLM
ports is: exit 0 + coherent EOS-terminated text. If you can't eyeball
the output, the port is not done.

## trust_remote_code

Many community LLMs (Ling-mini-2.0, dots.ocr, etc.) live as Hub
modeling files outside the transformers package. They require
`trust_remote_code=True` on both `AutoTokenizer.from_pretrained` and
`AutoModelForCausalLM.from_pretrained`. They also frequently trip on
API drift between the Hub modeling file and the installed transformers
version — see [`_hf_compat.py`](../../src/tt_symbiote/_hf_compat.py) for
the shim pattern (Ling needed two shims: `is_torch_fx_available` and
`ROPE_INIT_FUNCTIONS["default"]`).

When adding a new shim, name the function after the missing symbol and
add a comment explaining why the transformers version stopped exposing
it.

## KV cache

CPU-first LLM ports keep HF's `DynamicCache` — no `make_kv_cache`
implementation needed. The `@register_recipe` no-op default covers it.

Full TTNN LLM ports must implement `make_kv_cache` returning a paged
attention cache. See
[`BailingMoEV2Recipe.make_kv_cache`](../../src/tt_symbiote/models/bailing_moe_v2/modeling_bailing_moe_v2.py).
The cache is allocated by `set_device` (Phase 5 Q9) and attached as
`model._tt_kv_cache`; demo scripts pass it to `generate` as
`past_key_values=`.

## Mesh shape per variant

Causal LMs scale with parameter count:

| Param count | Mesh | Example |
|---|---|---|
| ≤ ~2 B BF16 | `(1, 1)` | `Qwen3-1.7B` |
| ~2–12 B BF16 | `(1, 1)` (single chip) or `(1, 2)` (N300) | `Ling-mini-2.0` (~16 B MoE but fits T3K), `Qwen3-8B` |
| ≥ ~16 B BF16 | `(1, 8)` (T3K, requires tensor-parallel sharded linears) | `Gemma-4-31B`, `Qwen3-32B` |

The CPU-first port works at any size as long as the host has the RAM.

## Out-of-scope class enumeration

For LLM ports, `out_of_scope` is typically just the output dataclasses
(`CausalLMOutputWithPast`, etc.). The full text decoder enumeration goes
into `cpu_fallback`.

## Acceptance gate

A LLM CPU-first port is done when:

1. `from_pretrained` + `set_device` + `generate` end-to-end succeeds.
2. Decoded text is coherent and terminates on EOS (manual check).
3. `compatibility.report(model)["runtime_observed"]["unexpected"]` is empty.
