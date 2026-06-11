# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""End-to-end demo for DiffusionGemma-26B-A4B-it on Tenstorrent P150x4.

Runs the full 26B block-diffusion sparse-MoE model on a 4-chip Blackhole mesh
(tensor-parallel TP=4, expert-parallel 32/chip, tied encoder<->decoder weights,
~12 GB/chip) via the sharded TTNN pipeline, with Metal-Trace replay: the 30-layer
decoder stack is captured ONCE and replayed for every block-diffusion denoising
step. The model is non-autoregressive -- the prompt conditions an encoder once,
then a canvas of tokens is denoised in parallel over N steps.

Prompt format
-------------
The checkpoint ships NO chat template and uses Gemma4 turn tokens
``<|turn>`` (105) / ``<turn|>`` (106) -- NOT the standard ``<start_of_turn>``
strings (which tokenize to ``<unk>`` here and yield empty/garbage output).
``format_prompt`` builds the correct turn-delimited instruction string, matching
how the Gemma4 family is prompted.

Run (MUST use the tt-metal python_env, which has torch + ttnn)::

    TT_METAL_HOME=/path/to/tt-metal \
    $TT_METAL_HOME/python_env/bin/python -m tt_symbiote.models.diffusion_gemma.demo \
        --prompt "What is the capital of France? Answer in one sentence."

Options::

    --prompt TEXT     one prompt (repeatable); omit to run the built-in demo set
    --steps N         denoising steps (default 24)
    --canvas N        canvas length / tokens denoised per block (default 128, multiple of 32)
    --no-trace        run eager (NORMAL) instead of Metal-Trace replay
    --model ID        HuggingFace model id (default google/diffusiongemma-26B-A4B-it)
"""
from __future__ import annotations

import argparse
import os
import time

MODEL_ID = "google/diffusiongemma-26B-A4B-it"

# Gemma4 end-of-turn / eos markers (from the model's generation_config eos_token_id).
_EOS_TOKENS = {1, 106, 50}

DEFAULT_PROMPTS = [
    "What is the capital of France? Answer in one sentence.",
    "Name three primary colors.",
    "Write a one-line haiku about the ocean.",
]


def format_prompt(text: str) -> str:
    """Wrap a user instruction in the Gemma4 turn format the -it model expects."""
    return f"<|turn>user\n{text}<turn|>\n<|turn>model\n"


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _open_mesh(trace_region_size: int):
    """Enable FABRIC_1D (required for CCL all_reduce on the line mesh) and open the
    1x4 P150x4 mesh."""
    import ttnn

    n = ttnn.GetNumAvailableDevices()
    if n < 4:
        raise RuntimeError(f"need 4 chips, GetNumAvailableDevices()={n} (cluster wedged? `tt-smi -r 0,1,2,3`)")
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    md = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=trace_region_size)
    assert tuple(md.shape) == (1, 4), f"mesh shape {tuple(md.shape)} != (1,4)"
    return md


def _close_mesh(md) -> None:
    import ttnn

    ttnn.close_mesh_device(md)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def _decode_answer(tok, token_ids):
    """Truncate the denoised canvas at the first end-of-turn/eos marker and decode."""
    cut = next((i for i, t in enumerate(token_ids) if t in _EOS_TOKENS), len(token_ids))
    return tok.decode(token_ids[:cut], skip_special_tokens=True).strip()


def run_demo(prompts, *, steps=24, canvas=128, trace=True, model_id=MODEL_ID):
    """Build the sharded TTNN pipeline once and generate for each prompt.

    The pipeline is built a single time; under trace mode the decoder-stack trace is
    captured on the first prompt and replayed for the rest. Returns a list of dicts
    ``{prompt, text, token_ids}``.
    """
    # Bind env BEFORE importing tt_symbiote (run mode + device arch are read at import).
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("MESH_DEVICE", "P150x4")
    os.environ["TT_SYMBIOTE_RUN_MODE"] = "TRACED" if trace else "NORMAL"

    import torch  # noqa: E402
    from transformers import AutoTokenizer  # noqa: E402
    from transformers.models.diffusion_gemma.modeling_diffusion_gemma import (  # noqa: E402
        DiffusionGemmaForBlockDiffusion,
    )

    import tt_symbiote.core.module as _mod  # noqa: E402
    from tt_symbiote.core.run_config import _RUN_MODE_REGISTRY, TracedRun  # noqa: E402
    from tt_symbiote.models.diffusion_gemma.pipeline import TTNNDiffusionGemmaPipeline  # noqa: E402

    # Force the run implementation explicitly: `python -m ...demo` imports the package
    # __init__ chain (which binds module.TENSOR_RUN_IMPLEMENTATION from the env at
    # import) BEFORE run_demo runs, so setting the env var above is too late. Rebind
    # the process-global directly so trace replay actually engages (or NORMAL eager).
    _mod.TENSOR_RUN_IMPLEMENTATION = _RUN_MODE_REGISTRY["TRACED" if trace else "NORMAL"]

    _log(f"loading tokenizer + real weights ({model_id}) ...")
    tok = AutoTokenizer.from_pretrained(model_id)
    t0 = time.time()
    model = DiffusionGemmaForBlockDiffusion.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    ).eval()
    _log(
        f"  HF weights loaded ({sum(p.numel() for p in model.parameters())/1e9:.1f}B params) "
        f"in {time.time()-t0:.1f}s"
    )

    results = []
    md = _open_mesh(trace_region_size=1_200_000_000)
    try:
        t0 = time.time()
        pipe = TTNNDiffusionGemmaPipeline.from_hf_model(model, md, canvas_length=canvas, max_denoising_steps=steps)
        _log(
            f"sharded TTNN pipeline built ({len(pipe.decoder_stack.layers)} layers, "
            f"TP=4, canvas={canvas}, steps={steps}, mode={'TRACED' if trace else 'NORMAL'}) "
            f"in {time.time()-t0:.1f}s"
        )
        del model  # free the host copy; sharded weights are resident on device

        for i, prompt_text in enumerate(prompts):
            input_ids = tok(format_prompt(prompt_text), return_tensors="pt").input_ids
            t0 = time.time()
            canvas_out = pipe.generate(input_ids, max_denoising_steps=steps)
            dt = time.time() - t0
            ids = canvas_out[0].tolist()
            text = _decode_answer(tok, ids)
            _log(
                f"prompt {i+1}/{len(prompts)} [{'trace capture+replay' if trace else 'eager'}] "
                f"{dt:.1f}s (traces={TracedRun.cache_size()})"
            )
            results.append({"prompt": prompt_text, "text": text, "token_ids": ids})
            # The captured decoder-stack trace bakes THIS prompt's encoder KV / position
            # embeddings by reference (they are constant across the prompt's denoise
            # steps -- that is where replay pays off). They DIFFER per prompt, so release
            # between prompts; the next prompt re-captures against its own encoder KV.
            if trace:
                TracedRun.release_all()

            print("\n" + "=" * 72, flush=True)
            print(f"PROMPT : {prompt_text}", flush=True)
            print(f"OUTPUT : {text!r}", flush=True)
            print("=" * 72 + "\n", flush=True)
    finally:
        if trace:
            TracedRun.release_all()
        _close_mesh(md)
    return results


def main():
    ap = argparse.ArgumentParser(description="DiffusionGemma-26B P150x4 e2e demo")
    ap.add_argument(
        "--prompt", action="append", default=None, help="a prompt (repeatable); omit to run the built-in set"
    )
    ap.add_argument("--steps", type=int, default=24, help="denoising steps (default 24)")
    ap.add_argument("--canvas", type=int, default=128, help="canvas length, multiple of 32 (default 128)")
    ap.add_argument("--no-trace", action="store_true", help="run eager (NORMAL), no Metal-Trace")
    ap.add_argument("--model", default=MODEL_ID, help="HuggingFace model id")
    args = ap.parse_args()

    prompts = args.prompt if args.prompt else DEFAULT_PROMPTS
    if args.canvas % 32 != 0:
        ap.error("--canvas must be a multiple of 32")

    results = run_demo(prompts, steps=args.steps, canvas=args.canvas, trace=not args.no_trace, model_id=args.model)

    print("\n##### SUMMARY #####", flush=True)
    for r in results:
        print(f"- {r['prompt'][:48]!r:52} -> {r['text']!r}", flush=True)


if __name__ == "__main__":
    main()
