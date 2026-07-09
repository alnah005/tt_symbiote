# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""TTNN modeling skeleton for baidu/Unlimited-OCR (DeepSeek-OCR-style VLM + MoE).

SCAFFOLD STAGE — this module defines the class tree, base-class contracts, and
``from_torch`` wiring for the TTNN port. Every ``forward`` is a device-guarded
stub that raises ``NotImplementedError`` with a pointer to the reference
implementation it should follow. Real compute bodies land in the later
pcc_test_gen stage (see CLAUDE.md "Bottom-Up Tuning"); NOTHING here performs
TTNN or torch compute in a ``forward`` path.

Architecture (see deep-work/unlimited_ocr/architecture_map.md):
  UnlimitedOCRForCausalLM
    -> DeepEncoder            (SAM ViT-B  +  CLIP-L, conv 16x compressor)
    -> MlpProjector           (Linear 2048 -> 1280)
    -> DeepseekV2Model        (embed + 12 DecoderLayer + RMSNorm)
       DecoderLayer L0        = dense DeepseekMLP (SwiGLU)
       DecoderLayer L1..L11   = DeepseekV2MoE (64 routed experts, k=6, 2 shared)
       self_attn              = SlidingWindowLlamaAttention (plain Llama MHA,
                                use_mla=False; RoPE theta=10000; sliding win=128)
    -> lm_head                (Linear 1280 -> 129280)

Base-class contract (core/module.py):
  * StatelessTTNNModule — forward mutates no persistent state under the trace
    double-run (all vision, MLP, MoE, norms, model/for-causal-lm shells).
  * StatefulTTNNModule — forward writes persistent state (KV cache). Used by the
    LM attention (real ``reset_trace_state``) and the decoder layer that owns it
    (documented no-op ``reset_trace_state``, mirroring dots_ocr_decoder_layer.py).

Reference implementations to consult when filling in the forwards:
  $TT_METAL_HOME/models/tt_transformers/tt/{attention,mlp,decoder,model}.py
  $TT_METAL_HOME/models/tt_transformers/tt/rope.py, lm_head.py
  src/tt_symbiote/models/dots_ocr/ (closest OCR-VLM analog)
  src/tt_symbiote/models/bailing_moe_v2/ (DeepSeek-lineage MoE decoder)
  src/tt_symbiote/modules/ttnn_{linear,attention,normalization,moe,conv,embedding,rope,activation}.py
"""

from __future__ import annotations

from tt_symbiote.core.module import (
    DeviceArch,
    StatefulTTNNModule,
    StatelessTTNNModule,
    run_on_devices,
)
from tt_symbiote.core.run_config import trace_enabled
from tt_symbiote.modules.ttnn_linear import TTNNLinear

# tt-metal version this port is developed against (see CLAUDE.md commit-tracking).
TT_METAL_COMMIT = "a0b506c780979538b6d2fc1e57fdbfdfdabc7e31"


def _todo(what: str, ref: str) -> "NotImplementedError":
    """Uniform scaffold stub error: what to implement + a reference pointer."""
    return NotImplementedError(f"{what} — ref: {ref}")


class UnlimitedOcrKVCache:
    """Simple contiguous per-layer K/V cache for FULL-CAUSAL cached decode.

    NOT a ``TTNNModule`` -- a plain device-tensor holder threaded as
    ``past_key_values`` through the LM stack (mirroring how HF threads a
    ``Cache`` object). Each layer stores the rope'd K/V for every token seen so
    far as a single contiguous ``[1, H, L, D]`` device tensor (TILE layout).

    PREFILL fills a layer's slot with the whole prompt's K/V; DECODE appends the
    single new token's K/V (``ttnn.concat`` along the seq dim). Attention then
    runs over ALL cached K/V, which -- for causal attention with the query at the
    newest position -- is mathematically identical to re-prefilling the whole
    sequence. This is the honest, exact O(n) win for the currently-validated
    full-causal regime (the window-128 ring buffer is a separate refinement).
    """

    def __init__(self, num_layers: int):
        self.num_layers = int(num_layers)
        self.k = [None] * self.num_layers
        self.v = [None] * self.num_layers

    def reset(self) -> None:
        import ttnn

        for i in range(self.num_layers):
            for buf in (self.k, self.v):
                if buf[i] is not None:
                    try:
                        ttnn.deallocate(buf[i])
                    except Exception:  # noqa: BLE001 — best-effort free
                        pass
                    buf[i] = None

    def seq_len(self, layer_idx: int = 0) -> int:
        k = self.k[layer_idx]
        return 0 if k is None else int(k.shape[2])


def _lm_compute_kernel_config():
    """HiFi4 + fp32 dest-accum compute config for the language-model matmuls.

    The shared ``TTNNLinear.forward`` issues ``ttnn.linear`` with NO
    ``compute_kernel_config`` -> the LoFi default. LoFi's reduced matmul
    fidelity is invisible per-layer (each LM tier test greens at 0.999) but
    COMPOUNDS across the 12 real decoder layers: full-model TTNN logits drop to
    PCC~0.89 / argmax-mismatch while torch-bf16 (same algorithm, full fidelity)
    holds 0.9997. HiFi4 + fp32 accumulation recovers the per-op fidelity so the
    12-layer stack matches the torch reference. Blackhole accepts the
    ``WormholeComputeKernelConfig`` (the greened MoE-gate matmul uses it too).
    """
    import ttnn

    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )


_PROJECTOR_PROGRAM_CONFIG = None


def _projector_matmul_program_config():
    """Explicit 2D block-multicast matmul program config for the vision->text
    projector Linear(2048 -> 1280) applied over the 256 concatenated CLIP+SAM
    vision tokens (prefill-only; M is ALWAYS 256, including under DP(4,1) where the
    batch dim is sharded but per-device M stays 256).

    Derived (and validated GREEN) by the matmul program-config op-sweep
    (sweep_results/matmul_progcfg.csv, label ``2d_8x8_blk4_1x1``) as ~2.24x faster
    than ttnn-AUTO on the isolated projector op (28352 ns vs 63564 ns, tracy DEVICE
    KERNEL DURATION; PCC 0.999956 vs the 0.999953 AUTO baseline -- program config is
    numerically inert, PCC is a sanity check). Params follow the tt_transformers
    ``matmul_config`` 2D formula (models/tt_transformers/tt/model_config.py) for the
    FIXED projector shape M=256/K=2048/N=1280 (Mt=8, Kt=64, Nt=40) on the 8x8
    Blackhole compute grid:
        per_core_M      = ceil(Mt/gy) = ceil(8/8) = 1
        per_core_N      = ceil(Nt/gx) = ceil(40/8) = 5
        in0_block_w     = largest divisor(<=4) of Kt=64 = 4
        out_subblock_h  = 1
        out_subblock_w  = get_out_subblock_w(per_core_N=5, 1) = 1  (5 % osw==0, osw*1<=4)
    All constraints hold with fp32_dest_acc_en (out_subblock_h*out_subblock_w=1<=4),
    so the config is valid under the projector's HiFi4 + fp32-accum compute kernel and
    under single-device, DP(4,1) (per-device M=256 unchanged) and trace. NOT applied
    to lm_qproj / lm_gate: their sweep wins were measured at an unrepresentative
    fixture M=8 (tiny text probe) whose per_core_M does NOT match the real prefill
    (M~277) or decode (M=1) paths, so those configs are shape-mismatched and left on
    ttnn-AUTO (a real application would require a re-sweep at the actual prefill/decode
    M; decode is already trace-bound so the marginal gain is negligible).
    """
    global _PROJECTOR_PROGRAM_CONFIG
    if _PROJECTOR_PROGRAM_CONFIG is None:
        import ttnn

        _PROJECTOR_PROGRAM_CONFIG = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(8, 8),
            in0_block_w=4,
            out_subblock_h=1,
            out_subblock_w=1,
            per_core_M=1,
            per_core_N=5,
            transpose_mcast=False,
            fused_activation=None,
            fuse_batch=False,
        )
    return _PROJECTOR_PROGRAM_CONFIG


def _lm_decode_sdpa_compute_kernel_config():
    """Compute config for the paged decode flash-SDPA (``paged_scaled_dot_product_
    attention_decode``). HiFi4 matmul fidelity + ``fp32_dest_acc_en=True`` +
    ``packer_l1_acc=True`` -- this is the tt_transformers authoritative decode-SDPA
    recipe (``model_config.compute_kernel_config_hifi4`` selected for
    ``OpGroup.SDPA_DECODE``; tt-metal a0b506c...).

    MEASURED, not assumed (Blackhole P150, 4 sample docs, teacher-forced 16-token
    decode-logits PCC + greedy-argmax match vs the CPU torch-fp32 reference):

      * The paged DECODE kernel does NOT share the vision flash-SDPA
        ``fp32_dest_acc_en=True`` regression (see ``_vision_sdpa_compute_kernel_config``,
        which is a DIFFERENT kernel: ``scaled_dot_product_attention``). Toggling
        ``fp32_dest_acc_en`` here (True vs False, HiFi4 either way) leaves aggregate
        token-match unchanged (60/64) and only shuffles WHICH near-tie flips:
        False "fixes" doc3 step-3 (a sub-ULP tie) but INTRODUCES a genuine
        mis-rank on doc4 step-3 (torch top-1 34.121 vs the token TTNN picks 33.544,
        a real 0.58 margin), which ``fp32_dest_acc_en=True`` gets RIGHT. So True is
        the correct choice, contradicting an earlier (now-corrected) docstring that
        claimed False.
      * ``exp_approx_mode`` on this decode op is bit-inert: passing an explicit
        ``SDPAProgramConfig(exp_approx_mode=False)`` produced byte-identical argmax
        on all 4 docs (unlike the SAM global SDPA, where it helped 0.982->0.999),
        so it is deliberately NOT plumbed in here.

    The residual token flips that remain under EVERY decode config (doc1 step-3
    gap 0.040, doc1 step-12, doc3 step-3 gap 0.027, doc4 step-6 gap 0.087 -- all
    bbox-coordinate tokens) are GENUINE mathematical ties: the competing torch-fp32
    logits differ by less than one bf16 ULP (~0.15-0.31 at logit magnitude 39-79),
    so the shared bf16 lm_head/hidden logit path -- NOT the decode SDPA -- cannot
    resolve them. Raising decode-SDPA precision cannot fix a sub-ULP tie in the
    downstream bf16 logits."""
    import ttnn

    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )


def _paged_kv_cache_cls():
    """Lazily import the proven dots_ocr paged KV-cache class (reused, not copied).

    Imported lazily so plain (software-only) collection of this module never
    forces the ttnn/transformers imports that ``dots_ocr._attention`` pulls at
    import time. Returns ``None`` if unavailable."""
    try:
        from tt_symbiote.models.dots_ocr._attention import TTNNPagedAttentionKVCache

        return TTNNPagedAttentionKVCache
    except Exception:  # noqa: BLE001 -- absence just disables the paged path
        return None


def _is_paged_kv_cache(obj) -> bool:
    cls = _paged_kv_cache_cls()
    return cls is not None and isinstance(obj, cls)


def _apply_rope(t, cos, sin):
    """HF-Llama rotate_half RoPE over the last dim; works for both the prefill
    layout ``[1, H, S, D]`` (cos/sin ``[1, 1, S, D]``) and the decode layout
    ``[1, 1, H, D]`` (cos/sin ``[1, 1, 1, D]`` broadcast over the head axis).

    Identical math to the original in-forward closure (kept byte-equivalent for
    the ``past_key_value=None`` PCC path)."""
    import ttnn

    shp = list(t.shape)
    d = shp[-1]
    lo = [0] * len(shp)
    hi_half = list(shp[:-1]) + [d // 2]
    lo_half = [0] * (len(shp) - 1) + [d // 2]
    x1 = ttnn.slice(t, lo, hi_half)
    x2 = ttnn.slice(t, lo_half, list(shp))
    rot = ttnn.concat([ttnn.neg(x2), x1], dim=-1)
    return ttnn.add(ttnn.multiply(t, cos), ttnn.multiply(rot, sin))


def _decode_head_sharded_mem_config(device, head_dim):
    """HEIGHT_SHARDED ``[TILE_SIZE, head_dim]`` on a single core -- the layout
    ``paged_update_cache`` requires for its ``[1, batch=1, num_kv_heads, head_dim]``
    K/V input (see paged_update_cache_device_operation.cpp: input must be sharded,
    ROW_MAJOR, shard-width == head_dim). Mirrors dots_ocr ``_build_head_output_mem_config``
    at batch=1."""
    import ttnn

    return ttnn.create_sharded_memory_config(
        shape=(ttnn.TILE_SIZE, head_dim),
        core_grid=ttnn.CoreGrid(y=1, x=1),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def _vision_sdpa_compute_kernel_config():
    """High-accuracy compute config for the VISION flash-SDPA calls (CLIP + SAM
    global). HiFi4 fidelity, but ``fp32_dest_acc_en=False``.

    KERNEL-BUG WORKAROUND (measured on Blackhole P150, tt-metal
    a0b506c780979538b6d2fc1e57fdbfdfdabc7e31): enabling ``fp32_dest_acc_en=True``
    on ``ttnn.transformer.scaled_dot_product_attention`` REGRESSES accuracy versus
    bf16 -- the opposite of the physically-correct behaviour. On the real CLIP-L
    tower (24 blocks) fed the real SAM features, isolated toggles measured
    (CLIP-encoder PCC vs torch-fp32):

        baseline (all bf16 default) ...................... 0.99723
        SDPA fp32_dest_acc_en=True, HiFi4, packer=False .. 0.96719   <-- REGRESSION
        SDPA fp32_dest_acc_en=True, HiFi4, packer=True ... 0.89466   <-- REGRESSION
        SDPA fp32_dest_acc_en=True, HiFi2 ................ 0.90689   <-- REGRESSION
        SDPA fp32_dest_acc_en=FALSE, HiFi4 .............. 0.99785   <-- BEST
        (linears fp32-acc, SDPA fp32_dest=False) ........ 0.99902

    Genuine fp32 dest-accumulation lives on the vision *linears*
    (:class:`TTNNUnlimitedOcrLinear`, which strictly improves PCC); the flash-SDPA
    accumulator MUST stay bf16 (``fp32_dest_acc_en=False``) while keeping HiFi4
    matmul fidelity -- the highest-accuracy path the buggy kernel offers. A manual
    fp32 QK^T->softmax->AV attention was also measured (0.99852) but is strictly
    less accurate AND far more expensive than flash HiFi4 (0.99902), and would OOM
    on the SAM global layers (4096x4096 fp32 scores)."""
    import ttnn

    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )


_LM_DECODE_PROGCFG_CACHE: dict = {}


def _lm_decode_1d_progcfg(n_tiles):
    """1D (mcast_in0) MatmulProgramConfig for a DECODE-shape LM projection.

    DECODE always processes exactly ONE token -> Mt == 1 (tall/skinny), so the
    natural matmul is the 1D MatmulMultiCoreReuseMultiCast1DProgramConfig with
    ``mcast_in0`` -- the same class the tt_transformers decode path
    (models/tt_transformers/tt/model_config.py matmul_1d_config) selects for
    single-token matmuls. Ported from the vetted formula in
    tests/experimental/unlimited_ocr/sweep_results/sweep_matmul_progcfg.py::mk_1d
    at grid 8x4 (the tracy-winning grid, ``1d_8x4_blk4_...``): ncores=32,
    per_core_M=1, per_core_N=ceil(Nt/32), in0_block_w=largest_divisor(Kt<=4)=4.

    Program config is numerically INERT (affects DEVICE TIME only). The real-shape
    re-sweep (matmul_progcfg_lm.csv, tt-metal a0b506c7) measured, vs ttnn AUTO at
    the REAL decode M=1, WITH the model's own LM compute kernel (HiFi4 /
    fp32_dest_acc=False / packer_l1_acc=False):
      * lm_qproj (Nt=40):  AUTO 20361ns -> 1d_8x4_blk4_1x2 11658ns (1.75x)
      * lm_gate  (Nt=214): AUTO 33936ns -> 1d_8x4_blk4_1x1 31624ns (1.07x)
    both PCC-identical to AUTO (0.99977). A config built for Mt==1 is ALWAYS valid
    for decode (M is fixed at 1), so it is safe to apply unconditionally on the
    decode path. Cached per (Kt, Nt).
    """
    import math

    import ttnn

    key = n_tiles
    cfg = _LM_DECODE_PROGCFG_CACHE.get(key)
    if cfg is None:
        Kt, Nt = n_tiles
        gx, gy = 8, 4
        ncores = gx * gy
        per_core_N = math.ceil(Nt / ncores)
        in0_block_w = max(i for i in range(1, 5) if Kt % i == 0)
        # The real LM compute kernel uses packer_l1_acc=False -> the hardware
        # out-subblock limit is out_subblock_h*out_subblock_w <= 4 (matmul_device_
        # operation.cpp available_reg_count). per_core_M==1 => out_subblock_h==1, so
        # cap out_subblock_w at 4. (Nt=40 -> per_core_N=2 -> osw=2 for lm_qproj;
        # Nt=214 -> per_core_N=7 -> osw=1 for lm_gate.)
        osw = max(i for i in range(1, 5) if per_core_N % i == 0)
        cfg = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(gx, gy),
            in0_block_w=in0_block_w,
            out_subblock_h=1,
            out_subblock_w=osw,
            per_core_M=1,
            per_core_N=per_core_N,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=True,
        )
        _LM_DECODE_PROGCFG_CACHE[key] = cfg
    return cfg


class TTNNUnlimitedOcrLinear(TTNNLinear):
    """Language-model Linear with HiFi4 + fp32-accum matmul (see
    :func:`_lm_compute_kernel_config`). Weights stay bf16 (torch-bf16 ceiling is
    0.9997); only the matmul math fidelity is raised. Used for the DeepSeek
    attention/MLP/MoE-expert/lm_head projections; vision linears keep the LoFi
    default (validated GREEN independently)."""

    @run_on_devices(DeviceArch.P150, DeviceArch.P150x4, DeviceArch.T3K)
    def forward(self, input_tensor):
        import ttnn

        if input_tensor.layout != ttnn.TILE_LAYOUT:
            input_tensor = ttnn.to_layout(input_tensor, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        input_tensor_shape = list(input_tensor.shape)
        input_shape = list(input_tensor_shape)
        # Prepend the rank-padding 1s so the leading token dim stays the matmul M
        # (not a batch): a 2D [M, K] must become [1, 1, M, K], else ttnn.linear runs
        # M tiny per-token batched matmuls instead of one full-grid matmul.
        while len(input_shape) < 4:
            input_shape.insert(0, 1)
        input_tensor = ttnn.reshape(input_tensor, input_shape)
        # Optional shape-specific matmul program config (numerically INERT; DEVICE
        # TIME only). Opt-in PER INSTANCE via ``self._decode_progcfg``: set ONLY on
        # the LM decode-path projections (lm_qproj/lm_gate) whose real-shape
        # re-sweep found a 1D config faster than ttnn AUTO at the fixed decode M==1.
        # M is the flattened leading-dim product; the config is applied ONLY when
        # M == 1 (decode) and falls back to AUTO (program_config=None) otherwise
        # (e.g. prefill, where the sweep found AUTO optimal / the fixed config
        # unsafe across varying prompt length). Default: attribute unset -> AUTO,
        # so every other linear is byte-for-byte unchanged.
        program_config = None
        decode_pc = getattr(self, "_decode_progcfg", None)
        if decode_pc is not None:
            flat_m = 1
            for d in input_shape[:-1]:
                flat_m *= int(d)
            if flat_m == 1:
                program_config = decode_pc
        tt_output = ttnn.linear(
            input_tensor,
            self.tt_weight,
            bias=self.tt_bias,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=_lm_compute_kernel_config(),
            program_config=program_config,
        )
        tt_output = ttnn.reshape(tt_output, input_tensor_shape[:-1] + [self.out_features])
        return tt_output


class TTNNUnlimitedOcrLinearBf8(TTNNUnlimitedOcrLinear):
    """Config-optimize variant of :class:`TTNNUnlimitedOcrLinear` whose WEIGHT is
    stored ``bfloat8_b`` instead of ``bfloat16`` (bias stays bf16; the forward's
    HiFi4 + fp32-dest-accum compute-kernel config is inherited unchanged).

    APPLIED to the compute-bound "bottleneck" matmuls the op-sweep measured as
    essentially lossless under bfloat8_b weights (sweep_results/linear_pcc.csv +
    recommendation.json: per-op PCC >= 0.99998 for clip qkv/fc*, sam qkv/lin*,
    projector, and the LM q/k/v/o + MLP/expert gate/up/down at HiFi4). Per
    data_formats.md a bfloat8_b weight halves weight bandwidth and speeds the
    matmul, cutting the compute-bound vision prefill at negligible accuracy cost.

    DELIBERATELY NOT applied to:
      * ``lm_head`` (final-logit projection) -- kept bf16 via the parent class so
        bfloat8_b rounding cannot flip argmax on near-tie bbox-coordinate tokens.
      * the MoE router gate matmul -- its weight is fp32 (accuracy-critical
        routing) and lives on TTNNUnlimitedOcrMoE, not a TTNNLinear.
      * the KV cache -- bf16 (bfloat8_b KV corrupts OCR); only WEIGHT dtype changes.

    Bias is intentionally kept bf16 (tiny; matches how the op-sweep measured the
    bfloat8_b points and how ttnn adds bias into the bf16 matmul output).
    """

    def preprocess_weights_impl(self):
        import ttnn
        from ttnn.model_preprocessing import preprocess_linear_bias, preprocess_linear_weight

        self.tt_weight_host = preprocess_linear_weight(
            self.weight, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT
        )
        self.tt_bias_host = None
        if self.bias is not None:
            self.tt_bias_host = preprocess_linear_bias(
                self.bias, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
            )


# =============================================================================
# Tier 2 — language-model composites (attention / MLP / MoE)
# =============================================================================
@trace_enabled
class TTNNUnlimitedOcrLlamaMHA(StatefulTTNNModule):
    """Plain Llama-style multi-head attention for the DeepSeek-V2 decoder.

    STATEFUL: when a ``UnlimitedOcrKVCache`` is threaded in as ``past_key_value``,
    forward appends this layer's rope'd K/V to the cache on decode. use_mla=False
    => head_dim=128, num_heads=num_kv_heads=10 (no GQA); RoPE theta=10000 via the
    HF Llama rotary (NOT the interleaved dead-code variant). q/k/v/o are
    Linear(1280,1280) bias=False.

    Attention is FULL-CAUSAL in both regimes (matching the validated
    prefill/torch reference): prefill = full causal SDPA; decode = the single new
    query attends over ALL cached K/V (exact re-prefill equivalent for causal
    attention). The window-128 ring buffer used by the model's real long-form
    generate() is deliberately NOT implemented here -- it is a separate
    long-form-parity refinement; full-causal cached decode is exact for the
    currently-validated regime.

    ref: $TT_METAL_HOME/models/tt_transformers/tt/attention.py ;
         src/tt_symbiote/models/dots_ocr/dots_ocr_attention.py
    """

    def __init__(self):
        super().__init__()
        self.q_proj = None
        self.k_proj = None
        self.v_proj = None
        self.o_proj = None
        self.rotary_emb = None
        self.layer_idx = None
        self.num_heads = None
        self.head_dim = None
        # KV-cache write position mirror (advanced on prefill/decode). The K/V
        # tensors themselves live on the external UnlimitedOcrKVCache threaded in
        # as past_key_value; this counter is a per-layer observability mirror.
        self._cache_seq_len = 0

    def reset_trace_state(self) -> None:
        # Reset the per-layer write-position mirror to a known baseline before the
        # trace warm-up + capture double-run. The K/V tensors are owned by the
        # external UnlimitedOcrKVCache (past_key_value), whose own reset()/pipeline
        # lifecycle re-baselines the actual device buffers.
        self._cache_seq_len = 0

    @classmethod
    def from_torch(cls, torch_layer, layer_idx=None):
        new = cls()
        new._fallback_torch_layer = torch_layer
        new.layer_idx = layer_idx if layer_idx is not None else getattr(torch_layer, "layer_idx", None)
        new.q_proj = TTNNUnlimitedOcrLinearBf8.from_torch(torch_layer.q_proj)
        # Real-shape re-sweep decode win (matmul_progcfg_lm.csv): apply the M==1
        # 1D program config to lm_qproj on the decode path (20361->11658ns, 1.75x
        # vs AUTO, PCC identical). Inert + M==1-guarded -> safe (decode M is always 1).
        new.q_proj._decode_progcfg = _lm_decode_1d_progcfg(
            (torch_layer.q_proj.in_features // 32, torch_layer.q_proj.out_features // 32)
        )
        new.k_proj = TTNNUnlimitedOcrLinearBf8.from_torch(torch_layer.k_proj)
        new.v_proj = TTNNUnlimitedOcrLinearBf8.from_torch(torch_layer.v_proj)
        new.o_proj = TTNNUnlimitedOcrLinearBf8.from_torch(torch_layer.o_proj)
        # head geometry: use_mla=False => head_dim=128, num_heads=num_kv_heads=10.
        new.head_dim = getattr(torch_layer, "head_dim", None) or 128
        new.num_heads = getattr(torch_layer, "num_heads", None) or (
            torch_layer.q_proj.out_features // new.head_dim
        )
        return new

    def _forward_decode_paged(self, hidden_states, cos, sin, past_key_value,
                              cache_position, H, D, scale):
        # Trace-capturable 1-token paged decode. cache_position is the DEVICE-side
        # cur_pos tensor ([batch] int32) supplied by the pipeline; it is fixed during
        # the trace double-run and advanced OUTSIDE the trace. No host->device writes,
        # no allocation-to-self, no shape changes here -> safe under trace capture.
        import ttnn

        cur_pos_tt = cache_position

        def _proj_head(linear):
            # hidden_states: [1, 1, hidden]. Project then reshape to the decode head
            # layout [1, batch=1, n_heads, head_dim] (natural row-major reshape).
            o = linear.forward(hidden_states)            # [1, 1, H*D]
            if o.dtype != ttnn.bfloat16:
                o = ttnn.typecast(o, ttnn.bfloat16)
            return ttnn.reshape(o, [1, 1, H, D])         # [1, 1, H, D]

        q = _apply_rope(_proj_head(self.q_proj), cos, sin)  # [1,1,H,D]
        k = _apply_rope(_proj_head(self.k_proj), cos, sin)  # [1,1,H,D]
        v = _proj_head(self.v_proj)                          # [1,1,H,D]

        # K/V -> HEIGHT_SHARDED [TILE_SIZE, head_dim] for paged_update_cache.
        sharded_mc = _decode_head_sharded_mem_config(self.device, D)
        k_s = ttnn.to_memory_config(k, sharded_mc)
        v_s = ttnn.to_memory_config(v, sharded_mc)
        ttnn.deallocate(k)
        ttnn.deallocate(v)
        past_key_value.paged_update_on_device(k_s, v_s, layer_idx=self.layer_idx, current_pos=cur_pos_tt)
        ttnn.deallocate(k_s)
        ttnn.deallocate(v_s)

        # paged_sdpa_decode requires Q in DRAM interleaved.
        q = ttnn.to_memory_config(q, ttnn.DRAM_MEMORY_CONFIG)
        attn = past_key_value.paged_sdpa_decode(
            q, self.layer_idx, current_pos=cur_pos_tt, scale=scale,
            compute_kernel_config=_lm_decode_sdpa_compute_kernel_config(),
        )
        ttnn.deallocate(q)

        # attn: [1, batch=1, padded_n_heads, head_dim]. Concat heads -> [1,1,batch,H*D].
        sdpa_out_mc = _decode_head_sharded_mem_config(self.device, D)
        attn = ttnn.to_memory_config(attn, sdpa_out_mc)
        attn = ttnn.experimental.nlp_concat_heads_decode(attn, num_heads=H)
        attn = ttnn.to_memory_config(attn, ttnn.L1_MEMORY_CONFIG)
        attn = ttnn.slice(attn, [0, 0, 0, 0], [1, 1, 1, H * D])
        attn = ttnn.reshape(attn, [1, 1, H * D])
        return self.o_proj.forward(attn)

    @run_on_devices(DeviceArch.P150, DeviceArch.P150x4, DeviceArch.T3K)
    def forward(self, hidden_states, position_embeddings=None, attention_mask=None,
                past_key_value=None, cache_position=None, **kwargs):
        # q/k/v proj -> HF-Llama RoPE (rotate_half) -> SDPA(scale=1/sqrt(head_dim))
        # -> o_proj. Pure ttnn.* ops; children invoked via .forward (bypassing the
        # per-child module_run/arch guard). position_embeddings = (cos, sin), each
        # broadcastable to [1,1,S,head_dim] (S == the query length of THIS call).
        #
        # Two paths, selected by ``past_key_value``:
        #   * None  -> PREFILL-only PCC path (unchanged behaviour): full causal SDPA
        #              over the freshly-computed q/k/v. No cache is touched.
        #   * a UnlimitedOcrKVCache -> FULL-CAUSAL CACHED path:
        #       - first call for this layer (slot empty)  == PREFILL: fill the slot
        #         with the whole prompt's rope'd K/V, then causal SDPA (identical
        #         math to the None path).
        #       - later calls (slot filled, S==1)         == DECODE: append the new
        #         token's rope'd K/V and attend the single query over ALL cached K/V
        #         (is_causal=False -- the query is the newest position, so every
        #         cached key is a valid past/self key). This is mathematically
        #         identical to re-prefilling the full sequence for causal attention.
        import math

        import ttnn

        if position_embeddings is None:
            raise _todo(
                "prefill/decode require position_embeddings=(cos,sin) at the query's "
                "absolute position(s); RoPE position is carried by cos/sin, cache "
                "position by the filled cache length. ",
                "$TT_METAL_HOME/models/tt_transformers/tt/attention.py",
            )
        cos, sin = position_embeddings
        # cos/sin may arrive fp32 (generic tensor-wrap); SDPA/eltwise need bf16.
        if cos.dtype != ttnn.bfloat16:
            cos = ttnn.typecast(cos, ttnn.bfloat16)
        if sin.dtype != ttnn.bfloat16:
            sin = ttnn.typecast(sin, ttnn.bfloat16)
        H, D = self.num_heads, self.head_dim
        shape = list(hidden_states.shape)
        S = shape[-2]
        scale = 1.0 / math.sqrt(D)

        # -------------------------------------------------------------------
        # PAGED DECODE (S == 1, past_key_value is a paged KV cache):
        #   1-token q/k/v -> RoPE at the absolute pos (carried by cos/sin) ->
        #   IN-PLACE paged_update_cache(k,v) at the DEVICE-side cur_pos (cache_position)
        #   -> paged_sdpa_decode(q, cur_pos). Nothing allocates-to-self or changes
        #   shape, so this path is trace-capturable (mirrors dots_ocr
        #   TTNNDotsOCRAttention._forward_decode_paged). cache_position IS the
        #   device-side cur_pos tensor ([batch] int32), fixed during the trace
        #   double-run and advanced OUTSIDE the trace by the pipeline.
        # -------------------------------------------------------------------
        if S == 1 and _is_paged_kv_cache(past_key_value):
            return self._forward_decode_paged(
                hidden_states, cos, sin, past_key_value, cache_position, H, D, scale
            )

        def _proj(linear):
            o = linear.forward(hidden_states)          # [.., S, H*D]
            if o.dtype != ttnn.bfloat16:
                o = ttnn.typecast(o, ttnn.bfloat16)
            o = ttnn.reshape(o, [1, S, H, D])
            return ttnn.permute(o, (0, 2, 1, 3))       # [1, H, S, D]

        def _rope(t):
            return _apply_rope(t, cos, sin)

        q = _rope(_proj(self.q_proj))
        k = _rope(_proj(self.k_proj))
        v = _proj(self.v_proj)

        # PAGED PREFILL fill (S > 1): populate the paged cache for the prompt, then
        # run the SAME full-causal SDPA as the None path (math identical). The rope'd
        # K/V are [1, num_kv_heads, S, head_dim] bf16 TILE -- exactly the
        # paged_fill_cache input contract ([1, num_heads, seq_len, head_dim]).
        if _is_paged_kv_cache(past_key_value):
            k_fill = k if k.layout == ttnn.TILE_LAYOUT else ttnn.to_layout(
                k, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            v_fill = v if v.layout == ttnn.TILE_LAYOUT else ttnn.to_layout(
                v, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            past_key_value.paged_fill_on_device(k_fill, v_fill, layer_idx=self.layer_idx, batch_idx=0)
            attn = ttnn.transformer.scaled_dot_product_attention(
                q, k, v, is_causal=True, scale=scale
            )
            attn = ttnn.permute(attn, (0, 2, 1, 3))        # [1, S, H, D]
            attn = ttnn.reshape(attn, [1, S, H * D])
            return self.o_proj.forward(attn)

        if past_key_value is None:
            # PREFILL-only PCC path (byte-for-byte the original behaviour).
            attn = ttnn.transformer.scaled_dot_product_attention(
                q, k, v, is_causal=True, scale=scale
            )
        else:
            li = self.layer_idx
            if past_key_value.k[li] is None:
                # First call for this layer == PREFILL fill: keep the rope'd K/V as
                # the cache slot (do NOT deallocate -- they ARE the cache), then run
                # the identical full causal SDPA over them.
                past_key_value.k[li] = k
                past_key_value.v[li] = v
                self._cache_seq_len = S
                attn = ttnn.transformer.scaled_dot_product_attention(
                    q, k, v, is_causal=True, scale=scale
                )
            else:
                # DECODE: append the new token's K/V, attend over ALL cached K/V.
                old_k, old_v = past_key_value.k[li], past_key_value.v[li]
                full_k = ttnn.concat([old_k, k], dim=2)
                full_v = ttnn.concat([old_v, v], dim=2)
                ttnn.deallocate(old_k)
                ttnn.deallocate(old_v)
                ttnn.deallocate(k)
                ttnn.deallocate(v)
                past_key_value.k[li] = full_k
                past_key_value.v[li] = full_v
                self._cache_seq_len = int(full_k.shape[2])
                attn = ttnn.transformer.scaled_dot_product_attention(
                    q, full_k, full_v, is_causal=False, scale=scale
                )
        attn = ttnn.permute(attn, (0, 2, 1, 3))        # [1, S, H, D]
        attn = ttnn.reshape(attn, [1, S, H * D])
        return self.o_proj.forward(attn)


class TTNNUnlimitedOcrDeepseekMLP(StatelessTTNNModule):
    """Dense SwiGLU MLP (DeepseekV2MLP): gate/up Linear(1280,6848), down
    Linear(6848,1280), SiLU gate. Used by decoder layer 0 and as the MoE
    shared/routed expert primitive.

    ref: $TT_METAL_HOME/models/tt_transformers/tt/mlp.py ;
         src/tt_symbiote/modules/ttnn_moe.py::TTNNGlm4MoeMLP
    """

    def __init__(self):
        super().__init__()
        self.gate_proj = None
        self.up_proj = None
        self.down_proj = None

    @classmethod
    def from_torch(cls, torch_layer):
        new = cls()
        new._fallback_torch_layer = torch_layer
        new.gate_proj = TTNNUnlimitedOcrLinearBf8.from_torch(torch_layer.gate_proj)
        # Real-shape re-sweep decode win (matmul_progcfg_lm.csv): apply the M==1
        # 1D program config to lm_gate ONLY at the swept shape -- the dense-MLP gate
        # 1280->6848 (decoder layer 0). 33936->31624ns, 1.07x vs AUTO at decode
        # M==1, PCC identical.
        # Inert + M==1-guarded -> safe. This SAME class also builds the MoE routed
        # (inter=896) and shared (inter=1792) expert gates, which were NOT swept
        # (AUTO can beat the formula config, cf. lm_gate_prefill); those keep AUTO.
        if torch_layer.gate_proj.out_features == 6848:
            new.gate_proj._decode_progcfg = _lm_decode_1d_progcfg(
                (torch_layer.gate_proj.in_features // 32,
                 torch_layer.gate_proj.out_features // 32)
            )
        new.up_proj = TTNNUnlimitedOcrLinearBf8.from_torch(torch_layer.up_proj)
        new.down_proj = TTNNUnlimitedOcrLinearBf8.from_torch(torch_layer.down_proj)
        return new

    @run_on_devices(DeviceArch.P150, DeviceArch.P150x4, DeviceArch.T3K)
    def forward(self, hidden_states, **kwargs):
        # SwiGLU: down_proj(silu(gate_proj(x)) * up_proj(x)). Children TTNNLinear
        # forwards are invoked directly (bypassing per-child module_run/arch guard,
        # as in TTNNLayerStack); inputs here are already ttnn tensors on device.
        import ttnn

        gate = self.gate_proj.forward(hidden_states)
        up = self.up_proj.forward(hidden_states)
        gate = ttnn.silu(gate)
        prod = ttnn.multiply(gate, up)
        ttnn.deallocate(gate)
        ttnn.deallocate(up)
        out = self.down_proj.forward(prod)
        ttnn.deallocate(prod)
        return out


class TTNNUnlimitedOcrMoE(StatelessTTNNModule):
    """DeepSeek-V2 mixture-of-experts block (decoder layers 1..11).

    Gate: MoEGate weight[64,1280], fp32 SOFTMAX scoring (scoring_func='softmax',
    NOT sigmoid), greedy top-k k=6, n_group=1 (group logic inert),
    norm_topk_prob=False, routed_scaling_factor=1.0. 64 routed experts
    DeepseekMLP(inter=896) + 2 shared experts DeepseekMLP(inter=1792).

    Hardest-on-TTNN item (dynamic routing / data-dependent shapes); reuse the
    DeepSeek-lineage stack in modules/ttnn_moe.py, adapting the router to softmax
    scoring per bailing_moe_v2's _adapt_config/_consolidate_experts/_adapt_gate.

    ref: src/tt_symbiote/modules/ttnn_moe.py ;
         src/tt_symbiote/models/bailing_moe_v2/
    """

    def __init__(self):
        super().__init__()
        # Router (softmax-scoring, greedy top-k) parameters.
        self._gate_weight_torch = None   # torch [n_routed_experts, hidden]
        self.tt_gate_weight = None       # ttnn [hidden, n_routed_experts] (fp32, on device)
        self.top_k = 6
        self.n_routed_experts = 64
        self.norm_topk_prob = False
        self.routed_scaling_factor = 1.0
        # Expert primitives (each a green TTNNUnlimitedOcrDeepseekMLP).
        self.experts = []               # list of routed experts (inter=896)
        self.shared_experts = None      # single dense MLP (inter=1792) or None
        # Stacked routed-expert weights for the grouped decode (T==1) path: the
        # 64-expert loop is replaced by 3 batched matmuls over these tensors.
        self._w_gate_3d_torch = None    # [1,E,H,I]
        self._w_up_3d_torch = None      # [1,E,H,I]
        self._w_down_3d_torch = None    # [1,E,I,H]
        self.tt_w_gate_3d = None
        self.tt_w_up_3d = None
        self.tt_w_down_3d = None

    @classmethod
    def from_torch(cls, torch_layer):
        import torch

        new = cls()
        new._fallback_torch_layer = torch_layer

        # --- Router gate (MoEGate): softmax scoring, greedy top-k -----------------
        gate = getattr(torch_layer, "gate", None)
        if gate is None or not hasattr(gate, "weight"):
            raise ValueError("TTNNUnlimitedOcrMoE.from_torch: torch MoE has no gate.weight")
        new._gate_weight_torch = gate.weight.detach().float()  # [n_experts, hidden]
        new.n_routed_experts = int(new._gate_weight_torch.shape[0])
        new.top_k = int(
            getattr(gate, "top_k", None)
            or getattr(torch_layer, "num_experts_per_tok", None)
            or 6
        )
        new.norm_topk_prob = bool(getattr(gate, "norm_topk_prob", False))
        new.routed_scaling_factor = float(getattr(gate, "routed_scaling_factor", 1.0) or 1.0)

        # --- Routed experts + shared experts (DeepseekV2MLP -> green SwiGLU) -------
        experts = getattr(torch_layer, "experts", [])
        new.experts = [TTNNUnlimitedOcrDeepseekMLP.from_torch(e) for e in experts]
        shared = getattr(torch_layer, "shared_experts", None)
        new.shared_experts = (
            TTNNUnlimitedOcrDeepseekMLP.from_torch(shared) if shared is not None else None
        )

        # Stack routed-expert weights into 3D tensors for the grouped decode path.
        # torch nn.Linear weight is [out, in]; ttnn.matmul(x, W) wants [in, out], so
        # transpose each expert then stack over the expert dim and add a leading 1.
        if experts:
            wg = torch.stack(
                [e.gate_proj.weight.detach().t().contiguous() for e in experts]
            )  # [E, H, I]
            wu = torch.stack(
                [e.up_proj.weight.detach().t().contiguous() for e in experts]
            )  # [E, H, I]
            wd = torch.stack(
                [e.down_proj.weight.detach().t().contiguous() for e in experts]
            )  # [E, I, H]
            new._w_gate_3d_torch = wg.unsqueeze(0)  # [1, E, H, I]
            new._w_up_3d_torch = wu.unsqueeze(0)    # [1, E, H, I]
            new._w_down_3d_torch = wd.unsqueeze(0)  # [1, E, I, H]
        return new

    def preprocess_weights_impl(self):
        import ttnn

        # Gate weight in fp32 for a HiFi4 router matmul: torch nn.Linear-style
        # weight is [n_experts, hidden]; ttnn.linear expects [hidden, n_experts].
        w = self._gate_weight_torch.t().contiguous()  # [hidden, n_experts]
        self.tt_gate_weight = ttnn.from_torch(w, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)
        # Stacked routed-expert weights for the grouped decode path, bfloat8_b to
        # match the per-expert dense loop's dtype. Kept alongside the per-expert
        # children, which serve the prefill dense loop.
        if self._w_gate_3d_torch is not None:
            self.tt_w_gate_3d = ttnn.from_torch(
                self._w_gate_3d_torch, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT
            )
            self.tt_w_up_3d = ttnn.from_torch(
                self._w_up_3d_torch, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT
            )
            self.tt_w_down_3d = ttnn.from_torch(
                self._w_down_3d_torch, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT
            )
        # Routed/shared expert children are preprocessed independently by set_device
        # (they live in a list / attr and are visited by the graph walk).
        return self

    def move_weights_to_device_impl(self):
        import ttnn

        self.tt_gate_weight = ttnn.to_device(self.tt_gate_weight, self.device)
        if self.tt_w_gate_3d is not None:
            self.tt_w_gate_3d = ttnn.to_device(self.tt_w_gate_3d, self.device)
            self.tt_w_up_3d = ttnn.to_device(self.tt_w_up_3d, self.device)
            self.tt_w_down_3d = ttnn.to_device(self.tt_w_down_3d, self.device)
        return self

    @run_on_devices(DeviceArch.P150, DeviceArch.P150x4, DeviceArch.T3K)
    def forward(self, hidden_states, **kwargs):
        # DeepSeek-V2 single-device MoE (softmax scoring, greedy top-k, no groups):
        #   scores = softmax(x_fp32 @ gate.T)            [T, E]
        #   (w_e, idx_e) = topk(scores, k)               top-k per token
        #   G[t, e]      = scores[t, e] if e in top-k(t) else 0  (routed_scaling * ...)
        #   out[t]       = sum_e G[t, e] * expert_e(x[t]) + shared_experts(x[t])
        # Dense expert compute (E small-matmul triples) is robust + correctness-first.
        import ttnn

        E, K = self.n_routed_experts, self.top_k
        orig_shape = list(hidden_states.shape)
        hidden = orig_shape[-1]
        T = 1
        for d in orig_shape[:-1]:
            T *= d

        if hidden_states.layout != ttnn.TILE_LAYOUT:
            hidden_states = ttnn.to_layout(
                hidden_states, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
        x2d = ttnn.reshape(hidden_states, [T, hidden])

        # --- Router: fp32 logits -> fp32 softmax scores ---------------------------
        x_f32 = x2d if x2d.dtype == ttnn.float32 else ttnn.typecast(x2d, ttnn.float32)
        logits = ttnn.linear(
            x_f32,
            self.tt_gate_weight,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=True,
            ),
        )
        if x_f32 is not x2d:
            ttnn.deallocate(x_f32)
        logits = ttnn.reshape(logits, [1, 1, T, E])
        scores_f32 = ttnn.softmax(logits, dim=-1)
        ttnn.deallocate(logits)

        # --- Greedy top-k selection ------------------------------------------------
        # ttnn.topk needs bf16; select indices there, but recover the exact fp32
        # top-k scores (via gather) so the routing weights keep full precision.
        # A dense gate matrix is built by thresholding (ttnn.scatter rejects fp32
        # tiled tensors): G[t,e] = scores[t,e] where scores[t,e] >= kth-largest(t).
        scores_bf16 = ttnn.typecast(scores_f32, ttnn.bfloat16)
        _, topk_idx = ttnn.topk(scores_bf16, k=K, dim=3, largest=True, sorted=True)
        ttnn.deallocate(scores_bf16)
        topk_w = ttnn.gather(scores_f32, dim=3, index=topk_idx)  # [1,1,T,K] fp32
        ttnn.deallocate(topk_idx)

        if self.norm_topk_prob and K > 1:
            denom = ttnn.sum(topk_w, dim=3, keepdim=True)
            denom = ttnn.add(denom, 1e-20)
            gate_scale = ttnn.reciprocal(denom)
            ttnn.deallocate(denom)
        else:
            gate_scale = None

        # kth-largest score per token = min over the K selected (sorted desc).
        threshold = ttnn.min(topk_w, dim=3, keepdim=True)  # [1,1,T,1] fp32
        ttnn.deallocate(topk_w)
        mask = ttnn.ge(scores_f32, threshold)               # [1,1,T,E] (1.0/0.0)
        ttnn.deallocate(threshold)
        gate_dense = ttnn.multiply(scores_f32, mask)
        ttnn.deallocate(scores_f32)
        ttnn.deallocate(mask)
        if gate_scale is not None:
            gate_dense = ttnn.multiply(gate_dense, gate_scale)
            ttnn.deallocate(gate_scale)
        if self.routed_scaling_factor != 1.0:
            gate_dense = ttnn.multiply(gate_dense, self.routed_scaling_factor)
        # NOTE: ttnn.reshape here returns a VIEW aliasing gate_dense's buffer, so we
        # must NOT deallocate gate_dense while gate_2d is live (doing so is a
        # use-after-free that silently corrupts the routing gates -> full-model MoE
        # PCC collapses from 0.9999 to ~0.91). The single final deallocate(gate_2d)
        # below frees the shared buffer once, after the expert loop.
        if T == 1 and self.tt_w_gate_3d is not None:
            # Decode (T==1): grouped experts -- 3 batched matmuls over the stacked
            # expert weights + one weighted sum, replacing the 64-expert loop. All E
            # experts are computed; the gate_dense mask (0 for non-selected) makes the
            # weighted sum equal the top-k accumulate. Prefill (T>1) uses the dense loop.
            ck = _lm_compute_kernel_config()
            x4 = ttnn.reshape(x2d, [1, 1, T, hidden])
            x_rep = ttnn.repeat(x4, ttnn.Shape([1, E, 1, 1]))       # [1,E,T,hidden]
            g = ttnn.matmul(x_rep, self.tt_w_gate_3d, compute_kernel_config=ck,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG)   # [1,E,T,inter]
            u = ttnn.matmul(x_rep, self.tt_w_up_3d, compute_kernel_config=ck,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(x_rep)
            inter = ttnn.multiply(ttnn.silu(g), u)                  # [1,E,T,inter]
            ttnn.deallocate(g)
            ttnn.deallocate(u)
            d = ttnn.matmul(inter, self.tt_w_down_3d, compute_kernel_config=ck,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG)   # [1,E,T,hidden]
            ttnn.deallocate(inter)
            gate_perm = ttnn.permute(gate_dense, (0, 3, 2, 1))       # [1,E,T,1]
            if gate_perm.dtype != d.dtype:
                gate_perm = ttnn.typecast(gate_perm, d.dtype)
            weighted = ttnn.multiply(d, gate_perm)                   # [1,E,T,hidden]
            ttnn.deallocate(d)
            ttnn.deallocate(gate_perm)
            summed = ttnn.sum(weighted, dim=1, keepdim=True)         # [1,1,T,hidden]
            ttnn.deallocate(weighted)
            out = ttnn.reshape(summed, [T, hidden])
        else:
            gate_2d = ttnn.reshape(gate_dense, [T, E])  # [T, E]

            # --- Dense routed-expert compute + weighted accumulate ----------------
            out = None
            for e, expert in enumerate(self.experts):
                g_e = ttnn.slice(gate_2d, [0, e], [T, e + 1])  # [T, 1] fp32
                y_e = expert.forward(x2d)                       # [T, hidden]
                if g_e.dtype != y_e.dtype:
                    g_e = ttnn.typecast(g_e, y_e.dtype)
                term = ttnn.multiply(y_e, g_e)                  # broadcast [T,1] over hidden
                ttnn.deallocate(y_e)
                ttnn.deallocate(g_e)
                if out is None:
                    out = term
                else:
                    out = ttnn.add(out, term)
                    ttnn.deallocate(term)
            ttnn.deallocate(gate_2d)

        # --- Shared experts (added to every token) --------------------------------
        if self.shared_experts is not None:
            shared = self.shared_experts.forward(x2d)
            if out is None:
                out = shared
            else:
                if shared.dtype != out.dtype:
                    shared = ttnn.typecast(shared, out.dtype)
                out = ttnn.add(out, shared)
                ttnn.deallocate(shared)

        return ttnn.reshape(out, orig_shape)


class TTNNUnlimitedOcrMlpProjector(StatelessTTNNModule):
    """Vision->text projector: Linear(2048, 1280) applied to
    cat(clip[:,1:], sam.flatten) features -> [B,256,1280].

    ref: src/tt_symbiote/modules/ttnn_linear.py::TTNNLinear
    """

    def __init__(self):
        super().__init__()
        self.proj = None

    @classmethod
    def from_torch(cls, torch_layer):
        new = cls()
        new._fallback_torch_layer = torch_layer
        # torch_layer may be an nn.Linear or an nn.Sequential/module wrapping one.
        linear = torch_layer if hasattr(torch_layer, "weight") else getattr(torch_layer, "layers", torch_layer)
        # fp32-acc linear (HiFi4 + fp32 dest-accum): strictly improves vs the LoFi
        # default; the vision path enables genuine fp32 accumulation on all matmuls.
        new.proj = TTNNUnlimitedOcrLinearBf8.from_torch(linear)
        return new

    @run_on_devices(DeviceArch.P150, DeviceArch.P150x4, DeviceArch.T3K)
    def forward(self, features, **kwargs):
        # Vision->text projector: a single Linear(2048 -> 1280) over the 256
        # concatenated CLIP+SAM tokens. Inlines the child TTNNLinear.forward body so
        # the matmul carries the swept EXPLICIT 2D program config
        # (_projector_matmul_program_config, ~2.24x vs ttnn-AUTO). Weights/bias/compute
        # config are the child's committed bf8 weight + HiFi4/fp32-accum kernel --
        # ONLY the program_config is added; program config is numerically inert.
        import ttnn

        lin = self.proj
        input_tensor = features
        if input_tensor.layout != ttnn.TILE_LAYOUT:
            input_tensor = ttnn.to_layout(
                input_tensor, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
        input_tensor_shape = list(input_tensor.shape)
        input_shape = list(input_tensor_shape)
        # Prepend the rank-padding 1s so the leading token dim stays the matmul M
        # (not a batch): a 2D [M, K] must become [1, 1, M, K], else ttnn.linear runs
        # M tiny per-token batched matmuls instead of one full-grid matmul.
        while len(input_shape) < 4:
            input_shape.insert(0, 1)
        input_tensor = ttnn.reshape(input_tensor, input_shape)
        tt_output = ttnn.linear(
            input_tensor,
            lin.tt_weight,
            bias=lin.tt_bias,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=_lm_compute_kernel_config(),
            program_config=_projector_matmul_program_config(),
        )
        tt_output = ttnn.reshape(tt_output, input_tensor_shape[:-1] + [lin.out_features])
        return tt_output


# =============================================================================
# Vision helpers (pure-ttnn building blocks reused by SAM + CLIP)
# =============================================================================
_VISION_ARCHS = (DeviceArch.P150, DeviceArch.P150x4, DeviceArch.T3K)


def _ln(norm, x, eps):
    """Inline LayerNorm using a child TTNNLayerNorm's device weights + explicit eps.

    The reused TTNNLayerNorm carries a T3K-only @run_on_devices guard on its forward,
    so (as with the RMSNorm in the decoder layer) we invoke ttnn.layer_norm directly
    with the child's set_device-prepared tt_weight / tt_bias.
    """
    import ttnn

    if x.layout != ttnn.TILE_LAYOUT:
        x = ttnn.to_layout(x, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    return ttnn.layer_norm(x, weight=norm.tt_weight, bias=norm.tt_bias, epsilon=eps)


def _window_partition(x, ws):
    """[1,H,W,C] -> ([num_windows, ws, ws, C], (Hp, Wp)). Pure ttnn (fixed-res)."""
    import ttnn

    B = x.shape[0]
    H, W, C = x.shape[1], x.shape[2], x.shape[3]
    pad_h = (ws - H % ws) % ws
    pad_w = (ws - W % ws) % ws
    x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
    if pad_h > 0 or pad_w > 0:
        x = ttnn.pad(x, [(0, 0), (0, pad_h), (0, pad_w), (0, 0)], value=0.0)
    Hp, Wp = H + pad_h, W + pad_w
    x = ttnn.reshape(x, [B, Hp // ws, ws, Wp // ws, ws, C])
    x = ttnn.permute(x, (0, 1, 3, 2, 4, 5))
    x = ttnn.reshape(x, [B * (Hp // ws) * (Wp // ws), ws, ws, C])
    return x, (Hp, Wp)


def _window_unpartition(windows, ws, Hp, Wp, H, W):
    """[num_windows, ws, ws, C] -> [1, H, W, C] (crop padding). Pure ttnn."""
    import ttnn

    C = windows.shape[3]
    B = windows.shape[0] // ((Hp // ws) * (Wp // ws))
    windows = ttnn.to_layout(windows, ttnn.ROW_MAJOR_LAYOUT)
    x = ttnn.reshape(windows, [B, Hp // ws, Wp // ws, ws, ws, C])
    x = ttnn.permute(x, (0, 1, 3, 2, 4, 5))
    x = ttnn.reshape(x, [B, Hp, Wp, C])
    if Hp > H or Wp > W:
        x = ttnn.slice(x, [0, 0, 0, 0], [B, H, W, C])
    return x


# =============================================================================
# Tier 1 — vision leaf ops (custom activation / normalization)
# =============================================================================
class TTNNUnlimitedOcrQuickGELU(StatelessTTNNModule):
    """CLIP quick_gelu = x * sigmoid(1.702 * x). NON-standard (do NOT use ttnn.gelu)."""

    @classmethod
    def from_torch(cls, torch_layer=None):
        new = cls()
        new._fallback_torch_layer = torch_layer
        return new

    @run_on_devices(*_VISION_ARCHS)
    def forward(self, x, **kwargs):
        import ttnn

        return ttnn.multiply(x, ttnn.sigmoid(ttnn.multiply(x, 1.702)))


class TTNNUnlimitedOcrLayerNorm2d(StatelessTTNNModule):
    """SAM neck LayerNorm2d (channel-first LN over dim=1 on NCHW).

    We keep the whole SAM tower in NHWC, so the channel axis is the LAST dim and a
    channel-first LayerNorm2d is exactly a standard ttnn.layer_norm over the last dim.
    """

    def __init__(self):
        super().__init__()
        self._weight_torch = None
        self._bias_torch = None
        self.eps = 1e-6
        self.tt_weight = None
        self.tt_bias = None

    @classmethod
    def from_torch(cls, ln2d):
        new = cls()
        new._fallback_torch_layer = ln2d
        new._weight_torch = ln2d.weight.detach().float()
        new._bias_torch = ln2d.bias.detach().float()
        new.eps = float(getattr(ln2d, "eps", 1e-6))
        return new

    def preprocess_weights_impl(self):
        import ttnn

        self.tt_weight = ttnn.from_torch(self._weight_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        self.tt_bias = ttnn.from_torch(self._bias_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        return self

    def move_weights_to_device_impl(self):
        import ttnn

        self.tt_weight = ttnn.to_device(self.tt_weight, self.device)
        self.tt_bias = ttnn.to_device(self.tt_bias, self.device)
        return self

    @run_on_devices(*_VISION_ARCHS)
    def forward(self, x, **kwargs):
        import ttnn

        if x.layout != ttnn.TILE_LAYOUT:
            x = ttnn.to_layout(x, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return ttnn.layer_norm(x, weight=self.tt_weight, bias=self.tt_bias, epsilon=self.eps)


# =============================================================================
# Tier 2 — vision composites (SAM + CLIP attention / FFN)
# =============================================================================
class TTNNUnlimitedOcrClipFFN(StatelessTTNNModule):
    """CLIP NoTPFeedForward: fc2(quick_gelu(fc1(x))). fc1 1024->4096, fc2 4096->1024."""

    def __init__(self):
        super().__init__()
        self.fc1 = None
        self.fc2 = None

    @classmethod
    def from_torch(cls, torch_layer):
        new = cls()
        new._fallback_torch_layer = torch_layer
        # fp32-acc linears (bias applied identically to the shared TTNNLinear).
        new.fc1 = TTNNUnlimitedOcrLinearBf8.from_torch(torch_layer.fc1)
        new.fc2 = TTNNUnlimitedOcrLinearBf8.from_torch(torch_layer.fc2)
        return new

    @run_on_devices(*_VISION_ARCHS)
    def forward(self, x, **kwargs):
        import ttnn

        h = self.fc1.forward(x)
        h = ttnn.multiply(h, ttnn.sigmoid(ttnn.multiply(h, 1.702)))  # quick_gelu
        return self.fc2.forward(h)


class TTNNUnlimitedOcrClipAttention(StatelessTTNNModule):
    """CLIP-L attention (NoTPAttention, plain non-causal SDPA). width=1024, heads=16.

    Fused qkv_proj Linear(1024,3072,bias=True); head_dim=64; out_proj Linear(1024,1024).
    Mirrors the greened LlamaMHA prefill minus RoPE and with is_causal=False, no mask.

    ref: $TT_METAL_HOME/tech_reports/ViT-TTNN/vit.md ; deepencoder.py::NoTPAttention
    """

    def __init__(self):
        super().__init__()
        self.qkv_proj = None
        self.out_proj = None
        self.num_heads = None
        self.head_dim = None

    @classmethod
    def from_torch(cls, torch_layer):
        new = cls()
        new._fallback_torch_layer = torch_layer
        # fp32-acc linears (bias applied identically to the shared TTNNLinear).
        new.qkv_proj = TTNNUnlimitedOcrLinearBf8.from_torch(torch_layer.qkv_proj)
        new.out_proj = TTNNUnlimitedOcrLinearBf8.from_torch(torch_layer.out_proj)
        new.num_heads = int(torch_layer.num_heads)
        new.head_dim = int(
            getattr(torch_layer, "head_dim", None) or (torch_layer.out_proj.in_features // int(torch_layer.num_heads))
        )
        return new

    @run_on_devices(*_VISION_ARCHS)
    def forward(self, hidden_states, **kwargs):
        import math

        import ttnn

        H, D = self.num_heads, self.head_dim
        C = H * D
        S = hidden_states.shape[-2]
        qkv = self.qkv_proj.forward(hidden_states)  # [1, S, 3C]
        if qkv.dtype != ttnn.bfloat16:
            qkv = ttnn.typecast(qkv, ttnn.bfloat16)
        # FUSED head split: nlp_create_qkv_heads replaces the 3x (slice+reshape+
        # permute) manual split. Input contract is [B, 1, S, 3*head_dim*num_heads];
        # the [Q;K;V] block order + per-head [S,H,D]->[H,S,D] shuffle is identical
        # to the manual _head, so it is numerically bit-close (only fewer TM ops).
        qkv = ttnn.reshape(qkv, [1, 1, S, 3 * C])
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            qkv, num_heads=H, num_kv_heads=H, transpose_k_heads=False,
        )
        ttnn.deallocate(qkv)
        attn = ttnn.transformer.scaled_dot_product_attention(
            q, k, v, is_causal=False, scale=1.0 / math.sqrt(D),
            compute_kernel_config=_vision_sdpa_compute_kernel_config(),
        )
        ttnn.deallocate(q)
        ttnn.deallocate(k)
        ttnn.deallocate(v)
        # FUSED head merge: nlp_concat_heads [1,H,S,D] -> [1,1,S,C] (replaces the
        # permute+reshape). Reshape to [1,S,C] for out_proj (free RM reshape).
        attn = ttnn.experimental.nlp_concat_heads(attn)  # [1, 1, S, C]
        attn = ttnn.reshape(attn, [1, S, C])
        return self.out_proj.forward(attn)


class TTNNUnlimitedOcrSamAttention(StatelessTTNNModule):
    """SAM ViT-B attention with decomposed relative-position bias.

    12 heads, head_dim 64, fused qkv Linear(768,2304,bias=True), proj Linear(768,768).
    The decomposed rel-pos indexing (get_rel_pos) is PRE-BAKED at fixed resolution in
    preprocess_weights (no F.interpolate); the two einsums are computed in forward as
    grid-batched ttnn matmuls and folded into an SDPA attn_mask (matching the reference
    F.scaled_dot_product_attention(q,k,v,attn_mask=rel_h+rel_w) call exactly).

    Input is [Bn, g, g, C] (window layer: Bn=num_windows, g=window_size; global: Bn=1,
    g=64). Windowing is performed by the owning block.

    ref: deepencoder.py::Attention + add_decomposed_rel_pos ; SDPA attn_mask API
    """

    def __init__(self):
        super().__init__()
        self.qkv = None
        self.proj = None
        self.num_heads = None
        self.head_dim = None
        self.use_rel_pos = True
        self.grid = None
        self._rel_pos_h_torch = None
        self._rel_pos_w_torch = None
        self.tt_RhT = None  # [g, hd, g] pre-baked get_rel_pos(h) transposed
        self.tt_RwT = None

    @classmethod
    def from_torch(cls, attn):
        new = cls()
        new._fallback_torch_layer = attn
        # fp32-acc linears (bias applied identically to the shared TTNNLinear).
        new.qkv = TTNNUnlimitedOcrLinearBf8.from_torch(attn.qkv)
        new.proj = TTNNUnlimitedOcrLinearBf8.from_torch(attn.proj)
        new.num_heads = int(attn.num_heads)
        new.head_dim = int(attn.qkv.in_features // attn.num_heads)
        new.use_rel_pos = bool(getattr(attn, "use_rel_pos", True))
        if new.use_rel_pos:
            new._rel_pos_h_torch = attn.rel_pos_h.detach().float()  # [2g-1, hd]
            new._rel_pos_w_torch = attn.rel_pos_w.detach().float()
            new.grid = (new._rel_pos_h_torch.shape[0] + 1) // 2
        return new

    def preprocess_weights_impl(self):
        import torch
        import ttnn

        if self.use_rel_pos:
            g = self.grid
            # get_rel_pos with q_size == k_size: relative_coords[i,j] = i - j + (g-1),
            # no interpolation (rel_pos length already == 2g-1). Rh[i,j,c]=rel_pos_h[coords].
            idx = (torch.arange(g)[:, None] - torch.arange(g)[None, :] + (g - 1)).long()  # [g,g]
            Rh = self._rel_pos_h_torch[idx]  # [g, g, hd]
            Rw = self._rel_pos_w_torch[idx]  # [g, g, hd]
            RhT = Rh.permute(0, 2, 1).contiguous()  # [g, hd, g]
            RwT = Rw.permute(0, 2, 1).contiguous()
            self.tt_RhT = ttnn.from_torch(RhT, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
            self.tt_RwT = ttnn.from_torch(RwT, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        return self

    def move_weights_to_device_impl(self):
        import ttnn

        if self.use_rel_pos:
            self.tt_RhT = ttnn.to_device(self.tt_RhT, self.device)
            self.tt_RwT = ttnn.to_device(self.tt_RwT, self.device)
        return self

    def _rel_pos_bias(self, q, Bn, nh, g, hd):
        # q: [Bn, nh, HW, hd]  ->  attn_bias [Bn, nh, HW, HW] (HW = g*g)
        import ttnn

        HW = g * g
        B_ = Bn * nh
        q_rm = ttnn.to_layout(q, ttnn.ROW_MAJOR_LAYOUT)
        r_q = ttnn.reshape(q_rm, [B_, g, g, hd])  # [B_, h, w, c]

        # rel_h = einsum("bhwc,hkc->bhwk"): batch the matmul over the query-row index h.
        rq_h = ttnn.permute(r_q, (1, 0, 2, 3))  # [h, B_, w, c]
        rq_h = ttnn.reshape(rq_h, [g, B_ * g, hd])
        rq_h = ttnn.to_layout(rq_h, ttnn.TILE_LAYOUT)
        # fp32 dest-accum on the rel-pos matmul (pure matmul -> fp32 acc is safe
        # and improves precision, unlike the flash-SDPA fp32-dest kernel bug).
        rel_h = ttnn.matmul(rq_h, self.tt_RhT, compute_kernel_config=_lm_compute_kernel_config())  # [h, B_*w, k_h]
        rel_h = ttnn.to_layout(rel_h, ttnn.ROW_MAJOR_LAYOUT)
        rel_h = ttnn.reshape(rel_h, [g, B_, g, g])  # [h, B_, w, k_h]
        rel_h = ttnn.permute(rel_h, (1, 0, 2, 3))  # [B_, h, w, k_h]
        rel_h = ttnn.reshape(rel_h, [B_, HW, g])  # [B_, HW, k_h]

        # rel_w = einsum("bhwc,wkc->bhwk"): batch the matmul over the query-col index w.
        rq_w = ttnn.permute(r_q, (2, 0, 1, 3))  # [w, B_, h, c]
        rq_w = ttnn.reshape(rq_w, [g, B_ * g, hd])
        rq_w = ttnn.to_layout(rq_w, ttnn.TILE_LAYOUT)
        rel_w = ttnn.matmul(rq_w, self.tt_RwT, compute_kernel_config=_lm_compute_kernel_config())  # [w, B_*h, k_w]
        rel_w = ttnn.to_layout(rel_w, ttnn.ROW_MAJOR_LAYOUT)
        rel_w = ttnn.reshape(rel_w, [g, B_, g, g])  # [w, B_, h, k_w]
        rel_w = ttnn.permute(rel_w, (1, 2, 0, 3))  # [B_, h, w, k_w]
        rel_w = ttnn.reshape(rel_w, [B_, HW, g])  # [B_, HW, k_w]

        # attn_bias[b, hw, kh, kw] = rel_h[b,hw,kh] + rel_w[b,hw,kw] -> [B_, HW, g, g],
        # flattened (kh,kw)->HW_k and reshaped to the SDPA-mask [Bn, nh, HW, HW] (TILE).
        # ttnn.add broadcasts [.,g,1] + [.,1,g] -> [.,g,g] natively (verified PCC 0.99999);
        # the two explicit ttnn.repeat expansions to the full [.,g,g] are REDUNDANT (removed).
        #
        # LAYOUT LEVER for the broadcast-add (measured on Blackhole P150, tt-metal
        # a0b506c780979538b6d2fc1e57fdbfdfdabc7e31; add+flatten stage, device kernel ns):
        #   GLOBAL (B_=12, HW=4096, g=64): the ROW_MAJOR double-broadcast add costs
        #     21.4ms; the same add in TILE costs 4.5ms. Doing the add in TILE and
        #     flattening [.,g,g]->[Bn,nh,HW,HW] via a single untilize (RM reshape is
        #     free once contiguous) then re-tilize beats the RM add end-to-end:
        #     29.8ms -> 20.8ms per global layer (-9ms x 4 global layers = -36ms).
        #     Tilizing the tile-aligned [B_,HW,g] (g=64) then TILE-reshaping to the
        #     broadcast shapes avoids the val-padded tilize of [.,g,1] (P3r).
        #   WINDOW (B_=300, HW=196, g=14): g=14 is not tile-aligned, so the TILE path
        #     pays a val-padded (14->32) tilize/untilize round-trip that makes it SLOWER
        #     (6.4ms RM vs 9.2ms TILE). Window keeps the ROW_MAJOR add.
        # Both paths are numerically identical (only layout differs; bf16 tilize/untilize
        # are exact) -- PCC bit-close to the RM baseline.
        if HW > 1024:
            # GLOBAL: TILE broadcast-add, untilize-flatten, re-tilize (P3r).
            rel_h = ttnn.to_layout(rel_h, ttnn.TILE_LAYOUT)
            rel_w = ttnn.to_layout(rel_w, ttnn.TILE_LAYOUT)
            rel_h = ttnn.reshape(rel_h, [B_, HW, g, 1])
            rel_w = ttnn.reshape(rel_w, [B_, HW, 1, g])
            bias = ttnn.add(rel_h, rel_w)  # TILE [B_, HW, g, g]
            bias = ttnn.to_layout(bias, ttnn.ROW_MAJOR_LAYOUT)
            bias = ttnn.reshape(bias, [Bn, nh, HW, HW])
            return ttnn.to_layout(bias, ttnn.TILE_LAYOUT)
        # WINDOW: ROW_MAJOR broadcast-add (g not tile-aligned).
        rel_h = ttnn.reshape(rel_h, [B_, HW, g, 1])
        rel_w = ttnn.reshape(rel_w, [B_, HW, 1, g])
        bias = ttnn.add(rel_h, rel_w)  # RM [B_, HW, g, g]
        bias = ttnn.reshape(bias, [Bn, nh, HW, HW])
        return ttnn.to_layout(bias, ttnn.TILE_LAYOUT)

    @run_on_devices(*_VISION_ARCHS)
    def forward(self, x, **kwargs):
        import math

        import ttnn

        Bn, g = x.shape[0], x.shape[1]
        nh, hd = self.num_heads, self.head_dim
        C = nh * hd
        HW = g * g
        if x.dtype != ttnn.bfloat16:
            x = ttnn.typecast(x, ttnn.bfloat16)
        x2 = ttnn.reshape(x, [Bn, HW, C])
        if Bn > 1:
            # Window-fold: the per-token QKV projection would otherwise run as Bn small
            # per-window batched matmuls; fold the windows into M ([1, Bn*HW, C]) for a
            # single full-grid matmul. HW is not tile-aligned, so fold via ROW_MAJOR.
            xr = ttnn.to_layout(x2, ttnn.ROW_MAJOR_LAYOUT)
            xr = ttnn.reshape(xr, [1, Bn * HW, C])
            xr = ttnn.to_layout(xr, ttnn.TILE_LAYOUT)
            qkv = self.qkv.forward(xr)  # [1, Bn*HW, 3C]
            qkv = ttnn.to_layout(qkv, ttnn.ROW_MAJOR_LAYOUT)
            qkv = ttnn.reshape(qkv, [Bn, HW, 3 * C])
            qkv = ttnn.to_layout(qkv, ttnn.TILE_LAYOUT)
        else:
            qkv = self.qkv.forward(x2)  # global (Bn=1): already a single full-grid matmul
        if qkv.dtype != ttnn.bfloat16:
            qkv = ttnn.typecast(qkv, ttnn.bfloat16)
        # FUSED head split: nlp_create_qkv_heads replaces the 3x (slice+reshape+
        # permute). Input contract [B, 1, S, 3*head_dim*num_heads]; the [Q;K;V] block
        # order + per-head shuffle is identical to the manual _head, so q/k/v (hence
        # the rel-pos bias computed from q) are numerically bit-close.
        qkv = ttnn.reshape(qkv, [Bn, 1, HW, 3 * C])
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            qkv, num_heads=nh, num_kv_heads=nh, transpose_k_heads=False,
        )  # each [Bn, nh, HW, hd]
        ttnn.deallocate(qkv)

        attn_mask = self._rel_pos_bias(q, Bn, nh, g, hd) if self.use_rel_pos else None
        # Precision lift: for the WINDOW layers (small HW) run attention manually in
        # fp32 (QK^T -> +rel_pos_bias -> softmax -> AV) instead of the bf16 flash-SDPA
        # kernel, whose bf16 q/k/v error compounds across the 12-deep SAM tower. GLOBAL
        # layers (HW=4096) keep flash-SDPA -- fp32 scores [Bn,nh,4096,4096] would OOM.
        scale = 1.0 / math.sqrt(hd)
        if HW <= 1024:
            ck = _lm_compute_kernel_config()
            qf = ttnn.typecast(q, ttnn.float32)
            kf = ttnn.typecast(k, ttnn.float32)
            vf = ttnn.typecast(v, ttnn.float32)
            kT = ttnn.permute(kf, (0, 1, 3, 2))  # [Bn,nh,hd,HW]
            scores = ttnn.matmul(qf, kT, compute_kernel_config=ck)  # [Bn,nh,HW,HW]
            scores = ttnn.multiply(scores, scale)
            if attn_mask is not None:
                scores = ttnn.add(scores, ttnn.typecast(attn_mask, ttnn.float32))
            probs = ttnn.softmax(scores, dim=-1)
            out = ttnn.matmul(probs, vf, compute_kernel_config=ck)  # [Bn,nh,HW,hd]
            out = ttnn.typecast(out, ttnn.bfloat16)
        else:
            # GLOBAL layers (HW=4096): flash-SDPA. ACCURACY LEVER (measured on
            # Blackhole P150, tt-metal a0b506c780979538b6d2fc1e57fdbfdfdabc7e31):
            # the flash-softmax exponent defaults to the APPROXIMATE mode
            # (SDPAProgramConfig.exp_approx_mode defaults to True in
            # sdpa_program_factory.cpp::get_exp_approx_mode). Passing an explicit
            # program_config with ``exp_approx_mode=False`` and matched q/k chunk
            # sizes uses the exact exponent and raises PCC. Standalone probe vs
            # torch-fp32 on the real global shape [1,12,4096,64]+rel-pos mask:
            #   default (exp_approx=True) ................ 0.999891
            #   exp_approx=False, q256/k256 (matched) .... 0.999918  <-- BEST
            #   exp_approx=False, q256/k512 (mismatched) . 0.999832  <-- worse
            # Chunk sizes MUST be equal (mismatched regresses) and divisible by
            # TILE_WIDTH (32); 4096 is a multiple of 256. Orthogonal to the
            # fp32_dest_acc kernel bug -- accumulator stays bf16 (see
            # _vision_sdpa_compute_kernel_config).
            grid = self.device.compute_with_storage_grid_size()
            sdpa_prog_cfg = ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=grid,
                q_chunk_size=256,
                k_chunk_size=256,
                exp_approx_mode=False,
            )
            out = ttnn.transformer.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, is_causal=False, scale=scale,
                compute_kernel_config=_vision_sdpa_compute_kernel_config(),
                program_config=sdpa_prog_cfg,
            )
        # FUSED head merge: nlp_concat_heads [Bn,nh,HW,hd] -> [Bn,1,HW,C] (replaces
        # the permute). Reshape to the [Bn,g,g,C] proj input (free RM reshape).
        out = ttnn.experimental.nlp_concat_heads(out)  # [Bn, 1, HW, C]
        out = ttnn.to_layout(out, ttnn.ROW_MAJOR_LAYOUT)
        if Bn > 1:
            # Window-fold the per-token output projection too (see QKV above).
            out = ttnn.reshape(out, [1, Bn * HW, C])       # free RM reshape
            out = ttnn.to_layout(out, ttnn.TILE_LAYOUT)
            p = self.proj.forward(out)                     # [1, Bn*HW, C]
            p = ttnn.to_layout(p, ttnn.ROW_MAJOR_LAYOUT)
            p = ttnn.reshape(p, [Bn, g, g, C])
            return ttnn.to_layout(p, ttnn.TILE_LAYOUT)
        out = ttnn.reshape(out, [Bn, g, g, C])
        return self.proj.forward(out)


# =============================================================================
# Tier 3 — block / decoder layers
# =============================================================================
class TTNNUnlimitedOcrClipEmbeddings(StatelessTTNNModule):
    """CLIPVisionEmbeddings: prepend class token + add absolute pos embed (pre-baked).

    forward receives the SAM patch features as a channels-last SEQUENCE [B, 256, 1024]
    (== torch patch_embeds.flatten(2).transpose(1,2)); prepends class_embedding to
    [B,257,1024]; adds position_embedding(arange) which needs no interpolation at the
    fixed 1024 candidate resolution (get_abs_pos returns as-is when src==tgt).
    """

    def __init__(self):
        super().__init__()
        self._class_embedding_torch = None
        self._pos_embed_torch = None
        self.tt_class = None
        self.tt_pos = None

    @classmethod
    def from_torch(cls, emb):
        new = cls()
        new._fallback_torch_layer = emb
        new._class_embedding_torch = emb.class_embedding.detach().float()  # [C]
        # position_ids = arange(num_positions); get_abs_pos returns the embedding as-is
        # at the candidate resolution -> full position_embedding weight [num_pos, C].
        new._pos_embed_torch = emb.position_embedding.weight.detach().float().unsqueeze(0)  # [1, num_pos, C]
        return new

    def preprocess_weights_impl(self):
        import ttnn

        C = self._class_embedding_torch.shape[0]
        self.tt_class = ttnn.from_torch(
            self._class_embedding_torch.view(1, 1, C), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
        )
        self.tt_pos = ttnn.from_torch(self._pos_embed_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        return self

    def move_weights_to_device_impl(self):
        import ttnn

        self.tt_class = ttnn.to_device(self.tt_class, self.device)
        self.tt_pos = ttnn.to_device(self.tt_pos, self.device)
        return self

    @run_on_devices(*_VISION_ARCHS)
    def forward(self, patch_embeds_seq, **kwargs):
        import ttnn

        x = patch_embeds_seq
        if x.dtype != ttnn.bfloat16:
            x = ttnn.typecast(x, ttnn.bfloat16)
        if x.layout != ttnn.TILE_LAYOUT:
            x = ttnn.to_layout(x, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        emb = ttnn.concat([self.tt_class, x], dim=1)  # [B, num_pos, C]
        return ttnn.add(emb, self.tt_pos)


class TTNNUnlimitedOcrSamBlock(StatelessTTNNModule):
    """SAM ViT-B Block: LayerNorm -> Attention(rel-pos, windowed) -> LayerNorm ->
    MLPBlock (GELU). Pre-norm residual; window partition/unpartition around attention.

    ref: deepencoder.py::Block ; architecture_map.md §Vision encoder
    """

    def __init__(self):
        super().__init__()
        self.norm1 = None
        self.attn = None
        self.norm2 = None
        self.lin1 = None
        self.lin2 = None
        self.n1_eps = 1e-6
        self.n2_eps = 1e-6
        self.window_size = 0

    @classmethod
    def from_torch(cls, blk):
        from tt_symbiote.modules.ttnn_normalization import TTNNLayerNorm

        new = cls()
        new._fallback_torch_layer = blk
        new.norm1 = TTNNLayerNorm.from_torch(blk.norm1)
        new.norm2 = TTNNLayerNorm.from_torch(blk.norm2)
        new.n1_eps = float(getattr(blk.norm1, "eps", 1e-6))
        new.n2_eps = float(getattr(blk.norm2, "eps", 1e-6))
        new.attn = TTNNUnlimitedOcrSamAttention.from_torch(blk.attn)
        # fp32-acc linears (bias applied identically to the shared TTNNLinear).
        new.lin1 = TTNNUnlimitedOcrLinearBf8.from_torch(blk.mlp.lin1)
        new.lin2 = TTNNUnlimitedOcrLinearBf8.from_torch(blk.mlp.lin2)
        new.window_size = int(getattr(blk, "window_size", 0))
        return new

    @run_on_devices(*_VISION_ARCHS)
    def forward(self, x, **kwargs):
        import ttnn

        if x.dtype != ttnn.bfloat16:
            x = ttnn.typecast(x, ttnn.bfloat16)
        if x.layout != ttnn.TILE_LAYOUT:
            x = ttnn.to_layout(x, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        shortcut = x
        h = _ln(self.norm1, x, self.n1_eps)
        H, W = h.shape[1], h.shape[2]
        pad_hw = None
        if self.window_size > 0:
            h, pad_hw = _window_partition(h, self.window_size)
        h = self.attn.forward(h)
        if self.window_size > 0:
            h = _window_unpartition(h, self.window_size, pad_hw[0], pad_hw[1], H, W)
        if h.dtype != ttnn.bfloat16:
            h = ttnn.typecast(h, ttnn.bfloat16)
        if h.layout != ttnn.TILE_LAYOUT:
            h = ttnn.to_layout(h, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        x = ttnn.add(shortcut, h)

        # MLPBlock: lin2(gelu(lin1(x))). Pre-norm residual.
        h2 = _ln(self.norm2, x, self.n2_eps)
        h2 = self.lin1.forward(h2)
        h2 = ttnn.gelu(h2)
        h2 = self.lin2.forward(h2)
        if h2.dtype != ttnn.bfloat16:
            h2 = ttnn.typecast(h2, ttnn.bfloat16)
        return ttnn.add(x, h2)


class TTNNUnlimitedOcrClipBlock(StatelessTTNNModule):
    """CLIP-L NoTPTransformerBlock: LN -> NoTPAttention (SDPA) -> +res -> LN ->
    NoTPFeedForward (quick_gelu) -> +res. Pre-norm residual.

    ref: deepencoder.py::NoTPTransformerBlock
    """

    def __init__(self):
        super().__init__()
        self.layer_norm1 = None
        self.self_attn = None
        self.layer_norm2 = None
        self.mlp = None
        self.ln1_eps = 1e-5
        self.ln2_eps = 1e-5

    @classmethod
    def from_torch(cls, torch_layer):
        from tt_symbiote.modules.ttnn_normalization import TTNNLayerNorm

        new = cls()
        new._fallback_torch_layer = torch_layer
        new.layer_norm1 = TTNNLayerNorm.from_torch(torch_layer.layer_norm1)
        new.layer_norm2 = TTNNLayerNorm.from_torch(torch_layer.layer_norm2)
        new.ln1_eps = float(getattr(torch_layer.layer_norm1, "eps", 1e-5))
        new.ln2_eps = float(getattr(torch_layer.layer_norm2, "eps", 1e-5))
        attn = getattr(torch_layer, "self_attn", getattr(torch_layer, "attn", None))
        new.self_attn = TTNNUnlimitedOcrClipAttention.from_torch(attn)
        new.mlp = TTNNUnlimitedOcrClipFFN.from_torch(torch_layer.mlp)
        return new

    @run_on_devices(*_VISION_ARCHS)
    def forward(self, hidden_states, **kwargs):
        import ttnn

        if hidden_states.dtype != ttnn.bfloat16:
            hidden_states = ttnn.typecast(hidden_states, ttnn.bfloat16)
        h = _ln(self.layer_norm1, hidden_states, self.ln1_eps)
        h = self.self_attn.forward(h)
        if h.dtype != ttnn.bfloat16:
            h = ttnn.typecast(h, ttnn.bfloat16)
        h = ttnn.add(hidden_states, h)

        h2 = _ln(self.layer_norm2, h, self.ln2_eps)
        h2 = self.mlp.forward(h2)
        if h2.dtype != ttnn.bfloat16:
            h2 = ttnn.typecast(h2, ttnn.bfloat16)
        return ttnn.add(h, h2)


@trace_enabled
class TTNNUnlimitedOcrDecoderLayer(StatefulTTNNModule):
    """DeepSeek-V2 decoder layer: input_layernorm(RMSNorm) -> self_attn
    (SlidingWindowLlamaAttention) -> post_attention_layernorm -> mlp
    (dense DeepseekMLP for layer 0; DeepseekV2MoE for layers 1..11). Pre-norm residual.

    STATEFUL solely because it OWNS the stateful ``self_attn`` (which writes the KV
    cache). This layer's own forward holds no persistent trace state (residual adds,
    RMSNorm, MLP only), so its ``reset_trace_state`` is a documented no-op — the
    framework's trace tree-reset walks the subtree and resets ``self_attn``
    independently. Mirrors src/tt_symbiote/models/dots_ocr/dots_ocr_decoder_layer.py.

    ref: $TT_METAL_HOME/models/tt_transformers/tt/decoder.py ;
         src/tt_symbiote/models/dots_ocr/dots_ocr_decoder_layer.py
    """

    def __init__(self):
        super().__init__()
        self.input_layernorm = None
        self.self_attn = None
        self.post_attention_layernorm = None
        self.mlp = None
        self.layer_idx = None

    def reset_trace_state(self) -> None:
        # Documented no-op: no persistent trace state lives on THIS layer. The KV
        # write lives entirely in self.self_attn, which the trace tree-reset resets
        # independently. (Reasoned decision, not the bare inherited hook — see the
        # identical rationale in dots_ocr_decoder_layer.TTNNDotsOCRDecoderLayer.)
        return None

    @classmethod
    def from_torch(cls, torch_layer, layer_idx=None):
        from tt_symbiote.modules.ttnn_normalization import TTNNRMSNorm

        new = cls()
        new._fallback_torch_layer = torch_layer
        new.layer_idx = layer_idx if layer_idx is not None else getattr(torch_layer, "layer_idx", None)
        new.input_layernorm = TTNNRMSNorm.from_torch(torch_layer.input_layernorm)
        new.post_attention_layernorm = TTNNRMSNorm.from_torch(torch_layer.post_attention_layernorm)
        new.self_attn = TTNNUnlimitedOcrLlamaMHA.from_torch(torch_layer.self_attn, layer_idx=new.layer_idx)
        # Dense SwiGLU MLP on layer 0, MoE on layers 1..11 (first_k_dense_replace=1).
        mlp = torch_layer.mlp
        if type(mlp).__name__.endswith("MoE") or hasattr(mlp, "experts"):
            new.mlp = TTNNUnlimitedOcrMoE.from_torch(mlp)
        else:
            new.mlp = TTNNUnlimitedOcrDeepseekMLP.from_torch(mlp)
        return new

    @run_on_devices(DeviceArch.P150, DeviceArch.P150x4, DeviceArch.T3K)
    def forward(self, hidden_states, position_embeddings=None, attention_mask=None,
                past_key_value=None, cache_position=None, **kwargs):
        # Pre-norm residual: h + attn(norm1(h)); then h + mlp(norm2(h)).
        # Children invoked via .forward (bypassing per-child module_run/arch guard).
        # ``past_key_value`` (a UnlimitedOcrKVCache) + ``cache_position`` are threaded
        # to self_attn for the full-causal cached decode path (None => prefill).
        import ttnn

        if hidden_states.dtype != ttnn.bfloat16:
            hidden_states = ttnn.typecast(hidden_states, ttnn.bfloat16)

        # The reused TTNNRMSNorm carries a T3K-only @run_on_devices guard on its
        # forward, so we invoke ttnn.rms_norm inline with the child's already-
        # prepared weight (built by set_device) instead of child.forward().
        def _rmsnorm(norm, x):
            if x.layout != ttnn.TILE_LAYOUT:
                x = ttnn.to_layout(x, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            return ttnn.rms_norm(x, weight=norm.tt_weight, epsilon=norm.torch_layer.variance_epsilon)

        residual = hidden_states
        h = _rmsnorm(self.input_layernorm, hidden_states)
        h = self.self_attn.forward(
            h, position_embeddings=position_embeddings,
            past_key_value=past_key_value, cache_position=cache_position,
        )
        if h.dtype != ttnn.bfloat16:
            h = ttnn.typecast(h, ttnn.bfloat16)
        h = ttnn.add(residual, h)

        residual = h
        h2 = _rmsnorm(self.post_attention_layernorm, h)
        h2 = self.mlp.forward(h2)
        if h2.dtype != ttnn.bfloat16:
            h2 = ttnn.typecast(h2, ttnn.bfloat16)
        return ttnn.add(residual, h2)


# =============================================================================
# Tier 4 — encoders / full model / for-causal-lm
# =============================================================================
class TTNNUnlimitedOcrSamEncoder(StatelessTTNNModule):
    """SAM ViT-B ImageEncoderViT: patch_embed Conv2d(3,768,k16,s16) + pos_embed ->
    12 SamBlock -> neck (Conv2d k1 + LayerNorm2d + Conv2d k3p1 + LayerNorm2d) ->
    net_2 Conv2d(256,512,k3,s2,p1) -> net_3 Conv2d(512,1024,k3,s2,p1) => [B,1024,16,16].

    ref: $TT_METAL_HOME/models/demos (SAM/ViT) ; src/tt_symbiote/modules/ttnn_conv.py
    """

    def __init__(self):
        super().__init__()
        self.patch_embed = None
        self.blocks = []
        self.neck_conv1 = None
        self.neck_ln1 = None
        self.neck_conv2 = None
        self.neck_ln2 = None
        self.net_2 = None
        self.net_3 = None
        self._pos_embed_torch = None
        self.tt_pos = None

    @classmethod
    def from_torch(cls, torch_layer):
        from tt_symbiote.modules.ttnn_conv import TTNNConv2dNHWC

        new = cls()
        new._fallback_torch_layer = torch_layer
        new.patch_embed = TTNNConv2dNHWC.from_torch(torch_layer.patch_embed.proj)
        new.blocks = [TTNNUnlimitedOcrSamBlock.from_torch(b) for b in torch_layer.blocks]
        neck = torch_layer.neck  # nn.Sequential(Conv2d, LayerNorm2d, Conv2d, LayerNorm2d)
        new.neck_conv1 = TTNNConv2dNHWC.from_torch(neck[0])
        new.neck_ln1 = TTNNUnlimitedOcrLayerNorm2d.from_torch(neck[1])
        new.neck_conv2 = TTNNConv2dNHWC.from_torch(neck[2])
        new.neck_ln2 = TTNNUnlimitedOcrLayerNorm2d.from_torch(neck[3])
        new.net_2 = TTNNConv2dNHWC.from_torch(torch_layer.net_2)
        new.net_3 = TTNNConv2dNHWC.from_torch(torch_layer.net_3)
        if getattr(torch_layer, "pos_embed", None) is not None:
            new._pos_embed_torch = torch_layer.pos_embed.detach().float()  # [1,64,64,768] NHWC
        return new

    def _configure_convs_on_device(self):
        """Force every SAM conv onto the ON-DEVICE 2D path: ``reshape_output=False``
        (skips the T3K-only ``TTNNReshape`` that torch-falls-back on P150) with the
        conv's spatial H,W pre-recorded in ``model_config[<module>]["input_shapes"]``.

        Pre-recording H,W is REQUIRED for the convs fed the FLATTENED ``[1,1,H*W,C]``
        output of the previous conv (neck_conv2 / net_2 / net_3): without it the conv
        would (wrongly) read H,W from the flattened shape (H=1, W=H*W). patch_embed /
        neck_conv1 receive genuine 4D input so H,W is derivable, but we set them too
        for uniformity. Keyed by each conv's own ``module_name`` (each conv owns its
        own ``_model_config`` dict). Idempotent -- safe under the trace double-run.

        Spatial dims are fixed by the SAM ViT-B architecture at 1024x1024 input:
        patch_embed k16s16 -> 64x64; neck (k1 / k3p1) keeps 64x64; net_2 s2 -> 32x32;
        net_3 s2 -> 16x16 (output 16*16=256 tokens x 1024 channels).
        """
        specs = [
            (self.patch_embed, [1, 1024, 1024, 3]),  # -> [1,1,4096,768]
            (self.neck_conv1, [1, 64, 64, 768]),     # -> [1,1,4096,256]
            (self.neck_conv2, [1, 64, 64, 256]),     # -> [1,1,4096,256]
            (self.net_2, [1, 64, 64, 256]),          # -> [1,1,1024,512]
            (self.net_3, [1, 32, 32, 512]),          # -> [1,1, 256,1024]
        ]
        for conv, shape in specs:
            conv.set_model_config({conv.module_name: {"input_shapes": [shape], "reshape_output": False}})

    def preprocess_weights_impl(self):
        import ttnn

        if self._pos_embed_torch is not None:
            self.tt_pos = ttnn.from_torch(self._pos_embed_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        self._configure_convs_on_device()
        return self

    def move_weights_to_device_impl(self):
        import ttnn

        if self.tt_pos is not None:
            self.tt_pos = ttnn.to_device(self.tt_pos, self.device)
        return self

    @run_on_devices(DeviceArch.P150, DeviceArch.P150x4, DeviceArch.T3K)
    def forward(self, pixel_values, **kwargs):
        # pixel_values: NHWC ttnn tensor [1, 1024, 1024, 3]. Entire tower stays NHWC and
        # FULLY ON-DEVICE: every conv uses reshape_output=False (returns its native
        # flattened [1,1,H*W,C] WITHOUT the T3K-only TTNNReshape host round-trip), and
        # LayerNorm2d(channel-first) == ttnn.layer_norm over the last (channel) dim so it
        # operates on the flattened form directly. The only shape change is a PURE
        # ttnn.reshape (P150-native) from the flattened patch_embed output back to 4D for
        # the windowed SAM blocks. The final flattened [1,1,256,1024] == torch
        # sam.flatten(2).T (the caller reshapes to [1,256,1024]).
        import ttnn

        # patch_embed conv -> flattened [1,1,4096,768]; ttnn.reshape to 4D for the blocks.
        x = self.patch_embed.forward(pixel_values, reshape_output=False)  # [1,1,4096,768]
        x = ttnn.reshape(x, [1, 64, 64, 768])                            # pure on-device reshape
        if x.dtype != ttnn.bfloat16:
            x = ttnn.typecast(x, ttnn.bfloat16)
        if self.tt_pos is not None:
            if x.layout != ttnn.TILE_LAYOUT:
                x = ttnn.to_layout(x, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            x = ttnn.add(x, self.tt_pos)
        for blk in self.blocks:
            x = blk.forward(x)  # 4D [1,64,64,768]
        # neck + net_2 + net_3 (16x conv compressor) staying flattened [1,1,H*W,C] on device.
        x = self.neck_conv1.forward(x, reshape_output=False)  # [1,1,4096,256]
        x = self.neck_ln1.forward(x)                          # LN2d on flattened (channel-last)
        x = self.neck_conv2.forward(x, reshape_output=False)  # [1,1,4096,256]
        x = self.neck_ln2.forward(x)
        x = self.net_2.forward(x, reshape_output=False)       # [1,1,1024,512]
        x = self.net_3.forward(x, reshape_output=False)       # [1,1, 256,1024]
        return x


class TTNNUnlimitedOcrClipEncoder(StatelessTTNNModule):
    """CLIP-L vision transformer: embeddings (cls + pos_embed) + pre_layrnorm ->
    24 ClipBlock -> [B,257,1024]. Receives SAM features as patch_embeds.

    ref: $TT_METAL_HOME/tech_reports/ViT-TTNN/vit.md ; src/tt_symbiote/models/dots_ocr/dots_ocr_vision.py
    """

    def __init__(self):
        super().__init__()
        self.embeddings = None
        self.pre_layrnorm = None
        self.pre_eps = 1e-5
        self.blocks = []

    @classmethod
    def from_torch(cls, torch_layer):
        from tt_symbiote.modules.ttnn_normalization import TTNNLayerNorm

        new = cls()
        new._fallback_torch_layer = torch_layer
        new.embeddings = TTNNUnlimitedOcrClipEmbeddings.from_torch(torch_layer.embeddings)
        new.pre_layrnorm = TTNNLayerNorm.from_torch(torch_layer.pre_layrnorm)
        new.pre_eps = float(getattr(torch_layer.pre_layrnorm, "eps", 1e-5))
        # VitModel nests blocks under .transformer.layers; tolerate either arrangement.
        transformer = getattr(torch_layer, "transformer", torch_layer)
        layers = getattr(transformer, "layers", getattr(torch_layer, "layers", []))
        new.blocks = [TTNNUnlimitedOcrClipBlock.from_torch(b) for b in layers]
        return new

    @run_on_devices(DeviceArch.P150, DeviceArch.P150x4, DeviceArch.T3K)
    def forward(self, patch_embeds_seq, **kwargs):
        # patch_embeds_seq: SAM features as a channels-last sequence [B, 256, 1024].
        x = self.embeddings.forward(patch_embeds_seq)  # [B, 257, 1024]
        x = _ln(self.pre_layrnorm, x, self.pre_eps)
        for blk in self.blocks:
            x = blk.forward(x)
        return x  # [B, 257, 1024]


class TTNNUnlimitedOcrDeepEncoder(StatelessTTNNModule):
    """DeepEncoder = SAM ViT-B (sam_model) + CLIP-L (vision_model). Produces
    cat(clip[:,1:], sam.flatten(2).permute) => [B,256,2048] for the projector.

    ref: architecture_map.md §Vision encoder ; src/tt_symbiote/models/dots_ocr/dots_ocr_vision.py
    """

    def __init__(self):
        super().__init__()
        self.sam_model = None
        self.vision_model = None

    @classmethod
    def from_torch(cls, sam_model, vision_model):
        new = cls()
        new.sam_model = TTNNUnlimitedOcrSamEncoder.from_torch(sam_model)
        new.vision_model = TTNNUnlimitedOcrClipEncoder.from_torch(vision_model)
        return new

    @run_on_devices(DeviceArch.P150, DeviceArch.P150x4, DeviceArch.T3K)
    def forward(self, pixel_values, **kwargs):
        # pixel_values: NHWC ttnn tensor [1, 1024, 1024, 3].
        # sam_model -> NHWC [1,16,16,1024]; flatten to [1,256,1024] (== torch
        # sam.flatten(2).permute(0,2,1)); feed as CLIP patch_embeds; then merge
        # cat(clip[:,1:], sam.flatten) -> [1, 256, 2048] for the projector.
        import ttnn

        sam_out = self.sam_model.forward(pixel_values)  # NHWC [1,16,16,1024]
        B, C = sam_out.shape[0], sam_out.shape[3]
        HW = sam_out.shape[1] * sam_out.shape[2]
        sam_seq = ttnn.reshape(sam_out, [B, HW, C])  # [1, 256, 1024]

        clip_out = self.vision_model.forward(sam_seq)  # [1, 257, 1024]
        clip_out = ttnn.to_layout(clip_out, ttnn.ROW_MAJOR_LAYOUT)
        # drop the CLS token: clip[:, 1:, :]
        clip_patch = ttnn.slice(clip_out, [0, 1, 0], [B, clip_out.shape[1], clip_out.shape[2]])  # [1,256,1024]

        sam_seq_rm = ttnn.to_layout(sam_seq, ttnn.ROW_MAJOR_LAYOUT)
        if sam_seq_rm.dtype != clip_patch.dtype:
            sam_seq_rm = ttnn.typecast(sam_seq_rm, clip_patch.dtype)
        feats = ttnn.concat([clip_patch, sam_seq_rm], dim=-1)  # [1, 256, 2048]
        return feats


class TTNNUnlimitedOcrDeepseekModel(StatelessTTNNModule):
    """DeepSeek-V2 language model: embed_tokens nn.Embedding(129280,1280) ->
    12 DecoderLayer -> final DeepseekV2RMSNorm(1280, eps=1e-6).

    ref: $TT_METAL_HOME/models/tt_transformers/tt/model.py ;
         src/tt_symbiote/models/bailing_moe_v2/
    """

    def __init__(self):
        super().__init__()
        self.embed_tokens = None
        self.layers = []
        self.norm = None

    @classmethod
    def from_torch(cls, torch_layer):
        from tt_symbiote.modules.ttnn_embedding import TTNNEmbedding
        from tt_symbiote.modules.ttnn_normalization import TTNNRMSNorm

        new = cls()
        new._fallback_torch_layer = torch_layer
        new.embed_tokens = TTNNEmbedding.from_torch(torch_layer.embed_tokens)
        new.layers = [
            TTNNUnlimitedOcrDecoderLayer.from_torch(layer, layer_idx=i)
            for i, layer in enumerate(torch_layer.layers)
        ]
        new.norm = TTNNRMSNorm.from_torch(torch_layer.norm)
        return new

    @run_on_devices(DeviceArch.P150, DeviceArch.P150x4, DeviceArch.T3K)
    def forward(self, inputs_embeds=None, input_ids=None, attention_mask=None,
                past_key_values=None, position_embeddings=None, cache_position=None, **kwargs):
        # Text-only DeepSeek-V2 stack: embed -> N decoder layers -> final RMSNorm.
        # Children are invoked via .forward directly (bypassing per-child arch guard,
        # as in the decoder layer); embedding + final RMSNorm are inlined because
        # the reused TTNNEmbedding / TTNNRMSNorm carry a T3K-only forward guard.
        # position_embeddings=(cos,sin) are threaded to every decoder layer's attn;
        # past_key_values (a UnlimitedOcrKVCache) + cache_position drive full-causal
        # cached decode (None => prefill, prefill-equivalent behaviour preserved).
        import ttnn

        if inputs_embeds is not None:
            h = inputs_embeds
        else:
            tt_ids = input_ids
            if tt_ids.dtype != ttnn.uint32:
                tt_ids = ttnn.typecast(tt_ids, ttnn.uint32)
            if tt_ids.layout != ttnn.ROW_MAJOR_LAYOUT:
                tt_ids = ttnn.to_layout(tt_ids, ttnn.ROW_MAJOR_LAYOUT)
            h = ttnn.embedding(
                tt_ids,
                self.embed_tokens.tt_weight,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        if h.dtype != ttnn.bfloat16:
            h = ttnn.typecast(h, ttnn.bfloat16)

        for layer in self.layers:
            h = layer.forward(
                h, position_embeddings=position_embeddings,
                past_key_value=past_key_values, cache_position=cache_position,
            )

        if h.layout != ttnn.TILE_LAYOUT:
            h = ttnn.to_layout(h, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return ttnn.rms_norm(
            h, weight=self.norm.tt_weight, epsilon=self.norm.torch_layer.variance_epsilon
        )


class TTNNUnlimitedOcrForCausalLM(StatelessTTNNModule):
    """Top-level Unlimited-OCR: vision (DeepEncoder) -> projector -> scatter vision
    embeds into text inputs_embeds at <image> tokens (id 128815) -> DeepseekV2Model
    -> lm_head Linear(1280, 129280).

    Uses the dots_ocr scatter-merge idiom (ttnn.embedding + ttnn.where) in place of
    HF masked_scatter for image-token injection.

    ref: src/tt_symbiote/models/dots_ocr/pipeline.py (scatter-merge) ;
         $TT_METAL_HOME/models/tt_transformers/tt/lm_head.py
    """

    def __init__(self):
        super().__init__()
        self.deep_encoder = None
        self.projector = None
        self.model = None
        self.lm_head = None
        self.image_token_id = 128815
        # Learned vision-layout parameters (UnlimitedOCRModel.image_newline /
        # .view_seperator); host tensors stashed at from_torch, moved at set_device.
        self._image_newline_torch = None
        self._view_seperator_torch = None
        self.tt_image_newline = None   # [1, 1, H] bf16 (row token appended per grid row)
        self.tt_view_seperator = None  # [1, H]    bf16 (trailing separator row)
        self.tt_scatter_zero = None    # [1, H]    bf16 (row 0 of the scatter table)

    @classmethod
    def from_torch(cls, torch_model):
        new = cls()
        new._fallback_torch_layer = torch_model
        # UnlimitedOCRModel (inner) owns sam_model / vision_model / projector plus the
        # DeepSeek language model; lm_head sits on the outer ForCausalLM.
        inner = torch_model.get_model() if hasattr(torch_model, "get_model") else torch_model.model
        new.deep_encoder = TTNNUnlimitedOcrDeepEncoder.from_torch(inner.sam_model, inner.vision_model)
        new.projector = TTNNUnlimitedOcrMlpProjector.from_torch(inner.projector)
        # The DeepSeek text stack: inner itself (embed_tokens/layers/norm) or a nested .model.
        language_model = inner if hasattr(inner, "layers") else getattr(inner, "model", inner)
        new.model = TTNNUnlimitedOcrDeepseekModel.from_torch(language_model)
        new.lm_head = TTNNUnlimitedOcrLinear.from_torch(torch_model.lm_head)
        new.image_token_id = getattr(getattr(torch_model, "config", None), "image_token_id", 128815)
        # Learned scatter-layout parameters (image_newline / view_seperator live on
        # UnlimitedOCRModel; absent in the reduced synthetic text model -> optional).
        nl = getattr(inner, "image_newline", None)
        vs = getattr(inner, "view_seperator", None)
        new._image_newline_torch = nl.detach().float() if nl is not None else None
        new._view_seperator_torch = vs.detach().float() if vs is not None else None
        return new

    def preprocess_weights_impl(self):
        import torch
        import ttnn

        H = None
        if self._image_newline_torch is not None:
            H = int(self._image_newline_torch.numel())
            self.tt_image_newline = ttnn.from_torch(
                self._image_newline_torch.view(1, 1, H), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT
            )
        if self._view_seperator_torch is not None:
            H = int(self._view_seperator_torch.numel())
            self.tt_view_seperator = ttnn.from_torch(
                self._view_seperator_torch.view(1, H), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT
            )
        if H is not None:
            # Row 0 of the scatter table = "no vision" (all-zero embedding row).
            self.tt_scatter_zero = ttnn.from_torch(
                torch.zeros(1, H, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
            )
        return self

    def move_weights_to_device_impl(self):
        import ttnn

        if self.tt_image_newline is not None:
            self.tt_image_newline = ttnn.to_device(self.tt_image_newline, self.device)
        if self.tt_view_seperator is not None:
            self.tt_view_seperator = ttnn.to_device(self.tt_view_seperator, self.device)
        if self.tt_scatter_zero is not None:
            self.tt_scatter_zero = ttnn.to_device(self.tt_scatter_zero, self.device)
        return self

    def _build_vision_block(self, vision):
        # vision: projector output [1, g*g, H] -> global-only vision-token block
        # [g*(g+1) + 1, H]: view [g,g,H]; append image_newline per row -> [g,g+1,H];
        # flatten -> [g*(g+1), H]; append view_seperator -> [g*(g+1)+1, H].
        # Mirrors UnlimitedOCRModel.forward global-only branch (modeling_unlimitedocr.py:564-572).
        import ttnn

        H = vision.shape[-1]
        HW = vision.shape[-2]
        g = int(round(HW ** 0.5))
        vision = ttnn.to_layout(vision, ttnn.ROW_MAJOR_LAYOUT)
        if vision.dtype != ttnn.bfloat16:
            vision = ttnn.typecast(vision, ttnn.bfloat16)
        vision = ttnn.reshape(vision, [g, g, H])
        nl_rows = ttnn.repeat(self.tt_image_newline, ttnn.Shape([g, 1, 1]))  # [g,1,H]
        vision = ttnn.concat([vision, nl_rows], dim=1)  # [g, g+1, H]
        vision = ttnn.reshape(vision, [g * (g + 1), H])
        block = ttnn.concat([vision, self.tt_view_seperator], dim=0)  # [g*(g+1)+1, H]
        # Tile layout for the downstream scatter-table concat + ttnn.embedding.
        return ttnn.to_layout(block, ttnn.TILE_LAYOUT)

    def _scatter_merge(self, text_embeds, vision_block, vision_idx, vision_mask):
        # dots_ocr scatter-merge idiom (pipeline._scatter_fuse_text_and_vision):
        #   table = concat(zero_row, vision_block)         [Nv+1, H]  (row 0 = no vision)
        #   full  = ttnn.embedding(vision_idx, table)      [1, S, H]  (0 -> zero row)
        #   fused = ttnn.where(vision_mask, full, text)    inject at <image> positions
        import ttnn

        table = ttnn.concat([self.tt_scatter_zero, vision_block], dim=0)  # [Nv+1, H]
        full_vision = ttnn.embedding(
            vision_idx, table, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        if full_vision.dtype != ttnn.bfloat16:
            full_vision = ttnn.typecast(full_vision, ttnn.bfloat16)
        fused = ttnn.where(vision_mask, full_vision, text_embeds, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(full_vision)
        return fused

    @run_on_devices(DeviceArch.P150, DeviceArch.P150x4, DeviceArch.T3K)
    def forward(self, input_ids=None, pixel_values=None, position_embeddings=None,
                vision_idx=None, vision_mask=None, attention_mask=None,
                past_key_values=None, cache_position=None, **kwargs):
        # TEXT-ONLY (pixel_values=None): embed input_ids -> DeepseekModel -> lm_head.
        #   With past_key_values set this is BOTH the text prefill and the O(n) decode
        #   step (single new token id) of the KV-cache pipeline.
        # VLM (pixel_values set): embed text; DeepEncoder+projector -> global-only vision
        # block; scatter-merge into text embeds at <image> positions -> model -> lm_head.
        #   Runs ONCE at prefill (fills the cache); decode steps take the text-only path.
        import ttnn

        if pixel_values is None:
            hidden = self.model.forward(
                input_ids=input_ids, position_embeddings=position_embeddings,
                past_key_values=past_key_values, cache_position=cache_position,
            )
        else:
            tt_ids = input_ids
            if tt_ids.dtype != ttnn.uint32:
                tt_ids = ttnn.typecast(tt_ids, ttnn.uint32)
            if tt_ids.layout != ttnn.ROW_MAJOR_LAYOUT:
                tt_ids = ttnn.to_layout(tt_ids, ttnn.ROW_MAJOR_LAYOUT)
            text_e = ttnn.embedding(
                tt_ids, self.model.embed_tokens.tt_weight,
                layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            if text_e.dtype != ttnn.bfloat16:
                text_e = ttnn.typecast(text_e, ttnn.bfloat16)
            feats = self.deep_encoder.forward(pixel_values)      # [1,256,2048]
            vision = self.projector.forward(feats)               # [1,256,1280]
            vision_block = self._build_vision_block(vision)      # [273,1280]
            text_e = self._scatter_merge(text_e, vision_block, vision_idx, vision_mask)
            hidden = self.model.forward(
                inputs_embeds=text_e, position_embeddings=position_embeddings,
                past_key_values=past_key_values, cache_position=cache_position,
            )

        logits = self.lm_head.forward(hidden)
        return logits


__all__ = [
    "TT_METAL_COMMIT",
    "UnlimitedOcrKVCache",
    "TTNNUnlimitedOcrLinear",
    "TTNNUnlimitedOcrLinearBf8",
    "TTNNUnlimitedOcrLlamaMHA",
    "TTNNUnlimitedOcrDeepseekMLP",
    "TTNNUnlimitedOcrMoE",
    "TTNNUnlimitedOcrMlpProjector",
    "TTNNUnlimitedOcrQuickGELU",
    "TTNNUnlimitedOcrLayerNorm2d",
    "TTNNUnlimitedOcrClipFFN",
    "TTNNUnlimitedOcrClipEmbeddings",
    "TTNNUnlimitedOcrSamAttention",
    "TTNNUnlimitedOcrClipAttention",
    "TTNNUnlimitedOcrSamBlock",
    "TTNNUnlimitedOcrClipBlock",
    "TTNNUnlimitedOcrDecoderLayer",
    "TTNNUnlimitedOcrSamEncoder",
    "TTNNUnlimitedOcrClipEncoder",
    "TTNNUnlimitedOcrDeepEncoder",
    "TTNNUnlimitedOcrDeepseekModel",
    "TTNNUnlimitedOcrForCausalLM",
]
