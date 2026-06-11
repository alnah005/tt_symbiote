# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""TTNN bring-up for ``DiffusionGemmaForBlockDiffusion`` (model_type ``diffusion_gemma``).

Model: ``google/diffusiongemma-26B-A4B-it`` (block-diffusion, non-autoregressive
26B sparse-MoE: 128 experts, ~A4B active; sliding+full attention; partial RoPE;
per-head Q/K/V RMSNorm; gelu-tanh gated MLP; final logit softcapping).

Device target: P150x4 (4-chip Blackhole mesh -- detected hardware is 4x p300c).
Integration path: MANUAL (block-diffusion model; the HF class is native to
``transformers`` 5.11.0 but is NOT a standard ``*ForCausalLM``, so the
Recipe/Auto-CausalLM path does not apply directly).

Bring-up scope (this file -- "scaffold + Tier1/Tier2"):
-------------------------------------------------------
IMPLEMENTED (pure-TTNN forward, Tier1/Tier2 validated):
  * ``TTNNDiffusionGemmaRMSNorm``     <- ``DiffusionGemmaRMSNorm`` (with_scale True & False)
  * ``TTNNDiffusionGemmaTextMLP``     <- ``DiffusionGemmaText4MLP`` (gelu-tanh gated MLP)
  * ``TTNNDiffusionGemmaTextRouter``  <- ``DiffusionGemmaTextRouter`` (norm+scale+proj+softmax+topk)

SCAFFOLDED (forward raises NotImplementedError pending HW-validated impl; the
bespoke pieces deferred to the post-Tier1/2 review per the bring-up scope):
  * ``TTNNDiffusionGemmaTextExperts``           <- ``DiffusionGemmaTextExperts`` (sparse 128-expert MoE)
  * ``TTNNDiffusionGemmaEncoderTextAttention``  <- encoder attention (causal/bidir, sliding/full, partial RoPE)
  * ``TTNNDiffusionGemmaDecoderTextAttention``  <- decoder attention (bidirectional, read-only encoder KV)

Reference implementations consulted:
  * $TT_METAL_HOME/models/tt_transformers/tt/{attention,mlp,decoder}.py
  * $TT_METAL_HOME/models/demos/gemma4/tt/moe.py, models/demos/deepseek_v3/tt/moe.py
  * tt_symbiote.models.gemma4.modeling_gemma4_text (direct Gemma-4 analogue)
  * transformers.models.diffusion_gemma.modeling_diffusion_gemma (torch reference)
"""

from __future__ import annotations

import os

import ttnn

from tt_symbiote.core.module import DeviceArch, StatelessTTNNModule, run_on_devices
from tt_symbiote.core.run_config import trace_enabled
from tt_symbiote.modules.ttnn_linear import TTNNLinear

# tt-metal checkout this bring-up was developed against
# (git -C $TT_METAL_HOME rev-parse HEAD).
TT_METAL_COMMIT = "a9e84ad6ce70d53729ba2f558c113136e1c5cb20"

# Default device arch for this bring-up. Detected hardware is 4x p300c (Blackhole);
# the user selected a 4-chip mesh. Widen/adjust after validation.
_ARCH = DeviceArch.P150x4

# Tensor-parallel layout on the 1x4 Blackhole line mesh.
_TP = 4  # number of shards == mesh dim-1 size
_TP_AXIS = 1  # cluster_axis for CCL (sharded mesh dim)
_NUM_LINKS = 1
_TOPO = ttnn.Topology.Linear  # NEVER Ring on this mesh (framework-forbidden)

# fp32 dest-accumulation compute-kernel config (PCC lever; threaded into linears,
# sparse_matmuls and SDPA). WormholeComputeKernelConfig is the only *ComputeKernelConfig
# struct in ttnn and is the one tt_transformers/model_config.py uses on Blackhole.
_FP32_ACC = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    math_approx_mode=False,
    fp32_dest_acc_en=True,
    packer_l1_acc=True,
)


# --- per-shard tilization cache (gemma4 dtype_to_str / tp_suffix convention) ---
def _dtype_to_str(dtype):
    if dtype == ttnn.bfloat16:
        return "bf16"
    if dtype == ttnn.bfloat8_b:
        return "bfp8"
    if dtype == ttnn.float32:
        return "fp32"
    return str(dtype)


def _cache_dir(weights_kind="real"):
    """`~/.cache/tt_symbiote/diffusion_gemma/tensor_cache/{real,random}/` per the plan.

    Selected by the DIFFUSION_GEMMA_WEIGHTS env var ("real"/"random"); random-weight
    runs MUST NOT alias real-weight tiles.
    """
    kind = os.environ.get("DIFFUSION_GEMMA_WEIGHTS", weights_kind)
    d = os.path.expanduser(f"~/.cache/tt_symbiote/diffusion_gemma/tensor_cache/{kind}")
    os.makedirs(d, exist_ok=True)
    return d


def _cache_enabled():
    # Random-weight probes regenerate weights each run; a fixed role would serve a
    # stale tile. Set DIFFUSION_GEMMA_NO_CACHE=1 to disable the disk cache entirely.
    return os.environ.get("DIFFUSION_GEMMA_NO_CACHE", "0") != "1"


def _cache_path(role, suffix):
    if not role or not _cache_enabled():
        return None
    return os.path.join(_cache_dir(), f"{role}_{suffix}")


def _content_sig(host_tensor):
    """Short content hash of a host weight tensor for the disk-cache key.

    The role+shape+dtype suffix alone does NOT detect a WEIGHT-VALUE change (e.g.
    a random-weight build, a re-sliced proxy, or a different layer mapped to the
    same role+shape). Without this, ``ttnn.as_tensor(cache_file_name=...)`` would
    silently serve a STALE tile whose bytes no longer match the requested weight
    -> catastrophically wrong KV/logits while every shape assert still passes
    (the diffusion_gemma incoherence root cause). Hashing the raw bytes makes the
    cache key content-addressed: a value change yields a new key, never a reuse.
    """
    import hashlib

    import torch

    t = host_tensor.detach().to("cpu", torch.float32).contiguous()
    return hashlib.blake2b(t.numpy().tobytes(), digest_size=8).hexdigest()


def _as_tensor_cached(host, device, dtype, mapper, role):
    """Per-shard tilize + DRAM upload with disk cache. ``role`` is a stable cache
    key fragment that MUST encode shard-dims + dtype + any unfuse/EP signature so a
    dtype/sharding flip never reuses a stale tile. Empty role -> no cache (transients)."""
    _shape_sig = "x".join(str(int(d)) for d in host.shape)
    cache_file = _cache_path(role, f"{_shape_sig}_{_dtype_to_str(dtype)}")
    return ttnn.as_tensor(
        host,
        device=device,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=mapper,
        cache_file_name=cache_file,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


# ---------------------------------------------------------------------------
# P150x4 collective Linear subclasses (model-local; smallest blast radius).
# The shared T3K variants hardcode Topology.Ring -> not reusable here. These use
# Topology.Linear / cluster_axis=1 / fp32-acc, per-shard as_tensor tilization cache.
# Convention (tp_axis=1): column-parallel shards the weight OUTPUT (last) dim;
# row-parallel shards the weight INPUT (2nd-to-last) dim + all_reduce back to H.
# ---------------------------------------------------------------------------
class _ShardedLinearP150x4(TTNNLinear):
    """Base for the P150x4 col/row-parallel linears. ``_weight_shard_dim`` selects
    the sharded weight axis on the POST-``preprocess_linear_weight`` (TILE) tensor:
    -1 = output (column-parallel), -2 = input (row-parallel)."""

    _weight_shard_dim = -1  # overridden by subclasses

    def set_cache_role(self, role):
        self._cache_role = role
        return self

    def preprocess_weights_impl(self):
        # Keep the raw torch weight; shard+tilize happens in move_weights_to_device.
        self._host_weight = self.weight
        self._host_bias = self.bias

    def move_weights_to_device_impl(self):
        pass

        role = getattr(self, "_cache_role", None)
        sdim = self._weight_shard_dim
        # Transpose [out,in] -> [in,out] on host (torch; matches preprocess_linear_weight)
        # so the TTNN weight is [in,out]. Shard axis on this layout:
        #   column-parallel (sdim=-1) shards the OUTPUT (last) dim;
        #   row-parallel    (sdim=-2) shards the INPUT (2nd-to-last) dim.
        w_t = self._host_weight.detach().t().contiguous()  # [in, out]
        # Encode the full weight shape AND a content signature so a config change
        # with the SAME role (e.g. a layer flipping sliding head_dim 256 -> full
        # head_dim 512) OR a WEIGHT-VALUE change (random/proxy/different layer at the
        # same shape) can never silently reuse a stale tile.
        cache_file = _cache_path(role, f"dim{sdim}_tp{_TP}_{w_t.shape[0]}x{w_t.shape[1]}_bf16_{_content_sig(w_t)}")
        self.tt_weight = ttnn.as_tensor(
            w_t,
            device=self.device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.shard_tensor_to_mesh_mapper(self.device, dim=sdim),
            cache_file_name=cache_file,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        # bias (only the lm_head/router may carry one; diffusion_gemma linears are bias-free).
        self.tt_bias = None
        if self._host_bias is not None:
            bmap = (
                ttnn.shard_tensor_to_mesh_mapper(self.device, dim=-1)
                if sdim == -1
                else ttnn.ReplicateTensorToMesh(self.device)
            )
            self.tt_bias = ttnn.from_torch(
                self._host_bias.detach(),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                mesh_mapper=bmap,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )


class _ColParallelLinearP150x4(_ShardedLinearP150x4):
    """Column-parallel: shard weight output dim; replicated input; NO CCL; output
    is sharded-on-output (caller closes the loop with a row-parallel + all_reduce)."""

    _weight_shard_dim = -1

    @run_on_devices(DeviceArch.P150x4)
    def forward(self, input_tensor: ttnn.Tensor) -> ttnn.Tensor:
        if input_tensor.layout != ttnn.TILE_LAYOUT:
            input_tensor = ttnn.to_layout(input_tensor, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ishape = list(input_tensor.shape)
        shp = list(ishape)
        while len(shp) < 4:
            shp.insert(1, 1)
        input_tensor = ttnn.reshape(input_tensor, shp)
        out = ttnn.linear(
            input_tensor,
            self.tt_weight,
            bias=self.tt_bias,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=_FP32_ACC,
        )
        return ttnn.reshape(out, ishape[:-1] + [-1])


class _KVRepeatColParallelLinearP150x4(_ColParallelLinearP150x4):
    """Column-parallel K/V projection for FULL-attention layers where
    ``num_kv_heads < TP`` (here kv=2, TP=4). The GQA q->kv mapping is by contiguous
    blocks; head-sharding q renumbers local q-heads 0..lqh-1 per chip, so a plain
    replicated kv would make every chip's SDPA map its local q-heads to local kv0
    (wrong for chips whose global q-heads map to a higher kv head).

    Fix: REPEAT each kv head ``TP/num_kv_heads`` times on host so the projection
    outputs TP kv heads ``[kv0,..,kv0, kv1,..,kv1]`` aligned to the q-head blocks,
    then column-shard /TP -> exactly 1 kv head/chip that matches that chip's q-heads.
    Mathematically identical (kv values duplicated, never summed)."""

    def set_kv_repeat(self, num_kv_heads, head_dim):
        self._kv_heads = num_kv_heads
        self._hd = head_dim
        self._repeat = _TP // num_kv_heads
        return self

    def move_weights_to_device_impl(self):
        pass

        role = getattr(self, "_cache_role", None)
        kv, hd, rep = self._kv_heads, self._hd, self._repeat
        # weight [kv*hd, H] -> [kv, hd, H] -> repeat_interleave on heads -> [TP*hd, H]
        w = self._host_weight.detach()
        H = w.shape[1]
        w = w.reshape(kv, hd, H).repeat_interleave(rep, dim=0).reshape(_TP * hd, H)
        w_t = w.t().contiguous()  # [H, TP*hd]
        cache_file = _cache_path(role, f"kvrep_tp{_TP}_{w_t.shape[0]}x{w_t.shape[1]}_bf16_{_content_sig(w_t)}")
        self.tt_weight = ttnn.as_tensor(
            w_t,
            device=self.device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.shard_tensor_to_mesh_mapper(self.device, dim=-1),
            cache_file_name=cache_file,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.tt_bias = None


class _RowParallelLinearAllReduceP150x4(_ShardedLinearP150x4):
    """Row-parallel: shard weight input dim; per-chip partial matmul; native
    all_reduce(sum) over cluster_axis=1 -> replicated full-H output."""

    _weight_shard_dim = -2

    @run_on_devices(DeviceArch.P150x4)
    def forward(self, input_tensor: ttnn.Tensor) -> ttnn.Tensor:
        if input_tensor.layout != ttnn.TILE_LAYOUT:
            input_tensor = ttnn.to_layout(input_tensor, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ishape = list(input_tensor.shape)
        shp = list(ishape)
        while len(shp) < 4:
            shp.insert(1, 1)
        input_tensor = ttnn.reshape(input_tensor, shp)
        partial = ttnn.linear(
            input_tensor, self.tt_weight, memory_config=ttnn.DRAM_MEMORY_CONFIG, compute_kernel_config=_FP32_ACC
        )
        out = ttnn.all_reduce(
            partial, cluster_axis=_TP_AXIS, num_links=_NUM_LINKS, topology=_TOPO, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        if self.tt_bias is not None:
            out = ttnn.add(out, self.tt_bias)
        return ttnn.reshape(out, ishape[:-1] + [-1])


__all__ = [
    "TTNNDiffusionGemmaRMSNorm",
    "TTNNDiffusionGemmaTextMLP",
    "TTNNDiffusionGemmaTextRouter",
    "TTNNDiffusionGemmaTextExperts",
    "TTNNDiffusionGemmaEncoderTextAttention",
    "TTNNDiffusionGemmaDecoderTextAttention",
    "TTNNDiffusionGemmaEncoderTextLayer",
    "TTNNDiffusionGemmaDecoderTextLayer",
    "TTNNDiffusionGemmaTextScaledWordEmbedding",
    "TTNNDiffusionGemmaEncoderTextModel",
    "TTNNDiffusionGemmaSelfConditioning",
    "TTNNDiffusionGemmaDecoderTextModel",
    "TTNNDiffusionGemmaLMHead",
    "share_text_weights",
]


# ---------------------------------------------------------------------------
# RMSNorm -- handles both ``with_scale=True`` (decoder/router/q/k norms with a
# learnable scale) and ``with_scale=False`` (router pre-norm, per-head v_norm).
# ---------------------------------------------------------------------------
#
# HF ``DiffusionGemmaRMSNorm`` computes, in fp32:
#     normed = x * pow(mean(x^2) + eps, -0.5)
#     out    = normed * weight            (only when with_scale)
# Unlike some Gemma norms there is NO ``(1 + weight)`` offset here -- the weight
# multiplies directly. ``ttnn.rms_norm(x, weight=W, epsilon=eps)`` matches the
# scaled form; the weightless form drops the ``weight`` kwarg.
class TTNNDiffusionGemmaRMSNorm(StatelessTTNNModule):
    """``DiffusionGemmaRMSNorm`` -> ``ttnn.rms_norm``.

    ``with_scale=True``  -> ``ttnn.rms_norm(x, weight=W, epsilon=eps)``
    ``with_scale=False`` -> ``ttnn.rms_norm(x, epsilon=eps)`` (no weight)
    """

    @classmethod
    def from_torch(cls, rms_norm):
        new = cls()
        new._fallback_torch_layer = rms_norm
        new._eps = float(getattr(rms_norm, "eps", getattr(rms_norm, "variance_epsilon", 1e-6)))
        new._with_scale = bool(getattr(rms_norm, "with_scale", True)) and (
            getattr(rms_norm, "weight", None) is not None
        )
        return new

    def preprocess_weights_impl(self):
        if not self._with_scale:
            self.tt_weight = None
            return
        weight = self.torch_layer.weight
        # Store as [1, dim] so rms_norm broadcasts across [B, ..., dim] inputs.
        self.tt_weight = ttnn.from_torch(
            weight.unsqueeze(0),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )

    def move_weights_to_device_impl(self):
        if self.tt_weight is None:
            return
        self.tt_weight = ttnn.to_device(self.tt_weight, self.device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    @run_on_devices(_ARCH)
    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        if x.layout != ttnn.TILE_LAYOUT:
            x = ttnn.to_layout(x, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if self.tt_weight is None:
            return ttnn.rms_norm(x, epsilon=self._eps)
        return ttnn.rms_norm(x, weight=self.tt_weight, epsilon=self._eps)


# ---------------------------------------------------------------------------
# Gated MLP (dense): down_proj(act(gate_proj(x)) * up_proj(x))
# ---------------------------------------------------------------------------
#
# HF ``DiffusionGemmaText4MLP`` uses ``ACT2FN[config.hidden_activation]`` which
# for this model is ``"gelu_pytorch_tanh"`` (tanh approximation). ``ttnn.gelu``
# uses the tanh approximation by default -- the small drift is within bf16 noise
# (same convention as the Gemma-4 port).
class TTNNDiffusionGemmaTextMLP(StatelessTTNNModule):
    """``DiffusionGemmaText4MLP`` -> on-device gelu-gated MLP."""

    @classmethod
    def from_torch(cls, mlp, role_prefix=""):
        new = cls()
        new._fallback_torch_layer = mlp
        # gate/up: column-parallel (shard output Ff); down: row-parallel + all_reduce.
        new.gate_proj = _ColParallelLinearP150x4.from_torch(mlp.gate_proj).set_cache_role(f"{role_prefix}mlp_gate")
        new.up_proj = _ColParallelLinearP150x4.from_torch(mlp.up_proj).set_cache_role(f"{role_prefix}mlp_up")
        new.down_proj = _RowParallelLinearAllReduceP150x4.from_torch(mlp.down_proj).set_cache_role(
            f"{role_prefix}mlp_down"
        )
        return new

    def preprocess_weights_impl(self):
        self.gate_proj.preprocess_weights()
        self.up_proj.preprocess_weights()
        self.down_proj.preprocess_weights()
        super().preprocess_weights_impl()

    def move_weights_to_device_impl(self):
        self.gate_proj.move_weights_to_device()
        self.up_proj.move_weights_to_device()
        self.down_proj.move_weights_to_device()
        super().move_weights_to_device_impl()

    @run_on_devices(_ARCH)
    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        gate = ttnn.gelu(self.gate_proj(x))
        up = self.up_proj(x)
        intermediate = ttnn.multiply(gate, up)
        ttnn.deallocate(gate)
        ttnn.deallocate(up)
        return self.down_proj(intermediate)


# ---------------------------------------------------------------------------
# MoE Router: RMSNorm(no scale) -> *scale*scalar_root -> proj -> softmax -> topk
# ---------------------------------------------------------------------------
#
# HF ``DiffusionGemmaTextRouter.forward`` (torch):
#     h = norm(hidden)                         # RMSNorm, with_scale=False
#     h = h * self.scale * (hidden_size ** -0.5)
#     logits = proj(h)                          # [N, num_experts], bias-free
#     probs  = softmax(logits, fp32)
#     w, idx = topk(probs, k)                   # [N, K]
#     w     /= w.sum(-1, keepdim=True)
#     w      = w * per_expert_scale[idx]
#     return probs, w, idx
#
# The router exposes the heavy compute (norm/scale/proj/softmax/topk) which maps
# cleanly to TTNN. ``per_expert_scale[idx]`` is a gather over a [num_experts]
# vector -> done with ttnn.embedding on the int32 topk indices.
class TTNNDiffusionGemmaTextRouter(StatelessTTNNModule):
    """``DiffusionGemmaTextRouter`` -> on-device router (returns probs, weights, indices)."""

    @classmethod
    def from_torch(cls, router):
        new = cls()
        new._fallback_torch_layer = router
        new.proj = TTNNLinear.from_torch(router.proj)
        new._eps = float(getattr(router, "eps", 1e-6))
        new._scalar_root_size = float(router.scalar_root_size)
        new._top_k = int(router.config.top_k_experts)
        new._num_experts = int(router.config.num_experts)
        return new

    def preprocess_weights_impl(self):
        self.proj.preprocess_weights()
        # Per-dim router input scale (shape [hidden]) -> [1, hidden] for broadcast.
        self.tt_scale = ttnn.from_torch(
            self.torch_layer.scale.detach().unsqueeze(0),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )
        # Per-expert output scale (shape [num_experts]) -> embedding table
        # [num_experts, 1] so a topk-index lookup yields the per-token scale.
        self.tt_per_expert_scale = ttnn.from_torch(
            self.torch_layer.per_expert_scale.detach().unsqueeze(-1),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        super().preprocess_weights_impl()

    def move_weights_to_device_impl(self):
        self.proj.move_weights_to_device()
        self.tt_scale = ttnn.to_device(self.tt_scale, self.device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        self.tt_per_expert_scale = ttnn.to_device(
            self.tt_per_expert_scale, self.device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        super().move_weights_to_device_impl()

    @run_on_devices(_ARCH)
    def forward(self, hidden_states: ttnn.Tensor):
        if hidden_states.layout != ttnn.TILE_LAYOUT:
            hidden_states = ttnn.to_layout(hidden_states, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        # RMSNorm without learnable scale, then * per-dim scale * hidden**-0.5.
        h = ttnn.rms_norm(hidden_states, epsilon=self._eps)
        h = ttnn.multiply(h, self.tt_scale)
        h = ttnn.multiply(h, self._scalar_root_size)

        logits = self.proj(h)
        probs = ttnn.softmax(logits, dim=-1)

        top_k_weights, top_k_index = ttnn.topk(probs, self._top_k, dim=-1)

        # Normalize the top-k weights so they sum to 1 per token.
        denom = ttnn.sum(top_k_weights, dim=-1, keepdim=True)
        top_k_weights = ttnn.divide(top_k_weights, denom)

        # per_expert_scale[idx] via embedding gather, then elementwise multiply.
        # ttnn.embedding requires a 2D [rows, cols] UINT32 index tensor (a >2D
        # index collapses incorrectly). Works for any input rank (router is
        # called on both 3D [B,S,H] and flattened 2D [N,H] hidden states): flatten
        # all index dims to [1, total*k], gather, reshape back to the index shape.
        idx_shape = [int(d) for d in top_k_index.shape]
        total_k = 1
        for d in idx_shape:
            total_k *= d
        idx_rm = ttnn.to_layout(top_k_index, ttnn.ROW_MAJOR_LAYOUT)
        if idx_rm.dtype != ttnn.uint32:
            idx_rm = ttnn.typecast(idx_rm, ttnn.uint32)
        idx_flat = ttnn.reshape(idx_rm, (1, total_k))
        gathered = ttnn.embedding(idx_flat, self.tt_per_expert_scale)  # [1, total*k, 1]
        gathered = ttnn.reshape(gathered, idx_shape)
        gathered = ttnn.to_layout(gathered, ttnn.TILE_LAYOUT)
        top_k_weights = ttnn.multiply(top_k_weights, gathered)

        return probs, top_k_weights, top_k_index


# ---------------------------------------------------------------------------
# MoE experts -- ttnn.sparse_matmul (prefill-style), the canonical multi-token
# MoE primitive. Reference: $TT_METAL_HOME/models/demos/gemma4/tt/experts/.
# ---------------------------------------------------------------------------
#
# HF ``DiffusionGemmaTextExperts`` stores fused ``gate_up_proj [E, 2I, H]`` and
# ``down_proj [E, H, I]`` and does a sparse host loop with ``index_add_``. We use
# the on-device ``ttnn.sparse_matmul`` MoE path (gpt_oss / gemma4 prefill
# pattern): the sequence is grouped into 32-token tiles, every expert is computed
# per group (all-ones sparsity -- different tokens in a tile pick different
# experts, so the per-tile union is ~all experts; this is the canonical
# multi-token approach), then the per-token routing weights select/scale the
# active experts after the down projection and reduce over the expert dim.
#
# The dense routing mask [1,1,N,E] (zero except the top-k, holding the
# normalized * per-expert weight) is built from the router's compact (index,
# weight) pairs via the proven identity-matrix ttnn.embedding one-hot path.
#
# ``nnz == count_nonzero(sparsity)`` is mandatory (a mismatch hangs the kernel);
# with all-ones sparsity this is exactly ``num_experts * group_size``.
_TILE = 32
_MAX_GRID_CORES = 64  # Blackhole has ~110 Tensix cores; stay under for safety.


def _gcd3(a, b, c):
    import math

    return math.gcd(math.gcd(a, b), c)


def _grid_coord(cores):
    """Factor ``cores`` into cx<=8, cy (grid area == cores: one core per block)."""
    for cx in range(min(8, cores), 0, -1):
        if cores % cx == 0:
            return ttnn.CoreCoord(cx, cores // cx)
    return ttnn.CoreCoord(1, cores)


def _build_sparse_matmul_config(n, grid_cores):
    """Program config for ttnn.sparse_matmul. ``grid_cores`` MUST divide n_tiles
    (so per_core_N tiles the output N cleanly) and equals the chunk's block count
    (one (expert,group) block per core). ``mcast_in0`` must be True."""
    import math

    n_tiles = max(1, int(math.ceil(n / _TILE)))
    per_core_N = max(1, n_tiles // grid_cores)
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=_grid_coord(grid_cores),
        in0_block_w=1,
        out_subblock_h=1,
        out_subblock_w=1,
        out_block_h=1,
        out_block_w=per_core_N,
        per_core_M=1,
        per_core_N=per_core_N,
        fuse_batch=False,
        fused_activation=None,
        mcast_in0=True,
    )


class TTNNDiffusionGemmaTextExperts(StatelessTTNNModule):
    """``DiffusionGemmaTextExperts`` -> on-device SwiGLU MoE via ttnn.sparse_matmul.

    Ported from gemma4's prefill experts (all-ones sparsity -> compute every
    expert, then mask by per-token routing weights). Because 128 experts exceed
    the Blackhole core grid, experts are processed in CHUNKS of ``G`` and the
    sequence one 32-token tile at a time, so each sparse_matmul has exactly ``G``
    (expert x group) blocks on a ``G``-core grid. ``G = gcd(gate_n_tiles,
    down_n_tiles, num_experts)`` (capped) divides the experts (no padding) and
    both n_tiles (valid per_core_N). Partial chunk outputs are routing-weighted
    and accumulated. (A larger mesh would allow bigger G / fewer chunks.)
    """

    @classmethod
    def from_torch(cls, experts, role_prefix=""):
        new = cls()
        new._fallback_torch_layer = experts
        new._num_experts = int(experts.num_experts)  # global E (=128); top-k/one-hot use this
        new._hidden_dim = int(experts.hidden_dim)
        new._intermediate_dim = int(experts.intermediate_dim)
        # EXPERT-PARALLEL: each chip owns E/TP local experts. _num_experts_local is
        # used ONLY for the weight-slice loop / chunk count / local sparsity width.
        new._num_experts_local = max(1, new._num_experts // _TP)
        new._role_prefix = role_prefix
        return new

    def preprocess_weights_impl(self):
        import torch

        gu = self.torch_layer.gate_up_proj.detach()  # [E, 2I, H]
        dn = self.torch_layer.down_proj.detach()  # [E, H, I]
        i = self._intermediate_dim
        e = self._num_experts
        # Unfuse + transpose to sparse_matmul layout (gemma4 weights.py):
        #   gate/up: [E, I, H] -> [1, E, H, I];  down: [E, H, I] -> [1, E, I, H]
        # Built as full-E host tensors here (G6 hazard); EXPERT-dim (dim 1) shard
        # mapper applied in move_weights_to_device_impl via as_tensor.
        self._h_gate = gu[:, :i, :].transpose(-2, -1).unsqueeze(0).contiguous()  # [1,E,H,I]
        self._h_up = gu[:, i:, :].transpose(-2, -1).unsqueeze(0).contiguous()  # [1,E,H,I]
        self._h_down = dn.transpose(-2, -1).unsqueeze(0).contiguous()  # [1,E,I,H]
        # tt_eye [E,E] identity; COLUMN-sharded on dim 1 -> chip i owns columns
        # [i*El:(i+1)*El]. embedding(global_idx, eye_sharded) yields, per chip, the
        # one-hot restricted to that chip's local experts -> the E-sharded routing
        # mask with NO per-chip index arithmetic (indices stay 0..E-1).
        self._h_eye = torch.eye(e)
        # All-ones [1,1,1,El] local sparsity (one entry per LOCAL expert); nnz = G.
        self._h_base_sparsity = torch.ones(1, 1, 1, self._num_experts_local)

    def move_weights_to_device_impl(self):
        emap = ttnn.shard_tensor_to_mesh_mapper(self.device, dim=1)  # shard EXPERT dim
        cmap = ttnn.shard_tensor_to_mesh_mapper(self.device, dim=1)  # shard eye COLUMNS
        rp = self._role_prefix

        def _ep_suffix(t):
            # content sig: a re-sliced/different-weight expert tensor at the SAME
            # shape must never reuse a stale tile (see _content_sig rationale).
            return f"ep{_TP}_{'x'.join(str(int(d)) for d in t.shape)}_bf16_{_content_sig(t)}"

        self.tt_gate = ttnn.as_tensor(
            self._h_gate,
            device=self.device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=emap,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=_cache_path(rp + "exp_gate", _ep_suffix(self._h_gate)),
        )
        self.tt_up = ttnn.as_tensor(
            self._h_up,
            device=self.device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=emap,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=_cache_path(rp + "exp_up", _ep_suffix(self._h_up)),
        )
        self.tt_down = ttnn.as_tensor(
            self._h_down,
            device=self.device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=emap,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=_cache_path(rp + "exp_down", _ep_suffix(self._h_down)),
        )
        # eye column-sharded (ROW_MAJOR for embedding); base sparsity replicated.
        self.tt_eye = ttnn.from_torch(
            self._h_eye,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
            mesh_mapper=cmap,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.tt_base_sparsity = ttnn.from_torch(
            self._h_base_sparsity,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        for nm in ("_h_gate", "_h_up", "_h_down", "_h_eye", "_h_base_sparsity"):
            if hasattr(self, nm):
                delattr(self, nm)

    def _routing_mask(self, top_k_index, top_k_weights, n, k):
        """E-sharded [1,1,N,El] routing weights via the COLUMN-sharded identity one-hot.

        ``tt_eye`` is column-sharded so chip i sees only its El local-expert columns;
        ``ttnn.embedding`` over global indices yields the per-chip restricted one-hot."""
        el = self._num_experts_local
        # Cast to uint32 BEFORE reshape: topk returns uint16, which ROW_MAJOR
        # reshape rejects (only bf16/fp32/int32/uint32 supported).
        idx_flat = ttnn.to_layout(top_k_index, ttnn.ROW_MAJOR_LAYOUT)
        if idx_flat.dtype != ttnn.uint32:
            idx_flat = ttnn.typecast(idx_flat, ttnn.uint32)
        idx_flat = ttnn.reshape(idx_flat, (1, n * k))
        onehot = ttnn.embedding(idx_flat, self.tt_eye)  # [1, N*K, El] per chip
        onehot = ttnn.to_layout(ttnn.reshape(onehot, (n, k, el)), ttnn.TILE_LAYOUT)
        wk = ttnn.reshape(top_k_weights, (n, k, 1))
        routing = ttnn.sum(ttnn.multiply(onehot, wk), dim=1)  # [N, El]
        return ttnn.reshape(routing, (1, 1, n, el))

    @run_on_devices(_ARCH)
    def forward(self, hidden_states, top_k_index, top_k_weights):
        # EXPERT-PARALLEL: per-chip e == _num_experts_local; tt_gate/up/down hold the
        # chip's El local experts; routing mask is E-sharded to the same El columns.
        e, i, h = self._num_experts_local, self._intermediate_dim, self._hidden_dim
        n = int(hidden_states.shape[0])
        k = int(top_k_index.shape[-1])
        gate_nt, down_nt = i // _TILE, h // _TILE
        g = max(1, _gcd3(gate_nt, down_nt, e))
        while g > _MAX_GRID_CORES:
            # halve to the next common divisor (g divides e and both n_tiles).
            for d in range(g - 1, 0, -1):
                if g % d == 0 and e % d == 0 and gate_nt % d == 0 and down_nt % d == 0:
                    g = d
                    break
        out_tile = ttnn.Tile([_TILE, _TILE])
        gate_cfg = _build_sparse_matmul_config(i, g)
        down_cfg = _build_sparse_matmul_config(h, g)

        routing = self._routing_mask(top_k_index, top_k_weights, n, k)  # [1,1,N,El] (E-sharded)
        # [1,1,1,G] all-ones chunk sparsity (slice of the precomputed base); nnz = G.
        chunk_sp = ttnn.slice(self.tt_base_sparsity, [0, 0, 0, 0], [1, 1, 1, g], [1, 1, 1, 1])

        hidden_4d = ttnn.reshape(hidden_states, (1, 1, n, h))
        # Token count need not be tile-aligned (e.g. an encoder prompt of 37 tokens):
        # pad the token dim up to a _TILE multiple so the per-tile loop covers ALL
        # tokens. The trailing pad rows route to zero (routing padded with zeros) ->
        # contribute nothing -> are sliced off the output. The decoder canvas is
        # already a _TILE multiple, so this is a no-op there.
        n_pad = ((n + _TILE - 1) // _TILE) * _TILE
        if n_pad != n:
            pad = [(0, 0), (0, 0), (0, n_pad - n), (0, 0)]
            hidden_4d = ttnn.pad(hidden_4d, pad, value=0.0)
            routing = ttnn.pad(routing, pad, value=0.0)
        tiles, chunks = n_pad // _TILE, e // g
        tile_outs = []
        for t in range(tiles):
            hid_t = ttnn.slice(hidden_4d, [0, 0, t * _TILE, 0], [1, 1, (t + 1) * _TILE, h], [1, 1, 1, 1])  # [1,1,32,H]
            acc = None
            for c in range(chunks):
                gw = ttnn.slice(self.tt_gate, [0, c * g, 0, 0], [1, (c + 1) * g, h, i], [1, 1, 1, 1])
                uw = ttnn.slice(self.tt_up, [0, c * g, 0, 0], [1, (c + 1) * g, h, i], [1, 1, 1, 1])
                dw = ttnn.slice(self.tt_down, [0, c * g, 0, 0], [1, (c + 1) * g, i, h], [1, 1, 1, 1])
                gate = ttnn.sparse_matmul(
                    hid_t,
                    gw,
                    sparsity=chunk_sp,
                    nnz=g,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    output_tile=out_tile,
                    program_config=gate_cfg,
                    dtype=ttnn.bfloat16,
                )
                sm_i = int(gate.shape[-1])
                gate = ttnn.reshape(ttnn.transpose(gate, 1, 3), (1, g, _TILE, sm_i))
                up = ttnn.sparse_matmul(
                    hid_t,
                    uw,
                    sparsity=chunk_sp,
                    nnz=g,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    output_tile=out_tile,
                    program_config=gate_cfg,
                    dtype=ttnn.bfloat16,
                )
                up = ttnn.reshape(ttnn.transpose(up, 1, 3), (1, g, _TILE, sm_i))
                down_in = ttnn.multiply(ttnn.gelu(gate, fast_and_approximate_mode=True), up)
                down = ttnn.sparse_matmul(
                    down_in,
                    dw,
                    sparsity=chunk_sp,
                    nnz=g,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    output_tile=out_tile,
                    program_config=down_cfg,
                    is_input_a_sparse=True,
                    dtype=ttnn.bfloat16,
                )
                down = ttnn.reshape(down, (1, g, _TILE, h))  # [1,G,32,H]
                r = ttnn.slice(
                    routing, [0, 0, t * _TILE, c * g], [1, 1, (t + 1) * _TILE, (c + 1) * g], [1, 1, 1, 1]
                )  # [1,1,32,G]
                r = ttnn.permute(r, (0, 3, 2, 1))  # [1,G,32,1]
                part = ttnn.sum(ttnn.multiply(down, r), dim=1)  # [1,32,H]
                acc = part if acc is None else ttnn.add(acc, part)
            tile_outs.append(acc)
        out = ttnn.concat(tile_outs, dim=1) if len(tile_outs) > 1 else tile_outs[0]  # [1,N_pad,H] partial
        out = ttnn.reshape(out, (1, 1, n_pad, h))
        # Each chip summed only its El local experts -> ONE final all_reduce(sum)
        # across the TP axis gives the exact full-E (128-expert) routing-weighted sum,
        # replicated full-H. (all-ones sparsity => no token dispatch needed.)
        out = ttnn.all_reduce(
            out, cluster_axis=_TP_AXIS, num_links=_NUM_LINKS, topology=_TOPO, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        out = ttnn.reshape(out, (n_pad, h))
        if n_pad != n:
            out = ttnn.slice(out, [0, 0], [n, h], [1, 1])
        return out


# ---------------------------------------------------------------------------
# Attention (encoder & decoder)
# ---------------------------------------------------------------------------
#
# Both share: q/k/v projections; per-head q_norm/k_norm (RMSNorm, scaled) and
# v_norm (RMSNorm, weightless); full-width RoPE (the rotary module already emits
# cos/sin at the full head_dim -- 256 for sliding layers, 512 for full layers --
# so the standard rotate-half formula applies directly); GQA (sliding kv=8, full
# kv=2); scaling=1.0; o_proj. When ``v_proj`` is absent (full layers) the value
# stream is the pre-norm key projection, then v_norm'd. ``scaled_dot_product_
# attention`` is NOT softcapped here (only the LM head logits are, elsewhere).
#
# Encoder ``is_causal`` follows config.use_bidirectional_attention; decoder
# attention is always bidirectional and (in the full model) prepends a read-only
# encoder KV cache. For Tier2 (single module, no KV cache, no mask) both reduce
# to the same full bidirectional attention.


def _rotate_half_apply(x, cos, sin):
    """(x * cos) + (rotate_half(x) * sin), pure TTNN.

    x: ``[B, S, H, hd]``; cos/sin broadcast as ``[B, S, 1, hd]``.
    """
    shp = [int(d) for d in x.shape]
    half = shp[-1] // 2
    x1 = ttnn.slice(x, [0, 0, 0, 0], [shp[0], shp[1], shp[2], half], [1, 1, 1, 1])
    x2 = ttnn.slice(x, [0, 0, 0, half], shp, [1, 1, 1, 1])
    rot = ttnn.concat([ttnn.neg(x2), x1], dim=-1)
    return ttnn.add(ttnn.multiply(x, cos), ttnn.multiply(rot, sin))


class _DiffusionGemmaAttentionBase(StatelessTTNNModule):
    """Shared pure-TTNN attention. Reference: tt_transformers/tt/attention.py."""

    _IS_CAUSAL = False

    @classmethod
    def from_torch(cls, attn, role_prefix=""):
        new = cls()
        new._fallback_torch_layer = attn
        new._head_dim = int(attn.head_dim)
        new._num_heads = int(attn.q_proj.out_features) // new._head_dim
        new._num_kv_heads = int(attn.k_proj.out_features) // new._head_dim
        new._scaling = float(attn.scaling)
        new._eps = float(getattr(attn.config, "rms_norm_eps", 1e-6))
        new._has_v_proj = attn.v_proj is not None
        # Honor the HF module's is_causal. The REAL model runs sdpa, under which
        # attention_mask=None + is_causal=True yields CAUSAL attention (verified:
        # model-forward layer with mask=None == explicit-causal, PCC 1.0). Encoder
        # attention is_causal=True (causal); decoder is_causal=False (bidirectional,
        # cross-attends read-only encoder KV). A bare standalone HF layer (impl=None,
        # eager) is non-causal, but that path is NOT the real model.
        new._is_causal = bool(getattr(attn, "is_causal", cls._IS_CAUSAL))
        # Sharding plan: q always column-parallel (head-shard num_heads/TP).
        # KV column-parallel iff num_kv_heads is divisible by TP (sliding, kv=8 -> 2/chip);
        # else (full layers, kv=2 < TP=4) REPLICATE kv. o_proj row-parallel + all_reduce,
        # its input dim sharded to local_q_heads*hd.
        new._local_q_heads = new._num_heads // _TP
        new._kv_sharded = new._num_kv_heads % _TP == 0
        # local kv heads per chip: sliding -> kv/TP; full (kv<TP) -> repeat kv to TP -> 1/chip.
        new._local_kv_heads = (
            (new._num_kv_heads // _TP) if new._kv_sharded else (_TP // new._num_kv_heads) * new._num_kv_heads // _TP
        )
        new.q_proj = _ColParallelLinearP150x4.from_torch(attn.q_proj).set_cache_role(f"{role_prefix}attn_q")
        if new._kv_sharded:
            new.k_proj = _ColParallelLinearP150x4.from_torch(attn.k_proj).set_cache_role(f"{role_prefix}attn_k")
            if new._has_v_proj:
                new.v_proj = _ColParallelLinearP150x4.from_torch(attn.v_proj).set_cache_role(f"{role_prefix}attn_v")
        else:
            # kv heads < TP (full layers): repeat kv to TP heads then col-shard -> 1/chip,
            # aligned to the chip's q-head GQA block.
            new.k_proj = (
                _KVRepeatColParallelLinearP150x4.from_torch(attn.k_proj)
                .set_cache_role(f"{role_prefix}attn_k")
                .set_kv_repeat(new._num_kv_heads, new._head_dim)
            )
            if new._has_v_proj:
                new.v_proj = (
                    _KVRepeatColParallelLinearP150x4.from_torch(attn.v_proj)
                    .set_cache_role(f"{role_prefix}attn_v")
                    .set_kv_repeat(new._num_kv_heads, new._head_dim)
                )
        new.o_proj = _RowParallelLinearAllReduceP150x4.from_torch(attn.o_proj).set_cache_role(f"{role_prefix}attn_o")
        new.q_norm = TTNNDiffusionGemmaRMSNorm.from_torch(attn.q_norm)
        new.k_norm = TTNNDiffusionGemmaRMSNorm.from_torch(attn.k_norm)
        new.v_norm = TTNNDiffusionGemmaRMSNorm.from_torch(attn.v_norm)
        return new

    def _children(self):
        kids = [self.q_proj, self.k_proj, self.o_proj, self.q_norm, self.k_norm, self.v_norm]
        if self._has_v_proj:
            kids.append(self.v_proj)
        return kids

    def preprocess_weights_impl(self):
        for c in self._children():
            c.preprocess_weights()
        super().preprocess_weights_impl()

    def move_weights_to_device_impl(self):
        for c in self._children():
            c.move_weights_to_device()
        super().move_weights_to_device_impl()

    def _qkv(self, hidden_states, position_embeddings):
        """Compute per-head-normed + RoPE'd Q,K,V, permuted to [B, n*, S, hd].

        Sharded: q is head-sharded (``_local_q_heads``/chip); kv is either
        head-sharded (``_local_kv_heads``/chip, sliding) or replicated (full layers)."""
        cos, sin = position_embeddings
        b, s = int(hidden_states.shape[0]), int(hidden_states.shape[1])
        hd = self._head_dim
        lqh, lkvh = self._local_q_heads, self._local_kv_heads
        cos = ttnn.reshape(cos, (b, s, 1, hd))
        sin = ttnn.reshape(sin, (b, s, 1, hd))

        q = ttnn.reshape(self.q_proj(hidden_states), (b, s, lqh, hd))
        q = self.q_norm(q)
        q = _rotate_half_apply(q, cos, sin)

        k_lin = ttnn.reshape(self.k_proj(hidden_states), (b, s, lkvh, hd))
        # Value stream: dedicated v_proj, else the pre-norm key projection.
        v = ttnn.reshape(self.v_proj(hidden_states), (b, s, lkvh, hd)) if self._has_v_proj else k_lin
        k = self.k_norm(k_lin)
        k = _rotate_half_apply(k, cos, sin)
        v = self.v_norm(v)

        # [B, S, n, hd] -> [B, n, S, hd]
        return (ttnn.permute(q, (0, 2, 1, 3)), ttnn.permute(k, (0, 2, 1, 3)), ttnn.permute(v, (0, 2, 1, 3)))

    def compute_kv(self, hidden_states, position_embeddings):
        """Export this attention's post-norm+RoPE K,V [B, n_kv, S, hd] (encoder cache)."""
        _, k, v = self._qkv(hidden_states, position_embeddings)
        return k, v

    def _attend(self, hidden_states, position_embeddings, encoder_kv=None):
        b, s = int(hidden_states.shape[0]), int(hidden_states.shape[1])
        lqh, hd = self._local_q_heads, self._head_dim
        q, k, v = self._qkv(hidden_states, position_embeddings)
        if encoder_kv is not None:
            # Decoder cross-attention: prepend the read-only encoder KV on the seq dim.
            enc_k, enc_v = encoder_kv
            k = ttnn.concat([enc_k, k], dim=2)
            v = ttnn.concat([enc_v, v], dim=2)

        attn = ttnn.transformer.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=self._is_causal,
            scale=self._scaling,
            compute_kernel_config=_FP32_ACC,
        )
        attn = ttnn.permute(attn, (0, 2, 1, 3))  # -> [B, S, local_q_heads, hd]
        # o_proj is row-parallel: its sharded input dim == local_q_heads*hd; it
        # all_reduces back to the replicated full-H output.
        attn = ttnn.reshape(attn, (b, s, lqh * hd))
        return self.o_proj(attn)


class TTNNDiffusionGemmaEncoderTextAttention(_DiffusionGemmaAttentionBase):
    """Encoder attention. ``is_causal`` follows use_bidirectional_attention; for
    the Tier2 single-module path (no mask) it is full bidirectional attention."""

    @run_on_devices(_ARCH)
    def forward(self, hidden_states, position_embeddings, attention_mask=None, **kwargs):
        return self._attend(hidden_states, position_embeddings)


class TTNNDiffusionGemmaDecoderTextAttention(_DiffusionGemmaAttentionBase):
    """Decoder attention (always bidirectional; full model prepends read-only
    encoder KV -- not exercised by the Tier2 single-module path)."""

    @run_on_devices(_ARCH)
    def forward(
        self, hidden_states, position_embeddings, attention_mask=None, past_key_values=None, encoder_kv=None, **kwargs
    ):
        return self._attend(hidden_states, position_embeddings, encoder_kv=encoder_kv)


# ---------------------------------------------------------------------------
# Decoder/Encoder layer (Tier3) -- dual-FFN sandwich (MLP branch + MoE branch)
# ---------------------------------------------------------------------------
#
# HF ``DiffusionGemma{Encoder,Decoder}TextLayer.forward`` (identical structure;
# they differ only in the attention class):
#
#   residual = x
#   x = input_layernorm(x); x = self_attn(x, pos, mask); x = post_attention_layernorm(x)
#   x = residual + x
#   residual = x
#   x1 = post_feedforward_layernorm_1(mlp(pre_feedforward_layernorm(x)))     # MLP branch
#   f  = residual.reshape(-1, H)                                             # flat tokens
#   _, w, idx = router(f)                                                    # router on UN-normed
#   x2 = post_feedforward_layernorm_2(experts(pre_feedforward_layernorm_2(f), idx, w))  # MoE branch
#   x  = post_feedforward_layernorm(x1 + x2)
#   x  = (residual + x) * layer_scalar
#
# Norms are scaled RMSNorm; residual adds / layer_scalar multiply are pure-TTNN
# compute-path ops. Children are invoked via .forward() for unambiguous raw-ttnn
# tensor flow (same convention validated for the attention/MoE modules).
_LAYER_NORMS = (
    "input_layernorm",
    "post_attention_layernorm",
    "pre_feedforward_layernorm",
    "post_feedforward_layernorm",
    "post_feedforward_layernorm_1",
    "post_feedforward_layernorm_2",
    "pre_feedforward_layernorm_2",
)


class _DiffusionGemmaTextLayerBase(StatelessTTNNModule):
    """Shared dual-FFN decoder/encoder layer. Subclasses set ``_ATTN_CLS``."""

    _ATTN_CLS = None

    @classmethod
    def from_torch(cls, layer, role_prefix=""):
        new = cls()
        new._fallback_torch_layer = layer
        new.self_attn = cls._ATTN_CLS.from_torch(layer.self_attn, role_prefix=role_prefix)
        new.mlp = TTNNDiffusionGemmaTextMLP.from_torch(layer.mlp, role_prefix=role_prefix)
        new.router = TTNNDiffusionGemmaTextRouter.from_torch(layer.router)
        new.experts = TTNNDiffusionGemmaTextExperts.from_torch(layer.experts, role_prefix=role_prefix)
        for name in _LAYER_NORMS:
            setattr(new, name, TTNNDiffusionGemmaRMSNorm.from_torch(getattr(layer, name)))
        new._layer_scalar = float(layer.layer_scalar.detach().reshape(-1)[0])
        return new

    def _children(self):
        kids = [self.self_attn, self.mlp, self.router, self.experts]
        kids += [getattr(self, n) for n in _LAYER_NORMS]
        return kids

    def preprocess_weights_impl(self):
        for c in self._children():
            c.preprocess_weights()
        super().preprocess_weights_impl()

    def move_weights_to_device_impl(self):
        for c in self._children():
            c.move_weights_to_device()
        super().move_weights_to_device_impl()

    def compute_kv(self, hidden_states, position_embeddings):
        """Export this layer's self-attention K,V (encoder cache for the decoder)."""
        return self.self_attn.compute_kv(self.input_layernorm.forward(hidden_states), position_embeddings)

    def _layer_forward(self, hidden_states, position_embeddings, attention_mask=None, encoder_kv=None):
        b, s, hdim = (int(hidden_states.shape[0]), int(hidden_states.shape[1]), int(hidden_states.shape[-1]))
        residual = hidden_states
        h = self.input_layernorm.forward(hidden_states)
        h = self.self_attn.forward(h, position_embeddings, attention_mask, encoder_kv=encoder_kv)
        h = self.post_attention_layernorm.forward(h)
        h = ttnn.add(residual, h)

        residual = h
        # MLP branch.
        h1 = self.post_feedforward_layernorm_1.forward(self.mlp.forward(self.pre_feedforward_layernorm.forward(h)))

        # MoE branch -- operates on the flattened post-attn residual.
        flat = ttnn.reshape(residual, (b * s, hdim))
        _, top_k_weights, top_k_index = self.router.forward(flat)
        experts_in = self.pre_feedforward_layernorm_2.forward(flat)
        h2 = self.experts.forward(experts_in, top_k_index, top_k_weights)  # [N,H]
        h2 = ttnn.reshape(h2, (b, s, hdim))
        h2 = self.post_feedforward_layernorm_2.forward(h2)

        h = self.post_feedforward_layernorm.forward(ttnn.add(h1, h2))
        h = ttnn.add(residual, h)
        return ttnn.multiply(h, self._layer_scalar)


@trace_enabled
class TTNNDiffusionGemmaEncoderTextLayer(_DiffusionGemmaTextLayerBase):
    """Encoder layer (uses encoder attention)."""

    _ATTN_CLS = TTNNDiffusionGemmaEncoderTextAttention

    @run_on_devices(_ARCH)
    def forward(self, hidden_states, position_embeddings=None, attention_mask=None, **kwargs):
        return self._layer_forward(hidden_states, position_embeddings, attention_mask)


@trace_enabled
class TTNNDiffusionGemmaDecoderTextLayer(_DiffusionGemmaTextLayerBase):
    """Decoder layer (uses decoder attention; bidirectional, read-only encoder KV)."""

    _ATTN_CLS = TTNNDiffusionGemmaDecoderTextAttention

    @run_on_devices(_ARCH)
    def forward(
        self,
        hidden_states,
        position_embeddings=None,
        attention_mask=None,
        past_key_values=None,
        encoder_kv=None,
        **kwargs,
    ):
        return self._layer_forward(hidden_states, position_embeddings, attention_mask, encoder_kv=encoder_kv)


# ---------------------------------------------------------------------------
# Embeddings + Encoder text model (Tier4 assembly)
# ---------------------------------------------------------------------------
class TTNNDiffusionGemmaTextScaledWordEmbedding(StatelessTTNNModule):
    """``DiffusionGemmaTextScaledWordEmbedding`` -> embedding lookup * sqrt(hidden).

    VOCAB-SHARDED: weight ``[V, H]`` is sharded on dim 0 -> each chip owns
    ``[V/TP, H]`` rows. ``ttnn.embedding`` indexes LOCALLY and auto-zeros local-OOR
    POSITIVE indices, so the forward subtracts the per-chip vocab offset, clamps
    invalid (negative or >= V_local) indices to V_local (a positive OOR row that
    auto-zeros), looks up, and ``all_reduce(sum)`` combines the 4 per-chip partials
    into the replicated full-H embedding. Verified on HW (Stage 0). Shared with the
    decoder.embed_tokens and (separate V-out axis) the lm_head per the SHARE step.
    """

    _SHARDED = True  # set False to fall back to a replicated bf16 embedding

    @classmethod
    def from_torch(cls, embedding, role_prefix="embed"):
        import math

        new = cls()
        new._fallback_torch_layer = embedding
        new._scale = float(getattr(embedding, "scalar_embed_scale", 0.0)) or math.sqrt(embedding.weight.shape[-1])
        new._vocab = int(embedding.weight.shape[0])
        new._role = role_prefix
        return new

    @property
    def weight(self):
        return self.torch_layer.weight

    def preprocess_weights_impl(self):
        self._h_weight = self.torch_layer.weight.detach()

    def move_weights_to_device_impl(self):
        if self._SHARDED:
            vlocal = self._vocab // _TP
            self._v_local = vlocal
            # Embedding weight must be ROW_MAJOR (ttnn.embedding requirement); shard on
            # vocab dim 0 -> [V/TP, H]/chip. (TILE-layout weights produce NaN in embedding.)
            self.tt_weight = ttnn.from_torch(
                self._h_weight,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.device,
                mesh_mapper=ttnn.shard_tensor_to_mesh_mapper(self.device, dim=0),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            # per-chip vocab offset [0, vlocal, 2*vlocal, 3*vlocal] sharded on dim 1.
            import torch

            off = torch.arange(_TP, dtype=torch.float32).reshape(1, _TP) * float(vlocal)
            self.tt_offset = ttnn.from_torch(
                off,
                dtype=ttnn.float32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.device,
                mesh_mapper=ttnn.shard_tensor_to_mesh_mapper(self.device, dim=1),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        else:
            self.tt_weight = ttnn.from_torch(
                self._h_weight,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.device,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        if hasattr(self, "_h_weight"):
            del self._h_weight

    @run_on_devices(_ARCH)
    def forward(self, tt_indices):
        if not self._SHARDED:
            out = ttnn.embedding(
                tt_indices, self.tt_weight, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            return ttnn.multiply(out, self._scale)
        # Vocab-sharded: local = global_idx - chip_offset. Out-of-local-range indices
        # are NOT auto-zeroed by ttnn.embedding (reading row == V_local on an exactly
        # V_local-row shard returns inf), so clamp invalid -> 0 (a valid in-range row)
        # and ZERO the embedding OUTPUT for invalid positions via the validity mask;
        # then all_reduce(sum) over chips assembles the full embedding.
        vlocal = self._v_local
        idx_f = ttnn.typecast(tt_indices, ttnn.float32) if tt_indices.dtype != ttnn.float32 else tt_indices
        local = ttnn.subtract(idx_f, self.tt_offset)  # broadcast [N] - [1]
        valid = ttnn.multiply(ttnn.ge(local, 0.0), ttnn.lt(local, float(vlocal)))  # [.., N]
        safe = ttnn.multiply(local, valid)  # invalid -> 0 (in-range, never OOR)
        safe_u = ttnn.typecast(safe, ttnn.uint32)
        part = ttnn.embedding(
            safe_u, self.tt_weight, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )  # [.., N, H]
        pshape = list(part.shape)
        part4 = ttnn.reshape(part, (1, 1, -1, pshape[-1]))
        # zero out rows whose global index this chip does not own.
        mask = ttnn.reshape(valid, (1, 1, -1, 1))
        mask = ttnn.to_layout(mask, ttnn.TILE_LAYOUT)
        part4 = ttnn.multiply(part4, mask)
        red = ttnn.all_reduce(
            part4, cluster_axis=_TP_AXIS, num_links=_NUM_LINKS, topology=_TOPO, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        red = ttnn.reshape(red, pshape)
        return ttnn.multiply(red, self._scale)


class TTNNDiffusionGemmaEncoderTextModel(StatelessTTNNModule):
    """``DiffusionGemmaEncoderTextModel`` -> embed -> encoder-layer stack -> norm.

    Self-contained (encoder uses plain self-attention). The forward takes
    pre-tokenized ``input_ids`` (ttnn uint32) and a per-layer-type ``cos/sin``
    dict (the host-side rotary tables), mirroring the HF model which builds
    position_embeddings per layer_type. Reduced-depth validation in Tier4;
    full 26B depth does not fit a 4-device Blackhole.
    """

    @classmethod
    def from_torch(cls, model):
        new = cls()
        new._fallback_torch_layer = model
        new.embed_tokens = TTNNDiffusionGemmaTextScaledWordEmbedding.from_torch(model.embed_tokens, role_prefix="embed")
        new.layers = [
            TTNNDiffusionGemmaEncoderTextLayer.from_torch(l, role_prefix=f"L{i}_") for i, l in enumerate(model.layers)
        ]
        new.norm = TTNNDiffusionGemmaRMSNorm.from_torch(model.norm)
        new._layer_types = list(model.config.layer_types)
        new._shared = False  # set by share_text_weights -> skip-load guard
        return new

    def _kids(self):
        return [self.embed_tokens, *self.layers, self.norm]

    def preprocess_weights_impl(self):
        if self._shared:
            return  # weights aliased from the decoder; nothing to preprocess.
        for c in self._kids():
            c.preprocess_weights()
        super().preprocess_weights_impl()

    def move_weights_to_device_impl(self):
        # Skip-load guard: when shared, every tt_* handle is already aliased from the
        # decoder (same DRAM buffer) -> do NOT re-upload (avoids the 2x doubling).
        if self._shared:
            return
        for c in self._kids():
            c.move_weights_to_device()
        super().move_weights_to_device_impl()

    @run_on_devices(_ARCH)
    def forward(self, input_ids, position_embeddings, attention_mask=None):
        h = self.embed_tokens.forward(input_ids)  # [B, S, H]
        for i, layer in enumerate(self.layers):
            h = layer.forward(h, position_embeddings[self._layer_types[i]], attention_mask)
        return self.norm.forward(h)

    @run_on_devices(_ARCH)
    def forward_with_cache(self, input_ids, position_embeddings, attention_mask=None):
        """Encoder forward that also returns the per-layer (K, V) cache the decoder
        cross-attends to. Each layer's K,V is computed from that layer's input
        (post input_layernorm), matching HF's ``past_key_values.update`` semantics."""
        h = self.embed_tokens.forward(input_ids)
        kv_cache = []
        for i, layer in enumerate(self.layers):
            pe = position_embeddings[self._layer_types[i]]
            kv_cache.append(layer.compute_kv(h, pe))
            h = layer.forward(h, pe, attention_mask)
        return self.norm.forward(h), kv_cache


# ---------------------------------------------------------------------------
# Self-conditioning + Decoder text model (Tier4 assembly)
# ---------------------------------------------------------------------------
#
# HF ``DiffusionGemmaSelfConditioning.forward``:
#   normed = pre_norm(sc_signal); sc = down(gelu(gate(normed)) * up(normed))
#   return post_norm(inputs_embeds + sc)            # post_norm is with_scale=False
class TTNNDiffusionGemmaSelfConditioning(StatelessTTNNModule):
    """Gated-MLP self-conditioning: combines prev-step soft embeddings into the
    decoder input embeddings (pre/post RMSNorm sandwich)."""

    @classmethod
    def from_torch(cls, sc, role_prefix="sc_"):
        new = cls()
        new._fallback_torch_layer = sc
        new.pre_norm = TTNNDiffusionGemmaRMSNorm.from_torch(sc.pre_norm)
        new.post_norm = TTNNDiffusionGemmaRMSNorm.from_torch(sc.post_norm)
        new.gate_proj = _ColParallelLinearP150x4.from_torch(sc.gate_proj).set_cache_role(f"{role_prefix}gate")
        new.up_proj = _ColParallelLinearP150x4.from_torch(sc.up_proj).set_cache_role(f"{role_prefix}up")
        new.down_proj = _RowParallelLinearAllReduceP150x4.from_torch(sc.down_proj).set_cache_role(f"{role_prefix}down")
        return new

    def _kids(self):
        return [self.pre_norm, self.post_norm, self.gate_proj, self.up_proj, self.down_proj]

    def preprocess_weights_impl(self):
        for c in self._kids():
            c.preprocess_weights()
        super().preprocess_weights_impl()

    def move_weights_to_device_impl(self):
        for c in self._kids():
            c.move_weights_to_device()
        super().move_weights_to_device_impl()

    @run_on_devices(_ARCH)
    def forward(self, inputs_embeds, sc_signal):
        normed = self.pre_norm.forward(sc_signal)
        sc = self.down_proj.forward(
            ttnn.multiply(ttnn.gelu(self.gate_proj.forward(normed)), self.up_proj.forward(normed))
        )
        return self.post_norm.forward(ttnn.add(inputs_embeds, sc))


class TTNNDiffusionGemmaDecoderTextModel(StatelessTTNNModule):
    """``DiffusionGemmaDecoderModel`` text path -> embed -> self-conditioning ->
    decoder-layer stack (cross-attends to the encoder KV cache) -> norm.

    forward args: ``input_ids`` (ttnn uint32), ``position_embeddings`` dict
    (decoder cos/sin per layer_type, at positions OFFSET past the encoder seq),
    ``encoder_kv_cache`` (list of (K,V) from the encoder's forward_with_cache),
    and optional ``sc_signal`` soft embeddings (zeros if None -> no self-cond).
    Reduced-depth validation; full 26B does not fit a 4-device Blackhole.
    """

    @classmethod
    def from_torch(cls, model):
        new = cls()
        new._fallback_torch_layer = model
        # Decoder is the SUPERSET stack and is built FIRST: its tt_* tensors are the
        # canonical shared handles the encoder later aliases (same cache keys -> tiled
        # ONCE). role_prefix MUST match the encoder's so the alias maps 1:1.
        new.embed_tokens = TTNNDiffusionGemmaTextScaledWordEmbedding.from_torch(model.embed_tokens, role_prefix="embed")
        new.layers = [
            TTNNDiffusionGemmaDecoderTextLayer.from_torch(l, role_prefix=f"L{i}_") for i, l in enumerate(model.layers)
        ]
        new.norm = TTNNDiffusionGemmaRMSNorm.from_torch(model.norm)
        new.self_conditioning = TTNNDiffusionGemmaSelfConditioning.from_torch(
            model.self_conditioning, role_prefix="sc_"
        )
        new._layer_types = list(model.text_config.layer_types)
        return new

    def _kids(self):
        return [self.embed_tokens, self.self_conditioning, *self.layers, self.norm]

    def preprocess_weights_impl(self):
        for c in self._kids():
            c.preprocess_weights()
        super().preprocess_weights_impl()

    def move_weights_to_device_impl(self):
        for c in self._kids():
            c.move_weights_to_device()
        super().move_weights_to_device_impl()

    @run_on_devices(_ARCH)
    def forward(self, input_ids, position_embeddings, encoder_kv_cache, sc_signal=None, attention_mask=None):
        embeds = self.embed_tokens.forward(input_ids)  # [B,S,H]
        # Self-conditioning: zeros signal (no prev step) reduces to post_norm(embeds).
        sig = sc_signal if sc_signal is not None else ttnn.multiply(embeds, 0.0)
        h = self.self_conditioning.forward(embeds, sig)
        for i, layer in enumerate(self.layers):
            h = layer.forward(
                h, position_embeddings[self._layer_types[i]], attention_mask, encoder_kv=encoder_kv_cache[i]
            )
        return self.norm.forward(h)


# ---------------------------------------------------------------------------
# LM head + final logit softcapping (decoder hidden -> vocab logits)
# ---------------------------------------------------------------------------
#
# HF ``DiffusionGemmaForBlockDiffusion``:
#   logits = lm_head(decoder_hidden)            # nn.Linear, weight tied to embed_tokens
#   logits = tanh(logits / softcap) * softcap   # final_logit_softcapping = 30.0
class TTNNDiffusionGemmaLMHead(StatelessTTNNModule):
    """``lm_head`` + final logit softcapping: tanh(lm_head(h)/cap)*cap."""

    @classmethod
    def from_torch(cls, lm_head, final_logit_softcapping=None):
        new = cls()
        new._fallback_torch_layer = lm_head
        # Column-parallel on V (output) -> [H, V/TP]/chip; per-chip softcap commutes;
        # host readback concats on the V axis (NO device all_gather).
        new.lm_head = _ColParallelLinearP150x4.from_torch(lm_head).set_cache_role("lm_head")
        new._softcap = float(final_logit_softcapping) if final_logit_softcapping else 0.0
        return new

    def preprocess_weights_impl(self):
        self.lm_head.preprocess_weights()
        super().preprocess_weights_impl()

    def move_weights_to_device_impl(self):
        self.lm_head.move_weights_to_device()
        super().move_weights_to_device_impl()

    @run_on_devices(_ARCH)
    def forward(self, hidden_states):
        # lm_head is V-sharded -> per-chip logits [B, S, V/TP]; softcap is elementwise
        # so it commutes with the V-shard. Host concats on V (no device all_gather).
        logits = self.lm_head.forward(hidden_states)
        if self._softcap:
            logits = ttnn.multiply(logits, 1.0 / self._softcap)
            logits = ttnn.tanh(logits)
            logits = ttnn.multiply(logits, self._softcap)
        return logits


# ---------------------------------------------------------------------------
# Weight-sharing (SHARE): encoder aliases the decoder's on-device tt_* handles so
# each unique (tied) weight is tiled/uploaded EXACTLY ONCE (~12.6 GB/chip, not ~25).
# Build the DECODER first (superset; owns self_conditioning), set_device it, THEN
# call share_text_weights(encoder, decoder) BEFORE set_device(encoder). The encoder
# then has _shared=True so its move/preprocess are no-ops (skip-load guard).
# ---------------------------------------------------------------------------
def _alias_tt_handles(dst, src):
    """Rebind every ``tt_*`` attribute and nested TTNNModule child of ``dst`` to the
    SAME object on ``src`` (same Python object => same DRAM buffer, no copy). Returns
    the number of tt_* tensor handles aliased."""
    n = 0
    # 1) direct tt_* tensor handles: copy EVERY tt_* from src onto dst (dst may not
    # have them yet -- its move_weights was skipped via the skip-load guard).
    for name in list(vars(src).keys()):
        if name.startswith("tt_"):
            setattr(dst, name, getattr(src, name))
            n += 1
    # also carry over the embedding vocab-shard helper offset if present.
    for name in ("_v_local", "_v_offset"):
        if hasattr(src, name):
            setattr(dst, name, getattr(src, name))
    # 2) recurse into child TTNNModules (matched by attribute name)
    for name, child in list(vars(dst).items()):
        if isinstance(child, StatelessTTNNModule):
            schild = getattr(src, name, None)
            if isinstance(schild, StatelessTTNNModule):
                n += _alias_tt_handles(child, schild)
                # mark child lifecycle complete so set_device skips it
                child._preprocessed_weight = True
                child._weights_on_device = True
                child._device = src.device
        elif isinstance(child, list):
            schild = getattr(src, name, None)
            if isinstance(schild, list) and len(schild) == len(child):
                for dc, sc in zip(child, schild):
                    if isinstance(dc, StatelessTTNNModule) and isinstance(sc, StatelessTTNNModule):
                        n += _alias_tt_handles(dc, sc)
                        dc._preprocessed_weight = True
                        dc._weights_on_device = True
                        dc._device = sc.device
    return n


def share_text_weights(encoder_ttnn, decoder_ttnn):
    """Alias the encoder text stack onto the decoder's on-device weights.

    Aliases per-layer attention q/k/v/o, dense-MLP gate/up/down, router proj/scale,
    experts gate/up/down/eye/sparsity, per-layer RMSNorm weights, plus embed_tokens
    and the final norm. The encoder's OWN forward methods (bidirectional) are kept;
    only the WEIGHT tensors are shared. Marks the encoder _shared so its set_device
    is a no-op. Returns the count of aliased tt_* handles."""
    assert len(encoder_ttnn.layers) == len(decoder_ttnn.layers), "layer count mismatch"

    def _mark_done(m, dev):
        # set_device recurses into every child TTNNModule and calls
        # preprocess/move DIRECTLY (bypassing the parent _shared guard), so each
        # aliased module must look already-bound or it re-uploads.
        m._device = dev
        m._preprocessed_weight = True
        m._weights_on_device = True

    n = 0
    dev = decoder_ttnn.device
    # embed + final norm
    n += _alias_tt_handles(encoder_ttnn.embed_tokens, decoder_ttnn.embed_tokens)
    _mark_done(encoder_ttnn.embed_tokens, dev)
    n += _alias_tt_handles(encoder_ttnn.norm, decoder_ttnn.norm)
    _mark_done(encoder_ttnn.norm, dev)
    # per-layer
    for el, dl in zip(encoder_ttnn.layers, decoder_ttnn.layers):
        n += _alias_tt_handles(el, dl)
        _mark_done(el, dev)
    _mark_done(encoder_ttnn, dev)
    encoder_ttnn._shared = True
    return n


# ---------------------------------------------------------------------------
# Trace-enabled layer stack (dots_ocr pattern: TTNNLayerStack subclass)
# ---------------------------------------------------------------------------
#
# Captures the ENTIRE encoder/decoder layer sequence as a single trace (the
# regime where Metal Trace actually pays off -- per-layer the model is
# device-bound, but a 30-layer stack replayed as one trace eliminates the
# per-layer host-dispatch + the MoE chunk-loop python overhead across all
# layers). DiffusionGemma layers are per-layer-type (sliding/full), so the
# forward dispatches each layer with its own cos/sin (and, for the decoder, its
# encoder KV); ``forward_with_cache`` exports the per-layer K,V for the decoder.
from tt_symbiote.core.module import TTNNLayerStack  # noqa: E402


@trace_enabled
class TTNNDiffusionGemmaLayerStack(TTNNLayerStack):
    """Stack of {Encoder,Decoder}TextLayer with per-layer-type position embeddings.

    THE trace unit (decorator-only tracing): under ``TT_SYMBIOTE_RUN_MODE=TRACED`` the
    whole layer sequence is captured ONCE and replayed each denoising step. Only the
    positional ``hidden_states`` is copied into the trace input buffer per replay; the
    constant ``position_embeddings`` (dict) and ``encoder_kv_cache`` (list of (K,V)) are
    captured by reference to their resident device buffers (computed once before the
    loop). ``_bypass_tensor_wrapping`` keeps the pure-ttnn boundary (raw ttnn in/out, no
    ``TorchTTNNTensor`` wrapping) so the eager embed/self-cond/norm/lm_head around the
    traced stack feed it directly."""

    @classmethod
    def from_layers(cls, layers, layer_types):
        new = cls(layers)
        new._layer_types = list(layer_types)
        new._bypass_tensor_wrapping = True
        return new

    @run_on_devices(_ARCH)
    def forward(self, hidden_states, position_embeddings=None, attention_mask=None, encoder_kv_cache=None):
        for i, layer in enumerate(self.layers):
            pe = position_embeddings[self._layer_types[i]]
            ekv = encoder_kv_cache[i] if encoder_kv_cache is not None else None
            hidden_states = layer.forward(hidden_states, pe, attention_mask, encoder_kv=ekv)
        return hidden_states

    def forward_with_cache(self, hidden_states, position_embeddings, attention_mask=None):
        """Encoder pass that also exports each layer's (K, V) for decoder cross-attn."""
        kv_cache = []
        for i, layer in enumerate(self.layers):
            pe = position_embeddings[self._layer_types[i]]
            kv_cache.append(layer.compute_kv(hidden_states, pe))
            hidden_states = layer.forward(hidden_states, pe, attention_mask)
        return hidden_states, kv_cache
