# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os

import torch
import ttnn

from tt_symbiote.core.d2d_bridge import D2DBridge
from tt_symbiote.core.d2d_pipeline import Pipeline
from tt_symbiote.core.d2d_transport import SocketTransport
from tt_symbiote.core.module import DeviceArch, StatelessTTNNModule, run_on_devices
from tt_symbiote.core.run_config import trace_enabled
from tt_symbiote.models.pi05.modeling_pi05_common import precompute_freqs_cis_meta
from tt_symbiote.models.pi05.modeling_pi05_gemma import TTNNPi05AdaRMSGemmaBlock, _linear_weight_to_tt
from tt_symbiote.models.pi05.modeling_pi05_suffix import TTNNPi05SuffixEmbedding
from tt_symbiote.models.pipelined_pi05.denoise_block import TTNNPi05DenoiseExpertBlock
from tt_symbiote.utils.device_management import set_device

TT_METAL_COMMIT = "7d1b555a394b7b5413444fc95e5c2d4a54ec197a"


def perf_action_horizon(default: int = 10) -> int:
    return int(os.environ.get("PI05_PERF_AH", default))


def perf_suffix_len(action_horizon: int) -> int:
    return ((action_horizon + 31) // 32) * 32
_L1 = ttnn.L1_MEMORY_CONFIG
_DRAM = ttnn.DRAM_MEMORY_CONFIG


def carve_four_submeshes(parent):
    shape = tuple(int(d) for d in parent.shape)
    if shape == (1, 4):
        coords = [(0, 0), (0, 1), (0, 2), (0, 3)]
    elif shape == (4, 1):
        coords = [(0, 0), (1, 0), (2, 0), (3, 0)]
    elif shape == (2, 2):
        coords = [(0, 0), (0, 1), (1, 0), (1, 1)]
    else:
        raise ValueError(
            f"carve_four_submeshes: unsupported parent mesh shape {shape}; "
            f"expected a 4-device mesh (1x4 / 4x1 / 2x2)."
        )
    return tuple(
        parent.create_submesh(ttnn.MeshShape(1, 1), ttnn.MeshCoordinate(r, c)) for (r, c) in coords
    )


_FILL_CACHE_SHIM_ATTR = "_pi05_denoise_fill_cache_shim"


def _install_fill_cache_shim():
    if getattr(ttnn, _FILL_CACHE_SHIM_ATTR, False):
        return
    _native_fill_cache = ttnn.fill_cache

    def _fill_cache_compat(cache_tensor, input_tensor, batch_idx, *, update_idx=0):
        if update_idx == 0:
            return _native_fill_cache(cache_tensor, input_tensor, batch_idx)
        s = input_tensor.shape[-2]
        hd = input_tensor.shape[-1]
        for i in range(s):
            row = ttnn.slice(input_tensor, [0, 0, i, 0], [1, 1, i + 1, hd])
            ttnn.update_cache(cache_tensor, row, update_idx + i, batch_offset=batch_idx)
            ttnn.deallocate(row)
        return cache_tensor

    ttnn.fill_cache = _fill_cache_compat
    setattr(ttnn, _FILL_CACHE_SHIM_ATTR, True)


_install_fill_cache_shim()

__all__ = [
    "carve_four_submeshes",
    "TTNNPi05DenoisePipelineStage",
    "build_denoise_pipeline",
    "build_n_stage_pipeline",
    "build_single_stage_reference",
    "TTNNPi05DenoiseExpertBlock",
    "perf_action_horizon",
    "perf_suffix_len",
    "build_denoise_loop_pipeline",
    "TTNNPi05DenoiseStreamedPipeline",
]


def _kv_dtype() -> "ttnn.DataType":
    return ttnn.bfloat16 if os.environ.get("PI05_VLM_KV_BF16") == "1" else ttnn.bfloat8_b


def _to_dram(tensors):
    out = []
    for t in tensors:
        d = ttnn.to_memory_config(t, _DRAM)
        ttnn.deallocate(t)
        out.append(d)
    return tuple(out)


def _slice_rope(cos, sin, seq_len, offset):
    hd = cos.shape[-1]
    c = ttnn.slice(cos, [0, 0, offset, 0], [1, 1, offset + seq_len, hd])
    s = ttnn.slice(sin, [0, 0, offset, 0], [1, 1, offset + seq_len, hd])
    return c, s


@trace_enabled
class TTNNPi05DenoisePipelineStage(StatelessTTNNModule):

    def __init__(
        self,
        *,
        blocks,
        expert_config,
        suffix=None,
        is_first=False,
        is_last=False,
        max_seq_len,
        rope_base,
        eps_expert,
        expert_width,
        prefix_len,
        suffix_len,
        position_offset,
        action_horizon,
        use_concat_kv=False,
    ):
        super().__init__()
        self._bypass_tensor_wrapping = True
        self._use_concat_kv = use_concat_kv
        self._prefix_kv = None
        self.blocks = list(blocks)
        self.suffix = suffix
        self._is_first = is_first
        self._is_last = is_last
        self._expert_config = expert_config
        self._max_seq_len = max_seq_len
        self._rope_base = rope_base
        self._eps_expert = eps_expert
        self._expert_width = expert_width
        self._prefix_len = prefix_len
        self._suffix_len = suffix_len
        self._position_offset = position_offset
        self._action_horizon = action_horizon
        self._raw_final_norm_mod_w = None
        self._raw_final_norm_mod_b = None
        self._precomputed_block_mods = None
        self._precomputed_final_mod = None
        self._attention_mask = None
        self._tt_final_mod_w = None
        self._tt_final_mod_b = None
        self._tt_expert_norm_ones = None
        self.tt_cos_expert = None
        self.tt_sin_expert = None

    def preprocess_weights_impl(self):
        if self._is_last and self._raw_final_norm_mod_w is not None:
            self._tt_final_mod_w = _linear_weight_to_tt(
                self._raw_final_norm_mod_w, dtype=ttnn.bfloat16
            )
            self._tt_final_mod_b = (
                ttnn.from_torch(
                    self._raw_final_norm_mod_b.reshape(1, -1).contiguous(),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                )
                if self._raw_final_norm_mod_b is not None
                else None
            )
            self._tt_expert_norm_ones = ttnn.from_torch(
                torch.ones(1, self._expert_width), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
            )

    def move_weights_to_device_impl(self):
        dev = self.device
        self.tt_cos_expert, self.tt_sin_expert = precompute_freqs_cis_meta(
            self._expert_config.head_dim, self._max_seq_len, dev, self._rope_base
        )
        if self._is_last and self._tt_final_mod_w is not None:
            self._tt_final_mod_w = ttnn.to_device(self._tt_final_mod_w, dev, memory_config=_DRAM)
            if self._tt_final_mod_b is not None:
                self._tt_final_mod_b = ttnn.to_device(self._tt_final_mod_b, dev, memory_config=_DRAM)
            self._tt_expert_norm_ones = ttnn.to_device(self._tt_expert_norm_ones, dev, memory_config=_DRAM)

    def _precompute_final_mod(self, cond_dev):
        m = ttnn.linear(cond_dev, self._tt_final_mod_w, bias=self._tt_final_mod_b, memory_config=_L1)
        b = m.shape[0]
        W = self._expert_width
        scale = ttnn.reshape(ttnn.slice(m, [0, 0], [b, W]), (b, 1, W))
        shift = ttnn.reshape(ttnn.slice(m, [0, W], [b, 2 * W]), (b, 1, W))
        ttnn.deallocate(m)
        scale1 = ttnn.add(scale, 1.0, memory_config=_L1)
        ttnn.deallocate(scale)
        return _to_dram((scale1, shift))

    def _ada_rms_norm_no_gate(self, x, precomputed):
        scale1, shift = precomputed
        normed = ttnn.rms_norm(x, weight=self._tt_expert_norm_ones, epsilon=self._eps_expert, memory_config=_L1)
        out = ttnn.multiply(normed, scale1, memory_config=_L1)
        out = ttnn.add(out, shift, memory_config=_L1)
        ttnn.deallocate(normed)
        return out

    @run_on_devices(DeviceArch.P150)
    def forward(self, x):
        if self._is_first:
            h = self.suffix.embed_actions(x)
        else:
            h = x
        s = h.shape[-2]
        cos, sin = _slice_rope(self.tt_cos_expert, self.tt_sin_expert, s, self._position_offset)
        for i, block in enumerate(self.blocks):
            block_mod = self._precomputed_block_mods[i]
            if self._use_concat_kv:
                pk, pv = self._prefix_kv[i]
                h, _ = block(h, cos, sin, None, self._attention_mask, (pk, pv), False, precomputed_mod=block_mod)
            else:
                h, _ = block(h, cos, sin, None, self._attention_mask, None, False, precomputed_mod=block_mod)
        ttnn.deallocate(cos)
        ttnn.deallocate(sin)
        if self._is_last:
            h = self._ada_rms_norm_no_gate(h, self._precomputed_final_mod)
            h = self.suffix.project_output(h)
        if h.memory_config().buffer_type != ttnn.BufferType.L1:
            h = ttnn.to_memory_config(h, _L1)
        return h


def _bind_stage_runtime(
    stages,
    submeshes_n,
    bounds,
    *,
    config,
    suffix_config,
    adarms_cond_torch,
    prefix_kv_cache,
    prefix_len,
    suffix_len,
    attention_mask_torch,
):
    kvd = _kv_dtype()
    for k, (lo, hi) in enumerate(bounds):
        st, mesh = stages[k], submeshes_n[k]
        if st._use_concat_kv:
            kvd_concat = _kv_dtype()
            st._prefix_kv = []
            for j, blk in enumerate(st.blocks):
                pk, pv = prefix_kv_cache[lo + j]
                pk_dev = ttnn.from_torch(pk, dtype=kvd_concat, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=_L1)
                pv_dev = ttnn.from_torch(pv, dtype=kvd_concat, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=_L1)
                st._prefix_kv.append((pk_dev, pv_dev))
        else:
            for j, blk in enumerate(st.blocks):
                pk, pv = prefix_kv_cache[lo + j]
                blk.init_static_kv(prefix_len, suffix_len)
                k_dev = ttnn.from_torch(pk, dtype=kvd, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=_DRAM)
                v_dev = ttnn.from_torch(pv, dtype=kvd, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=_DRAM)
                blk.fill_static_prefix(k_dev, v_dev)
                ttnn.deallocate(k_dev)
                ttnn.deallocate(v_dev)
        cond_dev = ttnn.from_torch(adarms_cond_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)
        st._precomputed_block_mods = [_to_dram(blk.precompute_mods(cond_dev)) for blk in st.blocks]
        if st._is_last:
            st._precomputed_final_mod = st._precompute_final_mod(cond_dev)
        ttnn.deallocate(cond_dev)
        st._attention_mask = ttnn.from_torch(
            attention_mask_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=_DRAM
        )


def build_n_stage_pipeline(
    reference_blocks,
    reference_final_mod_w,
    reference_final_mod_b,
    reference_suffix,
    config,
    suffix_config,
    parent_mesh,
    *,
    adarms_cond_torch,
    prefix_kv_cache,
    prefix_len,
    suffix_len=64,
    attention_mask_torch,
    position_offset,
    splits,
    submeshes=None,
    block_cls=TTNNPi05AdaRMSGemmaBlock,
    use_concat_kv=False,
) -> Pipeline:
    ec = config.expert_config
    n = len(splits)
    assert n >= 2 and sum(splits) == len(reference_blocks) == 18
    assert prefix_len % 32 == 0 and suffix_len % 32 == 0 and suffix_len <= 96
    assert attention_mask_torch is not None, "phantom-suffix mask is REQUIRED"

    tt_blocks = [block_cls.from_torch(b, ec) for b in reference_blocks]
    suffix0 = TTNNPi05SuffixEmbedding.from_torch(reference_suffix, suffix_config)
    suffixN = TTNNPi05SuffixEmbedding.from_torch(reference_suffix, suffix_config)
    bounds, acc = [], 0
    for sp in splits:
        bounds.append((acc, acc + sp))
        acc += sp
    common = dict(
        expert_config=ec,
        max_seq_len=config.max_seq_len,
        rope_base=ec.rope_base,
        eps_expert=ec.rms_norm_eps,
        expert_width=ec.width,
        prefix_len=prefix_len,
        suffix_len=suffix_len,
        position_offset=position_offset,
        action_horizon=suffix_config.action_horizon,
        use_concat_kv=use_concat_kv,
    )
    stages = []
    for k, (lo, hi) in enumerate(bounds):
        stages.append(
            TTNNPi05DenoisePipelineStage(
                blocks=tt_blocks[lo:hi],
                suffix=(suffix0 if k == 0 else (suffixN if k == n - 1 else None)),
                is_first=(k == 0),
                is_last=(k == n - 1),
                **common,
            )
        )
    stages[-1]._raw_final_norm_mod_w = reference_final_mod_w
    stages[-1]._raw_final_norm_mod_b = reference_final_mod_b
    sm = submeshes if submeshes is not None else carve_four_submeshes(parent_mesh)
    submeshes_n = sm[:n]
    for st, mesh in zip(stages, submeshes_n):
        set_device(st, mesh)
    _bind_stage_runtime(
        stages,
        submeshes_n,
        bounds,
        config=config,
        suffix_config=suffix_config,
        adarms_cond_torch=adarms_cond_torch,
        prefix_kv_cache=prefix_kv_cache,
        prefix_len=prefix_len,
        suffix_len=suffix_len,
        attention_mask_torch=attention_mask_torch,
    )
    bridges = [D2DBridge(stages[i], stages[i + 1], transport=SocketTransport(), tag=f"hop{i}") for i in range(n - 1)]
    return Pipeline(bridges, sync_on_return=True)


def build_denoise_pipeline(
    reference_blocks,
    reference_final_mod_w,
    reference_final_mod_b,
    reference_suffix,
    config,
    suffix_config,
    parent_mesh,
    *,
    adarms_cond_torch,
    prefix_kv_cache,
    prefix_len,
    suffix_len=64,
    attention_mask_torch,
    position_offset,
    splits=(5, 5, 4, 4),
    submeshes=None,
    block_cls=TTNNPi05AdaRMSGemmaBlock,
    use_concat_kv=False,
) -> Pipeline:
    assert len(splits) == 4
    return build_n_stage_pipeline(
        reference_blocks,
        reference_final_mod_w,
        reference_final_mod_b,
        reference_suffix,
        config,
        suffix_config,
        parent_mesh,
        adarms_cond_torch=adarms_cond_torch,
        prefix_kv_cache=prefix_kv_cache,
        prefix_len=prefix_len,
        suffix_len=suffix_len,
        attention_mask_torch=attention_mask_torch,
        position_offset=position_offset,
        splits=splits,
        submeshes=submeshes,
        block_cls=block_cls,
        use_concat_kv=use_concat_kv,
    )


def build_single_stage_reference(
    reference_blocks,
    reference_final_mod_w,
    reference_final_mod_b,
    reference_suffix,
    config,
    suffix_config,
    submesh,
    *,
    adarms_cond_torch,
    prefix_kv_cache,
    prefix_len,
    suffix_len=64,
    attention_mask_torch,
    position_offset,
    block_cls=TTNNPi05AdaRMSGemmaBlock,
    use_concat_kv=False,
) -> TTNNPi05DenoisePipelineStage:
    ec = config.expert_config
    assert len(reference_blocks) == 18
    assert prefix_len % 32 == 0 and suffix_len % 32 == 0 and suffix_len <= 96
    assert attention_mask_torch is not None
    tt_blocks = [block_cls.from_torch(b, ec) for b in reference_blocks]
    suffix = TTNNPi05SuffixEmbedding.from_torch(reference_suffix, suffix_config)
    stage = TTNNPi05DenoisePipelineStage(
        blocks=tt_blocks,
        suffix=suffix,
        is_first=True,
        is_last=True,
        expert_config=ec,
        max_seq_len=config.max_seq_len,
        rope_base=ec.rope_base,
        eps_expert=ec.rms_norm_eps,
        expert_width=ec.width,
        prefix_len=prefix_len,
        suffix_len=suffix_len,
        position_offset=position_offset,
        action_horizon=suffix_config.action_horizon,
        use_concat_kv=use_concat_kv,
    )
    stage._raw_final_norm_mod_w = reference_final_mod_w
    stage._raw_final_norm_mod_b = reference_final_mod_b
    set_device(stage, submesh)
    kvd = _kv_dtype()
    if use_concat_kv:
        kvd_concat = _kv_dtype()
        stage._prefix_kv = []
        for j, blk in enumerate(stage.blocks):
            pk, pv = prefix_kv_cache[j]
            pk_dev = ttnn.from_torch(pk, dtype=kvd_concat, layout=ttnn.TILE_LAYOUT, device=submesh, memory_config=_L1)
            pv_dev = ttnn.from_torch(pv, dtype=kvd_concat, layout=ttnn.TILE_LAYOUT, device=submesh, memory_config=_L1)
            stage._prefix_kv.append((pk_dev, pv_dev))
    else:
        for j, blk in enumerate(stage.blocks):
            pk, pv = prefix_kv_cache[j]
            blk.init_static_kv(prefix_len, suffix_len)
            k_dev = ttnn.from_torch(pk, dtype=kvd, layout=ttnn.TILE_LAYOUT, device=submesh, memory_config=_DRAM)
            v_dev = ttnn.from_torch(pv, dtype=kvd, layout=ttnn.TILE_LAYOUT, device=submesh, memory_config=_DRAM)
            blk.fill_static_prefix(k_dev, v_dev)
            ttnn.deallocate(k_dev)
            ttnn.deallocate(v_dev)
    cond_dev = ttnn.from_torch(adarms_cond_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=submesh)
    stage._precomputed_block_mods = [_to_dram(blk.precompute_mods(cond_dev)) for blk in stage.blocks]
    stage._precomputed_final_mod = stage._precompute_final_mod(cond_dev)
    ttnn.deallocate(cond_dev)
    stage._attention_mask = ttnn.from_torch(
        attention_mask_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=submesh, memory_config=_DRAM
    )
    return stage


def euler_schedule(num_steps):
    timesteps = [1.0 - i / num_steps for i in range(num_steps + 1)]
    dts = [timesteps[i + 1] - timesteps[i] for i in range(num_steps)]
    return timesteps, dts


class TTNNPi05DenoiseStreamedPipeline:

    def __init__(self, pipeline, *, num_steps, action_horizon, per_step_block_mods, per_step_final_mod, dts, drain="all"):
        self._pipe = pipeline
        self._stages = pipeline.stages
        self._meshes = pipeline.meshes
        self._bridges = [pipeline._hop_in[i] for i in range(1, len(self._stages))]
        self._n = num_steps
        self._ah = action_horizon
        self._block_mods = per_step_block_mods
        self._final_mod = per_step_final_mod
        self._dts = dts
        self._drain = drain
        self._stage0_mesh = self._meshes[0]
        self._last_mesh = self._meshes[-1]
        self._wrap_tp = SocketTransport()
        self._wrap_tag = "velocity_wrap"
        self._x_t = None
        self._hop_sock = None
        self._wrap_sock = None
        self._loop_tids = None

    def _set_step_mods(self, i):
        for k, st in enumerate(self._stages):
            st._precomputed_block_mods = self._block_mods[i][k]
            if st._is_last:
                st._precomputed_final_mod = self._final_mod[i]

    def _emit_step(self, i):
        self._set_step_mods(i)
        x_bf16 = ttnn.typecast(self._x_t, ttnn.bfloat16, memory_config=_L1)
        out0 = self._stages[0].forward(x_bf16)
        ttnn.deallocate(x_bf16)
        self._bridges[0].transport.send_only(_as_l1_dev(out0), self._hop_sock[1]["ss"])
        out = out0
        for s in range(1, len(self._stages)):
            sk = self._hop_sock[s]
            self._bridges[s - 1].transport.recv_only(sk["buf"], sk["rs"])
            out = self._stages[s].forward(_rewrap(out, sk["buf"]))
            if s < len(self._stages) - 1:
                self._bridges[s].transport.send_only(_as_l1_dev(out), self._hop_sock[s + 1]["ss"])
        velocity = out
        vsrc = _as_l1_dev(velocity)
        self._wrap_tp.send_only(vsrc, self._wrap_sock["ss"])
        self._wrap_tp.recv_only(self._wrap_sock["buf"], self._wrap_sock["rs"])
        recv = self._wrap_sock["buf"]
        v_fp32 = ttnn.typecast(recv, ttnn.float32, memory_config=_L1)
        v_scaled = ttnn.multiply(v_fp32, self._dts[i], memory_config=_L1)
        ttnn.deallocate(v_fp32)
        ttnn.add(self._x_t, v_scaled, output_tensor=self._x_t)
        ttnn.deallocate(v_scaled)

    def _warmup_caches(self, x_t_init):
        self._x_t = ttnn.from_torch(x_t_init, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=self._stage0_mesh, memory_config=_L1)
        self._hop_sock = [None]
        x_bf16 = ttnn.typecast(self._x_t, ttnn.bfloat16, memory_config=_L1)
        out = self._stages[0].forward(x_bf16)
        ttnn.deallocate(x_bf16)
        for s in range(1, len(self._stages)):
            b = self._bridges[s - 1]
            src = _as_l1_dev(out)
            ss, rs, buf = b.transport.prepare(src, b.mesh_b, tag=b.tag)
            b.transport.send_only(src, ss)
            b.transport.recv_only(buf, rs)
            self._hop_sock.append({"ss": ss, "rs": rs, "buf": buf})
            out = self._stages[s].forward(_rewrap(out, buf))
        velocity = out
        vsrc = _as_l1_dev(velocity)
        ws, wr, wbuf = self._wrap_tp.prepare(vsrc, self._stage0_mesh, tag=self._wrap_tag)
        self._wrap_tp.send_only(vsrc, ws)
        self._wrap_tp.recv_only(wbuf, wr)
        self._wrap_sock = {"ss": ws, "rs": wr, "buf": wbuf}
        v_fp32 = ttnn.typecast(wbuf, ttnn.float32, memory_config=_L1)
        v_scaled = ttnn.multiply(v_fp32, self._dts[0], memory_config=_L1)
        ttnn.deallocate(v_fp32)
        ttnn.add(self._x_t, v_scaled, output_tensor=self._x_t)
        ttnn.deallocate(v_scaled)
        for m in self._pipe._distinct_meshes(self._meshes):
            ttnn.synchronize_device(m)

    def stream_euler(self, x_t_init, *, capture=True):
        self._warmup_caches(x_t_init)
        ttnn.copy(
            ttnn.from_torch(x_t_init, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=self._stage0_mesh, memory_config=_L1),
            self._x_t,
        )
        if not capture:
            for i in range(self._n):
                self._emit_step(i)
                for m in self._pipe._distinct_meshes(self._meshes):
                    ttnn.synchronize_device(m)
        else:
            self._loop_tids = self._pipe.capture_loop(self._meshes, self._emit_step, self._n)
            drain_mesh = self._stage0_mesh if self._drain == "stage0" else None
            self._pipe.replay_loop(self._loop_tids, drain=self._drain, drain_mesh=drain_mesh)
        for m in self._pipe._distinct_meshes(self._meshes):
            ttnn.synchronize_device(m)
        host = ttnn.to_torch(self._x_t)
        return host[:, : self._ah, :]

    def replay(self):
        assert self._loop_tids is not None, "call stream_euler(capture=True) first"
        drain_mesh = self._stage0_mesh if self._drain == "stage0" else None
        self._pipe.replay_loop(self._loop_tids, drain=self._drain, drain_mesh=drain_mesh)
        for m in self._pipe._distinct_meshes(self._meshes):
            ttnn.synchronize_device(m)
        return ttnn.to_torch(self._x_t)[:, : self._ah, :]

    def close(self):
        self._pipe.release_loop(self._loop_tids)
        self._loop_tids = None
        try:
            self._wrap_tp.close()
        except Exception:
            pass


def _as_l1_dev(out):
    src = out if isinstance(out, ttnn.Tensor) else getattr(out, "ttnn_tensor", out)
    if src.memory_config().buffer_type != ttnn.BufferType.L1:
        src = ttnn.to_memory_config(src, _L1)
    return src


def _rewrap(template, recv):
    if isinstance(template, ttnn.Tensor):
        return recv
    try:
        return type(template)(recv)
    except Exception:
        return recv


def build_denoise_loop_pipeline(
    reference_blocks,
    reference_final_mod_w,
    reference_final_mod_b,
    reference_suffix,
    config,
    suffix_config,
    parent_mesh,
    *,
    adarms_cond_per_step,
    prefix_kv_cache,
    prefix_len,
    suffix_len,
    attention_mask_torch,
    position_offset,
    num_steps,
    action_horizon,
    splits=(5, 5, 4, 4),
    submeshes=None,
    block_cls=TTNNPi05DenoiseExpertBlock,
    use_concat_kv=True,
    drain="all",
):
    assert len(adarms_cond_per_step) == num_steps, "per-step adarms_cond REQUIRED (len == num_steps)"
    pipe = build_n_stage_pipeline(
        reference_blocks,
        reference_final_mod_w,
        reference_final_mod_b,
        reference_suffix,
        config,
        suffix_config,
        parent_mesh,
        adarms_cond_torch=adarms_cond_per_step[0],
        prefix_kv_cache=prefix_kv_cache,
        prefix_len=prefix_len,
        suffix_len=suffix_len,
        attention_mask_torch=attention_mask_torch,
        position_offset=position_offset,
        splits=splits,
        submeshes=submeshes,
        block_cls=block_cls,
        use_concat_kv=use_concat_kv,
    )
    stages = pipe.stages
    meshes = pipe.meshes
    per_step_block_mods = []
    per_step_final_mod = []
    for i in range(num_steps):
        step_block = []
        step_final = None
        for k, st in enumerate(stages):
            cond_dev = ttnn.from_torch(adarms_cond_per_step[i], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=meshes[k])
            step_block.append([_to_dram(blk.precompute_mods(cond_dev)) for blk in st.blocks])
            if st._is_last:
                step_final = st._precompute_final_mod(cond_dev)
            ttnn.deallocate(cond_dev)
        per_step_block_mods.append(step_block)
        per_step_final_mod.append(step_final)
    _, dts = euler_schedule(num_steps)
    return TTNNPi05DenoiseStreamedPipeline(
        pipe,
        num_steps=num_steps,
        action_horizon=action_horizon,
        per_step_block_mods=per_step_block_mods,
        per_step_final_mod=per_step_final_mod,
        dts=dts,
        drain=drain,
    )
