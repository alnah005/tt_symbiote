# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""One-time (CPU, torch) capture of per-op reference activations for FAST micro-tests.

The full-model PCC tests reload the ~3.3B model (~9s) and run the whole vision
tower every iteration -- a terrible inner loop for precision debugging. This
script runs the torch reference ONCE on CPU and dumps, for each SAM conv (and the
SAM encoder / CLIP / DeepEncoder), a small fixture containing:
    - the op's weights (state_dict)          -> rebuild the op standalone
    - its REAL input activation (NCHW/torch)  -> faithful distribution, not synthetic
    - its REAL fp32 output                    -> the PCC reference

Micro-tests then load one fixture, rebuild the single op, run it on TTNN, and
assert_pcc -- NO 3.3B model load, ~seconds per iteration.

Run (CPU only; does NOT need the TTNN device):
    export TT_METAL_HOME=/home/ttuser/salnahari/tt-metal
    $TT_METAL_HOME/python_env/bin/python \
        tests/experimental/unlimited_ocr/fixtures/capture_reference_activations.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

FIX = Path(__file__).resolve().parent
SEED = 0


def _save(name: str, payload: dict) -> None:
    p = FIX / f"{name}.pt"
    torch.save(payload, p)
    mb = p.stat().st_size / 1e6
    print(f"  saved {p.name}  ({mb:.1f} MB)")


def main() -> int:
    from tt_symbiote.models.unlimited_ocr.reference_loader import load_reference_model

    print("loading real 3.3B reference model (CPU, one-time)...")
    model, _cfg = load_reference_model()
    model.eval()
    torch.set_grad_enabled(False)

    inner = model.model if hasattr(model, "model") else model
    sam = inner.sam_model
    vis = inner.vision_model
    proj = inner.projector

    # ------------------------------------------------------------------
    # Op-sweep linear fixtures: capture the REAL (input, output) + weights of
    # the bottleneck vision / LM linears so the op-sweep can compute per-config
    # PCC from a cached reference WITHOUT reloading the 3.3B model per config.
    # Robust to attr naming: a global nn.Linear forward hook grabs the FIRST
    # occurrence matching each wanted (in_features, out_features). Vision linears
    # are triggered by the sam(px)/vis(px,sam_out)/proj(feats) runs below; the LM
    # linears (lm_*) by the extra text forward at the end of main().
    # ------------------------------------------------------------------
    import torch.nn as _nn

    _WANTED = {
        # key            (in_features, out_features)
        "clip_qkv":      (1024, 3072),   # CLIP-L NoTPAttention fused qkv_proj
        "clip_fc1":      (1024, 4096),   # CLIP-L FFN fc1 (largest CLIP matmul)
        "sam_qkv":       (768,  2304),   # SAM ViT-B fused qkv
        "sam_lin1":      (768,  3072),   # SAM MLPBlock lin1 (4096 tokens: largest SAM matmul)
        "projector":     (2048, 1280),   # vision->text projector
        "lm_gate":       (1280, 6848),   # DeepSeek dense-L0 MLP gate_proj (largest LM matmul)
        "lm_qproj":      (1280, 1280),   # DeepSeek attention q/k/v/o proj
        "lm_head":       (1280, 129280), # final-logit projection (kept bf16 in model)
    }
    _lin_cap: dict[str, dict] = {}

    def _lin_hook(_mod, inp, out):
        key = None
        for k, (i, o) in _WANTED.items():
            if k in _lin_cap:
                continue
            if _mod.in_features == i and _mod.out_features == o:
                key = k
                break
        if key is None:
            return
        _lin_cap[key] = {
            "state_dict": {kk: vv.detach().float().cpu() for kk, vv in _mod.state_dict().items()},
            "in_features": int(_mod.in_features),
            "out_features": int(_mod.out_features),
            "bias": _mod.bias is not None,
            "input": inp[0].detach().float().cpu(),
            "output": out.detach().float().cpu(),
        }

    _lin_handles = [
        m.register_forward_hook(_lin_hook)
        for m in model.modules()
        if isinstance(m, _nn.Linear)
    ]

    # The five SAM conv leaf modules (name -> module).
    neck = sam.neck  # nn.Sequential(Conv2d, LayerNorm2d, Conv2d, LayerNorm2d)
    convs = {
        "patch_embed": sam.patch_embed.proj,   # Conv2d(3,768,k16,s16)
        "neck_conv1": neck[0],                  # Conv2d(768,256,k1)
        "neck_conv2": neck[2],                  # Conv2d(256,256,k3,p1)
        "net_2": sam.net_2,                     # Conv2d(256,512,k3,s2,p1)
        "net_3": sam.net_3,                     # Conv2d(512,1024,k3,s2,p1)
    }

    # Hook each conv to grab its REAL (input, output) during the SAM forward.
    captured: dict[str, dict] = {}

    def _mk_hook(nm, m):
        def _h(_mod, inp, out):
            captured[nm] = {"input_nchw": inp[0].detach().float().cpu(),
                            "output_nchw": out.detach().float().cpu()}
        return _h

    handles = [m.register_forward_hook(_mk_hook(nm, m)) for nm, m in convs.items()]

    # ------------------------------------------------------------------
    # SDPA op-sweep fixtures. SDPA has NO weight dtype -> the sweep varies the
    # Q/K/V INPUT dtype + math_fidelity + fp32_dest_acc + exp_approx + chunk
    # size. Capture each SDPA call site's REAL (q,k,v[,attn_mask]) + the
    # torch-fp32 SDPA output as the PCC reference, by patching
    # torch.nn.functional.scaled_dot_product_attention for the 3 vision sites
    # (CLIP NoTPAttention, SAM window Attention, SAM global Attention) and by
    # reconstructing the LM prefill site (SlidingWindowLlamaAttention runs the
    # EAGER matmul->softmax->matmul path, NOT F.sdpa). Keyed by (seq, heads) so
    # the FIRST occurrence of each distinct shape is captured.
    # ------------------------------------------------------------------
    import math as _math
    import torch.nn.functional as _F

    _sdpa_cap: dict[str, dict] = {}
    _orig_sdpa = _F.scaled_dot_product_attention

    def _sdpa_patch(query, key, value, attn_mask=None, dropout_p=0.0,
                    is_causal=False, scale=None, **kw):
        out = _orig_sdpa(query, key, value, attn_mask=attn_mask,
                         dropout_p=dropout_p, is_causal=is_causal, scale=scale, **kw)
        try:
            q = query.detach()
            hd = int(q.shape[-1]); nh = int(q.shape[1]); seq = int(q.shape[-2])
            # site classification: CLIP=16 heads (no mask); SAM=12 heads (rel-pos
            # mask); window seq=196 vs global seq=4096.
            if nh == 16:
                site = "sdpa_clip"
            elif seq <= 1024:
                site = "sdpa_sam_window"
            else:
                site = "sdpa_sam_global"
            if site not in _sdpa_cap:
                m = None
                if attn_mask is not None:
                    m = attn_mask.detach().to(torch.bfloat16).cpu()
                _sdpa_cap[site] = {
                    "q": q.float().cpu(), "k": key.detach().float().cpu(),
                    "v": value.detach().float().cpu(),
                    "attn_mask": m,
                    "output": out.detach().float().cpu(),
                    "scale": float(scale) if scale is not None else 1.0 / _math.sqrt(hd),
                    "is_causal": bool(is_causal),
                    "heads": nh, "head_dim": hd, "seq": seq,
                }
                print(f"  [sdpa] captured {site}: q={tuple(q.shape)} "
                      f"mask={None if m is None else tuple(m.shape)}")
        except Exception as exc:  # noqa: BLE001
            print(f"  [sdpa] capture skipped: {exc}")
        return out

    _F.scaled_dot_product_attention = _sdpa_patch

    # ------------------------------------------------------------------
    # SAM GLOBAL rel-pos matmul fixture (rq_h @ RhT, static-weight bmm). Hook the
    # first GLOBAL SAM Attention module (rel_pos_h length 2g-1 == 127 -> g=64) and
    # reconstruct the TTNN _rel_pos_bias operands from its REAL input x, exactly as
    # modeling_unlimited_ocr.TTNNUnlimitedOcrSamAttention._rel_pos_bias does.
    # ------------------------------------------------------------------
    _relpos_cap: dict[str, dict] = {}

    def _capture_relpos(mod, x, tag, g):
        # mirror deepencoder Attention.forward q construction (lines 820-832) then
        # the ttnn rel-pos matmul layout (rq_h [g, B_*g, hd] @ RhT [g, hd, g]).
        B, H, W, _C = x.shape
        nh = int(mod.num_heads)
        qkv = mod.qkv(x).reshape(B, H * W, 3, nh, -1).permute(2, 0, 3, 1, 4)
        q = qkv.reshape(3, B * nh, H * W, -1).unbind(0)[0]  # [B*nh, HW, hd]
        q = q.view(B, nh, H * W, -1)                        # [B, nh, HW, hd]
        B_ = B * nh
        hd = q.shape[-1]
        r_q = q.reshape(B_, g, g, hd).float()               # [B_, h, w, c]
        rel_pos_h = mod.rel_pos_h.detach().float()          # [2g-1, hd]
        rel_pos_w = mod.rel_pos_w.detach().float()
        idx = (torch.arange(g)[:, None] - torch.arange(g)[None, :] + (g - 1)).long()
        Rh = rel_pos_h[idx]; Rw = rel_pos_w[idx]            # [g, g, hd]
        RhT = Rh.permute(0, 2, 1).contiguous()             # [g, hd, g]
        RwT = Rw.permute(0, 2, 1).contiguous()
        # rq_h = einsum row-index batching: [g, B_*g, hd]
        rq_h = r_q.permute(1, 0, 2, 3).reshape(g, B_ * g, hd)   # [h, B_*w, c]
        rq_w = r_q.permute(2, 0, 1, 3).reshape(g, B_ * g, hd)   # [w, B_*h, c]
        rel_h = torch.bmm(rq_h, RhT)                            # [g, B_*g, g]
        rel_w = torch.bmm(rq_w, RwT)
        _relpos_cap[f"relpos_h_{tag}"] = {"input": rq_h, "weight": RhT, "output": rel_h}
        _relpos_cap[f"relpos_w_{tag}"] = {"input": rq_w, "weight": RwT, "output": rel_w}
        print(f"  [relpos] captured {tag}: rq={tuple(rq_h.shape)} RhT={tuple(RhT.shape)}")

    _relpos_handles = []
    for m in sam.modules():
        if type(m).__name__ == "Attention" and getattr(m, "use_rel_pos", False):
            L = int(m.rel_pos_h.shape[0])
            g = (L + 1) // 2
            tag = "global" if g >= 32 else "window"
            def _mk_relpos(_m, _tag, _g):
                fired = {"done": False}
                def _h(mod, inp, out):
                    if fired["done"] or f"relpos_h_{_tag}" in _relpos_cap:
                        return
                    fired["done"] = True
                    try:
                        _capture_relpos(mod, inp[0], _tag, _g)
                    except Exception as exc:  # noqa: BLE001
                        print(f"  [relpos] {_tag} skipped: {exc}")
                return _h
            _relpos_handles.append(m.register_forward_hook(_mk_relpos(m, tag, g)))

    # ------------------------------------------------------------------
    # MoE router gate matmul fixture (x_fp32 @ gate.T). Hook MoEGate; reconstruct
    # the fp32 logits (pre-softmax) as the reference. Currently fp32 in the model
    # (accuracy-critical routing) -- sweep confirms whether fp32 is necessary.
    # ------------------------------------------------------------------
    _moe_cap: dict[str, dict] = {}

    def _moe_hook(mod, inp, out):
        if "moe_gate" in _moe_cap:
            return
        x = inp[0].detach()
        h = x.shape[-1]
        x2 = x.reshape(-1, h).float()
        w = mod.weight.detach().float()               # [n_experts, hidden]
        logits = torch.nn.functional.linear(x2, w)    # [T, n_experts]
        _moe_cap["moe_gate"] = {
            "input": x2, "weight": w, "output": logits,
            "n_experts": int(w.shape[0]), "hidden": int(h),
        }
        print(f"  [moe] captured moe_gate: x={tuple(x2.shape)} w={tuple(w.shape)}")

    _moe_handles = [m.register_forward_hook(_moe_hook)
                    for m in model.modules() if type(m).__name__ == "MoEGate"]

    # ------------------------------------------------------------------
    # LM prefill SDPA fixture (eager path -> reconstruct q,k,v,ref from module).
    # ------------------------------------------------------------------
    import sys as _sys

    def _lm_attn_hook(mod, args, kwargs, output):
        if "sdpa_lm_prefill" in _sdpa_cap:
            return
        try:
            hs = kwargs.get("hidden_states", args[0] if args else None)
            attention_mask = kwargs.get("attention_mask", args[1] if len(args) > 1 else None)
            position_ids = kwargs.get("position_ids", args[2] if len(args) > 2 else None)
            if hs is None or hs.shape[1] <= 1:
                return  # decode step, not prefill
            lm_mod = _sys.modules[type(mod).__module__]
            apply_rope = lm_mod._llama_apply_rotary_pos_emb
            repeat_kv = lm_mod._llama_repeat_kv
            cfg = mod.config
            nh = cfg.num_attention_heads
            nkv = cfg.num_key_value_heads
            hd = mod.head_dim
            groups = mod.num_key_value_groups
            bsz, q_len, _ = hs.shape
            q = mod.q_proj(hs).view(bsz, q_len, nh, hd).transpose(1, 2)
            k = mod.k_proj(hs).view(bsz, q_len, nkv, hd).transpose(1, 2)
            v = mod.v_proj(hs).view(bsz, q_len, nkv, hd).transpose(1, 2)
            cos, sin = mod.rotary_emb(v, position_ids)
            q, k = apply_rope(q, k, cos, sin)
            k = repeat_kv(k, groups)
            v = repeat_kv(v, groups)
            aw = torch.matmul(q, k.transpose(2, 3)) / _math.sqrt(hd)
            if attention_mask is not None:
                aw = aw + attention_mask[:, :, :, :k.shape[-2]]
            aw = torch.nn.functional.softmax(aw, dim=-1, dtype=torch.float32)
            out = torch.matmul(aw, v)                      # [bsz, nh, q_len, hd]
            _sdpa_cap["sdpa_lm_prefill"] = {
                "q": q.detach().float().cpu(), "k": k.detach().float().cpu(),
                "v": v.detach().float().cpu(), "attn_mask": None,
                "output": out.detach().float().cpu(),
                "scale": 1.0 / _math.sqrt(hd), "is_causal": True,
                "heads": int(nh), "head_dim": int(hd), "seq": int(q_len),
            }
            print(f"  [sdpa] captured sdpa_lm_prefill: q={tuple(q.shape)} (eager reconstruct)")
        except Exception as exc:  # noqa: BLE001
            print(f"  [sdpa] lm prefill skipped: {exc}")

    _lm_handles = [m.register_forward_hook(_lm_attn_hook, with_kwargs=True)
                   for m in model.modules()
                   if type(m).__name__ == "SlidingWindowLlamaAttention"]

    torch.manual_seed(SEED)
    px = (torch.rand(1, 3, 1024, 1024) - 0.5) / 0.5
    print("running SAM encoder (CPU) to trigger conv hooks...")
    sam_out = sam(px)                                   # [1,1024,16,16]
    for h in handles:
        h.remove()
    for h in _relpos_handles:
        h.remove()

    print("saving per-conv fixtures...")
    for nm, m in convs.items():
        cap = captured[nm]
        _save(
            f"conv_{nm}",
            {
                "state_dict": {k: v.detach().float().cpu() for k, v in m.state_dict().items()},
                "in_channels": m.in_channels, "out_channels": m.out_channels,
                "kernel_size": m.kernel_size, "stride": m.stride, "padding": m.padding,
                "bias": m.bias is not None,
                "input_nchw": cap["input_nchw"],     # feed (permuted to NHWC) to the TTNN conv
                "output_nchw": cap["output_nchw"],   # PCC reference (permute to NHWC to compare)
            },
        )

    # Encoder-level fixture (isolates the 12-block + neck/net compounding) and the
    # downstream CLIP / DeepEncoder references, all from the SAME px.
    print("running CLIP + projector (CPU)...")
    vit_out = vis(px, sam_out)                          # [1,257,1024]
    feats = torch.cat([vit_out[:, 1:], sam_out.flatten(2).permute(0, 2, 1)], dim=-1)  # [1,256,2048]
    proj_out = proj(feats)                              # [1,256,1280]
    _save("vision_pipeline", {
        "px": px,
        "sam_out": sam_out.detach().float().cpu(),      # SAM encoder ref
        "vit_out": vit_out.detach().float().cpu(),      # CLIP encoder ref
        "feats": feats.detach().float().cpu(),
        "proj_out": proj_out.detach().float().cpu(),    # DeepEncoder+projector ref
    })

    # LM linears: a short text-only forward triggers the DeepSeek decoder linears
    # (layer-0 dense MLP gate_proj + attention q_proj) captured by the hooks above.
    if not ({"lm_gate", "lm_qproj"} <= set(_lin_cap)):
        print("running text-only LM forward (CPU) to trigger LM linear hooks...")
        cfg = getattr(model, "config", None)
        vocab = int(getattr(cfg, "vocab_size", 129280))
        torch.manual_seed(SEED)
        input_ids = torch.randint(0, vocab, (1, 8), dtype=torch.int64)
        try:
            model(input_ids=input_ids, images=None, use_cache=False, return_dict=True)
        except Exception as exc:  # noqa: BLE001 -- fall back to the bare LM stack
            print(f"  full-model text forward failed ({exc}); trying inner LM stack...")
            lm = inner if hasattr(inner, "layers") else getattr(inner, "model", inner)
            emb = inner.embed_tokens(input_ids)
            lm(inputs_embeds=emb, use_cache=False, return_dict=True)

    for h in _lin_handles:
        h.remove()
    for h in _moe_handles:
        h.remove()
    for h in _lm_handles:
        h.remove()
    _F.scaled_dot_product_attention = _orig_sdpa

    print("saving linear op-sweep fixtures...")
    for key in _WANTED:
        if key not in _lin_cap:
            print(f"  WARNING: linear fixture '{key}' not captured (shape not seen)")
            continue
        _save(f"linear_{key}", _lin_cap[key])

    print("saving SDPA op-sweep fixtures...")
    for site in ("sdpa_clip", "sdpa_sam_window", "sdpa_sam_global", "sdpa_lm_prefill"):
        if site not in _sdpa_cap:
            print(f"  WARNING: SDPA fixture '{site}' not captured")
            continue
        _save(site, _sdpa_cap[site])

    print("saving rel-pos matmul fixtures...")
    for tag, payload in _relpos_cap.items():
        _save(tag, payload)
    if not _relpos_cap:
        print("  WARNING: no rel-pos matmul fixtures captured")

    print("saving MoE-gate matmul fixture...")
    if "moe_gate" in _moe_cap:
        _save("moe_gate", _moe_cap["moe_gate"])
    else:
        print("  WARNING: moe_gate fixture not captured")

    print("DONE. fixtures in", FIX)
    return 0


if __name__ == "__main__":
    sys.exit(main())
