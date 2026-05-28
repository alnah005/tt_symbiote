# ResNet end-to-end demos

One script per canonical Microsoft ResNet variant. All five share the
single [`ResNetRecipe`](../../../src/tt_symbiote/models/resnet/modeling_resnet.py)
— variants differ only in the HF model id; the recipe's
``build_module_dict`` selects between
:class:`TTNNResNetBasicLayer` (ResNet-18/34) and
:class:`TTNNResNetBottleNeckLayer` (ResNet-50/101/152) based on
``config.layer_type``.

Unlike the VLM demos, this recipe is a **full TTNN port**: the
embedder, conv layers, shortcuts, and the basic / bottleneck blocks
all run on Tenstorrent silicon. The final
``nn.AdaptiveAvgPool2d`` + ``nn.Linear`` classifier head still runs in
HF's host fallback path (it's after the NHWC→NCHW boundary).

## Variant matrix

| Script | HF model id | Block type | Mesh | HW verified | Notes |
|---|---|---|---|---|---|
| [`run_resnet18.py`](run_resnet18.py) | `microsoft/resnet-18` | basic | `(1, 1)` (N150) | ⏳ | Structurally supported |
| [`run_resnet34.py`](run_resnet34.py) | `microsoft/resnet-34` | basic | `(1, 1)` (N150) | ⏳ | Structurally supported |
| [`run_resnet50.py`](run_resnet50.py) | `microsoft/resnet-50` | bottleneck | `(1, 1)` (N150) | ✅ | Phase 6 reference reproducer |
| [`run_resnet101.py`](run_resnet101.py) | `microsoft/resnet-101` | bottleneck | `(1, 1)` (N150) | ⏳ | Structurally supported |
| [`run_resnet152.py`](run_resnet152.py) | `microsoft/resnet-152` | bottleneck | `(1, 1)` (N150) | ⏳ | Structurally supported |

`hw_verified` flag flips to `True` in
[`RESNET_TTNN_TUNING`](../../../src/tt_symbiote/models/resnet/configuration_resnet.py)
once a script produces a plausible top-1 ImageNet label on the listed
hardware target.

## Shared shape

Every script in this folder is the same ~70-line template:

1. `ttnn.set_fabric_config(DISABLED)` + `ttnn.open_mesh_device((1,1), l1_small_size=245760)`
2. `AutoModelForImageClassification.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)`
3. `set_device(model, mesh_device)` (this actually moves modules to device — full TTNN port)
4. Synthetic 1×3×224×224 BF16 input (swap to a real image via `AutoImageProcessor` for semantic verification — see `run_resnet50.py` "synthetic input" comment block).
5. `model(pixel_values=...)` forward
6. `compatibility.report(model)` → printed and written to `<script>_coverage.json`
7. `ttnn.close_mesh_device(mesh_device)`

The per-script `_coverage.json` artefacts aggregate into
[`docs/cpu_vs_device_coverage.md`](../../../docs/cpu_vs_device_coverage.md).
Note that the ResNet recipe predates the Phase 7 compatibility-list
convention, so its design-time `tt_implemented` / `cpu_fallback` /
`out_of_scope` lists are currently empty — the runtime ledger is the
authoritative source for the ResNet CPU/device split until those lists
are populated (tracked as a follow-up in `docs/migration_notes.md`).

## Reproducing a clean run

```bash
source /home/<you>/tt_symbiote/.venv/bin/activate
python examples/e2e/resnet/run_resnet50.py
```

Expected wall-clock on a warm cache for ResNet-50: a few seconds
(synthetic input + a single forward).
