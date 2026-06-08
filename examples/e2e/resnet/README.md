# ResNet end-to-end demos

One script per canonical Microsoft ResNet variant, all sharing the single
[`ResNetRecipe`](../../../src/tt_symbiote/models/resnet/modeling_resnet.py);
`build_module_dict` selects `TTNNResNetBasicLayer` (ResNet-18/34) vs
`TTNNResNetBottleNeckLayer` (ResNet-50/101/152) by `config.layer_type`. It is a
**full TTNN port**: embedder, conv layers, shortcuts, and the basic / bottleneck
blocks all run on device; only the final `nn.AdaptiveAvgPool2d` + `nn.Linear`
classifier head runs in HF's host fallback (after the NHWC→NCHW boundary), so a
clean run reports `regressions == []`.

## Variant matrix

| Script | HF model id | Block type | Mesh | HW verified | Notes |
|---|---|---|---|---|---|
| [`run_resnet18.py`](run_resnet18.py) | `microsoft/resnet-18` | basic | `(1, 1)` (N150) | ⏳ | structurally supported |
| [`run_resnet34.py`](run_resnet34.py) | `microsoft/resnet-34` | basic | `(1, 1)` (N150) | ⏳ | structurally supported |
| [`run_resnet50.py`](run_resnet50.py) | `microsoft/resnet-50` | bottleneck | `(1, 1)` (N150) | ✅ | reference reproducer |
| [`run_resnet101.py`](run_resnet101.py) | `microsoft/resnet-101` | bottleneck | `(1, 1)` (N150) | ⏳ | structurally supported |
| [`run_resnet152.py`](run_resnet152.py) | `microsoft/resnet-152` | bottleneck | `(1, 1)` (N150) | ⏳ | structurally supported |

`hw_verified` flips to `True` in
[`RESNET_TTNN_TUNING`](../../../src/tt_symbiote/models/resnet/configuration_resnet.py)
once a script produces a plausible top-1 ImageNet label on its hardware target.

## ResNet run shape (image classification)

Open mesh (`ttnn.open_mesh_device((1,1), l1_small_size=245760)`) →
`AutoModelForImageClassification.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)`
→ `set_device(model, mesh)` → synthetic
`pixel_values = torch.randn(1, 3, 224, 224, dtype=torch.bfloat16)` (swap in
`AutoImageProcessor` on a real image for a semantic check) →
`model(pixel_values=...)` → `logits.argmax(-1)` top-1 ImageNet label →
`compatibility.report(model)` written to the gitignored `<script>_coverage.json`
(regenerated per run) → `ttnn.close_mesh_device(mesh)`. Aggregated summary:
[`docs/development/cpu_vs_device_coverage.md`](../../../docs/development/cpu_vs_device_coverage.md).

## Reproduce

```bash
source .venv/bin/activate
python examples/e2e/resnet/run_resnet50.py
```
