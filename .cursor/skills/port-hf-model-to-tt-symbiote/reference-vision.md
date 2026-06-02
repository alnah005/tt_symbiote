# Vision (image classification) port specifics

Reference material for `port-hf-model-to-tt-symbiote` when the target HF
model maps to `AutoModelForImageClassification`. Read this in addition
to [`SKILL.md`](SKILL.md).

The canonical vision port is
[`microsoft/resnet-50`](../../src/tt_symbiote/models/resnet/) (Phase 6).
Like the LLM reference it is a **full TTNN** port — the CPU-first
default this skill produces is much simpler.

## CPU-first variant

```
src/tt_symbiote/models/<NAME>/
├── __init__.py
├── configuration_<NAME>.py
└── modeling_<NAME>.py          # @register_recipe(hf_class_name="<NAME>ForImageClassification")
                                #   build_module_dict -> {}
                                #   post_register -> device patch + runtime_config
```

`<HF_CLASS>` ends in `ForImageClassification`. `<AUTO_CLASS>` is
`AutoModelForImageClassification`.

## Image processor surface

Vision demos use `AutoImageProcessor` directly:

```python
from PIL import Image
from transformers import AutoImageProcessor

processor = AutoImageProcessor.from_pretrained(MODEL_ID)
image = Image.open("tests/images/<some_image>.png").convert("RGB")
inputs = processor(image, return_tensors="pt")
pixel_values = inputs["pixel_values"].to(torch.bfloat16)
```

Forward + decode:

```python
outputs = model(pixel_values=pixel_values)
predicted = int(outputs.logits.argmax(dim=-1).item())
label = model.config.id2label.get(predicted, f"<class {predicted}>")
```

For synthetic inputs (offline-runnable demo) pass random tensors of the
right shape — they yield arbitrary but valid logits and a top-1 class.

## Semantic check

Top-1 on a real image should be plausible. For ResNet-50 on the
canonical COCO val cat image
(`images.cocodataset.org/val2017/000000039769.jpg`), top-1 is
`'tiger cat'` (top-2: `'tabby, tabby cat'`).

There is no project-canonical vision test image; pick a known image
relevant to the model's training distribution.

## Mesh shape per variant

Vision backbones are usually small and fit a single chip:

| Backbone family | Typical mesh |
|---|---|
| ResNet, ConvNeXt, MobileNet | `(1, 1)` |
| ViT (Base / Large) | `(1, 1)` |
| ViT-Huge / Giant | `(1, 2)` or `(1, 4)` |

## `l1_small_size` for TTNN ports

If the model has a large initial conv (e.g. ResNet's 7x7 stride-2 stem),
the TTNN sliding-window halo metadata needs L1 small region. Default
`l1_small_size=0` triggers an OOM at first forward; use `245760` for
ResNet-style stems. For CPU-first ports this knob is harmless — keep it
at the default `245760` for symmetry with the full TTNN port path.

## Class enumeration

Image classification models follow:

```
<Model>ForImageClassification
├── <model>: <Model>Model      # the backbone
│   ├── embedder: <Model>Embeddings / <Model>PatchEmbed
│   ├── encoder: <Model>Encoder (stages / blocks / layers)
│   └── pooler / layernorm
└── classifier: nn.Linear   (sometimes nn.Sequential(Flatten, Linear))
```

Every backbone class goes into `cpu_fallback`. The output dataclasses
(`<Model>ModelOutput`, `<Model>ImageClassifierOutput`) go into
`out_of_scope`.

## Out-of-scope on the demo path

- Object detection / segmentation alternative heads (`<Model>ForObjectDetection`, `<Model>ForSemanticSegmentation`).
- Backbone-only top-level (`<Model>Model`) if the demo uses the classification head — the backbone runs as a child of the classification head, so the backbone classes stay in `cpu_fallback`.

## Acceptance gate

A vision CPU-first port is done when:

1. `from_pretrained` + `set_device` + `model(pixel_values=...)` succeeds.
2. `outputs.logits.shape == (1, model.config.num_labels)` and finite.
3. Top-1 on a real image is plausible (manual eyeball).
4. `compatibility.report(model)["runtime_observed"]["unexpected"]` is empty.
