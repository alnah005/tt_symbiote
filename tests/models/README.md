# `tests/models/` — RICH per-model tests (e2e-traced-correct)

This is the **rich** test tree: models here are expected to be end-to-end
correct and trace-validated, with the full tiered PCC test layout. It is the
promotion target for models that graduate out of `tests/experimental/`.

**It is currently empty.** All in-flight models live under
[`tests/experimental/`](../experimental/) for now; a model is promoted here once
its Tier1–4 PCC tests pass on hardware.

## Layout (per model)

```
tests/models/<name>/
  __init__.py
  test_config.json          # tt_metal_commit, device_arch, pcc_threshold, hf_model_id, hf_revision
  shapes.json  op_map.json  # optional artifacts (kept at the model root)
  Tier1/  test_ops_<name>.py            # leaf ops
  Tier2/  test_composites_<name>.py     # attention / mlp / moe / norm
  Tier3/  test_decoder_<name>.py        # decoder / block layer
  Tier4/  test_modeling_<name>.py       # full model + e2e + trace + semantic
```

Each `Tier{1..4}/` is a real package (carries an `__init__.py`). The structure
is enforced by `scripts/check_tier_structure.py` (the `tier-structure`
pre-commit hook). See [`CLAUDE.md`](../../CLAUDE.md) for the full conventions.
