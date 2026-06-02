# Dropped models / tests (Phase 2 migration)

The following items existed in
`tt-metal/models/experimental/tt_symbiote/` on the
`alnah005/tt_symbiote_v2` branch (commit `43d758d`) but were **not migrated**
into the new repo. Each entry records why and when, so a future Claude porting
skill (see [`docs/development/PROJECT_PROPOSAL.md`](../docs/development/PROJECT_PROPOSAL.md) §7.3) can revive them
if appropriate.

| Item                       | Reason for drop                                              | Dropped at | Could be revived? |
|----------------------------|--------------------------------------------------------------|------------|-------------------|
| `tests/test_yunet.py`      | YuNet is not a Hugging Face Hub model; out of scope for an HF-mirrored library. | Phase 2.6  | No — would need a new home if revived. |
| `tests/test_deepseek_ocr.py` | `dots.ocr` reached a hardware limit per the team call; cannot make further progress on current HW. | Phase 2.6  | Yes — re-port via the porting skill if HW story changes. |
| `tests/test_training.py`   | Training is explicitly out of scope for the `v0.1.0` release.  | Phase 2.6  | Yes — separate workstream when training is in scope. |
| `vllm/` directory          | Empty in `tt_symbiote_v2` (only contained `__pycache__`).      | Phase 2.6  | N/A — there was nothing to migrate. |

Tests **kept** but currently un-collectable (due to unresolved cross-repo imports
from `tt-metal/models/demos/`) are listed separately in
[`tests/SKIP_DURING_BOOTSTRAP.md`](../tests/SKIP_DURING_BOOTSTRAP.md). Those tests
are not dropped — their model is still in scope; they need the corresponding
helper to be vendored or re-implemented during the model's port.
