# tt_symbiote documentation

User-facing guides:

- [`install_prerequisites.md`](install_prerequisites.md) — install the
  `(ttnn, sfpi)` runtime that `pip install tt_symbiote` depends on.
- [`supported_models.md`](supported_models.md) — catalog of recipes
  that ship with hardware-verified status.
- [`ling_mini_2_0_guide.md`](ling_mini_2_0_guide.md) — end-to-end
  walkthrough of the causal-LM recipe and the `Auto*` factory.
- [`dropped_models.md`](dropped_models.md) — recipes that were
  intentionally *not* included in 0.1.0, and why.

Maintainer and release documentation lives under
[`development/`](development/) and is excluded from the published
sdist via `MANIFEST.in`.
