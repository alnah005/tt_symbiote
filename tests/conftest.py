# Empty top-level pytest conftest.
#
# Per-suite fixtures live in:
#   - tests/auto/conftest.py        (recipe/auto-class tests; ttnn deferred)
#   - tests/capabilities/conftest.py (TTNN capability/PCC tests, when present)
#
# Keep this file present so pytest treats `tests/` as the rootdir.
