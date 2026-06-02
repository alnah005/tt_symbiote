# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Experimental / unregistered model packages.

This subpackage is the source-side companion to ``tests/experimental/``.
It holds modeling code that survives from earlier phases but whose
recipe is NOT in
:data:`tt_symbiote.models._RECIPE_BEARING_SUBPACKAGES`. Modules under
``tt_symbiote._experimental`` are:

- **Not eagerly imported.** ``tt_symbiote.models.__init__`` does not
  touch them, so importing the package never triggers
  ``@register_recipe`` for any experimental model. Users have to import
  the module by its full ``tt_symbiote._experimental.<name>`` path
  themselves.
- **Excluded from the published sdist / wheel.** See the
  ``[tool.setuptools.packages.find]`` ``exclude`` entry in
  ``pyproject.toml``. ``pip install tt_symbiote`` will NOT ship this
  subtree.
- **No API stability.** Files here may move, rename, or disappear at any
  time without a deprecation cycle.

To promote a module out of ``_experimental``, follow the "Resurrecting
a port" recipe in ``tests/experimental/README.md``.
"""
