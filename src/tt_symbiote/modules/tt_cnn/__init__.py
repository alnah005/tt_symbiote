# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Trimmed vendor of tt-metal's ``models/tt_cnn/tt/`` package.

Source: ``models/tt_cnn/tt/`` in ``tenstorrent/tt-metal`` (vendored at
commit ``43d758d972e7f3f3610236723ea22dfdbcecbc85`` on 2026-05-27).

Why vendored
------------
``tt_cnn`` lives inside tt-metal's ``models/`` source tree. ``models/``
is not part of the ``ttnn`` PyPI wheel, so ``pip install ttnn`` does
not give us these convenience wrappers — we have to ship our own copy
to keep ``pip install tt_symbiote`` self-contained.

What's included
---------------
Only [`builder.py`](builder.py) is vendored — it provides
``Conv2dConfiguration``, ``MaxPool2dConfiguration``, ``TtConv2d``,
``TtMaxPool2d`` (the four symbols
:mod:`tt_symbiote.modules.ttnn_conv` consumes).

What's intentionally absent
---------------------------
Upstream ``tt_cnn`` also ships ``executor.py`` and ``pipeline.py``
(``Executor`` / ``PipelineConfig`` / ``create_pipeline_from_config`` /
the ``MultiCQ*Executor`` family). They are not used anywhere in
``tt_symbiote`` and were dropped from the vendor to keep the wheel
lean. If a future port needs them, copy the matching upstream files
back into this directory and update the resync banner at the top of
each file.
"""
