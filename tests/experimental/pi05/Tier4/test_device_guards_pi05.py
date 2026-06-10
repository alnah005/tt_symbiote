# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Software-only guard tests for the pi0.5 port (no hardware / no weights).

Verifies the hard conventions: every TTNN forward/device method carries a
``@run_on_devices`` guard (stamped as ``__tt_allowed_archs__``), and the package
imports cleanly. Runnable in CI without a device.
"""

from __future__ import annotations

import inspect

import pytest

DEVICE_METHODS = {
    "modeling_pi05_gemma": [
        ("TTNNPi05GemmaMLP", ["forward"]),
        ("TTNNPi05GemmaAttention", ["forward"]),
        ("TTNNPi05GemmaBlock", ["forward"]),
        ("TTNNPi05AdaRMSGemmaBlock", ["forward"]),
    ],
    "modeling_pi05_siglip": [
        ("TTNNPi05SigLIPVisionTower", ["forward"]),
        ("TTNNPi05MultiModalProjector", ["forward"]),
    ],
    "modeling_pi05_suffix": [("TTNNPi05SuffixEmbedding", ["embed_actions", "embed_adarms_cond", "project_output"])],
    "modeling_pi05_paligemma": [
        ("TTNNPi05PaliGemmaBackbone", ["embed_image", "embed_language_tokens", "forward_vlm", "forward_expert"])
    ],
    "modeling_pi05": [("TTNNPi05Model", ["forward", "sample_actions"])],
}


def test_package_imports():
    import tt_symbiote.models.pi05  # noqa: F401
    from tt_symbiote.models.pi05 import TTNNPi05Model  # noqa: F401


@pytest.mark.parametrize("module_name", list(DEVICE_METHODS.keys()))
def test_run_on_devices_present(module_name):
    import importlib

    mod = importlib.import_module(f"tt_symbiote.models.pi05.{module_name}")
    for cls_name, methods in DEVICE_METHODS[module_name]:
        cls = getattr(mod, cls_name)
        for m in methods:
            fn = getattr(cls, m)
            assert hasattr(fn, "__tt_allowed_archs__"), f"{cls_name}.{m} missing @run_on_devices guard"
            from tt_symbiote.core.module import DeviceArch

            assert DeviceArch.P150 in fn.__tt_allowed_archs__, f"{cls_name}.{m} not guarded for P150"


def test_no_torch_in_device_forwards():
    """AST check: no ``torch.*`` attribute access inside @run_on_devices methods."""
    import ast
    import importlib.util
    import pathlib

    pkg_dir = pathlib.Path(importlib.util.find_spec("tt_symbiote.models.pi05").submodule_search_locations[0])
    offenders = []
    for path in pkg_dir.glob("modeling_pi05*.py"):
        tree = ast.parse(path.read_text())
        for cls in [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]:
            for fn in [n for n in cls.body if isinstance(n, ast.FunctionDef)]:
                if not any("run_on_devices" in ast.dump(d) for d in fn.decorator_list):
                    continue
                for sub in ast.walk(fn):
                    if isinstance(sub, ast.Attribute) and isinstance(sub.value, ast.Name) and sub.value.id == "torch":
                        offenders.append(f"{path.name}:{cls.name}.{fn.name} -> torch.{sub.attr}")
    assert not offenders, "torch.* in device forward bodies: " + "; ".join(offenders)
