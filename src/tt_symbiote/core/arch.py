# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: (C) 2023 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

# Vendored from tt-metal commit 43d758d972e7f3f3610236723ea22dfdbcecbc85 on 2026-05-27.
# Resync manually if upstream materially changes.
#
# Sources:
#   - is_blackhole, is_wormhole_b0:  models/common/utility_functions.py:1029-1037
#   - determine_device_name:         models/tt_transformers/tt/model_config.py:4194-4237
#
# All three are tiny wrappers around ttnn.get_arch_name() / mesh_device APIs.
# Vendoring rather than importing from tt_transformers because tt_transformers
# is not a pip-installable package; it lives only as a source tree inside the
# tt-metal monorepo.

import ttnn


def is_blackhole() -> bool:
    arch_name = ttnn.get_arch_name()
    return "blackhole" in arch_name


def is_wormhole_b0() -> bool:
    arch_name = ttnn.get_arch_name()
    return "wormhole_b0" in arch_name


def determine_device_name(mesh_device) -> str:
    """Determine device name based on number of devices and architecture.

    Args:
        mesh_device (MeshDevice): MeshDevice object (or None for CPU).

    Returns:
        str: Device name (e.g., "CPU", "N150", "P100", etc.).

    Raises:
        ValueError: If architecture or device count is unsupported.
    """
    num_devices = mesh_device.get_num_devices() if mesh_device else 0
    arch_name = ttnn.get_arch_name()
    dram_grid_size = mesh_device.dram_grid_size() if mesh_device else None

    if num_devices == 0:
        return "CPU"

    if is_blackhole():
        dict_device_names = {
            1: "P100" if dram_grid_size and dram_grid_size.x == 7 else "P150",
            2: "P300",
            4: "P150x4",
            8: "P150x8",
            32: "BHGLX",
        }
    elif is_wormhole_b0():
        dict_device_names = {
            1: "N150",
            2: "N300",
            4: "N150x4",
            8: "T3K",
            32: "TG",
        }
    else:
        raise ValueError(f"Unsupported architecture: {arch_name}")

    if num_devices in dict_device_names:
        return dict_device_names[num_devices]
    raise ValueError(f"Unsupported number of devices: {num_devices} for {arch_name}")
