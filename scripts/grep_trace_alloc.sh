#!/usr/bin/env bash
# SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
# deep-plan_0 §9.2: per-restart trace/allocator sniff + §6 thrash measurement.
# Usage: grep_trace_alloc.sh <server.log>
LOG="${1:?usage: grep_trace_alloc.sh <server.log>}"
echo "=== $LOG ==="
echo -n "allocator.cpp:105 (AUTHORITATIVE, must be 0): "; grep -c "allocator.cpp:105" "$LOG"
echo -n "active trace                               : "; grep -c "active trace" "$LOG"
echo -n "released N trace (barrier releases)        : "; grep -cE "released [0-9]+ trace" "$LOG"
echo -n "before_device_allocation calls             : "; grep -c "before_device_allocation" "$LOG"
echo -n "Capturing Trace / Capturing trace          : "; grep -ciE "Capturing [Tt]race" "$LOG"
echo -n "TRACE-ALLOC-GUARD warnings                 : "; grep -c "TRACE-ALLOC-GUARD" "$LOG"
echo -n "POST /v1 requests                          : "; grep -c "POST /v1" "$LOG"
