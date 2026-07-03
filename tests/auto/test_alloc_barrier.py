# SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Software-only self-test for the TracedRun allocation-safety seam (deep-plan_0 §9.6).

Validates the reusable barrier primitive without hardware: before_device_allocation
returns 0 on empty cache, releases+returns N on non-empty and leaves _warmup_keys intact,
device_allocation_barrier empties the cache around an alloc and raises under
_strict_alloc_guard, and _trace_lock is re-entrant.

tt-metal commit pinned: c09f09c35a1a59a428f0e1b5cdaa8fe59fb1b195
"""
import sys
import types
from unittest import mock

import pytest


@pytest.fixture
def traced_run(monkeypatch):
    """Import TracedRun with ttnn.release_trace stubbed so no hardware is touched."""
    import ttnn

    released = []
    monkeypatch.setattr(ttnn, "release_trace", lambda dev, tid: released.append(tid), raising=False)
    from tt_symbiote.core.run_config import TracedRun

    # snapshot + reset class state
    orig_cache = TracedRun._trace_cache
    orig_warm = TracedRun._warmup_keys
    orig_strict = TracedRun._strict_alloc_guard
    orig_count = TracedRun._barrier_release_count
    TracedRun._trace_cache = {}
    TracedRun._warmup_keys = set()
    TracedRun._strict_alloc_guard = False
    TracedRun._barrier_release_count = 0
    yield TracedRun, released
    TracedRun._trace_cache = orig_cache
    TracedRun._warmup_keys = orig_warm
    TracedRun._strict_alloc_guard = orig_strict
    TracedRun._barrier_release_count = orig_count


def _fake_entry():
    e = types.SimpleNamespace()
    e.device = "dev"
    e.trace_id = object()
    return e


def test_has_active_captures(traced_run):
    TR, _ = traced_run
    assert TR.has_active_captures() is False
    TR._trace_cache[("m", "k")] = _fake_entry()
    assert TR.has_active_captures() is True


def test_before_device_allocation_empty(traced_run):
    TR, released = traced_run
    assert TR.before_device_allocation("x") == 0
    assert released == []
    assert TR._barrier_release_count == 0


def test_before_device_allocation_releases_and_preserves_warmup(traced_run):
    TR, released = traced_run
    TR._warmup_keys = {("m", "k1"), ("m", "k2")}
    TR._trace_cache[("m", "k1")] = _fake_entry()
    TR._trace_cache[("m", "k2")] = _fake_entry()
    n = TR.before_device_allocation("cold")
    assert n == 2
    assert len(released) == 2
    assert TR._trace_cache == {}
    # _warmup_keys MUST be untouched (released keys re-CAPTURE, not re-warm)
    assert TR._warmup_keys == {("m", "k1"), ("m", "k2")}
    assert TR._barrier_release_count == 1


def test_invalidate_alias_delegates(traced_run):
    TR, released = traced_run
    TR._trace_cache[("m", "k")] = _fake_entry()
    TR._warmup_keys = {("m", "k")}
    assert TR.invalidate_captures_for_cold_compile() == 1
    assert TR._trace_cache == {}
    assert TR._warmup_keys == {("m", "k")}


def test_barrier_empties_cache_around_alloc(traced_run):
    TR, released = traced_run
    TR._trace_cache[("m", "k")] = _fake_entry()
    with TR.device_allocation_barrier("req"):
        assert TR._trace_cache == {}  # released before the alloc body
    assert len(released) == 1


def test_barrier_raises_under_strict(traced_run):
    TR, _ = traced_run
    TR._strict_alloc_guard = True
    TR._trace_cache[("m", "k")] = _fake_entry()
    with pytest.raises(RuntimeError, match="active captured trace"):
        with TR.device_allocation_barrier("req"):
            pass


def test_barrier_strict_noop_when_no_captures(traced_run):
    TR, _ = traced_run
    TR._strict_alloc_guard = True
    # no captures -> no raise
    with TR.device_allocation_barrier("req"):
        pass


def test_trace_lock_reentrant(traced_run):
    TR, _ = traced_run
    with TR._trace_lock:
        with TR._trace_lock:  # re-entrant acquire must not deadlock
            assert True
