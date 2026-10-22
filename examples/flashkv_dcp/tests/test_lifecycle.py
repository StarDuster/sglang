"""Memory ownership protocol tests with actual adapter and fake native client."""

import ast
import logging
import threading
import time
import types
from unittest.mock import patch
import pytest
import torch
from cpu_harness import *


def unregisters(client):
    return [c for c in client.calls if c[0] == "unregister"]


def test_registration_dedup_and_close_exactly_once():
    s, p = store()
    before = list(s.flashkv_client.calls)
    spans = dict(s._registered_buffers)
    for tensor in spans.values():
        s._register_buffer(tensor)
    assert s.flashkv_client.calls == before
    s.close()
    assert s._lifecycle_state == "closed" and s._registered_buffers == {}
    assert len(unregisters(s.flashkv_client)) == len(spans)
    assert set(pair for c in unregisters(s.flashkv_client) for pair in c[1]) == set(
        spans
    )
    before = list(s.flashkv_client.calls)
    s.close()
    assert s.flashkv_client.calls == before


@pytest.mark.parametrize(
    "operation",
    [
        lambda s: s._batch_query_impl([]),
        lambda s: s._batch_get_impl([], [], []),
        lambda s: s._batch_put_impl([], [], []),
        lambda s: s.batch_get_v1([], torch.tensor([])),
        lambda s: s.batch_set_v1([], torch.tensor([])),
        lambda s: s.batch_exists_v2([]),
        lambda s: s.batch_get_v2([]),
        lambda s: s.batch_set_v2([]),
        lambda s: s._register_buffer(torch.ones(1)),
    ],
)
def test_closed_state_rejects_io_even_empty(operation):
    s, _ = store()
    s.close()
    before = list(s.flashkv_client.calls)
    with pytest.raises(RuntimeError):
        operation(s)
    assert s.flashkv_client.calls == before


@pytest.mark.parametrize("failure", ["status", "exception"])
def test_failed_unregister_retains_tensor_and_retries_only_failed_span(failure):
    s, _ = store()
    native = s.flashkv_client.unregister_memory
    victim = next(iter(s._registered_buffers))
    tensor = s._registered_buffers[victim]
    attempts = []

    def fail_one(buffers):
        ident = (buffers[0].ptr, buffers[0].size)
        attempts.append(ident)
        if ident == victim:
            if failure == "exception":
                raise RuntimeError("synthetic transport error")
            return Status(9)
        return native(buffers)

    s.flashkv_client.unregister_memory = fail_one
    with pytest.raises(RuntimeError):
        s.close()
    assert s._lifecycle_state == "closing" and s._registered_buffers == {victim: tensor}
    with pytest.raises(RuntimeError):
        s.batch_exists([])
    s.flashkv_client.unregister_memory = native
    before = len(unregisters(s.flashkv_client))
    s.close()
    assert len(unregisters(s.flashkv_client)) == before + 1
    assert s._registered_buffers == {} and s._lifecycle_state == "closed"


@pytest.mark.parametrize("rollback_fails", [False, True])
def test_partial_registration_rollback_owns_new_buffers_only(rollback_fails):
    s, _ = store()
    existing = dict(s._registered_buffers)
    oldpools = dict(s.registered_pools)
    new = pool(config(), "mamba")
    reg = s.flashkv_client.register_memory
    unreg = s.flashkv_client.unregister_memory
    count = 0

    def fail_second(buffers):
        nonlocal count
        count += 1
        return Status(9) if count == 2 else reg(buffers)

    s.flashkv_client.register_memory = fail_second
    if rollback_fails:
        s.flashkv_client.unregister_memory = lambda buffers: Status(9)
    with pytest.raises(RuntimeError):
        s.register_mem_host_pool_v2(new, PoolName.MAMBA)
    assert s.registered_pools == oldpools
    assert all(s._registered_buffers[k] is v for k, v in existing.items())
    if rollback_fails:
        assert s._lifecycle_state == "closing"
        leftover = set(s._registered_buffers) - set(existing)
        assert (
            len(leftover) == 1
            and s._registered_buffers[next(iter(leftover))] is new.temporal_buffer
        )
    else:
        assert (
            set(s._registered_buffers) == set(existing) and s._lifecycle_state == "open"
        )
    s.flashkv_client.unregister_memory = unreg
    s.close()
    assert not s._registered_buffers


def test_close_waits_for_inflight_operation_then_blocks_new_io():
    s, _ = store()
    entered = threading.Event()
    release = threading.Event()
    finished = threading.Event()

    def inflight():
        with s._operation():
            entered.set()
            assert release.wait(3)

    worker = threading.Thread(target=inflight)
    worker.start()
    assert entered.wait(2)
    closer = threading.Thread(target=lambda: (s.close(), finished.set()))
    closer.start()
    deadline = time.monotonic() + 2
    while s._lifecycle_state == "open" and time.monotonic() < deadline:
        time.sleep(0.005)
    assert s._lifecycle_state == "closing" and not finished.is_set()
    with pytest.raises(RuntimeError):
        s._batch_query_impl([])
    release.set()
    worker.join(2)
    closer.join(2)
    assert finished.is_set() and s._lifecycle_state == "closed"


def test_constructor_failure_closes_ownership_without_inventing_client_stop():
    s = FlashKVDCPStorage.__new__(FlashKVDCPStorage)
    with (
        patch.object(Client, "start", return_value=Status(9)),
        patch("torch.cuda.current_device", return_value=0),
    ):
        with pytest.raises(RuntimeError):
            FlashKVDCPStorage.__init__(s, config())
    assert s._lifecycle_state == "closed" and not s._registered_buffers


def test_exact_controller_detach_retains_failed_backend_for_retry():
    source = SG / "srt/managers/cache_controller.py"
    tree = ast.parse(source.read_text())
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "HiCacheController"
    )
    fn = next(
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "detach_storage_backend"
    )
    scope = {"logger": logging.getLogger("lifecycle-test")}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(source), "exec"), scope)
    s, _ = store()
    stop = threading.Event()
    stop.set()
    native = s.flashkv_client.unregister_memory
    s.flashkv_client.unregister_memory = lambda buffers: Status(9)
    c = types.SimpleNamespace(
        _stop_storage_threads=lambda: None,
        _destroy_sync_groups=lambda groups: None,
        prefetch_hits_sync_groups=[],
        prefetch_completion_sync_groups=[],
        storage_backend=s,
        storage_backend_type="dynamic",
        enable_storage=True,
        _generic_page_get=object(),
        _generic_page_set=object(),
        storage_stop_event=stop,
    )
    with pytest.raises(RuntimeError):
        scope["detach_storage_backend"](c)
    assert c.storage_backend is s and not c.enable_storage and stop.is_set()
    assert s._registered_buffers
    s.flashkv_client.unregister_memory = native
    scope["detach_storage_backend"](c)
    assert c.storage_backend is None and not stop.is_set() and not s._registered_buffers
