"""Independent observable contract tests, not native RDMA/GPU validation."""

import ast
import copy
import types
from unittest.mock import patch
import pytest
import torch
from cpu_harness import *


@pytest.fixture(autouse=True)
def reset_transport():
    Client.directory.clear()
    Client.instances.clear()


@pytest.mark.parametrize("tp,dcp", [(8, 8), (4, 2), (8, 1)])
def test_shard_identity_equivalence_classes(tp, dcp):
    stores = [store(config(rank=r, tp=tp, dcp=dcp))[0] for r in range(tp)]
    for a in range(tp):
        for b in range(tp):
            for name in (PoolName.KV, PoolName.DRAFT):
                assert (identity(stores[a], name) == identity(stores[b], name)) == (
                    a % dcp == b % dcp
                )
            assert (
                identity(stores[a], PoolName.MAMBA)
                == identity(stores[b], PoolName.MAMBA)
            ) == (a == b)


@pytest.mark.parametrize(
    "field,value",
    [
        ("model_revision", "different-target"),
        ("draft_revision", "different-draft"),
        ("namespace", "different-job"),
    ],
)
def test_revision_and_namespace_isolation(field, value):
    a, _ = store()
    cfg = config()
    cfg.extra_config["flashkv_dcp"][field] = value
    b, _ = store(cfg)
    for name in (PoolName.KV, PoolName.DRAFT, PoolName.MAMBA):
        assert identity(a, name) != identity(b, name)


def test_pool_model_dtype_layout_isolation():
    a, _ = store()
    cfg = config(model_name="different-model")
    b, _ = store(cfg)
    assert identity(a, PoolName.KV) != identity(b, PoolName.KV)
    names = [identity(a, n) for n in (PoolName.KV, PoolName.MAMBA, PoolName.DRAFT)]
    assert len(set(names)) == 3
    for overrides in (
        {"dtype": torch.float16},
        {"layout": "page_first"},
        {"layer_num": 3},
        {"kv_cache_dim": 16},
    ):
        cfg = config()
        c, _ = store(cfg, kv=pool(cfg, **overrides))
        assert identity(a, PoolName.KV) != identity(c, PoolName.KV)


def test_topology_and_pp_isolation():
    a, _ = store(config(tp=8, dcp=8))
    b, _ = store(config(tp=4, dcp=2))
    c, _ = store(config(pp_size=2, pp_rank=1))
    assert len({identity(s, PoolName.KV) for s in (a, b, c)}) == 3


@pytest.mark.parametrize("missing", ["model_revision", "schema_version"])
def test_bad_contract_rejected_before_native_constructor(missing):
    cfg = config()
    del cfg.extra_config["flashkv_dcp"][missing]
    with pytest.raises(ValueError):
        store(cfg)
    assert not Client.instances


@pytest.mark.parametrize(
    "change",
    [
        {"layout": "layer_first"},
        {"logical_page_size": 64},
        {"dcp_rank": 1},
        {"dtype": torch.float32},
        {"kv_scale_buffer": object()},
        {"kv_buffer": None},
    ],
)
def test_invalid_pool_rejected_before_native_constructor(change):
    cfg = config()
    bad = pool(cfg, **change)
    with pytest.raises(ValueError):
        store(cfg, kv=bad)
    assert not Client.instances


def test_late_invalid_registration_has_no_native_effect():
    s, _ = store()
    calls = list(s.flashkv_client.calls)
    before = s.key_namespace
    cfg = config()
    bad = pool(cfg, layout="layer_first")
    with pytest.raises(ValueError):
        s.register_mem_host_pool_v2(bad, PoolName.DRAFT)
    assert s.flashkv_client.calls == calls
    assert s.key_namespace == before


def test_namespace_stable_on_registration_and_independent_instances():
    cfg = config()
    with patch("torch.cuda.current_device", return_value=0):
        a = FlashKVDCPStorage(cfg)
    ns = a.key_namespace
    a.register_mem_pool_host(pool(cfg))
    assert a.key_namespace == ns
    a.register_mem_host_pool_v2(pool(cfg), PoolName.DRAFT)
    assert a.key_namespace == ns
    b, _ = store(cfg)
    assert b.key_namespace == ns


@pytest.mark.parametrize("name", [PoolName.KV, PoolName.DRAFT, PoolName.MAMBA])
def test_cpu_payload_roundtrip_and_component_failure(name):
    a, pa = store()
    b, pb = store()
    # Fresh reader buffers, same registered page format/topology.
    for buf in pb[name]._buffers:
        buf.fill_(-1)
    size = getattr(pa[name], "logical_page_size", pa[name].page_size)
    indices = torch.arange(2 * size)
    if name == PoolName.KV:
        assert a.batch_set_v1(["p1", "p2"], indices) == [True, True]
        assert b.batch_get_v1(["p1", "p2"], indices) == [True, True]
    else:
        transfer = PoolTransfer(name=name, keys=["p1", "p2"], host_indices=indices)
        assert a.batch_set_v2([transfer])[name] == [True, True]
        assert b.batch_get_v2([transfer])[name] == [True, True]
    for x, y in zip(pa[name]._buffers, pb[name]._buffers):
        assert torch.equal(x[:2], y[:2])
    # One missing component makes only its page fail.
    del Client.directory[a.key_namespace, object_keys(a, name, ["p2"])[-1]]
    got = (
        b.batch_get_v1(["p1", "p2"], indices)
        if name == PoolName.KV
        else b.batch_get_v2([transfer])[name]
    )
    assert got == [True, False]


@pytest.mark.parametrize("name", [PoolName.KV, PoolName.DRAFT])
@pytest.mark.parametrize("count", [16, 127, 129, 255])
def test_exact_logical_index_count_required(name, count):
    s, _ = store()
    calls = list(s.flashkv_client.calls)
    with pytest.raises(ValueError):
        if name == PoolName.KV:
            s.batch_set_v1(["p1"], torch.arange(count))
        else:
            s.batch_set_v2(
                [PoolTransfer(name=name, keys=["p1"], host_indices=torch.arange(count))]
            )
    assert s.flashkv_client.calls == calls


@pytest.mark.parametrize(
    "malformation", ["missing_pointer", "extra_size", "zero_pointer", "zero_size"]
)
def test_buffer_shape_rejection_prevents_zip_truncation(malformation):
    s, pools = store()
    p = pools[PoolName.DRAFT]
    bad = {
        "missing_pointer": ([], []),
        "extra_size": ([1], [1, 2]),
        "zero_pointer": ([0], [4]),
        "zero_size": ([1], [0]),
    }
    p.get_page_buffer_meta = lambda indices: bad[malformation]
    calls = list(s.flashkv_client.calls)
    with pytest.raises(ValueError):
        s.batch_set_v2(
            [
                PoolTransfer(
                    name=PoolName.DRAFT, keys=["p1"], host_indices=torch.arange(128)
                )
            ]
        )
    assert s.flashkv_client.calls == calls


def test_short_native_result_fails_closed():
    s, _ = store()
    s._batch_get_impl = lambda *args: [True]
    assert s.batch_get_v1(["p1", "p2"], torch.arange(256)) == [False, False]


def populate_sparse(
    s, checkpoints, draft_pages=(1, 2, 3, 4), target_pages=(1, 2, 3, 4)
):
    present(s, PoolName.KV, [f"p{i}" for i in target_pages])
    present(s, PoolName.MAMBA, [f"p{i}" for i in checkpoints])
    present(s, PoolName.DRAFT, [f"p{i}" for i in draft_pages])
    transfers = [
        PoolTransfer(
            name=PoolName.MAMBA,
            keys=["checkpoint"],
            hit_policy=PoolHitPolicy.TRAILING_PAGES,
        ),
        PoolTransfer(name=PoolName.DRAFT, hit_policy=PoolHitPolicy.ALL_PAGES),
    ]
    return s.batch_exists_v2(["p1", "p2", "p3", "p4"], transfers)


def test_sparse_checkpoint_intersection_across_ranks():
    a, _ = store(config(rank=0))
    b, _ = store(config(rank=1))
    ra = populate_sparse(a, {2, 4})
    rb = populate_sparse(b, {2, 3})
    assert set(ra.restorable_prefix_pages) == {2, 4}
    assert set(rb.restorable_prefix_pages) == {2, 3}
    assert max(set(ra.restorable_prefix_pages) & set(rb.restorable_prefix_pages)) == 2
    # Exact unmodified #41187 sync function with CPU Torch tensors and modeled MIN collective.
    source = SG / "srt/mem_cache/hybrid_cache/hybrid_cache_controller.py"
    tree = ast.parse(source.read_text())
    cl = next(
        x
        for x in tree.body
        if isinstance(x, ast.ClassDef) and x.name == "HybridCacheController"
    )
    fn = next(
        x
        for x in cl.body
        if isinstance(x, ast.FunctionDef) and x.name == "_sync_prefetch_hit_query"
    )
    scope = {
        "torch": torch,
        "PoolHitPolicy": PoolHitPolicy,
        "PoolName": PoolName,
        "_NO_POOL_VERDICT": -1,
    }
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(source), "exec"), scope)
    peer = torch.tensor(
        [1, 0, 1, 1, 0] + [rb.extra_pool_hit_pages.get(p, -1) for p in PoolName],
        dtype=torch.int,
    )

    def reduce(t, *args):
        t.copy_(torch.minimum(t, peer))

    controller = types.SimpleNamespace(
        page_size=128, prefetch_hits_sync_groups=[], _all_reduce=reduce
    )
    operation = types.SimpleNamespace(
        pool_transfers=[
            PoolTransfer(name=PoolName.MAMBA, hit_policy=PoolHitPolicy.TRAILING_PAGES)
        ],
        token_ids=list(range(512)),
        pool_storage_result=ra,
        query_pool_hit_pages=ra.extra_pool_hit_pages.copy(),
    )
    assert scope["_sync_prefetch_hit_query"](controller, operation, 4 * 128) == 2 * 128


@pytest.mark.parametrize(
    "checkpoints,draft,target,expected",
    [
        ({2, 4}, (1, 2, 4), (1, 2, 3, 4), {2}),
        ({2, 4}, (1, 3, 4), (1, 2, 3, 4), set()),
        ({2, 4}, (1, 2, 3, 4), (1, 2, 4), {2}),
        ({4}, (1, 2, 3, 4), (1, 2, 4), set()),
        (set(), (1, 2, 3, 4), (1, 2, 3, 4), set()),
    ],
)
def test_target_and_sidepool_holes(checkpoints, draft, target, expected):
    s, _ = store()
    r = populate_sparse(s, checkpoints, draft, target)
    assert set(r.restorable_prefix_pages) == expected
    assert r.kv_hit_pages == max(expected, default=0)


def test_missing_kda_component_is_miss():
    s, _ = store()
    populate_sparse(s, {2})
    del Client.directory[s.key_namespace, object_keys(s, PoolName.MAMBA, ["p2"])[-1]]
    r = s.batch_exists_v2(
        ["p1", "p2"],
        [
            PoolTransfer(
                name=PoolName.MAMBA, keys=["x"], hit_policy=PoolHitPolicy.TRAILING_PAGES
            )
        ],
    )
    assert r.kv_hit_pages == 0 and not r.restorable_prefix_pages


@pytest.mark.parametrize("namespace", ["", "explicit-test-space"])
def test_destructive_clear_always_disabled(namespace):
    cfg = config()
    cfg.extra_config["flashkv_dcp"]["namespace"] = namespace
    s, _ = store(cfg)
    calls = list(s.flashkv_client.calls)
    with pytest.raises(NotImplementedError):
        s.clear()
    assert s.flashkv_client.calls == calls


factory_module = load(
    "sglang.srt.mem_cache.storage.backend_factory",
    SG / "srt/mem_cache/storage/backend_factory.py",
)
Factory = factory_module.StorageBackendFactory


def dynamic_config(**kwargs):
    cfg = config(**kwargs)
    cfg.extra_config.update(
        backend_name="flashkv-dcp",
        module_path="apodex_flashkv.storage",
        class_name="FlashKVDCPStorage",
    )
    return cfg


@pytest.mark.parametrize(
    "caps",
    [
        frozenset(),
        True,
        {"mla_sharded"},
        frozenset({"mla_sharded", "logical_page_indices"}),
    ],
)
def test_factory_rejects_missing_or_malformed_capabilities_before_client(caps):
    cfg = dynamic_config()
    pools = {
        PoolName.KV: pool(cfg),
        PoolName.MAMBA: pool(cfg, "mamba"),
        PoolName.DRAFT: pool(cfg),
    }

    class Candidate(FlashKVDCPStorage):
        dcp_storage_capabilities = caps

    with patch.object(Factory, "_load_backend_class", return_value=Candidate):
        with pytest.raises((NotImplementedError, TypeError)):
            Factory.create_backend(
                "dynamic", cfg, pools[PoolName.KV], storage_host_pools=pools
            )
    assert not Client.instances


def test_extra_config_boolean_cannot_enable_unadapted_vendor():
    cfg = dynamic_config()
    cfg.extra_config.update(supports_dcp=True, dcp_storage_capabilities=True)
    with patch.object(
        Factory, "_load_backend_class", return_value=vendor.FlashKVStorage
    ):
        with pytest.raises(NotImplementedError):
            Factory.create_backend("dynamic", cfg, pool(cfg))
    assert not Client.instances


def test_factory_invokes_plugin_geometry_validation_before_constructor():
    cfg = dynamic_config()
    pools = {
        PoolName.KV: pool(cfg),
        PoolName.MAMBA: pool(cfg, "mamba", temporal_dtype=torch.int8),
    }
    with pytest.raises(ValueError):
        Factory.create_backend(
            "dynamic", cfg, pools[PoolName.KV], storage_host_pools=pools
        )
    assert not Client.instances


def test_factory_allows_adapted_dynamic_and_preserves_options():
    cfg = dynamic_config()
    pools = {
        PoolName.KV: pool(cfg),
        PoolName.MAMBA: pool(cfg, "mamba"),
        PoolName.DRAFT: pool(cfg),
    }
    with patch("torch.cuda.current_device", return_value=0):
        s = Factory.create_backend(
            "dynamic", cfg, pools[PoolName.KV], storage_host_pools=pools
        )
    assert isinstance(s, FlashKVDCPStorage)
    assert len(Client.instances) == 1
    assert s._options["model_revision"] == "target-sha-a"


@pytest.mark.parametrize("name", [PoolName.KV, PoolName.DRAFT])
def test_factory_denies_bad_actual_pool_geometry(name):
    cfg = dynamic_config()
    pools = {PoolName.KV: pool(cfg), PoolName.DRAFT: pool(cfg)}
    pools[name].page_size = 128
    with pytest.raises(NotImplementedError):
        Factory.create_backend(
            "dynamic", cfg, pools[PoolName.KV], storage_host_pools=pools
        )
    assert not Client.instances


def test_dcp1_factory_checks_opted_in_mamba_schema():
    cfg = dynamic_config(dcp=1)
    pools = {
        PoolName.KV: pool(cfg),
        PoolName.MAMBA: pool(cfg, "mamba", temporal_dtype=torch.int8),
    }
    with pytest.raises(ValueError):
        Factory.create_backend(
            "dynamic", cfg, pools[PoolName.KV], storage_host_pools=pools
        )
    assert not Client.instances


@pytest.mark.parametrize("dcp", [1, 8])
def test_other_tp_rank_cannot_claim_kda_from_rank_zero(dcp):
    a, _ = store(config(rank=0, dcp=dcp))
    b, _ = store(config(rank=1, dcp=dcp))
    present(a, PoolName.KV, ["p1"])
    present(a, PoolName.MAMBA, ["p1"])
    if dcp == 8:
        present(b, PoolName.KV, ["p1"])
    r = b.batch_exists_v2(
        ["p1"],
        [
            PoolTransfer(
                name=PoolName.MAMBA,
                keys=["p1"],
                hit_policy=PoolHitPolicy.TRAILING_PAGES,
            )
        ],
    )
    assert r.kv_hit_pages == 0


def test_all_small_sparse_directories_against_independent_recovery_oracle():
    # Exhaust all 4-page KDA checkpoint and draft-presence subsets. Recovery
    # requires every target/draft page through boundary and state at boundary.
    s, _ = store()
    keys = ["p1", "p2", "p3", "p4"]
    for kda_mask in range(16):
        for draft_mask in range(16):
            Client.directory.clear()
            kda = {i + 1 for i in range(4) if kda_mask & 1 << i}
            draft = {i + 1 for i in range(4) if draft_mask & 1 << i}
            expected = {
                p for p in range(1, 5) if p in kda and set(range(1, p + 1)) <= draft
            }
            r = populate_sparse(s, kda, draft)
            assert set(r.restorable_prefix_pages) == expected, (kda, draft)
            assert r.kv_hit_pages == max(expected, default=0)


def test_empty_batches_never_issue_native_io():
    s, _ = store()
    calls = list(s.flashkv_client.calls)
    assert s.batch_set_v1([], torch.tensor([], dtype=torch.int64)) == []
    assert s.batch_get_v2(
        [
            PoolTransfer(
                name=PoolName.DRAFT,
                keys=[],
                host_indices=torch.tensor([], dtype=torch.int64),
            )
        ]
    ) == {PoolName.DRAFT: []}
    assert s.flashkv_client.calls == calls


def test_overlapping_page_text_does_not_alias_components():
    s, _ = store()
    all_keys = []
    for name in (PoolName.KV, PoolName.MAMBA, PoolName.DRAFT):
        all_keys.extend(
            object_keys(s, name, ["a", "a:b", "b", "kv:1:a", "temporal:1:a"])
        )
    assert len(all_keys) == len(set(all_keys))


@pytest.mark.parametrize("native_failure", [False, True])
def test_exact_controller_late_sidecar_rejects_before_group_or_native_mutation(
    native_failure,
):
    # Only this test and sparse-sync test extract a controller method; full
    # controller construction requires runtime allocator/CUDA dependencies.
    from queue import Queue

    source = SG / "srt/mem_cache/hybrid_cache/hybrid_cache_controller.py"
    tree = ast.parse(source.read_text())
    cl = next(
        x
        for x in tree.body
        if isinstance(x, ast.ClassDef) and x.name == "HybridCacheController"
    )
    fn = next(
        x
        for x in cl.body
        if isinstance(x, ast.FunctionDef) and x.name == "register_host_pool_entry"
    )
    scope = {"HostPoolGroup": HostPoolGroup, "PoolEntry": object, "Queue": Queue}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(source), "exec"), scope)
    storage_package = package(
        "sglang.srt.mem_cache.storage", SG / "srt/mem_cache/storage"
    )
    storage_package.StorageBackendFactory = Factory
    cfg = dynamic_config()
    s, pools = store(cfg)
    events = []
    group = HostPoolGroup()
    group.entry_map = {PoolName.KV: object(), PoolName.MAMBA: object()}
    group.add_entry = lambda entry: events.append(entry.name)
    controller = types.SimpleNamespace(
        mem_pool_host=group,
        enable_storage=True,
        storage_backend=s,
        storage_backend_type="dynamic",
        storage_config=cfg,
        _storage_host_pools=lambda: {
            PoolName.KV: pools[PoolName.KV],
            PoolName.MAMBA: pools[PoolName.MAMBA],
        },
        extra_host_mem_release_queues={},
    )
    entry = types.SimpleNamespace(
        name=PoolName.DRAFT,
        host_pool=pool(cfg, layout="layer_first"),
        is_primary_index_anchor=False,
    )
    before = list(s.flashkv_client.calls)
    with pytest.raises((ValueError, NotImplementedError)):
        scope["register_host_pool_entry"](controller, entry)
    assert events == [] and s.flashkv_client.calls == before
    assert controller.extra_host_mem_release_queues == {}
    entry.host_pool = pool(cfg)
    if native_failure:
        register = s.flashkv_client.register_memory
        s.flashkv_client.register_memory = lambda buffers: Status(9)
        with pytest.raises(RuntimeError):
            scope["register_host_pool_entry"](controller, entry)
        assert events == [] and controller.extra_host_mem_release_queues == {}
        s.flashkv_client.register_memory = register
    scope["register_host_pool_entry"](controller, entry)
    assert events == [PoolName.DRAFT]
    assert PoolName.DRAFT in controller.extra_host_mem_release_queues


@pytest.mark.parametrize("value", [None, 0, 2, "1"])
def test_interface_v1_required_before_native_initialization(value):
    cfg = config()
    cfg.extra_config["interface_v1"] = value
    with pytest.raises(ValueError):
        store(cfg)
    assert not Client.instances


def test_registration_failure_restores_python_pool_identity():
    s, pools = store()
    before_prefix = dict(s._pool_prefixes)
    before_pools = dict(s.registered_pools)
    before_anchor = s.mem_pool_host

    def fail_registration(buffers):
        return Status(9)

    s.flashkv_client.register_memory = fail_registration
    new_draft = pool(config())
    with pytest.raises(RuntimeError):
        s.register_mem_host_pool_v2(new_draft, PoolName.DRAFT)
    assert s._pool_prefixes == before_prefix
    assert s.registered_pools == before_pools
    with pytest.raises(RuntimeError):
        s.register_mem_pool_host(pool(config()))
    assert s._pool_prefixes == before_prefix and s.mem_pool_host is before_anchor


def test_vendor_negative_controls_reproduce_original_contract_defects():
    def original(rank, dcp=8):
        cfg = config(rank=rank, dcp=dcp)
        with (
            patch.dict(
                "os.environ", {"FLASHKV_KEY_NAMESPACE": "isolated-vendor-control"}
            ),
            patch("torch.cuda.current_device", return_value=0),
        ):
            s = vendor.FlashKVStorage(cfg)
        s.register_mem_pool_host(pool(cfg))
        s.register_mem_host_pool_v2(pool(cfg, "mamba"), PoolName.MAMBA)
        s.register_mem_host_pool_v2(pool(cfg), PoolName.DRAFT)
        return s

    a = original(0)
    b = original(1)
    assert identity(a, PoolName.KV) == identity(b, PoolName.KV)
    assert identity(a, PoolName.DRAFT) == identity(b, PoolName.DRAFT)
    assert identity(original(0, 1), PoolName.MAMBA) == identity(
        original(1, 1), PoolName.MAMBA
    )
    with pytest.raises(AssertionError):
        a.batch_set_v2(
            [
                PoolTransfer(
                    name=PoolName.DRAFT, keys=["p1"], host_indices=torch.arange(128)
                )
            ]
        )


def test_legacy_best_effort_draft_mode_rejected():
    s, _ = store()
    with pytest.raises((ValueError, NotImplementedError)):
        s.set_draft(True)
    assert not s.has_draft


def test_draft_partial_prefix_policy_rejected():
    s, _ = store()
    present(s, PoolName.KV, ["p1", "p2"])
    with pytest.raises(ValueError):
        s.batch_exists_v2(
            ["p1", "p2"],
            [
                PoolTransfer(
                    name=PoolName.DRAFT,
                    keys=["p2"],
                    hit_policy=PoolHitPolicy.TRAILING_PAGES,
                )
            ],
        )


def test_duplicate_write_transfer_rejected_before_any_native_write():
    s, _ = store()
    before = list(s.flashkv_client.calls)
    tr = PoolTransfer(name=PoolName.DRAFT, keys=["p1"], host_indices=torch.arange(128))
    with pytest.raises(ValueError):
        s.batch_set_v2([tr, tr])
    assert s.flashkv_client.calls == before


def test_mha_draft_dcp1_not_admitted_without_writer_ownership_support():
    cfg = config(dcp=1)
    with pytest.raises(ValueError):
        store(cfg, draft=pool(cfg, "mha"))
    assert not Client.instances
