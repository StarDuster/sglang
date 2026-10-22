"""Observable adapter contract on CPU: object identity per shard, page I/O
through the real SGLang DCP page geometry, sparse checkpoint reporting, and
host-memory cleanup retry. Native transport and GPU paths are out of scope."""

import ast
import types
from unittest.mock import Mock

import pytest
import torch
from cpu_harness import *  # noqa: F401,F403


@pytest.fixture(autouse=True)
def reset_transport():
    Client.directory.clear()
    Client.instances.clear()


@pytest.mark.parametrize("tp,dcp", [(8, 8), (4, 2), (8, 1)])
def test_shard_identity_equivalence_classes(tp, dcp):
    """MLA KV and draft objects are shared by ranks holding the same DCP shard;
    KDA state is private to each TP rank."""
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


def _other(label):
    cfg = config()
    if label in ("model_revision", "draft_revision", "namespace"):
        cfg.extra_config["flashkv_dcp"][label] = "other"
        return store(cfg)[0]
    if label == "model_name":
        return store(config(model_name="other"))[0]
    if label == "pp_rank":
        return store(config(pp_size=2, pp_rank=1))[0]
    if label == "dcp_size":
        return store(config(tp=4, dcp=2))[0]
    if label == "layout":
        return store(
            cfg, kv=pool(cfg, layout="page_first"), draft=pool(cfg, layout="page_first")
        )[0]
    return store(
        cfg,
        kv=pool(
            cfg,
            **{
                label: {"dtype": torch.float16, "layer_num": 3, "kv_cache_dim": 16}[
                    label
                ]
            },
        ),
    )[0]


@pytest.mark.parametrize(
    "label",
    [
        "model_revision",
        "draft_revision",
        "namespace",
        "model_name",
        "pp_rank",
        "dcp_size",
        "dtype",
        "layout",
        "layer_num",
        "kv_cache_dim",
    ],
)
def test_every_identity_input_separates_objects(label):
    assert identity(store()[0], PoolName.KV) != identity(_other(label), PoolName.KV)


def _reject(label, cfg, host_pools):
    if label == "no model_revision":
        del cfg.extra_config["flashkv_dcp"]["model_revision"]
    elif label == "no interface_v1":
        del cfg.extra_config["interface_v1"]
    elif label == "draft without draft_revision":
        del cfg.extra_config["flashkv_dcp"]["draft_revision"]
    elif label == "mha target":
        cfg.is_mla_model = False
    elif label == "attention cp":
        cfg.attn_cp_size = 2
    elif label == "kv layer_first":
        host_pools[PoolName.KV].layout = "layer_first"
    elif label == "swa sidecar":
        host_pools[PoolName.SWA] = pool(cfg)
    elif label == "mamba slot side states":
        host_pools[PoolName.MAMBA].slot_state_device_tensors = [torch.empty(1)]


@pytest.mark.parametrize(
    "label",
    [
        "no model_revision",
        "no interface_v1",
        "draft without draft_revision",
        "mha target",
        "attention cp",
        "kv layer_first",
        "swa sidecar",
        "mamba slot side states",
    ],
)
def test_rejected_before_the_native_client_starts(label):
    cfg = config()
    host_pools = pools(cfg)
    _reject(label, cfg, host_pools)
    with pytest.raises((ValueError, NotImplementedError)):
        Factory.create_backend(
            "dynamic", cfg, host_pools[PoolName.KV], storage_host_pools=host_pools
        )
    assert not Client.instances


@pytest.mark.parametrize("name", [PoolName.KV, PoolName.DRAFT, PoolName.MAMBA])
def test_page_roundtrip_and_component_miss(name):
    a, pa = store()
    b, pb = store()
    for buf in pb[name]._buffers:
        buf.fill_(-1)
    size = LOGICAL_PAGE if name != PoolName.MAMBA else 1
    indices = torch.arange(2 * size)
    transfer = PoolTransfer(name=name, keys=["p1", "p2"], host_indices=indices)
    if name == PoolName.KV:
        assert a.batch_set_v1(["p1", "p2"], indices) == [True, True]
        assert b.batch_get_v1(["p1", "p2"], indices) == [True, True]
    else:
        assert a.batch_set_v2([transfer])[name] == [True, True]
        assert b.batch_get_v2([transfer])[name] == [True, True]
    for x, y in zip(pa[name]._buffers, pb[name]._buffers):
        assert torch.equal(x[:2], y[:2])
    # Losing one component object fails only that page.
    del Client.directory[a.key_namespace, object_keys(a, name, ["p2"])[-1]]
    got = (
        b.batch_get_v1(["p1", "p2"], indices)
        if name == PoolName.KV
        else b.batch_get_v2([transfer])[name]
    )
    assert got == [True, False]


def _real_geometry_pool(cfg, layout, reader=False):
    """Harness pool with the checkout's MLA page index math and page_num rows."""
    source = SG / "srt/mem_cache/pool_host/mla.py"
    tree = ast.parse(source.read_text())
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "MLATokenToKVPoolHost"
    )
    methods = [
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef)
        and n.name in ("_storage_page_index", "get_page_buffer_meta")
    ]
    namespace = {"torch": torch}
    exec(
        compile(ast.Module(body=methods, type_ignores=[]), str(source), "exec"),
        namespace,
    )
    p = pool(cfg, layout=layout)
    p.page_num = 6
    p.size = p.page_num * p.page_size
    p.logical_size = p.page_num * p.logical_page_size
    shape = (
        (p.size, p.layer_num, 1, p.kv_cache_dim)
        if layout == "page_first"
        else (p.page_num, p.layer_num, p.page_size, 1, p.kv_cache_dim)
    )
    count = torch.Size(shape).numel()
    p.kv_buffer = (
        torch.full((count,), -1, dtype=p.dtype)
        if reader
        else torch.arange(count, dtype=p.dtype)
    ).reshape(shape)
    p._buffers = [p.kv_buffer]
    for name in ("_storage_page_index", "get_page_buffer_meta"):
        setattr(p, name, types.MethodType(namespace[name], p))
    return p


@pytest.mark.parametrize("layout", ["page_first", "page_first_direct"])
@pytest.mark.parametrize("dcp,rank", [(1, 0), (8, 7)])
@pytest.mark.parametrize("name", [PoolName.KV, PoolName.DRAFT])
def test_logical_pages_land_on_this_ranks_physical_rows(layout, dcp, rank, name):
    cfg = config(dcp=dcp, rank=rank)
    writer, reader = (
        _real_geometry_pool(cfg, layout),
        _real_geometry_pool(cfg, layout, reader=True),
    )
    other = PoolName.DRAFT if name == PoolName.KV else PoolName.KV
    a, _ = store(cfg, **{str(name): writer, str(other): pool(cfg, layout=layout)})
    b, _ = store(cfg, **{str(name): reader, str(other): pool(cfg, layout=layout)})
    indices = torch.cat((torch.arange(128, 256), torch.arange(384, 512)))
    keys = ["page-1", "page-3"]
    if name == PoolName.KV:
        assert a.batch_set_v1(keys, indices) == [True, True]
        assert b.batch_get_v1(keys, indices) == [True, True]
    else:
        tr = PoolTransfer(name=name, keys=keys, host_indices=indices)
        assert a.batch_set_v2([tr]) == {name: [True, True]}
        assert b.batch_get_v2([tr]) == {name: [True, True]}
    # Only physical pages 1 and 3 of this rank's shard change, byte for byte.
    width = (128 // dcp) * writer.layer_num * writer.kv_cache_dim
    expected = torch.full_like(reader.kv_buffer, -1).reshape(-1)
    source = writer.kv_buffer.reshape(-1)
    for page in (1, 3):
        expected[page * width : (page + 1) * width] = source[
            page * width : (page + 1) * width
        ]
    assert torch.equal(reader.kv_buffer.reshape(-1), expected)


@pytest.mark.parametrize("is_set", [False, True])
@pytest.mark.parametrize("key_count", [1, 3])
def test_v2_rejects_logical_page_count_mismatch(is_set, key_count, monkeypatch):
    cfg = config(dcp=8, rank=7)
    draft = _real_geometry_pool(cfg, "page_first_direct")
    s, _ = store(cfg, draft=draft)
    native_io = Mock()
    monkeypatch.setattr(
        s.flashkv_client, "batch_put" if is_set else "batch_get", native_io
    )
    transfer = PoolTransfer(
        name=PoolName.DRAFT,
        keys=[f"page-{i}" for i in range(key_count)],
        host_indices=torch.arange(2 * LOGICAL_PAGE),
    )
    with pytest.raises(ValueError, match="logical host page count"):
        (s.batch_set_v2 if is_set else s.batch_get_v2)([transfer])
    native_io.assert_not_called()


@pytest.mark.parametrize("is_set", [False, True])
@pytest.mark.parametrize("shorten", ["pointers", "sizes"])
def test_v2_rejects_component_cardinality_mismatch(is_set, shorten, monkeypatch):
    cfg = config(dcp=8, rank=7)
    draft = _real_geometry_pool(cfg, "page_first_direct")
    s, _ = store(cfg, draft=draft)
    indices = torch.arange(2 * LOGICAL_PAGE)
    ptrs, sizes = draft.get_page_buffer_meta(indices)
    monkeypatch.setattr(
        draft,
        "get_page_buffer_meta",
        lambda _: (ptrs[:-1], sizes) if shorten == "pointers" else (ptrs, sizes[:-1]),
    )
    native_io = Mock()
    monkeypatch.setattr(
        s.flashkv_client, "batch_put" if is_set else "batch_get", native_io
    )
    transfer = PoolTransfer(
        name=PoolName.DRAFT, keys=["page-0", "page-1"], host_indices=indices
    )
    with pytest.raises(ValueError, match="keys, pointers and byte sizes"):
        (s.batch_set_v2 if is_set else s.batch_get_v2)([transfer])
    native_io.assert_not_called()


def _sparse(s, checkpoints, draft=(1, 2, 3, 4), target=(1, 2, 3, 4)):
    present(s, PoolName.KV, [f"p{i}" for i in target])
    present(s, PoolName.MAMBA, [f"p{i}" for i in checkpoints])
    present(s, PoolName.DRAFT, [f"p{i}" for i in draft])
    transfers = [
        PoolTransfer(
            name=PoolName.MAMBA,
            keys=["checkpoint"],
            hit_policy=PoolHitPolicy.TRAILING_PAGES,
        ),
        PoolTransfer(name=PoolName.DRAFT, hit_policy=PoolHitPolicy.ALL_PAGES),
    ]
    return s.batch_exists_v2(["p1", "p2", "p3", "p4"], transfers)


@pytest.mark.parametrize(
    "checkpoints,draft,target,expected",
    [
        ({2, 4}, (1, 2, 3, 4), (1, 2, 3, 4), {2, 4}),
        ({2, 4}, (1, 2, 4), (1, 2, 3, 4), {2}),
        ({2, 4}, (1, 3, 4), (1, 2, 3, 4), set()),
        ({2, 4}, (1, 2, 3, 4), (1, 2, 4), {2}),
        ({4}, (1, 2, 3, 4), (1, 2, 4), set()),
        (set(), (1, 2, 3, 4), (1, 2, 3, 4), set()),
    ],
)
def test_restorable_prefixes_are_reported_as_a_set(
    checkpoints, draft, target, expected
):
    """A KDA checkpoint can exist at 4 but not 3; the controller intersects the
    per-rank sets, so the result lists every restorable length, not the max."""
    r = _sparse(store()[0], checkpoints, draft, target)
    assert set(r.restorable_prefix_pages) == expected
    assert r.kv_hit_pages == max(expected, default=0)


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


def _unregisters(client):
    return [
        span for call in client.calls if call[0] == "unregister" for span in call[1]
    ]


@pytest.mark.parametrize("failure", ["status", "exception"])
def test_close_retries_from_the_first_failed_span(failure):
    s, _ = store()
    spans = list(s._registered_spans)
    native = s.flashkv_client.unregister_memory
    victim = spans[1]

    def fail_victim(buffers):
        if (buffers[0].ptr, buffers[0].size) == victim:
            if failure == "exception":
                raise RuntimeError("transport")
            return Status(9)
        return native(buffers)

    s.flashkv_client.unregister_memory = fail_victim
    with pytest.raises(RuntimeError):
        s.close()
    assert s._registered_spans == spans[:2]
    s.flashkv_client.unregister_memory = native
    s.close()
    assert s._registered_spans == []
    # Every span is released exactly once; the retry does not touch released ones.
    assert sorted(_unregisters(s.flashkv_client)) == sorted(spans)
