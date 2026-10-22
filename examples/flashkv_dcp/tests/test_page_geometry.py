"""Exact SGLang source-method replay with actual CPU tensor memory.

Allocator/kernel construction is omitted; _storage_page_index and
get_page_buffer_meta are extracted unchanged from the patched MLA host source.
"""

import ast
import types
import pytest
import torch
from cpu_harness import *

SOURCE = SG / "srt/mem_cache/pool_host/mla.py"
tree = ast.parse(SOURCE.read_text())
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
exec(compile(ast.Module(body=methods, type_ignores=[]), str(SOURCE), "exec"), namespace)


def actual_geometry_pool(cfg, layout, reader=False):
    p = pool(cfg, layout=layout)
    p.page_num = 6
    p.size = p.page_num * p.page_size
    p.logical_size = p.page_num * p.logical_page_size
    if layout == "page_first":
        shape = (p.size, p.layer_num, 1, p.kv_cache_dim)
    else:
        shape = (p.page_num, p.layer_num, p.page_size, 1, p.kv_cache_dim)
    count = 1
    for dim in shape:
        count *= dim
    p.kv_buffer = (
        torch.full((count,), -1, dtype=p.dtype)
        if reader
        else torch.arange(count, dtype=p.dtype)
    ).reshape(shape)
    p._buffers = [p.kv_buffer]
    for name, fn in namespace.items():
        if callable(fn) and name in ("_storage_page_index", "get_page_buffer_meta"):
            setattr(p, name, types.MethodType(fn, p))
    return p


@pytest.mark.parametrize("layout", ["page_first", "page_first_direct"])
@pytest.mark.parametrize("dcp,rank", [(1, 0), (1, 1), (8, 0), (8, 7)])
@pytest.mark.parametrize("name", [PoolName.KV, PoolName.DRAFT])
def test_nonadjacent_logical_pages_use_exact_source_mapping(layout, dcp, rank, name):
    Client.directory.clear()
    cfg = config(dcp=dcp, rank=rank)
    writer_pool = actual_geometry_pool(cfg, layout)
    reader_pool = actual_geometry_pool(cfg, layout, reader=True)
    a, _ = store(cfg, **{str(name): writer_pool})
    b, _ = store(cfg, **{str(name): reader_pool})
    indices = torch.cat((torch.arange(128, 256), torch.arange(384, 512)))
    ptrs, sizes = writer_pool.get_page_buffer_meta(indices)
    rows_per_page = 128 // dcp
    expected_bytes = (
        rows_per_page
        * writer_pool.layer_num
        * writer_pool.kv_cache_dim
        * writer_pool.kv_buffer.element_size()
    )
    assert sizes == [expected_bytes, expected_bytes]
    assert [ptr - writer_pool.kv_buffer.data_ptr() for ptr in ptrs] == [
        expected_bytes,
        3 * expected_bytes,
    ]
    keys = ["nonadjacent-page-1", "nonadjacent-page-3"]
    if name == PoolName.KV:
        assert a.batch_set_v1(keys, indices) == [True, True]
        assert b.batch_get_v1(keys, indices) == [True, True]
    else:
        tr = PoolTransfer(name=name, keys=keys, host_indices=indices)
        assert a.batch_set_v2([tr]) == {name: [True, True]}
        assert b.batch_get_v2([tr]) == {name: [True, True]}
    # Check full buffer including untouched pages; validates byte size and offset,
    # not just key count or success masks.
    expected = torch.full_like(reader_pool.kv_buffer, -1).reshape(-1)
    width = expected_bytes // writer_pool.kv_buffer.element_size()
    source = writer_pool.kv_buffer.reshape(-1)
    for page in (1, 3):
        expected[page * width : (page + 1) * width] = source[
            page * width : (page + 1) * width
        ]
    assert torch.equal(reader_pool.kv_buffer.reshape(-1), expected)
    assert all(
        len(Client.directory[a.key_namespace, key]) == expected_bytes
        for key in object_keys(a, name, keys)
    )


@pytest.mark.parametrize("start", [-128, 1, 129, 6 * 128])
def test_exact_dcp_geometry_rejects_bad_logical_page_starts(start):
    cfg = config()
    p = actual_geometry_pool(cfg, "page_first_direct")
    with pytest.raises((ValueError, IndexError)):
        p.get_page_buffer_meta(torch.arange(start, start + 128))
