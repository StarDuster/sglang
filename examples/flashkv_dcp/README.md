# FlashKV DCP adapter

A thin plugin over the PAI FlashKV Python integration (`flashkv-dist:1.0.0-df62dbba`,
wheel `1.0.0+gdf62dbb.cuda.cu13`) for SGLang's DCP-aware HiCache L3 contract. It
does not contain the vendor source, wheel, native library, weights or credentials.

The vendor keys MLA objects per PP rank only, so under DCP every rank's 1/dcp
KV shard collides, and it keys KDA state without the TP rank although that state
is TP-sharded. The adapter scopes a FlashKV namespace by deployment identity
(model name and revisions, TP/PP/DCP sizes, PP rank, user namespace) and prefixes
each object with a per-pool descriptor digest that carries the shard rank: the DCP
rank for MLA KV and draft pages, the TP rank for KDA state. It also reports sparse
KDA checkpoints as a set so the controller can intersect them across ranks, and
unregisters its host memory on detach with a retry that resumes at the first
failed buffer.

`clear()` removes all objects in the shared namespace, including caches visible
to other replicas with the same deployment identity. Use a separate namespace
for each isolated test; do not clear a namespace shared with serving replicas.

Requires the `codex/dynamic-dcp-storage-v2` branch (dynamic plugin capabilities on
top of SGLang PR #41187). A K3 EAGLE3 MLA draft additionally needs the separate
model and draft-DCP patches, which are not part of this branch.

## Configuration

```
--hicache-storage-backend dynamic
--hicache-storage-backend-extra-config '{
  "backend_name": "flashkv",
  "module_path": "apodex_flashkv.storage",
  "class_name": "FlashKVDCPStorage",
  "interface_v1": 1,
  "flashkv_dcp": {
    "model_revision": "<immutable target weights id>",
    "draft_revision": "<immutable draft weights id, when a draft pool exists>",
    "namespace": "<optional cache-sharing domain>"
  }
}'
```

Install with `pip install /path/to/flashkv_dcp` next to the authorized PAI client.

## CPU tests

Needs Python 3.12, CPU PyTorch, pytest, and the PAI FlashKV Python package source
(the PyPI package named `flashkv` is unrelated and must not be used).

```bash
export SGLANG_SOURCE=/path/to/sglang
export FLASHKV_PYTHON_SOURCE=/path/to/extracted-wheel/flashkv
python -m pytest -q examples/flashkv_dcp/tests
```

The harness imports the vendor module and the adapter unmodified and replaces the
native client with a byte-exact in-memory store; the MLA page index math is
replayed from the checkout. Run it in its own pytest process, since it replaces
allocator modules in `sys.modules`. Missing vendor source skips the suite; a run
consisting of skips is not a pass.

## Validation status

The 50-test CPU suite passes on the generic contract based on PR #41187 at
`0cbfb3ff2ec5231599c3fda1aa4f933c230b7845`, using the vendor Python integration
with its native client replaced by the byte-exact test store.

The same adapter implementation was also tested with the native ARM client in
an isolated K3 integration based on the earlier PR #41187 commit `e945243e9`,
plus separate model/draft prerequisites. Each worker used eight GB300 GPUs,
TP8/DCP8, BF16 KV, FP32 KDA state, and EAGLE3 speculation. The exact new-base
combination has CPU coverage only.

Controlled cross-worker readback used synthetic packed spans of real coding
histories, C1, 256 output tokens, three repeats per condition, and a cold reader
with device/host hits verified as zero. Native Prefill wall-time medians were:

| Input tokens | Cold recompute | L3 readback | Restored L3 tokens |
| ---: | ---: | ---: | ---: |
| 65,536 | 3.026 s | 0.832 s | 49,152 |
| 131,072 | 6.678 s | 0.991 s | 114,688 |
| 262,144 | 15.890 s | 1.312 s | 245,760 |

All nine measured L3 readers matched the corresponding cold output token IDs.
Decode rates were essentially unchanged. Prefill wall time includes scheduling
between chunks; standalone L3 I/O time was not measured. The common prefix was
input length minus 128 tokens; KDA restoration stopped at the preceding 16K
checkpoint, not at every logical 512-token page.

A separate fixed-history SWE-smith replay used 32 sessions and 380 requests per
run, with three off/on pairs at client concurrency 8 and 16 across two workers.
All 4,560 measured requests succeeded with zero retractions. L1/L2 already
covered 69.8% of input tokens. L3 reduced median Prefill latency, but median
end-to-end completion rates fell from 3.329 to 2.968 requests/s at C8 and from 4.451
to 4.257 requests/s at C16. Thus this workload did not gain aggregate throughput.
Concurrent outputs were not bitwise repeatable even in off/off controls; the
serial controls were repeatable. These are latency/cache checks, not coding
task-solving accuracy scores.
