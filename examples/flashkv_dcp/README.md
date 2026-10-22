# Experimental FlashKV DCP adapter

This package adapts the PAI FlashKV Python storage integration for SGLang's DCP-aware HiCache contract. It contains the Apodex adapter and portable CPU regression tests. It does not contain the vendor FlashKV source, wheel, native library, model weights, credentials, or deployment configuration.

The target SGLang checkout must include the DCP storage contract, dynamic-backend capability validation, EAGLE MLA host-pool integration, and explicit memory-cleanup handling represented by this change. An unrelated upstream version is not an equivalent test target.

This review branch is based on [SGLang #41187](https://github.com/sgl-project/sglang/pull/41187), at `e945243e9bff6fef0b0c36172859c1890101bef1`. It includes the K3/EAGLE3 MLA prerequisites as separate commits. See [VALIDATION.md](VALIDATION.md) for the measured scope and remaining integration work.

## CPU regression tests

Dependencies: Python 3.12 or later, CPU-capable PyTorch, pytest, NumPy, and the **actual PAI FlashKV Python integration source**. The public PyPI package named `flashkv` is not the required PAI implementation.

Set the checkout and vendor paths explicitly:

```bash
export SGLANG_SOURCE=/absolute/path/to/sglang
export FLASHKV_PYTHON_SOURCE=/absolute/path/to/extracted-wheel/flashkv

python -m pip install torch pytest numpy
python -m pytest -q -rs /absolute/path/to/flashkv_dcp/tests/test_flashkv_adapter.py \
  /absolute/path/to/flashkv_dcp/tests/test_page_geometry.py \
  /absolute/path/to/flashkv_dcp/tests/test_lifecycle.py
```

`FLASHKV_PYTHON_SOURCE` may point to the `flashkv` package directory or its parent wheel-extraction directory. It must contain `integration/sglang/flashkv_storage.py` and `packing_utils.py`. If unset, the tests locate these Python files in the installed PAI `flashkv` package **without importing its native extension**.

`SGLANG_SOURCE` may point to a checkout root or its `python/sglang` directory. If unset, tests search the current directory and their parent directories for a checkout. This permits copying this package inside the target repository without retaining any research-directory paths.

If PAI Python source is absent, pytest reports an explicit dependency skip for the affected test modules. A run consisting of skips is not a passing regression result. No substitute vendor Python implementation is synthesized. An explicitly invalid `SGLANG_SOURCE` is an error.

With an existing cached uv environment, an offline equivalent is:

```bash
SGLANG_SOURCE=/absolute/path/to/sglang \
FLASHKV_PYTHON_SOURCE=/absolute/path/to/extracted-wheel/flashkv \
uv run --offline --with torch --with pytest --with numpy \
  python -m pytest -q -rs /absolute/path/to/flashkv_dcp/tests
```

Run these tests in their own pytest process. The CPU harness intentionally replaces native-client and allocator modules in `sys.modules`; it must not share a process with full-runtime SGLang tests.

## What the CPU tests exercise

- Import the complete external vendor Python integration, adapter, checkout `HiCacheStorage` contract and backend factory.
- Use actual CPU Torch tensors and pointer-addressed byte copies to verify object identities, logical-page sizing, component ordering, sparse checkpoints, partial failures and memory-ownership state transitions.
- Replay exact source methods from the **selected current checkout** for MLA page address calculations, sparse checkpoint synchronization, late-pool registration and detach cleanup. No historical PR snapshot is loaded.
- Supply explicit test doubles for the native FlashKV client and CUDA device selection, and metadata-only replacements for GPU-allocating host classes. The distributed MIN collective is modeled locally; no distributed process group is launched.

The fake native client's partial-result behavior is a Python contract fixture, not a claim about the real native transport. In particular, mixed present/missing native Get batches can fail as a whole; production callers query and select recoverable pages before Get. Native ARM/CUDA ABI, RDMA, real allocator construction, distributed execution, model output correctness and performance require separate integration tests.

## Installing the adapter for an integration environment

```bash
python -m pip install /absolute/path/to/flashkv_dcp
```

Install the authorized PAI FlashKV client separately. The adapter requires `interface_v1=1` and a `flashkv_dcp` configuration with `schema_version=1`, immutable `model_revision`, and `draft_revision` when an MLA draft pool is present. `namespace` selects the intended cache-sharing domain. SGLang's dynamic-loader target is `apodex_flashkv.storage.FlashKVDCPStorage`.

With `--hicache-storage-backend dynamic`, use this extra configuration, replacing the revision placeholders with immutable identities of the actual weights and semantic configuration:

```json
{
  "backend_name": "flashkv",
  "module_path": "apodex_flashkv.storage",
  "class_name": "FlashKVDCPStorage",
  "interface_v1": 1,
  "flashkv_dcp": {
    "schema_version": 1,
    "model_revision": "IMMUTABLE_TARGET_REVISION",
    "draft_revision": "IMMUTABLE_DRAFT_REVISION",
    "namespace": "isolated-flashkv-test"
  }
}
```

The tested PAI client is the ARM64 Python 3.12 / CUDA 13 integration distributed with `flashkv-dist:1.0.0-df62dbba` (wheel version `1.0.0+gdf62dbb.cuda.cu13`). The corresponding KVS server must be deployed and discoverable using the vendor-supported runtime environment. The isolated integration used `write_through`, `page_first_direct`, TP8/DCP8, and an MLA EAGLE3 draft with top-k 1. This example does not configure the server, Redis, service discovery, or cloud permissions.

The adapter intentionally rejects unsupported pool formats/topologies and disables bulk namespace clearing. `close()` unregisters tracked successful native memory registrations; failed registrations with ambiguous native side effects and native transport-thread shutdown still require process-level cleanup.
