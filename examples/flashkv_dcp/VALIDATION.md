# Review scope and validation

## Dependency and commit layout

Base: SGLang PR [#41187](https://github.com/sgl-project/sglang/pull/41187), commit `e945243e9bff6fef0b0c36172859c1890101bef1`. The PR was open and this was its head when the review branch was prepared.

The prerequisite patches used by the K3 integration trial are separated from the storage extension:

1. MLA DCP draft execution, zero-KV validation, and top-k guard.
2. K3 EAGLE3 auxiliary-state capture and its registered unit test.
3. Loader configuration isolation when target and draft use different loaders.
4. MLA draft physical KV geometry and HiCache sidecar transfers.
5. Dynamic backend capability validation and host-memory registration/cleanup ownership.
6. The optional FlashKV adapter, portable CPU tests, and these review notes.

The adapter subclasses the separately installed PAI client. No vendor source, binary, model weights, cloud deployment configuration, credentials, or raw request logs are included. Machine-specific collective tuning from the integration environment is also excluded.

## CPU regression coverage

The three suites under `tests/` pass **107 tests** against this checkout, the actual vendor Python integration, and CPU PyTorch. Native transport, CUDA device selection, and GPU allocator classes are explicit test doubles. Source-method tests exercise the selected checkout's page geometry, sparse checkpoint synchronization, late registration, and detach paths. Run them in a separate pytest process as described in the README.

Coverage includes shard/model/namespace isolation, logical-page lengths, component cardinality, sparse KDA checkpoint intersections, missing objects, pre-construction rejection, late-sidecar registration, and explicit unregister/retry behavior. Missing vendor source is reported as a dependency skip, not as a successful test run. The registered K3 capture test is included separately; it is not part of the 107-test count.

## Completed integration evidence

The earlier isolated ARM/GB300 trial used the same modified framework source and adapter, plus the environment's machine-specific collective tuning. It exercised the real vendor client and KVS server. Native byte tests covered complete readback, missing objects, sparse checkpoint selection, and namespace isolation. A Get operation selected only objects proven restorable by Query; mixed present/missing native Get batches had failed as a whole in an earlier attempt and are not an accepted fallback path.

A full Kimi-K3 + EAGLE3 MLA trial used eight GB300 GPUs, TP8/DCP8, 8,194 input tokens, and 32 output tokens. Writer processes were stopped before fresh reader processes started. The reader reported:

| Request cache tier | Tokens |
| --- | ---: |
| Device | 0 |
| Host | 0 |
| Storage | 8,192 |

All 32 generated output token IDs exactly matched the writer. This is a functional cache-restoration check on one prompt, not a model-quality suite or an agentic throughput benchmark. Repeated TP metric series were not treated as additional request tokens.

## Remaining integration work

- Agentic coding replay comparing L3 off/on with identical local L1/L2 configuration, request traces, output lengths, routing, and concurrency.
- Fixed-session and cross-worker routing controls, whole-replay output throughput, latency distributions, and request-level storage usage with explicit denominators.
- Repeated paired rounds, including natural write/read races and cold-start costs.
- Repeated live attach/detach checks with pending I/O and resource-lifecycle observations.
- Model-output quality checks broader than the completed single-prompt token comparison.

The agentic dataset and replay harness were prepared, but the new test deployment stopped before serving any agentic request. No agentic hit-rate or throughput improvement is claimed by this branch. The runtime access issue and cloud provisioning tools are outside this code review.
