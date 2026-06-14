# Decode RoleInstance Recreate Loki Notes

## Symptom

`glm51-debug-decode-0` pods were deleted and recreated by the controller. This
was a RoleInstance-level recreate, not a same-pod container restart.

Kubernetes event:

- `ReCreateInstance`
- `RestartPolicy is RecreateInstanceOnPodRestart, recreate all pods of instance: llm-serving/glm51-debug-decode-0`

RoleInstance policy:

- `RecreateRoleInstanceOnPodRestart`

## Loki Evidence

Current deletion window:

- `2026-06-13T21:33:51Z`: `glm51-debug-decode-0-1` logged
  `All 1 scheduler process(es) initialized successfully`.
- Immediately after, the same pod failed in gRPC request-manager startup:
  - `sglang/srt/entrypoints/grpc_server.py:254`
  - `smg_grpc_servicer/sglang/server.py:114`
  - `smg_grpc_servicer/sglang/request_manager.py:196`
  - `sglang/srt/utils/network.py:393`
- Error:
  `zmq.error.ZMQError: Cannot assign requested address (addr='tcp://glm51-debug-decode-0-0.s-glm51-debug-decode.llm-serving:5002')`

Controller logs:

- `2026-06-13T21:33:57.935Z`: `pod runtime not ready` for
  `glm51-debug-decode-0-1`.
- Repeated controller state:
  `unavailableInstances=["glm51-debug-decode-0"]`.

Earlier deletion window showed the same failure mode on
`glm51-debug-decode-0-2`.

## Cause

`smg_grpc_servicer` starts `GrpcRequestManager` on every node. In multi-node
SGLang, only `node_rank == 0` should own the frontend/request-manager sockets.
Worker nodes should launch scheduler workers and connect through the rank0
frontend path.

The worker pod had `node_rank != 0`, but still tried to bind a ZMQ frontend
socket to the rank0 DNS address:

`tcp://glm51-debug-decode-0-0.s-glm51-debug-decode.llm-serving:5002`

That address is not local to the worker pod, so ZeroMQ correctly failed with
`Cannot assign requested address`.

## Code Fix

`sglang.srt.entrypoints.grpc_server.serve_grpc` now handles multi-node gRPC
workers before delegating to the third-party rank0 gRPC server path:

- `nnodes == 1`: unchanged.
- `nnodes > 1 && node_rank == 0`: unchanged, still starts gRPC frontend and
  request manager.
- `nnodes > 1 && node_rank != 0`: starts scheduler processes only, keeps the
  worker process alive, and does not construct `GrpcRequestManager`.

This prevents non-rank0 pods from binding frontend ZMQ sockets to the rank0
pod DNS address.
