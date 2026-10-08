"""Coordinate the K3 frontend's preprocessing and VMM producer CUDA work.

This is a scoped concurrency mitigation, not a native CUDA deadlock diagnosis.
The lock is process-local and never shared with Scheduler processes.  It must
always be acquired before the VMM metadata lock; cancellation uses metadata
only and must not wait for CUDA.
"""

import threading
from contextlib import contextmanager

_frontend_cuda_lock = threading.RLock()


@contextmanager
def frontend_cuda_section():
    """Serialize the known frontend CUDA submission/synchronization domains."""
    with _frontend_cuda_lock:
        yield


@contextmanager
def frontend_cuda_preprocess():
    """Finish preprocessing's current stream before another worker publishes.

    Preserve the existing stream and all image operations.  The explicit
    completion boundary also covers the exception path: a failed preprocessing
    call must not leave already-enqueued CUDA work racing the next producer.
    A native CUDA call can still hang here; the existing tracked worker and
    health timeout, and the external collector, remain responsible for that
    failure.  No timeout pretends to cancel a running native call.
    """
    import torch

    with _frontend_cuda_lock:
        stream = torch.cuda.current_stream()
        try:
            yield
        except BaseException as error:
            try:
                stream.synchronize()
            except BaseException as sync_error:
                error.add_note("K3 frontend preprocessing stream cleanup failed")
                raise error from sync_error
            raise
        else:
            stream.synchronize()
