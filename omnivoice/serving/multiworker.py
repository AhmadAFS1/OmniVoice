#!/usr/bin/env python3

"""Same-GPU multi-worker serving backend for OmniVoice."""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import queue
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from omnivoice.serving.service import (
    GenerationRequestPayload,
    GenerationResponsePayload,
    GenerationService,
    GenerationServiceConfig,
)

logger = logging.getLogger(__name__)


def _iso_utc_from_epoch(timestamp: Optional[float]) -> Optional[str]:
    if timestamp is None:
        return None
    return (
        datetime.fromtimestamp(timestamp, tz=timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def _worker_request_thread_count(config: GenerationServiceConfig) -> int:
    # Allow enough concurrent request handlers per worker so the local
    # GenerationBatcher can still aggregate bursts into one merged batch.
    return max(32, min(256, int(config.max_batch_requests) * 4))


@dataclass(frozen=True)
class WorkerStartupEnvelope:
    worker_id: str
    worker_pid: int
    ok: bool
    error: Optional[str] = None
    health: Optional[dict[str, Any]] = None


@dataclass
class _PendingResponse:
    request_id: str
    worker_id: str
    future: Future
    submitted_at: float


@dataclass
class _WorkerHandle:
    index: int
    worker_id: str
    worker_pid: int
    process: Any
    request_queue: Any
    response_queue: Any
    startup_health: dict[str, Any]
    reader_thread: Optional[threading.Thread] = None
    alive: bool = True
    pending_requests: int = 0
    total_assigned: int = 0
    total_completed: int = 0
    total_failed: int = 0
    ewma_batch_exec_ms: Optional[float] = None
    ewma_queue_wait_ms: Optional[float] = None
    last_response_headers: dict[str, str] = field(default_factory=dict)
    last_response_at: Optional[float] = None
    last_error: Optional[str] = None


def _worker_process_main(
    worker_index: int,
    config: GenerationServiceConfig,
    request_queue: Any,
    response_queue: Any,
    startup_queue: Any,
) -> None:
    log_level = os.environ.get("OMNIVOICE_LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s [pid=%(process)d] %(name)s %(levelname)s: %(message)s",
    )

    service_label = f"worker-{worker_index}"
    service: Optional[GenerationService] = None
    executor: Optional[ThreadPoolExecutor] = None

    try:
        service = GenerationService(config=config, service_label=service_label)
        startup_health = service.health()
        gpu_snapshot = (
            startup_health.get("batching", {}).get("gpu")
            if isinstance(startup_health.get("batching"), dict)
            else None
        ) or {}
        logger.info(
            "[%s] worker_ready pid=%s device=%s gpu_total_mb=%s gpu_used_mb=%s gpu_free_mb=%s torch_reserved_mb=%s",
            service_label,
            os.getpid(),
            startup_health.get("device"),
            gpu_snapshot.get("gpu_memory_total_mb"),
            gpu_snapshot.get("gpu_memory_used_mb"),
            gpu_snapshot.get("gpu_memory_free_mb"),
            gpu_snapshot.get("torch_reserved_mb"),
        )
        startup_queue.put(
            WorkerStartupEnvelope(
                worker_id=service.worker_id,
                worker_pid=service.worker_pid,
                ok=True,
                health=startup_health,
            )
        )

        executor = ThreadPoolExecutor(
            max_workers=_worker_request_thread_count(config),
            thread_name_prefix=f"omnivoice-{service_label}",
        )

        while True:
            payload = request_queue.get()
            if payload is None:
                break
            if not isinstance(payload, GenerationRequestPayload):
                logger.warning(
                    "[%s] Ignoring unexpected payload type: %s",
                    service_label,
                    type(payload).__name__,
                )
                continue
            executor.submit(_process_worker_request, service, payload, response_queue)
    except Exception as exc:
        logger.exception("[%s] worker startup failed", service_label)
        startup_queue.put(
            WorkerStartupEnvelope(
                worker_id=service_label,
                worker_pid=os.getpid(),
                ok=False,
                error=f"{type(exc).__name__}: {exc}",
            )
        )
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=False)
        if service is not None:
            service.close()
        try:
            response_queue.put(None)
        except Exception:
            logger.exception("[%s] Failed to post worker shutdown sentinel", service_label)


def _process_worker_request(
    service: GenerationService,
    payload: GenerationRequestPayload,
    response_queue: Any,
) -> None:
    response = service.generate(payload)
    response_queue.put(response)


class MultiWorkerGenerationBackend:
    """Router process that fans requests out to same-GPU worker subprocesses."""

    def __init__(
        self,
        config: GenerationServiceConfig,
        worker_count: int = 2,
        start_method: str = "spawn",
    ):
        if worker_count < 1:
            raise ValueError("worker_count must be >= 1")

        self._config = config
        self._worker_count = int(worker_count)
        self._start_method = start_method
        self._ctx = mp.get_context(start_method)
        self._lock = threading.Lock()
        self._pending: dict[str, _PendingResponse] = {}
        self._workers: list[_WorkerHandle] = []
        self._closing = False
        self._tie_breaker = 0

        for worker_index in range(1, self._worker_count + 1):
            self._workers.append(self._start_worker(worker_index))

    def _start_worker(self, worker_index: int) -> _WorkerHandle:
        request_queue = self._ctx.Queue()
        response_queue = self._ctx.Queue()
        startup_queue = self._ctx.Queue()
        process = self._ctx.Process(
            target=_worker_process_main,
            name=f"omnivoice-worker-{worker_index}",
            args=(
                worker_index,
                self._config,
                request_queue,
                response_queue,
                startup_queue,
            ),
        )
        process.start()

        startup_deadline = time.monotonic() + 900.0
        startup: Optional[WorkerStartupEnvelope] = None
        while time.monotonic() < startup_deadline:
            try:
                startup = startup_queue.get(timeout=1.0)
                break
            except queue.Empty:
                if not process.is_alive():
                    break

        if startup is None:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5.0)
            raise RuntimeError(
                f"Worker {worker_index} failed to report readiness before timeout."
            )

        if not startup.ok:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5.0)
            raise RuntimeError(
                f"Worker {worker_index} failed to start: {startup.error}"
            )

        handle = _WorkerHandle(
            index=worker_index,
            worker_id=startup.worker_id,
            worker_pid=startup.worker_pid,
            process=process,
            request_queue=request_queue,
            response_queue=response_queue,
            startup_health=startup.health or {},
        )
        handle.reader_thread = threading.Thread(
            target=self._reader_loop,
            args=(handle,),
            name=f"omnivoice-{handle.worker_id}-reader",
            daemon=True,
        )
        handle.reader_thread.start()

        gpu_snapshot = (
            handle.startup_health.get("batching", {}).get("gpu")
            if isinstance(handle.startup_health.get("batching"), dict)
            else None
        ) or {}
        logger.info(
            "worker_started worker_id=%s pid=%s pending=%d gpu_total_mb=%s gpu_used_mb=%s gpu_free_mb=%s torch_reserved_mb=%s",
            handle.worker_id,
            handle.worker_pid,
            handle.pending_requests,
            gpu_snapshot.get("gpu_memory_total_mb"),
            gpu_snapshot.get("gpu_memory_used_mb"),
            gpu_snapshot.get("gpu_memory_free_mb"),
            gpu_snapshot.get("torch_reserved_mb"),
        )
        return handle

    def _reader_loop(self, handle: _WorkerHandle) -> None:
        while True:
            try:
                item = handle.response_queue.get(timeout=0.5)
            except queue.Empty:
                if self._closing:
                    continue
                if not handle.process.is_alive():
                    self._mark_worker_failed(
                        handle,
                        RuntimeError(
                            f"{handle.worker_id} exited unexpectedly with code "
                            f"{handle.process.exitcode}"
                        ),
                    )
                    return
                continue

            if item is None:
                return

            if not isinstance(item, GenerationResponsePayload):
                logger.warning(
                    "Ignoring unexpected worker response type from %s: %s",
                    handle.worker_id,
                    type(item).__name__,
                )
                continue

            self._deliver_response(handle, item)

    def _deliver_response(
        self,
        handle: _WorkerHandle,
        response: GenerationResponsePayload,
    ) -> None:
        with self._lock:
            pending = self._pending.pop(response.request_id, None)
            handle.pending_requests = max(0, handle.pending_requests - 1)
            handle.last_response_headers = dict(response.headers)
            handle.last_response_at = time.time()
            if response.ok:
                handle.total_completed += 1
            else:
                handle.total_failed += 1
            self._update_ewma(handle, response)

        if pending is not None and not pending.future.done():
            pending.future.set_result(response)

    def _update_ewma(
        self,
        handle: _WorkerHandle,
        response: GenerationResponsePayload,
    ) -> None:
        batch_exec_ms = _header_float(
            response.headers,
            "X-OmniVoice-Batch-Exec-Ms",
        )
        queue_wait_ms = _header_float(
            response.headers,
            "X-OmniVoice-Queue-Wait-Ms",
        )
        handle.ewma_batch_exec_ms = _ewma(handle.ewma_batch_exec_ms, batch_exec_ms)
        handle.ewma_queue_wait_ms = _ewma(handle.ewma_queue_wait_ms, queue_wait_ms)

    def _mark_worker_failed(
        self,
        handle: _WorkerHandle,
        exc: Exception,
    ) -> None:
        failed: list[_PendingResponse] = []
        with self._lock:
            if not handle.alive:
                return
            handle.alive = False
            handle.last_error = f"{type(exc).__name__}: {exc}"
            for request_id, pending in list(self._pending.items()):
                if pending.worker_id != handle.worker_id:
                    continue
                failed.append(self._pending.pop(request_id))
            handle.pending_requests = 0

        logger.error(
            "worker_failed worker_id=%s pid=%s error=%s",
            handle.worker_id,
            handle.worker_pid,
            handle.last_error,
        )
        for pending in failed:
            if not pending.future.done():
                pending.future.set_exception(exc)

    def _check_worker_liveness(self) -> None:
        for handle in self._workers:
            if handle.alive and not handle.process.is_alive():
                self._mark_worker_failed(
                    handle,
                    RuntimeError(
                        f"{handle.worker_id} exited unexpectedly with code "
                        f"{handle.process.exitcode}"
                    ),
                )

    def _select_worker(self) -> _WorkerHandle:
        self._check_worker_liveness()
        with self._lock:
            candidates = [
                handle
                for handle in self._workers
                if handle.alive and handle.process.is_alive()
            ]
            if not candidates:
                raise RuntimeError("No alive OmniVoice workers are available.")

            sort_key = lambda item: (
                item.pending_requests,
                item.ewma_batch_exec_ms if item.ewma_batch_exec_ms is not None else 0.0,
                item.total_assigned,
            )
            best_key = min(sort_key(worker) for worker in candidates)
            tied = [worker for worker in candidates if sort_key(worker) == best_key]
            chosen = tied[self._tie_breaker % len(tied)]
            self._tie_breaker += 1

            chosen.pending_requests += 1
            chosen.total_assigned += 1
            return chosen

    def _router_error_response(
        self,
        request: GenerationRequestPayload,
        error: Exception,
        worker_id: Optional[str] = None,
        worker_pid: Optional[int] = None,
    ) -> GenerationResponsePayload:
        error_message = f"{type(error).__name__}: {error}"
        logger.exception(
            "router_request_failed request_id=%s worker_id=%s",
            request.request_id,
            worker_id or "-",
        )
        headers = {
            "X-OmniVoice-Request-Id": request.request_id,
        }
        if worker_id is not None:
            headers["X-OmniVoice-Worker-Id"] = worker_id
        if worker_pid is not None:
            headers["X-OmniVoice-Worker-Pid"] = str(worker_pid)
        return GenerationResponsePayload(
            request_id=request.request_id,
            ok=False,
            status_code=500,
            error=error_message,
            wav_bytes=None,
            headers=headers,
            worker_id=worker_id or "router",
            worker_pid=worker_pid or os.getpid(),
        )

    def generate(
        self,
        request: GenerationRequestPayload,
    ) -> GenerationResponsePayload:
        try:
            handle = self._select_worker()
        except Exception as exc:
            return self._router_error_response(request, exc)

        future: Future = Future()
        with self._lock:
            self._pending[request.request_id] = _PendingResponse(
                request_id=request.request_id,
                worker_id=handle.worker_id,
                future=future,
                submitted_at=time.time(),
            )

        try:
            handle.request_queue.put(request)
        except Exception as exc:
            with self._lock:
                self._pending.pop(request.request_id, None)
                handle.pending_requests = max(0, handle.pending_requests - 1)
            return self._router_error_response(
                request,
                exc,
                worker_id=handle.worker_id,
                worker_pid=handle.worker_pid,
            )

        try:
            return future.result()
        except Exception as exc:
            with self._lock:
                self._pending.pop(request.request_id, None)
                handle.pending_requests = max(0, handle.pending_requests - 1)
            return self._router_error_response(
                request,
                exc,
                worker_id=handle.worker_id,
                worker_pid=handle.worker_pid,
            )

    def health(self) -> dict[str, Any]:
        self._check_worker_liveness()
        with self._lock:
            workers = [
                {
                    "worker_id": handle.worker_id,
                    "worker_pid": handle.worker_pid,
                    "alive": handle.alive and handle.process.is_alive(),
                    "pending_requests": handle.pending_requests,
                    "total_assigned": handle.total_assigned,
                    "total_completed": handle.total_completed,
                    "total_failed": handle.total_failed,
                    "ewma_batch_exec_ms": _round_or_none(handle.ewma_batch_exec_ms),
                    "ewma_queue_wait_ms": _round_or_none(handle.ewma_queue_wait_ms),
                    "last_response_at": _iso_utc_from_epoch(handle.last_response_at),
                    "last_error": handle.last_error,
                    "last_response_headers": dict(handle.last_response_headers),
                    "startup_health": handle.startup_health,
                }
                for handle in self._workers
            ]
            inflight_counts = {
                handle.worker_id: handle.pending_requests for handle in self._workers
            }
        return {
            "backend_mode": "same_gpu_workers",
            "gpu_workers": self._worker_count,
            "worker_request_threads": _worker_request_thread_count(self._config),
            "pending_requests": sum(inflight_counts.values()),
            "inflight_counts": inflight_counts,
            "workers": workers,
        }

    def close(self) -> None:
        self._closing = True

        for handle in self._workers:
            try:
                handle.request_queue.put(None)
            except Exception:
                logger.exception("Failed to send shutdown signal to %s", handle.worker_id)

        for handle in self._workers:
            if handle.reader_thread is not None:
                handle.reader_thread.join(timeout=2.0)
            handle.process.join(timeout=5.0)
            if handle.process.is_alive():
                logger.warning(
                    "Force-terminating worker %s pid=%s",
                    handle.worker_id,
                    handle.worker_pid,
                )
                handle.process.terminate()
                handle.process.join(timeout=5.0)

        with self._lock:
            pending = list(self._pending.values())
            self._pending.clear()

        shutdown_error = RuntimeError("Router is shutting down.")
        for pending_item in pending:
            if not pending_item.future.done():
                pending_item.future.set_exception(shutdown_error)


def _ewma(current: Optional[float], latest: Optional[float], alpha: float = 0.35) -> Optional[float]:
    if latest is None:
        return current
    if current is None:
        return latest
    return (alpha * latest) + ((1.0 - alpha) * current)


def _header_float(headers: dict[str, str], key: str) -> Optional[float]:
    value = headers.get(key)
    if value in (None, ""):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _round_or_none(value: Optional[float], digits: int = 2) -> Optional[float]:
    if value is None:
        return None
    return round(value, digits)
