#!/usr/bin/env python3

"""Burst benchmark helper for the OmniVoice batching API server.

This script sends a concurrent burst of API requests and summarizes:

- total wall-clock throughput
- per-request latency / queue wait / batch execution timings
- observed batch sizes
- request success rate

It uses ``curl`` under the hood so it works against the existing multipart
form API without adding a new dependency.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Optional


DEFAULT_TEXTS = [
    "Hello there. This is a short batching benchmark request.",
    "The weather looks nice today. Please speak this naturally and clearly.",
    "We are measuring API throughput. Keep the delivery smooth and even.",
    "This is a short test sentence. It should be easy to batch with others.",
    "OmniVoice is running on a GPU server. This request checks queue behavior.",
    "Please read these two short sentences. They are meant for batching tests.",
    "This benchmark uses one or two sentences. We want realistic short requests.",
    "The server should process many users together. This request helps validate that.",
    "Fast response times matter. Efficient batching should reduce total queue pressure.",
    "This is another compact request. It is intentionally short for concurrency tests.",
]


@dataclass
class RequestResult:
    index: int
    ok: bool
    http_status: int | None
    error: str | None
    local_wall_ms: float
    request_id: str | None
    latency_ms: float | None
    queue_wait_ms: float | None
    batch_exec_ms: float | None
    batch_requests: int | None
    batch_prompts: int | None
    shape_bucket_id: str | None
    exact_shape_homogeneous: bool | None
    batch_target_tokens: int | None
    batch_max_sequence_length: int | None
    batch_estimated_memory_mb: float | None
    worker_id: str | None
    worker_pid: int | None
    gpu_utilization_peak_pct: float | None
    gpu_memory_total_mb: float | None
    gpu_memory_used_peak_mb: float | None
    gpu_memory_free_before_mb: float | None
    gpu_memory_free_after_mb: float | None
    gpu_allocator_allocated_mb: float | None
    gpu_allocator_reserved_mb: float | None
    gpu_allocator_peak_allocated_mb: float | None
    gpu_allocator_peak_reserved_mb: float | None
    audio_duration_s: float | None
    rtf: float | None
    clone_prompt_cache_hit: bool | None
    stage_timings_ms: dict[str, float]
    text: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Burst benchmark for the OmniVoice API batching path.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--url",
        default="http://127.0.0.1:8002/generate",
        help="Full /generate endpoint URL.",
    )
    parser.add_argument(
        "--requests",
        type=int,
        default=100,
        help="Number of requests to send in the burst.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=100,
        help="Maximum number of concurrent clients.",
    )
    parser.add_argument(
        "--launch-window-s",
        type=float,
        default=0.0,
        help="Spread request launches uniformly over this many seconds. Use 0 for a pure burst.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="Random seed for launch-time jitter and prompt selection.",
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "design", "clone"],
        default="design",
        help="API mode to benchmark.",
    )
    parser.add_argument(
        "--instruct",
        default="female, low pitch, british accent",
        help="Voice design instruction for design mode.",
    )
    parser.add_argument(
        "--num-step",
        type=int,
        default=16,
        help="Iterative decode steps to request.",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=2.0,
        help="Guidance scale to request.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=4.0,
        help="Optional fixed duration in seconds. Set <= 0 to omit.",
    )
    parser.add_argument(
        "--postprocess-output",
        action="store_true",
        default=False,
        help="Enable output post-processing. Disabled by default for cleaner benchmarks.",
    )
    parser.add_argument(
        "--ref-audio",
        default=None,
        help="Reference audio path for clone mode.",
    )
    parser.add_argument(
        "--ref-text",
        default="This is the transcript of the reference audio.",
        help="Reference text for clone mode.",
    )
    parser.add_argument(
        "--csv",
        default="batching-bench.csv",
        help="CSV file to write detailed per-request results.",
    )
    parser.add_argument(
        "--curl-bin",
        default="curl",
        help="curl binary to use.",
    )
    return parser


def parse_headers(raw: str) -> tuple[int | None, dict[str, str]]:
    blocks = [block for block in raw.split("\r\n\r\n") if block.strip()]
    headers: dict[str, str] = {}
    http_status = None

    for block in blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if not lines:
            continue
        if lines[0].startswith("HTTP/"):
            parts = lines[0].split()
            if len(parts) >= 2 and parts[1].isdigit():
                http_status = int(parts[1])
        for line in lines[1:]:
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            headers[key.strip().lower()] = value.strip()

    return http_status, headers


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return math.nan
    if len(values) == 1:
        return values[0]
    rank = (len(values) - 1) * pct
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return values[low]
    weight = rank - low
    return values[low] * (1 - weight) + values[high] * weight


def _header_float(headers: dict[str, str], key: str) -> float | None:
    value = headers.get(key)
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _header_int(headers: dict[str, str], key: str) -> int | None:
    value = headers.get(key)
    if value is None or value == "":
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _header_bool(headers: dict[str, str], key: str) -> bool | None:
    value = headers.get(key)
    if value is None or value == "":
        return None
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    return None


def _extract_stage_timings_ms(headers: dict[str, str]) -> dict[str, float]:
    prefix = "x-omnivoice-timing-"
    suffix = "-ms"
    timings: dict[str, float] = {}
    for key, value in headers.items():
        if not key.startswith(prefix) or not key.endswith(suffix):
            continue
        try:
            metric_value = float(value)
        except ValueError:
            continue
        metric_name = key[len(prefix) : -len(suffix)].replace("-", "_") + "_ms"
        timings[metric_name] = metric_value
    return timings


def run_one_request(args: argparse.Namespace, index: int, text: str) -> RequestResult:
    cmd = [
        args.curl_bin,
        "-sS",
        "-D",
        "-",
        "-o",
        os.devnull,
        "-X",
        "POST",
        args.url,
        "-F",
        f"mode={args.mode}",
        "-F",
        f"text={text}",
        "-F",
        f"num_step={args.num_step}",
        "-F",
        f"guidance_scale={args.guidance_scale}",
        "-F",
        f"postprocess_output={'true' if args.postprocess_output else 'false'}",
    ]

    if args.mode == "design":
        cmd.extend(["-F", f"instruct={args.instruct}"])
    elif args.mode == "clone":
        if not args.ref_audio:
            raise ValueError("--ref-audio is required for clone mode")
        cmd.extend(["-F", f"ref_text={args.ref_text}"])
        cmd.extend(["-F", f"ref_audio=@{args.ref_audio}"])

    if args.duration and args.duration > 0:
        cmd.extend(["-F", f"duration={args.duration}"])

    started = time.perf_counter()
    try:
        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception as exc:
        return RequestResult(
            index=index,
            ok=False,
            http_status=None,
            error=f"{type(exc).__name__}: {exc}",
            local_wall_ms=(time.perf_counter() - started) * 1000.0,
            request_id=None,
            latency_ms=None,
            queue_wait_ms=None,
            batch_exec_ms=None,
            batch_requests=None,
            batch_prompts=None,
            shape_bucket_id=None,
            exact_shape_homogeneous=None,
            batch_target_tokens=None,
            batch_max_sequence_length=None,
            batch_estimated_memory_mb=None,
            worker_id=None,
            worker_pid=None,
            gpu_utilization_peak_pct=None,
            gpu_memory_total_mb=None,
            gpu_memory_used_peak_mb=None,
            gpu_memory_free_before_mb=None,
            gpu_memory_free_after_mb=None,
            gpu_allocator_allocated_mb=None,
            gpu_allocator_reserved_mb=None,
            gpu_allocator_peak_allocated_mb=None,
            gpu_allocator_peak_reserved_mb=None,
            audio_duration_s=None,
            rtf=None,
            clone_prompt_cache_hit=None,
            stage_timings_ms={},
            text=text,
        )

    local_wall_ms = (time.perf_counter() - started) * 1000.0

    if completed.returncode != 0:
        return RequestResult(
            index=index,
            ok=False,
            http_status=None,
            error=completed.stderr.strip() or f"curl exited with {completed.returncode}",
            local_wall_ms=local_wall_ms,
            request_id=None,
            latency_ms=None,
            queue_wait_ms=None,
            batch_exec_ms=None,
            batch_requests=None,
            batch_prompts=None,
            shape_bucket_id=None,
            exact_shape_homogeneous=None,
            batch_target_tokens=None,
            batch_max_sequence_length=None,
            batch_estimated_memory_mb=None,
            worker_id=None,
            worker_pid=None,
            gpu_utilization_peak_pct=None,
            gpu_memory_total_mb=None,
            gpu_memory_used_peak_mb=None,
            gpu_memory_free_before_mb=None,
            gpu_memory_free_after_mb=None,
            gpu_allocator_allocated_mb=None,
            gpu_allocator_reserved_mb=None,
            gpu_allocator_peak_allocated_mb=None,
            gpu_allocator_peak_reserved_mb=None,
            audio_duration_s=None,
            rtf=None,
            clone_prompt_cache_hit=None,
            stage_timings_ms={},
            text=text,
        )

    http_status, headers = parse_headers(completed.stdout)
    ok = http_status == 200
    return RequestResult(
        index=index,
        ok=ok,
        http_status=http_status,
        error=None if ok else completed.stdout.strip(),
        local_wall_ms=local_wall_ms,
        request_id=headers.get("x-omnivoice-request-id"),
        latency_ms=_header_float(headers, "x-omnivoice-latency-ms"),
        queue_wait_ms=_header_float(headers, "x-omnivoice-queue-wait-ms"),
        batch_exec_ms=_header_float(headers, "x-omnivoice-batch-exec-ms"),
        batch_requests=_header_int(headers, "x-omnivoice-batch-requests"),
        batch_prompts=_header_int(headers, "x-omnivoice-batch-prompts"),
        shape_bucket_id=headers.get("x-omnivoice-shape-bucket"),
        exact_shape_homogeneous=_header_bool(
            headers, "x-omnivoice-exact-shape-homogeneous"
        ),
        batch_target_tokens=_header_int(headers, "x-omnivoice-batch-target-tokens"),
        batch_max_sequence_length=_header_int(
            headers, "x-omnivoice-batch-max-sequence-length"
        ),
        batch_estimated_memory_mb=_header_float(
            headers, "x-omnivoice-batch-estimated-memory-mb"
        ),
        worker_id=headers.get("x-omnivoice-worker-id"),
        worker_pid=_header_int(headers, "x-omnivoice-worker-pid"),
        gpu_utilization_peak_pct=_header_float(
            headers, "x-omnivoice-gpu-utilization-peak-pct"
        ),
        gpu_memory_total_mb=_header_float(headers, "x-omnivoice-gpu-memory-total-mb"),
        gpu_memory_used_peak_mb=_header_float(
            headers, "x-omnivoice-gpu-memory-used-peak-mb"
        ),
        gpu_memory_free_before_mb=_header_float(
            headers, "x-omnivoice-gpu-memory-free-before-mb"
        ),
        gpu_memory_free_after_mb=_header_float(
            headers, "x-omnivoice-gpu-memory-free-after-mb"
        ),
        gpu_allocator_allocated_mb=_header_float(
            headers, "x-omnivoice-gpu-allocator-allocated-mb"
        ),
        gpu_allocator_reserved_mb=_header_float(
            headers, "x-omnivoice-gpu-allocator-reserved-mb"
        ),
        gpu_allocator_peak_allocated_mb=_header_float(
            headers, "x-omnivoice-gpu-allocator-peak-allocated-mb"
        ),
        gpu_allocator_peak_reserved_mb=_header_float(
            headers, "x-omnivoice-gpu-allocator-peak-reserved-mb"
        ),
        audio_duration_s=_header_float(headers, "x-omnivoice-audio-duration-s"),
        rtf=_header_float(headers, "x-omnivoice-rtf"),
        clone_prompt_cache_hit=_header_bool(
            headers, "x-omnivoice-clone-prompt-cache-hit"
        ),
        stage_timings_ms=_extract_stage_timings_ms(headers),
        text=text,
    )


def run_scheduled_request(
    args: argparse.Namespace,
    index: int,
    text: str,
    launch_offset_s: float,
    launch_base: float,
) -> RequestResult:
    target = launch_base + launch_offset_s
    remaining = target - time.perf_counter()
    if remaining > 0:
        time.sleep(remaining)
    return run_one_request(args, index=index, text=text)


def write_csv(path: str, results: list[RequestResult]) -> None:
    stage_timing_columns = sorted(
        {
            f"timing_{timing_name}"
            for result in results
            for timing_name in result.stage_timings_ms.keys()
        }
    )
    fieldnames = [
        "index",
        "ok",
        "http_status",
        "error",
        "local_wall_ms",
        "request_id",
        "latency_ms",
        "queue_wait_ms",
        "batch_exec_ms",
        "batch_requests",
        "batch_prompts",
        "shape_bucket_id",
        "exact_shape_homogeneous",
        "batch_target_tokens",
        "batch_max_sequence_length",
        "batch_estimated_memory_mb",
        "worker_id",
        "worker_pid",
        "gpu_utilization_peak_pct",
        "gpu_memory_total_mb",
        "gpu_memory_used_peak_mb",
        "gpu_memory_free_before_mb",
        "gpu_memory_free_after_mb",
        "gpu_allocator_allocated_mb",
        "gpu_allocator_reserved_mb",
        "gpu_allocator_peak_allocated_mb",
        "gpu_allocator_peak_reserved_mb",
        "audio_duration_s",
        "rtf",
        "clone_prompt_cache_hit",
        *stage_timing_columns,
        "text",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            row = {
                "index": result.index,
                "ok": result.ok,
                "http_status": result.http_status,
                "error": result.error,
                "local_wall_ms": result.local_wall_ms,
                "request_id": result.request_id,
                "latency_ms": result.latency_ms,
                "queue_wait_ms": result.queue_wait_ms,
                "batch_exec_ms": result.batch_exec_ms,
                "batch_requests": result.batch_requests,
                "batch_prompts": result.batch_prompts,
                "shape_bucket_id": result.shape_bucket_id,
                "exact_shape_homogeneous": result.exact_shape_homogeneous,
                "batch_target_tokens": result.batch_target_tokens,
                "batch_max_sequence_length": result.batch_max_sequence_length,
                "batch_estimated_memory_mb": result.batch_estimated_memory_mb,
                "worker_id": result.worker_id,
                "worker_pid": result.worker_pid,
                "gpu_utilization_peak_pct": result.gpu_utilization_peak_pct,
                "gpu_memory_total_mb": result.gpu_memory_total_mb,
                "gpu_memory_used_peak_mb": result.gpu_memory_used_peak_mb,
                "gpu_memory_free_before_mb": result.gpu_memory_free_before_mb,
                "gpu_memory_free_after_mb": result.gpu_memory_free_after_mb,
                "gpu_allocator_allocated_mb": result.gpu_allocator_allocated_mb,
                "gpu_allocator_reserved_mb": result.gpu_allocator_reserved_mb,
                "gpu_allocator_peak_allocated_mb": result.gpu_allocator_peak_allocated_mb,
                "gpu_allocator_peak_reserved_mb": result.gpu_allocator_peak_reserved_mb,
                "audio_duration_s": result.audio_duration_s,
                "rtf": result.rtf,
                "clone_prompt_cache_hit": result.clone_prompt_cache_hit,
                "text": result.text,
            }
            for column in stage_timing_columns:
                timing_name = column[len("timing_") :]
                row[column] = result.stage_timings_ms.get(timing_name)
            writer.writerow(row)


def print_summary(
    results: list[RequestResult],
    total_wall_s: float,
    requests: int,
    concurrency: int,
    launch_window_s: float,
    csv_path: str,
) -> None:
    successes = [r for r in results if r.ok]
    failures = [r for r in results if not r.ok]

    print("Batching Benchmark Summary")
    print(f"Requests: {requests}")
    print(f"Concurrency: {concurrency}")
    print(f"Launch window: {launch_window_s:.2f}s")
    print(f"Successes: {len(successes)}")
    print(f"Failures: {len(failures)}")
    print(f"Total wall time: {total_wall_s:.2f}s")
    print(
        f"Effective throughput: {(len(successes) / total_wall_s):.2f} req/s"
        if total_wall_s > 0
        else "Effective throughput: n/a"
    )

    def emit_metric(label: str, values: list[float]) -> None:
        if not values:
            print(f"{label}: n/a")
            return
        values = sorted(values)
        print(
            f"{label}: mean={statistics.mean(values):.2f} "
            f"p50={statistics.median(values):.2f} "
            f"p95={percentile(values, 0.95):.2f} "
            f"p99={percentile(values, 0.99):.2f} "
            f"min={values[0]:.2f} max={values[-1]:.2f}"
        )

    emit_metric(
        "Latency ms",
        [r.latency_ms for r in successes if r.latency_ms is not None],
    )
    emit_metric(
        "Queue wait ms",
        [r.queue_wait_ms for r in successes if r.queue_wait_ms is not None],
    )
    emit_metric(
        "Batch exec ms",
        [r.batch_exec_ms for r in successes if r.batch_exec_ms is not None],
    )
    emit_metric(
        "Local wall ms",
        [r.local_wall_ms for r in successes if r.local_wall_ms is not None],
    )

    stage_timing_keys = sorted(
        {
            key
            for result in successes
            for key in result.stage_timings_ms.keys()
        }
    )
    for timing_key in stage_timing_keys:
        emit_metric(
            f"Timing {timing_key}",
            [
                result.stage_timings_ms[timing_key]
                for result in successes
                if timing_key in result.stage_timings_ms
            ],
        )

    batch_hist: dict[int, int] = {}
    for result in successes:
        if result.batch_requests is not None:
            batch_hist[result.batch_requests] = batch_hist.get(result.batch_requests, 0) + 1
    if batch_hist:
        print(
            "Observed batch_requests histogram: "
            + ", ".join(
                f"{batch_size}=>{count}" for batch_size, count in sorted(batch_hist.items())
            )
        )

    shape_bucket_hist: dict[str, int] = {}
    for result in successes:
        if result.shape_bucket_id:
            shape_bucket_hist[result.shape_bucket_id] = (
                shape_bucket_hist.get(result.shape_bucket_id, 0) + 1
            )
    if shape_bucket_hist:
        print(
            "Shape bucket distribution: "
            + ", ".join(
                f"{bucket_id}=>{count}"
                for bucket_id, count in sorted(shape_bucket_hist.items())
            )
        )

    exact_shape_hist: dict[str, int] = {}
    for result in successes:
        if result.exact_shape_homogeneous is None:
            continue
        key = "true" if result.exact_shape_homogeneous else "false"
        exact_shape_hist[key] = exact_shape_hist.get(key, 0) + 1
    if exact_shape_hist:
        print(
            "Exact shape homogeneous: "
            + ", ".join(
                f"{key}=>{count}" for key, count in sorted(exact_shape_hist.items())
            )
        )

    worker_hist: dict[str, int] = {}
    for result in successes:
        if result.worker_id:
            worker_hist[result.worker_id] = worker_hist.get(result.worker_id, 0) + 1
    if worker_hist:
        print(
            "Worker distribution: "
            + ", ".join(
                f"{worker_id}=>{count}"
                for worker_id, count in sorted(worker_hist.items())
            )
        )

    cache_hit_true = sum(1 for result in successes if result.clone_prompt_cache_hit is True)
    cache_hit_false = sum(
        1 for result in successes if result.clone_prompt_cache_hit is False
    )
    if cache_hit_true or cache_hit_false:
        print(
            f"Clone prompt cache hits: true=>{cache_hit_true}, false=>{cache_hit_false}"
        )

    emit_metric(
        "Batch requests",
        [float(r.batch_requests) for r in successes if r.batch_requests is not None],
    )
    emit_metric(
        "Batch target tokens",
        [
            float(r.batch_target_tokens)
            for r in successes
            if r.batch_target_tokens is not None
        ],
    )
    emit_metric(
        "Batch estimated memory mb",
        [
            r.batch_estimated_memory_mb
            for r in successes
            if r.batch_estimated_memory_mb is not None
        ],
    )
    emit_metric(
        "GPU utilization peak pct",
        [
            r.gpu_utilization_peak_pct
            for r in successes
            if r.gpu_utilization_peak_pct is not None
        ],
    )
    emit_metric(
        "GPU memory total mb",
        [
            r.gpu_memory_total_mb
            for r in successes
            if r.gpu_memory_total_mb is not None
        ],
    )
    emit_metric(
        "GPU memory used peak mb",
        [
            r.gpu_memory_used_peak_mb
            for r in successes
            if r.gpu_memory_used_peak_mb is not None
        ],
    )
    emit_metric(
        "GPU memory free before mb",
        [
            r.gpu_memory_free_before_mb
            for r in successes
            if r.gpu_memory_free_before_mb is not None
        ],
    )
    emit_metric(
        "GPU memory free after mb",
        [
            r.gpu_memory_free_after_mb
            for r in successes
            if r.gpu_memory_free_after_mb is not None
        ],
    )
    emit_metric(
        "GPU allocator allocated mb",
        [
            r.gpu_allocator_allocated_mb
            for r in successes
            if r.gpu_allocator_allocated_mb is not None
        ],
    )
    emit_metric(
        "GPU allocator reserved mb",
        [
            r.gpu_allocator_reserved_mb
            for r in successes
            if r.gpu_allocator_reserved_mb is not None
        ],
    )
    emit_metric(
        "GPU allocator peak allocated mb",
        [
            r.gpu_allocator_peak_allocated_mb
            for r in successes
            if r.gpu_allocator_peak_allocated_mb is not None
        ],
    )
    emit_metric(
        "GPU allocator peak reserved mb",
        [
            r.gpu_allocator_peak_reserved_mb
            for r in successes
            if r.gpu_allocator_peak_reserved_mb is not None
        ],
    )

    if failures:
        print("Failures:")
        for failure in failures[:10]:
            print(
                f"  request={failure.index} status={failure.http_status} error={failure.error}"
            )

    print(f"Detailed CSV: {csv_path}")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.requests <= 0:
        parser.error("--requests must be > 0")
    if args.concurrency <= 0:
        parser.error("--concurrency must be > 0")
    if args.launch_window_s < 0:
        parser.error("--launch-window-s must be >= 0")
    if args.mode == "clone" and not args.ref_audio:
        parser.error("--ref-audio is required for clone mode")

    rng = random.Random(args.seed)
    texts = [rng.choice(DEFAULT_TEXTS) for _ in range(args.requests)]
    if args.launch_window_s > 0:
        launch_offsets = sorted(
            rng.uniform(0.0, args.launch_window_s) for _ in range(args.requests)
        )
    else:
        launch_offsets = [0.0] * args.requests

    started = time.perf_counter()
    launch_base = time.perf_counter() + 0.05
    results: list[Optional[RequestResult]] = [None] * args.requests

    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {
            executor.submit(
                run_scheduled_request,
                args,
                index,
                text,
                launch_offset,
                launch_base,
            ): index
            for index, (text, launch_offset) in enumerate(
                zip(texts, launch_offsets), start=1
            )
        }
        for future in as_completed(futures):
            index = futures[future]
            results[index - 1] = future.result()

    total_wall_s = time.perf_counter() - started
    finalized = [result for result in results if result is not None]
    write_csv(args.csv, finalized)
    print_summary(
        results=finalized,
        total_wall_s=total_wall_s,
        requests=args.requests,
        concurrency=args.concurrency,
        launch_window_s=args.launch_window_s,
        csv_path=args.csv,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
