#!/usr/bin/env python3

"""Small profiling helpers for OmniVoice inference instrumentation."""

from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager
from typing import Iterator, MutableMapping, Optional

import torch

logger = logging.getLogger(__name__)

_NVTX_ENABLED = os.getenv("OMNIVOICE_ENABLE_NVTX", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}


def add_timing(
    metrics: Optional[MutableMapping[str, float]],
    key: str,
    elapsed_ms: float,
) -> None:
    """Accumulate a millisecond timing value into a metrics mapping."""

    if metrics is None:
        return
    metrics[key] = metrics.get(key, 0.0) + float(elapsed_ms)


@contextmanager
def timed_stage(
    metrics: Optional[MutableMapping[str, float]],
    key: str,
    nvtx_label: Optional[str] = None,
) -> Iterator[None]:
    """Measure a code block and optionally emit an NVTX range."""

    started = time.perf_counter()
    pushed_nvtx = False
    if nvtx_label and _NVTX_ENABLED and torch.cuda.is_available():
        try:
            torch.cuda.nvtx.range_push(nvtx_label)
            pushed_nvtx = True
        except Exception:
            logger.debug("Failed to push NVTX range %s", nvtx_label, exc_info=True)
    try:
        yield
    finally:
        if pushed_nvtx:
            try:
                torch.cuda.nvtx.range_pop()
            except Exception:
                logger.debug("Failed to pop NVTX range %s", nvtx_label, exc_info=True)
        add_timing(metrics, key, (time.perf_counter() - started) * 1000.0)
