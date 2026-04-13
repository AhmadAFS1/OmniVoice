# Throughput Metrics

This document records the throughput and latency measurements collected during
the API-server testing session in this chat, along with the most important
serving lessons learned so far.

Important scope note:

- All measurements below are from `mode=design`
- These are not voice-cloning benchmarks
- Early phases below were collected on the pre-batching server
- Later phases were collected after the API micro-batcher was implemented
- The current server is no longer using the old global generate lock path for
  `/generate`

Relevant implementation detail:

- [api_server.py](/workspace/OmniVoice/omnivoice/cli/api_server.py)
- [batching.py](/workspace/OmniVoice/omnivoice/serving/batching.py)
- [omnivoice.py](/workspace/OmniVoice/omnivoice/models/omnivoice.py)

## Test Request

Most measurements in this document used:

```text
mode=design
text=This is a benchmark request for comparing RTX 3080 and RTX 3090 inference.
instruct=female, low pitch, british accent
guidance_scale=2.0
```

The later repeated benchmark additionally fixed:

```text
duration=6.2
postprocess_output=false
```

## High-Level Findings

- In the historical single-request baseline, RTX 3080 and RTX 3090 were nearly
  tied for this workload shape.
- After online API micro-batching was added, throughput improved materially,
  but one GPU was still far below the offered load in the `100 requests over 2s`
  benchmark.
- Under realistic 100-request concurrency, the main latency contributor became
  queue wait rather than one-request inference time.
- Increasing batch size from `8` to `16` reduced queueing, but also made each
  batch much slower. The net gain was small.
- Making batching more VRAM-aware allowed larger merged batches, but larger
  batches alone did not guarantee better latency or higher throughput.
- The practical goal is not simply “maximize VRAM.” The better target is:
  maximize throughput subject to a reasonable batch execution SLA.
- For this workload, a single GPU is currently sustaining only about
  `5.5-6.1 req/s`, far below an offered load of `50 req/s`.

## Phase 1: Early One-Off Runs

These runs were taken before the system was fully warmed up, so they are useful
as context but should not be treated as the cleanest comparison.

| GPU | Steps | Latency (ms) | Audio Duration (s) | RTF |
|---|---:|---:|---:|---:|
| RTX 3090 | 16 | 1624.62 | 5.950 | 0.2730 |
| RTX 3090 | 32 | 1035.52 | 5.580 | 0.1856 |
| RTX 3080 | 16 | 1369.36 | 6.090 | 0.2249 |
| RTX 3080 | 32 | 1071.84 | 5.730 | 0.1871 |

Notes:

- These results showed obvious warmup effects.
- `32-step` appearing faster than `16-step` in some early samples is a strong
  sign that startup or kernel warmup cost was still dominating the first run.

## Phase 2: Warmed One-Off Runs

These runs were collected after warming the servers more.

| GPU | Steps | Latency (ms) | Audio Duration (s) | RTF |
|---|---:|---:|---:|---:|
| RTX 3080 | 16 | 589.73 | 6.120 | 0.0964 |
| RTX 3080 | 32 | 1109.90 | 5.180 | 0.2143 |
| RTX 3090 | 16 | 684.53 | 5.260 | 0.1301 |
| RTX 3090 | 32 | 1032.26 | 5.470 | 0.1887 |

Notes:

- These were much more believable than the first cold-sensitive samples.
- The one-off warmed numbers still had varying audio durations, so they were
  not yet perfect apples-to-apples comparisons.

## Phase 3: 20-Run Fixed-Duration Benchmark

This was the cleanest benchmark collected so far.

Conditions:

- `mode=design`
- `duration=6.2`
- `postprocess_output=false`
- sequential request loop
- current API server with serialization lock

### Aggregate Summary

| GPU | Steps | Runs | Mean Latency (ms) | Median Latency (ms) | Min (ms) | Max (ms) | Latency StdDev (ms) | Mean RTF | Median RTF | Min RTF | Max RTF | RTF StdDev |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| RTX 3080 | 16 | 20 | 540.50 | 535.60 | 526.67 | 581.32 | 15.07 | 0.0872 | 0.0864 | 0.0849 | 0.0938 | 0.0024 |
| RTX 3080 | 32 | 20 | 1053.73 | 1047.37 | 1031.26 | 1116.26 | 23.54 | 0.1700 | 0.1689 | 0.1663 | 0.1800 | 0.0038 |
| RTX 3090 | 16 | 20 | 540.60 | 547.13 | 520.58 | 565.99 | 12.83 | 0.0872 | 0.0882 | 0.0840 | 0.0913 | 0.0021 |
| RTX 3090 | 32 | 20 | 1060.01 | 1065.86 | 1022.62 | 1105.74 | 22.84 | 0.1710 | 0.1719 | 0.1649 | 0.1783 | 0.0037 |

### Interpretation

- `16-step`: effectively tied
- `32-step`: effectively tied
- The differences are small enough that they look like noise-level variation
  under this current serving path
- The test currently reflects single-request latency much more than true
  throughput under queue pressure

Approximate throughput under this specific benchmark shape:

- `16-step`: about `1.85 requests/sec`
- `32-step`: about `0.95 requests/sec`

These are rough single-server, single-request-path numbers inferred from mean
latency, not a batched concurrency benchmark.

## Raw CSV: RTX 3080

```csv
host,steps,run,latency_ms,audio_duration_s,rtf
e9b27791c166,16,1,581.32,6.200,0.0938
e9b27791c166,32,1,1048.17,6.200,0.1691
e9b27791c166,16,2,535.65,6.200,0.0864
e9b27791c166,32,2,1046.57,6.200,0.1688
e9b27791c166,16,3,532.10,6.200,0.0858
e9b27791c166,32,3,1043.38,6.200,0.1683
e9b27791c166,16,4,536.00,6.200,0.0865
e9b27791c166,32,4,1042.74,6.200,0.1682
e9b27791c166,16,5,538.71,6.200,0.0869
e9b27791c166,32,5,1049.28,6.200,0.1692
e9b27791c166,16,6,535.55,6.200,0.0864
e9b27791c166,32,6,1064.91,6.200,0.1718
e9b27791c166,16,7,550.59,6.200,0.0888
e9b27791c166,32,7,1044.64,6.200,0.1685
e9b27791c166,16,8,536.95,6.200,0.0866
e9b27791c166,32,8,1100.91,6.200,0.1776
e9b27791c166,16,9,533.44,6.200,0.0860
e9b27791c166,32,9,1048.78,6.200,0.1692
e9b27791c166,16,10,535.89,6.200,0.0864
e9b27791c166,32,10,1040.27,6.200,0.1678
e9b27791c166,16,11,531.63,6.200,0.0857
e9b27791c166,32,11,1116.26,6.200,0.1800
e9b27791c166,16,12,534.87,6.200,0.0863
e9b27791c166,32,12,1049.25,6.200,0.1692
e9b27791c166,16,13,535.02,6.200,0.0863
e9b27791c166,32,13,1031.31,6.200,0.1663
e9b27791c166,16,14,580.48,6.200,0.0936
e9b27791c166,32,14,1049.68,6.200,0.1693
e9b27791c166,16,15,527.89,6.200,0.0851
e9b27791c166,32,15,1036.65,6.200,0.1672
e9b27791c166,16,16,526.67,6.200,0.0849
e9b27791c166,32,16,1102.79,6.200,0.1779
e9b27791c166,16,17,557.29,6.200,0.0899
e9b27791c166,32,17,1037.59,6.200,0.1674
e9b27791c166,16,18,537.01,6.200,0.0866
e9b27791c166,32,18,1040.17,6.200,0.1678
e9b27791c166,16,19,530.03,6.200,0.0855
e9b27791c166,32,19,1050.03,6.200,0.1694
e9b27791c166,16,20,532.99,6.200,0.0860
e9b27791c166,32,20,1031.26,6.200,0.1663
```

## Raw CSV: RTX 3090

```csv
host,steps,run,latency_ms,audio_duration_s,rtf
7d7877499228,16,1,565.99,6.200,0.0913
7d7877499228,32,1,1068.42,6.200,0.1723
7d7877499228,16,2,539.10,6.200,0.0870
7d7877499228,32,2,1073.69,6.200,0.1732
7d7877499228,16,3,547.60,6.200,0.0883
7d7877499228,32,3,1081.02,6.200,0.1744
7d7877499228,16,4,548.29,6.200,0.0884
7d7877499228,32,4,1063.35,6.200,0.1715
7d7877499228,16,5,535.11,6.200,0.0863
7d7877499228,32,5,1070.68,6.200,0.1727
7d7877499228,16,6,541.45,6.200,0.0873
7d7877499228,32,6,1073.87,6.200,0.1732
7d7877499228,16,7,547.75,6.200,0.0883
7d7877499228,32,7,1065.08,6.200,0.1718
7d7877499228,16,8,553.27,6.200,0.0892
7d7877499228,32,8,1053.74,6.200,0.1700
7d7877499228,16,9,548.90,6.200,0.0885
7d7877499228,32,9,1073.83,6.200,0.1732
7d7877499228,16,10,521.16,6.200,0.0841
7d7877499228,32,10,1057.31,6.200,0.1705
7d7877499228,16,11,547.92,6.200,0.0884
7d7877499228,32,11,1064.51,6.200,0.1717
7d7877499228,16,12,521.93,6.200,0.0842
7d7877499228,32,12,1028.18,6.200,0.1658
7d7877499228,16,13,550.82,6.200,0.0888
7d7877499228,32,13,1077.56,6.200,0.1738
7d7877499228,16,14,547.75,6.200,0.0883
7d7877499228,32,14,1066.63,6.200,0.1720
7d7877499228,16,15,536.98,6.200,0.0866
7d7877499228,32,15,1081.08,6.200,0.1744
7d7877499228,16,16,548.82,6.200,0.0885
7d7877499228,32,16,1105.74,6.200,0.1783
7d7877499228,16,17,546.67,6.200,0.0882
7d7877499228,32,17,1023.19,6.200,0.1650
7d7877499228,16,18,520.58,6.200,0.0840
7d7877499228,32,18,1026.41,6.200,0.1656
7d7877499228,16,19,520.98,6.200,0.0840
7d7877499228,32,19,1022.62,6.200,0.1649
7d7877499228,16,20,520.97,6.200,0.0840
7d7877499228,32,20,1023.30,6.200,0.1650
```

## Phase 4: First Realistic 100-Request Batch Test

This was the first benchmark taken after the online API micro-batcher was
implemented.

Conditions:

- `requests=100`
- `concurrency=100`
- `launch-window-s=2`
- `mode=design`
- `num_step=16`
- `guidance_scale=2.0`
- `duration=4.0`
- default micro-batcher behavior at the time, which effectively filled
  `8-request` merged batches

### Aggregate Summary

| GPU | Total Wall Time (s) | Effective Throughput (req/s) | Mean Latency (ms) | P50 Latency (ms) | P95 Latency (ms) | Mean Queue Wait (ms) | Mean Batch Exec (ms) | Batch Histogram |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| RTX 3080 | 17.13 | 5.84 | 5582.61 | 6186.81 | 6983.61 | 4347.87 | 1224.31 | `1=>1, 3=>3, 8=>96` |
| RTX 3090 | 18.06 | 5.54 | 5891.45 | 6547.43 | 7387.30 | 4582.09 | 1292.00 | `1=>1, 3=>3, 8=>96` |

### Interpretation

- Batching was clearly working.
- Almost every request landed in a full `8-request` merged batch.
- The dominant latency contributor was queue wait, not batch execution.
- Offered load was about `50 req/s`, but one GPU only sustained about
  `5.5-5.8 req/s`.
- This means queue growth was inevitable under this load shape even with
  batching enabled.

## Phase 5: Static Batch-Size Test (`bs=16`)

This test increased the request and prompt caps to `16` to see whether larger
merged batches would turn spare VRAM into more useful throughput.

Conditions:

- `requests=100`
- `concurrency=100`
- `launch-window-s=2`
- `mode=design`
- `num_step=16`
- `guidance_scale=2.0`
- `duration=4.0`
- `max_batch_requests=16`
- `max_batch_prompts=16`

### Aggregate Summary

| GPU | Total Wall Time (s) | Effective Throughput (req/s) | Mean Latency (ms) | Mean Queue Wait (ms) | Mean Batch Exec (ms) | Batch Histogram |
|---|---:|---:|---:|---:|---:|---|
| RTX 3080 | 16.49 | 6.06 | 5532.00 | 3192.28 | 2321.32 | `1=>1, 3=>3, 16=>96` |
| RTX 3090 | 18.33 | 5.45 | 6171.71 | 3540.04 | 2581.21 | `1=>1, 3=>3, 16=>96` |

### Interpretation

- Larger merged batches did reduce queue wait on both cards.
- But batch execution time almost doubled compared with the `8-request` run.
- Net outcome:
  - small improvement on the `3080`
  - slight regression on the `3090`
- This showed that “bigger batch” was not automatically “better throughput.”

## Phase 6: VRAM-Aware Batch Admission

After the first batch-size experiments, the batcher was reworked to use an
estimated incremental GPU memory budget instead of relying primarily on a
small fixed request cap.

Relevant implementation detail:

- [batching.py](/workspace/OmniVoice/omnivoice/serving/batching.py)
- [omnivoice.py](/workspace/OmniVoice/omnivoice/models/omnivoice.py)
- [api_server.py](/workspace/OmniVoice/omnivoice/cli/api_server.py)

What changed:

- default `max_batch_requests` / `max_batch_prompts` increased to `32`
- the server now estimates batch memory from:
  - padded sequence length
  - target token length
  - CFG duplication
  - logits / masks / scratch tensors
- the batcher now compares the estimated batch memory to a budget derived from:
  - current free CUDA memory
  - `gpu_memory_utilization`
  - `gpu_memory_reserve_mb`

Tested startup command:

```bash
cd /workspace/OmniVoice
export OMNIVOICE_LOG_LEVEL=INFO
.venv/bin/python -m omnivoice.cli.api_server \
  --model k2-fsa/OmniVoice \
  --device cuda \
  --ip 0.0.0.0 \
  --port 8002 \
  --no-asr \
  --batch-collect-ms 10 \
  --gpu-memory-utilization 0.90 \
  --gpu-memory-reserve-mb 1024
```

### Aggregate Summary

| GPU | Total Wall Time (s) | Effective Throughput (req/s) | Mean Latency (ms) | Mean Queue Wait (ms) | Mean Batch Exec (ms) | Batch Histogram |
|---|---:|---:|---:|---:|---:|---|
| RTX 3090 | 17.15 | 5.83 | 5746.89 | 3177.43 | 2541.29 | `1=>1, 3=>3, 16=>96` |
| RTX 3080 | 16.41 | 6.09 | 5579.30 | 1795.55 | 3756.21 | `1=>1, 8=>16, 19=>19, 32=>64` |

### Interpretation

- The VRAM-aware path was active because the two GPUs no longer followed the
  same batch pattern.
- The `3090` stayed mostly at `16-request` merged batches.
- The `3080` admitted batches up to `32`.
- That did **not** mean the `3080` suddenly became a better serving card.
- Instead:
  - queue wait fell
  - but batch execution time became much longer
- Net result: only a small throughput gain and no major latency breakthrough.

This is the clearest evidence so far that:

- maximizing admitted VRAM is not the same as maximizing throughput
- the next scheduler target should be “best throughput under a batch-exec SLA”
  rather than “largest batch that fits”

## Phase 7: Corrected GPU-Telemetry Benchmark

After fixing a stale-server issue on the `3090`, the benchmark was rerun with
the new GPU telemetry headers enabled on both cards.

Conditions:

- `requests=100`
- `concurrency=100`
- `launch-window-s=2`
- `mode=design`
- `num_step=16`
- `guidance_scale=2.0`
- `duration=4.0`
- VRAM-aware batcher enabled
- GPU telemetry headers enabled

### Aggregate Summary

| GPU | Total Wall Time (s) | Effective Throughput (req/s) | Mean Latency (ms) | Mean Queue Wait (ms) | Mean Batch Exec (ms) | Peak GPU Util Mean (%) | Peak GPU Mem Used Mean (MB) | Peak Torch Reserved Mean (MB) | Batch Histogram |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| RTX 3080 | 16.39 | 6.10 | 5572.49 | 1776.29 | 3759.27 | 99.32 | 6247.28 | 5966.40 | `1=>1, 8=>16, 19=>19, 32=>64` |
| RTX 3090 | 16.92 | 5.91 | 5725.11 | 1598.74 | 4080.97 | 99.00 | 6347.00 | 6006.00 | `1=>1, 8=>16, 19=>19, 32=>64` |

### Detailed 3080 Summary

- Total wall time: `16.39s`
- Effective throughput: `6.10 req/s`
- Latency ms:
  - mean `5572.49`
  - p50 `5973.71`
  - p95 `6707.69`
  - p99 `6762.94`
  - min `1369.50`
  - max `6806.28`
- Queue wait ms:
  - mean `1776.29`
  - p50 `1287.08`
  - p95 `5353.93`
  - p99 `5410.45`
  - min `10.34`
  - max `5453.27`
- Batch exec ms:
  - mean `3759.27`
  - p50 `4680.66`
  - p95 `4692.34`
  - p99 `4692.34`
  - min `1253.91`
  - max `4692.34`
- Batch requests:
  - mean `25.38`
  - p50 `32`
  - p95 `32`
  - p99 `32`
  - min `1`
  - max `32`
- Batch target tokens:
  - mean `2538.00`
  - p50 `3200.00`
  - p95 `3200.00`
  - p99 `3200.00`
  - min `100.00`
  - max `3200.00`
- Batch estimated memory mb:
  - mean `551.69`
  - p50 `695.86`
  - p95 `695.86`
  - p99 `695.86`
  - min `21.41`
  - max `695.86`
- GPU utilization peak pct:
  - mean `99.32`
  - p50 `100.00`
  - p95 `100.00`
  - p99 `100.00`
  - min `32.00`
  - max `100.00`
- GPU memory used peak mb:
  - mean `6247.28`
  - p50 `6287.00`
  - p95 `6287.00`
  - p99 `6287.00`
  - min `2315.00`
  - max `6287.00`
- GPU allocator peak allocated mb:
  - mean `3347.47`
  - p50 `3713.33`
  - p95 `3713.33`
  - p99 `3713.33`
  - min `2001.25`
  - max `3713.33`
- GPU allocator peak reserved mb:
  - mean `5966.40`
  - p50 `6006.00`
  - p95 `6006.00`
  - p99 `6006.00`
  - min `2046.00`
  - max `6006.00`

### Detailed 3090 Summary

- Total wall time: `16.92s`
- Effective throughput: `5.91 req/s`
- Latency ms:
  - mean `5725.11`
  - p50 `5771.16`
  - p95 `6582.55`
  - p99 `6598.63`
  - min `695.95`
  - max `6611.28`
- Queue wait ms:
  - mean `1598.74`
  - p50 `1302.72`
  - p95 `5194.75`
  - p99 `5207.61`
  - min `10.67`
  - max `5214.81`
- Batch exec ms:
  - mean `4080.97`
  - p50 `5084.73`
  - p95 `5187.95`
  - p99 `5187.95`
  - min `676.55`
  - max `5187.95`
- Batch requests:
  - mean `25.38`
  - p50 `32`
  - p95 `32`
  - p99 `32`
  - min `1`
  - max `32`
- Batch target tokens:
  - mean `2538.00`
  - p50 `3200.00`
  - p95 `3200.00`
  - p99 `3200.00`
  - min `100.00`
  - max `3200.00`
- Batch estimated memory mb:
  - mean `551.69`
  - p50 `695.86`
  - p95 `695.86`
  - p99 `695.86`
  - min `21.41`
  - max `695.86`
- GPU utilization peak pct:
  - mean `99.00`
  - p50 `100.00`
  - p95 `100.00`
  - p99 `100.00`
  - min `0.00`
  - max `100.00`
- GPU memory used peak mb:
  - mean `6347.00`
  - p50 `6347.00`
  - p95 `6347.00`
  - p99 `6347.00`
  - min `6347.00`
  - max `6347.00`
- GPU allocator peak allocated mb:
  - mean `3347.47`
  - p50 `3713.33`
  - p95 `3713.33`
  - p99 `3713.33`
  - min `2001.25`
  - max `3713.33`
- GPU allocator peak reserved mb:
  - mean `6006.00`
  - p50 `6006.00`
  - p95 `6006.00`
  - p99 `6006.00`
  - min `6006.00`
  - max `6006.00`

### Interpretation

- Both cards are now showing near-100% **peak** GPU utilization during merged
  batches.
- Both cards are reserving roughly `6.0 GB` of CUDA allocator memory during
  these runs.
- This is strong evidence that the current bottleneck is **not** simply
  “unused VRAM.”
- Instead, the current serving path appears to be hitting compute-path and/or
  single-in-flight-batch limits before VRAM capacity is exhausted.
- The `3090` has much more total VRAM available, but under the current model
  path it is not turning that extra memory into higher throughput.
- Same-GPU multi-worker serving is still worth testing, but it should now be
  treated as an experiment to reduce head-of-line blocking or exploit idle
  gaps between batches, not as an automatically correct “use all the VRAM”
  strategy.

## Current Practical Conclusions

For the current implementation and current benchmark shape:

- batching is working
- one GPU is still far below the offered load in the `100 requests over 2s`
  scenario
- queueing dominates latency under bursty high-concurrency load
- larger merged batches help until batch execution time grows too much
- the current code path is not yet efficient enough to turn the `3090`'s extra
  VRAM into a decisive latency advantage
- the current server still has one in-flight merged batch per process
- optimizing for raw VRAM usage alone is the wrong target
- the new telemetry strongly suggests the current workload is compute-bound
  during batches rather than memory-capacity-bound

The best next-step hypotheses are:

- adaptive batch sizing using both memory headroom and observed batch execution
  time
- possible multi-replica same-GPU serving on the `3090`, but as an experiment
  rather than the default assumption
- further stage-1 efficiency improvements
- more stage-2 decode batching
- separate optimization passes for voice cloning

## Recommended Next Measurements

To keep future experiments comparable, use the following benchmark family:

```bash
.venv/bin/python scripts/benchmark_api_batching.py \
  --requests 100 \
  --concurrency 100 \
  --launch-window-s 2 \
  --mode design \
  --num-step 16 \
  --guidance-scale 2.0 \
  --duration 4.0
```

And when testing server-side changes, always record:

- total wall time
- effective throughput
- mean / p50 / p95 latency
- mean queue wait
- mean batch exec
- batch histogram
- `/health` batching stats

This document should be extended before changing multiple knobs at once so the
serving history stays interpretable.
