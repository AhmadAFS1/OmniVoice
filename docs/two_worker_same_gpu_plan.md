# Two-Worker Same-GPU Plan

This document describes the implementation plan for running **two OmniVoice
workers on the same GPU**, starting with the `RTX 3090` test environment.

The goal is to reduce head-of-line blocking and improve request latency under
load by allowing **two smaller in-flight batches** on one GPU instead of one
very large in-flight batch per process.

This is an execution plan, not just a brainstorming note.

## Current Implementation Status

This branch now includes a first working version of the same-GPU multi-worker
design described below.

Implemented pieces:

- `omnivoice/serving/service.py`
  - extracted the request-to-audio generation path into a reusable local
    service object
- `omnivoice/serving/multiworker.py`
  - added a parent router backend that starts multiple local worker processes
    on the same CUDA device
  - routes requests by least pending worker, with EWMA tie-breaking from recent
    batch timings
  - tracks per-worker pending counts, failures, and last-response metrics
- `omnivoice/cli/api_server.py`
  - added backend selection between direct mode and same-GPU worker mode
  - added `--same-gpu-workers` / `--gpu-workers`
  - added `--api-thread-limit` so the HTTP layer can actually hold large
    concurrent bursts while requests wait on worker responses
- `scripts/benchmark_api_batching.py`
  - records `worker_id`, `worker_pid`, and worker VRAM headers
  - prints a worker-distribution summary so we can confirm load is split

Current worker behavior:

- each worker loads its own OmniVoice model replica on the same GPU
- each worker keeps its own existing `GenerationBatcher`
- each worker uses a local request thread pool so concurrent requests inside a
  worker can still merge into one inference batch
- request and batch logs now include worker IDs, PIDs, and extra VRAM fields
  such as free-before/free-after and allocator reserved memory

Recommended first launch command:

```bash
.venv/bin/python -m omnivoice.cli.api_server \
  --device cuda:0 \
  --same-gpu-workers 2 \
  --api-thread-limit 256 \
  --batch-collect-ms 10 \
  --max-batch-requests 12 \
  --max-batch-prompts 12
```

Recommended first benchmark:

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

## Latest Benchmark Findings

The implementation is now in place, and we have enough measurements to draw a
few concrete conclusions about what the current same-GPU worker pool can and
cannot do on the `RTX 3090`.

Important benchmark context:

- server mode: `--same-gpu-workers 2`
- device: `cuda:0`
- concurrency test: `100 requests` over a `2s` launch window
- benchmark script: [scripts/benchmark_api_batching.py](/workspace/OmniVoice/scripts/benchmark_api_batching.py:78)
- all results below achieved `100/100` success with balanced worker routing

### Design Mode: Per-Worker Batch Cap = 12

Command:

```bash
.venv/bin/python -m omnivoice.cli.api_server \
  --device cuda:0 \
  --same-gpu-workers 2 \
  --api-thread-limit 256 \
  --batch-collect-ms 10 \
  --max-batch-requests 12 \
  --max-batch-prompts 12
```

Observed benchmark summary:

| Metric | Result |
|---|---:|
| Total wall time | `17.77s` |
| Effective throughput | `5.63 req/s` |
| Mean latency | `9902 ms` |
| Mean queue wait | `6170 ms` |
| Mean batch exec | `3717 ms` |
| Dominant batch size | `12` |
| Peak GPU memory used | `~10.7 GB` |
| Worker split | `worker-1=>49`, `worker-2=>51` |

Interpretation:

- the router is distributing requests correctly
- batches are filling to the configured cap almost every time
- VRAM usage increased compared with the single-process path
- the GPU is already effectively compute-saturated at `~100%` utilization

### Design Mode: Per-Worker Batch Cap = 24

Command:

```bash
.venv/bin/python -m omnivoice.cli.api_server \
  --device cuda:0 \
  --same-gpu-workers 2 \
  --api-thread-limit 256 \
  --batch-collect-ms 10 \
  --max-batch-requests 24 \
  --max-batch-prompts 24
```

Observed benchmark summary:

| Metric | Result |
|---|---:|
| Total wall time | `17.40s` |
| Effective throughput | `5.75 req/s` |
| Mean latency | `11536 ms` |
| Mean queue wait | `4253 ms` |
| Mean batch exec | `7258 ms` |
| Dominant batch size | `24` |
| Peak GPU memory used | `~13.2 GB` |
| Worker split | `worker-1=>50`, `worker-2=>50` |

Interpretation:

- raising the batch cap from `12` to `24` did increase VRAM usage
- throughput improved only marginally: `5.63 -> 5.75 req/s`
- batch execution time nearly doubled: `~3.7s -> ~7.3s`
- queue wait improved, but overall request latency got worse

Main lesson:

- the batch cap is binding, but it is not the main bottleneck
- larger merged batches on this workload increase `batch_exec_ms` nearly as fast
  as they increase batch size
- that means the GPU is still compute-bound, not merely “under-filled”

### Clone Mode: Warmed Prompt Cache, Generated Reference Clip

To measure warmed clone mode instead of cold prompt-creation overhead, a design
mode clip was generated first and reused as the clone reference:

- audio: [bench_assets/generated_ref.wav](/workspace/OmniVoice/bench_assets/generated_ref.wav)
- transcript: [bench_assets/generated_ref.txt](/workspace/OmniVoice/bench_assets/generated_ref.txt)
- prompt duration: `6.2s`

Clone benchmark command:

```bash
.venv/bin/python scripts/benchmark_api_batching.py \
  --url http://127.0.0.1:8002/generate \
  --requests 100 \
  --concurrency 100 \
  --launch-window-s 2 \
  --mode clone \
  --ref-audio /workspace/OmniVoice/bench_assets/generated_ref.wav \
  --ref-text "Hello, this is a generated reference voice for OmniVoice clone mode benchmarking. We are using this clip to measure throughput under concurrent clone requests." \
  --num-step 16 \
  --guidance-scale 2.0 \
  --duration 4.0 \
  --csv batching-clone-generated-ref.csv
```

Observed benchmark summary:

| Metric | Result |
|---|---:|
| Total wall time | `35.62s` |
| Effective throughput | `2.81 req/s` |
| Mean latency | `23361 ms` |
| Mean queue wait | `10213 ms` |
| Mean batch exec | `13107 ms` |
| Dominant batch sizes | mixed: `24`, `19`, `16`, `8`, `6`, `2`, `1` |
| Peak GPU memory used | `~21.9 GB` |
| Worker split | `worker-1=>50`, `worker-2=>50` |

Interpretation:

- clone mode is much heavier than design mode on this server
- prompt caches were already warmed, so this result mostly reflects the actual
  conditioned generation path, not one-time prompt preparation
- the `6.2s` reference clip and its transcript materially increase conditioning
  length and per-item work
- clone batches are harder to keep perfectly packed, and even full batches are
  far more expensive than design-mode batches

### Phase 1 Sweep Result: Design Mode With `num_step=8`, `guidance_scale=1.0`

This was the first concrete Phase 1 throughput sweep result after Phase 0
instrumentation was added.

Important caveat:

- this result is best interpreted as a **throughput ceiling test**, not
  automatically as the preferred production preset
- the product preference is still to stay closer to `num_step=16` with
  reasonably strong guidance for quality

Benchmark command:

```bash
.venv/bin/python scripts/benchmark_api_batching.py \
  --url http://127.0.0.1:8002/generate \
  --requests 100 \
  --concurrency 100 \
  --launch-window-s 2 \
  --mode design \
  --num-step 8 \
  --guidance-scale 1.0 \
  --duration 4.0
```

Observed benchmark summary:

| Metric | Result |
|---|---:|
| Total wall time | `8.68s` |
| Effective throughput | `11.53 req/s` |
| Mean latency | `4910 ms` |
| P50 latency | `4855 ms` |
| P95 latency | `7183 ms` |
| Mean queue wait | `1709 ms` |
| Mean batch exec | `3185 ms` |
| Dominant batch sizes | mixed: `24`, `22`, `21`, `18`, `7`, `6`, `1` |
| Peak GPU memory used | `~14.0 GB` |
| Worker split | `worker-1=>50`, `worker-2=>50` |

Phase 0 timing breakdown:

| Timing Metric | Result |
|---|---:|
| `batch_generate_tokens_ms` mean | `2830 ms` |
| `batch_decode_tokens_ms` mean | `354 ms` |
| `model_iterative_total_ms` mean | `2827 ms` |
| `model_iterative_forward_ms` mean | `1775 ms` |
| `model_iterative_scoring_ms` mean | `686 ms` |
| `model_iterative_update_ms` mean | `233 ms` |
| `model_iterative_setup_ms` mean | `132 ms` |
| `wav_serialize_ms` mean | `13 ms` |
| `request_prep_ms` mean | `< 1 ms` |

Interpretation:

- throughput nearly doubled versus the `num_step=16`, `guidance_scale=2.0`
  design-mode baseline on this branch
- mean batch execution time dropped from roughly `7.31s` to `3.19s`
- the forward pass is still the dominant cost, but both forward and scoring
  costs dropped substantially with the lower-step / lower-guidance preset
- decode and request-side overhead remain small relative to token generation

Why this result matters:

- it confirms that `num_step` and `guidance_scale` are major compute
  multipliers in practice, not just in theory
- it validates the Phase 1 strategy of testing product knobs before deeper
  kernel or masking rewrites

Why this does **not** end the optimization work:

- the preferred production quality target is still closer to `num_step=16`
  with decent guidance
- if that quality bar must be preserved, the remaining throughput gains must
  come from later phases:
  - dense-mask elimination
  - shape bucketing
  - CUDA Graphs
  - Triton fused kernels

### Phase 2 Result: Dense Mask Elimination With Flex Attention

Phase 2 replaced the dense iterative attention mask with a batch-aware
`BlockMask`. Because Qwen3 was still using the `sdpa` attention backend, the
model path was also updated to switch to `flex_attention` whenever a
`BlockMask` is passed into `forward(...)`.

That change introduced a very important operational behavior:

- the **first** large benchmark after server startup pays a significant compile
  and warmup cost
- subsequent runs are materially faster
- to avoid exposing that cold-start penalty to real traffic, the worker startup
  path now runs a synthetic CUDA warmup batch that compiles the
  `flex_attention` path before the worker is marked ready

Benchmark command used for both measurements:

```bash
.venv/bin/python scripts/benchmark_api_batching.py \
  --url http://127.0.0.1:8002/generate \
  --requests 100 \
  --concurrency 100 \
  --launch-window-s 2 \
  --mode design \
  --num-step 16 \
  --guidance-scale 2.0 \
  --duration 4.0
```

Observed benchmark summary:

| Metric | Pre-Phase-2 Baseline | Phase-2 Cold Run | Phase-2 Warm Run |
|---|---:|---:|---:|
| Total wall time | `17.36s` | `31.45s` | `15.91s` |
| Effective throughput | `5.76 req/s` | `3.18 req/s` | `6.29 req/s` |
| Mean latency | `11487 ms` | `20426 ms` | `9928 ms` |
| Mean queue wait | `4154 ms` | `15166 ms` | `4093 ms` |
| Mean batch exec | `7310 ms` | `5222 ms` | `5814 ms` |
| `model_iterative_forward_ms` mean | `4212 ms` | `2781 ms` | `3378 ms` |
| `model_iterative_scoring_ms` mean | `1997 ms` | `977 ms` | `1533 ms` |
| Peak GPU memory used | `~13.3 GB` | `~12.9 GB` | `~15.6 GB` |

Interpretation:

- the cold-run regression was dominated by first-use `flex_attention` compile
  overhead, not steady-state throughput
- the warmed Phase 2 path improved steady-state throughput versus the
  pre-Phase-2 baseline: `5.76 -> 6.29 req/s`
- warmed mean latency improved as well: `11487 ms -> 9928 ms`
- the biggest measured win landed in the forward pass:
  `model_iterative_forward_ms` dropped from `4212 ms` to `3378 ms`
- this validates the dense-mask removal work, but it also means worker startup
  warmup is mandatory for production-like behavior

### Phase 3-5 Result: Shape Bucketing, Exact-Shape CUDA Graphs, And Triton Groundwork

The next implementation slice added three connected changes:

- Phase 3 shape bucketing in the batcher, with an exact-shape-first selection
  pass and a coarse bucket fallback pass
- Phase 4 experimental CUDA Graph replay for homogeneous exact-shape batches
- Phase 5 low-risk Triton groundwork by compiling the scoring path with
  Inductor/Triton when possible, but automatically falling back to eager
  execution if that path fails at runtime

Benchmark command:

```bash
.venv/bin/python scripts/benchmark_api_batching.py \
  --url http://127.0.0.1:8002/generate \
  --requests 100 \
  --concurrency 100 \
  --launch-window-s 2 \
  --mode design \
  --num-step 16 \
  --guidance-scale 2.0 \
  --duration 4.0
```

Observed benchmark summary:

| Metric | Clean Pre-Phase-3 Baseline | Phase 3-5 Run |
|---|---:|---:|
| Total wall time | `20.62s` | `19.18s` |
| Effective throughput | `4.85 req/s` | `5.21 req/s` |
| Mean latency | `14734 ms` | `12022 ms` |
| Mean queue wait | `6745 ms` | `6883 ms` |
| Mean batch exec | `7953 ms` | `5108 ms` |
| `model_iterative_forward_ms` mean | `2801 ms` | `1493 ms` |
| `model_iterative_update_ms` mean | `3861 ms` | `2661 ms` |
| Peak GPU memory used | `~13.0 GB` | `~13.7 GB` |

Additional Phase 3-5 telemetry from the validated run:

- shape bucket distribution:
  - `short_mixed|cfg|p1|t112|c128 => 83`
  - `short_mixed|cfg|p1|t112|c160 => 17`
- exact-shape homogeneous responses: `true => 18`, `false => 82`
- CUDA Graph telemetry appeared in live responses for homogeneous batches:
  - `model_iterative_graph_capture_ms mean ~= 944 ms`
  - `model_iterative_graph_replay_ms mean ~= 7.7 ms`

Interpretation:

- Phase 3 is working as intended:
  - requests are now landing in a small number of stable bucket classes
  - the benchmark script surfaces those bucket IDs directly
- Phase 4 is working in a limited but real way:
  - homogeneous batches are now graph-capturable
  - replay removes most of the forward-call overhead for those exact shapes
  - graph replay is not yet the dominant throughput win because only a subset
    of batches in this benchmark are fully homogeneous
- Phase 5 is only groundwork on this branch:
  - the compiled scoring path currently hits an Inductor/TLS assertion in this
    server context
  - the runtime now catches that failure once, disables the compiled path, and
    falls back to eager scoring without failing the request

Current practical conclusion:

- the combined Phase 3-5 slice improves the immediate clean baseline, mainly by
  reducing `batch_exec_ms` and cutting the forward path again
- the main remaining bottleneck is still the update stage inside
  `_generate_iterative(...)`
- the Triton work here should be considered a safe starting point, not the full
  custom-kernel Phase 5 described below

### What These Results Tell Us

The same-GPU 2-worker implementation is functioning correctly:

- request routing is balanced
- both workers stay busy
- VRAM usage scales upward when two model replicas and larger batches are used
- worker-local batching and telemetry are working as intended

However, the results also show clear limitations:

- this `RTX 3090` setup is already compute-saturated on the tested workload
- simply increasing the request cap does **not** unlock a large throughput gain
- for design mode, larger batches mostly trade lower queue wait for longer
  `batch_exec_ms`
- for clone mode, a long reference prompt makes throughput much worse and pushes
  memory usage close to the card limit

Current practical conclusions:

- `24` is slightly better than `12` for raw design-mode throughput, but only by
  a very small margin
- `12` is better than `24` for design-mode latency
- clone mode with a `6.2s` reference clip is not a good fit for the
  `100 requests in 2s` target on a single `RTX 3090`
- removing the batch limit entirely is unlikely to help much and is more likely
  to create very large batches with worse tail latency and higher OOM risk

## Revised Roadmap For `100 Requests In 2 Seconds` With High Quality Output

This is the current go-forward plan and should be treated as the primary
roadmap from this point onward.

The target is:

- `100 concurrent requests`
- finishing in about `2s`
- at the preferred high-quality design-mode preset:
  - `num_step=16`
  - `guidance_scale=2.0`

That target implies an aggregate throughput of about:

- `50 req/s`

Current reference points on this branch:

| Scenario | Throughput | Gap To `50 req/s` |
|---|---:|---:|
| Best warmed high-quality run so far | `6.29 req/s` | `~7.95x short` |
| Current Phase 3-5 branch | `5.21 req/s` | `~9.60x short` |
| Lower-quality sweep (`num_step=8`, `guidance_scale=1.0`) | `11.53 req/s` | `~4.34x short` |

This leads to an important planning conclusion:

- on a single `RTX 3090`, software-only optimization is very unlikely to reach
  `50 req/s` while keeping `num_step=16` and `guidance_scale=2.0`
- the realistic strategy is:
  1. raise the single-GPU ceiling as much as possible
  2. then add more total GPU compute to actually hit the product target

### What The Current Data Says

At high-quality settings, the dominant remaining costs are still inside
`_generate_iterative(...)`:

- the forward path improved substantially after Phase 2 and again after the
  Phase 3-5 slice
- the current biggest remaining timed stage is the update path:
  - `model_iterative_update_ms ~= 2661 ms`
- the scoring path is still meaningful
- CUDA Graph replay is only helping a subset of traffic because only
  `18 / 100` requests in the validated run were part of exact-shape homogeneous
  batches

What is *not* the main problem right now:

- HTTP thread limits
- request validation / prep
- WAV serialization
- simply increasing `max_batch_requests`
- removing `max_batch_prompts`

Those knobs can affect latency or burst admission, but they are not the reason
the server is still about an order of magnitude short of the throughput target.

### New Phase Sequence

The next phases should be executed in this order.

### Phase A: Re-Baseline And Remove Self-Inflicted Regressions

Goal:

- make sure we are optimizing from the best current serving profile, not from a
  partially regressed experimental branch

Why this phase exists:

- the best warmed high-quality run so far is still `6.29 req/s`
- the current Phase 3-5 stack is more graph-friendly and better structured, but
  only at `5.21 req/s`
- before going deeper, we need to quantify how much throughput is being lost to:
  - stricter bucketing
  - reduced average batch size
  - graph capture overhead on cold or partially warm paths

Work items:

- run a controlled ablation matrix on the current branch:
  - current stack
  - `--no-shape-bucketing`
  - `--no-cuda-graphs`
  - `--no-triton-score-fusion`
- sweep current design-mode batch caps at `12`, `16`, `20`, `24`
- record:
  - average batch size
  - exact-shape homogeneous rate
  - graph capture rate
  - graph replay rate
- choose one clean default benchmark profile for later phases

Success criteria:

- we either recover or beat the `6.29 req/s` warmed high-quality result
- or we have clear, quantified evidence about which experimental feature is
  trading throughput for structure

Expected impact:

- low direct throughput gain
- very high decision value

### Phase B: Optimize The Update And Scoring Path

Goal:

- attack the largest remaining hot path after the forward pass improvements

Why this is next:

- the current branch already cut `model_iterative_forward_ms` down to
  `~1493 ms`
- the update stage is still about `~2661 ms`
- if we do not reduce update/scoring cost, more graph work on the forward path
  alone will not get us close to the target

Work items:

- keep more state flattened across steps instead of reshaping and reindexing
  large tensors every iteration
- reduce or eliminate repeated `permute(...)` and gather work when extracting
  conditional and unconditional target logits
- precompute per-step helper tensors outside the decode loop where possible
- tighten the masked top-k and scatter-write path
- prototype a fused CUDA/Triton kernel for:
  - score transformation
  - masked top-k selection
  - sparse token writeback
- only consider confidence-score approximations if parity on the saved quality
  reference set is strong

Success criteria:

- `model_iterative_update_ms` becomes smaller than the forward path
- combined `model_iterative_scoring_ms + model_iterative_update_ms` drops by at
  least `2x` on the `16 / 2.0` design benchmark

Expected impact:

- high

### Phase A Result: Current Stack Ablations

Phase A was run on the current high-quality design-mode benchmark:

```bash
.venv/bin/python scripts/benchmark_api_batching.py \
  --url http://127.0.0.1:8002/generate \
  --requests 100 \
  --concurrency 100 \
  --launch-window-s 2 \
  --mode design \
  --num-step 16 \
  --guidance-scale 2.0 \
  --duration 4.0
```

Measured one-GPU serving variants:

| Variant | Throughput | Mean Latency | Mean Batch Exec |
|---|---:|---:|---:|
| Current stack | `5.68 req/s` | `11289 ms` | `5053 ms` |
| `--no-shape-bucketing` | `5.48 req/s` | `12096 ms` | `6856 ms` |
| `--no-cuda-graphs` | `5.46 req/s` | `11802 ms` | `5302 ms` |
| `--no-triton-score-fusion` | `5.49 req/s` | `11822 ms` | `5338 ms` |

Interpretation:

- the current stack is the best of the tested serving toggles on this branch
- shape bucketing is helping even though it slightly reduces average batch size
- CUDA Graph replay is helping enough to justify keeping it enabled
- the current Triton scoring toggle is not a large gain, but disabling it still
  did not improve the end-to-end benchmark

Current Phase A decision:

- keep:
  - shape bucketing enabled
  - CUDA Graph replay enabled
  - the existing Triton scoring toggle enabled
- do not spend more time trying to recover throughput by turning serving knobs
  off

### Phase B Result: First Update-Path Experiments

Two first-pass update-path optimizations were prototyped and benchmarked:

1. a flatter linear-index update path with more precomputed indexing metadata
2. a compiled helper for the selection/update substep using
   `torch.compile(..., options={\"triton.cudagraphs\": False})`

Observed outcomes:

- the flatter linear-index rewrite did **not** beat the current stack
- the compiled helper reduced `model_iterative_update_ms` in isolation, but it
  hurt end-to-end throughput in the real server because compile and shape
  overhead outweighed the gain
- both experiments were reverted from the serving hot path

What stayed:

- the branch now keeps finer-grained update telemetry so later iterations can
  separate:
  - `model_iterative_update_select_ms`
  - `model_iterative_update_write_ms`

Current Phase B conclusion:

- the update path is still the right bottleneck to attack
- but the next improvement likely needs to be more structural than a small
  indexing rewrite
- the most promising next Phase B directions are:
  - a fused or compiled selection/write helper with stable shapes
  - broader graph capture that includes more of the per-step body
  - a custom Triton kernel for masked top-k plus sparse token writeback

### Latest High-Quality Rerun After Phase B Telemetry Split

A later rerun on the same high-quality benchmark produced a newer
`batching-bench.csv` with the per-update substage timings enabled:

- `model_iterative_update_ms`
- `model_iterative_update_select_ms`
- `model_iterative_update_write_ms`

Important note:

- the pasted console output did not include the top summary lines
  (`Total wall time`, `Effective throughput`)
- however, the CSV still lets us compare the run against the previous
  high-quality Phase A baseline

Observed metrics from the CSV:

| Metric | Phase A Baseline | Latest Rerun |
|---|---:|---:|
| Mean latency | `11289 ms` | `9687 ms` |
| Mean queue wait | `6205 ms` | `4590 ms` |
| Mean batch exec | `5053 ms` | `5067 ms` |
| Mean batch requests | `18.92` | `17.82` |
| Peak GPU memory used mean | `14064 MB` | `13273 MB` |
| Exact-shape homogeneous | `19` | `19` |

Throughput implication:

- from the CSV, the maximum observed `local_wall_ms` was about `14582 ms`
- with a `2s` launch window, total wall time for the run must therefore be at
  most about `16.58s`
- that implies a throughput of at least about `6.03 req/s`
- this is better than the earlier Phase A baseline of `5.68 req/s`

New update substage timings from that rerun:

| Timing Metric | Result |
|---|---:|
| `model_iterative_update_ms` mean | `2953 ms` |
| `model_iterative_update_select_ms` mean | `6.6 ms` |
| `model_iterative_update_write_ms` mean | `34.8 ms` |

Interpretation:

- this rerun appears to be a real improvement over the earlier `5.68 req/s`
  Phase A baseline
- latency and queueing improved materially
- batch execution time stayed roughly flat
- average batch size was slightly smaller, so the improvement is not coming
  from simply packing more requests into each launch
- the new substage timings strongly suggest that the expensive part of the
  update block is **not** the final sparse write itself
- instead, most of the true update cost is still tied to the selection /
  synchronization path around masked top-k and data-dependent indexing

### Phase C: Increase Shape Homogeneity And Graph Hit Rate

Goal:

- make graph replay apply to most traffic, not just the exact-shape minority

Why this matters:

- the current graph path is real, but only `18%` of responses in the validated
  run were exact-shape homogeneous
- exact-shape-only capture is too narrow to move total throughput enough

Work items:

- move from exact-shape-only graph capture toward bucket-level padded-shape
  capture
- capture by dominant padded bucket classes such as:
  - target-token bucket
  - conditioning-token bucket
  - CFG on/off
  - lane
- pad requests to bucket maxima where the extra compute cost is smaller than the
  benefit from graph reuse
- optionally run separate queue lanes per dominant bucket instead of mixing them
  inside the same worker queue
- expand worker startup warmup to compile the dominant bucket shapes, not just a
  single synthetic warmup

Success criteria:

- most requests fall into a small number of replayable bucket classes
- graph-replayable batches become the common case instead of the exception
- graph capture becomes rare after startup warmup

Expected impact:

- high

### Phase D: Expand Graph Scope Beyond The Forward Call

Goal:

- replay more of the iterative step body, not just the transformer forward

Why this matters:

- today the graph scope mostly removes forward-call overhead
- the remaining update and scoring work still runs outside the graph
- after Phase B and Phase C, the step body should be static enough to broaden
  graph coverage

Work items:

- capture the full iterative step for fixed padded buckets when shapes allow
- evaluate full-loop capture for homogeneous fixed-length workloads
- keep static buffers for:
  - logits
  - scores
  - selected indices
  - token writeback state
- preserve eager fallbacks for mixed-shape traffic

Success criteria:

- graph replay covers most of the per-step hot path for stable buckets
- Python overhead in the iterative loop becomes negligible in the hot case

Expected impact:

- medium to high

### Phase E: Real Triton Kernel Fusion

Goal:

- cut bandwidth-heavy transformer and scoring overhead with custom kernels, not
  just opportunistic `torch.compile(...)`

Why this is still needed:

- the current Phase 5 slice is only safe groundwork
- the compiled scoring path still falls back to eager in this server context
- if we want another material single-GPU gain, we need true fused kernels in
  the model hot path

Work items:

- implement or port Triton kernels for:
  - RMSNorm
  - SwiGLU
  - fused residual + norm
- add a fused CFG scoring kernel for the design-mode hot path
- benchmark against the saved multilingual quality references before enabling by
  default
- keep all kernel patches behind feature flags

Success criteria:

- stable warm serving with no eager fallback in the intended path
- measurable end-to-end gain on the `16 / 2.0` benchmark beyond graphing alone

Expected impact:

- medium to high

### Phase F: Scale Out To More Total GPU Compute

Goal:

- translate the single-GPU ceiling gains into the actual product throughput
  target

Why this phase is mandatory:

- even an aggressive single-GPU optimization outcome is unlikely to reach
  `50 req/s` on one `RTX 3090`
- the target almost certainly requires more total GPU compute in addition to
  software wins

Work items:

- generalize the backend from:
  - `2 workers on 1 GPU`
  - to `N workers across M GPUs`
- keep the same batching and bucketing rules across replicas
- add queue telemetry per GPU and per worker
- benchmark aggregate throughput on:
  - `2 GPUs`
  - `4 GPUs`
  - or stronger GPU classes if available

Success criteria:

- aggregate throughput at high quality reaches or exceeds `50 req/s`
- p95 latency remains operationally acceptable under the `100 requests / 2s`
  burst

Expected impact:

- essential for the final target

### Phase G: Clone-Mode Specialization

Goal:

- keep clone traffic from destroying the design-mode throughput envelope

Work items:

- give clone requests distinct batch lanes and caps
- enforce or strongly recommend shorter reference clips for high-throughput
  workloads
- surface prompt-cache hit/miss telemetry more clearly
- build separate quality and throughput targets for clone mode

Success criteria:

- clone mode becomes predictable and isolated
- design-mode traffic does not regress when clone traffic is present

Expected impact:

- medium operational gain

### Phase H: TensorRT Evaluation

Goal:

- evaluate TensorRT only after the path is static enough to justify the effort

Why this is late:

- the current serving path still contains too much dynamic behavior
- TensorRT will be easier to assess after:
  - Phase B
  - Phase C
  - Phase D
  - and at least part of Phase E

Work items:

- run Torch-TensorRT `dryrun=True` on the narrowed hot path
- test coverage on the dominant padded buckets
- compare build/maintenance cost against the Triton path

Success criteria:

- high coverage on the real serving hot path
- clear evidence that TensorRT outperforms the Triton-based alternative enough
  to justify adopting it

Expected impact:

- unknown until the path is more static

### What We Should Not Prioritize Next

The current data does **not** support spending major engineering time next on:

- `api_thread_limit`
- removing `max_batch_prompts`
- unbounded batch sizes
- request-validation overhead
- WAV serialization
- clone-mode work before design mode is faster

Those may still matter operationally, but they are not the highest-ROI path to
the throughput target.

### Practical Execution Order

The recommended order from today forward is:

1. Phase A
2. Phase B
3. Phase C
4. Phase D
5. Phase E
6. Phase F
7. Phase G
8. Phase H

Operational rule:

- do not start the next phase until the current one has:
  - a controlled benchmark comparison
  - a written conclusion in this document
  - a clear decision about whether the phase stays enabled by default

Most important planning rule:

- treat single-GPU optimization and multi-GPU scale-out as **separate**
  objectives
- the first raises the ceiling
- the second is what is most likely to actually deliver `100 requests in 2s`
  at the desired quality bar

### Recommended Next Tests

The most useful next measurements are:

1. design mode with per-worker caps around `12`, `16`, and `20` to find the
   best throughput-latency tradeoff
2. clone mode with a much shorter reference clip, ideally `2.5-3.5s`
3. clone mode with a shorter transcript that matches the shorter prompt
4. clone mode with smaller per-worker caps such as `12` or `16`
5. optional clone-mode test with `denoise=false`

Operational takeaway:

- if the product target is truly `100 concurrent requests finishing in ~2s`,
  the current model path on a single `RTX 3090` is still far from that goal
- the remaining gains are more likely to come from reducing per-request work,
  shortening clone prompts, optimizing the generation path, or adding more GPU
  compute rather than simply allowing bigger merged batches

## Compute-Bound Throughput Improvement Plan

At this point the main limiter is no longer request routing or queueing
mechanics. The implementation is now clearly **compute-bound** on the GPU for
both design mode and clone mode.

This section lays out the recommended plan for improving throughput from the
compute side.

### Core Observation From The Current Model Path

The main short-form generation loop is implemented in
[omnivoice/models/omnivoice.py](/workspace/OmniVoice/omnivoice/models/omnivoice.py:1371).

The important properties of the current inference path are:

- the model runs an **iterative denoising loop** for `num_step` iterations
- for each step, it calls the full model again on the padded batch
- classifier-free guidance doubles the effective batch from `B` to `2B`
- the code currently builds a dense 4-D boolean attention mask of shape
  roughly `[2B, 1, S, S]`
- clone mode increases conditioning length because reference audio tokens are
  concatenated into the input sequence

Practical implication:

- throughput now scales mostly with:
  - `num_step`
  - sequence length
  - whether guidance is enabled
  - whether reference audio tokens are present

This means the best remaining wins are in the **inference loop itself**, not in
adding more queueing layers around it.

### What Not To Prioritize First

Do **not** make the first optimization attempt a traditional KV-cache rewrite.

Reason:

- this is not a standard left-to-right autoregressive decode loop
- the iterative denoising path progressively updates masked target positions
  across steps
- because the target region keeps changing, a normal autoregressive KV-cache
  does not map cleanly to this algorithm

That does not mean caching is impossible, but it should not be the first
compute optimization attempt.

## Priority 0: Add Real Per-Stage GPU Profiling

Before deeper code changes, add timing markers inside the hot path so we can
separate:

- prompt preprocessing
- iterative token generation
- token postprocessing / scoring
- audio tokenizer decode

Recommended instrumentation points:

- `GenerationService.generate(...)`
  - time `prepare_generation_task`
  - time `generation_batcher.submit(...)`
- `GenerationBatcher._process_batch(...)`
  - separate `generate_tokens` and `decode_tokens`
- `OmniVoice._generate_iterative(...)`
  - time the model forward per step
  - time `_predict_tokens_with_scoring`
- optional NVTX ranges around each of the above for Nsight Systems

Relevant files:

- [omnivoice/serving/service.py](/workspace/OmniVoice/omnivoice/serving/service.py:394)
- [omnivoice/serving/batching.py](/workspace/OmniVoice/omnivoice/serving/batching.py:638)
- [omnivoice/models/omnivoice.py](/workspace/OmniVoice/omnivoice/models/omnivoice.py:1463)

Why this is first:

- we already know we are GPU-bound
- we still need to know whether the dominant cost is:
  - LLM forward
  - attention mask overhead
  - token scoring
  - audio decode

## Priority 1: Reduce Or Remove CFG Doubling

This is the highest-leverage optimization exposed directly by the current code.

In `_generate_iterative(...)`, classifier-free guidance creates conditional and
unconditional branches and runs them together by doubling the effective batch:

- `batch_input_ids` shape uses `2 * B`
- `batch_audio_mask` shape uses `2 * B`
- `batch_attention_mask` shape uses `2 * B`

Relevant code:

- [omnivoice/models/omnivoice.py](/workspace/OmniVoice/omnivoice/models/omnivoice.py:1420)
- [omnivoice/models/omnivoice.py](/workspace/OmniVoice/omnivoice/models/omnivoice.py:1532)

Recommended plan:

1. benchmark `guidance_scale=0.0`
2. benchmark a reduced-guidance preset such as `0.5` or `1.0`
3. if quality is acceptable, expose a high-throughput serving lane that uses
   reduced guidance

Expected impact:

- potentially the single largest throughput gain available without changing the
  model architecture
- lower memory pressure
- lower batch execution time

Why this matters:

- if guidance can be weakened or disabled for some traffic classes, we remove a
  large fraction of current compute immediately

## Priority 2: Replace Dense Attention Masks In Inference

The current loop builds a dense boolean attention tensor for every batch:

- [omnivoice/models/omnivoice.py](/workspace/OmniVoice/omnivoice/models/omnivoice.py:1425)

This is expensive because:

- memory use grows with `S^2`
- mask writes and reads consume bandwidth
- dense masks can block more efficient attention kernel paths

Important code clue:

- `OmniVoice.forward(...)` already supports `document_ids` and `create_block_mask`
  instead of an explicit dense mask

Relevant code:

- [omnivoice/models/omnivoice.py](/workspace/OmniVoice/omnivoice/models/omnivoice.py:373)
- [omnivoice/models/omnivoice.py](/workspace/OmniVoice/omnivoice/models/omnivoice.py:382)

Recommended plan:

1. refactor `_generate_iterative(...)` to stop materializing the full dense
   `[2B, 1, S, S]` mask
2. use packed or block-mask metadata instead
3. verify that the chosen path preserves the same masking semantics
4. re-benchmark design mode and clone mode

Expected impact:

- moderate to large throughput gain
- lower memory bandwidth pressure
- better scaling at larger batch sizes and longer clone prompts

## Priority 3: Shape Bucketing + CUDA Graphs / `torch.compile`

Right now batch shapes vary with:

- batch size
- prompt length
- target length
- clone vs non-clone conditioning length

That variability makes it hard to get full benefit from graph capture or
compilation.

Recommended plan:

1. bucket requests more aggressively by shape:
   - lane
   - `num_step`
   - guidance enabled vs disabled
   - target-token bucket
   - conditioning-length bucket
2. once shapes are stable, capture the iterative forward path with CUDA Graphs
   or apply `torch.compile` to the inference function
3. keep a small cache of compiled/graph variants per worker

Why this likely helps:

- the same inference loop runs repeatedly with very similar shapes
- graph replay can reduce Python and launch overhead
- compilation is more likely to help once padding and batch shapes are less
  chaotic

Practical note:

- this should come **after** attention-mask cleanup, because the current dense
  mask path is likely to limit graph/compile gains

## Priority 4: Vectorize The Per-Item Step Logic

After each model forward, the code loops over each item in the batch in Python:

- extracts per-item conditional/unconditional logits
- computes scores
- chooses positions to reveal
- updates the batched tensors

Relevant code:

- [omnivoice/models/omnivoice.py](/workspace/OmniVoice/omnivoice/models/omnivoice.py:1487)

Recommended plan:

1. detect the common case where all items in a merged batch share the same
   target length
2. vectorize `_predict_tokens_with_scoring` across the full batch
3. replace per-item `topk` and flatten/update logic with batched tensor ops
4. keep a slower fallback for variable-length batches

Why this is promising:

- your benchmark workloads already tend to use fixed durations
- fixed-duration traffic means target lengths are often identical
- this is exactly the situation where the current Python loop is most
  avoidably wasteful

Expected impact:

- likely smaller than the CFG and attention-mask wins
- still worthwhile because it attacks step-level overhead that repeats
  `num_step` times per batch

## Priority 5: Use Shorter And Lighter Clone Prompts

Clone mode is currently much more expensive than design mode because reference
audio tokens are included in the conditioning sequence.

Relevant code:

- [omnivoice/models/omnivoice.py](/workspace/OmniVoice/omnivoice/models/omnivoice.py:1351)
- [omnivoice/models/omnivoice.py](/workspace/OmniVoice/omnivoice/models/omnivoice.py:760)

The current clone benchmark used a `6.2s` prompt and pushed peak GPU memory to
roughly `22 GB`.

Recommended operational policy:

- cap reference audio to about `2.5-3.5s` for high-throughput clone traffic
- require `ref_text`
- keep clone traffic in its own batching lane
- use smaller clone batch caps than design mode

This is partly a workload policy rather than a kernel optimization, but it is
one of the highest-value ways to reduce actual compute for clone serving.

## Priority 6: Reconsider `num_step` As A Product Tiering Knob

Because the model reruns the forward path once per step, `num_step` is almost a
direct compute multiplier.

Relevant code:

- [omnivoice/models/omnivoice.py](/workspace/OmniVoice/omnivoice/models/omnivoice.py:1463)

Recommended plan:

1. benchmark `num_step=8`, `12`, and `16`
2. compare quality for:
   - design mode
   - clone mode
3. define explicit service tiers such as:
   - high-throughput
   - balanced
   - quality-priority

Expected impact:

- potentially close to linear throughput improvement if quality remains
  acceptable

## Priority 7: System-Level CUDA Hygiene

These are lower-risk environment checks that should accompany the code work:

- verify the GPU is not power- or thermal-throttling under sustained load
- keep persistence mode enabled
- monitor clocks, power draw, and throttling reasons during the burst tests
- ensure no unnecessary CPU-side contention or process oversubscription is
  reducing GPU feed rate

These checks are unlikely to produce the biggest gain by themselves, but they
can prevent false negatives while testing the real compute-path changes.

## Suggested Execution Order

Recommended order of work:

1. add fine-grained profiling and NVTX markers
2. benchmark `guidance_scale=0.0`, `0.5`, `1.0`, `2.0`
3. refactor the inference loop to eliminate the dense attention mask
4. add stronger shape bucketing
5. try CUDA Graphs or `torch.compile`
6. vectorize the per-item scoring/update path
7. re-tune per-worker batch caps after the compute path is faster

## Expected Outcome

The most realistic path to materially higher throughput on a single `RTX 3090`
is:

- reduce effective compute per request
- make the iterative loop cheaper
- keep shapes stable enough to unlock better kernels

Not all of the remaining gap to the `100 requests in 2s` goal will be closed by
serving changes alone, but the highest-probability wins are now clearly inside
the model inference path rather than in the outer worker scheduler.

## Expanded Throughput Analysis

This section summarizes the current end-to-end throughput picture using both the
measured benchmark results and the actual code path in this branch.

### Executive Summary

The current bottleneck is the **model inference loop**, not the HTTP server and
not the two-worker router.

The evidence is consistent across both design and clone mode:

- requests are being distributed evenly across workers
- batches are filling correctly
- GPU utilization is already pinned near `100%`
- raising the batch cap increases VRAM usage, but only marginally improves
  throughput
- larger batches mostly increase `batch_exec_ms` instead of increasing
  requests-per-second proportionally

Practical meaning:

- we are no longer primarily queue-bound
- we are now primarily **compute-bound**
- the best remaining wins are inside the OmniVoice inference path itself

### Request Path Versus Hot Path

At a high level, one API request goes through four stages:

1. request normalization in
   [omnivoice/serving/service.py](/workspace/OmniVoice/omnivoice/serving/service.py:453)
2. task preparation in
   [omnivoice/models/omnivoice.py](/workspace/OmniVoice/omnivoice/models/omnivoice.py:560)
3. token generation in
   [omnivoice/models/omnivoice.py](/workspace/OmniVoice/omnivoice/models/omnivoice.py:594)
4. audio decode and postprocess in
   [omnivoice/models/omnivoice.py](/workspace/OmniVoice/omnivoice/models/omnivoice.py:642)

For warmed design-mode traffic, stage 3 is by far the dominant cost. For clone
traffic, stage 3 still dominates steady-state cost, while stage 1 and stage 2
become more expensive on cache misses because prompt construction is heavier.

### The Real Hot Paths

#### 1. Iterative Denoising Loop

The single most important hot path is
[OmniVoice._generate_iterative(...)](/workspace/OmniVoice/omnivoice/models/omnivoice.py:1371).

Important properties:

- the model runs a full denoising loop for `num_step` iterations
- the full LLM backbone is called again on every step
- the padded batch is rebuilt and updated across the whole target sequence

This means `num_step` is very close to a direct compute multiplier.

#### 2. CFG Doubles Effective Batch Size

In
[estimate_generation_batch_memory_bytes(...)](/workspace/OmniVoice/omnivoice/models/omnivoice.py:763),
the code explicitly computes:

- `effective_batch = 2 * batch_size if guidance_scale != 0 else batch_size`

That same doubling shows up in the actual runtime tensors inside
[_generate_iterative(...)](/workspace/OmniVoice/omnivoice/models/omnivoice.py:1416):

- `batch_input_ids` uses shape `(2 * B, C, S)`
- `batch_audio_mask` uses shape `(2 * B, S)`
- `batch_attention_mask` uses shape `(2 * B, 1, S, S)`

So the current production setting of `guidance_scale=2.0` is paying for both
the conditional and unconditional branches on every decode step.

#### 3. Dense Attention Mask Materialization

The iterative path currently creates a dense boolean mask tensor here:

- [omnivoice/models/omnivoice.py](/workspace/OmniVoice/omnivoice/models/omnivoice.py:1425)

This is expensive because:

- memory grows with `S^2`
- writes happen on every merged batch
- the mask contributes bandwidth pressure even before the real transformer work
  begins

This is especially important because the model already contains a more compact
masking path in
[OmniVoice.forward(...)](/workspace/OmniVoice/omnivoice/models/omnivoice.py:370),
where `document_ids` can be used to build a packed block mask instead of an
explicit dense mask.

#### 4. Python Per-Item Scoring And Update Loop

After each model forward, the code loops item-by-item in Python:

- [omnivoice/models/omnivoice.py](/workspace/OmniVoice/omnivoice/models/omnivoice.py:1487)

Inside that loop it:

- slices conditional and unconditional logits
- computes guided scores
- applies penalties and optional Gumbel noise
- runs `topk`
- writes selected tokens back into the shared batch tensors

This is repeated `num_step` times, so even moderate Python overhead becomes
meaningful at scale.

#### 5. Clone Conditioning Length

Clone mode is heavier even after prompt caches are warm because reference audio
tokens are included directly in the conditioning sequence:

- sequence-length estimate includes `ref_audio_tokens.size(-1)` in
  [omnivoice/models/omnivoice.py](/workspace/OmniVoice/omnivoice/models/omnivoice.py:728)
- inference input appends those tokens in
  [omnivoice/models/omnivoice.py](/workspace/OmniVoice/omnivoice/models/omnivoice.py:1351)

That means longer clone prompts increase:

- attention length
- hidden-state work
- logits footprint
- total batch execution time

This matches the observed benchmark regression from design mode to clone mode.

#### 6. Clone Prompt Construction On Cache Miss

On cache misses, clone mode also pays additional request-side cost in
[create_voice_clone_prompt(...)](/workspace/OmniVoice/omnivoice/models/omnivoice.py:836):

- load or resample audio
- RMS normalization
- optional trimming and silence removal
- optional Whisper transcription if `ref_text` is absent
- audio-token encoding

The service already caches these prompts in
[omnivoice/serving/service.py](/workspace/OmniVoice/omnivoice/serving/service.py:474),
so this is not the main steady-state throughput bottleneck, but it does matter
for cold clone traffic and for cache-miss heavy workloads.

#### 7. Audio Decode And Postprocess

Audio decode is implemented in
[decode_tokens(...)](/workspace/OmniVoice/omnivoice/models/omnivoice.py:642),
with optional silence removal and fade/pad in
[_post_process_audio(...)](/workspace/OmniVoice/omnivoice/models/omnivoice.py:974).

This stage is real work, but it is still secondary relative to the iterative
token-generation loop for the benchmark profile we have been running.

### What The Two-Worker Benchmarks Actually Proved

The two-worker architecture test was still useful, because it ruled out some
wrong intuitions:

- the router is not the bottleneck
- one worker being overloaded while the other idles is not the bottleneck
- “just remove the batch cap” is not the answer by itself

The measurements show:

- per-worker cap `12`: about `5.63 req/s`
- per-worker cap `24`: about `5.75 req/s`

That tiny gain came with a large increase in `batch_exec_ms`, which means the
GPU is already doing all the work it can on this request shape.

### Design Mode Versus Clone Mode

Design mode is currently the best throughput path for this server. Clone mode is
materially slower because it carries extra conditioning tokens and more memory
pressure.

Observed practical behavior:

- design mode at `24` per-worker cap: `~5.75 req/s`, peak GPU memory `~13.2 GB`
- warmed clone mode with a `6.2s` reference clip: `~2.81 req/s`, peak GPU
  memory `~21.9 GB`

That means long reference prompts are not just a little slower; they are a
major throughput limiter on a single `RTX 3090`.

### Secondary Costs Worth Tracking But Not Prioritizing First

These are real costs, but they are not the first place to spend engineering
effort:

- request parsing and WAV serialization in
  [omnivoice/serving/service.py](/workspace/OmniVoice/omnivoice/serving/service.py:59)
- router-side worker selection in
  [omnivoice/serving/multiworker.py](/workspace/OmniVoice/omnivoice/serving/multiworker.py:380)
- online micro-batch selection in
  [omnivoice/serving/batching.py](/workspace/OmniVoice/omnivoice/serving/batching.py:560)

These matter operationally, but they are not where the 3090 is currently
spending most of its time.

## External Optimization Analysis: Triton Vs TensorRT

An external suggestion was to look at the open-source `omnivoice-triton`
project, which reports a large OmniVoice speedup using Triton kernel fusion,
CUDA Graphs, and optional SageAttention.

Relevant sources:

- `omnivoice-triton` GitHub:
  `https://github.com/newgrit1004/omnivoice-triton`
- `omnivoice-triton` PyPI:
  `https://pypi.org/project/omnivoice-triton/`
- NVIDIA TensorRT dynamic shapes docs:
  `https://docs.nvidia.com/deeplearning/tensorrt/10.13.3/inference-library/work-dynamic-shapes.html`
- Torch-TensorRT performance guide:
  `https://docs.pytorch.org/TensorRT/user_guide/performance_tuning.html`

### What Looks Promising About The Triton Approach

The external Triton project is compelling because its optimization targets match
this repo’s actual hotspots:

- fused RMSNorm
- fused SwiGLU
- fused norm + residual
- CUDA Graph replay of repeated fixed-shape forwards

That maps well onto the current OmniVoice path here, where:

- the same LLM forward runs repeatedly in the iterative loop
- the model uses norm and MLP-heavy transformer blocks
- batch shapes are often similar across requests in the same benchmark class

### Important Caveats About The Reported Speedup

The external project’s headline benchmark should be treated carefully:

- it was measured on an `RTX 5090`
- it was measured at `batch size 1`
- it uses Python `>=3.12`
- it targets the pip `omnivoice` package rather than this serving fork

So the reported `~3.4x` should be viewed as a directional signal, not an
expected result for this branch on a `3090` under `100` concurrent requests.

### Current Environment Compatibility

The current local environment on this branch is:

- Python `3.11.15`
- PyTorch `2.8.0+cu128`
- CUDA `12.8`
- GPU: `NVIDIA GeForce RTX 3090`
- `triton`: installed
- `sageattention`: not installed
- `torch_tensorrt`: not installed

Implication:

- a repo-native Triton implementation is feasible
- the external package is **not** a direct drop-in as-is because of the Python
  version mismatch and model/package integration differences

### Why TensorRT Is Not The First Move

TensorRT is powerful, but it is not the best first optimization step for this
branch.

Reasons:

- this code path is highly shape-dynamic today
- TensorRT requires optimization profiles for dynamic shapes
- shape ranges can materially change memory usage and tactic availability
- Torch-TensorRT performance depends heavily on good graph coverage and low
  graph-break rates
- even a partial TRT compile would still leave the iterative Python orchestration
  outside the engine unless the whole path is refactored first

In other words, TensorRT becomes much more attractive **after**:

- masking is simplified
- request shapes are bucketed
- the hot forward path is made more static

### SageAttention Note

One subtle but important observation from the external code is that its
SageAttention path only applies when `attention_mask is None`.

That means the highest-value parts for this repo are likely:

- Triton fused norms and MLP kernels
- CUDA Graphs

not SageAttention itself, at least not until the current masking path changes.

### Recommendation

The best external-optimization sequence for this repo is:

1. implement repo-native mask cleanup and shape bucketing
2. add CUDA Graph capture per worker for a small set of shape buckets
3. port Triton fused kernels for RMSNorm, SwiGLU, and fused norm+residual
4. only then evaluate whether TensorRT is worth exploring for a narrower,
   more static subgraph

## Updated Phase Roadmap

This roadmap supersedes the earlier high-level rollout notes later in this
document. The intent here is to give us a clear phase-by-phase path that we can
execute and benchmark one step at a time.

### Phase 0: Instrumentation And Ground Truth

Goal:

- measure exactly where batch time is spent before changing the hot path

Work items:

- add fine-grained timers around:
  - request normalization
  - clone prompt cache hit vs miss path
  - `prepare_generation_task`
  - `generate_tokens`
  - `decode_tokens`
  - per-step model forward
  - per-step scoring/update
- add optional NVTX ranges for Nsight Systems
- log stage timing in worker responses and CSV benchmark output

Success criteria:

- we can break down one benchmark into stage-level timings instead of just
  `queue_wait_ms` and `batch_exec_ms`

Implementation status on this branch:

- Phase 0 instrumentation is now implemented in the live serving path
- request responses emit per-stage timing headers using the
  `X-OmniVoice-Timing-*-Ms` pattern
- the benchmark script now captures and summarizes those stage timings
- optional NVTX ranges can be enabled with `OMNIVOICE_ENABLE_NVTX=1`

Why first:

- every later optimization should be attributable to a known hot stage

### Phase 1: Product-Knob Throughput Sweep

Goal:

- identify the cheapest throughput wins with no algorithm rewrite

Work items:

- benchmark `guidance_scale` at `0.0`, `0.5`, `1.0`, `2.0`
- benchmark `num_step` at `8`, `12`, `16`
- benchmark `postprocess_output=false` for throughput-only lanes
- benchmark shorter clone prompts around `2.5-3.5s`

Success criteria:

- define at least one "high-throughput" preset and one "balanced" preset
- quantify how much throughput each product knob buys

Expected impact:

- this is the fastest path to a meaningful gain because CFG and `num_step` are
  direct compute multipliers

### Phase 2: Dense Mask Elimination

Goal:

- remove the `S^2` dense attention-mask materialization from the iterative path

Status:

- implemented on this branch
- `_generate_iterative(...)` now builds a batch-aware `BlockMask` once during
  setup instead of allocating a dense `[2B, 1, S, S]` bool tensor
- `forward(...)` now switches to `flex_attention` when a `BlockMask` is used so
  Qwen3 does not try to route the sparse mask through the `sdpa` backend
- the new mask preserves the previous inference semantics:
  - conditional rows attend within their active prefix only
  - unconditional rows attend within their active prefix and keep pad-diagonal
    self-attention for the padded tail
- this is an inference-specific mask path and does not change the existing
  packed-training `document_ids` path in `forward(...)`
- worker startup now runs a synthetic CUDA warmup batch so the `flex_attention`
  compile cost is paid before the worker starts serving real requests

Work items:

- refactor `_generate_iterative(...)` to use packed or block-mask metadata
- reuse the `document_ids` / `create_block_mask` path already supported by
  `forward(...)`
- verify output parity against the current path

Success criteria:

- no dense `[2B, 1, S, S]` mask allocation in the hot iterative path
- lower batch memory pressure
- measurable reduction in `batch_exec_ms`

Expected impact:

- medium to high

Historical note:

- the original Phase `0-8` rollout below is preserved as implementation
  history and background
- use the **Revised Roadmap For `100 Requests In 2 Seconds` With High Quality
  Output** above as the current go-forward plan

### Phase 3: Shape Bucketing

Goal:

- make inference shapes stable enough to benefit from graph replay and better
  kernel reuse

Status:

- implemented on this branch
- each queued request now carries:
  - a coarse shape bucket ID based on lane, CFG on/off, prompt-count bucket,
    target-token bucket, and conditioning-length bucket
  - an exact shape signature based on the request's concrete target and
    conditioning lengths
- batch selection now prefers exact-shape matches first, then falls back to
  coarse bucket matches within the same generation-compatibility key
- shape bucket IDs and exact-shape homogeneity flags are now emitted in:
  - batch logs
  - response headers
  - benchmark CSV output
  - benchmark summary output

Work items:

- bucket requests by:
  - lane
  - guidance enabled vs disabled
  - batch-size bucket
  - target-token bucket
  - conditioning-length bucket
- keep merged batches more homogeneous
- surface bucket IDs in logs and benchmark output

Success criteria:

- repeated requests hit a small number of stable shape classes
- padding ratio and shape churn both decrease

Expected impact:

- modest alone, but foundational for the next phases

### Phase 4: CUDA Graph Capture Per Worker

Goal:

- reduce kernel-launch and Python overhead for repeated fixed-shape forwards

Status:

- implemented experimentally on this branch for homogeneous exact-shape batches
- the iterative path now:
  - detects homogeneous batches where all requests share the same conditioning
    and target lengths
  - captures the iterative forward call into a CUDA Graph for those exact
    shapes
  - replays the graph on subsequent matching batches inside the same worker
- graph capture and replay timings are now visible in stage telemetry when the
  path is active
- this is intentionally conservative:
  - it only activates for homogeneous exact-shape batches
  - it falls back to eager execution for mixed-shape batches or failed captures

Work items:

- add CUDA Graph capture keyed by the Phase 3 shape buckets
- keep a bounded graph cache per worker
- add graph-hit / graph-miss telemetry

Success criteria:

- repeated bursts reuse captured graphs reliably
- `batch_exec_ms` decreases for stable-shape traffic

Expected impact:

- potentially large for short-form repeated workloads

### Phase 5: Triton Kernel Fusion

Goal:

- reduce bandwidth-heavy transformer operator overhead

Status:

- partially started on this branch as low-risk groundwork, not as full custom
  transformer kernel fusion
- the scoring path now attempts to use a compiled Inductor/Triton version when
  running on CUDA and when class sampling is disabled
- runtime failures in that compiled path are now caught once and force a safe
  eager fallback for the rest of the worker lifetime
- full custom Triton kernels for RMSNorm / SwiGLU / fused residual+norm are
  still pending

Work items:

- port or reimplement Triton kernels for:
  - RMSNorm
  - SwiGLU
  - fused residual + norm
- patch the OmniVoice backbone inside each worker
- validate quality parity on design and clone mode

Success criteria:

- kernel patches are selectable by feature flag
- no observable regression in output quality
- measurable end-to-end improvement beyond CUDA Graphs alone

Expected impact:

- medium to high, depending on how dominant these kernels are after Phase 4

### Phase 6: Vectorize The Per-Step Update Logic

Goal:

- remove repeated Python overhead in the scoring and token-update loop

Status:

- implemented early on this branch, ahead of shape bucketing
- `_generate_iterative(...)` now batches target-logit extraction,
  `_predict_tokens_with_scoring`, top-k position selection, and conditional /
  unconditional token writes instead of iterating request-by-request in Python
- this phase was pulled forward because Phase 0 showed scoring and update logic
  were the second-largest timed stage after the forward pass

Work items:

- batch `_predict_tokens_with_scoring`
- vectorize position selection and token update in the common fixed-duration case
- keep a slower variable-length fallback

Success criteria:

- fewer Python-side per-item operations inside each decode step
- measurable reduction in per-step overhead

Expected impact:

- medium

### Phase 7: Clone-Mode Specialization

Goal:

- make clone traffic less destructive to server-wide throughput

Work items:

- enforce or recommend shorter reference clips for high-throughput lanes
- keep clone traffic in distinct batching lanes
- optionally use lower clone batch caps than design mode
- strengthen prompt-cache metrics and reuse reporting

Success criteria:

- clone traffic has predictable memory and latency envelopes
- warmed clone performance improves without hurting design-mode traffic

Expected impact:

- medium operational gain, especially for production stability

### Phase 8: TensorRT Evaluation

Goal:

- evaluate TensorRT only after the inference path is sufficiently static

Work items:

- run Torch-TensorRT `dryrun=True` on the narrowed hot forward path
- inspect graph coverage and graph breaks
- test representative optimization profiles for the dominant shape buckets

Success criteria:

- TensorRT shows high coverage on the narrowed path
- compile complexity is justified by measured improvement

Expected impact:

- unknown today; intentionally deferred until the path is more static

### How We Should Execute The Phases

The recommended order is:

1. Phase 0
2. Phase 1
3. Phase 2
4. Phase 3
5. Phase 4
6. Phase 5
7. Phase 6
8. Phase 7
9. Phase 8

Rule for moving forward:

- do not start the next phase until the current one has:
  - code landed
  - benchmark deltas recorded
  - a short written conclusion added to this document

### Benchmark Gate For Every Phase

Every phase should be measured with the same core benchmark unless the phase is
clone-specific:

```bash
.venv/bin/python scripts/benchmark_api_batching.py \
  --url http://127.0.0.1:8002/generate \
  --requests 100 \
  --concurrency 100 \
  --launch-window-s 2 \
  --mode design \
  --num-step 16 \
  --guidance-scale 2.0 \
  --duration 4.0
```

Record at minimum:

- effective throughput
- total wall time
- mean / p50 / p95 latency
- queue wait
- batch exec time
- worker distribution
- peak GPU memory used
- allocator peak allocated / reserved

### Most Likely Highest-ROI Phases

If engineering time is limited, the most likely high-return sequence is:

1. Phase 1
2. Phase 2
3. Phase 4
4. Phase 5

That sequence attacks the biggest known compute multipliers first.

## Why We Are Testing This

Recent telemetry-backed benchmarks showed:

- the current server can batch successfully
- the current server still allows only **one in-flight merged batch per process**
- both `RTX 3080` and `RTX 3090` reached about **99% peak GPU utilization**
  during merged batches
- both cards used only about `~6 GB` of CUDA allocator reserved memory during
  those runs
- larger merged batches reduced queue wait, but also increased `batch_exec_ms`
  substantially

This suggests:

- we are not simply “failing to fill VRAM”
- the current bottleneck is more likely:
  - very large merged batch execution time
  - single in-flight batch per process
  - head-of-line blocking
  - model-path efficiency as batch size grows

The hypothesis behind a two-worker same-GPU setup is:

- two smaller workers may outperform one very large worker
- each worker can process smaller batches faster
- the router can distribute requests to whichever worker has less queue
- this may lower mean latency and p50/p95 latency even if absolute GPU peak
  utilization is already high during each batch

Important note:

- this is **not** guaranteed to improve throughput
- same-GPU multi-worker serving is an experiment justified by the current
  single-in-flight-batch limitation
- if the GPU is fully compute-bound even across the whole request timeline,
  two workers may simply compete and regress performance

## Success Criteria

The 2-worker same-GPU mode is successful if, on the `RTX 3090`, it improves at
least one of the following under the standard benchmark:

- higher effective throughput than the single-worker same-GPU setup
- lower mean latency
- lower p50 latency
- lower p95 latency
- lower queue wait

while keeping:

- success rate at `100%`
- no CUDA OOMs
- acceptable `batch_exec_ms`

The first benchmark to compare against is:

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

## Non-Goals For Initial Multi-Worker Phase

The first version of this plan does **not** try to solve everything at once.

Phase 1 will **not**:

- implement shared clone prompt caches across worker processes
- implement multi-GPU routing
- implement adaptive batch sizing
- rewrite the OmniVoice iterative decode path
- move long-form decoding to a new streaming protocol

Those can come later.

## Chosen Architecture

We will implement a **router + local worker-process** design.

### Topology

One parent process:

- binds the public HTTP port
- validates requests
- reads uploaded files
- routes each request to one of two local workers
- waits for the chosen worker’s response
- returns the WAV and headers to the client

Two child worker processes:

- each loads its own OmniVoice model on the **same CUDA device**
- each owns its own `GenerationBatcher`
- each handles batched inference independently
- each returns WAV bytes + telemetry + headers to the router

This gives us:

- 1 HTTP ingress process
- 2 local inference workers
- 2 independent in-flight merged batches max on one GPU

### Why Router + Workers Instead Of Threads

We do **not** want multiple Python threads calling one model instance.

Reasons:

- model state is GPU-resident
- current batching is process-local
- per-request lock-free model access would be difficult to reason about
- thread contention would be harder to observe and debug

Separate processes are cleaner:

- each process has one clear model owner
- each worker keeps its own batch queue and telemetry
- crashes are isolated
- routing decisions are explicit

## Process Model

### Router Process

Responsibilities:

- start child workers
- keep a control channel to each worker
- expose the FastAPI endpoints
- perform request validation
- receive uploads and normalize form data
- assign a request ID
- choose a worker
- forward request payload
- await worker result
- propagate worker response headers back to the client

The router does **not** load OmniVoice itself.

### Worker Process

Each worker process:

- loads OmniVoice on the configured GPU device
- creates its own `ClonePromptCache`
- creates its own `GenerationBatcher`
- receives request envelopes over IPC
- converts envelopes into local request handling calls
- returns:
  - WAV bytes
  - timing data
  - batching data
  - GPU telemetry
  - worker ID / PID

## Request Lifecycle

The same-GPU 2-worker flow will be:

1. client sends `POST /generate`
2. router validates request
3. router reads uploaded audio bytes if present
4. router chooses a worker using a load policy
5. router forwards a `WorkerRequestEnvelope`
6. worker processes request using its local batcher
7. worker returns `WorkerResponseEnvelope`
8. router returns the worker’s WAV + headers to the client

## IPC Design

Phase 1 should use **multiprocessing queues / pipes**, not sockets.

Recommended design:

- one request queue per worker
- one shared response queue
- request ID used to match responses

### Request Envelope

Suggested structure:

```python
@dataclass
class WorkerRequestEnvelope:
    request_id: str
    mode: str
    text: str
    language: str | None
    instruct: str | None
    ref_text: str | None
    ref_audio_bytes: bytes | None
    ref_audio_filename: str | None
    num_step: int
    guidance_scale: float
    speed: float | None
    duration: float | None
    denoise: bool
    preprocess_prompt: bool
    postprocess_output: bool
    created_at: float
```

### Response Envelope

Suggested structure:

```python
@dataclass
class WorkerResponseEnvelope:
    request_id: str
    ok: bool
    status_code: int
    error: str | None
    wav_bytes: bytes | None
    headers: dict[str, str]
    worker_id: str
    worker_pid: int
```

## Routing Policy

The initial routing policy should be simple and observable.

Use:

- least pending requests
- then least in-flight batches
- then least recent batch execution EWMA

Do **not** start with a complicated policy.

The router should maintain per-worker load state:

- pending request count
- in-flight batch flag
- recent mean `batch_exec_ms`
- recent mean queue wait
- recent success/failure counts

The router should route each request to the worker with the lowest current
load score.

### Why Not Round-Robin

Round-robin is easy, but it ignores:

- if one worker is already executing a large merged batch
- if one worker’s queue is much deeper
- if one worker is slower due to batch composition

We want a basic load-aware policy from the beginning.

## Per-Worker Batch Configuration

The whole point of this architecture is to let each worker run **smaller**
faster batches.

Recommended initial settings for the `RTX 3090` experiment:

- worker count: `2`
- per-worker `max_batch_requests`: `8` or `12`
- per-worker `max_batch_prompts`: same as request cap
- keep `batch_collect_ms=10`
- keep the existing VRAM-aware admission logic

The goal is:

- replace one very large `32-request` merged batch
- with two smaller workers that can each execute more quickly

We will benchmark at:

- `8 / 8`
- `12 / 12`
- maybe `16 / 16` if stable

## Memory Strategy

Two worker processes mean:

- duplicated model weights
- duplicated PyTorch allocator pools
- duplicated prompt caches

So the `RTX 3090` plan needs a preflight memory fit check.

### Required Preflight

Before starting the second worker:

1. start worker 1
2. measure idle CUDA used / reserved
3. start worker 2
4. verify both fit without OOM
5. leave additional reserve for real batches

We should add startup logging for:

- idle GPU memory used
- idle allocator reserved
- free CUDA memory after each worker starts

If two workers do not fit safely, the server should fail early with a clear
error.

## Clone Prompt Cache Behavior

Phase 1 choice:

- each worker keeps its own local `ClonePromptCache`

This is simpler than trying to share prompt objects across processes.

Tradeoff:

- repeated clone prompts may be cached twice
- but process-local correctness is simple

Phase 2 optional improvement:

- shared CPU-side prompt preparation cache in the router
- workers receive normalized prompt payloads instead of raw audio

That is not required for the first 2-worker experiment.

## API Surface Changes

The router should preserve the existing external API:

- `GET /health`
- `GET /languages`
- `POST /generate`

We should add worker metadata to responses:

- `X-OmniVoice-Worker-Id`
- `X-OmniVoice-Worker-Pid`

And `/health` should expose:

- router stats
- each worker’s current load snapshot
- each worker’s last batch summary
- each worker’s GPU telemetry

## File-Level Implementation Plan

### 1. New Serving Module

Add a new module, likely:

- `omnivoice/serving/multiworker.py`

Suggested classes:

```python
@dataclass
class WorkerProcessConfig: ...

@dataclass
class WorkerRequestEnvelope: ...

@dataclass
class WorkerResponseEnvelope: ...

class WorkerState: ...

class MultiWorkerRouter: ...

class WorkerProcessMain: ...
```

### 2. `api_server.py`

Update `create_app(...)` to support:

- single-worker mode
- multi-worker mode

Suggested new CLI flags:

- `--same-gpu-workers`
- `--worker-max-batch-requests`
- `--worker-max-batch-prompts`

Behavior:

- if `same_gpu_workers == 1`, use current code path
- if `same_gpu_workers > 1`, start router + workers

### 3. Worker-Side Request Handler

Refactor the current per-request logic in `api_server.py` into a reusable
function that can run inside a worker process without the outer HTTP layer.

Suggested helper:

```python
def process_generation_request(
    model_bundle,
    request: WorkerRequestEnvelope,
) -> WorkerResponseEnvelope:
    ...
```

### 4. Health Reporting

Extend `/health` to report:

- router queue stats
- worker queue depths
- worker EWMA latency
- worker GPU snapshots

## Scheduling Details

### Worker Selection Score

Initial load score:

```text
score =
  pending_requests * 1.0
  + in_flight_batches * 4.0
  + (recent_batch_exec_ms / 1000.0)
```

This is intentionally simple.

We can tune later.

### Backpressure

If both workers have queues beyond a threshold, the router should:

- either reject with `503`
- or keep the request and let latency rise

Phase 1 can keep queueing behavior unchanged, but we should add:

- `max_router_pending_requests`

for future overload protection.

## Metrics We Must Add

Per worker:

- pending request count
- batches started / completed
- mean batch exec
- mean queue wait
- GPU peak utilization
- GPU peak memory used
- allocator peak allocated / reserved

Per router:

- total inflight requests
- per-worker assignment counts
- per-worker failure counts
- average routing wait

Per response:

- worker ID
- worker PID
- all existing batch headers

## Benchmark Plan

We should compare:

### Baseline

- single worker
- `max_batch_requests=32`
- current 3090 benchmark

### Experiment A

- `same_gpu_workers=2`
- per-worker `max_batch_requests=8`

### Experiment B

- `same_gpu_workers=2`
- per-worker `max_batch_requests=12`

### Experiment C

- `same_gpu_workers=2`
- per-worker `max_batch_requests=16`

Use the same benchmark each time:

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

We should record:

- throughput
- mean / p50 / p95 latency
- queue wait
- batch exec
- worker distribution
- GPU telemetry

## Risks

### 1. Compute Contention

Two workers may compete for the same GPU and regress throughput.

This is the biggest risk.

### 2. Duplicate Model Memory

Two model replicas may consume enough VRAM to reduce useful headroom or cause
OOMs under larger batches.

### 3. Duplicate Prompt Caches

Clone prompt caching will be less memory-efficient until we add a shared
preparation layer.

### 4. More Operational Complexity

The server becomes:

- router process
- multiple worker processes
- IPC channels
- worker lifecycle management

### 5. Misleading Peak Utilization

Peak GPU utilization can be high even if average utilization over the request
timeline still has gaps.

This is why the experiment is still worth trying even though peak utilization
already looks high.

## Earlier Multi-Worker Rollout Plan (Historical)

### Phase 1

- add router + two worker processes
- add worker headers
- add worker health stats
- keep per-worker caches local
- benchmark design mode only

### Phase 2

- benchmark clone mode
- benchmark mixed design + clone traffic
- tune per-worker caps

### Phase 3

- optional shared clone prompt preparation
- optional adaptive routing policy
- optional adaptive batch sizing per worker

## Earlier Multi-Worker Recommendation (Historical)

Implement the first same-GPU experiment on the `RTX 3090` as:

- `same_gpu_workers=2`
- per-worker `max_batch_requests=8`
- per-worker `max_batch_prompts=8`
- current `batch_collect_ms=10`
- current VRAM telemetry retained

That is the safest initial configuration for testing whether:

- two smaller in-flight workers are better than one large in-flight worker

If that improves latency or throughput, we can tune upward.

If it regresses, we will know the next effort should go into:

- adaptive batch sizing
- model-path efficiency
- or a different hardware / multi-GPU strategy
