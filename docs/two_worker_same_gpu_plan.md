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

## Non-Goals For Phase 1

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

## Rollout Plan

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

## Recommendation

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
