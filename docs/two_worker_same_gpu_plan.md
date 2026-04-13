# Two-Worker Same-GPU Plan

This document describes the implementation plan for running **two OmniVoice
workers on the same GPU**, starting with the `RTX 3090` test environment.

The goal is to reduce head-of-line blocking and improve request latency under
load by allowing **two smaller in-flight batches** on one GPU instead of one
very large in-flight batch per process.

This is an execution plan, not just a brainstorming note.

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

