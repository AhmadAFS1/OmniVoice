# Batching And Scaling Plan

This document captures the batching and concurrency plan for evolving the
current OmniVoice API server from single-request serialized inference into a
production-oriented serving system that can support large concurrent user loads.

The focus is on:

- simple TTS
- voice design TTS
- zero-shot voice-clone TTS
- workloads on the order of `100`, `500`, and `1000` concurrent users

## Executive Summary

The current API server can process requests correctly, but it is not designed
for high-concurrency serving.

Right now:

- the API server serializes inference with a global lock
- requests are handled one at a time per server process
- voice-clone preprocessing happens inline inside the request path
- no online batching or queue-based scheduling exists

Relevant code:

- [api_server.py](/workspace/OmniVoice/omnivoice/cli/api_server.py:360)
- [api_server.py](/workspace/OmniVoice/omnivoice/cli/api_server.py:557)

The right scaling strategy is not “add more Python threads around the existing
endpoint.”

The right strategy is:

- async HTTP ingress
- queue-based request scheduling
- dedicated preprocessing stages
- GPU microbatching
- one hot worker process per GPU
- separate batching lanes for incompatible request types
- prompt caching for voice cloning

## What The Current Code Already Supports

The core model already supports batched inference at the Python API layer.

`OmniVoice.generate(...)` accepts list inputs for:

- `text`
- `language`
- `instruct`
- `duration`
- `speed`
- clone inputs

Relevant code:

- [omnivoice.py](/workspace/OmniVoice/omnivoice/models/omnivoice.py:457)

The model also already has:

- a batched short-form iterative decode path
- a chunked long-form generation path
- some internal batching across chunks for long requests

Relevant code:

- [omnivoice.py](/workspace/OmniVoice/omnivoice/models/omnivoice.py:560)
- [omnivoice.py](/workspace/OmniVoice/omnivoice/models/omnivoice.py:760)
- [omnivoice.py](/workspace/OmniVoice/omnivoice/models/omnivoice.py:1118)

This means the core problem is not whether the model can batch.
It can.

The real problem is that there is no online scheduler feeding it batches.

## Why The Current Server Does Not Scale

The current `/generate` endpoint performs the full request lifecycle inline:

1. validate request
2. optionally save uploaded reference audio to a temp file
3. call `model.generate(...)`
4. wait for generation to finish
5. serialize output to WAV
6. return the response

Relevant code:

- [api_server.py](/workspace/OmniVoice/omnivoice/cli/api_server.py:442)
- [api_server.py](/workspace/OmniVoice/omnivoice/cli/api_server.py:551)

This is protected by:

- `app.state.generate_lock`

Relevant code:

- [api_server.py](/workspace/OmniVoice/omnivoice/cli/api_server.py:360)

So even if:

- there are many HTTP clients
- the app server has many threads
- multiple users hit the endpoint at once

the current implementation still effectively runs one generation at a time per
server process.

## Why “Multithreaded” Alone Is Not Enough

There are two separate ideas:

1. multithreaded request handling
2. batched GPU inference

Multithreading is useful for:

- HTTP ingress
- file upload handling
- temp-file work
- audio loading
- audio preprocessing
- response serialization

Multithreading is not the main way to speed up GPU inference for this model.

For the GPU path, the better design is:

- one dedicated worker process per GPU
- one active inference batch per worker
- many queued requests waiting to be coalesced into microbatches

Trying to drive one GPU model instance from many Python threads is typically
worse than using one controlled scheduler.

## Major Bottlenecks In The Current Code

### 1. API-Level Serialization

Inference is serialized by a global lock.

Relevant code:

- [api_server.py](/workspace/OmniVoice/omnivoice/cli/api_server.py:360)
- [api_server.py](/workspace/OmniVoice/omnivoice/cli/api_server.py:557)

### 2. Voice Clone Preprocessing Is Inline

Clone requests can trigger:

- audio loading
- resampling
- silence trimming
- optional ASR transcription
- prompt tokenization

Relevant code:

- [omnivoice.py](/workspace/OmniVoice/omnivoice/models/omnivoice.py:583)
- [omnivoice.py](/workspace/OmniVoice/omnivoice/models/omnivoice.py:658)
- [audio.py](/workspace/OmniVoice/omnivoice/utils/audio.py:32)
- [audio.py](/workspace/OmniVoice/omnivoice/utils/audio.py:70)

These steps are too expensive to repeat inline for large volumes of repeated
clone traffic.

### 3. Iterative Decode Pads To The Longest Item In The Batch

The model builds padded tensors using the largest request in the batch:

- `max_c_len`
- dense `batch_attention_mask`
- padded target token tensor

Relevant code:

- [omnivoice.py](/workspace/OmniVoice/omnivoice/models/omnivoice.py:1159)
- [omnivoice.py](/workspace/OmniVoice/omnivoice/models/omnivoice.py:1163)
- [omnivoice.py](/workspace/OmniVoice/omnivoice/models/omnivoice.py:1172)

So naive batching without length control creates waste and poor throughput.

### 4. Iterative Decode Re-Runs The Model For Every Step

The iterative loop re-invokes the model `num_step` times:

- `16-step`
- `32-step`
- or higher

Relevant code:

- [omnivoice.py](/workspace/OmniVoice/omnivoice/models/omnivoice.py:1227)

So batching policy must consider:

- `num_step`
- target token count
- prompt length

### 5. Decode And Postprocess Are Still Per-Item

After token generation, outputs are still decoded and postprocessed item by
item.

Relevant code:

- [omnivoice.py](/workspace/OmniVoice/omnivoice/models/omnivoice.py:572)
- [omnivoice.py](/workspace/OmniVoice/omnivoice/models/omnivoice.py:684)
- [omnivoice.py](/workspace/OmniVoice/omnivoice/models/omnivoice.py:721)

This is not the biggest bottleneck yet, but it matters under large batch sizes.

## Important Constraint: Mixed Clone And Non-Clone Batches Are Not Safe Today

This is one of the most important implementation details.

In `_preprocess_all(...)`, if `ref_audio` is present, the code tries to create
clone prompts batch-wise. The logic is not robust for arbitrary mixtures of:

- some requests with `ref_audio`
- some requests without `ref_audio`

Relevant code:

- [omnivoice.py](/workspace/OmniVoice/omnivoice/models/omnivoice.py:917)
- [omnivoice.py](/workspace/OmniVoice/omnivoice/models/omnivoice.py:923)
- [omnivoice.py](/workspace/OmniVoice/omnivoice/models/omnivoice.py:933)

There is also an explicit long-form constraint:

- chunked inference does not support mixed ref/non-ref batches

Relevant code:

- [omnivoice.py](/workspace/OmniVoice/omnivoice/models/omnivoice.py:793)

Operational implication:

- clone and non-clone requests should be batched separately
- long clone and long non-clone traffic should be separated as well

## High-Level Serving Architecture

The recommended architecture is:

1. HTTP ingress layer
2. request normalization layer
3. CPU preprocessing workers
4. scheduler queues by request lane
5. one GPU batch worker per GPU
6. response assembly and delivery

### Stage 1: HTTP Ingress

Responsibilities:

- accept requests quickly
- validate basic fields
- return request IDs immediately to internal scheduler

This should be async and lightweight.

### Stage 2: Request Normalization

Convert incoming HTTP form data into an internal request object containing:

- `request_id`
- `mode`
- `text`
- `language`
- `instruct`
- `num_step`
- `guidance_scale`
- `duration`
- `speed`
- `postprocess_output`
- estimated token budget
- prompt type
- prepared clone prompt if available
- a future/promise for result delivery

This isolates HTTP from model execution.

### Stage 3: CPU Preprocessing Workers

For clone requests, move prompt preparation out of the GPU scheduler path.

Responsibilities:

- load reference audio
- resample
- trim silence
- optionally transcribe if ASR is allowed
- audio-tokenize prompt
- produce a reusable `VoiceClonePrompt`

Relevant current prompt abstraction:

- [omnivoice.py](/workspace/OmniVoice/omnivoice/models/omnivoice.py:74)
- [omnivoice.py](/workspace/OmniVoice/omnivoice/models/omnivoice.py:583)

### Stage 4: Scheduler Queues

Requests should not all go into one queue.

Recommended initial queue lanes:

- `short_simple_fast`
- `short_simple_quality`
- `short_clone_fast`
- `short_clone_quality`
- `long_simple`
- `long_clone`

Definitions:

- `simple` = auto/design without clone prompt
- `clone` = prepared clone prompt
- `fast` = lower-step path like `16-step`
- `quality` = higher-step path like `32-step`
- `short` vs `long` = below or above chunk threshold

### Stage 5: GPU Batch Workers

Each GPU should run:

- one warm model instance
- one scheduler loop
- one active batch at a time

Responsibilities:

- pop requests from compatible lanes
- group them into a microbatch
- call `model.generate(...)` with list inputs
- complete the result futures/promises

### Stage 6: Response Delivery

After generation:

- serialize returned waveform(s)
- return response to user
- optionally persist outputs or metrics

## Batching Strategy

### Use Dynamic Microbatching

Do not wait for huge static batches.

Instead:

- collect requests for a short time window, such as `10-30 ms`
- dispatch earlier if enough work accumulates
- cap latency with a max-wait threshold

Each queue lane should support:

- `max_wait_ms`
- `max_batch_size`
- `max_total_target_tokens`
- `max_total_prompt_tokens`
- `max_padding_ratio`

### Batch By Token Budget, Not Just Request Count

The right batching unit is not:

- “10 users”
- “100 users”

The right batching unit is:

- prompt length
- target token count
- total token budget
- padding cost
- generation-step class

This is necessary because iterative decode cost is dominated by the largest
sequence in the batch.

### Use Length Bucketing

Requests should be grouped by approximate target length before forming the final
microbatch.

A practical first version:

- estimate target tokens
- group into coarse buckets
- batch within the same bucket

There is already an offline example of clustering by duration in the batch CLI:

- [infer_batch.py](/workspace/OmniVoice/omnivoice/cli/infer_batch.py:291)

That logic is not directly sufficient for online serving, but it is a useful
starting reference.

## Voice Clone Strategy

Voice cloning should become a two-step workflow in production.

### Recommended API Pattern

1. `POST /prepare-clone-prompt`
2. `POST /generate` with `prompt_id`

Why this matters:

- users often reuse the same reference voice
- repeated prompt preprocessing is wasted work
- repeated ASR is wasted work
- prompt tokenization is expensive enough to cache

### Prompt Cache

Cache key should include:

- audio hash
- `ref_text`
- `preprocess_prompt`
- any other fields that materially affect prompt preparation

Cache value should include:

- `VoiceClonePrompt`
- prompt metadata
- maybe estimated prompt token length

### Require `ref_text` In The Fast Clone Lane

For large-scale serving, clone requests should ideally provide `ref_text`.

Reason:

- omitting `ref_text` can invoke Whisper ASR
- ASR dramatically increases per-request overhead
- ASR is better treated as a separate slow path or preprocessing service

Relevant code:

- [omnivoice.py](/workspace/OmniVoice/omnivoice/models/omnivoice.py:658)
- [omnivoice.py](/workspace/OmniVoice/omnivoice/models/omnivoice.py:290)

## Long-Form Request Strategy

The model already supports chunked long-form generation internally.

Relevant code:

- [omnivoice.py](/workspace/OmniVoice/omnivoice/models/omnivoice.py:760)
- [omnivoice.py](/workspace/OmniVoice/omnivoice/models/omnivoice.py:830)

Operationally:

- long requests should not share the same main queue as short requests
- long requests should have their own lanes
- they should be scheduled with stricter token budgets

Eventually, long-form output should be streamed chunk-by-chunk so time-to-first-
audio improves even when total completion time is large.

## What Needs To Change In The Codebase

### API Server Changes

Replace direct serialized inference with:

- async queue insertion
- background scheduler loop(s)
- batch worker(s)
- separate preprocessing stage
- future/promise result handling

The current direct call pattern:

- [api_server.py](/workspace/OmniVoice/omnivoice/cli/api_server.py:551)

needs to be replaced with:

- create internal request object
- enqueue it
- await completion

### Model-Side Changes

The following changes would make batching much cleaner:

1. Fix mixed clone/non-clone preprocessing in `_preprocess_all(...)`
2. Add a prepared-request generation path that accepts prebuilt clone prompts
3. Optionally add a lower-level `generate_prepared(batch)` API
4. Optionally batch decode/postprocess

The current mixed-mode assumptions in `_preprocess_all(...)` make a universal
single-queue batcher risky.

## Why One GPU Worker Per GPU Is The Right Default

For this model, the best first production pattern is:

- one worker process per GPU
- one hot model instance per worker
- one active inference batch at a time
- many queued requests feeding that worker

This is better than many concurrent model calls because it:

- avoids Python thread contention around one GPU
- reduces memory fragmentation
- makes batching deterministic
- gives clear control over queueing and SLA

## Why The Current 3080 vs 3090 Results Look Tied

The benchmark history is stored in:

- [throughput_metrics.md](/workspace/OmniVoice/docs/throughput_metrics.md:1)

The measured design-mode benchmarks showed that 3080 and 3090 are nearly tied
under the current server design.

This does not necessarily mean the GPUs are truly equivalent for production.

It more likely means:

- the workload is single-request oriented
- no online batching exists
- the GPU is not being driven at a scale where 3090 memory or throughput
  advantages show clearly

Under proper batch-heavy workloads, the 3090 should have more room to
differentiate.

## Capacity Framing For 100 / 500 / 1000 Users

The right way to think about user load is not:

- one batch of 100
- one batch of 500
- one batch of 1000

The right way to think about it is:

- arrival rate
- output duration
- SLA target
- available GPU-seconds per second

From the current serialized benchmark shape:

- `16-step`: about `1.85 requests/sec` for a `6.2s` clip
- `32-step`: about `0.95 requests/sec` for a `6.2s` clip

Equivalent audio throughput:

- `16-step`: about `11.5 audio-sec/sec/GPU`
- `32-step`: about `5.9 audio-sec/sec/GPU`

If `1000` users each request one `6.2s` clip every `60s`, offered load is:

```text
1000 * 6.2 / 60 = 103.3 audio-sec/sec
```

At current measured efficiency, that implies roughly:

- around `9` GPUs at current `16-step` efficiency before headroom
- around `18` GPUs at current `32-step` efficiency before headroom

Practical deployments should add headroom beyond these theoretical minimums.

## Recommended Implementation Phases

### Phase 1: Basic Online Batching

Build:

- internal request queue
- microbatch scheduler
- one GPU worker per GPU
- separate simple vs clone lanes
- separate short vs long lanes

Operational rule:

- require `ref_text` for the main clone benchmark lane

### Phase 2: Clone Prompt Preparation And Caching

Build:

- `/prepare-clone-prompt`
- `prompt_id` workflow
- prompt cache
- prompt metrics

This will likely deliver the highest win for clone traffic.

### Phase 3: Model Interface Cleanup

Implement:

- robust mixed preprocessing semantics
- prepared-batch generation API
- more explicit lower-level model-serving hooks

### Phase 4: Streaming And SLA-Aware Scheduling

Implement:

- chunk streaming for long-form requests
- per-lane max wait
- priority queues
- fairness logic
- overload shedding and admission control

## Practical First Recommendation

For the next engineering pass, do not build:

- one universal multi-threaded endpoint that tries to run many `generate(...)`
  calls directly

Instead, build:

- a queued scheduler inside the API server
- separate lanes for simple TTS and clone TTS
- a dedicated clone prompt preparation path
- batched `model.generate(...)` calls using list inputs
- one model worker per GPU

This is the serving design that can realistically move the system toward
supporting large concurrent user populations.

## Current Conclusion

Yes, batching both simple TTS and voice-clone TTS is feasible.

But the path is:

- queueing
- token-budgeted microbatching
- prompt caching
- lane separation
- controlled GPU workers

not:

- “just add more threads to the current endpoint”
