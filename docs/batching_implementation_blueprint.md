# OmniVoice Batching Implementation Blueprint

This document describes the concrete batching architecture ported from the
`chatterbox-fastest` pattern onto OmniVoice, along with the Phase 1 server
implementation now present in this repo.

## Goal

Move OmniVoice from:

- one HTTP request
- one inline `model.generate(...)`
- one serialized response

to:

- request preparation
- clone prompt caching
- in-memory queueing
- anchor-based micro-batching
- one batched stage-1 generation call
- downstream decode
- per-request response delivery

## Current Phase 1 Implementation

The current code now includes:

- a serving batcher module in [batching.py](/workspace/OmniVoice/omnivoice/serving/batching.py:1)
- a clone prompt cache in [batching.py](/workspace/OmniVoice/omnivoice/serving/batching.py:95)
- API integration in [api_server.py](/workspace/OmniVoice/omnivoice/cli/api_server.py:292)
- new model serving primitives in [omnivoice.py](/workspace/OmniVoice/omnivoice/models/omnivoice.py:457)

This is the implemented request flow:

1. `/generate` validates request data.
2. Clone requests prepare or reuse a cached `VoiceClonePrompt`.
3. The request is normalized into a one-item `GenerationTask`.
4. The request is assigned a compatibility key and batching lane.
5. The request becomes a `PendingGeneration` job and is enqueued.
6. A background `GenerationBatcher` waits a short collection window.
7. Compatible jobs are merged into one batched `GenerationTask`.
8. One stage-1 model generation batch is executed.
9. Token outputs are decoded into waveform audio.
10. Each request receives its own response plus queue and batch timing headers.

## Main Components

### 1. `PendingGeneration`

Implemented in [batching.py](/workspace/OmniVoice/omnivoice/serving/batching.py:60).

Represents one queued API request. It stores:

- request ID
- mode
- creation timestamp
- compatibility key
- prepared `GenerationTask`
- generation config
- per-prompt postprocess flags
- estimated sequence lengths
- response future

### 2. `GenerationBatchKey`

Implemented in [batching.py](/workspace/OmniVoice/omnivoice/serving/batching.py:26).

This is the compatibility key used to decide whether requests can join the
same batch.

Fields:

- `num_step`
- `guidance_scale`
- `t_shift`
- `layer_penalty_factor`
- `position_temperature`
- `class_temperature`
- `denoise`
- `audio_chunk_duration`
- `audio_chunk_threshold`
- `lane`

Notably, the key does **not** include:

- language
- instruct
- clone prompt identity
- `speed`
- `duration`
- `postprocess_output`
- `preprocess_prompt`

That is intentional. OmniVoice already supports per-item conditioning during the
batched stage-1 path, and `postprocess_output` is handled after stage 1.

### 3. `GenerationBatcher`

Implemented in [batching.py](/workspace/OmniVoice/omnivoice/serving/batching.py:199).

This is the background batching thread. It:

- holds pending jobs in memory
- waits up to `collect_ms` to gather nearby arrivals
- uses the first pending job as the anchor
- selects compatible jobs under configured caps
- runs one merged batch at a time
- distributes outputs back to request futures

This is intentionally similar to the current Chatterbox pattern:

- one process
- one in-flight batch
- anchor-based selection
- exact config-key matching

### 4. Clone Prompt Cache

Implemented in [batching.py](/workspace/OmniVoice/omnivoice/serving/batching.py:95).

The cache stores prepared `VoiceClonePrompt` objects instead of raw uploaded
audio. Cache entries are keyed by:

- reference audio bytes hash
- `ref_text`
- `preprocess_prompt`

Prompts are stored with CPU-resident audio tokens to avoid growing persistent
GPU memory usage.

### 5. Model Serving Primitives

Added in [omnivoice.py](/workspace/OmniVoice/omnivoice/models/omnivoice.py:583):

- `prepare_generation_task(...)`
- `generate_tokens(...)`
- `decode_tokens(...)`
- `estimate_inference_sequence_length(...)`

These methods split the serving pipeline into:

- request normalization / preparation
- stage-1 token generation
- stage-2 waveform decode

This is the key change that makes API-level batching practical.

## Batching Lanes

The Phase 1 lane model is:

- `short_mixed`
- `long_ref`
- `long_no_ref`

Meaning:

- all short requests can share one lane, including mixed auto/design/clone
- long clone traffic stays separate
- long no-reference traffic stays separate

This matches OmniVoice’s current internal limitations:

- short-form iterative generation can mix per-item conditioning cleanly
- long-form chunked generation does **not** support mixed ref/non-ref batches

Relevant code:

- [omnivoice.py](/workspace/OmniVoice/omnivoice/models/omnivoice.py:760)
- [omnivoice.py](/workspace/OmniVoice/omnivoice/models/omnivoice.py:793)

## Batch Selection Rules

Implemented in [batching.py](/workspace/OmniVoice/omnivoice/serving/batching.py:279).

The batcher enforces:

- `max_batch_requests`
- `max_batch_prompts`
- `max_total_target_tokens`
- `max_total_conditioning_tokens`
- `max_padding_ratio`

This is better aligned with OmniVoice than a simple request-count cap because
OmniVoice’s iterative decode path pads to the longest sequence in the batch.

## What Gets Batched

### Batched in Phase 1

- short-form stage-1 token generation
- mixed auto/design/clone short requests
- different languages in one short batch
- different style instructions in one short batch
- different clone prompts in one short batch
- long-form requests within their own compatible lanes

### Not Fully Batched Yet

- long-form chunk decode remains structurally sequential by chunk index
- final per-request response serialization is still per request
- chunked long-form audio stitching is still per request
- clone prompt preparation itself is cached, but not yet precomputed by a
  separate background worker pool

## Decoder Behavior

OmniVoice differs from Chatterbox in a useful way.

Chatterbox’s downstream S3Gen path remains heavily serialized.

OmniVoice’s audio tokenizer decoder actually accepts batched audio codes, so
Phase 1 now batch-decodes short-form outputs that share the same token length.

Implemented in [omnivoice.py](/workspace/OmniVoice/omnivoice/models/omnivoice.py:623).

Current behavior:

- chunked outputs are still decoded one item at a time
- non-chunk outputs are grouped by token length
- each equal-length group is decoded together
- postprocessing remains per item

This is already a small improvement over a strictly serialized stage-2 tail.

## API Server Changes

Implemented in [api_server.py](/workspace/OmniVoice/omnivoice/cli/api_server.py:536).

The old server path used:

- upload handling
- direct `model.generate(...)`
- a global `generate_lock`

The new server path now uses:

- prompt cache lookup / prompt preparation
- `prepare_generation_task(...)`
- compatibility-key assignment
- `GenerationBatcher.submit(...)`
- response assembly after batch completion

The old single-inference lock has been replaced by queued batching.

## New Response Headers

The API now returns batching-aware timing headers in addition to total latency:

- `X-OmniVoice-Queue-Wait-Ms`
- `X-OmniVoice-Batch-Exec-Ms`
- `X-OmniVoice-Batch-Requests`
- `X-OmniVoice-Batch-Prompts`
- `X-OmniVoice-Batch-Target-Tokens`
- `X-OmniVoice-Batch-Max-Sequence-Length`

These are useful when tuning batch collection windows and queue behavior on
your vast.ai instances.

## Current Tuning Parameters

Added as CLI flags in [api_server.py](/workspace/OmniVoice/omnivoice/cli/api_server.py:155):

- `--batch-collect-ms`
- `--max-batch-requests`
- `--max-batch-prompts`
- `--max-batch-target-tokens`
- `--max-batch-conditioning-tokens`
- `--max-batch-padding-ratio`
- `--clone-prompt-cache-size`

## Known Limitations

Phase 1 is a real batching architecture, but it is not the final state.

Important remaining limitations:

- one server process still runs one in-flight merged batch at a time
- there is not yet one worker process per GPU with cross-process routing
- there is no separate preprocessing pool for clone prompt creation
- long-form streaming is not implemented
- sentence splitting / regrouping is not exposed at the API layer yet
- fairness is still anchor-based, so long jobs can head-of-line block later jobs

## Recommended Next Phases

### Phase 2

- add a dedicated `/prepare-clone-prompt` endpoint
- return `prompt_id`
- allow generate requests to reference cached prompt IDs directly

### Phase 3

- split preprocessing from request threads
- move clone prompt creation into a background prep pool
- add queue metrics per lane

### Phase 4

- one model worker process per GPU
- central router / dispatcher
- SLA-aware lane scheduling
- admission control for overload

### Phase 5

- stream long-form chunk outputs
- optional prompt splitting and regrouping
- more aggressive batched stage-2 decode

## File Map

Files directly involved in the current batching implementation:

- [api_server.py](/workspace/OmniVoice/omnivoice/cli/api_server.py:1)
- [batching.py](/workspace/OmniVoice/omnivoice/serving/batching.py:1)
- [omnivoice.py](/workspace/OmniVoice/omnivoice/models/omnivoice.py:457)

Related planning docs:

- [batching_scaling_plan.md](/workspace/OmniVoice/docs/batching_scaling_plan.md:1)
- [throughput_metrics.md](/workspace/OmniVoice/docs/throughput_metrics.md:1)
