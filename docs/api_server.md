# API Server

This document explains the FastAPI server implemented in
[omnivoice/cli/api_server.py](/workspace/OmniVoice/omnivoice/cli/api_server.py:1).

The server wraps the existing `OmniVoice` Python inference API and exposes
simple HTTP endpoints for health checks, language discovery, and text-to-speech
generation.

## Overview

The server is designed to be a minimal HTTP layer around:

- `OmniVoice.from_pretrained(...)`
- `model.generate(...)`
- optional reference-audio uploads for clone mode

It supports three generation modes:

- `auto`: text-only generation with no explicit voice prompt
- `design`: text plus `instruct`
- `clone`: text plus uploaded reference audio, with optional `ref_text`

The server returns generated audio as `audio/wav`.

## Entry Point

The CLI entrypoint is:

```bash
omnivoice-api
```

It is registered in [pyproject.toml](/workspace/OmniVoice/pyproject.toml:60).

You can also run it directly as a module:

```bash
.venv/bin/python -m omnivoice.cli.api_server --model k2-fsa/OmniVoice --ip 0.0.0.0 --port 8002
```

## Startup Behavior

At startup, `create_app(...)`:

1. Detects the best device if one is not provided.
2. Selects `float16` for GPU or `float32` for CPU.
3. Loads the model with `OmniVoice.from_pretrained(...)`.
4. Optionally loads ASR at startup unless `--no-asr` is used.
5. Stores the model and runtime settings in `app.state`.

Relevant code:

- [api_server.py](/workspace/OmniVoice/omnivoice/cli/api_server.py:259)
- [api_server.py](/workspace/OmniVoice/omnivoice/cli/api_server.py:278)

## CLI Options

Main runtime flags:

- `--model`: checkpoint path or Hugging Face repo ID
- `--device`: explicit device such as `cuda`, `cuda:0`, `cpu`, or `mps`
- `--ip`: bind address
- `--port`: bind port
- `--root-path`: reverse-proxy root path
- `--no-asr`: skip loading Whisper ASR at startup
- `--save-dir`: persist a copy of each generated WAV

Optional accelerator flags are also accepted:

- `--coreml-backbone`
- `--coreml-decoder`
- `--onnx-backbone`
- `--onnx-decoder`

Important note:

This repo currently does not implement the Core ML / ONNX model-loading hooks
referenced by these flags. The server accepts the flags, but if you pass them
without the corresponding model methods existing, startup will fail with a
clear `RuntimeError`.

Relevant code:

- [api_server.py](/workspace/OmniVoice/omnivoice/cli/api_server.py:71)
- [api_server.py](/workspace/OmniVoice/omnivoice/cli/api_server.py:221)

## Endpoints

### `GET /health`

Returns basic runtime status:

- model checkpoint
- device
- sampling rate
- whether ASR is loaded
- whether save-dir is configured
- optional accelerator metadata if available

Relevant code:

- [api_server.py](/workspace/OmniVoice/omnivoice/cli/api_server.py:373)

Example:

```bash
curl http://127.0.0.1:8002/health
```

### `GET /languages`

Returns the supported language list derived from `LANG_NAME_TO_ID`.

Each item includes:

- `id`
- `name`
- `display_name`

Relevant code:

- [api_server.py](/workspace/OmniVoice/omnivoice/cli/api_server.py:417)

Example:

```bash
curl http://127.0.0.1:8002/languages
```

### `POST /generate`

Generates speech and returns a WAV file.

Relevant code:

- [api_server.py](/workspace/OmniVoice/omnivoice/cli/api_server.py:429)

The endpoint expects multipart form data.

## Request Modes

### `mode=auto`

Use when you want basic text-to-speech without explicit voice guidance.

Allowed:

- `text`
- optional `language`
- generation parameters such as `num_step`, `guidance_scale`, `speed`, `duration`

Rejected:

- `ref_audio`
- `ref_text`
- `instruct`

### `mode=design`

Use when you want voice design from an instruction string.

Required:

- `text`
- `instruct`

Rejected:

- `ref_audio`
- `ref_text`

### `mode=clone`

Use when you want zero-shot voice cloning from a reference audio clip.

Required:

- `text`
- `ref_audio`

Optional:

- `ref_text`

Behavior:

- If `ref_text` is omitted, the model may auto-transcribe with ASR.
- If native Core ML runtime is active, `ref_text` is required.

Validation logic lives in:

- [api_server.py](/workspace/OmniVoice/omnivoice/cli/api_server.py:463)

## Request Fields

The `/generate` endpoint accepts:

- `mode`
- `text`
- `language`
- `instruct`
- `ref_text`
- `num_step`
- `guidance_scale`
- `speed`
- `duration`
- `denoise`
- `preprocess_prompt`
- `postprocess_output`
- `ref_audio`

Some details:

- `text` is always required.
- `duration` must be greater than `0` if provided.
- `speed` is only forwarded if it differs from `1.0`.
- `language="Auto"` or blank is normalized to `None`.

Relevant code:

- [api_server.py](/workspace/OmniVoice/omnivoice/cli/api_server.py:434)
- [api_server.py](/workspace/OmniVoice/omnivoice/cli/api_server.py:515)

## Response

Successful responses return:

- HTTP `200`
- `Content-Type: audio/wav`
- inline WAV bytes in the response body

The server sets:

- `Content-Disposition: inline; filename="omnivoice.wav"`

Relevant code:

- [api_server.py](/workspace/OmniVoice/omnivoice/cli/api_server.py:560)

## Latency And Throughput Logging

The server includes explicit per-request timing metadata for testing.

### Response Headers

Every successful `/generate` response includes:

- `X-OmniVoice-Request-Id`
- `X-OmniVoice-Started-At`
- `X-OmniVoice-Finished-At`
- `X-OmniVoice-Latency-Ms`
- `X-OmniVoice-Audio-Duration-S`
- `X-OmniVoice-RTF` if audio duration is non-zero

If `--save-dir` is configured, it also includes:

- `X-OmniVoice-Saved-Path`

Relevant code:

- [api_server.py](/workspace/OmniVoice/omnivoice/cli/api_server.py:560)

### Server Logs

On success, the server logs one structured `INFO` line per request including:

- request ID
- mode
- latency in milliseconds
- output audio duration in seconds
- RTF
- input text length
- whether reference audio was present
- normalized language
- device
- saved output path if applicable

Relevant code:

- [api_server.py](/workspace/OmniVoice/omnivoice/cli/api_server.py:577)

Example log shape:

```text
request_id=abc123 status=success mode=design started_at=2026-04-12T19:10:00Z finished_at=2026-04-12T19:10:01Z latency_ms=842.11 audio_s=3.104 rtf=0.2713 text_chars=22 has_ref_audio=False language=auto device=cuda saved_path=-
```

### What `RTF` Means

`RTF` is real-time factor:

```text
RTF = generation_time_seconds / output_audio_seconds
```

Interpretation:

- `RTF < 1.0`: faster than real-time
- `RTF = 1.0`: real-time
- `RTF > 1.0`: slower than real-time

The server computes it from wall-clock request latency and the generated audio
duration.

Relevant code:

- [api_server.py](/workspace/OmniVoice/omnivoice/cli/api_server.py:553)

## Upload Handling

For clone mode:

1. Uploaded audio is written to a temporary file.
2. That file path is passed into `model.generate(...)`.
3. The temp file is removed in `finally`.

Relevant code:

- [api_server.py](/workspace/OmniVoice/omnivoice/cli/api_server.py:179)
- [api_server.py](/workspace/OmniVoice/omnivoice/cli/api_server.py:540)
- [api_server.py](/workspace/OmniVoice/omnivoice/cli/api_server.py:614)

## Save Directory

If `--save-dir` is set:

- the WAV response is still returned inline
- a copy is also persisted locally
- the saved path is added to the response headers

Filename format:

```text
{timestamp}_{mode}_{slug}_{short_uuid}.wav
```

Relevant code:

- [api_server.py](/workspace/OmniVoice/omnivoice/cli/api_server.py:206)
- [api_server.py](/workspace/OmniVoice/omnivoice/cli/api_server.py:568)

## Concurrency Model

This server currently uses:

- one in-process model instance
- one `threading.Lock()` around `model.generate(...)`

That means requests are serialized within a single server process.

Relevant code:

- [api_server.py](/workspace/OmniVoice/omnivoice/cli/api_server.py:352)
- [api_server.py](/workspace/OmniVoice/omnivoice/cli/api_server.py:548)

This is simple and safe for initial testing, but it is not a high-throughput
production scheduling design. It prevents concurrent inference within the same
process and is mainly useful for correctness, smoke tests, and early latency
measurement.

## Error Handling

The server maps common failures to HTTP responses:

- validation issues: `400`
- user-facing `ValueError`: `400`
- `RuntimeError`: `500`
- unexpected exceptions: `500`

Relevant code:

- [api_server.py](/workspace/OmniVoice/omnivoice/cli/api_server.py:596)

## Example Requests

### Auto Mode

```bash
curl -i -X POST http://127.0.0.1:8002/generate \
  -F mode=auto \
  -F text='Hello from OmniVoice'
```

### Design Mode

```bash
curl -i -X POST http://127.0.0.1:8002/generate \
  -F mode=design \
  -F text='Hello from OmniVoice' \
  -F instruct='male, british accent'
```

### Clone Mode

```bash
curl -i -X POST http://127.0.0.1:8002/generate \
  -F mode=clone \
  -F text='Hello from OmniVoice' \
  -F ref_audio=@ref.wav \
  -F ref_text='Hello from OmniVoice'
```

### Save Generated WAVs

```bash
.venv/bin/python -m omnivoice.cli.api_server \
  --model k2-fsa/OmniVoice \
  --save-dir ./generated_wavs
```

## Recommended Testing Workflow

For first-pass validation:

1. Start the server locally.
2. Hit `/health`.
3. Send one request for `auto`, `design`, and `clone`.
4. Inspect:
   - HTTP status
   - WAV playback
   - latency headers
   - server logs

Example health check:

```bash
curl http://127.0.0.1:8002/health
```

Example latency-focused request:

```bash
curl -i -X POST http://127.0.0.1:8002/generate \
  -F mode=design \
  -F text='Testing latency logging' \
  -F instruct='female, low pitch'
```

Look for:

- `X-OmniVoice-Started-At`
- `X-OmniVoice-Finished-At`
- `X-OmniVoice-Latency-Ms`
- `X-OmniVoice-Audio-Duration-S`
- `X-OmniVoice-RTF`

## HTTP Request File

A ready-to-use request file lives at:

- [api_server.http](/workspace/OmniVoice/examples/api_server.http:1)

It includes example requests for:

- `/health`
- `/languages`
- `mode=auto`
- `mode=design`
- `mode=clone`

The clone example uses a local file include placeholder:

```text
< {{ref_audio_path}}
```

Before sending the clone request, update `@ref_audio_path` in the `.http` file
to point to a real local WAV file.

## Comparing 3080 vs 3090

For your GPU tests, the most useful response headers are:

- `X-OmniVoice-Started-At`
- `X-OmniVoice-Finished-At`
- `X-OmniVoice-Latency-Ms`
- `X-OmniVoice-Audio-Duration-S`
- `X-OmniVoice-RTF`

Suggested comparison method:

1. Run the same request file on the 3080 server and the 3090 server.
2. Keep `text`, `mode`, `num_step`, `guidance_scale`, `duration`, and prompt inputs identical.
3. Record `X-OmniVoice-Latency-Ms`.
4. Record `X-OmniVoice-RTF`.
5. Compare the server logs using `request_id`, `started_at`, and `finished_at`.

Interpretation:

- Lower `X-OmniVoice-Latency-Ms` is better for end-user response time.
- Lower `X-OmniVoice-RTF` is better for synthesis efficiency.
- `RTF < 1.0` means faster than real-time.

## Current Limitations

- No batching across HTTP requests
- No streaming audio output
- Requests are serialized with a global generate lock
- Accelerator flags are only usable if the underlying model implementation
  provides those optional methods
- Clone-mode uploads are written to temp files rather than streamed directly

These limitations are acceptable for functional testing but not sufficient for
the high-concurrency serving goals discussed elsewhere.
