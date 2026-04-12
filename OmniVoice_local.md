# OmniVoice On-Device Mobile Feasibility

Reviewed: 2026-04-10

## Short Verdict

Running this OmniVoice repo directly inside a React Native app is not realistic.

This repository is a Python/PyTorch/Transformers project designed for server, desktop, or batch GPU inference. The inference entry points assume PyTorch devices such as CUDA, Apple Silicon MPS, or CPU, and the repo does not contain React Native, Core ML, TFLite, ONNX Runtime, or native mobile integration code.

Running an OmniVoice-derived model on a phone might be possible after a separate mobile inference port, but reliable real-time production performance is unlikely at the default settings. A high-end-phone prototype is most plausible with an exported and quantized ONNX model, fewer generation steps, no Whisper ASR, and short utterances.

## Why This Repo Is Not React Native Ready

The package dependencies in `pyproject.toml` are server/desktop Python dependencies:

- `torch>=2.4`
- `torchaudio>=2.4`
- `transformers>=5.3.0`
- `accelerate`
- `pydub`
- `gradio`
- `soundfile`

Relevant code:

- `pyproject.toml`
- `omnivoice/cli/infer.py`
- `omnivoice/cli/api_server.py`
- `omnivoice/cli/demo.py`
- `omnivoice/models/omnivoice.py`

The CLI/server device detection explicitly targets:

- CUDA
- MPS
- CPU

There is no React Native bridge, no mobile asset packaging path, and no mobile-native inference runtime in this repo.

## Model Architecture Summary

The main runtime is in `omnivoice/models/omnivoice.py`.

OmniVoice is a Hugging Face `PreTrainedModel` wrapper around:

- a text LLM backbone, loaded through `AutoModel`
- audio token embeddings
- a linear audio prediction head
- a `HiggsAudioV2TokenizerModel` audio tokenizer/codec
- a Hugging Face text tokenizer
- optional Whisper ASR for auto-transcribing reference audio

The model config for the published checkpoint uses a Qwen3-style LLM:

- 28 hidden layers
- hidden size 1024
- intermediate size 3072
- 16 attention heads
- 8 key/value heads
- vocabulary size 151,676
- 8 audio codebooks
- audio vocabulary size 1025

This is a substantial model for a mobile app, especially when combined with the audio tokenizer/codec.

## Artifact Size

The local repo does not include model weights. The code downloads them from Hugging Face by default via:

```python
OmniVoice.from_pretrained("k2-fsa/OmniVoice")
```

Published artifact sizes from the `k2-fsa/OmniVoice` Hugging Face commit metadata:

| Artifact | Size |
|---|---:|
| `model.safetensors` | about 2.45 GB |
| `audio_tokenizer/model.safetensors` | about 806 MB |
| `tokenizer.json` | about 11 MB |
| Main model + audio tokenizer + tokenizer JSON | about 3.27 GB |

That is before runtime memory, activations, temporary tensors, ONNX/Core ML conversion overhead, or optional ASR.

Optional Whisper ASR is especially important to avoid on mobile. If `ref_text` is omitted for voice cloning, the repo can load `openai/whisper-large-v3-turbo` to transcribe the reference audio. That is not appropriate for an on-device TTS path unless replaced with a much smaller ASR flow or removed from the product requirement.

## Inference Behavior

The generation path is not a simple streaming vocoder call. It is iterative masked audio-token generation:

1. Build style and text tokens.
2. Add optional reference audio tokens for cloning.
3. Fill the target audio region with mask tokens.
4. Run `num_step` iterative decoding rounds.
5. Decode final audio tokens back to waveform with the audio tokenizer.
6. Post-process audio by removing silence, normalizing, fading, and padding.

Default generation config:

- `num_step = 32`
- `guidance_scale = 2.0`
- `audio_chunk_duration = 15.0`
- `audio_chunk_threshold = 30.0`

The docs say `num_step=16` can be used for faster inference, but this is a quality/speed tradeoff.

The expensive part is `_generate_iterative(...)` in `omnivoice/models/omnivoice.py`. For each decoding step, it runs a full model forward pass over a doubled conditional/unconditional batch for classifier-free guidance. This is a good shape for GPU throughput, but a difficult shape for low-latency phone inference.

The final output is returned after token generation and audio decoding. This repo is not currently structured as a low-latency streaming TTS engine.

## Real-Time Claim Context

The README says OmniVoice can reach very low RTF, down to 0.025. The OmniVoice paper reports very fast results on GPU hardware, including H20 GPU measurements for 16-step and 32-step settings.

That should not be interpreted as a phone performance claim. GPU server throughput and on-device mobile latency are different constraints:

- Phone memory bandwidth is much lower than server GPU bandwidth.
- Mobile NPUs/GPUs often do not support arbitrary Transformer graphs as cleanly as CUDA.
- React Native itself is not the inference runtime; it must call into ONNX Runtime, Core ML, TFLite, or native code.
- Long context attention and iterative decoding multiply latency.
- Thermal throttling matters after repeated generations.
- Cold model load time can dominate UX.

## Mobile Feasibility Assessment

### Direct Python/PyTorch in React Native

Not recommended.

Reasons:

- React Native cannot directly run this Python package in production app code.
- PyTorch Mobile is no longer the normal path for modern Transformers-style mobile deployment.
- The repo uses Hugging Face `transformers` classes and custom generation logic.
- `torchaudio`, `pydub`, and server tooling add more non-mobile assumptions.
- The main + audio tokenizer artifacts are too large for a straightforward app bundle.

### Core ML or TFLite Port

Possible in theory, but high effort.

Likely blockers:

- custom `OmniVoice` wrapper
- Qwen3 model conversion
- dynamic sequence lengths
- attention masks
- iterative decoding loop
- audio tokenizer encode/decode graph
- post-processing and prompt audio preprocessing
- quality preservation after quantization

This would be an engineering project, not a small integration task.

### ONNX Runtime React Native

Most plausible path.

ONNX Runtime has a React Native package: `onnxruntime-react-native`.

A third-party Hugging Face repo, `Gigsu/vocoloco-onnx`, appears to provide an ONNX export related to OmniVoice/VocoLoco-style inference. It lists:

- INT8 main model around 586 MB
- decoder around 83 MB
- voice-cloning encoder around 624 MB
- web/mobile/low-memory oriented ONNX usage

This is the best starting point for a phone experiment because it avoids porting this Python repo directly.

### TensorRT

Not a useful route for the React Native phone app.

TensorRT is NVIDIA's inference optimizer/runtime for NVIDIA GPUs. It is useful when the deployment target has NVIDIA CUDA-capable hardware, such as a server GPU or some NVIDIA edge devices. Normal iPhones and Android phones do not have NVIDIA CUDA GPUs.

Where TensorRT could help:

- Server-side OmniVoice on a Vast.ai-style NVIDIA GPU instance.
- Possibly NVIDIA Jetson-style edge hardware.

Where TensorRT does not help:

- Running OmniVoice locally inside a normal iPhone app.
- Running OmniVoice locally inside a normal Android phone app.

For this project, TensorRT is worth considering only if the goal changes to making the server cheaper/faster. It is not the right path for local React Native smartphone inference.

### Core ML

Core ML is the Apple-native route for iPhone inference.

Using Core ML could be faster than plain ONNX Runtime CPU inference because Core ML can use Apple hardware backends such as CPU, GPU, and the Neural Engine when the model graph is supported. However, converting OmniVoice to Core ML is not a simple one-step conversion.

Likely challenges:

- The repo is not one static model graph. It is a full Python inference pipeline.
- The main model is a custom Hugging Face `PreTrainedModel` wrapper around a Qwen3-style LLM.
- Generation uses an iterative decoding loop, not a single forward pass.
- Classifier-free guidance doubles conditional/unconditional passes.
- The pipeline also needs a text tokenizer and audio tokenizer/decoder.
- Dynamic sequence lengths, masks, attention patterns, and custom post-processing may not export cleanly.
- Voice cloning adds reference audio encoding and possibly ASR.

The most realistic Core ML strategy is not "convert the whole repo." It is to split the pipeline:

- Keep the text/token generation orchestration in native code or a React Native native module.
- Convert the heavy neural pieces first, likely the main model and audio decoder/tokenizer path.
- Keep voice cloning in scope, but do not require the voice-clone prompt encoder to be exported in the first milestone.
- Avoid Whisper ASR initially by requiring `ref_text` when testing clone mode.
- Use short fixed-shape test cases first, then expand once latency and memory are understood.

### ONNX vs Core ML

ONNX by itself is a model format and portability layer. It does not guarantee that the model will run fast on a phone. Speed depends on the runtime and execution provider.

Relevant paths:

- ONNX Runtime CPU only: easiest to try, but probably too slow for OmniVoice real-time on iPhone 12 Pro.
- ONNX Runtime React Native: useful for wiring the benchmark into a React Native app.
- ONNX Runtime with CoreML Execution Provider on iOS: can route supported ONNX subgraphs through Core ML, which may use Apple acceleration.
- Direct Core ML `.mlpackage`: most iOS-native route and likely the better final iPhone performance target if conversion works.

Practical recommendation:

Use ONNX first as the fastest experiment, especially because a third-party ONNX export candidate already exists. If it runs but is too slow, then try converting the heavy pieces to direct Core ML. For iPhone performance, the target should be direct Core ML or ONNX Runtime using the CoreML Execution Provider, not ONNX Runtime CPU-only.

## Localhost ONNX Runtime Phase Plan

This is the best localhost-first path for testing whether ONNX Runtime can improve throughput before touching React Native.

### Current inference path

The current FastAPI server simply loads the model once and calls `model.generate(...)` for each request.

Inside `generate(...)`, the current flow is:

1. text preprocessing and tokenization
2. optional voice prompt preparation
3. iterative diffusion-style decoding
4. audio token decoding
5. waveform post-processing

The most expensive part is the iterative decoding loop in `_generate_iterative(...)`. It runs the backbone forward pass once per diffusion step, and with classifier-free guidance it evaluates both conditional and unconditional paths.

### What should move to ONNX first

The practical export order is:

1. OmniVoice backbone
2. audio decoder
3. voice-clone prompt encoder
4. ASR only if absolutely needed

Why this order:

- The backbone runs every diffusion step, so it dominates latency.
- The decoder runs once at the end.
- Voice-clone prompt creation runs once per request.
- Whisper ASR is not desirable in the first ORT milestone.

### Voice cloning requirement

Voice cloning can stay working in the first ONNX Runtime milestone.

The way to do that is:

- keep `create_voice_clone_prompt(...)` in PyTorch first
- export only the heavy backbone first
- continue passing `ref_audio` and `ref_text`
- strongly prefer requiring `ref_text` so Whisper does not load

That preserves clone mode while still attacking the main throughput bottleneck.

### What ONNX Runtime actually replaces

ONNX Runtime should not replace the whole `model.generate(...)` method at first.

Instead, keep the current Python orchestration and replace the expensive backbone forward call inside the diffusion loop.

In other words:

- keep current preprocessing in Python
- keep current scheduling and diffusion loop in Python
- keep current scoring logic in Python
- keep current post-processing in Python
- replace the repeated backbone forward pass with an ONNX Runtime session

This is the fastest way to answer whether ONNX Runtime helps on localhost.

### Export target for phase 1

The first export target should be a backbone wrapper with inputs and outputs that mirror the current diffusion loop:

- `input_ids: [B, C, S]`
- `audio_mask: [B, S]`
- `attention_mask: [B, 1, S, S]`
- output `logits: [B, C, S, V]`

Keeping `attention_mask` explicit avoids baking in too many assumptions too early and keeps closer parity with the current PyTorch behavior.

### Existing external evidence

There is already a useful public reference project that split OmniVoice into separate exported backbone and decoder modules:

- `acul3/OmniVoice-LiteRT`

Its approach matches the recommended direction here:

- export backbone separately
- export decoder separately
- keep the diffusion loop outside the exported model

This is a strong signal that the model can be decomposed in a useful way for ONNX Runtime or other edge runtimes.

### Best ONNX Runtime provider for localhost

For a local MacBook Air test, the most relevant ONNX Runtime provider is:

- `CoreMLExecutionProvider`

with CPU fallback.

That does not mean React Native yet. It just means the localhost Python benchmark should try ONNX Runtime on macOS with Core ML acceleration where possible.

Dynamic shapes may work, but the CoreML Execution Provider documentation notes that dynamic shapes can hurt performance. If the first dynamic-shape export works but is not fast enough, the next optimization should be bucketed static-shape exports such as fixed sequence lengths.

### Current phase map

Phase 1:

- Export backbone to ONNX.
- Add a localhost Python ONNX Runtime backend.
- Fix parity and audio quality.
- Status: done enough to move forward. Dynamic-sequence ONNX restored quality parity for the backbone path.

Phase 2:

- Make Apple-accelerated ONNX work through CoreMLExecutionProvider.
- Re-check parity and localhost throughput.
- Status: working, but still slower than PyTorch + MPS on this 2022 MacBook Air.

Phase 3:

- Remove the remaining inference-time PyTorch dependency from the hot path.
- Export and validate the audio decoder.
- Keep clone mode in scope, with `ref_text` required so Whisper stays out of the runtime path.
- Target outcome: a full neural inference path that can run without PyTorch at request time.

Phase 4:

- Build the first true mobile-style runtime harness.
- Compare ONNX Runtime Mobile versus direct Core ML packaging for the migrated models.
- Benchmark on a real iPhone instead of inferring from the Mac.

### Expected benefit and risk

Best case:

- backbone ONNX Runtime is faster than PyTorch MPS
- localhost throughput improves materially
- the same exported model architecture becomes a better stepping stone toward React Native

Main risks:

- ONNX Runtime may not beat PyTorch MPS on this Mac
- dynamic shapes may limit Core ML acceleration
- large model export may require external data files
- decoder and clone prompt paths may still remain PyTorch bottlenecks after backbone export

### Recommended localhost order

1. Backbone-only ONNX export
2. Local Python ONNX Runtime backend
3. Benchmark against current PyTorch API
4. Add decoder export only if backbone export is promising
5. Keep voice cloning supported by leaving prompt creation in PyTorch until later

### Phase 1 implementation status

Phase 1 has been wired into this repo as an optional backend.

What is implemented:

- ONNX backbone export script: `omnivoice-export-onnx-backbone`
- optional ONNX Runtime backbone loading in `OmniVoice`
- API flag: `--onnx-backbone`
- API flag: `--onnx-provider auto|cpu|coreml`
- single-item CLI flag: `--onnx_backbone`
- single-item CLI flag: `--onnx_provider auto|cpu|coreml`

What still stays in PyTorch during phase 1:

- voice-clone prompt creation
- audio decoder
- post-processing
- all Python orchestration around the diffusion loop

What this means in practice:

- the heavy repeated backbone forward pass can run through ONNX Runtime
- voice cloning can still work if `ref_text` is provided
- Whisper ASR should still be avoided during testing
- if the request shape does not match the exported ONNX backbone shape, the code falls back to the normal PyTorch backbone

Example export command:

```bash
uv sync --extra onnx
uv run omnivoice-export-onnx-backbone \
  --model k2-fsa/OmniVoice \
  --output artifacts/onnx/omnivoice_backbone.onnx \
  --seq-len 1024 \
  --batch-size 2
```

Example localhost API command:

```bash
uv run omnivoice-api \
  --model k2-fsa/OmniVoice \
  --device mps \
  --no-asr \
  --onnx-backbone artifacts/onnx/omnivoice_backbone.onnx \
  --onnx-provider coreml \
  --ip 127.0.0.1 \
  --port 8002
```

Example single-item CLI command:

```bash
uv run omnivoice-infer \
  --model k2-fsa/OmniVoice \
  --device mps \
  --onnx_backbone artifacts/onnx/omnivoice_backbone.onnx \
  --onnx_provider coreml \
  --text "This is a short test of local ONNX Runtime inference." \
  --instruct "female, american accent" \
  --num_step 16 \
  --output ort_test.wav
```

## Recommended Prototype Plan

1. Start with a localhost Python ONNX Runtime benchmark before React Native.

   The goal is to answer one question first: does ONNX Runtime improve throughput on the current MacBook Air versus PyTorch MPS?

2. Keep voice cloning working during the ONNX Runtime rollout.

   For the first ONNX milestone, voice cloning should continue to work by keeping prompt creation in PyTorch. Require `ref_text` during testing so Whisper does not load. Exporting the prompt encoder can be deferred until later.

3. Export the backbone before anything else.

   This is the main throughput target because it runs every diffusion step.

4. Build a tiny React Native benchmark app with `onnxruntime-react-native` only after localhost ORT proves useful.

   The benchmark should measure:

   - cold load time
   - warm generation latency
   - real-time factor
   - peak RAM
   - app bundle/model download size
   - battery and thermal behavior after repeated generations
   - audio quality at 8, 16, and 32 steps if supported

5. Use short and medium utterances first.

   Suggested test strings:

   - 2 seconds of speech
   - 5 seconds of speech
   - 10 seconds of speech

   Also include at least one clone-mode test with a supplied `ref_text`, because voice cloning is a hard requirement for the product.

6. Define a minimum supported device.

   Do not judge by a flagship device only. Pick the actual lowest-end iPhone and Android device you are willing to support. Test on real hardware, not only a simulator.

7. Use a pass/fail threshold.

   A practical first target:

   - warm RTF below 1.0 for 5-10 second utterances
   - no app crash or OS memory kill
   - acceptable first-response latency
   - no severe thermal throttling after repeated use
   - acceptable quality at the fastest step count

## Local MacBook Air Test Results

Environment:

- MacBook Air, 2022 Apple Silicon, `arm64`
- Localhost API server with model kept loaded in memory
- Device: `mps`
- Mode: design voice
- Instruct: `female, american accent`
- `num_step=16`
- API endpoint: `http://127.0.0.1:8002/generate`

Important interpretation:

- The CLI command `uv run omnivoice-infer ...` starts a fresh Python process each time, so it reloads weights from disk into memory every run.
- The API server loads the model once, then keeps it warm for repeated requests.
- The first ever model run downloaded about 2.45 GB of main model weights and about 817 MB for the audio tokenizer. Later runs used the Hugging Face cache and did not redownload the files.
- The API server is the better latency test because it avoids repeated model load time.

Short sentence tested:

```text
This is a short test of local text to speech.
```

Command:

```bash
curl -X POST http://127.0.0.1:8002/generate \
  -F mode=design \
  -F text="This is a short test of local text to speech." \
  -F instruct="female, american accent" \
  -F num_step=16 \
  --output local_test.wav
```

Observed curl runs:

| Run | Curl wall time | Received size | Notes |
|---|---:|---:|---|
| 1 | about 5 seconds | about 117 KB | warm API request |
| 2 | about 4 seconds | about 120 KB | warm API request |
| 3 | about 4 seconds | about 132 KB | warm API request |

The retained `local_test.wav` from the last short run measured:

| File | Audio duration | Sample rate | Format | Approx RTF |
|---|---:|---:|---|---:|
| `local_test.wav` | 2.82 seconds | 24000 Hz | WAV PCM 16-bit | about 1.42 |

Longer sentence tested:

```text
Today I am testing whether OmniVoice can generate natural sounding speech quickly enough for a mobile app, while keeping the voice clear, expressive, and stable across a longer sentence.
```

Command:

```bash
curl -X POST http://127.0.0.1:8002/generate \
  -F mode=design \
  -F text="Today I am testing whether OmniVoice can generate natural sounding speech quickly enough for a mobile app, while keeping the voice clear, expressive, and stable across a longer sentence." \
  -F instruct="female, american accent" \
  -F num_step=16 \
  --output local_test_10sec.wav
```

Observed curl result:

| File | Curl wall time | Received size | Audio duration | Sample rate | Format | Approx RTF |
|---|---:|---:|---:|---:|---|---:|
| `local_test_10sec.wav` | about 22 seconds | about 494 KB | 10.55 seconds | 24000 Hz | WAV PCM 16-bit | about 2.09 |

Takeaway:

The warm local API path is much faster than repeatedly running the CLI, but this 2022 MacBook Air still did not reach real-time at `num_step=16` in these tests. The short sentence was around 1.4x real time, and the longer sentence was around 2.1x real time. This increases the risk for on-device phone inference unless the ONNX/mobile runtime path gives a large optimization win or the app accepts lower quality settings.

## Recommended Product Strategy

For production quality today, keep OmniVoice server-side.

For cost reduction, test an on-device ONNX path in parallel and only move suitable use cases local if the benchmark passes on target phones.

Likely split:

- Server: voice cloning, long-form generation, best quality, broad language support, fallback for low-end devices.
- On-device: short auto/design TTS, cached voices, offline mode, premium devices only if benchmarks pass.

If server cost is the main problem, also consider:

- batching requests
- caching repeated generated phrases
- caching reusable voice clone prompts
- reducing `num_step` to 16 for lower-latency modes
- using GPU instances only while busy
- using a lower-cost TTS model for simple local responses

## Sources Checked

Local repo:

- `README.md`
- `pyproject.toml`
- `omnivoice/models/omnivoice.py`
- `omnivoice/cli/infer.py`
- `omnivoice/cli/api_server.py`
- `omnivoice/cli/demo.py`
- `omnivoice/cli/infer_batch.py`
- `docs/generation-parameters.md`
- `docs/training.md`
- `vast_startup.sh`

External:

- `k2-fsa/OmniVoice` Hugging Face model: https://huggingface.co/k2-fsa/OmniVoice
- OmniVoice paper: https://arxiv.org/abs/2604.00688
- ONNX Runtime React Native docs: https://onnxruntime.ai/docs/get-started/with-javascript/react-native.html
- ONNX Runtime CoreML Execution Provider docs: https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html
- PyTorch ONNX export docs: https://pytorch.org/docs/stable/onnx.html
- `acul3/OmniVoice-LiteRT`: https://huggingface.co/acul3/OmniVoice-LiteRT
- Third-party ONNX export candidate: https://huggingface.co/Gigsu/vocoloco-onnx

## Phase 1 ONNX Runtime Validation

Status as of 2026-04-11: the phase-1 ONNX Runtime migration is functionally integrated, and the dynamic-sequence backbone export fixes the major quality failure caused by fixed-shape padding. It is still not a throughput win on this MacBook Air.

### What Was Implemented

Phase 1 keeps the existing OmniVoice generation orchestration and swaps only the repeated backbone forward pass to ONNX Runtime.

Implemented pieces:

- optional ONNX backbone loader inside `OmniVoice`
- phase-1 ONNX Runtime session helper
- backbone export wrapper and export CLI
- CLI flag support in `omnivoice-infer`
- API flag support in `omnivoice-api`
- voice cloning preserved by keeping prompt creation in PyTorch

This is a hybrid migration, not a full PyTorch removal.

### Export Results

Machine constraints:

- 2022 MacBook Air
- 8 GB RAM (`hw.memsize = 8589934592`)

Successful fixed-shape exports:

| Export | Result | Wall time | Max RSS |
|---|---|---:|---:|
| `batch=2, seq=64` | success | 52.41s | about 2.18 GB |
| `batch=2, seq=640` | success | 66.69s | about 2.49 GB |

Artifacts:

- `artifacts/onnx/omnivoice_backbone_bs2_seq640.onnx`
- external weight files emitted by PyTorch ONNX export in the same directory

The earlier `seq=1024` attempt did not complete, so it was not used for validation.

Successful dynamic export:

| Export | Result | Wall time | Max RSS |
|---|---|---:|---:|
| `dynamic seq, example batch=2, example seq=128` | success | 63.43s | about 1.86 GB |

Dynamic artifact used for parity:

- `artifacts/onnx_dynamic/omnivoice_backbone_dynamic.onnx`
- external weight files emitted alongside it in `artifacts/onnx_dynamic/`

### Measured Sequence Lengths

To avoid exporting a larger graph than necessary, I measured the actual prepared input lengths for the current test prompts.

Design voice:

| Case | Target audio tokens | Input sequence length |
|---|---:|---:|
| short sentence | 66 | 88 |
| longer sentence | 286 | 331 |

Voice cloning using `local_test_10sec.wav` plus provided `ref_text`:

| Case | Reference audio tokens | Target audio tokens | Input sequence length |
|---|---:|---:|---:|
| short target text | 259 | 59 | 372 |
| longer target text | 259 | 259 | 595 |

That is why the validated export target was `seq=640`.

### CoreML Execution Provider Result

ONNX Runtime is installed with CoreML support on this Mac:

```python
['CoreMLExecutionProvider', 'AzureExecutionProvider', 'CPUExecutionProvider']
```

However, CoreML EP did not actually work for this exported OmniVoice backbone.

Tested combinations all failed with the same fallback behavior:

- `ModelFormat=MLProgram`, `MLComputeUnits=ALL`
- `ModelFormat=NeuralNetwork`, `MLComputeUnits=ALL`
- `ModelFormat=NeuralNetwork`, `MLComputeUnits=CPUAndGPU`
- `ModelFormat=MLProgram`, `MLComputeUnits=CPUAndGPU`
- `RequireStaticInputShapes=1`
- `ModelCacheDirectory` enabled

Observed runtime behavior:

```text
EP Error SystemError : 20
Falling back to ['CPUExecutionProvider'] and retrying.
```

So on this machine, the ONNX path is currently CPU-only in practice.

### Fixed-Shape Root Cause

The original fixed-shape ONNX export was not quality-safe for short prompts.

Direct parity measurements showed:

| Comparison | Max abs diff | Mean abs diff | Argmax mismatch rate |
|---|---:|---:|---:|
| native-length PyTorch vs padded PyTorch | 41.63 | 1.95 | 0.8366 |
| padded PyTorch vs ONNX Runtime | 0.000412 | 0.000038 | 0.0 |

Interpretation:

- ONNX Runtime was faithfully reproducing the padded graph.
- The real quality bug came from padding a short request up to the fixed export length.
- That padding caused the pitch/spacing corruption heard in earlier ONNX audio tests.

Because of that, the runtime no longer pads requests to fit a fixed-shape ONNX export. Fixed-shape exports are now treated as exact-shape only.

### Dynamic-Sequence Parity Validation

The dynamic-sequence export removes the padding path and restores backbone parity at native prompt length.

Measured with `omnivoice/scripts/debug_onnx_parity.py` against `artifacts/onnx_dynamic/omnivoice_backbone_dynamic.onnx`:

| Comparison | Max abs diff | Mean abs diff | Argmax mismatch rate |
|---|---:|---:|---:|
| native-length PyTorch vs dynamic-shape PyTorch baseline | 0.0 | 0.0 | 0.0 |
| dynamic-shape PyTorch baseline vs ONNX Runtime | 0.000519 | 0.000038 | 0.0 |
| native-length PyTorch vs ONNX Runtime | 0.000519 | 0.000038 | 0.0 |

Interpretation:

- The dynamic ONNX backbone matches native-length PyTorch closely enough for backbone parity.
- The large quality regression from the earlier ONNX test was caused by fixed-shape padding, not by ORT fundamentally changing the backbone.

### End-to-End Runtime Validation

The ONNX path now works end to end for both design mode and clone mode.

CLI validation:

| Mode | Command shape | Result | Output |
|---|---|---|---|
| design | ONNX backbone, provider=`cpu`, `num_step=2` | success | `ort_short_num2.wav` |
| clone | ONNX backbone, provider=`cpu`, `num_step=2`, `ref_audio` + `ref_text` | success | `ort_clone_num2.wav` |

Measured outputs:

| File | Duration |
|---|---:|
| `ort_short_num2.wav` | 2.84s |
| `ort_clone_num2.wav` | 2.50s |

So voice cloning remains functional in the phase-1 hybrid path as long as `ref_text` is provided and Whisper ASR stays disabled.

Dynamic export status:

- the dynamic backbone loads through the same ONNX Runtime path
- fixed-shape padding is no longer used
- the dynamic export is now the recommended phase-1 artifact

Deterministic end-to-end generation check (`num_step=16`, `position_temperature=0`, `class_temperature=0`) on CPU:

| Check | Result |
|---|---|
| generated token mismatch rate | about `0.00189` |
| number of mismatched tokens | `1` out of `528` |

The one remaining mismatch appeared at:

- codebook layer `7`
- position `51`
- PyTorch token `625`
- ONNX token `467`

This means the dynamic export is not bit-exact through the full iterative decode, but it is dramatically closer than the broken fixed-shape path and no longer exhibits the gross spacing/pitch corruption mechanism caused by padding.

Deterministic comparison WAVs saved locally:

- `local/phase1_pt_det.wav`
- `local/phase1_ort_det.wav`

### Localhost API Validation

Validated ONNX-backed API server:

```bash
uv run omnivoice-api \
  --model k2-fsa/OmniVoice \
  --device mps \
  --no-asr \
  --onnx-backbone artifacts/onnx/omnivoice_backbone_bs2_seq640.onnx \
  --onnx-provider cpu \
  --ip 127.0.0.1 \
  --port 8002
```

Observed health response:

```json
{
  "status": "ok",
  "model": "k2-fsa/OmniVoice",
  "device": "mps",
  "sampling_rate": 24000,
  "asr_loaded": false,
  "onnx_backbone": "artifacts/onnx/omnivoice_backbone_bs2_seq640.onnx",
  "onnx_provider": "cpu"
}
```

### Throughput Comparison on This Mac

Apples-to-apples API test:

- same short sentence
- design mode
- same `num_step=2`
- same localhost request path
- same output duration: `2.67s`

| Backend | Request wall time | Output duration | Approx RTF |
|---|---:|---:|---:|
| PyTorch + MPS API | 10.491s | 2.67s | about 3.93 |
| ONNX Runtime + CPU API | 21.305s | 2.67s | about 7.98 |

Result:

- ONNX Runtime on this Mac is about 2.0x slower than the current PyTorch/MPS path for this validated short request.
- The reason is not the hybrid architecture by itself. The main issue is that CoreML EP is failing and ORT is falling back to CPU.

### Additional Warm Benchmark Update

User-ran warm-server comparison on 2026-04-10 with the same short design prompt:

```text
This is a short test of local text to speech.
```

Server startup observations from logs:

| Server | Startup window | Approx startup time |
|---|---|---:|
| ONNX Runtime + CPU API (`8002`) | `20:57:20.097` -> `20:57:30.160` | about 10.06s |
| PyTorch + MPS API (`8003`) | `20:57:52.198` -> `20:57:59.424` | about 7.23s |

Warm request timings:

| Backend | Run | Request wall time | Output duration | Approx RTF |
|---|---:|---:|---:|---:|
| ONNX Runtime + CPU API | 1 | 18.935730s | 2.84s | about 6.67 |
| ONNX Runtime + CPU API | 2 | 19.164227s | 2.84s | about 6.75 |
| PyTorch + MPS API | 1 | 4.007727s | 2.48s | about 1.62 |
| PyTorch + MPS API | 2 | 1.368771s | 2.48s | about 0.55 |

Additional CPU-only baseline:

| Backend | Request wall time |
|---|---:|
| PyTorch + CPU API (`8004`) | 15.269604s |

Interpretation:

- The newer warm runs show a larger performance gap than the earlier one-off measurement.
- PyTorch + MPS is the clear winner on this MacBook Air.
- PyTorch + CPU is still faster than the current ONNX Runtime + CPU path.
- Current backend ranking on this machine is:
  1. PyTorch + MPS
  2. PyTorch + CPU
  3. ONNX Runtime + CPU

This means the current ONNX phase-1 path is not only losing to MPS acceleration. It is also losing to plain PyTorch on CPU.

### Current Verdict

The phase-1 ONNX migration is complete enough to validate functionally:

- export works
- ONNX Runtime loading works
- CLI works
- API works
- voice cloning still works

But it is not performance-ready on this MacBook Air:

- CoreML EP currently fails with `SystemError : 20`
- the ORT path falls back to CPU
- CPU ORT is slower than both PyTorch/MPS and PyTorch/CPU

So the current state is:

1. the bad fixed-shape padded ONNX path is no longer acceptable and has been effectively superseded
2. the dynamic-sequence backbone export works and restores backbone parity
3. end-to-end dynamic decode is very close, but not fully bit-exact through all generation steps
4. phase 1 is complete enough to move forward with the dynamic export as the only acceptable ONNX backbone artifact
5. phase 1 still does not improve throughput on this machine
6. the next bottleneck to solve is CoreML EP compatibility or a different Apple-accelerated deployment path

## Phase 2 CoreMLExecutionProvider

Status as of 2026-04-11: CoreMLExecutionProvider is now working with the dynamic ONNX backbone on this Mac, but it is still slower than the existing PyTorch + MPS path for full generation.

### What Changed

The initial CoreML EP attempts failed with:

```text
EP Error SystemError : 20
```

That failure mode was tied to letting CoreML EP accept dynamic input shapes directly.

The working configuration is:

- dynamic ONNX backbone export
- CoreMLExecutionProvider enabled
- `RequireStaticInputShapes=1`
- `EnableOnSubgraphs=0`
- `ModelFormat=NeuralNetwork`
- `MLComputeUnits=ALL`
- `ModelCacheDirectory=<persistent cache dir>`
- ORT graph optimizations disabled for fidelity

This allows the dynamic ONNX model to keep native prompt length while still letting CoreML EP claim supported partitions.

### Runtime Integration

The ONNX runtime helper now:

- uses a persistent CoreML cache directory next to the ONNX model
- defaults CoreML EP to `NeuralNetwork` format on macOS
- forces static input shapes at the provider level
- exposes actual loaded ORT providers through the API health endpoint

Example health response now includes:

```json
{
  "onnx_provider": "coreml",
  "onnx_runtime_providers": ["CoreMLExecutionProvider", "CPUExecutionProvider"]
}
```

This confirms the session is not falling back to CPU-only at load time.

### CoreML Parity Check

The dynamic ONNX parity script with `--provider coreml` still shows clean backbone parity:

| Comparison | Max abs diff | Mean abs diff | Argmax mismatch rate |
|---|---:|---:|---:|
| native-length PyTorch vs ONNX Runtime CoreML-backed session | 0.000458 | 0.000034 | 0.0 |

So CoreML EP did not reintroduce the earlier quality bug.

### CoreML Runtime Benchmark

CoreML-backed ONNX server command:

```bash
/Users/ahmadsmacair/OmniVoice/.venv/bin/omnivoice-api \
  --model k2-fsa/OmniVoice \
  --device mps \
  --no-asr \
  --onnx-backbone /Users/ahmadsmacair/OmniVoice/artifacts/onnx_dynamic/omnivoice_backbone_dynamic.onnx \
  --onnx-provider coreml \
  --save-dir /Users/ahmadsmacair/OmniVoice/local/api_outputs/onnx_coreml \
  --ip 127.0.0.1 \
  --port 8006
```

Measured localhost `/generate` timings at `num_step=16` for the same short design prompt:

| Backend | Run | Wall time |
|---|---:|---:|
| ONNX Runtime + CoreML EP | 1 | 30.453s |
| ONNX Runtime + CoreML EP | 2 | 25.521s |
| PyTorch + MPS | 1 | 12.632s |
| PyTorch + MPS | 2 | 5.095s |

Interpretation:

- CoreML EP is active and quality is preserved.
- On this 2022 MacBook Air, the current hybrid ONNX + CoreML path is still slower than PyTorch + MPS.
- The gap is smaller than the old broken fixed-shape ONNX path in terms of correctness, but not yet in terms of speed.

### Phase 2 Verdict

What is now true:

- CoreMLExecutionProvider is no longer broken for the dynamic backbone.
- The dynamic ONNX backbone can run through CoreML EP on macOS.
- Audio quality remains aligned with the PyTorch baseline.

What is still not true:

- CoreML-backed ONNX is not yet faster than PyTorch + MPS on this machine.
- The current stack is still hybrid; the decoder remains in PyTorch.
- This is still not a true iPhone deployment path.

So the current Apple-side ranking on this Mac is:

1. PyTorch + MPS
2. ONNX Runtime + CoreML EP
3. PyTorch + CPU
4. ONNX Runtime + CPU

The next likely optimization path is no longer “fix CoreML EP”. That part is now working. The next question is whether direct Core ML conversion of the backbone and decoder can beat the hybrid ORT path, or whether the PyTorch/MPS path simply remains the best Mac-local option while the iPhone path is built separately.

## Current Plan

As of 2026-04-11, the project is no longer blocked on ONNX quality parity. The main blocker is now the backbone runtime: it needs to move to direct native Core ML so Apple hardware can be used without ORT/CoreML partitioning overhead.

### What is done

- Dynamic ONNX backbone export works.
- Fixed-shape padding is no longer the accepted path.
- ONNX Runtime parity is good enough for design-mode audio quality.
- CoreMLExecutionProvider is active for the dynamic backbone.
- The runtime now enforces true CoreML instead of silently accepting CPU fallback when `provider=coreml` is requested.
- The direct Core ML path is now the intended destination for the backbone, with ORT/CoreML kept mainly as a validation and comparison path.
- A productionized direct native Core ML backbone export/runtime path now exists for local benchmarking.

### What is still missing

- The audio decoder no longer has to run in PyTorch; an ONNX Runtime path now exists, but it still needs more end-to-end validation in clone mode and more throughput testing.
- Voice-clone prompt creation still runs in PyTorch.
- The current server is still hybrid, not a real iOS deployment path.
- CoreML-backed ONNX is still slower than PyTorch + MPS on this Mac.
- Verified dynamic-CoreML backbone timings are still roughly `11-25s` depending on prompt length, with the backbone dominating almost all of that time.
- The current fixed `seq128` ORT/CoreML artifact does not initialize with true CoreML on this machine.
- The direct Core ML backbone path is implemented, but it is still in the first benchmarking phase and not yet the default runtime path for all requests.
- The direct native Core ML backbone path is now implemented, but it still needs broader benchmarking and decoder/cloning integration before it can be treated as the default Apple path.

### Phase 3: Direct Core ML Backbone

Goal:

- Move the backbone off ONNX Runtime and onto direct native Core ML so GPU/ANE usage is controlled by Apple's runtime rather than ORT graph partitioning.

Implementation steps:

1. Productionize direct Core ML backbone export from the current prototype script.
2. Export the backbone as native `.mlpackage` artifacts, starting with the backbone because it is the dominant bottleneck.
3. Add a native Core ML runtime path for the backbone in the local inference stack so it can be benchmarked against ORT/CoreML and PyTorch/MPS.
4. Benchmark direct Core ML backbone latency and warm-run behavior on Mac before touching iPhone integration.
5. After the backbone path is trusted, keep the decoder on CPU ORT temporarily and continue clone-mode validation.

Exit criteria:

- A native Core ML backbone artifact runs end to end with correct audio quality.
- The direct Core ML backbone beats or clearly characterizes the ORT/CoreML backbone path.
- The resulting path is trustworthy enough to benchmark for iPhone feasibility.

### Phase 3 status as of 2026-04-11

The decoder export and parity task is now implemented.

What was added:

- `omnivoice/models/onnx_decoder.py`
- `omnivoice/scripts/export_onnx_decoder.py`
- `omnivoice/scripts/debug_decoder_parity.py`
- runtime support in the model, CLI, and API for `--onnx-decoder`

The decoder can now be loaded independently from the backbone:

```bash
/Users/ahmadsmacair/OmniVoice/.venv/bin/python -m omnivoice.cli.infer \
  --model k2-fsa/OmniVoice \
  --device mps \
  --onnx_backbone /Users/ahmadsmacair/OmniVoice/artifacts/onnx_dynamic/omnivoice_backbone_dynamic.onnx \
  --onnx_provider coreml \
  --onnx_decoder /Users/ahmadsmacair/OmniVoice/artifacts/onnx_dynamic/omnivoice_decoder_dynamic.onnx \
  --onnx_decoder_provider coreml \
  --text "This is a short test of local text to speech." \
  --instruct "female, american accent" \
  --num_step 4 \
  --output /Users/ahmadsmacair/OmniVoice/local/phase3_full_coreml.wav
```

#### Decoder export artifact

Successful dynamic decoder export:

```bash
/Users/ahmadsmacair/OmniVoice/.venv/bin/python -m omnivoice.scripts.export_onnx_decoder \
  --model k2-fsa/OmniVoice \
  --output artifacts/onnx_dynamic/omnivoice_decoder_dynamic.onnx \
  --seq-len 128 \
  --dynamic-seq
```

Artifact:

- `/Users/ahmadsmacair/OmniVoice/artifacts/onnx_dynamic/omnivoice_decoder_dynamic.onnx`

#### Decoder parity validation

CPU ORT parity is extremely tight.

Command:

```bash
/Users/ahmadsmacair/OmniVoice/.venv/bin/python -m omnivoice.scripts.debug_decoder_parity \
  --model k2-fsa/OmniVoice \
  --onnx-decoder artifacts/onnx_dynamic/omnivoice_decoder_dynamic.onnx \
  --provider cpu \
  --save-prefix local/phase3_decoder_det
```

Result:

```json
{
  "provider": "cpu",
  "loaded_runtime_providers": ["CPUExecutionProvider"],
  "random_codes": {
    "pt_num_samples": 122880,
    "ort_num_samples": 122880,
    "shared_num_samples": 122880,
    "max_abs": 0.00000370,
    "mean_abs": 0.00000011
  },
  "generated_codes": {
    "pt_num_samples": 63360,
    "ort_num_samples": 63360,
    "shared_num_samples": 63360,
    "max_abs": 0.00001157,
    "mean_abs": 0.00000018
  },
  "generated_token_seq_len": 66
}
```

Saved WAVs for listening:

- `/Users/ahmadsmacair/OmniVoice/local/phase3_decoder_det_pt.wav`
- `/Users/ahmadsmacair/OmniVoice/local/phase3_decoder_det_ort.wav`

Both files are valid `24000` Hz WAVs with the same duration:

- `2.64s`

#### CoreML EP note for the decoder

The first CoreML decoder configuration was not numerically safe.

- `ModelFormat=NeuralNetwork` with `MLComputeUnits=ALL` produced non-finite decoder output.

The working decoder-side CoreML configuration is:

- `ModelFormat=MLProgram`
- `MLComputeUnits=ALL`
- `RequireStaticInputShapes=1`
- `EnableOnSubgraphs=0`

With that configuration, CoreML-backed decoder output becomes finite and reasonably close:

```json
{
  "provider": "coreml",
  "loaded_runtime_providers": ["CoreMLExecutionProvider", "CPUExecutionProvider"],
  "random_codes": {
    "max_abs": 0.01199312,
    "mean_abs": 0.00033379
  },
  "generated_codes": {
    "max_abs": 0.02624599,
    "mean_abs": 0.00049674
  }
}
```

That is materially worse than CPU ORT parity, so the decoder-side CoreML path should still be treated as in-progress rather than fully trusted for production audio judgments.

#### Phase 3 checkpoint

What is now true:

- the decoder has a productionized ONNX export path
- the runtime can execute the decoder without PyTorch
- the CLI and API can load the decoder independently
- CPU decoder parity is strong enough to trust as a migration step
- one full design-mode inference run with both ONNX backbone and ONNX decoder loaded completed successfully

What is still not true:

- clone mode has not yet been revalidated end to end with the ONNX decoder path
- decoder CoreML parity is not yet as tight as CPU parity
- the entire mobile runtime path is still not complete because clone-prompt encoding remains in PyTorch

#### Fixed-shape bucket experiment (CoreML backbone)

Status as of 2026-04-11: the original `seq128` fixed-bucket timings looked promising, but after adding strict provider validation they cannot be treated as proof that the fixed bucket was actually running on true CoreML.

Fixed-shape backbone artifact:

- `/Users/ahmadsmacair/OmniVoice/artifacts/onnx_fixed/seq128/omnivoice_backbone_bs2_seq128.onnx`

Server notes:

- User tested via a server on `http://127.0.0.1:8008`.
- The server saved WAVs under: `/Users/ahmadsmacair/OmniVoice/local/api_outputs/onnx_fixed128/` (from `x-omnivoice-saved-path`).

Measured design-mode timings (`num_step=16`) from the earlier `8008` experiment:

- Text: `Hello world.` -> `fixed128_total=3.629878`
- Text: `This is a short test of local text to speech.` -> `fixed128_total=4.650628`
- Text: `Today I am testing whether OmniVoice can generate natural sounding speech quickly enough for a mobile app, while keeping the voice clear, expressive, and stable across a longer sentence.` -> `fixed128_long=14.736434`

Saved output for the long-sentence run:

- `/Users/ahmadsmacair/OmniVoice/local/api_outputs/onnx_fixed128/20260411-162019_design_today-i-am-testing-whether-omnivoice-can_6a0ff71d.wav`

One log line reported `dynamic_total=0.000482`, which is not plausible for real inference and is likely due to a curl formatting/line-continuation issue rather than an actual generation runtime.

What later validation showed:

- the fixed `seq128` artifact does **not** initialize with true `CoreMLExecutionProvider` on this machine
- ORT reports `SystemError : 20` during CoreML EP setup for that artifact
- before the strict-provider fix, ORT would silently fall back to CPU and continue
- so the earlier `seq128` timing numbers should be treated as "interesting earlier behavior" rather than verified fixed-bucket CoreML throughput
- the old `fixed128_long=14.736434` result is very likely explained by fallback behavior in the earlier runtime:
  - the fixed bucket could not run on true CoreML
  - that server was started without `--onnx-backbone-allow-fixed-padding`
  - for a long sentence, `seq128` cannot serve the request directly
  - fallback warnings were deduplicated, so later requests could use a PyTorch backbone path without printing the same warning again

#### Verified dynamic CoreML timings

After strict CoreML enforcement and per-request backbone-path logging were added, the clean dynamic-CoreML measurements on `http://127.0.0.1:8007` are:

- Text: `Hello world.` -> `11.590467`
- Text: `Hello world. this is just a test to see how this thing works` -> `24.746319`

Server-side generation profiles for the same dynamic-CoreML server:

- Short request:
  - `total=12.681s`
  - `iterative=12.285s`
  - `backbone=11.907s`
  - `decoder=0.378s`
- Warm short request:
  - `total=11.571s`
  - `iterative=11.223s`
  - `backbone=11.150s`
  - `decoder=0.344s`
- Longer request:
  - `total=24.736s`
  - `iterative=24.095s`
  - `backbone=23.898s`
  - `decoder=0.626s`

These runs explicitly logged:

- `onnx_backbone_path=/Users/ahmadsmacair/OmniVoice/artifacts/onnx_dynamic/omnivoice_backbone_dynamic.onnx`
- `onnx_backbone_fixed_seq_len=None`
- `onnx_backbone_providers=['CoreMLExecutionProvider', 'CPUExecutionProvider']`

So these are the first trustworthy dynamic ONNX + CoreML backbone timings after the runtime was made strict about rejecting fake-CoreML sessions.

#### Bucket routing implementation

Status as of 2026-04-11: the inference runtime now supports loading multiple ONNX backbone exports and routing each request to the smallest compatible backbone bucket.

Implemented behavior:

- `--onnx-backbone` can now point to:
  - a single ONNX file
  - a directory containing multiple `.onnx` backbone exports
  - a comma-separated list of ONNX paths
- the runtime loads all matching backbone sessions
- requests are routed to the smallest fixed-shape session that can satisfy the request
- if no fixed bucket can satisfy the request, the runtime falls back to a dynamic session if one is loaded

Current selection rule:

- prefer the smallest compatible fixed `seq_len`
- otherwise use the loaded dynamic export

This is now implemented in:

- `omnivoice/utils/onnx_runtime.py`
- `omnivoice/models/omnivoice.py`
- `omnivoice/cli/api_server.py`
- `omnivoice/cli/infer.py`

Smoke test result:

- loaded:
  - `artifacts/onnx_fixed/seq128/omnivoice_backbone_bs2_seq128.onnx`
  - `artifacts/onnx_dynamic/omnivoice_backbone_dynamic.onnx`
- selection behavior:
  - `seq_len=64` -> fixed `seq128`
  - `seq_len=128` -> fixed `seq128`
  - `seq_len=129` -> dynamic backbone
  - `seq_len=300` -> dynamic backbone

Important caveat from the smoke test:

- on this machine, directly loading the current `seq128` fixed artifact through CoreML EP triggers `SystemError : 20`
- after the strict-provider fix, that failed fixed session is now rejected instead of being treated as usable
- the dynamic artifact still loads with `CoreMLExecutionProvider` + `CPUExecutionProvider`

So the bucket router itself is implemented and selecting correctly, but fixed buckets are only useful if they can actually initialize with CoreML. Right now the current `seq128` ORT/CoreML artifact does not.

#### Strict CoreML enforcement

Status as of 2026-04-11: the runtime now treats `provider=coreml` as a hard requirement instead of allowing silent CPU fallback.

What changed:

- ONNX backbone and decoder session creation now verifies the requested provider actually loaded
- if `provider=coreml` is requested and ORT only loads `CPUExecutionProvider`, that session is rejected
- a single explicit fixed-bucket path that cannot load on CoreML now fails startup
- a mixed list like `fixed + dynamic` now keeps only the sessions that truly loaded with CoreML

Validated behavior:

- single fixed path:
  - `/Users/ahmadsmacair/OmniVoice/artifacts/onnx_fixed/seq128/omnivoice_backbone_bs2_seq128.onnx`
  - `provider=coreml`
  - result: startup fails with `RuntimeError` because actual providers are `['CPUExecutionProvider']`
- mixed fixed + dynamic path:
  - fixed `seq128` session is skipped
  - dynamic backbone remains loaded with `['CoreMLExecutionProvider', 'CPUExecutionProvider']`

The generation profile log now also records which ONNX backbone session actually served the request:

- `onnx_backbone_path`
- `onnx_backbone_fixed_seq_len`
- `onnx_backbone_providers`

This removes the earlier ambiguity around whether a request used `dynamic-coreml`, `fixed-cpu`, or a PyTorch fallback.

#### Direct native Core ML backbone implementation

Status as of 2026-04-11: the direct native Core ML backbone path is implemented in code, but the currently exported backbone `.mlpackage` is not reliable enough to call the full Apple path complete.

What was added:

- `omnivoice/models/coreml_backbone.py`
- `omnivoice/utils/coreml_runtime.py`
- `omnivoice/scripts/export_coreml_backbone.py`
- CLI/API flags for `--coreml-backbone` and `--coreml-compute-units`
- health/profile reporting for native Core ML backbone selection

Packaging:

- optional dependency group:
  - `coreml`
- install command:

```bash
uv sync --extra coreml --extra onnx
```

Exporter command:

```bash
uv run omnivoice-export-coreml-backbone \
  --model k2-fsa/OmniVoice \
  --output /Users/ahmadsmacair/OmniVoice/artifacts/coreml_dynamic/omnivoice_backbone_dynamic_seq128.mlpackage \
  --seq-len 128 \
  --batch-size 2 \
  --dynamic-seq \
  --target macos13
```

This produces:

- `/Users/ahmadsmacair/OmniVoice/artifacts/coreml_dynamic/omnivoice_backbone_dynamic_seq128.mlpackage`
- `/Users/ahmadsmacair/OmniVoice/artifacts/coreml_dynamic/omnivoice_backbone_dynamic_seq128.metadata.json`

Earlier local CLI smoke test:

```bash
uv run python -m omnivoice.cli.infer \
  --model k2-fsa/OmniVoice \
  --device mps \
  --coreml_backbone /Users/ahmadsmacair/OmniVoice/artifacts/coreml_dynamic/omnivoice_backbone_dynamic_seq128.mlpackage \
  --coreml_compute_units all \
  --onnx_decoder /Users/ahmadsmacair/OmniVoice/artifacts/onnx_dynamic/omnivoice_decoder_dynamic.onnx \
  --onnx_decoder_provider cpu \
  --text "Hello world." \
  --instruct "female, american accent" \
  --num_step 1 \
  --output /Users/ahmadsmacair/OmniVoice/local/coreml_direct_test.wav
```

Observed result:

- generation completed successfully
- saved file:
  - `/Users/ahmadsmacair/OmniVoice/local/coreml_direct_test.wav`
- generation profile:
  - `total=7.039s`
  - `iterative=6.632s`
  - `backbone=6.329s`
  - `decoder=0.364s`
  - `coreml_backbone_path=/Users/ahmadsmacair/OmniVoice/artifacts/coreml_dynamic/omnivoice_backbone_dynamic_seq128.mlpackage`
  - `coreml_backbone_compute_units=all`

Caveats:

- `coremltools==9.0` warns that `torch==2.8.0` is not a tested combination
- local runs print an Apple runtime cache warning:
  - `error: attempting to parse a byte at the end of the bytecode`
  - generation still completed successfully in the validated smoke test

Current blocker discovered during revalidation:

- the existing direct Core ML backbone artifact at
  - `/Users/ahmadsmacair/OmniVoice/artifacts/coreml_dynamic/omnivoice_backbone_dynamic_seq128.mlpackage`
  now fails direct `predict()` in the native runtime with:
  - `RuntimeError: Error compiling model: "compiler error: Error reading protobuf spec. validator error: Model specification version field missing or corrupt."`
- a fresh re-export attempt to a new backbone `.mlpackage` failed during save because the machine ran out of disk space:
  - `LLVM ERROR: IO failure on output stream: No space left on device`

So the direct Core ML backbone implementation exists, but the currently validated state is:

- export path implemented
- runtime path implemented
- CLI/API wiring implemented
- backbone artifact reliability is still blocked by the local environment and needs a clean re-export

#### Direct native Core ML decoder implementation

Status as of 2026-04-11: the native Core ML decoder path is now implemented and validated locally.

What was added:

- `omnivoice/models/coreml_decoder.py`
- `omnivoice/scripts/export_coreml_decoder.py`
- decoder support inside `omnivoice/utils/coreml_runtime.py`
- model integration in `omnivoice/models/omnivoice.py`
- CLI/API flags for:
  - `--coreml_decoder`
  - `--coreml_decoder_compute_units`
- compatibility wrappers:
  - `scripts/export_audio_decoder.py`
  - `scripts/export_coreml_decoder.py`

Exporter command:

```bash
uv run omnivoice-export-coreml-decoder \
  --model k2-fsa/OmniVoice \
  --output /Users/ahmadsmacair/OmniVoice/artifacts/coreml_dynamic/omnivoice_decoder_dynamic_seq128.mlpackage \
  --seq-len 128 \
  --dynamic-seq \
  --target macos13
```

This produces:

- `/Users/ahmadsmacair/OmniVoice/artifacts/coreml_dynamic/omnivoice_decoder_dynamic_seq128.mlpackage`
- `/Users/ahmadsmacair/OmniVoice/artifacts/coreml_dynamic/omnivoice_decoder_dynamic_seq128.metadata.json`

Validation:

- export completed successfully
- runtime load completed successfully
- direct dummy prediction completed successfully
- observed decoder runtime output:
  - output shape: `(1, 1, 15360)`
  - output dtype: `torch.float32`

Current local package sizes:

- backbone `.mlpackage`: about `310M`
- decoder `.mlpackage`: about `41M`

### Phase 4: Full Native Apple Path

Goal:

- Build the first realistic iPhone deployment candidate.

Implementation steps:

1. Re-export the direct Core ML backbone cleanly once disk headroom is available.
2. Revalidate native backbone `predict()` and then re-run full generation with:
   - direct Core ML backbone
   - direct Core ML decoder
3. Build a minimal native harness that loads the backbone and decoder artifacts and runs one full generation request.
4. Measure wall time, first-audio latency, memory use, and thermal behavior on a real iPhone.
5. Only after that, design the React Native bridge around the winning runtime path.

Exit criteria:

- One end-to-end prompt works on-device without Python.
- Backbone and decoder both run through the chosen native Apple path.
- Clone-critical runtime pieces are either migrated or explicitly moved off-device by design.

### Immediate next task

The next implementation task is now:

- free disk headroom, regenerate the direct Core ML backbone artifact, and revalidate full native generation with both backbone and decoder on Core ML

After that:

- decide whether clone-prompt creation stays off-device or also needs its own Apple-native migration path

### Repo-wide codebase audit (2026-04-11)

I audited the tracked repo files to answer a narrower question than "what does the project do": which parts actually matter for making OmniVoice fast enough and complete enough for iOS real-time inference.

Scope:

- audited tracked repo files (`rg --files` -> `85` files)
- excluded environment/generated noise like `.venv/`, `__pycache__/`, and large generated artifacts when deciding what counts as the codebase

#### What files actually control iOS inference feasibility

These are the files that materially determine whether local Apple execution can become fast and mobile-ready:

- `omnivoice/models/omnivoice.py`
- `omnivoice/utils/onnx_runtime.py`
- `omnivoice/models/onnx_backbone.py`
- `omnivoice/models/onnx_decoder.py`
- `omnivoice/cli/api_server.py`
- `omnivoice/cli/infer.py`
- `omnivoice/utils/audio.py`
- `omnivoice/utils/text.py`
- `omnivoice/utils/duration.py`
- `omnivoice/utils/voice_design.py`
- `omnivoice/utils/lang_map.py`
- `omnivoice/scripts/export_onnx_backbone.py`
- `omnivoice/scripts/export_onnx_decoder.py`
- `omnivoice/scripts/debug_onnx_parity.py`
- `omnivoice/scripts/debug_decoder_parity.py`
- `scripts/export_coreml_backbone.py`
- `scripts/export_audio_decoder.py`
- `docs/ios_deployment_plan.md`

Everything else in the repo is either:

- training-only (`omnivoice/training/*`, `omnivoice/data/*`)
- evaluation-only (`omnivoice/eval/*`)
- dataset preparation (`omnivoice/scripts/extract_audio_tokens*.py`, `jsonl_to_webdataset.py`, `denoise_audio.py`)
- examples/docs/config glue

Those parts matter for training and benchmarking, but they are not the current blockers for iPhone inference.

#### Main technical conclusion from the full audit

The repo is still a Python-first inference system. Even after the ONNX work, the current mobile path is not "a converted model"; it is still a hybrid runtime whose control plane lives in Python.

The bottleneck and blockers are now very clear:

1. The repeated backbone call inside `_generate_iterative(...)` is still the dominant latency cost.
2. Voice cloning is still not fully mobile-ready because prompt creation depends on PyTorch audio-token encoding, and clone mode without `ref_text` still depends on Whisper.
3. The decoder is migrated to ONNX, but the best-fidelity path is still ORT CPU, not a clearly faster Apple-native decoder backend.
4. The current direct Core ML export scripts in `scripts/` are prototype-level utilities, and the next work should turn them into the primary backbone deployment path rather than leaving them as side experiments.
5. None of the Python-side orchestration has been reimplemented in Swift or a native mobile runtime yet.

#### What the audit says about the current phases

Phase 1:

- effectively complete for design-mode quality parity
- dynamic ONNX backbone fixed the bad-audio regression from padded fixed-shape export

Phase 2:

- partially complete
- CoreML EP works
- fixed-shape bucket experiments are now the first real sign of Apple-side throughput improvement

Phase 3:

- partially complete
- decoder export exists and parity is strong on CPU ORT
- clone mode is still not fully revalidated through the migrated path

Phase 4:

- not started as an actual native app/runtime
- still only documented/analyzed

#### The most important repo-level insight

The backbone runtime path is now the strongest lever in the whole codebase.

Why:

- profiling in `omnivoice/models/omnivoice.py` already showed the backbone dominates runtime
- dynamic ONNX + CoreML preserved quality but stayed slow
- ORT/CoreML still leaves too much of the performance question inside ORT partitioning and fallback behavior
- direct native Core ML is now the cleanest way to answer whether Apple hardware can actually make the backbone fast

So the next performance phase should focus on the backbone shape strategy, not on post-processing, docs, training code, or evaluation code.

#### Next steps, in order

1. Productionize direct native Core ML for the backbone.

   Start with:

   - export `.mlpackage` backbone artifacts
   - add a native runtime path
   - benchmark direct Core ML against ORT/CoreML and PyTorch/MPS

   The backbone is the dominant bottleneck, so this is the highest-value next move.

2. Revalidate clone mode end to end on the migrated path.

   Required test shape:

   - ONNX/CoreML backbone
   - ONNX decoder
   - `ref_text` always supplied

   Do not treat mobile clone mode as viable until this passes cleanly.

3. Decide how clone prompt creation will work on device.

   There are only two serious options:

   - export and migrate the reference-audio encoder path too
   - or precompute reusable clone prompts off-device/server-side

   If on-device cloning is mandatory, this becomes a first-class migration task.

4. Quantize the backbone and decoder.

   Real iPhone deployment does not work with the current model sizes.

   Quantization is not optional if the goal is on-device inference with reasonable memory headroom.

   Start with:

   - backbone first
   - INT8 before more aggressive compression
   - parity and listening checks after each change

5. Compare ORT CoreML EP against direct Core ML artifacts.

   The repo now has enough evidence to justify this as a real decision point.

   If ORT CoreML remains slower than expected after bucket routing and quantization, move the backbone to direct `.mlpackage` benchmarking instead of assuming ORT will eventually catch up.

6. Build a minimal native iOS harness before touching React Native integration.

   The first native test should do only this:

   - tokenize text
   - build the prompt
   - run the iterative loop
   - decode waveform
   - measure wall time, memory, and first-audio latency

   React Native should come after the native runtime path is proven, not before.

#### What should not be the focus right now

These are not the next bottlenecks according to the repo audit:

- training code changes
- evaluation scripts
- dataset tokenization scripts
- API server features
- post-processing tweaks
- Gradio/demo work

Those files are not what is keeping iPhone real-time from happening.

#### Practical roadmap after this audit

The next implementation path should be:

1. bucketed fixed-shape CoreML backbone routing
2. clone-mode revalidation on the migrated path
3. quantization experiments
4. direct Core ML vs ORT CoreML decision
5. native iOS harness

That is now the cleanest path to answer the real question: can OmniVoice become fast enough on iPhone without breaking voice quality or clone behavior.
