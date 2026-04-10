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
- Avoid Whisper ASR and avoid voice cloning initially.
- Start with auto/design voice only.
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

## Recommended Prototype Plan

1. Do not start with voice cloning.

   Test auto/design voice first with the smallest useful model set. Avoid the voice-cloning encoder and avoid Whisper. Voice cloning can be evaluated later after the basic latency and memory profile is known.

2. Build a tiny React Native benchmark app with `onnxruntime-react-native`.

   The benchmark should measure:

   - cold load time
   - warm generation latency
   - real-time factor
   - peak RAM
   - app bundle/model download size
   - battery and thermal behavior after repeated generations
   - audio quality at 8, 16, and 32 steps if supported

3. Use short utterances first.

   Suggested test strings:

   - 2 seconds of speech
   - 5 seconds of speech
   - 10 seconds of speech

   Long-form generation is less important than proving the base path can run within mobile constraints.

4. Define a minimum supported device.

   Do not judge by a flagship device only. Pick the actual lowest-end iPhone and Android device you are willing to support. Test on real hardware, not only a simulator.

5. Use a pass/fail threshold.

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
- Third-party ONNX export candidate: https://huggingface.co/Gigsu/vocoloco-onnx
