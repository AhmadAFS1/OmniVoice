# OmniVoice iOS Real-Time Deployment Plan

**Created:** 2026-04-11
**Goal:** Real-time text-to-speech inference on iPhone (iOS 17+, A16+ / M-series)

---

## 1. Current State

### What exists today

| Component | Status | Runtime | Notes |
|-----------|--------|---------|-------|
| ONNX backbone (dynamic) | ✅ exported | CoreML EP / CPU | Quality matches PyTorch |
| ONNX decoder (dynamic) | ✅ exported | CPU ORT | CoreML decoder had NaN issues |
| API server with ONNX flags | ✅ working | Python + FastAPI | `--onnx-backbone`, `--onnx-decoder` |
| Parity validation scripts | ✅ working | Python | `debug_onnx_parity.py`, `debug_decoder_parity.py` |
| Design mode (ONNX path) | ✅ validated | Hybrid | Same quality as PyTorch |
| Clone mode (ONNX path) | ⚠️ not validated | Hybrid | Not yet tested end-to-end with ONNX decoder |
| Native iOS app | ❌ does not exist | — | — |
| CoreML `.mlpackage` models | ❌ not converted | — | Only ONNX via CoreML EP |
| Swift inference logic | ❌ does not exist | — | — |

### Current inference pipeline (Python, hybrid ONNX)

```
Python orchestration
  ├─ PyTorch: text tokenizer (AutoTokenizer, vocab 151K)
  ├─ PyTorch: prompt preparation (_prepare_inference_inputs)
  ├─ PyTorch: attention mask construction
  ├─ PyTorch: voice clone audio encoding (audio_tokenizer.encode)
  │
  ├─ ONNX Runtime: backbone × 16 iterations (CoreML EP)
  │   └─ Each iteration: CFG batch (2 × codebooks × seq_len)
  │
  ├─ ONNX Runtime: audio decoder (CPU)
  │
  └─ Python/NumPy: post-processing
      ├─ silence removal (energy-based VAD)
      ├─ RMS normalization
      ├─ fade in/out
      └─ cross-fade for chunked audio
```

### Current performance (MacBook Air M2, design mode, num_step=16)

| Backend | Wall time | Quality |
|---------|-----------|---------|
| PyTorch + MPS | ~5 sec | ✅ reference |
| ONNX + CoreML EP backbone + CPU decoder | ~25 sec | ✅ matches PyTorch |
| ONNX + CPU backbone + CPU decoder | ~25-30 sec | ✅ matches PyTorch |

The ONNX path is **5× slower** than PyTorch+MPS on Mac. This is the baseline
we need to beat dramatically before iOS real-time is possible.

---

## 2. What "Real-Time" Means

For TTS, real-time means **RTF < 1.0**: generating N seconds of audio in less
than N seconds of wall time. For good UX, we want:

| Metric | Target | Stretch |
|--------|--------|---------|
| RTF | < 1.0 | < 0.5 |
| Time-to-first-audio | < 3 sec | < 1.5 sec |
| Model load (cold start) | < 5 sec | < 3 sec |
| Memory (peak) | < 2 GB | < 1.5 GB |
| App bundle size (models) | < 500 MB | < 300 MB |

The published RTF of 0.025 is on an H20 GPU. iPhone A17 Pro has roughly
**15-20 TOPS** (ANE) + GPU. Server H20 has ~150 TFLOPS FP16. That is a
~10-50× compute gap depending on the operation.

---

## 3. Blockers to iOS Real-Time

### Blocker 1: Model Size

| Artifact | Current size | Target (quantized) |
|----------|-------------|-------------------|
| Backbone (Qwen3-0.6B) | ~2.45 GB (FP32) / ~1.2 GB (FP16) | ~300-600 MB (INT4/INT8) |
| Audio decoder | ~806 MB (FP32) / ~400 MB (FP16) | ~100-200 MB (INT8) |
| Tokenizer vocab | ~11 MB | ~11 MB (no change) |
| **Total** | **~3.27 GB** | **~400-800 MB** |

iPhones have 6-8 GB RAM total. A 3.27 GB model leaves almost nothing for the
OS and app. **Quantization is mandatory.**

### Blocker 2: Iterative Generation Loop

OmniVoice runs the backbone **16-32 times** per generation. Each call processes
a batch of 2 (conditional + unconditional for CFG) with shape
`[2, 8_codebooks, seq_len]`.

For a ~5 second utterance at 75 tokens/sec = 375 audio tokens.
Total sequence ≈ style(~20) + text(~50) + audio(375) ≈ 445 tokens.

Each backbone call: `[2, 8, 445]` input → full Transformer forward pass
(28 layers, 1024 hidden, 16 heads).

**16 iterations × 28 layers × 445 tokens × 2 batch = massive compute.**

### Blocker 3: No Streaming Architecture

The model generates all tokens before decoding any audio. There is no way to
start playing audio while generation is still running. For long text, this
means the user waits for the entire generation to complete.

### Blocker 4: Voice Clone Encoder

Clone mode requires encoding reference audio through the HiggsAudioV2 encoder.
This is ~806 MB and runs once per generation. It is not yet exported to ONNX.

### Blocker 5: Python-Only Orchestration

The entire generation loop (`_generate_iterative`), mask scheduling,
token selection, CFG combination, and post-processing are Python code in
`omnivoice.py`. None of this exists in Swift.

### Blocker 6: Text Tokenizer

The tokenizer is a SentencePiece/BPE model loaded via HuggingFace
`AutoTokenizer`. iOS needs a native tokenizer implementation.

---

## 4. Architecture for iOS

### Target stack

```
┌─────────────────────────────────────────────────────┐
│                    Swift App                         │
│                                                      │
│  ┌──────────────────┐  ┌──────────────────────────┐ │
│  │  Text Tokenizer   │  │  Prompt Builder          │ │
│  │  (swift-          │  │  (Swift reimplementation  │ │
│  │   tokenizers or   │  │   of _prepare_inference   │ │
│  │   custom BPE)     │  │   _inputs)                │ │
│  └────────┬──────────┘  └───────────┬──────────────┘ │
│           │                         │                 │
│           ▼                         ▼                 │
│  ┌───────────────────────────────────────────────┐   │
│  │  Generation Loop (Swift)                       │   │
│  │  for step in 0..<num_step:                     │   │
│  │    logits = backbone.predict(input_ids,        │   │
│  │                              audio_mask,       │   │
│  │                              attention_mask)   │   │
│  │    tokens = unmask_schedule(logits, step)      │   │
│  └──────────────────────┬────────────────────────┘   │
│                         │                             │
│                         ▼                             │
│  ┌───────────────────────────────────────────────┐   │
│  │  CoreML Backbone (.mlpackage)                  │   │
│  │  Quantized INT4/INT8, runs on ANE+GPU          │   │
│  │  Input: [2, 8, S] + [2, S] + [2, 1, S, S]    │   │
│  │  Output: [2, 8, S, 1025]                       │   │
│  └───────────────────────────────────────────────┘   │
│                         │                             │
│                         ▼                             │
│  ┌───────────────────────────────────────────────┐   │
│  │  CoreML Audio Decoder (.mlpackage)             │   │
│  │  Quantized INT8, single pass                   │   │
│  │  Input: [1, 8, T]  Output: [1, 1, T×320]     │   │
│  └──────────────────────┬────────────────────────┘   │
│                         │                             │
│                         ▼                             │
│  ┌───────────────────────────────────────────────┐   │
│  │  Audio Post-Processing (Accelerate framework)  │   │
│  │  - RMS normalization                           │   │
│  │  - Fade in/out                                 │   │
│  │  - Optional silence trimming                   │   │
│  └──────────────────────┬────────────────────────┘   │
│                         │                             │
│                         ▼                             │
│  ┌───────────────────────────────────────────────┐   │
│  │  AVAudioEngine playback                        │   │
│  └───────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
```

---

## 5. Implementation Phases

### Phase A: Model Optimization (Python-side, before any Swift)

**Goal:** Reduce model size and per-iteration cost while preserving quality.

#### A1. Quantize backbone to INT4 (palettized) or INT8

```python
# Using coremltools post-training quantization
import coremltools as ct
from coremltools.optimize.coreml import (
    OpPalettizerConfig,
    OptimizationConfig,
    palettize_weights,
)

# Load the FP16 CoreML model
mlmodel = ct.models.MLModel("OmniVoiceBackbone.mlpackage")

# 4-bit palettization (biggest size reduction)
config = OptimizationConfig(
    global_config=OpPalettizerConfig(nbits=4, mode="kmeans")
)
compressed = palettize_weights(mlmodel, config)
compressed.save("OmniVoiceBackbone_int4.mlpackage")
```

Expected size reduction:

| Precision | Backbone size | Decoder size |
|-----------|--------------|--------------|
| FP32 | ~2.45 GB | ~806 MB |
| FP16 | ~1.22 GB | ~403 MB |
| INT8 | ~612 MB | ~201 MB |
| INT4 | ~306 MB | ~201 MB (keep INT8) |

#### A2. Reduce num_step

Test quality at num_step=8 vs 16. If 8 is acceptable, that halves backbone calls.

```bash
# A/B test
curl -o step8.wav -X POST http://127.0.0.1:8003/generate \
  -F mode=design -F text="Test sentence." \
  -F instruct="female, american accent" -F num_step=8

curl -o step16.wav -X POST http://127.0.0.1:8003/generate \
  -F mode=design -F text="Test sentence." \
  -F instruct="female, american accent" -F num_step=16
```

#### A3. Explore CFG-free or reduced-CFG inference

Currently batch_size=2 because of classifier-free guidance. Options:
- **CFG distillation**: Train a student that doesn't need CFG (halves compute)
- **Reduced guidance_scale**: Test guidance_scale=1.0 (no CFG, batch=1)
- **Guidance at fewer steps**: Only apply CFG for the first N steps

```bash
# Test no-CFG quality
curl -o nocfg.wav -X POST http://127.0.0.1:8003/generate \
  -F mode=design -F text="Test sentence." \
  -F instruct="female, american accent" \
  -F num_step=16 -F guidance_scale=1.0
```

#### A4. Export voice clone encoder to ONNX

```python
# The encoder is the tokenizer's encode path
# audio_tokenizer.encode(audio_values) -> audio_codes
# This needs its own ONNX export wrapper
class CloneEncoderForOnnx(nn.Module):
    def __init__(self, audio_tokenizer):
        super().__init__()
        self.audio_tokenizer = audio_tokenizer

    def forward(self, audio_values: torch.Tensor) -> torch.Tensor:
        return self.audio_tokenizer.encode(audio_values).audio_codes
```

#### A5. Validate parity after each optimization

Use existing `debug_onnx_parity.py` and `debug_decoder_parity.py` patterns.
Every quantization level needs audio quality validation.

---

### Phase B: CoreML Conversion (Python-side)

**Goal:** Convert ONNX artifacts to native CoreML `.mlpackage` files.

#### B1. Convert backbone ONNX → CoreML

```python
import coremltools as ct

mlmodel = ct.converters.convert(
    "artifacts/onnx_dynamic/omnivoice_backbone_dynamic.onnx",
    convert_to="mlprogram",
    compute_precision=ct.precision.FLOAT16,
    minimum_deployment_target=ct.target.iOS17,
    compute_units=ct.ComputeUnit.ALL,  # ANE + GPU + CPU
)
mlmodel.save("artifacts/coreml/OmniVoiceBackbone.mlpackage")
```

**Known risks:**
- Dynamic shapes: CoreML prefers fixed or enumerated shapes. May need to
  export multiple shape variants or use `EnumeratedShapes`.
- Attention mask: The 4D boolean mask `[2, 1, S, S]` may not map cleanly
  to ANE. May need to convert to float multiply.
- Rotary embeddings (RoPE): Qwen3 uses rotary position embeddings. Some
  CoreML versions handle these poorly on ANE.

#### B2. Convert decoder ONNX → CoreML

```python
mlmodel = ct.converters.convert(
    "artifacts/onnx_dynamic/omnivoice_decoder_dynamic.onnx",
    convert_to="mlprogram",
    compute_precision=ct.precision.FLOAT16,
    minimum_deployment_target=ct.target.iOS17,
)
mlmodel.save("artifacts/coreml/OmniVoiceDecoder.mlpackage")
```

**Known risk:** Decoder CoreML previously produced NaN outputs with
`NeuralNetwork` format. `MLProgram` format worked but had worse parity
than CPU ORT. This needs careful validation.

#### B3. Profile CoreML compute plan

```python
# Check what runs on ANE vs GPU vs CPU
import coremltools as ct

mlmodel = ct.models.MLModel("OmniVoiceBackbone.mlpackage")
spec = mlmodel.get_spec()
# Inspect operation types
for layer in spec.mlProgram.functions["main"].block.operations:
    print(layer.type, layer.attributes)
```

On-device profiling with Xcode Instruments > CoreML Performance template
will show the actual ANE/GPU/CPU split.

#### B4. Enumerated shapes for CoreML

CoreML doesn't handle fully dynamic shapes well on ANE. Better to enumerate
the common sequence lengths:

```python
from coremltools.converters.mil import Builder as mb

# Instead of fully dynamic, enumerate likely shapes
# Short utterance: ~200 tokens, medium: ~450, long: ~700
shapes = [
    (2, 8, 200),   # ~2.5 sec audio
    (2, 8, 450),   # ~5 sec audio
    (2, 8, 700),   # ~8 sec audio
]
```

---

### Phase C: Swift Inference Engine

**Goal:** Reimplement the Python generation loop in Swift.

#### C1. Text tokenizer in Swift

Options:
- **swift-transformers** (huggingface/swift-transformers): Has BPE tokenizer
  support, can load `tokenizer.json` directly.
- **Custom BPE**: The `tokenizer.json` file is ~11 MB. Parse the vocab and
  merges, implement BPE encode in Swift.

```swift
import Tokenizers  // from swift-transformers

let tokenizer = try AutoTokenizer.from(pretrained: "k2-fsa/OmniVoice")
let ids = tokenizer.encode(text: "<|text_start|>Hello world<|text_end|>")
```

The special tokens that matter:
- `<|text_start|>`, `<|text_end|>`
- `<|lang_start|>`, `<|lang_end|>`
- `<|instruct_start|>`, `<|instruct_end|>`
- `<|denoise|>`
- Audio mask token ID (from config)

#### C2. Prompt builder in Swift

Reimplement `_prepare_inference_inputs` from `omnivoice.py`:

```swift
struct PreparedInputs {
    let inputIds: MLMultiArray    // [2, 8, S]
    let audioMask: MLMultiArray   // [2, S]
    let attentionMask: MLMultiArray // [2, 1, S, S]
    let audioStartIndex: Int
    let audioLength: Int
}

func prepareInferenceInputs(
    text: String,
    instruct: String?,
    language: String?,
    numTargetTokens: Int
) -> PreparedInputs {
    // 1. Tokenize style prefix
    // 2. Tokenize text
    // 3. Create audio mask region
    // 4. Build attention mask (causal for text, full for audio)
    // 5. Create CFG unconditional copy (batch dim 1)
}
```

Key logic to port from Python:
- `_estimate_target_tokens()` — duration estimation
- `_build_style_prefix()` — language/instruct token construction
- Voice design instruct validation (`_INSTRUCT_VALID_EN`, etc.)
- Attention mask construction (causal text + bidirectional audio)

#### C3. Generation loop in Swift

```swift
func generateIterative(
    inputs: PreparedInputs,
    backbone: MLModel,
    numSteps: Int,
    guidanceScale: Float,
    tShift: Float
) -> MLMultiArray {
    var currentIds = inputs.inputIds  // [2, 8, S]

    for step in 0..<numSteps {
        // 1. Run backbone
        let prediction = try backbone.prediction(from: currentIds, ...)
        let logits = prediction["logits"]  // [2, 8, S, 1025]

        // 2. Apply CFG
        let condLogits = logits[0]    // conditional
        let uncondLogits = logits[1]  // unconditional
        let guided = uncondLogits + guidanceScale * (condLogits - uncondLogits)

        // 3. Compute confidence scores
        // 4. Select positions to unmask (based on schedule)
        // 5. Sample tokens at unmasked positions
        // 6. Update currentIds
    }

    return currentIds[0]  // return conditional tokens
}
```

Key scheduling logic to port:
- Time-step schedule with t_shift
- Mask ratio per step: `ratio = cos(π/2 * t)`
- Position selection with layer penalty
- Temperature-based sampling
- Confidence scoring from logits

#### C4. Audio decoder call

```swift
let decoderInput = try MLDictionaryFeatureProvider(dictionary: [
    "audio_codes": audioTokens  // [1, 8, T]
])
let decoderOutput = try decoder.prediction(from: decoderInput)
let audioSamples = decoderOutput.featureValue(for: "audio_values")!
    .multiArrayValue!  // [1, 1, T*320]
```

#### C5. Post-processing in Swift

```swift
import Accelerate

func postprocessAudio(_ samples: [Float], sampleRate: Int) -> [Float] {
    // RMS normalization
    var rms: Float = 0
    vDSP_rmsqv(samples, 1, &rms, vDSP_Length(samples.count))
    let targetRMS: Float = 0.1
    let scale = targetRMS / max(rms, 1e-8)
    var normalized = samples
    vDSP_vsmul(samples, 1, [scale], &normalized, 1, vDSP_Length(samples.count))

    // Fade in/out (linear)
    let fadeSamples = Int(0.01 * Float(sampleRate))  // 10ms
    for i in 0..<fadeSamples {
        let factor = Float(i) / Float(fadeSamples)
        normalized[i] *= factor
        normalized[normalized.count - 1 - i] *= factor
    }

    return normalized
}
```

#### C6. Audio playback

```swift
import AVFoundation

let engine = AVAudioEngine()
let playerNode = AVAudioPlayerNode()
engine.attach(playerNode)

let format = AVAudioFormat(
    standardFormatWithSampleRate: 24000,
    channels: 1
)!
engine.connect(playerNode, to: engine.mainMixerNode, format: format)

let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: UInt32(samples.count))!
// Copy samples into buffer
memcpy(buffer.floatChannelData![0], samples, samples.count * 4)
buffer.frameLength = UInt32(samples.count)

try engine.start()
playerNode.scheduleBuffer(buffer, completionHandler: nil)
playerNode.play()
```

---

### Phase D: Optimization & Profiling

**Goal:** Hit RTF < 1.0 on iPhone.

#### D1. Profile with Xcode Instruments

- CoreML Performance instrument: shows ANE/GPU/CPU split per op
- Time Profiler: shows Swift overhead
- Metal System Trace: shows GPU utilization

#### D2. Optimize attention for ANE

The ANE prefers:
- Static shapes (use enumerated shapes)
- Float16 (not Float32 or Bool for masks)
- No gather/scatter ops
- Contiguous memory layouts

Convert boolean attention mask to float multiply:
```python
# Instead of: output = attn_weights.masked_fill(~mask, -inf)
# Use: output = attn_weights * mask_float + (1 - mask_float) * (-1e9)
```

#### D3. Benchmark per-component

```swift
let startBackbone = CFAbsoluteTimeGetCurrent()
let logits = try backbone.prediction(from: inputs)
let backboneTime = CFAbsoluteTimeGetCurrent() - startBackbone

let startDecoder = CFAbsoluteTimeGetCurrent()
let audio = try decoder.prediction(from: tokens)
let decoderTime = CFAbsoluteTimeGetCurrent() - startDecoder

print("Backbone: \(backboneTime)s, Decoder: \(decoderTime)s")
print("Backbone per-step: \(backboneTime / Double(numSteps))s")
```

#### D4. Consider ONNX Runtime Mobile as fallback

If CoreML conversion hits unsupported ops:

```swift
// Using onnxruntime-swift package
import OnnxRuntimeModule

let env = try ORTEnvironment(loggingLevel: .warning)
let session = try ORTSession(
    env: env,
    modelPath: backbonePath,
    sessionOptions: nil
)
```

ONNX Runtime Mobile supports CoreML EP on iOS as well, so this can still
use Apple acceleration without native `.mlpackage` files.

---

### Phase E: Voice Clone on iOS

**Goal:** Support clone mode without server.

#### E1. Export clone encoder

The encoder path (`audio_tokenizer.encode`) converts waveform → audio codes.
This is needed only for clone mode, not design or auto mode.

```python
class CloneEncoderWrapper(nn.Module):
    def __init__(self, audio_tokenizer):
        super().__init__()
        self.audio_tokenizer = audio_tokenizer

    def forward(self, audio_values: torch.Tensor) -> torch.Tensor:
        # audio_values: [1, 1, num_samples]
        return self.audio_tokenizer.encode(audio_values).audio_codes
        # returns: [1, 8, T]

# Export
torch.onnx.export(wrapper, dummy_audio, "clone_encoder.onnx", ...)
```

#### E2. Reference audio preprocessing in Swift

```swift
// Load reference audio
let audioFile = try AVAudioFile(forReading: refAudioURL)
// Resample to 24kHz
// Trim to 3-10 seconds
// Remove silence
// Compute RMS for normalization
```

#### E3. Clone prompt construction

Port `_build_clone_prompt()` logic: prepend reference audio tokens + text
tokens before the generation target region.

---

## 6. Risk Assessment

| Risk | Severity | Mitigation |
|------|----------|------------|
| CoreML conversion fails (unsupported ops) | High | Use ONNX Runtime Mobile with CoreML EP as fallback |
| ANE rejects the model (falls back to CPU) | High | Profile compute plan, restructure ops if needed |
| INT4 quantization degrades audio quality | Medium | Test INT8 first, use mixed precision |
| Attention mask shape causes ANE fallback | Medium | Convert to float multiply, use enumerated shapes |
| Memory exceeds iPhone budget | Medium | INT4 backbone mandatory; stream decoder if needed |
| RoPE not supported on ANE | Medium | Pre-compute and pass as input, or use GPU for that op |
| Generation loop overhead in Swift | Low | The loop is simple math; backbone calls dominate |
| Tokenizer mismatch | Low | Validate token IDs against Python reference |
| Dynamic seq_len causes recompilation | Medium | Use enumerated shapes (3-5 fixed lengths) |

---

## 7. Realistic Timeline Estimate

| Phase | Duration | Dependencies |
|-------|----------|-------------|
| A: Model optimization | 1-2 weeks | None |
| B: CoreML conversion | 1-2 weeks | Phase A |
| C: Swift inference engine | 2-4 weeks | Phase B |
| D: Optimization & profiling | 2-3 weeks | Phase C |
| E: Voice clone on iOS | 1-2 weeks | Phase C |
| **Total** | **7-13 weeks** | — |

---

## 8. Minimum Viable iOS Demo

For the fastest path to a working iOS prototype:

1. **Design mode only** (no clone, no ref audio encoding)
2. **num_step=8** (if quality is acceptable)
3. **guidance_scale=1.0** (no CFG, batch=1 instead of 2, halves compute)
4. **Short text only** (< 20 words, no chunking)
5. **INT8 quantized** CoreML models
6. **English only** (simplifies tokenizer validation)

This would demonstrate feasibility. Production quality can be iterated after.

---

## 9. Alternative: Server-Assisted Hybrid

If pure on-device real-time proves infeasible:

```
iPhone App
  ├─ Records reference audio locally
  ├─ Sends text + ref_audio to server
  ├─ Server runs PyTorch+GPU inference (~1-2 sec)
  ├─ Streams WAV back to phone
  └─ Plays audio via AVAudioEngine
```

This is what most production TTS apps do today (ElevenLabs, Play.ht, etc.).
The existing `api_server.py` already supports this architecture.

The on-device path is worth pursuing for:
- Offline capability
- Privacy (voice data stays on device)
- Latency (no network round-trip)
- Cost (no server infrastructure)

---

## 10. Files That Need Changes

### Python-side (export & optimization)

| File | Change |
|------|--------|
| `scripts/export_coreml_backbone.py` | Add enumerated shapes, quantization, ANE-friendly mask conversion |
| `scripts/export_audio_decoder.py` | Fix CoreML NaN issue, add INT8 quantization |
| `scripts/export_onnx_dynamic.py` | Already done, no changes needed |
| New: `scripts/export_clone_encoder.py` | Export audio encoder for clone mode |
| New: `scripts/quantize_coreml.py` | INT4/INT8 quantization script |
| New: `scripts/validate_coreml_parity.py` | End-to-end CoreML vs PyTorch comparison |

### Swift-side (new iOS project)

| File | Purpose |
|------|---------|
| `ios/OmniVoice/OmniVoiceEngine.swift` | Main inference engine, loads models, runs generation |
| `ios/OmniVoice/Tokenizer.swift` | BPE tokenizer wrapping swift-transformers |
| `ios/OmniVoice/PromptBuilder.swift` | Port of `_prepare_inference_inputs` |
| `ios/OmniVoice/GenerationLoop.swift` | Port of `_generate_iterative` |
| `ios/OmniVoice/AudioPostProcessor.swift` | Silence removal, RMS norm, fade |
| `ios/OmniVoice/AudioPlayer.swift` | AVAudioEngine playback |
| `ios/OmniVoice/Models/` | CoreML `.mlpackage` bundles |

---

## 11. Immediate Next Steps (This Week)

1. **Test guidance_scale=1.0 quality** — if acceptable, this halves compute
2. **Test num_step=8 quality** — if acceptable, this halves iterations
3. **Run `export_coreml_backbone.py`** — convert ONNX → CoreML `.mlpackage`
4. **Profile CoreML model** — check ANE vs GPU vs CPU op split
5. **Run `coremltools` quantization** — INT8 first, then INT4
6. **Validate quantized model quality** — listen to outputs